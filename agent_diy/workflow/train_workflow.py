#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import time

from kaiwu_agent.utils.common_func import Frame, attached
from tools.train_env_conf_validate import check_usr_conf, read_usr_conf

from agent_diy.feature.definition import sample_process, reward_shaping
from agent_diy.conf.conf import Config


@attached
def workflow(envs, agents, logger=None, monitor=None):
    """
    DQN training workflow.

    Loop structure
    --------------
    for each episode:
        reset env → observe s
        while not done:
            predict action a  (ε-greedy)
            step env → r, s', done
            store (s, a, r, s', done) in replay buffer via agent.learn
            s ← s'
        decay ε
    save model when converged or max episodes reached

    DQN 训练工作流。
    """

    # --- Configuration validation ---
    # 配置文件读取与校验
    usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
        return

    valid = check_usr_conf(usr_conf, logger)
    if not valid:
        logger.error("check_usr_conf returned False, please check configuration")
        return

    env, agent = envs[0], agents[0]

    EPISODES = Config.EPISODES
    MAX_TOTAL_STEPS = Config.MAX_TOTAL_STEPS

    # Step-based linear epsilon schedule
    # 基于全局步数的线性 ε 衰减
    eps_start = Config.EPSILON_START
    eps_min = Config.EPSILON_MIN
    eps_decay_steps = Config.EPSILON_DECAY_STEPS
    total_steps = 0

    # Monitoring state
    # 监控数据
    monitor_data = {
        "reward": 0,
        "diy_1": 0,   # total wins
        "diy_2": 0,   # epsilon
        "diy_3": 0,   # replay buffer size
        "diy_4": 0,
        "diy_5": 0,
    }
    last_report_time = time.time()

    total_reward_window = 0.0   # accumulated reward since last report
    win_cnt = 0
    start_t = time.time()

    logger.info("[DQN] Training started ...")

    for episode in range(EPISODES):

        # --- Reset environment ---
        # 重置环境
        obs, state = env.reset(usr_conf=usr_conf)
        if obs is None:
            continue

        obs_data = agent.observation_process(obs, state)
        done = False
        episode_reward = 0.0

        # --- Episode loop ---
        # 单轮交互循环
        while not done:
            # Select action (ε-greedy)
            # 选择动作（ε-贪心）
            act_data, _ = agent.predict(list_obs_data=[obs_data])
            act_data = act_data[0]
            act = agent.action_process(act_data)

            # Environment step
            # 执行动作，获取下一帧
            frame_no, _obs, score, terminated, truncated, _state = env.step(act)
            if _obs is None:
                break

            _obs_data = agent.observation_process(_obs, _state)

            # Reward shaping
            # 奖励塑形
            reward = reward_shaping(frame_no, score, terminated, truncated, obs, _obs)

            done = terminated or truncated
            if terminated:
                win_cnt += 1

            # Build training sample and learn
            # 构建训练样本并学习
            sample = Frame(
                state=obs_data.feature,
                action=act,
                reward=reward,
                next_state=_obs_data.feature,
                done=float(done),
            )
            agent.learn(sample_process([sample]))

            episode_reward += reward
            total_reward_window += reward
            obs = _obs
            obs_data = _obs_data

            # --- Step-based linear epsilon decay ---
            # 每步线性衰减 ε
            total_steps += 1
            frac = min(1.0, total_steps / eps_decay_steps)
            agent.epsilon = eps_start + frac * (eps_min - eps_start)

        # --- Stop if step budget exhausted ---
        if total_steps >= MAX_TOTAL_STEPS:
            logger.info(f"[DQN] Step budget exhausted at episode {episode + 1}, total_steps={total_steps}")
            break

        # --- Periodic logging ---
        # 定期上报训练进度
        now = time.time()
        if now - last_report_time > 60:
            win_rate = win_cnt / (episode + 1)
            buf_size = len(agent.algorithm.replay_buffer)
            logger.info(
                f"[DQN] Episode {episode + 1}/{EPISODES} | "
                f"WinRate={win_rate:.3f} | "
                f"ε={agent.epsilon:.4f} | "
                f"ReplayBuf={buf_size} | "
                f"AvgReward={total_reward_window:.2f}"
            )
            monitor_data["reward"] = total_reward_window
            monitor_data["diy_1"] = win_cnt
            monitor_data["diy_2"] = agent.epsilon
            monitor_data["diy_3"] = buf_size
            if monitor:
                monitor.put_data({os.getpid(): monitor_data})
            total_reward_window = 0.0
            last_report_time = now

        # --- Convergence check ---
        # 收敛判断：胜率超过 90% 且已训练超过 100 轮
        if episode > 100 and win_cnt / (episode + 1) > 0.9:
            logger.info(
                f"[DQN] Converged at episode {episode + 1} "
                f"(win_rate={win_cnt / (episode + 1):.3f})"
            )
            break

    end_t = time.time()
    logger.info(
        f"[DQN] Training finished: {episode + 1} episodes in {end_t - start_t:.1f}s"
    )

    # --- Save model ---
    # 保存模型
    agent.save_model()
