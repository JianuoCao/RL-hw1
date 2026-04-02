#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import random

from agent_diy.model.model import Model
from agent_diy.conf.conf import Config


class ReplayBuffer:
    """
    Fixed-capacity circular experience replay buffer.
    固定容量的循环经验回放缓冲区。
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class Algorithm:
    """
    DQN algorithm:
      - online Q-network (q_net) updated every training step
      - target Q-network (target_net) hard-updated every TARGET_UPDATE_FREQ steps
      - experience replay with a random uniform buffer

    DQN 算法：
      - 在线 Q 网络（q_net）每步更新
      - 目标 Q 网络（target_net）每 TARGET_UPDATE_FREQ 步硬更新
      - 均匀随机经验回放
    """

    def __init__(self, logger):
        self.logger = logger

        obs_shape = Config.OBSERVATION_SHAPE
        action_size = Config.ACTION_SIZE

        # Online network and target network
        # 在线网络与目标网络
        self.q_net = Model(obs_shape, action_size)
        self.target_net = Model(obs_shape, action_size)
        self._sync_target()          # initialise target = online
        self.target_net.eval()       # target net is never backprop'd through

        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=Config.LEARNING_RATE
        )

        self.replay_buffer = ReplayBuffer(Config.BUFFER_CAPACITY)

        self.gamma = Config.GAMMA
        self.batch_size = Config.BATCH_SIZE
        self.min_replay_size = Config.MIN_REPLAY_SIZE
        self.target_update_freq = Config.TARGET_UPDATE_FREQ

        # Global training step counter (used for target-network updates)
        # 全局训练步数计数（用于目标网络更新）
        self.train_step = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict_q(self, state: np.ndarray) -> np.ndarray:
        """
        Return Q-values for all actions given a single state (no grad).
        对单个状态计算所有动作的 Q 值（不计算梯度）。
        """
        self.q_net.eval()
        with torch.no_grad():
            q_values = self.q_net(state)   # (1, action_size)
        self.q_net.train()
        return q_values.squeeze(0).cpu().numpy()

    def store(self, state, action, reward, next_state, done):
        """Push one transition into the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self, list_sample_data):
        """
        Consume a list of SampleData objects produced by sample_process,
        store them in the replay buffer, and run one gradient update if
        the buffer is sufficiently large.

        接收 sample_process 产出的 SampleData 列表，存入回放缓冲区，
        缓冲区足够大时执行一次梯度更新。
        """
        # --- Store new transitions ---
        for sample in list_sample_data:
            self.replay_buffer.push(
                np.array(sample.state, dtype=np.float32),
                int(sample.action),
                float(sample.reward),
                np.array(sample.next_state, dtype=np.float32),
                float(sample.done),
            )

        # Wait until the buffer has enough samples before training
        # 缓冲区样本不足时暂不训练
        if len(self.replay_buffer) < self.min_replay_size:
            return None

        loss = self._update()
        return loss

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update(self):
        """Sample a mini-batch and perform one step of TD learning."""
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states_t = torch.FloatTensor(states)              # (B, obs_dim)
        actions_t = torch.LongTensor(actions)             # (B,)
        rewards_t = torch.FloatTensor(rewards)            # (B,)
        next_states_t = torch.FloatTensor(next_states)    # (B, obs_dim)
        dones_t = torch.FloatTensor(dones)                # (B,)  1.0 = done

        # Current Q-values for the taken actions: Q(s, a)
        # 当前步所采取动作的 Q 值: Q(s, a)
        q_values = self.q_net(states_t)                          # (B, A)
        q_current = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # (B,)

        # Double DQN: online net selects action, target net evaluates
        # 双重DQN：在线网络选动作，目标网络估值（减少Q值过高估计）
        with torch.no_grad():
            # Online net picks best action for next state
            q_next_online = self.q_net(next_states_t)            # (B, A)
            best_actions = q_next_online.argmax(dim=1, keepdim=True)  # (B, 1)
            # Target net evaluates that action
            q_next_target = self.target_net(next_states_t)       # (B, A)
            q_next_max = q_next_target.gather(1, best_actions).squeeze(1)  # (B,)
            q_target = rewards_t + self.gamma * q_next_max * (1.0 - dones_t)

        loss = F.smooth_l1_loss(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.train_step += 1

        # Hard-update target network
        # 硬更新目标网络
        if self.train_step % self.target_update_freq == 0:
            self._sync_target()
            if self.logger:
                self.logger.info(
                    f"[DQN] Target network updated at train_step={self.train_step}"
                )

        return loss.item()

    def _sync_target(self):
        """Copy online network weights to target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())
