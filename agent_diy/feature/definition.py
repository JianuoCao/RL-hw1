#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np
from kaiwu_agent.utils.common_func import create_cls, attached


# Each training sample carries: (s, a, r, s', done)
# 每条训练样本包含: (状态, 动作, 奖励, 下一状态, 是否终止)
SampleData = create_cls(
    "SampleData",
    state=None,
    action=None,
    reward=None,
    next_state=None,
    done=None,
)


@attached
def sample_process(list_game_data):
    """Convert a list of Frame objects into a list of SampleData."""
    return [SampleData(**i.__dict__) for i in list_game_data]


def reward_shaping(frame_no, score, terminated, truncated, obs, _obs):
    """
    宝箱优先的多层级奖励塑形，与 env score 结构对齐。

    Env score 比例: 终点(150) vs 宝箱(100/个)，宝箱价值 ≈ 终点的 67%
    Reward 比例:   终点(+80)  vs 宝箱(+55/个)，对齐 ≈ 69%

    Args:
        frame_no   : current step index within the episode
        score      : raw environment score (unused - for interface compatibility)
        terminated : True when the agent reaches the goal
        truncated  : True when the episode is cut short (max steps exceeded)
        obs        : raw observation vector at current step (shape: 250,)
        _obs       : raw observation vector at next step   (shape: 250,)

    Returns:
        reward (float)

    观测说明（raw env obs，非 processed feature）:
        obs[129]    : 终点距离 (0=最近, 6=最远)
        obs[130:140]: 10个宝箱距离 (0=无/已收集, 1-6=距离)
    """
    reward = 0.0

    # ===== Layer 1: Terminal State Reward (最高权重) =====
    if terminated:
        reward += 80.0  # 降低: 100 → 80，给宝箱让出空间
        return reward

    if truncated:
        reward -= 30.0
        return reward

    # ===== Layer 2: Goal Approach Reward (权重降低，避免压制宝箱探索) =====
    try:
        before_end_dist = float(obs[129])
        after_end_dist = float(_obs[129])
        goal_delta = before_end_dist - after_end_dist

        if goal_delta > 0:
            reward += goal_delta * 2.5   # 降低: 5.0 → 2.5
        else:
            reward += goal_delta * 0.5   # 降低: 1.0 → 0.5，允许绕道收宝箱
    except (IndexError, TypeError, ValueError):
        pass

    # ===== Layer 3: Per-Chest Approach Reward (追踪所有宝箱，不只是最近的) =====
    # 地图上宝箱分散在各处，对每个可见宝箱都给予靠近奖励，引导主动探索全图
    try:
        before_treasure = np.array(obs[130:140], dtype=float)
        after_treasure = np.array(_obs[130:140], dtype=float)

        for i in range(10):
            b = before_treasure[i]
            a = after_treasure[i]
            if b > 0 and a > 0:  # 前后两步该宝箱都可见（未收集）
                delta = b - a    # 正值=靠近，负值=远离
                if delta > 0:
                    reward += delta * 2.5   # 靠近任意宝箱都给奖励
                elif delta < 0:
                    reward += delta * 0.2   # 远离宝箱仅轻微惩罚
    except (IndexError, TypeError, ValueError):
        pass

    # ===== Layer 4: Treasure Collection Bonus (大幅提升，对齐 env 价值) =====
    # env 中每个宝箱=100分，宝箱价值=终点(80)×(100/150) ≈ 55
    try:
        before_treasure = np.array(obs[130:140], dtype=float)
        after_treasure = np.array(_obs[130:140], dtype=float)

        collected = np.sum((before_treasure > 0) & (after_treasure == 0))
        if collected > 0:
            reward += collected * 55.0   # 大幅提升: 15 → 55
    except (IndexError, TypeError, ValueError):
        pass

    # ===== Layer 5: Step Efficiency Penalty (低权重) =====
    reward -= 0.1

    return reward
