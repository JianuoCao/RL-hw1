#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class Config:

    # Observation / action dimensions
    # 观察维度 / 动作维度
    OBSERVATION_SHAPE = 134
    ACTION_SIZE = 4

    # DQN hyperparameters
    # DQN 超参数
    LEARNING_RATE = 5e-4
    GAMMA = 0.95

    # Epsilon-greedy exploration schedule (step-based linear decay)
    # ε-贪心探索衰减参数（基于步数的线性衰减）
    EPSILON_START = 1.0
    EPSILON_MIN = 0.05
    EPSILON_DECAY_STEPS = 200000  # linearly decay ε over this many steps

    # Replay buffer
    # 经验回放缓冲区
    BUFFER_CAPACITY = 30000
    BATCH_SIZE = 128
    MIN_REPLAY_SIZE = 500        # start training sooner

    # Target network hard-update frequency (in training steps)
    # 目标网络硬更新频率（按训练步数计）
    TARGET_UPDATE_FREQ = 200

    # Training budget
    # 训练预算
    MAX_TOTAL_STEPS = 500000
    EPISODES = 10000             # upper bound, will stop by step budget

    # Network hidden layer sizes (smaller = faster convergence)
    # 网络隐层大小（更小 = 更快收敛）
    HIDDEN_SIZE_1 = 128
    HIDDEN_SIZE_2 = 64

    # Sample dimensionality (state, action, reward, next_state, done)
    # 样本维度
    SAMPLE_DIM = 5
