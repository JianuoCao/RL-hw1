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
    OBSERVATION_SHAPE = 250
    ACTION_SIZE = 4

    # DQN hyperparameters
    # DQN 超参数
    LEARNING_RATE = 1e-3
    GAMMA = 0.99

    # Epsilon-greedy exploration schedule
    # ε-贪心探索衰减参数
    EPSILON_START = 1.0
    EPSILON_MIN = 0.05
    EPSILON_DECAY = 0.995        # multiplicative decay per episode

    # Replay buffer
    # 经验回放缓冲区
    BUFFER_CAPACITY = 50000
    BATCH_SIZE = 64
    MIN_REPLAY_SIZE = 1000       # start training only after buffer reaches this size

    # Target network hard-update frequency (in training steps)
    # 目标网络硬更新频率（按训练步数计）
    TARGET_UPDATE_FREQ = 500

    # Training episodes
    # 训练轮数
    EPISODES = 5000

    # Network hidden layer sizes
    # 网络隐层大小
    HIDDEN_SIZE_1 = 256
    HIDDEN_SIZE_2 = 128

    # Sample dimensionality (state, action, reward, next_state, done)
    # 样本维度
    SAMPLE_DIM = 5
