#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from kaiwu_agent.utils.common_func import attached


@attached
class Model(nn.Module):
    """
    Dueling DQN Q-network:
      shared trunk → split into Value stream + Advantage stream → combine.
    Dueling DQN Q网络：
      共享主干 → 分为价值流 + 优势流 → 合并输出Q值。
    """

    def __init__(self, state_shape, action_shape):
        super().__init__()

        if isinstance(state_shape, (tuple, list)):
            in_dim = int(np.prod(state_shape))
        else:
            in_dim = int(state_shape)

        if isinstance(action_shape, (tuple, list)):
            out_dim = int(np.prod(action_shape))
        else:
            out_dim = int(action_shape)

        from agent_diy.conf.conf import Config
        h1 = Config.HIDDEN_SIZE_1
        h2 = Config.HIDDEN_SIZE_2

        # Shared feature trunk
        # 共享特征主干
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.LayerNorm(h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.ReLU(),
        )

        # Value stream: V(s)
        # 价值流: 估计状态价值
        self.value_head = nn.Sequential(
            nn.Linear(h2, h2 // 2),
            nn.ReLU(),
            nn.Linear(h2 // 2, 1),
        )

        # Advantage stream: A(s, a)
        # 优势流: 估计每个动作的相对优势
        self.advantage_head = nn.Sequential(
            nn.Linear(h2, h2 // 2),
            nn.ReLU(),
            nn.Linear(h2 // 2, out_dim),
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        features = self.trunk(x)                    # (B, h2)
        value = self.value_head(features)           # (B, 1)
        advantage = self.advantage_head(features)   # (B, A)

        # Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
