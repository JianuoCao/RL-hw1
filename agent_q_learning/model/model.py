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


class Model(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()

        # User-defined network
        # 用户自定义网络
        state_dim = int(np.prod(state_shape)) if isinstance(state_shape, (tuple, list)) else int(state_shape)
        action_dim = int(np.prod(action_shape)) if isinstance(action_shape, (tuple, list)) else int(action_shape)

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.view(x.size(0), -1)
        return self.net(x)
