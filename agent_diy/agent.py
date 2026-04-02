#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np
import os
import torch
from kaiwu_agent.agent.base_agent import BaseAgent
from kaiwu_agent.utils.common_func import create_cls, attached
from kaiwu_agent.agent.base_agent import (
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
    predict_wrapper,
    exploit_wrapper,
    check_hasattr,
)
from agent_diy.conf.conf import Config
from agent_diy.algorithm.algorithm import Algorithm


ObsData = create_cls("ObsData", feature=None)
ActData = create_cls("ActData", act=None)


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        self.logger = logger

        # ε-greedy 探索参数
        self.action_size = Config.ACTION_SIZE
        self.epsilon = Config.EPSILON_START

        # 算法对象（含 Q 网络、目标网络、经验回放缓冲区）
        self.algorithm = Algorithm(logger)

        super().__init__(agent_type, device, logger, monitor)

    @predict_wrapper
    def predict(self, list_obs_data):
        """
        ε-greedy policy used during training.
        以概率 ε 随机探索，否则执行贪心动作。
        """
        state = np.array(list_obs_data[0].feature, dtype=np.float32)

        if np.random.rand() < self.epsilon:
            act = int(np.random.randint(0, self.action_size))
        else:
            q_values = self.algorithm.predict_q(state)   # (action_size,)
            act = int(np.argmax(q_values))

        return [ActData(act=act)]

    @exploit_wrapper
    def exploit(self, list_obs_data):
        """
        Pure greedy policy used during evaluation (no exploration).
        评估阶段纯贪心，不随机探索。
        """
        state = np.array(list_obs_data[0].feature, dtype=np.float32)
        q_values = self.algorithm.predict_q(state)
        act = int(np.argmax(q_values))
        return [ActData(act=act)]

    @learn_wrapper
    def learn(self, list_sample_data):
        """
        Store transitions and, once the replay buffer is warm, run one
        gradient update step on the online Q-network.
        将转换存入回放缓冲区，缓冲区预热后执行一步梯度更新。
        """
        return self.algorithm.learn(list_sample_data)

    def observation_process(self, raw_obs, game_info=None):
        """
        Build the fixed-length float32 feature vector fed to the Q-network.

        Feature layout (134-dim, compact representation):
          [0:2]     : normalized position (pos_x/63, pos_z/63)
          [2:23]    : end / treasure distances (21 dims, /6.0)
          [23:48]   : obstacle flat 5×5  (25 dims)
          [48:73]   : treasure flat 5×5  (25 dims)
          [73:98]   : end flat 5×5       (25 dims)
          [98:123]  : location memory 5×5 (25 dims, clipped & /10)
          [123:133] : treasure collection status (10 dims)

        构建输入 Q 网络的紧凑定长 float32 特征向量（134 维）。
        """
        if game_info is None:
            # Fallback: use raw_obs directly (e.g. during exploit without state)
            feature = np.asarray(raw_obs, dtype=np.float32)
            return ObsData(feature=_pad_or_trim(feature, Config.OBSERVATION_SHAPE))

        pos = [int(game_info.pos_x), int(game_info.pos_z)]

        # Feature 1: normalized position (2 dims, replaces 128-dim one-hot)
        pos_norm = [pos[0] / 63.0, pos[1] / 63.0]

        # Feature 2: distances from raw_obs (indices 129-149, normalized to [0,1])
        end_treasure_dists = list(raw_obs[129:150]) if len(raw_obs) >= 150 else list(raw_obs[129:])
        end_treasure_dists = [d / 6.0 for d in end_treasure_dists]

        # Feature 5: local view maps (obstacle / treasure / end), each 5×5
        local_view = [
            game_info.local_view[i: i + 5]
            for i in range(0, len(game_info.local_view), 5)
        ]
        obstacle_flat, treasure_flat, end_flat = [], [], []
        for row in local_view:
            obstacle_flat.extend([1 if v == 0 else 0 for v in row])
            treasure_flat.extend([1 if v == 4 else 0 for v in row])
            end_flat.extend([1 if v == 3 else 0 for v in row])

        # Feature 6: visited-cell memory within the local view window (clipped & normalized)
        view = game_info.view
        memory_flat = []
        for i in range(view * 2 + 1):
            idx_start = (pos[0] - view + i) * 64 + (pos[1] - view)
            memory_flat.extend(
                game_info.location_memory[idx_start: idx_start + view * 2 + 1]
            )
        memory_flat = [min(v, 10) / 10.0 for v in memory_flat]

        # Feature 7: treasure collection status (remap status=2 → 0)
        treasure_status = [x if x != 2 else 0 for x in game_info.treasure_status]

        feature = np.concatenate([
            pos_norm,
            end_treasure_dists,
            obstacle_flat,
            treasure_flat,
            end_flat,
            memory_flat,
            treasure_status,
        ]).astype(np.float32)

        return ObsData(feature=_pad_or_trim(feature, Config.OBSERVATION_SHAPE))

    def action_process(self, act_data):
        return act_data.act

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        """
        Persist online Q-network weights to disk.
        将在线 Q 网络参数保存到磁盘。
        """
        os.makedirs(path, exist_ok=True)
        model_file = f"{path}/model.ckpt-{id}.pt"
        torch.save(self.algorithm.q_net.state_dict(), model_file)
        if self.logger:
            self.logger.info(f"[DQN] save model {model_file} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        """
        Restore online Q-network weights and sync to target network.
        从磁盘加载 Q 网络参数，并同步到目标网络。
        """
        model_file = f"{path}/model.ckpt-{id}.pt"
        try:
            state_dict = torch.load(model_file, map_location="cpu")
            self.algorithm.q_net.load_state_dict(state_dict)
            self.algorithm._sync_target()
            if self.logger:
                self.logger.info(f"[DQN] load model {model_file} successfully")
        except FileNotFoundError:
            if self.logger:
                self.logger.error(f"[DQN] model file not found: {model_file}")
            raise


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _pad_or_trim(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Ensure exactly target_len elements by trimming or zero-padding."""
    if len(arr) >= target_len:
        return arr[:target_len]
    return np.concatenate(
        [arr, np.zeros(target_len - len(arr), dtype=np.float32)]
    )
