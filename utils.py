import numpy as np
import torch


def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    """
        适用于多商品四维张量的滑动窗口切分。

        参数:
        - X: 输入数据，形状为 [P, N, F, T]
        - num_timesteps_input: 输入序列长度
        - num_timesteps_output: 输出序列长度

        返回:
        - inputs: [T_in, P, num_timesteps_input, N, F]
        - targets:  [T_in, P, num_timesteps_output, N, F]
    """
    P, N, F, T = X.shape

    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(T - (num_timesteps_input + num_timesteps_output) + 1)]

    inputs, targets = [], []

    for i, j in indices:
        # X[:, :, :, i:i+input] → [P, N, F, input] → permute to [P, input, N, F]
        input_slice = X[:, :, :, i:i + num_timesteps_input].permute(0, 3, 1, 2)
        target_slice = X[:, :, :, i + num_timesteps_input:j].permute(0, 3, 1, 2)
        inputs.append(input_slice)
        targets.append(target_slice)

    inputs = torch.from_numpy(np.stack(inputs))
    targets = torch.from_numpy(np.stack(targets))
    return inputs.float(), targets.float()


def generate_adj_matrices(A, num_timesteps_input):
    """
        对输入张量进行滑动窗口切分，默认一个窗口用其第一个时间步

        参数：
        - tensor: 输入的 [P, T, N, N] 张量，例如 [7, 276, 76, 76]
        - num_timesteps_input: 输入序列长度

        输出：
        - inputs: [T_in, P, N, N]
    """
    P, T, N, _ = A.shape

    inputs = []

    for i in range(T - num_timesteps_input + 1):
        input_slice = A[:, i, :, :]
        inputs.append(input_slice)

    inputs = torch.stack(inputs, dim=0)

    return inputs
