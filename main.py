import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit

from model import STGCNAttModel
from utils import generate_dataset, generate_adj_matrices

torch.manual_seed(7)

epochs = 400
batch_size = 32
feature_dim1 = 76
# feature_dim2 = 5
feature_dim2 = 6  # 加上timestamp的特征就是6，不带就是5
emb_dim = 8

num_node = 76
num_timesteps_input = 12
num_timesteps_output = 1

torch.set_default_dtype(torch.float32)

parser = argparse.ArgumentParser(description='STGCN')
parser.add_argument('--enable-cuda', action='store_true',
                    help='Enable CUDA')
args = parser.parse_args()
args.device = None
if args.enable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')


def train_epoch(t_train_input, t_train_target, f_train_input, A_wave_train, batch_size):
    len_input = t_train_input.shape[0]

    model.train()
    epoch_loss = 0

    for i in range(0, len_input, batch_size):
        # 按照batch划分数据
        start_idx = i
        end_idx = min(i + batch_size, len_input)

        t_X_batch, t_y_batch = (t_train_input[start_idx:end_idx].to(device=args.device),
                                t_train_target[start_idx:end_idx].to(device=args.device))
        f_X_batch = f_train_input[start_idx:end_idx].to(device=args.device)
        A_wave_X_batch = A_wave_train[start_idx:end_idx].to(device=args.device)

        optimizer.zero_grad()
        output = model(t_X_batch, f_X_batch, product_emb, A_wave_X_batch)
        loss = criterion(output, t_y_batch)

        loss.backward()
        optimizer.step()

        batch_size_actual = end_idx - start_idx
        epoch_loss += loss.item() * batch_size_actual

    return epoch_loss / len_input


def validate(t_val_input, t_val_target, f_val_input, A_wave_val, batch_size):
    len_input = t_val_input.shape[0]

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for i in range(0, len_input, batch_size):
            start_idx = i
            end_idx = min(i + batch_size, len_input)

            t_X_batch, t_y_batch = (t_val_input[start_idx:end_idx].to(device=args.device),
                                    t_val_target[start_idx:end_idx].to(device=args.device))
            f_X_batch = f_val_input[start_idx:end_idx].to(device=args.device)
            A_wave_X_batch = A_wave_val[start_idx:end_idx].to(device=args.device)

            output = model(t_X_batch, f_X_batch, product_emb, A_wave_X_batch)
            loss = criterion(output, t_y_batch)

            batch_size_actual = end_idx - start_idx
            val_loss += loss.item() * batch_size_actual

    return val_loss / len_input


def test_model(t_test_input, t_test_target, f_test_input, A_wave_test, batch_size):
    model.eval()

    test_loss = 0
    all_preds = []
    all_trues = []

    len_input = t_test_input.shape[0]

    with torch.no_grad():
        for i in range(0, len_input, batch_size):
            start_idx = i
            end_idx = min(i + batch_size, len_input)

            t_X_batch, t_y_batch = (t_test_input[start_idx:end_idx].to(device=args.device),
                                    t_test_target[start_idx:end_idx].to(device=args.device))
            f_X_batch = f_test_input[start_idx:end_idx].to(device=args.device)
            A_wave_X_batch = A_wave_test[start_idx:end_idx].to(device=args.device)

            output = model(t_X_batch, f_X_batch, product_emb, A_wave_X_batch)
            loss = criterion(output, t_y_batch)

            batch_size_actual = end_idx - start_idx
            test_loss += loss.item() * batch_size_actual

            all_preds.append(output.cpu())
            all_trues.append(t_y_batch.cpu())

        preds = torch.cat(all_preds, dim=0)
        trues = torch.cat(all_trues, dim=0)

        mae = torch.abs(preds - trues).mean(dim=(0, 3, 4))
        mse = ((preds - trues) ** 2).mean(dim=(0, 3, 4))
        rmse = torch.sqrt(mse)

        non_zero = trues != 0
        ape = torch.zeros_like(preds)
        ape[non_zero] = (torch.abs(preds[non_zero] - trues[non_zero])
                         / trues[non_zero])
        mape = ape.sum(dim=(0, 3, 4)) / non_zero.sum(dim=(0, 3, 4))

        return {
            "loss": test_loss / len_input,
            "mae": mae.numpy(),
            "mse": mse.numpy(),
            "rmse": rmse.numpy(),
            "mape": mape.numpy()
        }


if __name__ == '__main__':
    file_path = r"../Processed_data/combined_data/"
    trade_flow = torch.load(file_path + "multi_trade.pt")
    A = torch.load(file_path + "multi_A.pt")
    features = torch.load(file_path + "multi_feature.pt")
    product_emb = torch.load(file_path + "product_embeddings.pt")

    # 对数据进行滑窗切分
    t_input, t_target = generate_dataset(trade_flow, num_timesteps_input, num_timesteps_output)
    f_input, _ = generate_dataset(features, num_timesteps_input, num_timesteps_output)
    A_wave = generate_adj_matrices(A, num_timesteps_input)

    tscv = TimeSeriesSplit(n_splits=3)
    all_metrics = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(t_input)):
        '''Sole'''
        if fold == 0 or fold == 1:
            continue
        n_samples = t_input.shape[0]
        split_index = int(n_samples * 0.8)
        train_idx = range(split_index)
        test_idx = range(split_index, n_samples)

        # 训练集: 前80%时间步的所有商品数据
        t_train_input, t_train_target = t_input[train_idx], t_target[train_idx]
        f_train_input = f_input[train_idx]
        A_wave_train = A_wave[train_idx]

        # 测试集: 后20%时间步 (保持时间连续)
        t_test_input, t_test_target = t_input[test_idx], t_target[test_idx]
        f_test_input = f_input[test_idx]
        A_wave_test = A_wave[test_idx]

        # 在训练集中进一步划分最后10%作为验证集
        val_split = int(0.9 * len(train_idx))
        t_train_input, t_val_input = t_train_input[:val_split], t_train_input[val_split:]
        t_train_target, t_val_target = t_train_target[:val_split], t_train_target[val_split:]
        f_train_input, f_val_input = f_train_input[:val_split], f_train_input[val_split:]
        A_wave_train, A_wave_val = A_wave_train[:val_split], A_wave_train[val_split:]

        # 初始化训练模型
        model = STGCNAttModel(node_size=num_node,
                              feature_dim1=feature_dim1,
                              feature_dim2=feature_dim2,
                              emb_dim=emb_dim,
                              num_timesteps_input=num_timesteps_input,
                              num_timesteps_output=num_timesteps_output).to(device=args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20

        '''Training & Validation'''
        for epoch in range(epochs):
            avg_train_loss = train_epoch(t_train_input, t_train_target, f_train_input, A_wave_train, batch_size)
            avg_val_loss = validate(t_val_input, t_val_target, f_val_input, A_wave_val, batch_size)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f"best_fold{fold}.pt")
            else:
                patience_counter += 1

            print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        '''Testing'''
        model.load_state_dict(torch.load(f"best_fold{fold}.pt"))
        fold_metrics = test_model(t_test_input, t_test_target, f_test_input, A_wave_test, batch_size)
        all_metrics.append(fold_metrics)

        '''Output'''
        rows = []
        for fold_idx, metrics in enumerate(all_metrics):
            for item_idx in range(len(metrics['mae'])):
                rows.append({
                    'Fold': fold_idx + 1,
                    'Item': item_idx + 1,
                    'MAE': float(metrics['mae'][item_idx].item()),
                    'MSE': float(metrics['mse'][item_idx].item()),
                    'RMSE': float(metrics['rmse'][item_idx].item()),
                    'MAPE': float(metrics['mape'][item_idx].item())
                })

        df_metrics = pd.DataFrame(rows)

        fold_avg = df_metrics.groupby('Fold', as_index=False).mean(numeric_only=True)
        fold_avg['Item'] = 'Average'

        df_metrics = pd.concat([df_metrics, fold_avg], ignore_index=True)

        avg_rows = df_metrics[df_metrics['Item'] == 'Average']
        final_avg = {
            'Fold': 'Overall Avg',
            'Item': 'All Items',
            'MAE': avg_rows['MAE'].mean(),
            'MSE': avg_rows['MSE'].mean(),
            'RMSE': avg_rows['RMSE'].mean(),
            'MAPE': avg_rows['MAPE'].mean(),
        }
        df_metrics = pd.concat([df_metrics, pd.DataFrame([final_avg])], ignore_index=True)

        df_metrics.to_csv("fold_item_metrics.csv", index=False)

    # 汇总所有 folds 的平均指标
    avg_mae = np.mean([np.mean(m['mae']) for m in all_metrics])
    avg_mse = np.mean([np.mean(m['mse']) for m in all_metrics])
    avg_rmse = np.mean([np.mean(m['rmse']) for m in all_metrics])
    avg_mape = np.mean([np.mean(m['mape']) for m in all_metrics])

    print("\n=== Final Evaluation Across All Folds ===")
    print(f"Average MAE:  {avg_mae:.6f}")
    print(f"Average MSE:  {avg_mse:.6f}")
    print(f"Average RMSE: {avg_rmse:.6f}")
    print(f"Average MAPE: {avg_mape:.6f}")

# ===================================================================================================================
'''交叉检验代码'''
#     from sklearn.model_selection import KFold
#
#     t_original_data = trade_data
#     print(t_original_data.shape[2])
#     f_original_data = features
#
#     # 创建KFold对象
#     kf = KFold(n_splits=10, shuffle=False)
#
#     # 存储每个折叠的损失
#     test_losses = []
#     test_maes = []
#     test_mses = []
#     test_rmses = []
#     test_mapes = []
#
#     for train_index, test_index in kf.split(range(t_original_data.shape[2])):  # 使用行数进行划分
#         print(kf)
#
#         t_train_original_data, t_test_original_data = t_original_data[:,:,train_index], t_original_data[:,:,test_index]
#         f_train_original_data, f_test_original_data = f_original_data[:,:,train_index], f_original_data[:,:,test_index]
#
#         # 生成输入和目标
#         t_train_input, t_train_target = generate_dataset(t_train_original_data, num_timesteps_input,
#                                                          num_timesteps_output)
#         t_test_input, t_test_target = generate_dataset(t_test_original_data, num_timesteps_input, num_timesteps_output)
#
#         f_train_input, f_train_target = generate_dataset(f_train_original_data, num_timesteps_input,
#                                                          num_timesteps_output)
#         f_test_input, f_test_target = generate_dataset(f_test_original_data, num_timesteps_input, num_timesteps_output)
#
#         # 转换为浮点数
#         t_train_input = t_train_input.float()
#         t_train_target = t_train_target.float()
#         t_test_input = t_test_input.float()
#         t_test_target = t_test_target.float()
#         f_train_input = f_train_input.float()
#         f_train_target = f_train_target.float()
#         f_test_input = f_test_input.float()
#         f_test_target = f_test_target.float()
#
#         # 这里可以继续进行模型训练和测试
#         # 初始化模型
#         model = STGCNAttModel(node_size, feature_dim1, feature_dim2, num_timesteps_input, num_timesteps_output)
#         optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#         loss_criterion = nn.MSELoss()
#         from utils import a_train_crossvalidation,a_test_crossvalidation
#         # 训练模型
#         for epoch in range(epochs):
#             print(epoch)
#             model.train()
#             optimizer.zero_grad()
#             a_hat_train = a_train_crossvalidation('1001_切片.csv',12,train_index)
#             a_hat_train = a_hat_train.float()
#             out = model(t_train_input, f_train_input, a_hat_train)  # a_hat_train需要根据你的逻辑进行处理
#             loss = loss_criterion(out, t_train_target)
#             loss.backward()
#             optimizer.step()
#
#
#
#         # 测试模型
#         model.eval()
#         with torch.no_grad():
#             a_hat_test = a_test_crossvalidation('1001_切片.csv', 12, test_index)
#             a_hat_test = a_hat_test.float()
#             out = model(t_test_input, f_test_input, a_hat_test)  # a_hat_test需要根据你的逻辑进行处理
#             val_loss = loss_criterion(out, t_test_target)
#
#             mae = torch.mean(torch.abs(out - t_test_target)).item()
#             mse = torch.mean((out - t_test_target) ** 2).item()
#             rmse = np.sqrt(mse)
#             non_zero_mask = t_test_target != 0
#             mape = ((t_test_target[non_zero_mask] - out[non_zero_mask]) / t_test_target[non_zero_mask]).abs().mean()
#
#             test_losses.append(val_loss.item())
#             test_maes.append(mae)
#             test_mses.append(mse)
#             test_rmses.append(rmse)
#             test_mapes.append(mape)
#
#
#     # 输出交叉验证结果
#     print("Test Losses for each fold:", test_losses)
#     print("Test MAE for each fold:", test_maes)
#     print("Test MSE for each fold:", test_mses)
#     print("Test RMSE for each fold:", test_rmses)
#     print("Test MAPE for each fold:", test_mapes)
#
#     print("Average Test Loss:", np.mean(test_losses))
#     print("Average Test MAE:", np.mean(test_maes))
#     print("Average Test MSE:", np.mean(test_mses))
#     print("Average Test RMSE:", np.mean(test_rmses))
#     print("Average Test MAPE:", np.mean(test_mapes))
