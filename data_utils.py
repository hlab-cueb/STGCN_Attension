import os

import numpy as np
import pandas as pd
import torch

# 矩阵归一化
def min_max_normalize(X):  # X格式为tensor
    min_vals = np.min(X, axis=(0, 1))
    max_vals = np.max(X, axis=(0, 1))
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # 避免分母为0

    # min_vals.reshape(1, 1, -1) 改变数组形状，第一、二个维度是1，第三个维度-1，即自动推断出维度的大小以适应原数组的数据总量
    # 改后的min_vals和range_vals可用于广播相减
    X = (X - min_vals.reshape(1, 1, -1)) / range_vals.reshape(1, 1, -1)

    return X

# 获取目标76个国家的Code
def get_target_country_codes():
    df_f = pd.read_csv(r'../../data/平均气温数据.csv')
    target_iso = df_f['IsoCode'].to_list()

    df = pd.read_csv(r'../../data/1001.csv', header=None)
    df = df.iloc[:, 6:8]  # 包含 Code 和 IsoCode
    df.columns = ['Code', 'IsoCode']

    target_codes_df = df[df['IsoCode'].isin(target_iso)]
    target_codes_df = target_codes_df.drop_duplicates(subset='IsoCode', keep='first')
    target_codes = target_codes_df['Code'].tolist()

    return sorted(target_codes)


# 将原始数据合并并预处理
def data_pre_process(folder_path):
    # 将多个文件连接成一个combine_df
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not files:
        print('No csv files found.')
        return None

    dfs = []
    for file in files:
        file_path = os.path.join(folder_path, file)
        # 防止csv文件为空
        if os.path.getsize(file_path) == 0:
            # empty_files.append(file_path)
            continue
        df = pd.read_csv(file_path, header=None).iloc[:, 1:]
        dfs.append(df)

    combined_df = pd.concat(dfs, axis=0, ignore_index=True)
    combined_df.columns = pd.read_csv("../../data/表头.csv").columns[1:]

    # 删除partnerCode为0的数据行
    combined_df = combined_df[combined_df['partnerCode'] != 0].reset_index(drop=True)
    # 处理Import&Export的汇报口径问题
    # 将Export的行中进出口国家信息调换
    export_mask = combined_df['flowDesc'] == 'Export'
    combined_df.loc[
        export_mask, ['reporterISO', 'reporterDesc', 'partnerISO', 'partnerDesc']] = (
        combined_df.loc[export_mask, ['partnerISO', 'partnerDesc', 'reporterISO',
                                      'reporterDesc']].values)
    combined_df.loc[export_mask, ['reporterCode', 'partnerCode']] = combined_df.loc[
        export_mask, ['partnerCode', 'reporterCode']].values.astype(int)
    combined_df.loc[export_mask, ['flowCode', 'flowDesc']] = [['M', 'Import']]
    # 重复数据行用均值合并
    cols_to_avg = combined_df.columns[[31, 35, 37, 39, 41, 42, 43]]
    sort_index = combined_df.columns[:17].tolist()
    combined_df[cols_to_avg] = combined_df.groupby(sort_index)[cols_to_avg].transform('mean')
    combined_df = combined_df.drop_duplicates(subset=sort_index, keep='first')

    return combined_df


# 贸易流量数据特征对齐&筛选
def align_pre_process(df, target_country_codes):
    # 删除特征不存在的节点
    df = df[~df[['reporterISO', 'partnerISO']].isin(['E19', '_X ', 'EUR', 'X2 ', 'S19']).any(axis=1)]

    # 只保留目标国家间的贸易数据
    df = df[df['reporterCode'].isin(target_country_codes) & df['partnerCode'].isin(target_country_codes)]

    # 截取出达到总体贸易流量70%的边
    df_sorted = df.sort_values(by='netWgt', ascending=False)
    total_netWgt = df_sorted['netWgt'].sum()
    df_sorted['cumulativeNetWgt'] = df_sorted['netWgt'].cumsum()
    df_top_70 = df_sorted[df_sorted['cumulativeNetWgt'] <= total_netWgt * 0.7]
    df_top_70 = df_top_70.drop(columns=['cumulativeNetWgt'])

    df_top_70.to_csv(save_path + folder_path + '.csv', index=False)
    return df_top_70


# 将合并数据转化为贸易流量数据
def get_trade_flow(df):
    # 将df处理为贸易流量trade_flow_df
    trade_flow_df = df.pivot_table(
        index=['reporterCode', 'partnerCode'],
        columns='period',
        values='netWgt',
        aggfunc='sum',
        fill_value=None
    ).reset_index()
    # 按时间顺序排序列
    time_columns_lst = sorted([col for col in trade_flow_df.columns if col not in ['reporterCode', 'partnerCode']])
    trade_flow_df = trade_flow_df[['reporterCode', 'partnerCode'] + time_columns_lst]
    trade_flow_df = trade_flow_df.drop_duplicates().reset_index(drop=True)

    return trade_flow_df


# 生成图邻接矩阵 - 贸易流量矩阵

# 生成贸易流量数据
def load_metr_la_data(df, target_country_code):
    # 建立国家编号到矩阵索引的映射字典
    nodes = sorted(target_country_code)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    reporter_codes = df['reporterCode'].map(node_to_idx).values
    partner_codes = df['partnerCode'].map(node_to_idx).values

    # 生成贸易流量矩阵
    trade_flow_cols = df.columns[2:]
    trade_flow_matrix = np.zeros((len(nodes), len(nodes), len(trade_flow_cols)))

    for i, col in enumerate(trade_flow_cols):
        # reporter_codes, partner_codes, i 传入2个尺寸一致的列表，返回值将为一个1D-Array
        trade_flow_matrix[reporter_codes, partner_codes, i] = df[col].fillna(0).values

    return trade_flow_matrix


# 生成图节点特征 - 特征数据
def get_feature():
    df = pd.read_csv(source_path + '1001.csv', header=None)
    df = df.iloc[:, 6:8]
    df.columns = ['Code', 'IsoCode']

    file_lst = ['平均气温数据.csv', '最低气温数据.csv', '最高气温数据.csv', '降水量数据.csv', '汇率数据.csv']
    result_dfs = []

    for file in file_lst:
        feature_df = pd.read_csv(source_path + file)
        # 将Code匹配特征数据
        result = pd.merge(feature_df, df, on='IsoCode', how='left')
        result.drop(result[result['Code'] == 736].index, inplace=True)  # 不考虑南苏丹
        result = result.drop_duplicates().sort_values(by='Code').reset_index(drop=True)
        if file == '汇率数据.csv':
            result = result.drop(columns=['IsoCode', 'Code'])
        else:
            result = result.drop(columns=['IsoName', 'IsoCode', 'Code'])

        result_dfs.append(result)

    # 生成时间戳(月份索引)
    timestamp = result_dfs[0].copy()
    month = [i % 12 + 1 for i in range(timestamp.shape[1])]
    timestamp[:] = month
    result_dfs.append(timestamp)

    # 生成2D-Array组成的列表(国家(节点)×时间×特征) - 维度从后往前看
    features = np.array([df.values for df in result_dfs])
    # 转化为(时间×国家(节点)×特征)
    features = features.transpose((1, 0, 2))
    return features


# 生成二进制邻接矩阵
def get_adj_matrices(df, target_codes):
    # 保证只保留目标国家之间的贸易数据
    df = df[df['reporterCode'].isin(target_codes) & df['partnerCode'].isin(target_codes)]

    time_steps = df.columns[2:]
    num_time_steps = len(time_steps)
    num_nodes = len(target_codes)

    # 创建国家编码到索引的映射（统一顺序）
    code_to_index = {code: idx for idx, code in enumerate(target_codes)}

    adjacency_matrices = []

    for i in range(num_time_steps):
        adj = np.zeros((num_nodes, num_nodes))
        # 填充邻接矩阵
        current_time_step = time_steps[i]
        for _, row in df.iterrows():
            reporter_code = row['reporterCode']
            partner_code = row['partnerCode']
            if pd.notna(row[current_time_step]):  # 检查是否有数据
                reporter_idx = code_to_index[reporter_code]
                partner_idx = code_to_index[partner_code]
                adj[reporter_idx, partner_idx] = 1  # 设置为1

        np.fill_diagonal(adj, 1)
        adjacency_matrices.append(adj)

    adjacency_tensor = torch.tensor(np.array(adjacency_matrices), dtype=torch.float32)  # [T, N, N]
    return adjacency_tensor


def adj_crossvalidation(filename, x, time_indices, target_codes):
    """
    通用交叉验证邻接矩阵生成函数，用于 train/test 阶段。
    :param filename: CSV 文件路径
    :param x: 输入时间步
    :param time_indices: 当前 fold 中的 train 或 test 索引（基于 features 时间列的索引）
    :param target_codes: 全部目标国家的 code 列表（长度为76，用于统一顺序）
    """
    data = pd.read_csv(filename)

    new_index = np.concatenate(([0, 1], time_indices + 2))  # 第0,1列是 reporterCode, partnerCode
    data = data.iloc[:, new_index]

    code_to_index = {code: idx for idx, code in enumerate(target_codes)}
    time_steps = data.columns[2:]
    num_nodes = len(target_codes)
    num_time_steps = len(time_steps)
    adjacency_matrices = []

    for i in range(num_time_steps - x):
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        for j in range(i, i + 1):
            current_time_step = time_steps[i]
            for _, row in data.iterrows():
                reporter_code = row['reporterCode']
                partner_code = row['partnerCode']
                if pd.notna(row[current_time_step]) and \
                        reporter_code in code_to_index and partner_code in code_to_index:
                    reporter_index = code_to_index[reporter_code]
                    partner_index = code_to_index[partner_code]
                    adjacency_matrix[reporter_index, partner_index] = 1

        for i in range(76):
            adjacency_matrix[i, i] = 1

        adjacency_matrices.append(adjacency_matrix)

    adjacency_tensor = torch.tensor(np.array(adjacency_matrices))
    return adjacency_tensor


if __name__ == '__main__':
    # 生成固定国家列表
    target_country_codes = get_target_country_codes()

    # 生成图邻接矩阵
    source_path = '..\\..\\data\\'
    dest_path = "..\\Processed_data\\"
    folder_paths = [entry.name for entry in os.scandir(source_path) if entry.is_dir()]

    # folders_path = ["1001"]
    # empty_files = []
    trade_flow_lst = []
    A_lst = []
    feature_lst = []

    # 生成图节点特征 - 特征数据
    features = get_feature()
    features = min_max_normalize(features)
    features = torch.tensor(features, dtype=torch.float32)

    for folder_path in folder_paths:
        file_path = source_path + folder_path + "\\"
        save_path = dest_path + folder_path + "\\"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        df = data_pre_process(file_path)
        df = align_pre_process(df, target_country_codes)

        # 贸易流量矩阵生成
        trade_flow_df = get_trade_flow(df)
        # trade_flow_df.to_csv(save_path + 'trade_flow.csv', index=False)
        trade_flow_matrix = load_metr_la_data(trade_flow_df, target_country_codes)
        trade_flow = min_max_normalize(trade_flow_matrix)
        torch.save(trade_flow,
                   save_path + 'trade_flow_matrix.pt')

        # 邻接矩阵生成
        A = get_adj_matrices(trade_flow_df, target_country_codes)

        if folder_path not in {"110314", "110411", "110421", "init"}:
            if trade_flow.shape == (76, 76, 276):
                trade_flow = torch.tensor(trade_flow, dtype=torch.float32)
                trade_flow_lst.append(trade_flow)
                A_lst.append(A)
                feature_lst.append(features.clone().detach())

            else:
                print(f"Skip {folder_path} {trade_flow.shape}")

        print("Process " + folder_path + "'s data successfully")

    multi_trade = torch.stack(trade_flow_lst, dim=0)
    multi_A = torch.stack(A_lst, dim=0)
    multi_feature = torch.stack(feature_lst, dim=0)

    torch.save(multi_trade, dest_path + "combined_data\\multi_trade.pt")
    torch.save(multi_A, dest_path + "combined_data\\multi_A.pt")
    torch.save(multi_feature, dest_path + "combined_data\\multi_feature.pt")

    # print(empty_files)
