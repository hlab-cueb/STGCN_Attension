# The Template for Hlab code（可根据项目需要进行修改）

## Requirements（项目环境依赖）

* Python >= 3.7
* PyTorch >= 1.9
* NumPy
* Pandas

> 完整依赖项已列入 `requirements.txt`，请通过以下命令安装：

```bash
pip install -r requirements.txt
```
---

## Training（训练方法与数据组织）

### 1. 模型训练方式 | Training Method

配置项详见 `config.py` 文件。可通过如下命令启动训练：

```bash
python main.py
```

模型结构定义在 `model.py`，包含 STGCN Block、自注意力机制、交叉注意力等模块。

---

### 2. 数据组织与格式 | Dataset

#### 输入张量格式说明：

| 输入名 Input Name     | 维度 Shape         | 描述 Description                            |
| ------------------ | ---------------- | ----------------------------------------- |
| `multi_trade.pt`   | `(P, 76, 76, T)` | 每个商品的贸易流量矩阵 Trade flow tensor per product |
| `multi_A.pt`       | `(P, T, 76, 76)` | 每个商品对应时间的邻接矩阵 Binary adjacency matrix     |
| `multi_feature.pt` | `(P, 76, F, T)`  | 每个商品的节点特征矩阵 Node feature tensor           |

> P: 商品数 Products，T: 时间步数 Time steps，F: 特征维度 Feature dimensions

#### 数据组织样例：

```python
# 每条数据包含一系列图结构输入
sample = {
    "trade_flow": torch.Tensor(P, 76, 76, T),  # 多商品贸易流量
    "adjacency": torch.Tensor(P, T, 76, 76),   # 二值邻接矩阵
    "node_features": torch.Tensor(P, 76, F, T) # 气候与经济特征
}
```

---

### 3. 代码功能特性 | Code Features

* [x] 支持多商品数据并联合建模
* [x] 完整的数据归一化、国家过滤与标准化处理流程
* [x] 支持按时间滑窗生成时序邻接张量
* [x] 自动构建节点特征：包括气温、降水、汇率等
* [x] 节点编码与邻接图构建模块已封装（支持通用调用）
* [ ] 拟支持图结构动态变化与边权学习
* [ ] 可扩展商品类型并自动匹配特征维度

---

## Test（模型测试）

运行如下脚本以进行模型评估或推理：

```bash
sh test.sh
```

>测试指标包括 RMSE、MAE、MAPE 等。

---

## 项目补充说明 | Additional Notes

### 文件结构说明

```
|- data_utils.py               # 数据预处理主模块
|- model.py                    # 多输入 STGCN 模型定义
|- main.py                     # 训练主函数
|- attention.py                # 注意力模块（FiLM / Cross / Self）
|- stgcn.py                    # STGCN block 模块定义
|- utils.py                    # 工具函数（窗口滑动函数）
|
|- Processed_data/            # 输出的模型输入数据
|  |- combined_data/          # 最终张量输出位置
|     |- multi_trade.pt
|     |- multi_A.pt
|     |- multi_feature.pt
|
|- assert/                    # 用于文档展示的图像资源
|  |- sample.png              # 数据流程图（示意）
|
|- README.md                  # 项目说明文件
|- requirements.txt           # 依赖文件
```

