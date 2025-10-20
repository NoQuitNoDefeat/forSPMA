# SPMA节点端机器学习项目

一个生产就绪的SPMA节点端机器学习系统，使用TCN预测非平稳COS（Channel Occupancy Status），DQN选择退避动作。

## 快速开始（5个命令）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 生成模拟数据
python scripts/make_fake_data.py

# 3. 训练TCN模型
python train_tcn.py

# 4. 训练DQN模型
python train_dqn.py

# 5. 运行推理基准测试
python scripts/bench_infer.py
```

## 数据架构

### 训练数据格式 (CSV)
- `timestamp`: 时间戳
- `cos_0` 到 `cos_13`: 14个信道的占用状态 (0-1)
- `backoff_action`: 退避动作 (-2到+2)
- `reward`: 奖励信号
- `capacity_used`: 当前容量使用率

### 特征工程
- TCN输入: 14维COS序列 + 2维优先级信息
- DQN状态: 10维状态向量（包括历史COS、容量使用率、奖励等）
- 动作空间: 5个离散动作 {-2, -1, 0, +1, +2} 对应退避步数

## 模型架构

### TCN (Temporal Convolutional Network)
- 因果卷积确保时间因果关系
- 膨胀卷积捕获长期依赖
- 残差连接加速训练
- 深度可分离卷积减少参数量
- 输出: 非平稳COS预测

### DQN (Deep Q-Network)
- 小MLP网络 (64->32->5)
- 经验重放缓冲区
- 目标网络稳定训练
- Epsilon贪婪策略探索
- 输出: 退避动作Q值

## 与ns-3集成

当前使用存根环境进行开发。集成ns-3的步骤：

1. 实现 `envs/ns3_bridge_env.py` 中的 `NS3BridgeEnvironment` 类
2. 替换 `envs/spma_stub_env.py` 为实际的ns-3接口
3. 配置ns-3参数在 `config.yaml` 中

```python
# TODO: 替换为ns-3桥接
class NS3BridgeEnvironment:
    def __init__(self, ns3_config):
        # 连接到ns-3仿真器
        # 设置网络拓扑
        # 初始化信道状态
        pass
```

## ARM部署和INT8导出

### 导出模型
```bash
python export_int8.py --model tcn --output models/tcn_int8.ptl
python export_int8.py --model dqn --output models/dqn_int8.ptl
```

### 在ARM上运行推理
```python
# 使用TorchScript优化的推理
import torch

# 加载INT8模型
model = torch.jit.load('models/tcn_int8.ptl')
model.eval()

# 推理循环
with torch.no_grad():
    prediction = model(input_tensor)
```

### ExecuTorch集成
```python
# TODO: 集成ExecuTorch进行更高效的ARM推理
# from executorch import exir
# model = exir.load('models/tcn_int8.pte')
```

## 延迟性能分析

### 基准测试
```bash
python scripts/bench_infer.py --model tcn --iterations 1000
python scripts/bench_infer.py --model dqn --iterations 1000
```

### 性能目标
- TCN推理: < 1ms (ARM Cortex-A78)
- DQN推理: < 0.5ms (ARM Cortex-A78)
- 内存占用: < 10MB

### 优化技巧
1. 使用INT8量化减少模型大小
2. 批处理多个推理请求
3. 预分配张量避免内存分配开销
4. 使用SIMD指令优化关键路径

## 配置说明

所有配置在 `config.yaml` 中：

```yaml
# TCN配置
tcn:
  window_size: 32        # 输入序列长度
  input_dim: 14          # COS特征维度
  channels: 16           # 卷积通道数
  levels: 3              # TCN层级数
  kernel_size: 3         # 卷积核大小
  n_prio: 2              # 优先级特征数

# DQN配置
dqn:
  state_dim: 10          # 状态维度
  action_dim: 5          # 动作数量
  hidden_dims: [64, 32]  # 隐藏层维度
  epsilon_start: 1.0     # 初始探索率
  epsilon_end: 0.01      # 最终探索率
  epsilon_decay: 0.995   # 探索率衰减

# 训练配置
training:
  batch_size: 32
  learning_rate: 0.001
  buffer_size: 10000
  target_update: 100
```

## 测试

```bash
# 运行所有测试
pytest -q

# 运行特定测试
pytest tests/test_tcn.py -v
pytest tests/test_dqn.py -v
```

## 文件结构

```
spma-ml/
├── README.md                 # 项目文档
├── requirements.txt          # Python依赖
├── config.yaml              # 配置文件
├── data/                    # 数据目录
│   ├── train.csv           # 训练数据
│   └── val.csv             # 验证数据
├── models/                  # 模型定义
│   ├── __init__.py
│   ├── tcn.py              # TCN模型
│   └── dqn.py              # DQN模型
├── envs/                    # 环境定义
│   ├── __init__.py
│   ├── spma_stub_env.py    # 存根环境
│   └── ns3_bridge_env.py   # ns-3桥接环境
├── utils/                   # 工具函数
│   ├── __init__.py
│   ├── dataset.py          # 数据集加载器
│   ├── metrics.py          # 评估指标
│   └── logger.py           # 日志工具
├── train_tcn.py            # TCN训练脚本
├── train_dqn.py            # DQN训练脚本
├── export_int8.py          # INT8导出脚本
├── infer_node.py           # 节点推理脚本
├── scripts/                # 辅助脚本
│   ├── make_fake_data.py   # 生成模拟数据
│   └── bench_infer.py      # 推理基准测试
├── tests/                  # 测试套件
│   ├── test_tcn.py
│   ├── test_dqn.py
│   └── test_dataset.py
└── .gitignore              # Git忽略文件
```

## 贡献指南

1. 遵循PEP 8代码风格
2. 添加类型提示和文档字符串
3. 为新功能编写测试
4. 更新配置文件而非硬编码参数
5. 保持文件长度 < 300行

## 许可证

MIT License
