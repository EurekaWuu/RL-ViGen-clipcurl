# RL-ViGen-CLIPCURL
一个结合了CLIP（对比语言-图像预训练）和CURL（对比无监督表示学习）的强化学习视觉泛化框架。提升强化学习智能体在复杂视觉环境中的泛化能力。

## 特点

- **多模态融合**: 结合CLIP的图像-文本理解能力与CURL的视觉对比学习
- **智能奖励机制**: 使用CLIP计算基于语义的辅助奖励信号
- **多环境支持**: 支持CARLA、DM-Control、Habitat、Robosuite等主流RL环境
- **视觉泛化**: 在背景、光照、视角、颜色等各种视觉变化下保持性能
- **跨具身泛化**: 支持不同机器人形态和任务场景

## 核心架构

### CLIPCURLAgent

1. **CLIPCURLHead**: 结合CLIP投影和CURL对比学习的混合头部网络
2. **多模态奖励**: 使用CLIP计算图像-文本匹配奖励，增强学习信号
3. **视觉表示学习**: CURL进行无监督的视觉特征学习

```python
# 核心算法使用示例
from algos.clipcurl import CLIPCURLAgent

agent = CLIPCURLAgent(
    obs_shape=(9, 84, 84),
    action_shape=action_spec.shape,
    device='cuda',
    lr=1e-4,
    clip_reward_scale=0.1
)
```

## 支持的环境

### 1. CARLA 自动驾驶
- **任务**: 自动驾驶导航
- **视觉变化**: 天气、光照、地图、车辆类型
- **配置**: `cfgs/carlaenv_config.yaml`

### 2. DM-Control 连续控制
- **任务**: 四足机器人行走、杂技等
- **视觉变化**: 背景干扰、光照、相机角度
- **配置**: `envs/DMCVGB/cfg/config.yaml`

### 3. Habitat 室内导航
- **任务**: 点导航、物体导航
- **视觉变化**: 场景、光照、相机参数
- **配置**: `envs/habitatVGB/cfg/config.yaml`

### 4. Robosuite 机器人操作
- **任务**: Door、Lift、TwoArmPegInhole
- **视觉变化**: 场景外观、动态背景、机器人类型
- **配置**: `envs/robosuiteVGB/cfg/robo_config.yaml`

## 安装

### 环境要求
- Python 3.8+
- CUDA 11.0+
- PyTorch 1.12+

```bash
# 克隆仓库
git clone https://github.com/gemcollector/RL-ViGen.git
cd RL-ViGen/

# 创建conda环境
conda create -n rl-vigen python=3.8
conda activate rl-vigen

# 运行安装脚本
bash setup/install_rlvigen.sh
```

### 特定安装

#### CARLA环境
```bash
# 下载CARLA 0.9.10
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.10.tar.gz
# 解压到third_party文件夹
```

#### Habitat环境
```bash
# 创建专用环境
conda create -n vigen-habitat python=3.8 cmake=3.14.0 -y
conda activate vigen-habitat
bash setup/install_vigen-habitat.sh
```

### 一键安装
```bash
bash setup/install_all.sh
```

## 使用

### 训练模型

#### CARLA训练
```bash
bash scripts/carlatrain.sh [wandb_group_name]
```

#### 通用训练
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    env=${env_name} \
    task=${task_name} \
    seed=1 \
    agent=clipcurl \
    use_wandb=true
```

### 模型评估
```bash
bash scripts/eval.sh

# 或者直接运行
python eval.py \
    env=${env_name} \
    task=${task_name} \
    agent=clipcurl \
    eval_mode=color_hard
```

### 自定义配置
```yaml
# 示例配置文件
agent:
  _target_: algos.clipcurl.CLIPCURLAgent
  obs_shape: [9, 84, 84]
  action_shape: [2]
  lr: 1e-4
  clip_reward_scale: 0.1
  feature_dim: 50
  hidden_dim: 512
```

## 核心算法原理

### 1. CLIP奖励机制
```python
def compute_clip_reward(self, obs):
    # 提取图像特征
    image_features = self.clip_model.encode_image(obs)
    # 获取任务描述的文本特征
    text_features = self.clip_model.encode_text(self.task_description)
    # 计算相似度作为奖励
    similarity = F.cosine_similarity(image_features, text_features)
    return similarity
```

### 2. CURL对比学习
```python
def update_curl(self, z_a, z_pos):
    # 计算对比学习损失
    curl_logits = self.curl_head.compute_curl_logits(z_a, z_pos)
    labels = torch.arange(curl_logits.shape[0]).long().to(self.device)
    curl_loss = F.cross_entropy(curl_logits, labels)
    return curl_loss
```

### 3. 混合奖励训练
```python
def update_critic(self, obs, action, reward, next_obs):
    # 计算CLIP奖励
    clip_reward = self.compute_clip_reward(obs)
    # 结合原始奖励和CLIP奖励
    total_reward = reward + self.clip_reward_scale * clip_reward
    # 正常的critic更新...
```


## 项目结构

```
RL-ViGen-clipcurl/
├── algos/                 # 核心算法实现
│   ├── clipcurl.py       # CLIP+CURL融合算法
│   ├── curl.py           # CURL基线算法
│   └── ...
├── envs/                 # 环境包装器
│   ├── carlaVGB/         # CARLA环境
│   ├── DMCVGB/           # DM-Control环境
│   ├── habitatVGB/       # Habitat环境
│   └── robosuiteVGB/     # Robosuite环境
├── cfgs/                 # 配置文件
├── scripts/              # 训练评估脚本
├── wrappers/             # 环境包装器
└── results/              # 实验结果
```

## 开发

### 添加新环境
1. 在`envs/`下创建新的环境文件夹
2. 实现环境包装器，符合标准的gym接口
3. 添加对应的配置文件
4. 在训练脚本中注册新环境

### 扩展算法
1. 继承`CLIPCURLAgent`类
2. 重写`update`方法实现新的学习逻辑
3. 添加新的网络模块
4. 在配置文件中注册新算法