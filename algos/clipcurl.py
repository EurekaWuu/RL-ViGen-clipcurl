import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import clip
from algos.curl import CURLAgent, CNNEncoder, CURLHead

class CLIPCURLHead(nn.Module):
    def __init__(self, repr_dim, clip_dim=512, temperature=0.07):
        super().__init__()
        self.temperature = temperature  # 控制相似度计算
        
        # CURL的对比学习head
        self.W = nn.Parameter(torch.rand(repr_dim, repr_dim))
        
        # CLIP的投影head
        self.clip_projector = nn.Sequential(
            nn.Linear(repr_dim, repr_dim),
            nn.LayerNorm(repr_dim),
            nn.ReLU(inplace=True),
            nn.Linear(repr_dim, clip_dim),
            nn.LayerNorm(clip_dim)
        )
        
    def compute_curl_logits(self, z_a, z_pos):
        Wz = torch.matmul(self.W, z_pos.T)  # [batch, batch]
        logits = torch.matmul(z_a, Wz)
        logits = logits / self.temperature 
        return logits
        
    def compute_clip_similarity(self, z_img, z_text):
        
        z_img = z_img.float()
        z_text = z_text.float()
        
        
        z_img = self.clip_projector(z_img)
        
        
        z_img = F.normalize(z_img, dim=-1)
        z_text = F.normalize(z_text, dim=-1)
        
        # 计算cosine相似度
        similarity = torch.matmul(z_img, z_text.T) / self.temperature
        return similarity
        
    def forward(self, z_a, z_pos, z_text):
        curl_logits = self.compute_curl_logits(z_a, z_pos)
        clip_sim = self.compute_clip_similarity(z_a, z_text)
        return curl_logits, clip_sim

class CLIPCURLAgent(CURLAgent):
    def __init__(
        self,
        obs_shape=None,
        action_shape=None,
        device=None,
        lr=None,
        critic_target_tau=None,
        update_every_steps=None,
        use_tb=None,
        num_expl_steps=None,
        hidden_dim=None,
        feature_dim=None,
        stddev_schedule=None,  
        stddev_clip=None,      
        # CLIP相关
        clip_model_name='ViT-B/32',
        clip_reward_scale=0.1,
        curl_weight=1.0,
        clip_weight=0.1,
        # 任务相关
        task_name=None,
        task_descriptions=None,
        **kwargs 
    ):
        
        super().__init__(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            lr=lr,
            critic_target_tau=critic_target_tau,
            update_every_steps=update_every_steps,
            use_tb=use_tb,
            num_expl_steps=num_expl_steps,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            stddev_schedule=stddev_schedule,  
            stddev_clip=stddev_clip,         
            **kwargs
        )
        
        
        self.action_dim = action_shape[0]
        
        
        self.task_name = task_name
        self.task_descriptions = task_descriptions
        
        
        self.clip_model, _ = clip.load(clip_model_name, device=self.device)
        # 冻结CLIP模型参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # 使用组合的head替换原有的curl_head
        self.curl_head = CLIPCURLHead(self.encoder.repr_dim).to(self.device)
        
        
        self.clip_reward_scale = clip_reward_scale
        self.curl_weight = curl_weight
        self.clip_weight = clip_weight
        
        
        if not self.task_descriptions:
            self.task_descriptions = [
                "agent performing the task successfully",
                "agent moving in a stable and controlled manner", 
                "agent achieving the desired goal"
            ]
            
        # 编码任务描述
        self.task_text = clip.tokenize(self.task_descriptions).to(self.device)
        
        # 预计算任务描述的特征
        with torch.no_grad():
            self.text_features = self.clip_model.encode_text(self.task_text)

    def compute_clip_reward(self, obs):
        with torch.no_grad():
            # 确保输入是float tensor并且在正确的设备上
            if isinstance(obs, np.ndarray):
                obs = torch.from_numpy(obs).float()
            obs = obs.to(self.device)
            
            # 如果是 3 维 => (9, 84, 84)，加一个 batch 维度 => (1, 9, 84, 84)
            # 这样在评估(只有1条数据)时也能与批量处理代码统一
            if obs.ndim == 3:
                obs = obs.unsqueeze(0)

            # obs 现在保证是 (batch_size, 9, 84, 84)
            clip_input = obs[:, -3:, :, :]

            clip_input = F.interpolate(
                clip_input,
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            )

            image_features = self.clip_model.encode_image(clip_input)

            encoded_input = self.encoder(obs)

            similarity = self.curl_head.compute_clip_similarity(
                encoded_input,
                self.text_features
            )
            
            clip_reward = similarity.mean(dim=1)

            # 如果是单条数据,返回标量值
            if obs.size(0) == 1:
                return clip_reward.item()
            # 否则返回批量奖励
            return clip_reward

    def update_curl(self, z_a, z_pos):
        metrics = dict()

        # CURL的对比学习loss
        curl_logits = self.curl_head.compute_curl_logits(z_a, z_pos)
        curl_labels = torch.arange(curl_logits.shape[0]).long().to(self.device)
        curl_loss = F.cross_entropy(curl_logits, curl_labels)
        
        # CLIP的相似度loss
        clip_similarity = self.curl_head.compute_clip_similarity(z_a, self.text_features)
        clip_loss = -clip_similarity.mean()  # 最大化相似度
        
        # 组合loss
        total_loss = self.curl_weight * curl_loss + self.clip_weight * clip_loss

        # 优化
        self.curl_optimizer.zero_grad(set_to_none=True)
        self.encoder_opt.zero_grad(set_to_none=True)
        total_loss.backward()
        self.curl_optimizer.step()
        self.encoder_opt.step()

        if self.use_tb:
            metrics['curl_loss'] = curl_loss.item()
            metrics['clip_loss'] = clip_loss.item()
            metrics['total_loss'] = total_loss.item()

        return metrics

    def update_critic(self, encoded_obs, action, reward, discount, encoded_next_obs, step, raw_obs=None):
        metrics = dict()

        # 更新critic网络,同时计算CLIP奖励
        # 使用 raw_obs 计算 clip_reward，不用已经encoder后的 obs
        with torch.no_grad():
            clip_reward = self.compute_clip_reward(raw_obs) 
            clip_reward = clip_reward.unsqueeze(1)
            total_reward = reward + self.clip_reward_scale * clip_reward

            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(encoded_next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(encoded_next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = total_reward + (discount * target_V)

        # 继续用 encoded_obs, action 来计算Q
        Q1, Q2 = self.critic(encoded_obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()
            metrics['clip_reward'] = clip_reward.mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = batch
        if step % 500 == 0:
            print(f"[DEBUG Agent Update] obs.shape={obs.shape}, action.shape={action.shape}")

        obs = obs.float().to(self.device)
        action = action.float().to(self.device)
        reward = reward.float().to(self.device)
        discount = discount.float().to(self.device)
        next_obs = next_obs.float().to(self.device)
        
        assert obs.shape[1:] == (9, 84, 84), f"Wrong observation shape: {obs.shape}"
        assert len(action.shape) == 2 and action.shape[1] == self.action_dim, \
               f"Wrong action shape: {action.shape}, expected (batch_size, {self.action_dim})"


        # 在对 obs 做数据增强与 encoder 处理前先保存一份原始 obs 用于 clip_reward
        raw_obs = obs.clone()
        
        obs = self.aug(obs)
        next_obs = self.aug(next_obs)
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()


        # 在 update_critic(...) 时把原始 obs 传进去 (raw_obs)，compute_clip_reward(...) 用像素数据
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step, raw_obs=raw_obs)
        )

        metrics.update(self.update_actor(obs.detach(), step))

        original_obs = self.aug(raw_obs.clone())  # 用原像素作CURL时的对比
        pos_obs = self.aug(raw_obs.clone())
        obs_enc = self.encoder(original_obs)
        with torch.no_grad():
            pos_enc = self.encoder(pos_obs)
        metrics.update(self.update_curl(obs_enc, pos_enc))

        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics 