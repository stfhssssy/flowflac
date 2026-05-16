import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.amp import autocast, GradScaler
import copy
from .utils import soft_update, hard_update
from .model import QNetwork, ValueNetwork, Policy_flow, C51QNetwork
import time
from torch.optim import Adam
import torch.optim as optim
import numpy as np

from utilis.utils import RunningMeanStd



mode = "max-autotune"
compile_model = True

class flowAC(object):
    def __init__(self, num_inputs, action_space, args):
        self.num_inputs = num_inputs
        self.gamma = args.gamma
        self.tau = args.tau
        self.noise_level = args.epsilon
        self.action_space = action_space
        self.sample_count = 0

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.device = torch.device(f"cuda:{args.device}" if args.cuda else "cpu")
        self.amp_enabled = args.cuda and torch.cuda.is_available()
        self.amp_dtype = torch.bfloat16
        self.scaler = GradScaler(enabled=self.amp_enabled and self.amp_dtype == torch.float16)

        self.obs_norm_clip = getattr(args, "obs_norm_clip", 10.0)
        self.obs_norm_eps = getattr(args, "obs_norm_eps", 1e-8)
        self.normalize_obs = bool(getattr(args, "normalize_obs", False))
        self.obs_rms = RunningMeanStd(num_inputs, device=self.device) if self.normalize_obs else None

        # LAC: Target kinetic energy (coef * action_dim)
        target_kinetic_coef = float(getattr(args, "target_kinetic_coef", 2.5))
        self.target_kinetic = target_kinetic_coef * action_space.shape[0]

        # LAC: Adaptive temperature parameter (alpha = exp(log_alpha))
        init_log_alpha = float(getattr(args, "init_log_alpha", 0.0))
        self.auto_alpha = bool(getattr(args, "auto_alpha", True))
        self.log_alpha = torch.tensor(
            [init_log_alpha],
            requires_grad=self.auto_alpha,
            device=self.device,
        )
        # Use a smaller LR for alpha to avoid overreacting.
        self.alpha_optim = optim.Adam([self.log_alpha], lr=args.lr * 0.1) if self.auto_alpha else None

        self.distributional_critic = bool(getattr(args, "distributional_critic", False))
        if self.distributional_critic:
            self.critic_num_atoms = int(getattr(args, "critic_num_atoms", 101))
            self.critic_v_min = float(getattr(args, "critic_v_min", -150.0))
            self.critic_v_max = float(getattr(args, "critic_v_max", 150.0))
            self.c51_atoms = torch.linspace(
                self.critic_v_min, self.critic_v_max, self.critic_num_atoms, device=self.device
            )
            self.c51_delta = (self.critic_v_max - self.critic_v_min) / (self.critic_num_atoms - 1)

        # ---------------------- Policy Network ----------------------
        if self.policy_type == "Flow":
            self.policy = Policy_flow(num_inputs, action_space.shape[0], args.hidden_size, args.steps, action_space).to(self.device)
            self.policy_optim = optim.Adam(self.policy.parameters(), lr=args.lr)
        else:
            pass

        # ---------------------- Critic Networks ----------------------
        if self.distributional_critic:
            self.critic = C51QNetwork(
                num_inputs,
                action_space.shape[0],
                args.hidden_size,
                num_atoms=self.critic_num_atoms,
            ).to(self.device)
        else:
            self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=args.lr)
        if self.distributional_critic:
            self.critic_target = C51QNetwork(
                num_inputs,
                action_space.shape[0],
                args.hidden_size,
                num_atoms=self.critic_num_atoms,
            ).to(self.device)
        else:
            self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        # ---------------------- Compile Models ----------------------
        if compile_model:
            self.critic = torch.compile(self.critic,mode=mode)
            self.critic_target = torch.compile(self.critic_target, mode=mode)
            # self.policy = torch.compile(self.policy, mode=mode)

    # only use for env step 
    def select_action(self, state, evaluate=False):

        # Noise schedule for exploration: In all tasks, we set the noise to 0.
        if not evaluate:
            self.sample_count += 1
            if self.sample_count % 1e5 == 0:
                self.noise_level = self.noise_level*0.8

        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        state = self._normalize_obs(state)

        if not evaluate:
            action, _, _ = self.policy.sample_env(state)
            noise = torch.rand_like(action) * 0.01 * self.noise_level
            noise = torch.clamp(noise, -0.25, 0.25)
            action = action + noise
        else:
            with torch.no_grad():
                action, _, _ = self.policy.sample_env(state)
        
        return action.detach().cpu().numpy()[0].clip(self.action_space.low, self.action_space.high)

    @torch.no_grad()
    def observe(self, state, next_state=None):
        if self.obs_rms is None:
            return

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        self.obs_rms.update(state_tensor)
        if next_state is not None:
            next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
            self.obs_rms.update(next_state_tensor)

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if self.obs_rms is None:
            return obs
        return self.obs_rms.normalize(obs, clip=self.obs_norm_clip, eps=self.obs_norm_eps)

    @torch.compile(mode=mode)
    def update_critic(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        """
        Critic update.
        - If distributional_critic: C51 cross-entropy on projected distribution.
        - Else: MSE TD error on scalar Q.
        Both include LAC kinetic penalty in the target:  r + gamma * (Q - alpha * kinetic).
        """
        with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
            with torch.no_grad():
                next_state_action, next_kinetic, _ = self.policy.sample(next_state_batch)
                alpha = self.log_alpha.exp()

                if self.distributional_critic:
                    qf1_next_target_logits, qf2_next_target_logits = self.critic_target(
                        next_state_batch, next_state_action
                    )
                    next_prob_1 = F.softmax(qf1_next_target_logits.float(), dim=-1)
                    next_prob_2 = F.softmax(qf2_next_target_logits.float(), dim=-1)

                    qf1_next_target = (next_prob_1 * self.c51_atoms).sum(dim=-1, keepdim=True)
                    qf2_next_target = (next_prob_2 * self.c51_atoms).sum(dim=-1, keepdim=True)
                    use_q1 = (qf1_next_target <= qf2_next_target)
                    next_prob = torch.where(use_q1, next_prob_1, next_prob_2)

                    # Project (r + gamma * (z - alpha * kinetic)) onto fixed support.
                    target_z = reward_batch + mask_batch * self.gamma * self.c51_atoms.view(1, -1)
                    target_z = target_z - (mask_batch * self.gamma * alpha * next_kinetic)
                    target_z = target_z.clamp(self.critic_v_min, self.critic_v_max)

                    b = (target_z - self.critic_v_min) / self.c51_delta
                    l = b.floor().to(torch.int64)
                    u = b.ceil().to(torch.int64)
                    l = l.clamp(0, self.critic_num_atoms - 1)
                    u = u.clamp(0, self.critic_num_atoms - 1)

                    m = torch.zeros_like(next_prob)
                    m_l = (u.to(b.dtype) - b)
                    m_u = (b - l.to(b.dtype))
                    eq = (u == l)
                    m_l = torch.where(eq, torch.ones_like(m_l), m_l)
                    m_u = torch.where(eq, torch.zeros_like(m_u), m_u)
                    m.scatter_add_(1, l, next_prob * m_l)
                    m.scatter_add_(1, u, next_prob * m_u)
                    target_dist = m
                else:
                    qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                    next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target - alpha * next_kinetic)

            # Update critic
            if self.distributional_critic:
                qf1_logits, qf2_logits = self.critic(state_batch, action_batch)
                log_p1 = F.log_softmax(qf1_logits.float(), dim=-1)
                log_p2 = F.log_softmax(qf2_logits.float(), dim=-1)
                qf1_loss = -(target_dist * log_p1).sum(dim=-1).mean()
                qf2_loss = -(target_dist * log_p2).sum(dim=-1).mean()
                qf_loss = qf1_loss + qf2_loss
            else:
                qf1, qf2 = self.critic(state_batch, action_batch)
                # Keep two independent targets to avoid accidental graph aliasing.
                qf1_loss = F.mse_loss(qf1, next_q_value)
                qf2_loss = F.mse_loss(qf2, next_q_value.clone())
                qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        self.scaler.scale(qf_loss).backward()
        self.scaler.step(self.critic_optim)
        self.scaler.update()
        return None


    @torch.compile(mode=mode)
    def update_policy(self, state_batch):
        """
        LAC policy + temperature update.
        Actor loss:  E[ -Q(s,a) + alpha * kinetic ]
        Alpha update (SAC-style on log_alpha): match mean kinetic to target_kinetic.
        """
        with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
            action, kinetic, _ = self.policy.sample(state_batch)
            alpha = self.log_alpha.exp()

            if self.distributional_critic:
                qf1_pi_logits, qf2_pi_logits = self.critic(state_batch, action)
                qf1_pi = (F.softmax(qf1_pi_logits.float(), dim=-1) * self.c51_atoms).sum(dim=-1, keepdim=True)
                qf2_pi = (F.softmax(qf2_pi_logits.float(), dim=-1) * self.c51_atoms).sum(dim=-1, keepdim=True)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
            else:
                qf1_pi, qf2_pi = self.critic(state_batch, action)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)

            policy_loss = (-min_qf_pi + alpha.detach() * kinetic).mean()

        # Update policy
        self.policy_optim.zero_grad()
        self.scaler.scale(policy_loss).backward()
        self.scaler.step(self.policy_optim)
        self.scaler.update()

        if self.auto_alpha:
            # Update alpha (SAC-style on log_alpha; stable when alpha is small).
            # We intentionally detach kinetic to avoid gradients flowing into the policy.
            kinetic_mean = kinetic.detach().mean()
            alpha_loss = self.log_alpha * (self.target_kinetic - kinetic_mean)

            self.alpha_optim.zero_grad()
            self.scaler.scale(alpha_loss).backward()
            self.scaler.step(self.alpha_optim)
            self.scaler.update()
        return None


    def update_parameters(self, memory, batch_size, updates, total_numsteps=None):
        """
        Update: Critic and Policy updates
        """
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        state_batch = self._normalize_obs(state_batch)
        next_state_batch = self._normalize_obs(next_state_batch)
        
        self.update_critic(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)

        # Update policy and alpha (with delayed update)
        if updates % self.target_update_interval == 0:
            self.update_policy(state_batch)
            with torch.no_grad():
                soft_update(self.critic_target, self.critic, self.tau)

        return None

    # Save model parameters
    def save_checkpoint(self, path, i_episode):
        ckpt_path = path + '/' + '{}.torch'.format(i_episode)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict(),
                    'alpha_optimizer_state_dict': self.alpha_optim.state_dict() if self.alpha_optim else None,
                    'log_alpha': self.log_alpha,
                    'obs_rms_state_dict': self.obs_rms.state_dict() if self.obs_rms is not None else None,
                    },
                    ckpt_path)

    # Load model parameters
    def load_checkpoint(self, path, i_episode, evaluate=False):
        # ckpt_path = path + '/' + '{}.torch'.format(i_episode)
        ckpt_path = path + '/' + 'checkpoint/'+'best.torch'
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            # Load alpha state if available
            if 'log_alpha' in checkpoint:
                self.log_alpha.data.copy_(checkpoint['log_alpha'].data)
            if self.alpha_optim is not None and checkpoint.get('alpha_optimizer_state_dict') is not None:
                self.alpha_optim.load_state_dict(checkpoint['alpha_optimizer_state_dict'])

            obs_rms_state_dict = checkpoint.get('obs_rms_state_dict')
            if obs_rms_state_dict is not None:
                if self.obs_rms is None:
                    self.normalize_obs = True
                    self.obs_rms = RunningMeanStd(self.num_inputs, device=self.device)
                self.obs_rms.load_state_dict(obs_rms_state_dict)

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
