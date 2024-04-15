# Adapted from https://github.com/vwxyzjn/cleanrl/blob/8cbca61360ef98660f149e3d76762350ce613323/cleanrl/sac_continuous_action.py

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer

from evaluate_policy import evaluate_policy

def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env.action_space.seed(seed)
        return env
    return thunk



# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, config):
        self.config = config
        hidden_dim = config["hidden_dim"]
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(hidden_dim, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        if self.config["spinning_up_log_std"]:
            log_std = torch.tanh(log_std)
            log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

default_config = {
    "learning_starts": 100,
    "autotune": True,
    "alpha": 1,
    "q_lr": 1e-3,
    "batch_size": 100,
    "policy_lr": 1e-3,
    "gamma": 0.99,
    "policy_frequency": 1,
    "target_network_frequency": 1,
    "tau": 0.005,
    "hidden_dim": 64,
    "spinning_up_log_std": False,
}

def train_cleanrl(config):
    import stable_baselines3 as sb3
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    device = torch.device("cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(config["environment_name"], config["seed"])])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs, config).to(device)
    qf1 = SoftQNetwork(envs, config["hidden_dim"]).to(device)
    qf2 = SoftQNetwork(envs, config["hidden_dim"]).to(device)
    qf1_target = SoftQNetwork(envs, config["hidden_dim"]).to(device)
    qf2_target = SoftQNetwork(envs, config["hidden_dim"]).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=config["q_lr"])
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=config["policy_lr"])

    # Automatic entropy tuning
    if config["autotune"]:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=config["q_lr"])
    else:
        alpha = config["alpha"]

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        config["n_steps"],
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    returns = []
    obs, _ = envs.reset(seed=config["seed"])
    for global_step in range(config["n_steps"]):
        # ALGO LOGIC: put action logic here
        if global_step < config["learning_starts"]:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        if global_step % config["evaluation_interval"] == 0:
            def policy(observation):
                return actor.get_action(torch.Tensor(observation).to(device).unsqueeze(0))[0][0].detach().cpu().numpy()
            current_returns = evaluate_policy(policy, config)
            print(f"Step: {global_step}, Returns: {np.array(current_returns).mean()}")
            returns.append(current_returns)

        # ALGO LOGIC: training.
        if global_step > config["learning_starts"]:
            data = rb.sample(config["batch_size"])
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * config["gamma"] * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % config["policy_frequency"] == 0:  # TD 3 Delayed update support
                for _ in range(
                    config["policy_frequency"]
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if config["autotune"]:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % config["target_network_frequency"] == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(config["tau"] * param.data + (1 - config["tau"]) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(config["tau"] * param.data + (1 - config["tau"]) * target_param.data)

    envs.close()
    return returns