# Adapted from https://github.com/vwxyzjn/cleanrl/blob/8cbca61360ef98660f149e3d76762350ce613323/cleanrl/td3_continuous_action.py

import gymnasium as gym
from gymnasium.experimental.wrappers import RescaleActionV0
import numpy as np

from evaluate_policy import evaluate_policy

def make_env(config):
    def thunk(**kwargs):
        env = gym.make(config["environment_name"], **kwargs)
        env = RescaleActionV0(env, -1, 1)
        env = gym.wrappers.ClipAction(env)
        env.reset(seed=config["seed"])
        return env
    return thunk

import gymnasium as gym
import numpy as np

default_config = {
    "policy_frequency": 2
}

def train_cleanrl(config):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from stable_baselines3.common.buffers import ReplayBuffer

    class QNetwork(nn.Module):
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


    class Actor(nn.Module):
        def __init__(self, env, hidden_dim):
            super().__init__()
            self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc_mu = nn.Linear(hidden_dim, np.prod(env.single_action_space.shape))
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
            x = torch.tanh(self.fc_mu(x))
            return x * self.action_scale + self.action_bias


    import stable_baselines3 as sb3
    import random
    import time

    # TRY NOT TO MODIFY: seeding
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(config)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs, config["hidden_dim"]).to(device)
    qf1 = QNetwork(envs, config["hidden_dim"]).to(device)
    qf2 = QNetwork(envs, config["hidden_dim"]).to(device)
    qf1_target = QNetwork(envs, config["hidden_dim"]).to(device)
    qf2_target = QNetwork(envs, config["hidden_dim"]).to(device)
    target_actor = Actor(envs, config["hidden_dim"]).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=config["learning_rate"])
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=config["learning_rate"])

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
    obs, _ = envs.reset(seed=config["seed"])
    returns = []
    for global_step in range(config["n_steps"]):
        if global_step % config["evaluation_interval"] == 0:
            def policy(observation):
                return actor.forward(torch.Tensor(observation).to(device).unsqueeze(0))[0][0].detach().cpu().numpy()
            current_returns = evaluate_policy(policy, config, make_env(config), render=config["render"] and global_step >= 0)
            print(f"Step: {global_step}, Returns: {np.array(current_returns).mean()}", flush=True)
            returns.append(current_returns)
        # ALGO LOGIC: put action logic here
        if global_step < config["learning_starts"]:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * config["exploration_noise"])
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > config["learning_starts"]:
            data = rb.sample(config["batch_size"])
            with torch.no_grad():
                noise_clip = config["target_next_action_noise_clip"]
                clipped_noise = (torch.randn_like(data.actions, device=device) * config["target_next_action_noise_std"]).clamp(
                    -noise_clip, noise_clip
                ) * target_actor.action_scale

                next_state_actions = (target_actor(data.next_observations) + clipped_noise).clamp(
                    envs.single_action_space.low[0], envs.single_action_space.high[0]
                )
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
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

            if global_step % config["policy_frequency"] == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(config["tau"] * param.data + (1 - config["tau"]) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(config["tau"] * param.data + (1 - config["tau"]) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(config["tau"] * param.data + (1 - config["tau"]) * target_param.data)
    
    envs.close()
    return returns
