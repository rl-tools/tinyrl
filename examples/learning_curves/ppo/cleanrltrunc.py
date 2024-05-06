# from https://github.com/vwxyzjn/cleanrl/blob/8cbca61360ef98660f149e3d76762350ce613323/cleanrl/ppo_continuous_action.py

import os
import random
import time

import gymnasium as gym
from gymnasium.experimental.wrappers import RescaleActionV0
import numpy as np
from evaluate_policy import evaluate_policy

def make_env(config):
    def thunk():
        env = gym.make(config["environment_name"])
        # env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        # env = gym.wrappers.ClipAction(env)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env = RescaleActionV0(env, -1, 1)
        env = gym.wrappers.ClipAction(env)
        env.reset(seed=config["seed"])
        return env
    return thunk


default_config = {
    "anneal_lr": False,
    "clip_vloss": False,
    "vf_coef": 1.0,
}

def train_cleanrltrunc(config):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions.normal import Normal
    import numpy as np
    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        # torch.nn.init.orthogonal_(layer.weight, std)
        # torch.nn.init.constant_(layer.bias, bias_const)
        return layer


    class Agent(nn.Module):
        def __init__(self, envs, hidden_dim):
            super().__init__()
            self.critic = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hidden_dim, hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hidden_dim, 1), std=1.0),
            )
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hidden_dim, hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hidden_dim, np.prod(envs.single_action_space.shape)), std=0.01),
            )
            self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(envs.single_action_space.shape))*np.log(config["initial_action_std"]))

        def get_value(self, x):
            return self.critic(x)

        def get_action_and_value(self, x, action=None):
            action_mean = self.actor_mean(x)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    # args = tyro.cli(Args)
    batch_size = int(config["n_environments"] * config["on_policy_runner_steps_per_env"])
    minibatch_size = config["batch_size"]
    num_iterations = config["n_steps"]
    # TRY NOT TO MODIFY: seeding
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(config) for i in range(config["n_environments"])]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, config["hidden_dim"]).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config["learning_rate"], eps=1e-8)

    # ALGO Logic: Storage setup
    obs = torch.zeros((config["on_policy_runner_steps_per_env"], config["n_environments"]) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((config["on_policy_runner_steps_per_env"], config["n_environments"]) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((config["on_policy_runner_steps_per_env"], config["n_environments"])).to(device)
    rewards = torch.zeros((config["on_policy_runner_steps_per_env"], config["n_environments"])).to(device)
    truncated = torch.zeros((config["on_policy_runner_steps_per_env"], config["n_environments"])).to(device)
    terminated = torch.zeros((config["on_policy_runner_steps_per_env"], config["n_environments"])).to(device)
    values = torch.zeros((config["on_policy_runner_steps_per_env"], config["n_environments"])).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(config["n_environments"]).to(device)

    evaluation_returns = []

    for iteration in range(1, config["n_steps"] + 1):
        # Annealing the rate if instructed to do so.
        if config["anneal_lr"]:
            frac = 1.0 - (iteration - 1.0) / config["n_steps"]
            lrnow = frac * config["learning_rate"]
            optimizer.param_groups[0]["lr"] = lrnow

        if (iteration % config["evaluation_interval"]) == 1 or (config["evaluation_interval"] == 1):
            def policy(observation):
                return agent.actor_mean(torch.Tensor(observation).to(device).unsqueeze(0))[0].detach().cpu().numpy()
            current_returns = evaluate_policy(policy, config, make_env(config))
            print(f"Step: {iteration}, Returns: {np.array(current_returns).mean()}", flush=True)
            evaluation_returns.append(current_returns)


        for step in range(0, config["on_policy_runner_steps_per_env"]):
            global_step += config["n_environments"]
            obs[step] = next_obs

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            # next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs= torch.Tensor(next_obs).to(device)
            terminated[step] = torch.Tensor(terminations)
            truncated[step] = torch.Tensor(np.logical_or(terminations, truncations))

        # bootstrap value if not done
        with torch.no_grad():
            # next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(config["on_policy_runner_steps_per_env"])):
                if t == config["on_policy_runner_steps_per_env"] - 1:
                    current_truncated = 1
                    next_value = 0
                else:
                    current_truncated = truncated[t]
                    next_value = values[t + 1] * (1-terminated[t])
                delta = (rewards[t] + config["gamma"] * next_value - values[t]) * (1-current_truncated*(1-terminated[t])) # delta should be zero if truncated (but not due to termination) if terminated the delta should be based on a zero next_value
                advantages[t] = lastgaelam = delta + config["gamma"] * config["gae_lambda"] * (1-current_truncated) * lastgaelam # in any case of truncation (truncation or termination) the advantage should be truncated
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(config["n_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > config["clip_coef"]).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if config["norm_advantage"]:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config["clip_coef"], 1 + config["clip_coef"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if config["clip_vloss"]:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -config["clip_coef"],
                        config["clip_coef"],
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config["entropy_coefficient"] * entropy_loss + v_loss * config["vf_coef"]

                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            # if args.target_kl is not None and approx_kl > args.target_kl:
            #     break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
    envs.close()
    return evaluation_returns