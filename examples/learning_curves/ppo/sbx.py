from evaluate_policy import evaluate_policy
import gymnasium as gym
from gymnasium.experimental.wrappers import RescaleActionV0
import numpy as np

default_config = {
    "max_grad_norm": np.inf
}

def train_sbx(config):
    import os, random
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from sbx import PPO as SBX_PPO
    from sbx.ppo.policies import PPOPolicy
    from stable_baselines3.common.callbacks import BaseCallback

    import jax
    import flax.linen as nn
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    def env_factory(seed=0):
        env = gym.make(config["environment_name"])
        env = RescaleActionV0(env, -1, 1)
        env = gym.wrappers.ClipAction(env)
        env.reset(seed=seed)
        return env
    # envs = DummyVecEnv([lambda: env_factory(seed=(config["seed"]*config["n_environments"] + i)) for i in range(config["n_environments"])])
    envs = env_factory()
    def policy_factory(obs_dim, action_dim, lr_schedule, **kwargs):
        return PPOPolicy(obs_dim, action_dim, lr_schedule, net_arch=[config["hidden_dim"], config["hidden_dim"]], optimizer_kwargs={"eps": 1e-8}, activation_fn=nn.relu, share_features_extractor=False, ortho_init=False, log_std_init=np.log(config["initial_action_std"]))
    model = SBX_PPO(policy_factory, envs, 
        learning_rate=config["learning_rate"],
        ent_coef=config["entropy_coefficient"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        batch_size=config["batch_size"],
        n_steps=config["on_policy_runner_steps_per_env"]*config["n_environments"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_coef"],
        normalize_advantage=config["norm_advantage"],
        seed=config["seed"]
    )
    returns = []
    class CustomCallback(BaseCallback):
        def __init__(self, verbose=0):
            super(CustomCallback, self).__init__(verbose)
            self.evaluation_step_i = 0
        def _on_training_start(self) -> None:
            pass
        def _on_step(self) -> bool:
            return True
        def _on_rollout_end(self) -> None:
            pass
        def _on_training_end(self) -> None:
            pass
        def _on_rollout_start(self) -> None:
            if self.evaluation_step_i % config["evaluation_interval"] == 0:
                def policy(observation):
                    return model.predict(observation, deterministic=True)[0]
                current_returns = evaluate_policy(policy, config, env_factory)
                print(f"Step {self.evaluation_step_i}/{config['n_steps']}: {np.mean(current_returns)}", flush=True)
                returns.append(current_returns)
            self.evaluation_step_i += 1
    model.learn(total_timesteps=config["n_environments"]*config["on_policy_runner_steps_per_env"]*config["n_steps"], callback=CustomCallback())
    return returns