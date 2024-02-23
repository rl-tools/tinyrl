import os, sys
import gymnasium as gym
from torch.utils.cpp_extension import load

enable_optimization = True

absolute_path = os.path.dirname(os.path.abspath(__file__))

def env_factory():
    return gym.make("Pendulum-v1")

cpp_std_flag = '-std=c++17' if not sys.platform.startswith('win') else '/std:c++17'
optimization_flag = ('-O3' if not sys.platform.startswith('win') else '/O2') if enable_optimization else ''
arch_flags = '-march=native' if not sys.platform.startswith('win') else '/arch:AVX2'
fast_math_flag = '-ffast-math' if not sys.platform.startswith('win') else '/fp:fast'

observation_dim_flag = f'-DTINYRL_OBSERVATION_DIM={env_factory().observation_space.shape[0]}'
action_dim_flag = f'-DTINYRL_ACTION_DIM={env_factory().action_space.shape[0]}'
print(f"Compiling the TinyRL interface...")
ppo = load(
    'rl_tools',
    sources=['src/python_environment/python_environment.cpp'],
    extra_include_paths=[
        os.path.join(absolute_path, "..", "external", "rl_tools", "include"),
    ],
    extra_cflags=[cpp_std_flag, optimization_flag, arch_flags, fast_math_flag, observation_dim_flag, action_dim_flag],
)
print(f"Finished compiling the TinyRL interface.")


ppo.set_environment_factory(env_factory)

finished = False
step = 0

loop_state = ppo.LoopState()

ppo.init(loop_state, 0)
while not finished:
    if(step % 100 == 0):
        print(f"Step {step}")
    finished = ppo.step(loop_state)
    step += 1