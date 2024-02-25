# TinyRL
A Python wrapper for RLtools ([https://rl.tools](https://rl.tools)). PyTorch is only used for its [utils that allow convenient wrapping of C++ code](https://pytorch.org/docs/stable/cpp_extension.html) to compile RLtools. The RLtools training code needs to be compiled at runtime because properties like the observation and action dimensions are not known at compile time. One of the fundamental principles of RLtools is that the sizes of all data structures and loops are known at compile-time so that the compiler can maximally reason about the code and heavily optimize it. Hence this wrapper takes an environment ([Gymnasium](https://github.com/Farama-Foundation/Gymnasium) interface) factory function as an input to infer the observation and action shapes and compile a bridge environment that is compatible with RLtools. 

This wrapper is work in progress and for now just exposes the SAC training loop and does not allow much modification of hyperparameters etc. yet. Stay tuned.

### Installation:
```
pip install tinyrl
```
For the following example:
```
pip install gymnasium
```

### Example:
```
from tinyrl import SAC
import gymnasium as gym

seed = 0xf00d
def env_factory():
    env = gym.make("Pendulum-v1")
    env.reset(seed=seed)
    return env

sac = SAC(env_factory)
state = sac.State(seed)

finished = False
while not finished:
    finished = state.step()
```

### Evaluating the Trained Policy
```
pip install gymnasium\[classic-control\]
```

```
env_replay = gym.make("Pendulum-v1", render_mode="human")

while True:
    observation, _ = env_replay.reset(seed=seed)
    finished = False
    while not finished:
        env_replay.render()
        action = state.action(observation)
        observation, reward, terminated, truncated, _ = env_replay.step(action)
        finished = terminated or truncated
```



# Acceleration

On macOS TinyRL automatically uses Accelerate. To use MKL on linux you can install TinyRL with the `mkl` option:
```
pip install tinyrl[mkl]
```