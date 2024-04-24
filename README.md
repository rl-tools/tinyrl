# TinyRL
A Python wrapper for RLtools ([https://rl.tools](https://rl.tools)). PyTorch is only used for its [utils that allow convenient wrapping of C++ code](https://pytorch.org/docs/stable/cpp_extension.html) to compile RLtools. The RLtools training code needs to be compiled at runtime because properties like the observation and action dimensions are not known at compile time. One of the fundamental principles of RLtools is that the sizes of all data structures and loops are known at compile-time so that the compiler can maximally reason about the code and heavily optimize it. Hence this wrapper takes an environment ([Gymnasium](https://github.com/Farama-Foundation/Gymnasium) interface) factory function as an input to infer the observation and action shapes and compile a bridge environment that is compatible with RLtools. 

This wrapper is work in progress and for now just exposes the SAC training loop and does not allow much modification of hyperparameters etc. yet. Stay tuned.

### Installation:
```
pip install tinyrl gymnasium
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
pip install gymnasium[classic-control]
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


### Saving and Loading Checkpoints

```
# Save
with open("pendulum_sac_checkpoint.h", "w") as f:
    f.write(state.export_policy())
# Load
from tinyrl import load_checkpoint_from_path
policy = load_checkpoint_from_path("pendulum_sac_checkpoint.h")
action = policy.evaluate(observation) # Note that e.g. SAC's policies output mean and std (concatenated)
```
# Custom C++ Enviroments

To get the maximum performance you should rewrite your environment in C++. Don't be scared it is actually quite straightforward and similar to creating a Gym environment. For an example of a custom pendulum environment see [examples/custom_environment](./examples/custom_environment/README.MD) (just 105 lines of code).

# Acceleration

On macOS TinyRL automatically uses Accelerate. To use MKL on linux you can install TinyRL with the `mkl` option:
```
pip install tinyrl[mkl]
```

# Windows

TinyRL also works on Windows but MKL is not integrated, yet. Please make sure to install Python from the installer from the Python website and not using the Windows Store Python version. The latter resides in a directory that requires admin privileges even for read access. Due to the just-in-time compilation of RLtools we need to be able to read the Python header and library files. After installing the right Python version the easies way to run TinyRL is by opening the cloned folder in Visual Studio Code and launching the preconfigured targets. Make sure to start Visual Studio Code from the Visual Studio Prompt (e.g. `Developer Command Prompt for VS 2022`) by running `code` so that `cl.exe` (MSVC) is available in the environment. 
