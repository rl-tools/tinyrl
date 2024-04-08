# Custom Environment

To use custom C++ environments you should provide an [environment.h](./environment.h) which defines a `ENVIRONMENT` typedef that defines the data structures (based on [my_pendulum.h](./my_pendulum/my_pendulum.h)). Furthermore you should provide implementations for the basic operations (example: [operations_generic.h](./my_pendulum/operations_generic.h)):
- `sample_initial_state`: Sample from the initial state distribution
- `initial_state`: Initial state for deterministic evals
- `step`: Forward simulation of the environment dynamics given a state and action
- `reward`: Reward function
- `terminated`: Relation that returns if a state is terminal

Then you can use TinyRL to train an agent for this environment by creating a specification for this environment:
```
custom_environment = {
    "path": os.path.dirname(os.path.abspath(__file__)),
    "action_dim": 1,
    "observation_dim": 3,
}
```
And training it in the usual way ([train.py](./train.py)).