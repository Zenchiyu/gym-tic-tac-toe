# Gym Tic Tac Toe
Classic Tic Tac Toe game environment with 3x3 board for RL against customizable opponent.

# Installation

```
git clone https://github.com/Zenchiyu/gym_tic_tac_toe.git
cd gym_tic_tac_toe
pip install -e .
```

# How to use ?

```python
import gym
import gym_tic_tac_toe
import numpy as np


env = gym.make("TicTacToe-v0")

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

s = env.reset()
done = False

while not done:
    a = np.random.choice(env.get_possible_actions())
    s_new, r, done, info = env.step(a)  # env.action_space.sample() not always permitted
    env.render()

print(f"The reward is: {r}")
env.close()
```

# Tutorials and references for how to create custom environment:
- [Cheesy AI](https://youtu.be/ZxXKISVkH6Y)
- [DataHubbs](https://youtu.be/WNVbJNiiADA)
- [OpenAI Gym](https://github.com/openai/gym)
- [Gym MA Toy - Mehdi Zouitine](https://github.com/MehdiZouitine/gym_ma_toy)
