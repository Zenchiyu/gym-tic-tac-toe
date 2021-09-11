import gym
# For "Cannot re-register id: TicTacToe" error, can comment it if
# no error
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
     if 'TicTacToe' in env:
          print('Remove {} from registry'.format(env))
          del gym.envs.registration.registry.env_specs[env]
          
import gym_tic_tac_toe
import numpy as np


env = gym.make("TicTacToe-v0")

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

# Fully observable environment
s = env.reset()
done = False

while not done:
    a = np.random.choice(env.get_possible_actions())
    s_new, r, done, info = env.step(a)  # env.action_space.sample() not always permitted
    env.render()

print(f"The reward is: {r}")
env.close()
