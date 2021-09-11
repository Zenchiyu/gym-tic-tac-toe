import gym
import gym_tic_tac_toe
import numpy as np


env = gym.make("TicTacToe-v0")

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

# Fully observable environment
s = env.reset()

for _ in range(9):
    a = np.random.choice(env.get_possibe_actions())
    print(env.step(a))  # env.action_space.sample() not always permitted
    env.render()
    
env.close()
