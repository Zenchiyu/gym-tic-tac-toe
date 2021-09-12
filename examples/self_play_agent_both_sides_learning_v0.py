# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 15:42:39 2021

@author: Stéphane Liem Nguyen

See td_learning_agent_random_opponent.py for more comments.
Trying to apply self-play but two tables instead of one.
"""

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
import matplotlib.pyplot as plt
import copy

# This part is the same as in td_learning_agent_random_opponent.py (without comments)
def sample_action(current_player, V, s, gamma=1, eps=0.2):
    actions = np.arange(env.action_space.n)
    s_nexts, rewards, _, _ = zip(*[env.simulate_one_step(s, a, current_player) for a in actions])
    if np.random.random() < eps:
        return np.random.choice(env.get_possible_actions())
    else:
        return np.argmax([reward + gamma*V[tuple(s_next)] if s[a] == 0 else -float("inf") for (s_next, reward, a) in zip(s_nexts, rewards, actions)])
    
def sample_greedily(current_player, V, s, gamma=1):
    actions = np.arange(env.action_space.n)
    s_nexts, rewards, _, _ = zip(*[env.simulate_one_step(s, a, current_player) for a in actions])
    return np.argmax([reward + gamma*V[tuple(s_next)] if s[a] == 0 else -float("inf") for (s_next, reward, a) in zip(s_nexts, rewards, actions)])
    
def get_all_states(observation_space):
    low = observation_space.low[0]
    high = observation_space.high[0]
    n = observation_space.low.shape[0]
    print(f"low: {low}, high: {high}, n: {n}")
    
    meshgrid_tuple = np.meshgrid(*([np.arange(low, high + 1)]*n))
    flattened_states = np.array(list(zip(*[a.flatten() for a in meshgrid_tuple])))
    return flattened_states

# ----------------------------------------

env = gym.make("TicTacToe-v0")
env.set_verbose(False)

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
# ----------------------------------------


# Init V1 and V2, we'll use a dictionary but each state is a tuple instead of numpy array
V1 = {tuple(state.tolist()): 0 for state in get_all_states(env.observation_space)}
V2 = {tuple(state.tolist()): 0 for state in get_all_states(env.observation_space)}
    
ALPHA = 0.1
# One for each agent
EPSILON_1 = 0.2
EPSILON_2 = 0.2

MAX_EPISODES = 30000 # 100000
GAMMA = 1

logs1 = []
logs2 = []

avg = 0
for e in range(MAX_EPISODES):
    # Fully observable environment
    s = env.reset()
    # for updates
    s_old_1 = copy.deepcopy(s)
    s_old_2 = copy.deepcopy(s)
    
    done = False
    while not done:
        current_player_before_move = env.get_current_player()
        # Behavior policies have the tendency to explore
        if current_player_before_move == 1:
            a = sample_action(current_player_before_move, V1, s, gamma=GAMMA, eps=EPSILON_1)
        else:
            a = sample_action(current_player_before_move, V2, s, gamma=GAMMA, eps=EPSILON_2)
            
        s_new, r, done, info = env.step(a)
        
        r = (info["outcome"] == "O") + (info["outcome"] == "X")
        # TODO: change it (what I wrote above) to something better because here
        # it's based on the knowledge of the rewards (added this due to second agent learning)
        
        # update V using TD without exploratory moves
        if current_player_before_move == 1:
            # remove this condition and unindent the content if want to include exploratory moves
            if sample_greedily(current_player_before_move, V1, s, gamma=GAMMA) == a:  # Note: it's a bit an approximate check for the greedy move..
                # print(f"Before: {V1[(tuple(s_old_1))]}")    
                V1[(tuple(s_old_1))] = V1[(tuple(s_old_1))] + ALPHA*(r + GAMMA*V1[(tuple(s_new))]-V1[(tuple(s_old_1))])
                # print(f"After: {V1[(tuple(s_old_1))]}")    
                EPSILON_1 *= 0.999  # /= 2
                s_old_1 = s_new
        elif current_player_before_move == 2:  # Exactly the same as our agent but with his V2 and s_old_2
            # Need to minimize instead of maximize if we take r directly
            # TODO: modify something here because it "locally" does not make sense
            # remove this condition and unindent the content if want to include exploratory moves
            if sample_greedily(current_player_before_move, V2, s, gamma=GAMMA) == a:
                V2[(tuple(s_old_2))] = V2[(tuple(s_old_2))] + ALPHA*(r + GAMMA*V2[(tuple(s_new))]-V2[(tuple(s_old_2))])
                EPSILON_2 *= 0.999  # /= 2
                s_old_2 = s_new
        
        s = s_new
        
        # env.render()
    avg = avg + 1/(e+1)*( (info["outcome"] == "X") - avg)
    
    logs1.append(V1[tuple([0]*9)])
    logs2.append(V2[tuple([0]*9)])
    
print(f"% number of times the player X won (while learning): {avg}")

# Plotting value of the empty board (without knowing what is the initial player)
plt.figure()
plt.plot(logs1, label="X")
plt.plot(logs2, label="O")
plt.title("Estimated probability of winning for each agent from the initial state for each episode")
plt.legend()

# Do not necessarily give the exact values..
print(f"V1[tuple([0]*9)]: {V1[tuple([0]*9)]}\nV2[tuple([0]*9)]: {V2[tuple([0]*9)]}\n\n")

# ----------------------------------------
# Now let's use the learned value function against random policy instead !!
avg = 0

for e in range(MAX_EPISODES):
    s = env.reset()
    done = False
    while not done:
        current_player_before_move = env.get_current_player()
        if current_player_before_move == 1:
            a = sample_greedily(env.get_current_player(), V1, s, gamma = GAMMA)
        else:
            a = np.random.choice(env.get_possible_actions())
        s_new, r, done, info = env.step(a)  # env.action_space.sample() not always permitted
        s = s_new
        # env.render()
    avg = avg + 1/(e+1)*( (info["outcome"] == "X") - avg)
    
# print(f"The reward is: {r} and the outcome is: {info['outcome']}")
# print(f"env.check_done_state(s_new) : {env.check_done_state(s_new)}")

print(f"% number of times the player X won: {avg}")
# TODO: Change the values below because can be outdated
# We obtain : 0.7163600000000023 where
# V1[tuple([0]*9)]: 0.4352624605862745
# V2[tuple([0]*9)]: 0.5647375394137243

env.close()
