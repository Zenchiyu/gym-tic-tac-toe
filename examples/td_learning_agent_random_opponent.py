# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 20:23:06 2021

@author: Stéphane Liem Nguyen

Trying to apply the same update rule as in Chap 1 introduction in Sutton
and Barto's book but without the same initial values.
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


def sample_action(current_player, V, s, gamma=1, eps=0.2):
    """
    Epsilon greedy strategy (where p(s', r | s, a) = 1 )
    This function is of limited use, cannot be applied in any problem
    """
    # TODO: Clean the code
    actions = np.arange(env.action_space.n)
    s_nexts, rewards, _, _ = zip(*[env.simulate_one_step(s, a, current_player) for a in actions])
    
    # have to convert the numpy array into tuple because of the dict
    if np.random.random() < eps:
        return np.random.choice(env.get_possible_actions())
    else:
        return np.argmax([reward + gamma*V[tuple(s_next)] if s[a] == 0 else -float("inf") for (s_next, reward, a) in zip(s_nexts, rewards, actions)])
    
def sample_greedily(current_player, V, s, gamma=1, eps=0.2):
    actions = np.arange(env.action_space.n)
    s_nexts, rewards, _, _ = zip(*[env.simulate_one_step(s, a, current_player) for a in actions])
    
    return np.argmax([reward + gamma*V[tuple(s_next)] if s[a] == 0 else -float("inf") for (s_next, reward, a) in zip(s_nexts, rewards, actions)])
    
def get_all_states(observation_space):
    """
    observation_space is a Box and we suppose the low values are all the same
    as well as the high values..
    
    This function is of limited use, cannot be applied in any problem
    """
    low = observation_space.low[0]
    high = observation_space.high[0]
    n = observation_space.low.shape[0]
    print(f"low: {low}, high: {high}, n: {n}")
    
    meshgrid_tuple = np.meshgrid(*([np.arange(low, high + 1)]*n))
    # Obtain flattened representation of all states.
    flattened_states = np.array(list(zip(*[a.flatten() for a in meshgrid_tuple])))
    # it's a array with one state per row
    return flattened_states

# ----------------------------------------

env = gym.make("TicTacToe-v0")
env.set_verbose(False)

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
# ----------------------------------------


# Init V, we'll use a dictionary but each state is a tuple instead of numpy array
V = {tuple(state.tolist()): 0  for state in get_all_states(env.observation_space)}

# V = {tuple(state.tolist()): 
#      0 if env.check_done_state(state)[-1]["outcome"] == "O" else
#      1 if env.check_done_state(state)[-1]["outcome"] == "X" else
#      0.5 for state in get_all_states(env.observation_space)}
    
# In general, probably won't do this, only init with all 0s ? and probably
# will prefer action values
    
ALPHA = 0.1
EPSILON = 0.2
MAX_EPISODES = 10000
GAMMA = 1  # because episodic tasks and the maximum number of steps
# of an episode is only 9

logs = []

for e in range(MAX_EPISODES):
    # Fully observable environment
    s = env.reset()
    s_old = copy.deepcopy(s)# for updates
    first_player = env.get_current_player()
    
    done = False
    while not done:
        current_player_before_move = env.get_current_player()
        if current_player_before_move == 1:
            # Behavior policy has the tendency to explore
            a = sample_action(env.get_current_player(), V, s, gamma = GAMMA, eps=EPSILON)
        else:
            a = np.random.choice(env.get_possible_actions())
        s_new, r, done, info = env.step(a)  # env.action_space.sample() not always permitted
        
        # update V using TD without exploratory moves
        if current_player_before_move == 1:
            # remove this condition and unindent the content if want to include exploratory moves
            if sample_greedily(env.get_current_player(), V, s, gamma = GAMMA, eps=EPSILON) == a:  # Note: it's a bit an approximate check for the greedy move..
                # Uncomment this if do not want to backup the if the first player
                # is X and previous state was directly the empty board.
                # if first_player == 1 and np.all(s_old == 0):
                #     continue
                V[(tuple(s_old))] = V[(tuple(s_old))] + ALPHA*(r + GAMMA*V[(tuple(s_new))]-V[(tuple(s_old))])
                EPSILON /= 2
                s_old = s_new
        
        s = s_new
        
        # env.render()
    logs.append(V[tuple([0]*9)])
    
# Plotting value of the empty board (without knowing what is the initial player)
plt.plot(logs)
plt.title("Estimated probability of winning for player X from the initial state for each episode")
# Do not necessarily give the exact values..
print(f"V[tuple([0]*9)]: {V[tuple([0]*9)]}\n\n")

# ----------------------------------------
# Now let's use the learned value function
avg = 0

for e in range(10*MAX_EPISODES):
    s = env.reset()
    done = False
    while not done:
        current_player_before_move = env.get_current_player()
        if current_player_before_move == 1:
            # From the learned value function with behavior policy having exploratory tendency
            # 1. If do not learn from exploration: acting greedily gives the most rewards
            # 2. If do learn from exploration: acting eps-greedily gives the most rewards
            
            # Example:
            # 1. V[tuple([0]*9)]: 0.7347304824736642
            # Average number of times the player X won (same learned value function):
            #  -greedy: 0.7156899999999914
            #  -eps-greedy: 0.715540000000005
                
            # 2. V[tuple([0]*9)]: 0.7942112886509693
            # Average number of times the player X won (same learned value function):
            #  -greedy: 0.7166800000000111
            #  -eps-greedy: 0.7180299999999962
            
            # a = sample_greedily(env.get_current_player(), V, s, gamma = GAMMA, eps=EPSILON)
            a = sample_action(env.get_current_player(), V, s, gamma = GAMMA, eps=EPSILON)
        else:
            a = np.random.choice(env.get_possible_actions())
        s_new, r, done, info = env.step(a)  # env.action_space.sample() not always permitted
        s = s_new
        # env.render()
    avg = avg + 1/(e+1)*( (info["outcome"] == "X") - avg)
    
# print(f"The reward is: {r} and the outcome is: {info['outcome']}")
# print(f"env.check_done_state(s_new) : {env.check_done_state(s_new)}")

print(f"Average number of times the player X won: {avg}")

env.close()
