Figures correspond to the estimated prob winning from empty board while learning with eps-greedy behavior policy.
They show the current learned value function where learning updates occurred either after only greedy moves or all moves.

# TD agent against random uniform opponent
From the learned value function with behavior policy having exploratory tendency
1. If do not learn from exploration: acting greedily gives the most rewards
2. If do learn from exploration: acting eps-greedily gives the most rewards
            
Numerical experiments, averaged over 100 000 episodes:
1. V[tuple([0]*9)]: 0.7347304824736642
Average number of times the player X won (same learned value function):
	-greedy: 0.7156899999999914
        -eps-greedy: 0.715540000000005
                
2. V[tuple([0]*9)]: 0.7942112886509693
Average number of times the player X won (same learned value function):
	-greedy: 0.7166800000000111
        -eps-greedy: 0.7180299999999962

We can also observe that we seems to get more reward when learning from exploration because the behavior policy takes
into account the tendency to explore.