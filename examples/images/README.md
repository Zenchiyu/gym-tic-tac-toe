# Examples
Figures correspond to the estimated prob winning from empty board while learning with eps-greedy behavior policy.
They show the current learned value function where learning updates occurred either after only greedy moves or all moves.

(Note: my explanations are not necessarily correct)
## TD agent against random uniform opponent (inspired by Exercise 1.4 Learning From Exploration from Sutton and Barto's book)

To learn the value function, the update rule had 0.1 as step-size parameter (but kept constant) and the undiscounted formulation of the return (discount factor/rate `gamma=1`).
In all cases, the behavior policy has a tendency to explore and is an epsilon-greedy strategy with `eps=0.2` (decaying over time)

After learning the value function by either
1. Excluding exploratory moves
2. Including exploratory moves

We can observe through numerical experiments that, acting greedily gives the most rewards in the first case while acting eps-greedily gives the most rewards in the second case.

To summarize, with an epsilon-greedy behavior policy, we get two cases depending on when learning updates occured:
1. Excluding exploratory moves: acting greedily gives the most rewards
2. Including exploratory moves: acting eps-greedily gives the most rewards

Each cases have their own purposes.

### Numerical experiments:
1. V[tuple([0]*9)] (over 10 000 episodes): 0.7347304824736642
Average number of times (over 100 000 episodes) the player X won (same learned value function):
    - greedy: 0.7156899999999914
    - eps-greedy: 0.715540000000005
                
2. V[tuple([0]*9)] (over 10 000 episodes): 0.7942112886509693
Average number of times (over 100 000 episodes) the player X won (same learned value function):
    - greedy: 0.7166800000000111
    - eps-greedy: 0.7180299999999962

We can also observe that we seem to get more reward during learning when learning from exploration. Doing so takes into account the tendency to explore of the behavior policy and consequently, the agent will try to get more rewards despite the random nature of exploration. On the other hand, if trying to maximize the amount of reward we can get while interacting with the environment is not that important, we can just learn from greedy moves and then obtain the optimal greedy policy at the end (1. acting greedily gives the most rewards).

