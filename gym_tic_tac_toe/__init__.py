from gym.envs.registration import register


# Draws and defeats are equally bad. No intermediate rewards
# ----------------------------------------
# r_draw = r_lost = 0
register(id='TicTacToe-v0',
    entry_point='gym_tic_tac_toe.envs:TicTacToeEnv',
    max_episode_steps=9,
    kwargs={"r_win": 1, "r_draw": 0, "r_lost": 0}
)

# r_draw = r_lost = -1
register(id='TicTacToe-v1',
    entry_point='gym_tic_tac_toe.envs:TicTacToeEnv',
    max_episode_steps=9,
    kwargs={"r_win": 1, "r_draw": -1, "r_lost": -1}
)

# Draws are better than defeats. No intermediate rewards
# ----------------------------------------
# r_draw = 0 > r_lost = -1
register(id='TicTacToe-v2',
    entry_point='gym_tic_tac_toe.envs:TicTacToeEnv',
    max_episode_steps=9,
    kwargs={"r_win": 1, "r_draw": 0, "r_lost": -1}
)

# r_draw = -1 > r_lost = -2
register(id='TicTacToe-v3',
    entry_point='gym_tic_tac_toe.envs:TicTacToeEnv',
    max_episode_steps=9,
    kwargs={"r_win": 1, "r_draw": -1, "r_lost": -2}
)
