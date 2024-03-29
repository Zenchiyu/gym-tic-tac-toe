import gym
import numpy as np
from gym import spaces
import copy


class TicTacToeEnv(gym.Env):
    """
    Classic Tic Tac Toe Environment 3x3 board against customizable opponent.

    Two players play at each turn.

    Observations are 9d numpy arrays, one for each position on the board and
    each position is either 0, 1 or 2 corresponding to no X and no O, X or O respectively.
    

    Actions and their corresponding positions on the board:
        0 1 2
        3 4 5
        6 7 8
    Each action is an integer and corresponds to placing an X or an O
    (depending on who is playing)

    Rewards are always from the point of view of player X (or 1)
    
    """
    def __init__(self, r_win=1, r_draw=0, r_lost=0):
        """
        Initialize the environment with v0 as default.
        No intermediate rewards (or reward of 0).
        
        Note: kwargs from gym register go in the constructor params.
        """
        
        self._r_win = r_win
        self._r_draw = r_draw
        self._r_lost = r_lost

        # TODO: remove print statement
        print(f"r_win={self._r_win}, r_draw={self._r_draw}, r_lost={self._r_lost}")

        # 9 possible actions
        self.action_space = spaces.Discrete(9)

        # 9 dimensions, 3 possible values for each
        # 0 for no X no O, 1 for X, 2 for O
        self.observation_space = spaces.Box(0, 2, shape=(9, ), dtype=int)
        
        self.current_player_dict = {1: "X", 2: "O"}
        
        self.verbose = True
        
    def step(self, action):
        """
        Performing an action either places a X or O at the corresponding location
        depending on who is currently playing.
        """
        info = {}
        
        # Check if action possible from the environment state
        if not(self.check_action_possible(action)):
            raise Exception("Action not permitted")

        # Place X or O, 1 or 2 respectively
        if self._current_player in [1, 2]:
            self._env_state[action] = self._current_player
            obs = copy.deepcopy(self._env_state)
        else:
            raise Exception("Current player should be X or O (1 or 2 respectively)")
        
        if self.verbose:
            print(f"Player {self.current_player_dict[self._current_player]} took action {action}")
        
        # Check if win, draw or lost based on board configuration
        reward, done, info = self.check_done()
        
        # Swap current player
        self._current_player = 3 - self._current_player
        
        return obs, reward, done, info

    def reset(self):
        # Current player can be either 1 (X) or 2 (O).
        self._current_player = np.random.randint(1, 3)  # [1, 3)
        self._env_state = np.zeros(9, dtype=int)
        return np.zeros(9, dtype=int)  # initial observation

    def render(self, mode="human", close=False):
        # parameters are there just to not have errors..
        # TODO: add prettier rendering !

        # env state = agent state
        print(self._env_state.reshape((3, 3)))
    
    def set_verbose(self, verbose):
        self.verbose = verbose
        
    def print_current_player(self):
        print(f"Current player is: {self.current_player_dict[self._current_player]}")
    
    def get_current_player(self):
        return self._current_player
    
    def check_action_possible(self, action):
        return self._env_state[action] == 0

    def get_possible_actions(self):
        return np.flatnonzero(self._env_state == 0)

    def check_done(self):
        return self.check_done_state(copy.deepcopy(self._env_state))
    
    def check_done_state(self, state):
        """
        Check outcome of state, can be different than our current game
        """
        # Check board configuration
        board = copy.deepcopy(state).reshape((3, 3))
        
        # For each player, only take a look at their pieces
        for player_num, b in [(player_num, (board == player_num)) for player_num in [1, 2]]:
            # Check horizontallym, vertically, diagonally
            three_in_horizontal = (3 in np.sum(b, axis=1))
            three_in_vertical = (3 in np.sum(b, axis=0))
            three_in_diagonal = (np.sum(np.diag(b)) == 3) or (np.sum(np.diag(np.fliplr(b))) == 3)
            
            if three_in_horizontal or three_in_vertical or three_in_diagonal:
                # r_win if player_num = 1, r_lose if player_num = 2
                return self._r_win*(2-player_num) + self._r_lost*(player_num-1), True, {"outcome": self.current_player_dict[player_num]}

        # check if board is filled
        if np.sum(state != 0) == 9:
            return self._r_draw, True, {"outcome": "draw"}
        
        # Still in an intermediate step !
        return 0, False, {"outcome": "none"}
    
    # These functions below are useful when you want to have the perfect
    # model of the environment : exact environment dynamics, rewards
    
    def simulate_one_step(self, state, action, current_player):
        """
        Get obs, reward, done, info
        after taking an action from a state with the current player.
        
        It does not affect the current state of the environment.
        """
        assert current_player in [1, 2], "Current player should be X or O (1 or 2 respectively)"
        if not(state[action] == 0):  # if empty space, action is possible
            return state, 0, False, {"outcome": "none"}  # TODO: maybe raise Exception
        
        s_new = copy.deepcopy(state)
        s_new[action] = current_player
        return s_new, *self.check_done_state(s_new)
    
    def p(self, s_new, reward, state, action):
        """
        Get the probability of moving into a new state, getting a reward
        after taking an action from a state.
        """
        # if empty space, action is possible and if reward is
        # the reward obtained when you check the new state.
        if (state[action] == 0) and (self.check_done_state(s_new)[0] == reward):
            return 1
        return 0
