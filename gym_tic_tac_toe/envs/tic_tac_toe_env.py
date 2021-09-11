import gym
import numpy as np
from gym import spaces


class TicTacToeEnv(gym.Env):
    """
    Classic Tic Tac Toe Environment 3x3 board against customizable opponent.

    Two players play at each turn.

    Observations have 9 dimensions, one for each position on the board and
    each position is either 0, 1 or 2 corresponding to no X and no O, X or O respectively.

    Actions and their corresponding positions on the board:
        0 1 2
        3 4 5
        6 7 8
    Each action corresponds to placing an X or an O (depending on who is playing)
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
        self.observation_space = spaces.Box(np.zeros(9), 2*np.ones(9), dtype=np.int)
        
        self.current_player_dict = {1: "X", 2: "O"}
        
        
        
    def step(self, action):
        """
        Performing an action either places a X or O at the corresponding location
        depending on who is currently playing.
        """
        info = {}

        # Check if action possible from the environment state
        if not(self.check_action_possible(action)):
            raise Exception("Action not permitted")

        # place X or O respectively
        if self.current_player == 1:
            print("place X")
            obs = self.Box(np.zeros(9)) # TODO: Change this
        elif self.current_player == 2:
            print("place O")
            obs = self.Box(np.zeros(9)) # TODO: Change this
        else:
            raise Exception("Current player should be X or O (1 or 2 respectively)")

        # Swap current player
        self._current_player = 3 - self._current_player

        # Check if win, draw or lost based on board configuration
        reward, done = self.check_done()
        
        return obs, reward, done, info

    def reset(self):
        # Current player can be either 1 (X) or 2 (O).
        self._current_player = np.random.randint(1, 3)  # [1, 3)
        
        pass

    def print_current_player():
        print(f"Current player is: {self.current_player_dict[self._current_player]}")
        

    def check_action_possible(self, action):
        pass

    def check_done(self):
        # Check board configuration
        
        # Return dummy stuffs
        return 0, False
