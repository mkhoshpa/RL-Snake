import random

UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'
ACTIONS = [None,UP,DOWN,LEFT,RIGHT]


class BaseAgent:
    """
    Abstract class for the snake player agent
    """
    def __init__(self, state=None):
        """
        constructor of the agent
        Args:
            state: optional: if is None means the start of the game, else shows the current state of the game
        """
        self.state = state if state is not None else (None,None)

    def update(self,board):
        """

        Args:
            board:

        Returns:

        """
        self.state = self.state[1], board

    def take_action(self):
        """
        Abstract method for taking the action by the agent according to it's state
        Returns:

        """
        pass


class RandomAgent(BaseAgent):
    """
    Concrete agent class that take actions randomly
    """
    def take_action(self):
        """
        take actions randomly
        Returns:
            str or None: represent the action
        """
        if self.state[0] is None or self.state[1] is None:
            return None
        return ACTIONS[random.randrange(len(ACTIONS))]