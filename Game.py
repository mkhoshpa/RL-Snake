from Board import Board
from Snake import Snake
import Agent
from GUI import App
import random

UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'
ACTIONS = [UP,DOWN,LEFT,RIGHT]


class Game:
    """

    """
    def __init__(self,board_size,agent,start_board_snake=None,cur_reward=0):
        """

        Args:
            board_size:
            start_board_snake:
            agent:
        """
        self.cur_reward = cur_reward
        self.agent = agent
        if start_board_snake is None:
            x, y = board_size
            self.board_size = board_size
            self.board = Board(x,y)
            self.snake = Snake(5,4,UP,board_size=(x,y))
            self.board.put_snake_on_board(self.snake)
            i, j = self.new_food_location()
            self.board.set_food((i,j))
            self.board.set_cell(i, j, 2)
        else:
            self.board = start_board_snake[0]
            self.snake = start_board_snake[1]
        self.agent.update(self.board.get_board())

    def one_time_step(self):
        """
        updates the board for one time step
        Returns:
            int: -1 if game ends, 0 otherwise

        """
        self.snake.move_one_step_without_eating()
        out = self.board.put_snake_on_board(self.snake)
        if out == 1:
            self.cur_reward += 1
            i,j = self.new_food_location()
            self.board.set_cell(i,j,2)
            self.board.set_food((i,j))
        self.agent.update(self.board.get_board())
        return out

    def receive_action(self):
        """
        Asks the agent to take a action and update the snake accordingly
        Returns:
            int index of action in ACTIONS if is 4 means do nothing
        """
        action = self.agent.take_action().item()
        if action < len(ACTIONS):
            self.snake.set_direction(ACTIONS[action])
        return action

    def new_food_location(self):
        empty_cells = []
        for i,col in enumerate(self.board.get_board()):
            for j,cell in enumerate(col):
                if cell == 0:
                    empty_cells.append((i,j))
        return empty_cells[random.randrange(len(empty_cells))]




def run_gui_game(board_size,agent,start_board_snake=None):
    """

    Args:
        board_size:
        start_board_snake:
        agent:

    Returns:

    """
    if start_board_snake is None:
        game = Game(board_size=board_size,agent=agent)
    else:
        game = Game(board_size=board_size,start_board_snake=start_board_snake,agent=agent)
    app = App(game=game,rows=board_size[0],cols=board_size[1])
    app.mainloop()
    #while True:
    #    game.receive_action()
    #    reward = game.one_time_step()
    #    print(game.board)
    #    if reward == -1:
    #        print('total reward was: ',game.cur_reward)
    #        break


if __name__ == "__main__":
    #run_gui_game((10,10))
    game = Game(board_size=(10,10), agent=Agent.DQNAgent((10,10), trained = False))