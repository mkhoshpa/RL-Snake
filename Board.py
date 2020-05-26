from Snake import Snake


class Board:
    """
    A class used to represent a board

    ...

    Attributes
    ----------
    size_x : int
        size of the board in x axis
    size_y : int
        size of the board in y axis
    board : list
        a list containing the each cell of the board
        0 represents empty cell
        1 represents snake body part
        20 represents food
    Methods
    -------
    put_snake_on_board(snake: Snake)
        puts the body parts of the snake on an empty board

    """

    def __init__(self,x,y):
        """

        :param x: int
            size of board in x axis
        :param y:int
            size of board in y axis
        """
        self.size_x = x
        self.size_y = y
        self.board = []
        self.food = tuple()

    def put_snake_on_board(self,snake):
        """

        Args:
            snake: Snake object to put in the board

        Returns:
            int: -10 if game ends, 10 if snake eats the apple, and 0 otherwise

        """

        board = [ [1 for j in range(self.size_y) ] if i == 0 or i == self.size_x-1 else [1 if j == 0 or j == self.size_y-1 else 0 for j in range(self.size_y)] for i in range(self.size_x)]
        if len(self.food) == 2:
            board[self.food[0]][self.food[1]] = -20
        reward = 0
        for body_part in snake.get_body():
            x,y = body_part[0],body_part[1]
            if x > self.size_x - 1 or x < 0:
                return -10
            if y > self.size_y -1 or y<0:
                return -10
            if board[x][y] == 1 or board[x][y] == 2:
                reward =  -10
            elif board[x][y] == -20:
                reward = 10
            board[x][y] = 1
        head_x,head_y = snake.get_head()
        board[head_x][head_y] = 2
        self.board = board
        return reward

    def __str__(self):
        """

        Returns:
            A string representation of the board
        """
        out = ""
        for j in range(self.size_y):
            for i in range(self.size_x):
                out += str(self.board[i][self.size_y - j - 1])
                out += ' '
            out += '\n'
        return out

    def get_board(self):
        """
        getter method for the game board
        Returns:
            list: representing the board list[i][j] is ith col and jth row
        """
        return self.board

    def set_cell(self,i,j,cell):
        """
        setter method for one cell of the board
        Args:
            i: col number
            j: row number
            cell: cells content

        Returns:
            None

        """
        self.board[i][j] = cell

    def set_food(self,food):
        """
        sets the location of the food in the food attribute
        Args:
            food: tuple of two ints
                column, row if the food.
        Returns:

        """
        self.food = food