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
        '0' represent empty cell
        'x' represent snake body part

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
            int: -1 if game ends, 0 otherwise

        """

        board = [[0 for j in range(self.size_y)] for i in range(self.size_x)]
        if len(self.food) == 2:
            board[self.food[0]][self.food[1]] = 'f'
        reward = 0
        for body_part in snake.get_body():
            x,y = body_part[0],body_part[1]
            if x > self.size_x - 1 or x < 0:
                return -1
            if y > self.size_y -1 or y<0:
                return -1
            if board[x][y] == 'x':
                return -1
            if board[x][y] == 'f':
                reward = 1
            board[x][y] = 'x'
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
        self.food = food