
UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'


class Snake:
    '''
    A class used to represent an Snake

    ...

    Attributes
    ----------
    head : tuple
        a tuple containing (x,y) of the head of the snake
    length : int
        length of the snake
    direction : str
        showing the direction that snake is moving from 'up', 'down', 'left','right'
    body : list
        a list containing all the pixels that build up the body of snake

    Methods
    -------
    get_body()
        Returns Body of snake
    '''

    def __init__(self,head_x,head_y,direction,length=3,board_size=(10,10)):
        """
        Parameters
        ----------
        head_x : int
            The name of the animal
        head_y : int
            The sound the animal makes
        direction : str, optional
            direction that the snake is moving
        length : int. optional
            length of the snake
        """
        self.head = head_x,head_y
        self.direction = direction
        self.length = length
        self.body=[self.head]
        self.board_size =  board_size
        for i in range(1, length):
            if self.direction == UP:
                self.body.append((self.head[0],self.head[1] - i))
            elif self.direction == DOWN:
                self.body.append((self.head[0], self.head[1] + i))
            elif self.direction == LEFT:
                self.body.append((self.head[0]+i, self.head[1]))
            elif self.direction == RIGHT:
                self.body.append((self.head[0] - i, self.head[1]))
            else:
                raise Exception('Direction is not valid')

    def get_body(self):
        """
        :return: list
            containing body of Snake
        """
        return self.body

    def _move_one_step(self):
        """
        :param func: a decorator function to work according to eating or not eating the food\
        :return: int
            0 if the snake does not hit the walls; -1 other wise
        """
        self.length += 1
        if self.direction == UP:
            self.head = (self.head[0], self.head[1] + 1)
            self.body.insert(0,self.head)
        elif self.direction == DOWN:
            self.head = (self.head[0], self.head[1] - 1)
            self.body.insert(0, self.head)
        elif self.direction == LEFT:
            self.head = (self.head[0] - 1, self.head[1])
            self.body.insert(0, self.head)
        elif self.direction == RIGHT:
            self.head = (self.head[0] + 1, self.head[1])
            self.body.insert(0, self.head)
        else:
            raise Exception('Direction is not valid')
        if self.head[0] < 0 or self.head[0] > self.board_size[0]\
                or self.head[1] < 0 or self.head[1] > self.board_size[1]:
            return -1
        return 0



    def move_one_step_without_eating(self):
        """
        removes the last part of the body, meaning moving one step
        :return: None
        """
        self._move_one_step()
        self.body.pop()

    def move_one_step_with_eating(self):
        """
        increasing the length of the snake, since it has just eaten a piece of food
        :return: None
        """
        self._move_one_step()
        self.length += 1

    def set_direction(self,direction):
        """

        :param direction: str:
            string representing the direction of the snake. From 'up', 'down', 'left','right'.
        :return: None
        """
        if direction in [LEFT,RIGHT,UP,DOWN]:
            if direction == LEFT and self.direction == RIGHT:
                return
            if direction == RIGHT and self.direction == LEFT:
                return
            if direction == UP and self.direction == DOWN:
                return
            if direction == DOWN and self.direction == UP:
                return
            self.direction = direction
        else:
            raise Exception('Direction is invalid')