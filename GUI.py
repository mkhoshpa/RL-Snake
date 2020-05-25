import tkinter as tk
UPDATE_RATE = 500

class App(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self)
        self.canvas = tk.Canvas(self, width=500, height=500, borderwidth=0, highlightthickness=0)
        self.canvas.pack(side="top", fill="both", expand="true")
        self.game = kwargs['game']
        self.board = self.game.board.get_board()
        self.rows = kwargs['rows']
        self.columns = kwargs['cols']
        self.tiles = {}
        self.canvas.bind("<Configure>", self.redraw)
        self.status = tk.Label(self, anchor="w")
        self.status.pack(side="bottom", fill="x")
        self.updater()

    def redraw(self, event=None):
        self.canvas.delete("rect")
        cellwidth = int(self.canvas.winfo_width()/self.columns)
        cellheight = int(self.canvas.winfo_height()/self.columns)
        for column,col in enumerate(self.board):
            for row,cell in enumerate(col):
                x1 = column*cellwidth
                y1 = row * cellheight
                x2 = x1 + cellwidth
                y2 = y1 + cellheight
                if self.board[column][row] == 0:
                    tile = self.canvas.create_rectangle(x1,y1,x2,y2, fill="grey", tags="rect")
                elif self.board[column][row] == 'x':
                    tile = self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", tags="rect")
                else:
                    tile = self.canvas.create_rectangle(x1, y1, x2, y2, fill="red", tags="rect")
                self.tiles[row,column] = tile

    def update_board(self):
        self.game.receive_action()
        reward = self.game.one_time_step()
        self.board = self.game.board.get_board()
        if reward == -1:
            print('total reward was: ',self.game.cur_reward)
            self.quit()

    def updater(self):
        self.update_board()
        self.redraw()
        self.after(UPDATE_RATE, self.updater)
