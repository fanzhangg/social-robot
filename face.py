class Face:

    def __init__(self, canvas):
        self._init_face(canvas)

    def _init_face(self, canvas):
        self.canvas = canvas

        canvas.create_oval(70, 70, 350, 350, fill='#FEE230', width=5, outline="#ED950D")
        canvas.create_oval(110, 110, 200, 200, fill='white', tags='left', width=5, outline="#ED950D")
        canvas.create_oval(110 + 30, 110 + 30, 110 + 60, 110 + 60, fill="black", tag='leftBall')
        canvas.create_oval(310 - 90, 110, 310, 110 + 90, fill='white', tags='right', width=5, outline="#ED950D")
        canvas.create_oval(220 + 30, 110 + 30, 220 + 60, 110 + 60, fill="black", tag="rightBall")
        canvas.create_line(145, 250, 275, 250, width=5, tags='mouth')

    def smile(self):
        self.canvas.delete('mouth')
        self.canvas.create_arc(125, 225, 275, 275, extent=-180, width=5, fill='white', tags='mouth')

    def sad(self):
        self.canvas.delete('mouth')
        self.canvas.create_arc(125, 250, 275, 300, extent=180, width=5, fill='white', tags='mouth')

    def wink(self):
        self.canvas.delete('mouth')
        self.canvas.create_line(125, 250, 275, 250, width=5, tags='mouth')

    def grin(self):
        self.canvas.delete('mouth')
        self.canvas.create_line(125, 250, 200, 250, 275, 215, width=5, smooth='true', tags='mouth')


if __name__ == "__main__":
    pass