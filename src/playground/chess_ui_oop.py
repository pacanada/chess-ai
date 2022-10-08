from typing import Tuple
import pyglet
from chess_python.chess import Chess

W = H = 480
PIECE_IMAGE_DICT = {
    1: pyglet.resource.image("resources/white_pawn.png"),
    2: pyglet.resource.image("resources/white_knight.png"),
    3: pyglet.resource.image("resources/white_bishop.png"),
    4: pyglet.resource.image("resources/white_rook.png"),
    5: pyglet.resource.image("resources/white_queen.png"),
    6: pyglet.resource.image("resources/white_king.png"),
    -1: pyglet.resource.image("resources/black_pawn.png"),
    -2: pyglet.resource.image("resources/black_knight.png"),
    -3: pyglet.resource.image("resources/black_bishop.png"),
    -4: pyglet.resource.image("resources/black_rook.png"),
    -5: pyglet.resource.image("resources/black_queen.png"),
    -6: pyglet.resource.image("resources/black_king.png"),
    }
SQUARE_SIZE = H/8



def from_coord_to_index(x,y)->int:

    file_index = int(x/SQUARE_SIZE)
    rank_index = int(y/SQUARE_SIZE)
    index = file_index+rank_index*8
    return index

def from_index_to_coord(index:int)->Tuple[int,int]:
    rank_index = index//8
    file_index = index%8

    x = SQUARE_SIZE/2 + file_index*SQUARE_SIZE
    y = SQUARE_SIZE/2 + rank_index*SQUARE_SIZE
    return x, y




DEBUG = True

class ChessBoard(pyglet.window.Window):
    def __init__(
        self,
        width: int,
        height: int,
        name: str = "Window",
        dt: float = 10,

    ):
        super().__init__(width, height, name, resizable=False)
        self.set_vsync(False)
        #pyglet.clock.schedule_interval(self.update, dt)
        self.fps_display = pyglet.window.FPSDisplay(window=self)
        self.game = Chess()
        self.background = pyglet.resource.image("resources/board.svg.png")
        self.pos_i = None
        self.pos_f = None
        self.allowed_moves_in_pos = []

    def draw_sprites_from_board(self):

        batch = pyglet.graphics.Batch()
        pieces = []
        for pos, piece in enumerate(self.game.state.board):
            if piece!=0:
                piece_sprite = pyglet.sprite.Sprite(PIECE_IMAGE_DICT[piece], (pos%8)*H/8, (pos//8)*W/8, batch=batch)
                piece_sprite.scale = 0.35

                pieces.append(piece_sprite)
        batch.draw()

    def draw_allowed_moves(self):
        batch_circle = pyglet.graphics.Batch()
        #allowed_moves = self.game.legal_moves_in_position(pos=pos)
        circles = []
        for index in self.allowed_moves_in_pos:
            x,y = from_index_to_coord(index)
            print(x,y)
            circle = pyglet.shapes.Circle(x, y, 8, color=(33,39,33), batch=batch_circle)

            circles.append(circle)
        batch_circle.draw()


    def on_draw(self):
        self.clear()
        self.background.blit(0,0)
        self.draw_sprites_from_board()
        self.draw_allowed_moves()


    def on_key_press(self, symbol, modifier): 
        # key "Q" get press
        if symbol == pyglet.window.key.Q:
            # close the window
            self.close()

    def on_mouse_press(self, x, y, button, modifiers):
        if self.pos_i is None:
            self.pos_i = from_coord_to_index(x,y)
            self.allowed_moves_in_pos = self.game.legal_moves_in_position(pos=self.pos_i)
            self.draw_allowed_moves(self.pos_i)
        else:
            self.pos_f = from_coord_to_index(x,y)
            self.game.move(move=[self.pos_i, self.pos_f, None], check_allowed_moves=True)
            print(self.game)
            self.draw_sprites_from_board()
            self.pos_i = None
            self.pos_f = None
        if DEBUG:
            print(f"Moused pressed {x=}, {y=}, {button=}, {modifiers=}", )
            print(f"Corresponding to index: {self.pos_i}")

    def on_mouse_release(self, x, y, button, modifiers):
        #pos_f = from_coord_to_index(x,y)
        #print(self.pos_i, pos_f)
        #self.game.move(move=[self.pos_i, pos_f, None], check_allowed_moves=True)
        #print(self.game)
        #self.draw_sprites_from_board()
        print(f"Moused realeased {x=}, {y=}, {button=}, {modifiers=}", )

    def on_mouse_drag(self,x, y, dx, dy, buttons, modifiers):
        print(f"Moused dragging {x=}, {y=},{dx=}, {dy=}, {buttons=}, {modifiers=}", )



app = ChessBoard(W,H)
pyglet.app.run()
        
        