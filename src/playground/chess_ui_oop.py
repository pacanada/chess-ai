from typing import Tuple
import pyglet
from chess_python.chess import Chess

W = H = 480
PIECE_IMAGE_DICT = {
    1: pyglet.resource.image("resources/white_pawn.png"),
    2: pyglet.resource.image("resources/white_bishop.png"),
    3: pyglet.resource.image("resources/white_knight.png"),
    4: pyglet.resource.image("resources/white_rook.png"),
    5: pyglet.resource.image("resources/white_queen.png"),
    6: pyglet.resource.image("resources/white_king.png"),
    -1: pyglet.resource.image("resources/black_pawn.png"),
    -2: pyglet.resource.image("resources/black_bishop.png"),
    -3: pyglet.resource.image("resources/black_knight.png"),
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




#pieces = generate_sprites_from_board(board=game.state.board)

class ChessBoard(pyglet.window.Window):
    def __init__(
        self,
        width: int,
        height: int,
        name: str = "Window",
        dt: float = 2,

    ):
        super().__init__(width, height, name, resizable=False)
        self.set_vsync(False)
        pyglet.clock.schedule_interval(self.update, dt)
        self.fps_display = pyglet.window.FPSDisplay(window=self)
        self.game = Chess()
        self.background = pyglet.resource.image("resources/board.svg.png")

    def draw_sprites_from_board(self):

        batch = pyglet.graphics.Batch()
        pieces = []
        for pos, piece in enumerate(self.game.state.board):
            if piece!=0:
                piece_sprite = pyglet.sprite.Sprite(PIECE_IMAGE_DICT[piece], (pos%8)*H/8, (pos//8)*W/8, batch=batch)
                piece_sprite.scale = 0.35

                pieces.append(piece_sprite)
        batch.draw()


    def on_draw(self):
        self.clear()
        self.background.blit(0,0)
        self.draw_sprites_from_board()


    def update(self, dt):
        self.clear()
        print("updating")

    def on_mouse_press(self, x, y, button, modifiers):

        pos_i = from_coord_to_index(x,y)
        print(f"Moused pressed {x=}, {y=}, {button=}, {modifiers=}", )
        print(f"Corresponding to index: {pos_i}")

    def on_mouse_release(x, y, button, modifiers):
        global pos_f
        pos_f = from_coord_to_index(x,y)
        print(pos_i, pos_f)
        game.move(move=[pos_i, pos_f, None], check_allowed_moves=True)
        print(game)
        pieces = generate_sprites_from_board(board=game.state.board)
        # background.blit(0,0)
        # for piece in pieces:
        #     piece.draw()
        print(f"Moused realeased {x=}, {y=}, {button=}, {modifiers=}", )

    def on_mouse_drag(self,x, y, dx, dy, buttons, modifiers):
        print(f"Moused dragging {x=}, {y=},{dx=}, {dy=}, {buttons=}, {modifiers=}", )



app = ChessBoard(W,H)
pyglet.app.run()
        
        