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

def generate_sprites_from_board(board):

    #batch = pyglet.graphics.Batch()
    pieces = []
    for pos, piece in enumerate(board):
        if piece!=0:
            piece_sprite = pyglet.sprite.Sprite(PIECE_IMAGE_DICT[piece], (pos%8)*H/8, (pos//8)*W/8)
            piece_sprite.scale = 0.35

            pieces.append(piece_sprite)
    return pieces

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




game = Chess()
window = pyglet.window.Window(W,H)
background = pyglet.resource.image("resources/board.svg.png")

pieces = generate_sprites_from_board(board=game.state.board)
circles = []





@window.event
def on_draw():
    window.clear()
    background.blit(0,0)
    for piece in pieces:
        piece.draw()

def on_update():
    window.clear()
    print("updating")
    background.blit(0,0)
    for piece in pieces:
        piece.draw()


@window.event
def on_mouse_press(x, y, button, modifiers):

    global pos_i
    pos_i = from_coord_to_index(x,y)
    print(f"Moused pressed {x=}, {y=}, {button=}, {modifiers=}", )
    print(f"Corresponding to index: {pos_i}")
    # allowed_moves = game.legal_moves_in_position(pos=pos)
    # circles = []
    # for index in allowed_moves:
    #     x,y = from_index_to_coord(index)
    #     circle = pyglet.shapes.Circle(x, y, 10, color=(0, 0, 0))
    #     circle.draw()
    #     window.on
    #     circles.append(circle)
    #     print(x,y)




@window.event
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
@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    print(f"Moused dragging {x=}, {y=},{dx=}, {dy=}, {buttons=}, {modifiers=}", )

pyglet.app.run()