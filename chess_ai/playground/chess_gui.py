from copy import deepcopy
from typing import List, Tuple

import pyglet
from chess_python.chess import Chess

from chess_ai.classical_agent.agent import Agent

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
SQUARE_SIZE = H / 8


def from_coord_to_index(x, y) -> int:

    file_index = int(x / SQUARE_SIZE)
    rank_index = int(y / SQUARE_SIZE)
    index = file_index + rank_index * 8
    return index


def from_index_to_coord(index: int) -> Tuple[float, float]:
    rank_index = index // 8
    file_index = index % 8

    x = SQUARE_SIZE / 2 + file_index * SQUARE_SIZE
    y = SQUARE_SIZE / 2 + rank_index * SQUARE_SIZE
    return x, y


DEBUG = False


class ChessBoard(pyglet.window.Window):
    def __init__(
        self,
        width: int,
        height: int,
        name: str = "Chess",
    ):
        super().__init__(width, height, name, resizable=False)
        self.set_vsync(False)
        self.fps_display = pyglet.window.FPSDisplay(window=self)
        self.game = Chess()
        self.background = pyglet.resource.image("resources/board.svg.png")
        self.pos_i = None
        self.pos_f = None
        self.allowed_moves_in_pos: List[int] = []
        self.result_label = None

    def draw_sprites_from_board(self):

        batch = pyglet.graphics.Batch()
        pieces = []
        for pos, piece in enumerate(self.game.state.board):
            if piece != 0:
                piece_sprite = pyglet.sprite.Sprite(
                    PIECE_IMAGE_DICT[piece],
                    (pos % 8) * H / 8,
                    (pos // 8) * W / 8,
                    batch=batch,
                )
                piece_sprite.scale = 0.35

                pieces.append(piece_sprite)
        batch.draw()

    def draw_allowed_moves(self):
        batch_legal_moves = pyglet.graphics.Batch()
        circles = []
        for index in self.allowed_moves_in_pos:
            x, y = from_index_to_coord(index)
            circle = pyglet.shapes.Circle(x, y, 8, color=(33, 39, 33), batch=batch_legal_moves)

            circles.append(circle)
        batch_legal_moves.draw()

    def draw_result(self):
        if self.game.result:
            result_dict = {1: "White wins", -1: "Black wins", 0: "Draw"}
            self.result_label = pyglet.text.Label(
                result_dict[self.game.result] + "\n Press q to quit.",
                font_name="Times New Roman",
                font_size=24,
                color=(0, 0, 0, 255),
                x=W // 2,
                y=W // 2,
                anchor_x="center",
                anchor_y="center",
            )
            self.result_label.draw()

    def on_draw(self):
        self.clear()
        self.background.blit(0, 0)
        self.draw_sprites_from_board()
        self.draw_allowed_moves()
        self.draw_result()

    def on_key_press(self, symbol, modifier):  # noqa U100
        # key "Q" get press
        if symbol == pyglet.window.key.Q:
            # close the window
            self.close()

    def on_mouse_press(self, x, y, button, modifiers):
        """Implementing main UI logic"""

        if (
            self.pos_i is None
            and self.game.state.board[from_coord_to_index(x, y)] * self.game.state.turn > 0
        ):
            self.pos_i = from_coord_to_index(x, y)
            self.allowed_moves_in_pos = self.game.legal_moves_in_position(pos=self.pos_i)
            self.draw_allowed_moves()
        elif self.pos_i is not None:
            if self.game.state.board[from_coord_to_index(x, y)] * self.game.state.turn > 0:
                self.allowed_moves_in_pos = self.game.legal_moves_in_position(pos=self.pos_i)
                self.draw_allowed_moves()
            else:
                self.pos_f = from_coord_to_index(x, y)
                try:
                    self.game.move(move=[self.pos_i, self.pos_f, None], check_allowed_moves=True)
                    self.game.update_outcome()
                    self.draw_result()
                    self.draw_sprites_from_board()

                except ValueError as e:
                    print("Error: Invalid move", e)
                    self.allowed_moves_in_pos = []
                    self.pos_i = None
                    self.pos_f = None
                    return None
                self.allowed_moves_in_pos = []
                self.pos_i = None
                self.pos_f = None
                # opponent
                self.handle_opponent_turn()

        if DEBUG:
            print(
                f"Moused pressed {x=}, {y=}, {button=}, {modifiers=}",
            )
            print(f"Corresponding to index: {self.pos_i}")

    def handle_opponent_turn(self):
        if self.game.result is None:
            self.make_opponent_move()
            self.game.update_outcome()
            self.draw_result()

    def on_mouse_release(self, x, y, button, modifiers):
        if DEBUG:
            print(
                f"Moused realeased {x=}, {y=}, {button=}, {modifiers=}",
            )

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if DEBUG:
            print(
                f"Moused dragging {x=}, {y=},{dx=}, {dy=}, {buttons=}, {modifiers=}",
            )

    def make_opponent_move(self):

        game_copy = deepcopy(self.game)
        agent = Agent(
            depth=3,  # actually is 4
            color=game_copy.state.turn,
            alpha_beta=True,
            move_ordering=True,
            use_transpositions=True,
        )

        recommended_moves = agent.recommend(node=game_copy, order=True, random_flag=False)

        # in case there are several moves with same value
        best_move = recommended_moves[0][0]
        if DEBUG:
            print("Agent recomendation: ", best_move)
        self.game.move(best_move)


app = ChessBoard(W, H)
pyglet.app.run()
