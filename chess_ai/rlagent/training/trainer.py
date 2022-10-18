import pickle
import random
from copy import deepcopy
from typing import List

import pandas as pd
from chess_python.chess import Chess, State
from sklearn.neural_network import MLPRegressor


def encode_state_to_bits(state: State):
    # board: 64*7figures*color ie: white pawn 0001 black pawn 1001

    # en passant: only 16 different possibilities at the same time 4 bits

    # castling 4 bits

    # turn 1 bit (0 white, 1 black)
    return state


def get_en_passant_index_offset(en_passant_allowed: List[int]) -> int:
    """Transform from en_passant encoding used in chess.state to simple 0 to 16 encoding. 0 nothing,
    1-8 white rank 3 and 9-16 black rank 5"""
    if len(en_passant_allowed) == 0:
        return 0
    index = en_passant_allowed[0]
    if index >= 16 and index <= 23:
        # a3 would be 1
        return index - 15
    if index >= 40 and index <= 47:
        return index - 31
    raise ValueError("Something went wrong")


def get_castling_encoding(c_e: List[int]):
    # {"Q": 0, "K": 1, "q": 2, "k": 3}
    bit_string = "".join(
        [
            "1" if 0 in c_e else "0",
            "1" if 1 in c_e else "0",
            "1" if 2 in c_e else "0",
            "1" if 3 in c_e else "0",
        ]
    )
    # 16 for [0,1,2,3] (1111)
    return int(bit_string, 2)


def encode_state(state: State):
    # TODO: this is far from efficient, there must be a clever way to encode a chess position for nn
    en_passant_encoding: int = get_en_passant_index_offset(state.en_passant_allowed)
    encoded_state = (
        state.board
        + [state.turn]
        + [en_passant_encoding]
        + [get_castling_encoding(state.castling_rights)]
    )
    return encoded_state


class Agent:
    def __init__(self, color, game, model):
        self.color = color
        self.game = game
        self.model = model

    def evaluate_position(self, state):

        self.model.predict(encode_state(state))
        # else:

        # # not implemented
        #    return random.random()*100-50

    def recommend(self, random_move=False):
        if random_move:
            return random.choice(self.game.legal_moves())
        list_moves = []
        for move in self.game.legal_moves():
            game_copied = deepcopy(self.game)
            value = self.evaluate_position(game_copied.move(move).state)
            list_moves.append((move, value))

        list_moves = sorted(list_moves, key=lambda item: item[1], reverse=self.color == 1)
        return list_moves[0][0]


class Simulation:
    """Take two agents, and run a game, storing the buffer of gameplay"""

    def __init__(self, model):
        self.buffer = []
        self.outcome = None
        self.game = Chess()
        self.model = model

    def run(self):
        while self.game.result is None:
            recommended_move = Agent(
                color=self.game.state.turn, game=self.game, model=self.model
            ).recommend()
            self.game.move(recommended_move)
            self.game.update_outcome()
            self.buffer.append(
                (deepcopy(self.game.state), encode_state(self.game.state), recommended_move)
            )
        self.outcome = self.game.result


class Trainer:
    """"""

    def __init__(self, n_sim: int, model=None):
        self.feature_columns = [f"x_{i}" for i in range(67)]
        dummy_dataset = pd.DataFrame([[0] * 67], columns=self.feature_columns)
        dummy_dataset["y"] = 0
        self.model = (
            MLPRegressor(hidden_layer_sizes=(10, 10), warm_start=True).fit(
                dummy_dataset[self.feature_columns], dummy_dataset["y"]
            )
            if model is None
            else model
        )
        self.n_sim = n_sim
        self.buffer = pd.DataFrame()

    def process_buffer(self, buffer_raw: List[List], result):
        df_buffer_per_sim = pd.DataFrame(
            [row[1] for row in buffer_raw], columns=self.feature_columns
        )
        # not clever, just for debugging
        df_buffer_per_sim["y"] = result * 100
        return df_buffer_per_sim

    def run_simulations(self):
        for _ in range(self.n_sim):
            sim = Simulation(self.model)
            sim.run()

            # process buffer and append
            buffer_per_sim = self.process_buffer(buffer_raw=sim.buffer, result=sim.outcome)
            self.buffer = pd.concat([self.buffer, buffer_per_sim])

    def train(self):
        self.model.fit(X=self.buffer[self.feature_columns], y=self.buffer["y"])


if __name__ == "__main__":

    trainer = Trainer(3)
    trainer.run_simulations()
    trainer.train()
    with open("chess_trainer.pickle", "wb") as f:
        # avoid api key to be serialized
        pickle.dump(trainer, f)
