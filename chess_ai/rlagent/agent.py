from copy import deepcopy
import random
import pandas as pd

from chess_ai.rlagent.training.trainer import encode_state


class Agent:
    def __init__(self, color, game, model):
        self.color = color
        self.game = game
        self.model = model

    def evaluate_position(self, state):

        return self.model.predict(pd.DataFrame([encode_state(state)], columns=self.model.feature_names_in_))
        #else:

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
        return list_moves