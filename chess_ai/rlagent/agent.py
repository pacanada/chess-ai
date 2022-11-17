from copy import deepcopy
import random
import numpy as np
import pandas as pd
from chess_ai.rlagent.muzero.all import MCTS
from chess_ai.rlagent.muzero.utils import MOVES

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

class AlphaZeroAgent:
    def __init__(self, game, model, n_sim):
        self.game = deepcopy(game)
        self.model = model
        self.n_sim = n_sim
    def recommend(self):
        mcts = MCTS(n_sim=self.n_sim, nn=self.model, game=self.game)
        mcts.run()
        s = hash(self.game.state)
        sum_N = sum(mcts.N[s])
        # actual policy
        P = [n/sum_N for n in mcts.N[s]]
        suggested_move = MOVES[np.random.choice(len(MOVES), p=P)]
        return [[suggested_move]]