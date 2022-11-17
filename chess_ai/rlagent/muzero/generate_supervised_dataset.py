import numpy as np
import pandas as pd
from torch import nn
import torch.nn.functional as F

from chess_ai.rlagent.muzero.utils import MOVES, encode_state
from chess_python.chess import Chess

import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

def softmax_with_temparature(vec: list, temperature=10):
    """
    https://stats.stackexchange.com/questions/419751/why-is-softmax-function-used-to-calculate-probabilities-although-we-can-divide-e
    """
    exp_vec = np.exp([value*temperature for value in vec])
    return exp_vec/sum(exp_vec)

def rank_moves(evaluation: list, turn: int)->list:
    if turn == -1:
        evaluation = [-value for value in evaluation]
    min_value, max_value = min(evaluation), max(evaluation)
    range_ = (max_value-min_value)
    out = [(value-min_value)/range_ if range_!=0 else 1 for value in evaluation]
    return out

def generate_dataset():
    """Generate dataset to train a supervised netwrok and fine tune easily the architecutre for the reinforcement learning approach"""
    from stockfish import Stockfish
    engine = Stockfish("/opt/homebrew/bin/stockfish")
    engine.set_depth(5)
    n_moves = 20
    buffer = pd.DataFrame()
    for j in range(1000):
        engine.set_position()
        i = 0
        buffer_game = pd.DataFrame()
        while True:
            turn = 1 if i%2==0 else -1
            top_moves = engine.get_top_moves(n_moves)
            moves = [stockfish_move["Move"] for stockfish_move in top_moves]
            move_evaluation = [stockfish_move["Centipawn"] if stockfish_move["Centipawn"] is not None else stockfish_move["Mate"]*10000  for stockfish_move in top_moves]
            if len(move_evaluation)==0 or i>500:
                if i>500:
                    print("Game too long")
                    #buffer_game.to_csv("game_too_long.csv")
                # reach end
                buffer = pd.concat([buffer, buffer_game])
                break
            # normalized
            move_evaluation_ranked = rank_moves(move_evaluation, turn)
            # Note: for black high doesnt mean better!
            policy = softmax_with_temparature(move_evaluation_ranked, 20)
            # policy with respect the 4208 possible moves
            policy_complete = [0]*len(MOVES)
            indexes = [MOVES.index(move) for move in moves]
            for ii, index in enumerate(indexes):
                policy_complete[index] = policy[ii]
            move = np.random.choice(moves, p=policy)
            # Evaluation of pos
            evaluation = engine.get_evaluation()
            state = encode_state(Chess(engine.get_fen_position()).state)
            # print(f"--------Turn:{turn}")
            # print("Moves:", moves)
            # print("Policy:", policy)
            # print("Evaluation of moves:", move_evaluation_ranked)
            # print("Move:", move)
            # print("Evaluation of position", evaluation)
            # print("State:", state)
            buffer_game = buffer_game.append(pd.DataFrame({"state": [state], "policy": [policy_complete], "evaluation": [evaluation], "move_number": [i], "move": [move], "game": [j] }))

            engine.make_moves_from_current_position([move])
            i+=1
        if j%10==0:
            print("Saving at ", j)
            buffer.reset_index().to_feather("supervised_dataset.feather")
    buffer.reset_index().to_feather("supervised_dataset_last.feather")

if __name__ =="__main__":
    generate_dataset()

