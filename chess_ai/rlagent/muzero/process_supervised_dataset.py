import pickle
import numpy as np
import pandas as pd
from chess_ai.rlagent.muzero.utils import MOVES, get_root_dir

MOVES_LENGTH = len(MOVES)

def from_set_to_general_moves(x):
    out = [0]*MOVES_LENGTH
    indexes = [MOVES.index(move) for move in x["moves"]]
    for ii, i in enumerate(indexes):
        out[i]=x["scores_softmax"][ii]
    return out


def convert_pov_score(x):
    """We have Cp(-10), Cp(10), Mate(-1).. and we want to convert that to numbers. Cp(-) is losing for the current player perspective"""
    scores = [score.relative.cp if hasattr(score.relative, "cp") else evaluate_mate(score.relative.moves) for score in x ]
    return scores

def convert_to_softmax(x):
    ranked_moves = rank_moves(x)
    out = softmax_with_temparature(ranked_moves, 1)
    return out

def softmax_with_temparature(vec: list, temperature=10):
    """
    https://stats.stackexchange.com/questions/419751/why-is-softmax-function-used-to-calculate-probabilities-although-we-can-divide-e
    """
    exp_vec = np.exp([value*temperature for value in vec])
    return exp_vec/sum(exp_vec)

def rank_moves(evaluation: list)->list:
    min_value, max_value = min(evaluation), max(evaluation)
    range_ = (max_value-min_value)
    out = [(value-min_value)/range_ if range_!=0 else 1 for value in evaluation]
    return out

def evaluate_mate(mate_value:int):
    """ The when checking for best moves or evaluation the position we can get "mate": -8, -1, 3..8..
    We want to give a corresponding value to that in terms of centipawns"""
    max_mate_value = 30
    max_ctp_no_mate = 5000
    sign = -1 if mate_value<0 else 1
    if abs(mate_value)>max_mate_value:
        raise ValueError(f"Not considered. Suggest increasing the max_mate_value. Max value seen {mate_value}")

    out = max_ctp_no_mate + (max_mate_value - abs(mate_value))*100
    return out*sign


def process_buffer_pickle(buffer):
    """"""
    # Cp(-..) from black is losing for black
    buffer_df = pd.DataFrame.from_dict(data=buffer, orient="index")
    buffer_df["scores_number"] = buffer_df.scores.apply(convert_pov_score)
    buffer_df["scores_softmax"] = buffer_df.scores_number.apply(convert_to_softmax)
    # the best position it can choose is the value (maximum score number would be the optimal for each player)
    buffer_df["value_number"] = buffer_df.scores_number.apply(lambda x: max(x))
    # softmax: visualize with plt.plot(x, 2/(1+np.exp(-1e-3*x))-1);plt.show()
    buffer_df["value"] = 2/(1+np.exp(-0.001*buffer_df.value_number))-1
    buffer_df["policy"] = buffer_df.apply(from_set_to_general_moves, axis=1)
    return buffer_df.reset_index()

buffer = pd.DataFrame()
for i in range(5):
    print(f"Processing {i+1}")
    with open(get_root_dir() / "data" / "pickle" / f"buffer_{i+1}.pickle", "rb") as f:
    #with open(get_root_dir() / "data" / "pickle" / f"buffer_new.pickle", "rb") as f:
        buffer_pickle= pickle.load(f)
    buffer_ = process_buffer_pickle(buffer_pickle)
    buffer = pd.concat([buffer, buffer_])


buffer.drop("scores", axis=1).reset_index().to_feather(get_root_dir() / "data" / "dataframe" / "buffer_df.feather") 

