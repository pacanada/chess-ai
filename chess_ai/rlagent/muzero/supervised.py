import numpy as np
from torch import nn
import torch.nn.functional as F
class AlphazeroNet(nn.Module):
    def __init__(self):
        super(AlphazeroNet, self).__init__()
        self.l1 = nn.Linear(67, 10)
        self.l2 = nn.Linear(10, 1)
        self.l3 = nn.Linear(10, 4208)
        self.activation = nn.Tanh()
    def forward(self, x):
        x = self.l1(x)
        #x = self.activation(x)
        x_pol = F.relu(self.l3(x))
        x = self.l2(x)
        return self.activation(x), F.softmax(x_pol, dim=1)

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
    engine.set_position()
    i = 0
    while True:
        turn = 1 if i%2==0 else -1
        top_moves = engine.get_top_moves(n_moves)
        moves = [stockfish_move["Move"] for stockfish_move in top_moves]
        move_evaluation = [stockfish_move["Centipawn"] if stockfish_move["Centipawn"] is not None else stockfish_move["Mate"]*10000  for stockfish_move in top_moves]
        if None in move_evaluation:
            print("here")
        if len(move_evaluation)==0:
            # reach end
            break
        # normalized
        move_evaluation_ranked = rank_moves(move_evaluation, turn)
        # Note: for black high doesnt mean better!
        policy = softmax_with_temparature(move_evaluation_ranked, 20)
        move = np.random.choice(moves, p=policy)
        # Evaluation of pos
        evaluation = engine.get_evaluation()
        print(f"--------Turn:{turn}")
        print("Moves:", moves)
        print("Policy:", policy)
        print("Evaluation of moves:", move_evaluation_ranked)
        print("Move:", move)
        print("Evaluation of position", evaluation)
        engine.make_moves_from_current_position([move])
        i+=1

if __name__ =="__main__":
    generate_dataset()

