#https://web.stanford.edu/~surag/posts/alphazero.html
#something is off with the copying

from copy import deepcopy
import math
import random
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from chess_python.chess import State, Chess, Optimizer
import torch
from chess_ai.rlagent.muzero.models import AlphazeroNet

from chess_ai.rlagent.muzero.utils import encode_state

SQUARES = [file+str(rank+1) for file in "abcdefgh" for rank in range(8)]


PROMOTION_MOVES_STRAIGHT = [file+"7"+file+"8" for file in "abcdefgh"]+[file+"2"+file+"1" for file in "abcdefgh"]
PROMOTION_MOVES_DIAG = ["a7b8", "b7a8", "b7c8","c7b8", "c7d8", "d7c8", "d7e8", "e7d8", "e7f8", "f7e8", "f7g8", "g7f8", "g7h8", "h7g8"] + ["a2b1", "b2a1", "b2c1","c2b1", "c2d1", "d2c1", "d2e1", "e2d1", "e2f1", "f2g1", "g2f1", "g2h1", "h2g1"]
PROMOTION_MOVES = PROMOTION_MOVES_DIAG +PROMOTION_MOVES_STRAIGHT
MOVES = [i+f for i in SQUARES for f in SQUARES if i!=f]+[move+promotion for move in PROMOTION_MOVES  for promotion in "nbrq"]



A = len(MOVES)

def process_buffer(result: int, buffer: List):
    df = pd.DataFrame(buffer, columns=["state", "policy", "player"])
    if result == 0:
        df["value"] = 0
    else:
        df["value"] = result*df["player"]
    return df



class NN:
    def predict(self, s:State)->Tuple[float, list]:
        v = np.random.rand(1)[0]*2-1 # (-1,1)
        p = np.random.rand(A)
        return v, p

class Simulation:
    pass
class MCTS:
    def __init__(self, n_sim: int, nn:NN, game:Chess):
        self.c = 1/3
        self.n_sim = n_sim
        self.nn = nn
        self.game = game #deepcopy(game)
        self.N: Dict[State, list]= {} # number of times a action is chosen
        self.P: Dict[State, list] = {}
        self.Q: Dict[State, list] = {}
        self.visited: List[list] = []
    def search(self, nn: NN, game:Chess)->float:
        s = hash(game.state)
        if game.result is not None:
            return -game.result
        if s not in self.visited:
            self.visited.append(s)
            v_model, policy = self.nn(torch.tensor(encode_state(game.state), dtype=torch.float32).view(-1,67)) #self.nn.predict(game.state)
            self.P[s] = policy.tolist()[0]
            self.N[s] = [0]*A
            self.Q[s] = [0]*A
            v = v_model.item()
            return -v

        max_u, best_a = -float("inf"), -1
        for move in game.legal_moves():
            a = MOVES.index(move)
            u = self.Q[s][a] + self.c*self.P[s][a]*math.sqrt(sum(self.N[s]))/(1+self.N[s][a])
            if u>max_u:
                max_u = u
                best_a = a
        a = best_a
        game = game.move(MOVES[a])
        game.update_outcome()
        v = self.search( nn=nn, game=game)
        self.Q[s][a] = (self.N[s][a]*self.Q[s][a] + v)/(1+self.N[s][a])
        self.N[s][a] += 1
        return -v
        
    def run(self):
        for _ in range(self.n_sim):
            game = deepcopy(self.game)
            self.search(nn=self.nn, game=game)


def run_episode():
    nn = AlphazeroNet()
    nn.load_state_dict(torch.load("nn.pth"))
    nn.eval()

    buffer = []
    game = Chess() #.move("e2e4")
    count = 0
    while True:
        mcts = MCTS( n_sim=10, nn=nn, game=game)
        mcts.run()
        s = hash(game.state)
        sum_N = sum(mcts.N[s])
        # actual policy
        P = [n/sum_N for n in mcts.N[s]]
        buffer.append([encode_state(game.state), P, game.state.turn])
        # When all are negative reward, it chooses the first move a1a2 because it cannot find a 
        # winning move. We remove the non explored actions by giving -inf
        Q_inf_for_neg = [a if a!=0 else -float("inf") for a in mcts.Q[s]]
        max_value = max(Q_inf_for_neg)
        #max_value = max(mcts.Q[s])
        index_move = mcts.Q[s].index(max_value)
        suggested_move = MOVES[index_move]
        # print(count, "Suggested move", suggested_move)
        count+=1
        game=game.move(suggested_move)
        game.update_outcome()
        if game.result is not None:
            print(game.result)
            print(game)
            return process_buffer(game.result, buffer)
if __name__=="__main__":
    buffer_df = pd.DataFrame()
    for i in range(10):
        buffer_df_episode = run_episode()
        buffer_df_episode["episode"] = i
        buffer_df = pd.concat([buffer_df, buffer_df_episode])
    buffer_df.to_csv("buffer.csv")

    # not visited actions are Q=0
    # Q_inf_for_neg = [a if a!=0 else -float("inf") for a in mcts.Q[hash(game.state)]]
    #max_value = max(Q_inf_for_neg)
    # max_value = max(mcts.Q[hash(game.state)])
    # index_move = mcts.Q[hash(game.state)].index(max_value)
    # print("game state", hash(game.state))
    # print("For first state", MOVES[index_move])
    # print("bla")