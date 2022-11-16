import random
import time
import numpy as np
import pandas as pd
from chess_ai.rlagent.muzero.utils import get_root_dir, BufferDataset, process_buffer_to_torch, encode_state, MOVES, LEN_MOVES, process_buffer
from chess_ai.rlagent.muzero.models import AlphazeroNet
from chess_python.chess import Chess, State
from typing import Dict, List
import torch
from torch.utils.data import DataLoader
import math
from typing import Dict, List
from copy import deepcopy

class MCTS:
    def __init__(self, n_sim: int, nn:AlphazeroNet, game:Chess):
        self.c = 1/3
        self.n_sim = n_sim
        self.nn = nn
        self.game = game #deepcopy(game)
        self.N: Dict[State, list]= {} # number of times a action is chosen
        self.P: Dict[State, list] = {}
        self.Q: Dict[State, list] = {}
        self.visited: List[list] = []
    def search(self, game:Chess)->float:
        s = hash(game.state)
        if game.result is not None:
            return -game.result
        if s not in self.visited:
            self.visited.append(s)
            v_model, policy = self.nn(torch.tensor(encode_state(game.state), dtype=torch.float32).view(-1,67)) #self.nn.predict(game.state)
            self.P[s] = policy.tolist()[0]
            self.N[s] = [0]*LEN_MOVES
            self.Q[s] = [0]*LEN_MOVES
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
        v = self.search(game=game)
        self.Q[s][a] = (self.N[s][a]*self.Q[s][a] + v)/(1+self.N[s][a])
        self.N[s][a] += 1
        return -v
        
    def run(self):
        for _ in range(self.n_sim):
            game = deepcopy(self.game)
            self.search(game=game)

class Episode:
    def __init__(self, n_games: int, n_mcts: int):
        self.n_mcts = n_mcts
        self.n_games = n_games
        self.buffer = pd.DataFrame()
        self.loss_list = []
        model = AlphazeroNet()
        model.load_state_dict(torch.load(get_root_dir() / "checkpoints/nn_latest.pth"))
        model.eval()
        self.model = model

    def run_episode(self):
        for i in range(self.n_games):
            buffer_per_game = self._run_game()
            buffer_per_game["episode"] = i
            self.buffer = pd.concat([self.buffer, buffer_per_game])

    def _run_game(self):
        buffer_per_game = []
        game = Chess()
        count = 0
        while True:
            mcts = MCTS(n_sim=self.n_mcts, nn=self.model, game=game)
            mcts.run()
            s = hash(game.state)
            sum_N = sum(mcts.N[s])
            # actual policy
            P = [n/sum_N for n in mcts.N[s]]
            buffer_per_game.append([encode_state(game.state), P, game.state.turn])
            # actually in the original paper introduce a temperature to map the N to the actual policy, for the first 30 moves is just proportional to the visit count then chosing the most visited one
            suggested_move = MOVES[np.random.choice(len(MOVES), p=P)]
            # print(count, "Suggested move", suggested_move)
            count+=1
            game=game.move(suggested_move)
            game.update_outcome()
            if game.result is not None:
                print(game.result)
                print(game)
                return process_buffer(game.result, buffer_per_game)
    def train(self, epochs, batch_size):
        x, y_value, y_policy = process_buffer_to_torch(self.buffer)
        dataset = BufferDataset(x=x,y_value=y_value, y_policy=y_policy)
        train_dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size)

        loss_v_f = torch.nn.MSELoss()
        loss_policy_f = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=0)
        self.model.train()

        for it in range(epochs):
            for x, y_value, y_policy in train_dataloader:
                optimizer.zero_grad()
                y_value_pred, y_policy_pred = self.model(x)        
                loss_value = loss_v_f(y_value_pred, y_value)
                loss_policy = loss_policy_f(y_policy_pred, y_policy)
                loss = loss_value+loss_policy
                loss.backward()
                optimizer.step()

            self.loss_list.append(loss.mean().detach().numpy())
            print(it, loss.mean())
        torch.save(self.model.state_dict(), get_root_dir() /"checkpoints/nn_latest.pth")
        torch.save(self.model.state_dict(), get_root_dir() / f"checkpoints/nn_{time.time()}.pth")
        
    def compete():
        pass

if __name__ == "__main__":
    for i in range(10):
        episode = Episode(n_games=30, n_mcts=5)
        episode.run_episode()
        episode.train(epochs=100, batch_size=500)