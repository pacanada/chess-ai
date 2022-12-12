from pathlib import Path
from typing import List
from chess_python.chess import State
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SQUARES = [file+str(rank+1) for file in "abcdefgh" for rank in range(8)]
PROMOTION_MOVES_STRAIGHT = [file+"7"+file+"8" for file in "abcdefgh"]+[file+"2"+file+"1" for file in "abcdefgh"]
PROMOTION_MOVES_DIAG = ["a7b8", "b7a8", "b7c8","c7b8", "c7d8", "d7c8", "d7e8", "e7d8", "e7f8", "f7e8", "f7g8", "g7f8", "g7h8", "h7g8"] + ["a2b1", "b2a1", "b2c1","c2b1", "c2d1", "d2c1", "d2e1", "e2d1", "e2f1", "f2g1", "f2e1", "g2f1", "g2h1", "h2g1"]
PROMOTION_MOVES = PROMOTION_MOVES_DIAG +PROMOTION_MOVES_STRAIGHT
MOVES = [i+f for i in SQUARES for f in SQUARES if i!=f]+[move+promotion for move in PROMOTION_MOVES  for promotion in "nbrq"]
LEN_MOVES = len(MOVES)

def get_root_dir():
    return Path(".").absolute() / "chess_ai" / "rlagent" 

class BufferDataset(Dataset):
    def __init__(self, x, y_value, y_policy):
        super(BufferDataset, self).__init__()
        assert x.shape[0] == y_value.shape[0] == y_policy.shape[0]
        self.x = x
        self.y_value = y_value
        self.y_policy = y_policy
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y_value[index], self.y_policy[index]

def process_buffer_to_torch(buffer: pd.DataFrame):
    # ouch
    x = torch.tensor(np.stack(buffer.state.values, axis=0), dtype=torch.float32)
    y_values = torch.tensor(buffer.value.values,dtype=torch.float32).view(-1,1) # [:,1]
    y_policy = torch.tensor(np.stack(buffer.policy.values, axis=0), dtype=torch.float32)
    return x, y_values, y_policy

def process_buffer_to_torch_state_64(buffer: pd.DataFrame):
    # ouch
    buffer["state_64"] = buffer.state.apply(lambda x: x[:-3])
    #print(len(buffer.state_64.iloc[0]))
    x = torch.tensor(np.stack(buffer.state_64.values, axis=0), dtype=torch.float32).view(-1,64)
    y_values = torch.tensor(buffer.value.values,dtype=torch.float32).view(-1,1) # [:,1]
    y_policy = torch.tensor(np.stack(buffer.policy.values, axis=0), dtype=torch.float32)
    return x, y_values, y_policy

def process_buffer_to_torch_state_72(buffer: pd.DataFrame):
    # ouch
    buffer["state_72"] = buffer.state.apply(lambda x: np.concatenate([x, np.array([0]*5)]))
    #print(len(buffer.state_64.iloc[0]))
    x = torch.tensor(np.stack(buffer.state_72.values, axis=0), dtype=torch.float32).view(-1,72)
    y_values = torch.tensor(buffer.value.values,dtype=torch.float32).view(-1,1) # [:,1]
    y_policy = torch.tensor(np.stack(buffer.policy.values, axis=0), dtype=torch.float32)
    return x, y_values, y_policy

def loss_policy_f(inputs, targets):
        return -torch.sum(targets * inputs) / targets.size()[0]

def get_en_passant_index_offset(en_passant_allowed: List[int])-> int:
    """Transform from en_passant encoding used in chess.state to simple 0 to 16 encoding. 0 nothing,
    1-8 white rank 3 and 9-16 black rank 5"""
    if len(en_passant_allowed)==0:
        return 0
    index = en_passant_allowed[0]
    if index >=16 and index <= 23:
        # a3 would be 1
        return index - 15 
    if index >=40 and index <= 47:
        return index - 31

def get_castling_encoding(c_e: List[int]):
    # {"Q": 0, "K": 1, "q": 2, "k": 3}
    bit_string = "".join(["1" if 0 in c_e else "0", "1" if 1 in c_e else "0", "1" if 2 in c_e else "0", "1" if 3 in c_e else "0"])
    #16 for [0,1,2,3] (1111)
    return int(bit_string, 2)

def encode_state(state: State):
    # TODO: this is far from efficient, there must be a clever way to encode a chess position for nn
    en_passant_encoding: int = get_en_passant_index_offset(state.en_passant_allowed)
    encoded_state = state.board+[state.turn]+[en_passant_encoding] + [get_castling_encoding(state.castling_rights)]
    return encoded_state

def process_buffer(result: int, buffer: List):
    df = pd.DataFrame(buffer, columns=["state", "policy", "player"])
    if result == 0:
        df["value"] = 0
    else:
        df["value"] = result*df["player"]
    return df

def softmax_with_temparature(vec: list, temperature=10):
    """
    https://stats.stackexchange.com/questions/419751/why-is-softmax-function-used-to-calculate-probabilities-although-we-can-divide-e
    """
    exp_vec = np.exp([value*temperature for value in vec])
    return exp_vec/sum(exp_vec)