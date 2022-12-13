from copy import deepcopy
import numpy as np
import torch

from chess_ai.rlagent.muzero.models import ChessNet
from chess_ai.rlagent.agent import AlphaZeroAgent
from chess_ai.rlagent.muzero.utils import MOVES, encode_state, get_root_dir
from chess_python.chess import Chess


model = ChessNet()
model.load_state_dict(torch.load(get_root_dir()/  "checkpoints/nn_supervised_conv_kaggle_with_5_runs.pth", map_location=torch.device('cpu')))
model.eval()
#"1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - 0 0"
#"1k1r4/pp1b1R2/6pp/4p3/2B5/4Q3/PPP2B2/3K4 b - - 0 2"
#"1k1r4/pp3R2/6pp/4p3/2B3b1/4Q3/PPP2B2/4K3 b - - 2 3"

game = Chess("1k1r4/pp1b1R2/6pp/4p3/2B5/4Q3/PPP2B2/3K4 b - - 0 2")
# right one d6d1
# state previous to checkmate -6500941900896903514

recommended_moves = AlphaZeroAgent(game=deepcopy(game), model=model, n_sim=1000).recommend()
x = torch.tensor(np.concatenate([encode_state(game.state), np.array([0]*5)]), dtype=torch.float32).view(-1,72)
v_model, policy = model(x)
p = np.array(policy.detach())
idx = (-p).argsort()[0][:5]
moves = [MOVES[i] for i in idx]
print("v of net:", v_model.item())
print("5 best moves based on pure policy", moves)
print(recommended_moves)