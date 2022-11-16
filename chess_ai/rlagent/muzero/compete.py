import numpy as np
import torch
from chess_ai.rlagent.muzero.all import MCTS
from chess_ai.rlagent.muzero.models import AlphazeroNet
from chess_python.chess import Chess, State

from chess_ai.rlagent.muzero.utils import MOVES, get_root_dir

white = AlphazeroNet()
white.load_state_dict(torch.load(get_root_dir() / "checkpoints/nn_latest.pth"))
white.eval()

black = AlphazeroNet()
black.load_state_dict(torch.load(get_root_dir() / "checkpoints/nn_1668530445.68187.pth"))
black.eval()

for i in range(10):

    game = Chess()
    while True:
        model = white if game.state.turn==1 else black
        
        #if game.state.turn==1:
            #suggested_move = np.random.choice(game.legal_moves())
        mcts = MCTS(n_sim=30, nn=model, game=game)
        mcts.run()
        s = hash(game.state)
        sum_N = sum(mcts.N[s])
        # actual policy
        P = [n/sum_N for n in mcts.N[s]]
        suggested_move = MOVES[np.random.choice(len(MOVES), p=P)]
        #else:
        #    suggested_move = np.random.choice(game.legal_moves())
        #print(suggested_move)
        game.move(suggested_move)
        game.update_outcome()
        #print(game)
        if game.result is not None:
            print(game.result)
            print(game)
            break


