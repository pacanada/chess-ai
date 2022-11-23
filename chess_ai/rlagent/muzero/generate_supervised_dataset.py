from copy import deepcopy, copy
import pickle
from chess import Board, Move
import chess.engine
import numpy as np

from chess_ai.rlagent.muzero.utils import MOVES, encode_state, softmax_with_temparature, get_root_dir
from chess_python.chess import Chess

def main():
    engine = chess.engine.SimpleEngine.popen_uci("/opt/homebrew/bin/stockfish")

    #  # Iterate through all moves, play them on a board and analyse them.
    #board = Board()
    n_pvs = 5

    weights = softmax_with_temparature(reversed(range(n_pvs)), temperature=1)
    buffer = {}

    for n in range(300):
        board = Board()
        i = 0
        while True:
            i+=1
            if board.outcome() is not None:
                print(board.outcome())
                break
            result = engine.analyse(board, chess.engine.Limit(time=0.01), multipv=5)
            
            moves = [res["pv"][0].uci() for res in result]
            scores = [res["score"] for res in result]
            # assuming equally good (which doesnt have to but to simplify)
            weights = [1/len(moves)]*len(moves) #range(len(moves))softmax_with_temparature(reversed(range(len(moves))), temperature=1)
            # choosing a random with weights to give some variability
            move = np.random.choice(moves, p=weights)

            fen = board.fen(en_passant="fen")
            encoded_state = encode_state(Chess(fen=fen).state)
            buffer[f"{n}_{i}"] = {"moves":moves, "scores":scores, "move": move, "state": encoded_state, "fen": fen}
            board.push(Move.from_uci(move))
    engine.quit()
    with open(get_root_dir() / "data" / "pickle" / f"buffer_4.pickle", "wb") as f:
        pickle.dump(buffer, f)




if __name__ =="__main__":
    # engine = chess.engine.SimpleEngine.popen_uci("/opt/homebrew/bin/stockfish")
    # board = Board("8/k7/8/8/QK6/8/8/8 b")
    # result_w = engine.analyse(board, chess.engine.Limit(time=0.01), multipv=5)
    #engine.quit()
    # engine = chess.engine.SimpleEngine.popen_uci("/opt/homebrew/bin/stockfish")
    # board = Board("8/k7/8/8/QK6/R7/8/8 b")
    # result_b = engine.analyse(board, chess.engine.Limit(time=0.01), multipv=5)
    # engine.quit()

    main()

