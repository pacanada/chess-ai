import chess
import chess.engine
from chess import Board, Move

game = chess

engine = chess.engine.SimpleEngine.popen_uci("/opt/homebrew/bin/stockfish")

#  # Iterate through all moves, play them on a board and analyse them.
board = Board()
result = engine.analysis(board, chess.engine.Limit(time=0.1), multipv=5)
print(result)

# board.push(Move.from_uci("e2e4"))
# print(board.fen())

# board = Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
# print(board.fen(en_passant="legal"))
# print(board.fen(en_passant="fen"))
# print(board.fen(en_passant="xfen"))
# for move in board.legal_moves:
#     print(move)
#     board.push(move)
#     print(engine.analyse(board, chess.engine.Limit(time=0.1))["score"])