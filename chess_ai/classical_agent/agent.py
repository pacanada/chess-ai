import random
import time
from copy import deepcopy
from typing import Dict, List, Literal, Tuple

from chess_python.chess import Chess, ChessUtils

PIECE_VALUES = {
    0: 0,
    1: 1,
    2: 3,
    3: 3,
    4: 5,
    5: 9,
    6: 0,
    -1: -1,
    -2: -3,
    -3: -3,
    -4: -5,
    -5: -9,
    -6: 0,
}
PIECE_VALUES_ABS = {
    0: 0,
    1: 1,
    2: 3,
    3: 3,
    4: 5,
    5: 9,
    6: 0,
}


class ClassicEvaluator:
    def evaluate(self, state):
        # Simple count
        base_evaluation = sum([PIECE_VALUES[piece] for piece in state.board])
        # pawn positions (-0.1 for each double, isolated or blocked)
        white_pawns_blocked = [
            piece
            for pos, piece in enumerate(state.board)
            if piece == 1 and state.board[pos + 8] != 0
        ]
        black_pawns_blocked = [
            piece
            for pos, piece in enumerate(state.board)
            if piece == -1 and state.board[pos - 8] != 0
        ]
        base_evaluation += -0.2 * len(white_pawns_blocked) + 0.2 * len(black_pawns_blocked)
        return base_evaluation


class MoveOrderer:
    """Order move to cut more branches with alpha-beta"""

    def order(self, board: List[int], moves: List[str], attacked_map: List[int]):
        def rank(move: str):
            # way more things can be done here
            move_rank = 0
            end_pos = ChessUtils.POSITION_DICT[move[2:4]]
            v_end = PIECE_VALUES_ABS[abs(board[end_pos])]
            v_ini = PIECE_VALUES_ABS[abs(board[ChessUtils.POSITION_DICT[move[:2]]])]
            # promote captures
            if v_end != 0:
                move_rank = v_end - v_ini + 10
            # penalize moving to attacked position
            move_rank -= v_end if end_pos in attacked_map else 0
            return move_rank

        moves_ordered = sorted(moves, key=lambda move: rank(move), reverse=True)
        return moves_ordered


class Agent:
    """Base class to make a recommendation of best move"""

    def __init__(
        self,
        depth: int,
        color: Literal[1, -1],
        alpha_beta: bool = True,
        move_ordering: bool = True,
        use_transpositions: bool = False,
    ):
        self.color = color
        self.nodes_visited = 0
        self.alpha_beta = alpha_beta
        self.move_ordering = move_ordering
        self.use_transpositions = use_transpositions
        self.depth = depth
        self.transpositions: Dict[int, Dict[int, float]] = {i: {} for i in range(depth + 1)}
        self.transpositions_found = 0

    def minimax(self, node: Chess, depth, maximize):
        if depth == 0:
            self.nodes_visited += 1
            return ClassicEvaluator().evaluate(node.state)
        if maximize:
            maxeva = -float("inf")
            for move in node.legal_moves():
                child = deepcopy(node).move(move)
                eva = self.minimax(child, depth - 1, False)
                maxeva = max(eva, maxeva)
            return maxeva
        else:
            mineva = +float("inf")
            for move in node.legal_moves():
                child = deepcopy(node).move(move)
                eva = self.minimax(child, depth - 1, True)
                mineva = min(eva, mineva)

            return mineva

    def minimax_ab(self, node: Chess, depth, alpha, beta, maximize):
        """Working. As class method to keep track of metrics"""

        node_hash = hash(node.state)
        if self.use_transpositions:
            if node_hash in self.transpositions[depth].keys():
                self.transpositions_found += 1
                return self.transpositions[depth][node_hash]
        if depth == 0:
            self.nodes_visited += 1
            value = ClassicEvaluator().evaluate(node.state)
            if self.use_transpositions:
                # having the value cached for the leaf nodes does not help much since the evaluator
                # is not very expensive
                self.transpositions[0][node_hash] = value
            return value

        if self.move_ordering:
            legal_moves = MoveOrderer().order(
                board=node.state.board,
                moves=node.legal_moves(),
                attacked_map=node.optimizer.attacked_map,
            )
        else:
            legal_moves = node.legal_moves()

        if maximize:
            value = -float("inf")
            for move in legal_moves:

                # no need to update the optimizer if is second to last node
                child = deepcopy(node).move(move, False, True if depth != 1 else False)
                value = max(value, self.minimax_ab(child, depth - 1, alpha, beta, False))
                if value >= beta:
                    break
                alpha = max(alpha, value)
            # caching on in beta cutoff
            if self.use_transpositions:
                self.transpositions[depth][node_hash] = value

        else:
            value = +float("inf")
            for move in legal_moves:
                child = deepcopy(node).move(move, False, True if depth != 1 else False)
                value = min(value, self.minimax_ab(child, depth - 1, alpha, beta, True))
                if value <= alpha:
                    break
                beta = min(beta, value)

        return value

    def recommend(self, node: Chess, order: bool = False, random_flag=False):
        if random_flag:
            # sanity check
            legal_moves = node.legal_moves()
            return [(random.choice(legal_moves), "", "")]

        list_moves: List[Tuple[str, float, str]] = []
        if self.move_ordering:
            legal_moves = MoveOrderer().order(
                board=node.state.board,
                moves=node.legal_moves(),
                attacked_map=node.optimizer.attacked_map,
            )
        else:
            legal_moves = node.legal_moves()
        maximize = self.color == -1
        for move in legal_moves:
            ti = time.time()
            if self.alpha_beta:

                value = self.minimax_ab(
                    node=deepcopy(node).move(move),
                    depth=self.depth,
                    alpha=-float("inf"),
                    beta=float("inf"),
                    maximize=maximize,
                )
            else:
                value = self.minimax(
                    node=deepcopy(node).move(move), depth=self.depth, maximize=maximize
                )
            tf = time.time()
            list_moves.append((move, value, f"{(tf-ti):.2f}"))
        if order:
            list_moves = sorted(list_moves, key=lambda item: item[1], reverse=self.color == 1)

        return list_moves
