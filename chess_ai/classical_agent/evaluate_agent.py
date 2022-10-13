import time

from chess_python.chess import Chess
from classical_agent.agent import Agent

from chess_ai.evaluation.utils import FEN_POSITIONS, evaluate_move

# depth 4 3.93 / 3.65
# depth 3 2.86 / 2.79
# depth 2 4.45 (weird) / 4.47
# depth 1 1.37 / with pawns 2.17


def test_engine():
    all_evals = []
    for fen_pos in FEN_POSITIONS:
        print("\n", fen_pos)
        chess = Chess(fen_pos)
        # something is wrong with transpositions
        agent = Agent(
            depth=2,
            color=chess.state.turn,
            alpha_beta=True,
            move_ordering=True,
            use_transpositions=True,
        )
        t0 = time.time()
        recommended_moves = agent.recommend(node=chess, order=True, random_flag=False)

        t1 = time.time()
        # in case there are several moves with same value
        best_value = recommended_moves[0][1]
        best_moves = [move[0] for move in recommended_moves if move[1] == best_value]
        print(best_moves)
        print(
            f"{agent.depth=}, {agent.alpha_beta=} {agent.move_ordering=}, Time: {(t1-t0):.2f}"
            f"Nodes visited: {agent.nodes_visited}. Transpositions found {agent.transpositions_found}"
        )
        print(recommended_moves[:], "\n")

        evaluations = []

        for move in best_moves:
            evaluations.append(evaluate_move(fen_pos, move))
        evaluation_avg = sum(evaluations) / len(evaluations)
        print("Evaluation", evaluation_avg)
        all_evals.append(evaluation_avg)
        if len(best_moves) > 1:
            print(
                "Evaluations has been averaged. All evals: ",
                [round(evalu, 2) for evalu in evaluations],
            )

    print(f"Overall score for depth {agent.depth} {sum(all_evals)/len(all_evals)}")


if __name__ == "__main__":
    test_engine()
