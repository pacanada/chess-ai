from copy import deepcopy
import pickle
import random
import numpy as np
import pandas as pd
from typing import List
from chess_python.chess import State, Chess
from sklearn.neural_network import MLPRegressor

from chess_ai.evaluation.utils import FEN_POSITIONS, evaluate_move

def encode_state_to_bits(state:State):
    # board: 64*7figures*color ie: white pawn 0001 black pawn 1001

    # en passant: only 16 different possibilities at the same time 4 bits

    # castling 4 bits

    # turn 1 bit (0 white, 1 black)
    pass


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


class Agent:
    def __init__(self, color, game, model):
        self.color = color
        self.game = game
        self.model = model

    def evaluate_position(self, state):

        #return self.model.predict(np.array(encode_state(state)).reshape(1, -1))
        return self.model.predict(pd.DataFrame([encode_state(state)], columns=self.model.feature_names_in_))

    def recommend(self, random_move=False):
        if random_move:
            return [(random.choice(self.game.legal_moves()), 0)]
        list_moves = []
        for move in self.game.legal_moves():
            game_copied = deepcopy(self.game)
            value = self.evaluate_position(game_copied.move(move).state)
            list_moves.append((move, value))

        list_moves = sorted(list_moves, key=lambda item: item[1], reverse=self.color == 1)
        return list_moves

class Simulation:
    """Take two agents, and run a game, storing the buffer of gameplay"""
    def __init__(self, model):
        self.buffer = []
        self.outcome = None
        self.game = Chess()
        self.model = model
    def run(self):
        cont = 0

        while self.game.result is None:
            random_move = True if cont%3==0 else False
            recommended_move = Agent(color=self.game.state.turn, game=self.game, model=self.model).recommend(random_move)[0][0]
            self.game.move(recommended_move)
            self.game.update_outcome()
            self.buffer.append((deepcopy(self.game.state), encode_state(self.game.state), recommended_move))
            cont +=1
        self.outcome = self.game.result



class Trainer:
    """"""
    def __init__(self, n_sim: int, model=None):
        self.feature_columns = [f"x_{i}" for i in range(67)]
        dummy_dataset = pd.DataFrame([[0]*67], columns=self.feature_columns)
        dummy_dataset["y"] = 0
        self.model = MLPRegressor(hidden_layer_sizes=(100,50,10),tol=1e-6, max_iter=300, n_iter_no_change=1e5, learning_rate_init=0.001, warm_start=True, verbose=True).fit(dummy_dataset[self.feature_columns], dummy_dataset["y"]) if model is None else model
        self.n_sim = n_sim
        self.buffer = pd.DataFrame()
       

    def process_buffer(self, buffer_raw:List[List], result):
        df_buffer_per_sim = pd.DataFrame([row[1] for row in buffer_raw], columns=self.feature_columns)
        # not clever, just for debugging
        df_buffer_per_sim["y"] = np.linspace(0,1,df_buffer_per_sim.shape[0])*10*result
        return df_buffer_per_sim

    def run_simulations(self):
        self.num_non_draw = 0
        for i in range(self.n_sim):
            sim = Simulation(self.model)
            sim.run()
            # process buffer and append
            #if sim.outcome!=0:
            self.num_non_draw+= 1 if sim.outcome!=0 else 0 
            buffer_per_sim = self.process_buffer(buffer_raw=sim.buffer, result=sim.outcome)
            self.buffer = pd.concat([self.buffer, buffer_per_sim])
        print(f"{self.num_non_draw=}")

    def train(self):
        self.model.fit(X=self.buffer[self.feature_columns], y= self.buffer["y"])

    def evaluate(self):
        all_evals = []
        for fen_pos in FEN_POSITIONS:
            chess = Chess(fen_pos)
            # something is wrong with transpositions
            recommended_moves = Agent(color=chess.state.turn, game=chess, model=self.model).recommend()

            # in case there are several moves with same value
            best_value = recommended_moves[0][1]
            best_moves = [move[0] for move in recommended_moves if move[1] == best_value]
            
            evaluations = []

            for move in best_moves:
                evaluations.append(evaluate_move(fen_pos, move))
            evaluation_avg = sum(evaluations) / len(evaluations)
            all_evals.append(evaluation_avg)

        print(f"Overall score {sum(all_evals)/len(all_evals)}")
        return sum(all_evals)/len(all_evals)


if __name__=="__main__":
    model=None
    
    list_evaluations = []
    for i in range(20):
        try:
            with open("model.pickle", "rb") as f:
                model = pickle.load(f)
        except:
            print("Model not found, initializing")
        print("it", i)

        trainer = Trainer(n_sim=20, model=model)

        trainer.run_simulations()


        trainer.train()
        # summary 
        print("Number of rows in the buffer", trainer.buffer.shape[0])

        with open(f"model.pickle", "wb") as f:
           pickle.dump(trainer.model, f)
        if i%1==0:
            evaluation = trainer.evaluate()
            list_evaluations.append(evaluation)
            if evaluation>4:
                with open(f"model_4.pickle", "wb") as f:
                    pickle.dump(trainer.model, f)


    print(list_evaluations)
    with open("list_evaluations.pickle", "wb") as fp:   #Pickling
        pickle.dump(list_evaluations, fp)


        


    





