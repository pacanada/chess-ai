import pickle
import pandas as pd
from chess_ai.rlagent.muzero.utils import get_root_dir



def process_buffer_pickle(buffer):
    """"""
    # Cp(-..) from black is lossing for black!!!
    buffer_df = pd.DataFrame.from_dict(data=buffer, orient="index")
    buffer_df

with open(get_root_dir() / "data" / "pickle" / f"buffer_1.pickle", "rb") as f:
    buffer = pickle.load(f)
process_buffer_pickle(buffer)

