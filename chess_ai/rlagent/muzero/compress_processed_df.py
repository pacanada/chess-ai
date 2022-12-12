import pickle

import torch
from chess_ai.rlagent.muzero.utils import get_root_dir, process_buffer_to_torch_state_72
import pandas as pd


buffer = pd.read_feather(get_root_dir() / "data" / "dataframe" / "buffer_df.feather")
x, y_value, y_policy = process_buffer_to_torch_state_72(buffer)

torch.save(x, get_root_dir() /  "data" / "pickle" / f"x.pt")
torch.save(y_value, get_root_dir() /  "data" / "pickle" / f"y.pt")
torch.save(y_policy.to_sparse(), get_root_dir() /  "data" / "pickle" / f"y_policy.pt")

# a = {}
# a["x"] = x
#a["y_value"] = y_value
#a["y_policy"] = y_policy

# with open(get_root_dir() / "data" / "pickle" / f"buffer.pickle", "wb") as f:
#     pickle.dump(buffer, f)