import pickle


with open("bug_trainer.pickle", "rb") as f:
    trainer = pickle.load(f)

print(trainer)