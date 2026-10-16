import pickle

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

path = "frictionSensitivity/solution_67.88_2.pkl"

sol = load_pickle(path)

print(sol["tau_w"])
