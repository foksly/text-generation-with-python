import pickle


def save_pkl(path, file):
    """
    Saves to pkl
    """
    with open(path, 'wb') as f:
        pickle.dump(file, f)


def load_pkl(path):
    """
    Loads pkl file
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data
