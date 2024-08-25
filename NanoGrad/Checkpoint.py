import pickle

class SaveCheckpoint:
    def __init__(self, model, path):
        self.model = model
        self.path = path

    def save(self):
        with open(self.path, 'wb') as f:
            pickle.dump(self.model.parameters(), f)

class LoadCheckpoint:
    def __init__(self, model, path):
        self.model = model
        self.path = path

    def load(self):
        with open(self.path, 'rb') as f:
            params = pickle.load(f)
        for p_loaded, p in zip(params, self.model.parameters()):
            p.data = p_loaded.data

