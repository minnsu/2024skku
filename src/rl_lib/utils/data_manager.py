import pickle
import gzip

class DataManager:
    def __init__(self):
        self._data = None

    def set_data(self, name: str, description: str, env: str, model: object):
        if name is None or description is None or env is None or model is None:
            raise ValueError("Please provide name, description, env and model")

        self._data = {
            'name': name,
            'description': description,
            'env': env,
            'model': model
        }

    def get_data(self):
        return self._data
    
    def __str__(self):
        if self._data is None:
            return "[No data]"
        
        return f"[{self._data['name']}]\n : {self._data['description']}\n env: {self._data['env']}\n model: {self._data['model']}\n"

    def save(self, path):
        if self._data is None:
            raise ValueError("Please set_data before saving")

        with gzip.open(path, 'wb') as f:
            pickle.dump(self._data, f)
    
    def load(self, path):
        del self._data

        with gzip.open(path, 'rb') as f:
            self._data = pickle.load(f)
        return self._data