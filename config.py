import json


class Config:

    def __init__(self, network_json):
        self._json_dict = self.parse_json(network_json)

    @property
    def json_dict(self):
        return self._json_dict

    @staticmethod
    def parse_json(network_json):
        with open(network_json, 'r') as f:
            j = json.load(f)  # dict
            return j

    def get_value(self, key, default_val=None):
        return self.json_dict.get(key, default_val)
