import json
import os


class Config:
    """
    Reads in json config file, and provides access to the key/value items.
    """

    def __init__(self, network_json):
        """
        :param network_json: A string, json config file path
        """
        self._json_dict = self.parse_json(network_json)

    @property
    def json_dict(self):
        return self._json_dict

    @staticmethod
    def parse_json(network_json):
        """
        Deserializes the json file into a dict

        :param network_json: A string, the json config file path
        :return: A dict representing the deserialized json
        """
        if os.path.isfile(network_json):
            with open(network_json, 'r') as f:
                j = json.load(f)  # dict
        else:
            assert isinstance(network_json, str)
            j = json.loads(network_json)
        return j

    def get_value(self, key, default_val=None):
        """
        Returns the value corresponding to the key value in the json file

        :param key: A string representing the key value
        :param default_val: If key is not found then return this value
        """

        value = self.json_dict.get(key, default_val)
        if value and value.lower() == 'none':
            return None
        return value
