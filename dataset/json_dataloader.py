import json

def get_data_dict(path):
    with open(path, encoding='utf-8') as f:
        data_dict = json.load(f)
    return data_dict