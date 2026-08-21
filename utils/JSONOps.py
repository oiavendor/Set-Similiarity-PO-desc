import json
from utils.enum import JSON_DIR, JSON_FILE_EXT

def read_json(json_file: str):
    try:
        with open(JSON_DIR + json_file + JSON_FILE_EXT, 'r') as file:
            data = json.load(file)
    except:
        data = {}
    return data

def write_json(json_file: str, data: dict):
    with open(JSON_DIR + json_file + JSON_FILE_EXT, 'w') as file:
        json.dump(data, file)
