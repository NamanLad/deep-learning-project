# from mppp.jsonl get all the values of the "text" field and print them

import json
import sys
import os
import logging

def get_text_from_jsonl(jsonl_file):
    """
    Get all the values of the "text" field from a jsonl file.
    :param jsonl_file: The path to the jsonl file.
    :return: A list of the values of the "text" field.
    """
    texts = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'text' in data:
                    texts.append(data['text'])
                if len(texts) >= 100:
                    break
            except json.JSONDecodeError:
                logging.error(f"Failed to decode JSON line: {line}")
    

    #save the texts to a file
    with open('prompts.txt', 'w') as f:
        for text in texts:
            f.write(text + '\n')

get_text_from_jsonl('mbpp.jsonl')



