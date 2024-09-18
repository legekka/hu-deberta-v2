from datasets import Dataset, DatasetDict

import os
import json

def load_husst_dataset(directory_path):

    with open(os.path.join(directory_path, 'data_cola_train.json'), encoding="utf-8") as f:
        train_data = json.load(f)["data"]
    with open(os.path.join(directory_path, 'data_cola_dev.json'), encoding="utf-8") as f:
        eval_data = json.load(f)["data"]
    with open(os.path.join(directory_path, 'data_cola_test.json'), encoding="utf-8") as f:
        test_data = json.load(f)["data"]

    
    data = {
        "Sent_id": [],
        "text": [],
        "label": []
    }

    for example in train_data:
        data["Sent_id"].append(example["Sent_id"])
        data["text"].append(example["Sent"])
        data["label"].append(int(example["Label"]))

    train_dataset = Dataset.from_dict(data)

    data = {
        "Sent_id": [],
        "text": [],
        "label": []
    }

    for example in eval_data:
        data["Sent_id"].append(example["Sent_id"])
        data["text"].append(example["Sent"])
        data["label"].append(int(example["Label"]))

    eval_dataset = Dataset.from_dict(data)

    data = {
        "Sent_id": [],
        "text": []
    }

    for example in test_data:
        data["Sent_id"].append(example["Sent_id"])
        data["text"].append(example["Sent"])

    test_dataset = Dataset.from_dict(data)

    return DatasetDict({
        'train': train_dataset,
        'eval': eval_dataset,
        'test': test_dataset
    })