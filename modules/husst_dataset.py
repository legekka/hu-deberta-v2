from datasets import Dataset, DatasetDict

import os
import json

def load_husst_dataset(directory_path):

    with open(os.path.join(directory_path, 'data_sst_train.json'), encoding="utf-8") as f:
        train_data = json.load(f)["data"]
    with open(os.path.join(directory_path, 'data_sst_dev.json'), encoding="utf-8") as f:
        eval_data = json.load(f)["data"]
    with open(os.path.join(directory_path, 'data_sst_test.json'), encoding="utf-8") as f:
        test_data = json.load(f)["data"]

    
    data = {
        "Sent_id": [],
        "text": [],
        "label": []
    }

    for example in train_data:
        data["Sent_id"].append(example["Sent_id"])
        data["text"].append(example["Sent"])
        data["label"].append(example["Label"])

    train_dataset = Dataset.from_dict(data)

    data = {
        "Sent_id": [],
        "text": [],
        "label": []
    }

    for example in eval_data:
        data["Sent_id"].append(example["Sent_id"])
        data["text"].append(example["Sent"])
        data["label"].append(example["Label"])

    eval_dataset = Dataset.from_dict(data)

    data = {
        "Sent_id": [],
        "text": []
    }

    for example in test_data:
        data["Sent_id"].append(example["Sent_id"])
        data["text"].append(example["Sent"])

    test_dataset = Dataset.from_dict(data)

    label2id = {"negative": 0, "neutral": 1, "positive": 2}

    train_dataset = train_dataset.map(lambda example: {"label": label2id[example["label"]]}, remove_columns=["Sent_id"])
    eval_dataset = eval_dataset.map(lambda example: {"label": label2id[example["label"]]}, remove_columns=["Sent_id"])

    return DatasetDict({
        'train': train_dataset,
        'eval': eval_dataset,
        'test': test_dataset
    })