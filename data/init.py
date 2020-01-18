import pandas as pd
import json
import os

DATA_DIR = "nyt10"


class Samples(object):
    def __init__(self):
        self.head = []
        self.tail = []
        self.text = []
        self.relation = []

    def add_head(self, head):
        self.head.append(head)

    def add_tail(self, tail):
        self.tail.append(tail)

    def add_text(self, text):
        self.text.append(text)

    def add_relation(self, relation):
        self.relation.append(relation)

    def get_head(self):
        return self.head

    def get_tail(self):
        return self.tail

    def get_text(self):
        return self.text

    def get_relation(self):
        return self.relation


def get_json_data(file):
    with open(file, 'r') as f:
        org_data = f.readlines()
    samples = Samples()
    for data in org_data:
        data = json.loads(data)
        samples.add_text(data["text"])
        samples.add_head(data["h"]["name"])
        samples.add_tail(data["t"]["name"])
        samples.add_relation(data["relation"])
    return samples


def sample_to_csv(data, type):
    samples = pd.DataFrame(
        {
            "head": data.get_head(),
            "tail": data.get_tail(),
            "relation": data.get_relation(),
            "text": data.get_text()
        }
    )
    samples.to_csv(os.path.join(DATA_DIR, type + ".csv"), sep="\t", header=None)


if __name__ == '__main__':
    train_samples = get_json_data(os.path.join(DATA_DIR, "nyt10_train.txt"))
    eval_samples = get_json_data(os.path.join(DATA_DIR,"nyt10_val.txt"))
    test_samples = get_json_data(os.path.join(DATA_DIR, "nyt10_test.txt"))
    sample_to_csv(train_samples, "train")
    sample_to_csv(eval_samples,"eval")
    sample_to_csv(test_samples, "test")
    relations = []
    for rel in train_samples.get_relation():
        if rel not in relations:
            relations.append(rel)
    for rel in eval_samples.get_relation():
        if rel not in relations:
            relations.append(rel)
    for rel in test_samples.get_relation():
        if rel not in relations:
            relations.append(rel)
    with open(os.path.join(DATA_DIR, "rel2id.txt"), 'w') as f:
        for r in rel:
            f.write(r + "\n")
    f.close()
