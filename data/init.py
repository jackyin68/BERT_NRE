import pandas as pd
import json
import os

DATA_DIR = "nyt10"


class Samples(object):
    def __init__(self):
        self.head = []
        self.tail = []
        self.sentence = []
        self.relation = []

    def add_head(self, head):
        self.head.append(head)

    def add_tail(self, tail):
        self.tail.append(tail)

    def add_sentence(self, sentence):
        self.sentence.append(sentence)

    def add_relation(self, relation):
        self.relation.append(relation)

    def get_head(self):
        return self.head

    def get_tail(self):
        return self.tail

    def get_sentence(self):
        return self.sentence

    def get_relation(self):
        return self.relation


def get_json_data(file):
    with open(file, 'r') as f:
        org_data = json.load(f)
    samples = Samples()
    for data in org_data:
        samples.add_sentence(data["sentence"])
        samples.add_head(data["head"]["word"])
        samples.add_tail(data["tail"]["word"])
        samples.add_relation(data["relation"])
    return samples


def sample_to_csv(data, type):
    samples = pd.DataFrame(
        {
            "head": data.get_head(),
            "tail": data.get_tail(),
            "sentence": data.get_sentence(),
            "relation": data.get_relation()
        }
    )
    samples.to_csv(os.path.join(DATA_DIR, type + ".csv"), sep="\t", header=None)


if __name__ == '__main__':
    train_samples = get_json_data(os.path.join(DATA_DIR, "train.json"))
    test_samples = get_json_data(os.path.join(DATA_DIR, "test.json"))
    sample_to_csv(train_samples, "train")
    sample_to_csv(test_samples, "test")
    relations = []
    for rel in train_samples.get_relation():
        if rel not in relations:
            relations.append(rel)
    for rel in test_samples.get_relation():
        if rel not in relations:
            relations.append(rel)
    with open(os.path.join(DATA_DIR, "rel2id"), 'w') as f:
        for r in rel:
            f.write(r + "\n")
    f.close()
