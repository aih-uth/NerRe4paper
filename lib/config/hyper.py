import json

from dataclasses import dataclass

@dataclass
class Hyper(object):
    def __init__(self, path: str):
        self.bert_path: str
        self.neologd_path: str
        self.manbyo_path: str
        self.exp_name: str
        self.data_path: str
        self.max_words: int
        self.idx_flag: str
        self.batch_size: int
        self.max_epoch: int
        self.skip_epoch: int
        self.task: str
        self.__dict__ = json.load(open(path, 'r'))

    def __post_init__(self):
        pass
