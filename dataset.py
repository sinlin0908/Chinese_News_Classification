import numpy as np
import torch
from torch.utils.data import Dataset
import jieba


def tokenize(s):
    return [w for w in jieba.cut(s)]


class TextDataset(Dataset):
    def __init__(
        self,
        max_len,
        file_path,
        label_dict,
        w2id_dict
    ):

        self.x = None

        self.max_len = max_len
        self.label_dict = label_dict
        self.w2id_dict = w2id_dict

        with open(file_path, 'r', encoding='utf-8') as f:
            self.x = f.readlines()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):

        data = self.x[index].split("\t")

        content = data[1]
        tokens = tokenize(content)

        sent_idxs = torch.LongTensor(
            self.max_len).fill_(self.w2id_dict['PAD'])

        for i, w in enumerate(tokens):
            if i >= self.max_len:
                break

            if w in self.w2id_dict.keys():
                sent_idxs[i] = self.w2id_dict[w]
            else:
                sent_idxs[i] = self.w2id_dict["OOV"]

        label = self.label_dict[data[0]]

        t_len = sent_idxs.size(0)

        return sent_idxs,  t_len, label
