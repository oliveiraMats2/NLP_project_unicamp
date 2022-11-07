import torch
from typing import List


class ImdbDataset(torch.utils.data.Dataset):
    def __init__(self, corpus: List[str], labels: List[bool], truncate: int, tokenizer):
        self.x = []
        self.y = []
        print("----init tokenize-----")
        self.tokens_list = tokenizer(corpus, padding=True,
                                     truncation=True, max_length=truncate,
                                     return_tensors="pt", verbose=True)
        print("----finished tokenize-----")

        for tokens, mask_tokens, label in zip(self.tokens_list.input_ids,
                                              self.tokens_list.attention_mask,
                                              labels):
            self.x.append((tokens, mask_tokens))
            self.y.append(label)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]