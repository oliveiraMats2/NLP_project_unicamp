import torch
from tqdm import tqdm
from typing import List


class ImdbDataset_old(torch.utils.data.Dataset):
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


class ImdbDataset(torch.utils.data.Dataset):
    def __init__(self, texts: List[str], tokenizer, max_seq_length: int):
        self.max_seq_length = max_seq_length
        self.examples = []

        tokenizer.pad_token = '[PAD]'
        tokenizer.eos_token = '[SEP]'
        tokenizer.cls_token = '[EOS]'

        self.tokenizer = tokenizer

        for text in tqdm(texts):
            tokenized_text = self.tokenizer(f'{text}',
                                            return_tensors='pt',
                                            add_special_tokens=False)

            self.examples.append(tokenized_text)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class ImdbDataset_v1(torch.utils.data.Dataset):
    def __init__(self, texts: List[str], tokenizer, max_seq_length: int):
        self.max_seq_length = max_seq_length
        self.examples = []

        tokenizer.pad_token = '[PAD]'
        tokenizer.eos_token = '[SEP]'
        tokenizer.cls_token = '[EOS]'

        self.tokenizer = tokenizer

        for text in tqdm(texts):
            # insere o vetor [101]
            # Não usei o batch_encoder_plus porque eu gostaria de ver o progresso do encoder.
            tokenized_text = self.tokenizer(f'[CLS] {text}',
                                            return_tensors=None,
                                            add_special_tokens=False).input_ids

            # tokenized_text += [self.tokenizer.vocab['[PAD]']] * max(0, 1 + max_seq_length - len(tokenized_text))
            tokenized_text += [self.tokenizer.cls_token_id]
            tokenized_text += [self.tokenizer.pad_token_id] * max(0,
                                                                  1 + max_seq_length - len(tokenized_text))

            for i in range(0, len(tokenized_text) - 1, max_seq_length):
                if i + max_seq_length < len(tokenized_text):
                    self.examples.append(tokenized_text[i: i + max_seq_length + 1])

                else:
                    self.examples.append(tokenized_text[-max_seq_length - 1:])

        self.examples = torch.LongTensor(self.examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx][:-1], self.examples[idx][1:]
