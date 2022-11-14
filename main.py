import random, torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, BertTokenizer
from imdb_dataset import ImdbDataset
from save_logs import SaveLoss
from utils import set_device, load_texts, train, evaluate


# save in yaml
configs = {'learning_rate': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
           'num_iterations': 50,
           'criterion': torch.nn.CrossEntropyLoss(),
           'batch_size_train': [1, 10000, 1000, 100, 10, 1],
           'batch_size_valid': 100,
           'context_size': 300,
           'models': ['google/bert_uncased_L-2_H-128_A-2',
                      'google/bert_uncased_L-4_H-256_A-4',
                      'google/bert_uncased_L-8_H-512_A-8']
           }

device = set_device()
x_train_pos = load_texts('aclImdb/train/pos')
x_train_neg = load_texts('aclImdb/train/neg')
x_test_pos = load_texts('aclImdb/test/pos')
x_test_neg = load_texts('aclImdb/test/neg')

x_train = x_train_pos + x_train_neg
x_test = x_test_pos + x_test_neg
y_train = [1] * len(x_train_pos) + [0] * len(x_train_neg)
y_test = [1] * len(x_test_pos) + [0] * len(x_test_neg)

max_valid = 5000

c = list(zip(x_train, y_train))
random.shuffle(c)
x_train, y_train = zip(*c)

x_valid = x_train[-max_valid:]
y_valid = y_train[-max_valid:]
x_train = x_train[:-max_valid]
y_train = y_train[:-max_valid]

print(len(x_train), 'amostras de treino.')
print(len(x_valid), 'amostras de desenvolvimento.')
print(len(x_test), 'amostras de teste.')

save_scores = SaveLoss('')

config = GPT2Config()

model = GPT2LMHeadModel.from_pretrained("gpt2")
#model = BertForSequenceClassification.from_pretrained(configs['models'][0])

#tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer = BertTokenizer.from_pretrained(configs['models'][0])

truncate = 50


train_dataset = ImdbDataset(x_train[:truncate], tokenizer, configs["context_size"])

valid_dataset = ImdbDataset(x_valid[:truncate], tokenizer, configs["context_size"])

train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=configs['batch_size_train'][3],
                                               shuffle=False)# change

valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=configs['batch_size_valid'],
                                               shuffle=False)

optimizer = torch.optim.SGD(model.parameters(), lr=configs['learning_rate'][3])

dict_statistics = {'train_loss': [],
                   'valid_loss': [],
                   'valid_accuracy': []}
#model, train_loader, valid_dataloader, optimizer, criterion, num_epochs, device
list_loss_train, accuracy_list_valid, list_loss_valid = train(model.to(device),
                                                              train_dataloader,
                                                              valid_dataloader,
                                                              optimizer,
                                                              configs['criterion'],
                                                              configs['num_iterations'],
                                                              device)

dict_statistics['train_loss'] += list_loss_train
dict_statistics['valid_loss'] += list_loss_valid
dict_statistics['valid_accuracy'] += accuracy_list_valid
