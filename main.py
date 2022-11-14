import random, torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, BertTokenizer, AutoModelForPreTraining
from imdb_dataset import ImdbDataset, ImdbDataset_slice
from save_logs import SaveLoss
from utils import set_device, load_texts, train, read_yaml

configs = read_yaml('configs/config_model.yaml')

device = set_device()
x_train_pos = load_texts('aclImdb/train/pos')
x_train_neg = load_texts('aclImdb/train/neg')
x_test_pos = load_texts('aclImdb/test/pos')
x_test_neg = load_texts('aclImdb/test/neg')

x_train = x_train_pos + x_train_neg
x_test = x_test_pos + x_test_neg
max_valid = 5000

c = list(x_train)
random.shuffle(c)

x_valid = x_train[-max_valid:]
x_train = x_train[:-max_valid]


print(len(x_train), 'amostras de treino.')
print(len(x_valid), 'amostras de desenvolvimento.')
print(len(x_test), 'amostras de teste.')

save_scores = SaveLoss('')

config = GPT2Config()

model = AutoModelForPreTraining.from_pretrained("gpt2")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

truncate = 50

train_dataset = ImdbDataset(x_train[:truncate], tokenizer, configs["context_size"])

valid_dataset = ImdbDataset(x_valid[:truncate], tokenizer, configs["context_size"])

train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=configs['batch_size_train'],
                                               shuffle=False)  # change

valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=configs['batch_size_valid'],
                                               shuffle=False)

optimizer = torch.optim.SGD(model.parameters(), lr=configs['learning_rate'])

dict_statistics = {'train_loss': [],
                   'valid_loss': [],
                   'valid_accuracy': []}

criterion = torch.nn.CrossEntropyLoss()
# model, train_loader, valid_dataloader, optimizer, criterion, num_epochs, device
list_loss_train, accuracy_list_valid, list_loss_valid = train(model.to(device),
                                                              train_dataloader,
                                                              valid_dataloader,
                                                              optimizer,
                                                              criterion,
                                                              configs['num_iterations'],
                                                              device)

dict_statistics['train_loss'] += list_loss_train
dict_statistics['valid_loss'] += list_loss_valid
dict_statistics['valid_accuracy'] += accuracy_list_valid
