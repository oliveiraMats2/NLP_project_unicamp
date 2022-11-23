import os
import random

import torch
import wandb
import yaml
from tqdm import trange

from save_models import SaveBestModel
from server_interface import ServerInterface

save_best_model = SaveBestModel()


def load_texts(folder):
    texts = []
    for path in os.listdir(folder):
        with open(os.path.join(folder, path)) as f:
            texts.append(f.read())
    return texts


def load_dataset_imdb(imdb_train_pos,
                      imdb_train_neg,
                      imdb_test_pos,
                      imdb_test_neg):
    x_train_pos = load_texts(imdb_train_pos)
    x_train_neg = load_texts(imdb_train_neg)
    x_test_pos = load_texts(imdb_test_pos)
    x_test_neg = load_texts(imdb_test_neg)

    x_train = x_train_pos + x_train_neg
    x_test = x_test_pos + x_test_neg
    max_valid = 500

    x_train = x_train[:max_valid + 30]#APENAS PARa debugger

    c = list(x_train)
    random.shuffle(c)

    x_valid = x_train[-max_valid:]
    x_train = x_train[:-max_valid]

    print(len(x_train), 'amostras de treino.')
    print(len(x_valid), 'amostras de desenvolvimento.')
    print(len(x_test), 'amostras de teste.')

    return x_train, x_valid


def read_yaml(file: str) -> yaml.loader.FullLoader:
    with open(file, "r") as yaml_file:
        configurations = yaml.load(yaml_file, Loader=yaml.FullLoader)

    return configurations


def set_device():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print('Using {}'.format(device))

    return device


def evaluate(model, loader, criterion, device):
    acc_loss = 0
    with torch.no_grad():
        with trange(len(loader), desc='Valid Loop') as progress_bar_valid:
            for batch_idx, sample_batch in zip(progress_bar_valid, loader):
                inputs = sample_batch[0].to(device)
                labels = sample_batch[1].to(device)

                outputs = model(inputs)
                logits = outputs.logits.permute(0, 2, 1)

                loss = criterion(logits, labels)
                acc_loss += loss.item()

                progress_bar_valid.set_postfix(
                    desc=f'iteration: {batch_idx:d}/{len(loader):d}, loss: {loss.item():.5f}, perplexity: {torch.exp(loss)} '
                )

    return acc_loss / (len(loader))


def train(model, train_loader, valid_dataloader, optimizer, criterion, num_epochs, device, configs, model_name, avaliable_time=3):
    train_loss = 0
    list_loss_valid = []
    accuracy_list_valid = []
    list_loss_train = []
    server_interface = ServerInterface(model_name=model_name, server_adress="http://127.0.0.1:8000/")  # "https://patrickctrf.loca.lt/")

    for epoch in range(num_epochs):
        with trange(len(train_loader), desc='Train Loop') as progress_bar:
            for batch_idx, sample_batch in zip(progress_bar, train_loader):
                optimizer.zero_grad()

                inputs = sample_batch[0].to(device)
                labels = sample_batch[1].to(device)

                outputs = model(inputs)
                logits = outputs.logits.permute(0, 2, 1)

                loss = criterion(logits, labels)
                train_loss += loss.item()

                progress_bar.set_postfix(
                    desc=f'[epoch: {epoch + 1:d}], iteration: {batch_idx:d}/{len(train_loader):d}, loss: {loss.item():.5f}, perplexity: {torch.exp(loss)}'
                )
                if (batch_idx + 1) % avaliable_time == 0:
                    valid_loss = evaluate(model, valid_dataloader, criterion, device)
                    # list_loss_train.append(train_loss / len(train_loader))
                    result_train_loss = train_loss / avaliable_time

                    if configs['wandb']:
                        wandb.log({'train_loss': result_train_loss,
                                   'valid_loss': valid_loss,
                                   'exp_loss_train': torch.exp(torch.Tensor([result_train_loss])),
                                   'exp_loss_valid': torch.exp(torch.Tensor([valid_loss]))})

                # with torch.no_grad():
                #     # print("single_loss: ", type(single_loss), single_loss)
                #     loss.set_(torch.randn_like(loss).detach().to(device))
                #
                # loss.backward(retain_graph=True)
                #
                # optimizer.step()
                # optimizer.zero_grad()

                model_state_dict = model.state_dict()

                server_interface.share_loss(model_state_dict)
                losses_dict = server_interface.receive_losses()

                loss.backward(retain_graph=True)

                # Retropropagamos cada loss enviada por cada modelo
                for k, single_loss in enumerate(losses_dict.values()):
                    # with torch.no_grad():
                    #     # print("single_loss: ", type(single_loss), single_loss)
                    #     loss.set_(torch.randn_like(loss).detach().to(device))
                    #     # loss.data = single_loss.data
                    loss.backward(retain_graph=True)

                    if k == 0:
                        optimizer.step()
                        optimizer.zero_grad()
