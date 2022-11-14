import torch
import os
from tqdm import trange
from save_models import SaveBestModel
import yaml

save_best_model = SaveBestModel()


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


def load_texts(folder):
    texts = []
    for path in os.listdir(folder):
        with open(os.path.join(folder, path)) as f:
            texts.append(f.read())
    return texts


def evaluate(model, loader, criterion, device):
    acc_loss = 0
    with torch.no_grad():
        model.eval()

        with trange(len(loader), desc='Train Loop') as progress_bar:
            for batch_idx, sample_batch in zip(progress_bar, loader):
                inputs = sample_batch[0].to(device)
                labels = sample_batch[1].to(device)

                outputs = model(inputs)
                logits = outputs.logits.permute(0, 2, 1)

                loss = criterion(logits, labels)
                acc_loss += loss.item()

                progress_bar.set_postfix(
                    desc=f'iteration: {batch_idx:d}/{len(loader):d}, loss: {loss.item():.5f}, perplexity: {torch.exp(loss)} '
                )

    return loss / (len(loader))


def train(model, train_loader, valid_dataloader, optimizer, criterion, num_epochs, device, avaliable_time=100):
    train_loss = 0
    list_loss_valid = []
    accuracy_list_valid = []
    list_loss_train = []

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

                loss.backward()
                optimizer.step()

            valid_loss = evaluate(model, valid_dataloader, criterion, device)
            list_loss_train.append(train_loss / len(train_loader))

    return list_loss_train, accuracy_list_valid, list_loss_valid
