import torch
import os
from tqdm import trange
from save_models import SaveBestModel

save_best_model = SaveBestModel()


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
    with trange(len(loader), desc='Train Loop') as progress_bar:
        for batch_idx, sample_batch in zip(progress_bar, loader):
            inputs = sample_batch.to(device)

            loss_valid = model(**inputs, labels=inputs["input_ids"]).loss

            acc_loss += loss_valid.item()

    return acc_loss / (len(loader))


def train(model, train_loader, valid_dataloader, optimizer, criterion, num_epochs, device):
    train_loss = 0
    list_loss_valid = []
    accuracy_list_valid = []
    list_loss_train = []

    for epoch in range(num_epochs):
        with trange(len(train_loader), desc='Train Loop') as progress_bar:
            for batch_idx, sample_batch in zip(progress_bar, train_loader):
                optimizer.zero_grad()

                inputs = sample_batch.to(device)

                ouputs = model(**inputs, labels=inputs["input_ids"])
                loss = ouputs.loss
                train_loss += loss.item()

                progress_bar.set_postfix(
                    desc=f'[epoch: {epoch + 1:d}], iteration: {batch_idx:d}/{len(train_loader):d}, loss: {loss.item():.5f}'
                )

                loss.backward()
                optimizer.step()

            list_loss_train.append(train_loss / len(train_loader))

            num_examples = len(valid_dataloader.dataset)

            device = 'cpu'

            evaluate(model, valid_dataloader, criterion, device)

    return list_loss_train, accuracy_list_valid, list_loss_valid
