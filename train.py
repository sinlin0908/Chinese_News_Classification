import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import TextDataset
from model import Model
import pickle
import copy
from tqdm import tqdm

torch.cuda.manual_seed(0)
torch.manual_seed(0)
device = torch.device('cuda:0')


def sort_batch_data(data):
    xs, lens, ys = data
    sorted_lens, sorted_id = lens.sort(dim=0, descending=True)
    sorted_xs = xs[sorted_id]
    sorted_ys = ys[sorted_id]

    return sorted_xs, sorted_lens, sorted_ys


def valid(model, valid_data_loader):
    n = 0
    acc = 0.0
    with torch.no_grad():
        for step, data in tqdm(enumerate(valid_data_loader),
                               total=len(valid_data_loader),
                               ascii=True):
            n += 1
            x, l, y = sort_batch_data(data)

            x, y = x.to(device), y.to(device)
            outs = model(x, l)

            _, predict = torch.max(outs.detach(), 1)

            correct_count = ((predict == y.detach()).sum()
                             ).double()
            acc += correct_count / y.size(0)

        return acc / n


def train(model, train_dataloader, valid_dataloader):
    model.train()

    best_acc = 0.0
    best_model_params = copy.deepcopy(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    epochs = 100

    for epoch in range(epochs):
        print(f'Epoch: {epoch + 1}/{epochs}')
        print('-' * len(f'Epoch: {epoch + 1}/{epochs}'))

        epoch_loss = 0.0
        acc = 0.0
        count = 0

        for step, data in tqdm(enumerate(train_dataloader),
                               total=len(train_dataloader),
                               ascii=True):

            count += 1
            optimizer.zero_grad()

            x, l, y = sort_batch_data(data)

            x, y, l = x.to(device), y.to(device), l.to(device)
            # print(x.size())
            outs = model(x, l)

            _, predict = torch.max(outs.detach(), 1)

            batch_loss = criterion(outs, y)
            batch_loss.backward()

            optimizer.step()

            correct_count = torch.sum(predict == y.detach())

            acc += correct_count.double() / y.size(0)

            epoch_loss += batch_loss.item() / y.size(0)

        epoch_loss = epoch_loss / count
        acc = acc / count

        print(
            f'Training loss: {epoch_loss:.4f} Acc: {acc:.4f}\n')
        if (epoch + 1) % 10 == 0:
            print("\n\n==================")
            print("Valid.....")
            model.eval()
            valid_acc = valid(model, valid_dataloader)
            print("epoch {} valid accuracy: {}".format(epoch + 1, valid_acc))
            print("==================\n\n")
            model.train()

            if valid_acc > best_acc:
                best_acc = valid_acc
                best_model_params = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_params)

    print("save model......")
    torch.save(model, 'best.model')


if __name__ == "__main__":

    with open('./emb_matrix.pickle', 'rb') as f:
        emb_matrix = pickle.load(f)

    with open('./w2id_dict.pickle', 'rb') as f:
        w2id_dict = pickle.load(f)

    labels = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    labels2id_dict = {l: i for i, l in enumerate(labels)}

    train_dataset = TextDataset(
        max_len=600,
        file_path='./data/cnews.train.txt',
        label_dict=labels2id_dict,
        w2id_dict=w2id_dict
    )

    valid_dataset = TextDataset(
        max_len=600,
        file_path='./data/cnews.val.txt',
        label_dict=labels2id_dict,
        w2id_dict=w2id_dict
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
    )

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
    )
    model = Model(
        vocab_size=len(w2id_dict),
        emb_dim=300,
        num_classes=len(labels),
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        weight_matrix=torch.FloatTensor(emb_matrix).to(device)
    ).to(device)

    print(model)

    train(model, train_dataloader, valid_dataloader)
