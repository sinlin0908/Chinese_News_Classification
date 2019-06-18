import torch
from dataset import TextDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import pickle
from tqdm import tqdm
from sklearn import metrics

device = torch.device('cuda:0')
labels = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']


def sort_batch_data(data):
    xs, lens, ys = data
    sorted_lens, sorted_id = lens.sort(dim=0, descending=True)
    sorted_xs = xs[sorted_id]
    sorted_ys = ys[sorted_id]

    return sorted_xs, sorted_lens, sorted_ys


def test(model, test_dataloader):
    model.eval()

    n = 0
    acc = 0.0

    all_y = []
    all_p = []

    with torch.no_grad():
        for data in tqdm(test_dataloader,
                         ascii=True, total=len(test_dataloader)):
            n += 1

            x, l, y = sort_batch_data(data)
            x, l, y = x.to(device), l.to(device), y.to(device)

            outs = model(x, l)

            _, predict = torch.max(outs.detach(), 1)

            correct_count = ((predict == y.detach()).sum()
                             ).double()

            all_y.extend(y.cpu().detach().numpy())
            all_p.extend(predict.cpu().detach().numpy())

            acc += correct_count / y.size(0)

        print(f"Test accuracy:{acc/n:.4f}")
        print(metrics.classification_report(
            y_true=all_y, y_pred=all_p, target_names=labels))


if __name__ == "__main__":

    with open('./emb_matrix.pickle', 'rb') as f:
        emb_matrix = pickle.load(f)

    with open('./w2id_dict.pickle', 'rb') as f:
        w2id_dict = pickle.load(f)

    labels2id_dict = {l: i for i, l in enumerate(labels)}

    test_dataset = TextDataset(
        max_len=600,
        file_path='./data/cnews.test.txt',
        label_dict=labels2id_dict,
        w2id_dict=w2id_dict
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
    )

    model = torch.load("./best.model").to(device)

    print(model)

    test(model, test_dataloader)
