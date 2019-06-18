# Chinese News Classification

使用 pytorch 練習 GRU 進行新聞分類

## Enviroment

- OS: unbuntu 18.04 LTS
- GPU: GTX 1070 8G
- RAM: 16G

## Requiremnet

- pytorch 1.0
- jieba
- numpy
- scikit-learn

## Dataset

10 label names: 

> 体育, 财经, 房产, 家居, 教育, 科技, 时尚, 时政, 游戏, 娱乐

[link](https://github.com/gaussic/text-classification-cnn-rnn)


- train dataset: 5000 * 10
- test dataset: 1000 * 10
- valid dataset: 500 * 10


## word embedding

Mixed-large 综合 in [link](https://github.com/Embedding/Chinese-Word-Vectors) 

word size : about 1292608


## Model

```bash
Model(
  (embedding_layer): Embedding(1291384, 300)
  (bigru): GRU(300, 64, num_layers=2, dropout=0.2, bidirectional=True)
  (output_layer): Linear(in_features=256, out_features=512, bias=True)
)

```

## Result
![](https://i.imgur.com/6Rhdzke.png)
