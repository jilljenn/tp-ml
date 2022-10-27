"""
Matrix completion on toy and Movielens datasets
JJV for Deep Learning course, 2022
"""
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


LEARNING_RATE = 0.1
EMBEDDING_SIZE = 3


DATA = 'toy'
# DATA = 'movielens'
if DATA == 'toy':
    N_EPOCHS = 1000
    DISPLAY_EPOCH_EVERY = 100
    BATCH_SIZE = 50
    N, K, M = 10, 3, 5
    U = np.random.normal(size=(N, K))
    V = np.random.normal(size=(M, K))
    R = U @ V.T
    X = []
    y = []
    for i in range(N):  # Can be done using pd.unstack
        for j in range(M):
            X.append((i, N + j))
            y.append(R[i, j])
else:
    N_EPOCHS = 50
    DISPLAY_EPOCH_EVERY = 2
    BATCH_SIZE = 1000
    df = pd.read_csv('ml-latest-small/ratings.csv')
    films = pd.read_csv('ml-latest-small/movies.csv')
    df = df.merge(films, on='movieId')
    df['user'] = np.unique(df['userId'], return_inverse=True)[1]
    df['item'] = np.unique(df['movieId'], return_inverse=True)[1]
    N = df['user'].nunique()
    M = df['item'].nunique()
    df['item'] += N
    X = torch.LongTensor(df[['user', 'item']].to_numpy())
    y = torch.Tensor(df['rating'])

X = torch.LongTensor(X)
y = torch.Tensor(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True)
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
train_iter = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = nn.Sequential(
    ...  # Your code here
)


class CF(nn.Module):
    """
    Recommender system
    """
    def __init__(self, embedding_size):
        super().__init__()
        # Your code here

    def forward(self, x):
        # Your code here


# model = CF(EMBEDDING_SIZE)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
losses = []
for epoch in tqdm(range(N_EPOCHS)):
    losses = []
    for indices, target in train_iter:
        outputs = model(indices).squeeze()
        # print(outputs.shape)
        loss = loss_function(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    if epoch % DISPLAY_EPOCH_EVERY == 0:
        print(f"Epoch {epoch}: Train MSE {np.mean(losses)}")

        y_pred = model(X_test).squeeze()
        loss = loss_function(y_pred, y_test)
        print('Test MSE', loss)


writer = SummaryWriter(log_dir='logs/embeddings')  # TBoard
item_embeddings = list(model.parameters())[1][N:]
user_meta = pd.DataFrame(np.arange(N), columns=('item',))
user_meta['title'] = ''
item_meta = df.sort_values('item')[['item', 'title']].drop_duplicates()
metadata = pd.concat((user_meta, item_meta), axis=0)
writer.add_embedding(
    item_embeddings, metadata=item_meta.values.tolist(),
    metadata_header=item_meta.columns.tolist())
writer.close()
