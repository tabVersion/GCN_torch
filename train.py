from utils import load_data, preprocess_features, preprocess_adj, masked_loss, masked_acc
import torch
import numpy as np
from model import Model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_decay = 1e-4
hidden_units = 128
learning_rate = 2e-4
epochs = 500
dropout = 0.5


def validation(model, features, support, val_label, val_mask):
    model.eval()
    output = model(features, support)
    loss_val = masked_loss(output, val_label, val_mask)
    acc_val = masked_acc(output, val_label, val_mask)
    print(f'[validation] loss = {loss_val} acc = {acc_val}')
    model.train()


def test_model(model, features, support, test_label, test_mask):
    model.eval()
    output = model(features, support)
    loss_val = masked_loss(output, test_label, test_mask)
    acc_val = masked_acc(output, test_label, test_mask)
    print(f'[test] loss = {loss_val} acc = {acc_val}')


def train(model, features, support, train_label, train_mask, val_label, val_mask, test_label, test_mask):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    optimizer.zero_grad()

    for epoch in range(epochs):
        output = model(features, support)
        loss_train = masked_loss(output, train_label, train_mask)
        loss_train += model.l2_loss() * weight_decay
        acc_train = masked_acc(output, train_label, train_mask)
        loss_train.backward()
        optimizer.step()
        print(f'epoch: {epoch} loss = {loss_train} acc: {acc_train}')
        if (epoch + 1) % 10 == 0:
            validation(model, features, support, val_label, val_mask)

    test_model(model, features, support, test_label, test_mask)


def init_network(model, method='kaiming', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    torch.nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    torch.nn.init.kaiming_normal_(w)
                else:
                    torch.nn.init.normal_(w)
            elif 'bias' in name:
                torch.nn.init.constant_(w, 0)
            else:
                pass


if __name__ == '__main__':
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora')

    print('adj:', adj.shape)
    print('features:', features.shape)
    print('y:', y_train.shape, y_val.shape, y_test.shape)
    print('mask:', train_mask.shape, val_mask.shape, test_mask.shape)

    features = preprocess_features(features)  # [49216, 2], [49216], [2708, 1433]
    supports = preprocess_adj(adj)

    train_label = torch.from_numpy(y_train).long().to(device)
    num_classes = train_label.shape[1]
    train_label = train_label.argmax(dim=1)
    train_mask = torch.from_numpy(train_mask.astype(np.int)).to(device)
    val_label = torch.from_numpy(y_val).long().to(device)
    val_label = val_label.argmax(dim=1)
    val_mask = torch.from_numpy(val_mask.astype(np.int)).to(device)
    test_label = torch.from_numpy(y_test).long().to(device)
    test_label = test_label.argmax(dim=1)
    test_mask = torch.from_numpy(test_mask.astype(np.int)).to(device)

    i = torch.from_numpy(features[0].astype(np.float)).long().to(device)
    v = torch.from_numpy(features[1]).to(device)
    feature = torch.sparse.FloatTensor(i.t(), v, features[2]).to(device)

    i = torch.from_numpy(supports[0].astype(np.float)).long().to(device)
    v = torch.from_numpy(supports[1]).to(device)
    support = torch.sparse.FloatTensor(i.t(), v, supports[2]).float().to(device)

    print('x :', feature)
    print('support:', support)
    num_features_nonzero = feature._nnz()
    feat_dim = feature.shape[1]

    model = Model(feat_dim, num_classes, hidden_units=hidden_units, dropout=dropout).to(device)
    # init_network(model)

    train(model, feature, support, train_label, train_mask, val_label, val_mask, test_label, test_mask)
