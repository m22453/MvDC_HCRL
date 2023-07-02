import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn import Linear
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
import sys 
sys.path.append("..") 
from cm_plot import clustering_score as eva
import scipy.io as sio
import copy



class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, z


class ATF(nn.Module):
    def __init__(self, vocab_size, d_model, o_dim, h_dim):
        # for contrastive learning with transformer based model
        super(ATF, self).__init__()

        self.emb = nn.Embedding(vocab_size, d_model)

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
        self.transformer_enc = nn.TransformerEncoder(enc_layer, num_layers=6)

        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8, batch_first=True)
        self.transformer_dec = nn.TransformerDecoder(dec_layer, num_layers=6)

        self.cov_layer = nn.Sequential(
                            # in_channels-特征维度, out_channels-卷积核的数量（feature map: 想把原来的特征维度转换为多少）, kernel_size-卷积方向核的大小
                            nn.Conv1d(in_channels=d_model, out_channels=h_dim, kernel_size=3, padding=1), 
                            nn.MaxPool1d(kernel_size=2, stride=1)
                        )
        self.lin_layer = nn.Sequential(
                            nn.Linear(h_dim, h_dim),
                            nn.ReLU(),
                            nn.Linear(h_dim, o_dim),
                        )


    def forward(self, x):
        print('x.size', x.size())
        word_embr = self.emb(x)
        print('word_embr.size', word_embr.size())
        enc_out = self.transformer_enc(word_embr)
        print('enc_out.size', enc_out.size())
        dec_out = self.transformer_dec(word_embr, enc_out)
        print('dec_out.size', dec_out.size())
        cov_in = dec_out.permute(0,2,1)
        print('cov_in.size',cov_in.size())
        cov_out = self.cov_layer(cov_in).permute(0,2,1)
        print('cov_out.size', cov_out.size())
        mean_out = cov_out.mean(dim=1)
        print('mean_out.size', mean_out.size())
        out = self.lin_layer(mean_out)
        print('out.size', out.size())
        return (enc_out, dec_out, out)


class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001 * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def pretrain_ae(model, dataset, y, n_epoch, data_name):
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=1e-3)
    for epoch in range(n_epoch):
        # adjust_learning_rate(optimizer, epoch)
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.cuda()

            x_bar, _ = model(x)
            loss = F.mse_loss(x_bar, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            x = torch.Tensor(dataset.x).cuda().float()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            print('{} loss: {}'.format(epoch, loss))           
            kmeans = KMeans(n_clusters=len(np.unique(y)), n_init=20).fit(z.data.cpu().numpy())
            print(eva(y, kmeans.labels_))

        torch.save(model.state_dict(), './MAT/non-main_view_model/{}_{}.pkl'.format(data_name, 'x2'))


def main():
    data_name = 'chinese_news_3'
    data = sio.loadmat('./MAT/'+data_name+'.mat')
    label = data['y'] 
    n_label = len(np.unique(label))
    # label = np.reshape(label, (-1, 1))
    views = [data['x0'], data['x1'], data['x2']]
    # for chinese_news_3, x1 for non-contextual view
    # the last of views is the main view

    x = views[-1]
    y = label[0]

    model = AE(
            n_enc_1=500,
            n_enc_2=500,
            n_enc_3=2000,
            n_dec_1=2000,
            n_dec_2=500,
            n_dec_3=500,
            n_input=x.shape[-1],
            n_z=n_label*2,).cuda()

    dataset = LoadDataset(x)
    num_epoch = 10
    pretrain_ae(model, dataset, y, num_epoch, data_name=data_name)


# n = 2
# o_dim = 5
# d_model=512
# h_dim = 100
# x = torch.LongTensor([[5,2,1,0,0],[1,3,1,4,0]])

# atf = ATF(10, d_model, o_dim, h_dim)
# out = atf(x)

