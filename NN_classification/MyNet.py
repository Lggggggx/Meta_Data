import os
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data as Data

class Net(torch.nn.Module):
    '''User-defined the NN. It has two hidden layers.

    '''
    def __init__(self, n_feature, first_n_hidden, second_n_hidden, n_output, activation=F.relu, batch_normalization=False):
        super(Net, self).__init__()
        self.do_bn = batch_normalization
        self.input_bn = torch.nn.BatchNorm1d(n_feature)
        self.first_hidden = torch.nn.Linear(n_feature, first_n_hidden)
        self.first_bn = torch.nn.BatchNorm1d(first_n_hidden)
        self.second_hidden = torch.nn.Linear(first_n_hidden, second_n_hidden)
        self.second_bn = torch.nn.BatchNorm1d(second_n_hidden)
        self.out = torch.nn.Linear(second_n_hidden, n_output)
        self.activation = activation
    
    def forward(self, x):
        if self.do_bn:
            x = self.input_bn(x)
        x = self.activation(self.first_hidden(x))
        if self.do_bn:
            x = self.first_bn(x)
        x = self.activation(self.second_hidden(x))
        if self.do_bn:
            x = self.second_bn(x)
        x = self.out(x)
        return x


class Net_3hiddens(torch.nn.Module):
    '''
    '''
    def __init__(self, n_feature, first_n_hidden, second_n_hidden, third_n_hidden, n_output, activation=F.relu, batch_normalization=False):
        super(Net_3hiddens, self).__init__()
        self.do_bn = batch_normalization
        self.input_bn = torch.nn.BatchNorm1d(n_feature)
        self.first_hidden = torch.nn.Linear(n_feature, first_n_hidden)
        self.first_bn = torch.nn.BatchNorm1d(first_n_hidden)
        self.second_hidden = torch.nn.Linear(first_n_hidden, second_n_hidden)
        self.second_bn = torch.nn.BatchNorm1d(second_n_hidden)
        self.third_hidden = torch.nn.Linear(second_n_hidden, third_n_hidden)
        self.third_bn = torch.nn.BatchNorm1d(third_n_hidden)        
        self.out = torch.nn.Linear(third_n_hidden, n_output)
        self.activation = activation
    
    def forward(self, x):
        if self.do_bn:
            x = self.input_bn(x)
        x = self.activation(self.first_hidden(x))
        if self.do_bn:
            x = self.first_bn(x)
        x = self.activation(self.second_hidden(x))
        if self.do_bn:
            x = self.second_bn(x)
        x = self.activation(self.third_hidden(x))
        if self.do_bn:
            x = self.third_bn(x)
        x = self.out(x)
        return x    


def MetaDataSet(metadata_dir, batch_size, n_workers, dataset_shuffle=True, task='classification'):
    '''

    Returns
    -------
    dataset_loader: torch.utils.data.DataLoader

    '''
    # load metadata
    metadata = None
    # Note that, metadata_dir depends on your folder path.
    

    for _, _, files in os.walk(metadata_dir):
        for file in files:
            temp_metadata = np.load(metadata_dir + file)
            if metadata is None:
                metadata = temp_metadata
            else:
                metadata = np.vstack((metadata, temp_metadata))
                
    # metadata1 = np.load('D:/generate_metadata/bigmetadata/wdbc/query_time/query_time_5interval/query_time50/2wdbc30_big_metadata50.npy')
    # metadata2 = np.load('D:/generate_metadata/bigmetadata/wdbc/query_time/query_time_5interval/query_time50/20wdbc30_big_metadata50.npy')
    # metadata3 = np.load('D:/generate_metadata/bigmetadata/wdbc/query_time/query_time_5interval/query_time50/40wdbc30_big_metadata50.npy')
    # metadata4 = np.load('D:/generate_metadata/bigmetadata/wdbc/query_time/query_time_5interval/query_time50/80wdbc30_big_metadata50.npy')
    # metadata5 = np.load('D:/generate_metadata/bigmetadata/wdbc/query_time/query_time_5interval/query_time50/10wdbc30_big_metadata50.npy')
    # metadata6 = np.load('D:/generate_metadata/bigmetadata/wdbc/query_time/query_time_5interval/query_time50/4wdbc30_big_metadata50.npy')

    # metadata = np.vstack((metadata1, metadata2, metadata3, metadata4, metadata5, metadata6 ))

    # X = metadata[:, 0:396]
    # y = metadata[:, 396]
    X = metadata[:, 0:418]
    y = metadata[:, 418]
    if task == 'classification':
        y[np.where(y>0)[0]] = 1
        y[np.where(y<=0)[0]] = 0

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    X_tensor = X_tensor.float()
    if task == 'classification':
        print('This is a classification')
        y_tensor = y_tensor.long()
    else:
        print('This is a regression')
        y_tensor = y_tensor.float()
        
    print('X`s dtype is : ', X_tensor.dtype)
    print('y`s dtype is : ', y_tensor.dtype)
    # print('the shape of y is : ', y_tensor.size())
    troch_dataset = Data.TensorDataset(X_tensor, y_tensor)
    dataset_loader = Data.DataLoader(dataset=troch_dataset,
                            batch_size=batch_size,
                            shuffle=dataset_shuffle,
                            num_workers=n_workers)   

    return dataset_loader

class MyLoss(torch.nn.Module):
    ''' Learn the loss prediction module by considering the difference 
        between a pair of loss predictions, which completely make the 
        loss prediction module discard the overall scale changes.
    Parameters:
    ----------
    xi: float
        ξ is a pre-defined positive margin.
    Reference
    ---------
    [1] Learning Loss for Active Learning
        https://arxiv.org/abs/1905.03677?context=cs.CV
    '''
    def __init__(self):
        super(MyLoss, self).__init__()
        # self.xi = xi
        # self.lam = lam

    def forward(self, output1, output2, pair_target, target, xi, lam):
        indicative_f = torch.sign(pair_target)
        indicative_f[indicative_f==0] = -1
        # temp = -indicative_f * (output1 - output2) + self.xi
        temp = -indicative_f * (output1 - output2) + xi
        pair_loss = torch.clamp(temp, min=0.0)
        pair_loss = pair_loss.sum()
        mse_loss = F.mse_loss(torch.cat((output1, output2))[:, 0], target)
        # loss = mse_loss + self.lam * pair_loss
        loss = mse_loss + lam * pair_loss
        if np.isnan(loss.cpu().detach().numpy()):
            print('indicative_f', indicative_f)
            print('-indicative_f',-indicative_f )
            print('output1',output1)
            print('output2',output2)

            print('(output1 - output2)',(output1 - output2))
            print('xi',xi)
            print('temp',temp)
            print('pair loss', pair_loss)
            print('mse loss', mse_loss)
            print('loss', loss)
        return loss

class MyLoss1(torch.nn.Module):
    ''' Learn the loss prediction module by considering the difference 
        between a pair of loss predictions, which completely make the 
        loss prediction module discard the overall scale changes.
    Parameters:
    ----------
    xi: float
        ξ is a pre-defined positive margin.
    Reference
    ---------
    [1] Learning Loss for Active Learning
        https://arxiv.org/abs/1905.03677?context=cs.CV
    '''
    def __init__(self, xi):
        super(MyLoss1, self).__init__()
        self.xi = xi

    def forward(self, output, target):
        indicative_f = torch.sign(target)
        indicative_f[indicative_f==0] = -1
        temp = -indicative_f * output + self.xi
        loss = torch.clamp(temp, 0.0)
        loss = loss.sum()
        return loss
        