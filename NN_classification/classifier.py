import numpy as np 
import torch
import torch.utils.data as Data
import torch.optim as optim

from MyNet import Net, MetaDataSet

# hyper parameters
BATCH_SIZE = 10
EPOCH_SIZE = 2
LR = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-6
num_WORKERS = 1

save_path = './'
metadata_dir = 'E:/metadata数据集/new_bigmetadata/australian/query_time/'

# net = Net(n_feature=396, first_n_hidden=100, second_n_hidden=20, n_output=2)
net_bn = Net(n_feature=418, first_n_hidden=200, second_n_hidden=100, n_output=2, batch_normalization=True)
net_no_bn = Net(n_feature=418, first_n_hidden=200, second_n_hidden=100, n_output=2, batch_normalization=False)

nets = [net_bn, net_no_bn]
optimizers = [optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY) for net in nets]
criterion = torch.nn.CrossEntropyLoss()

print('Preparing the dataset....')
loader = MetaDataSet(metadata_dir, BATCH_SIZE, num_WORKERS)
print("Dataset is ready!")

'''
# tarining the NN
for epoch in range(EPOCH_SIZE):
    running_loss = 0.0
    for step, (batch_x, batch_y) in enumerate(loader):
        for net, optimizer in zip(nets, optimizers):
            optimizer.zero_grad()
            outputs = net(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # print every 2000 mini-batches
            if step % 1000 == 999:    
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, step + 1, running_loss / 2000))
                running_loss = 0.0

# save the parameters in NN
torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, './net_parameters.pth')
'''