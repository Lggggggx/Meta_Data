import os
import numpy as np 
import torch
import torch.utils.data as Data
import torch.optim as optim

from sklearn.metrics import r2_score, accuracy_score

from MyNet import Net, MetaDataSet, MyLoss, Net_3hiddens



# tarining the NN
# for epoch in range(EPOCH_SIZE):
#     net.train()
#     # 
#     if epoch < EPOCH_SIZE * 0.3:
#         lam = 0
#         xi = 0.0005
#     elif epoch < EPOCH_SIZE * 0.6:
#         lam = 0.5
#         xi = 0.0005
#     else:
#         lam = 1
#         xi = 0.0001       
#     running_loss = 0.0
#     for step, (batch_x, batch_y) in enumerate(trainloader):
#         batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#         min_b = int(BATCH_SIZE/2)
#         pari_y = batch_y[0:min_b] - batch_y[min_b:]
#         optimizer.zero_grad()
#         output1 = net(batch_x[0:min_b,:])
#         output2 = net(batch_x[min_b:,:])
#         loss = criterion(output1, output2, pari_y, batch_y, xi, lam)
#         loss.backward()
#         optimizer.step()   
  
#         # print statistics
#         running_loss += loss.item()
#         # print every 1000 mini-batches
#         if step % 1000 == 999:    
#             print('[%d, %5d] , %.4f, mean loss: %.10f' %
#                 (epoch + 1, step + 1, step/len(trainloader), running_loss / 1000))
#             running_loss = 0.0
        
def train(epoch):
    print('\nEpoch: %d' % (epoch+1))
    net.train()
    # 
    if epoch < EPOCH_SIZE * 0.3:
        lam = 0
        xi = 0.0005
    elif epoch < EPOCH_SIZE * 0.6:
        lam = 0.5
        xi = 0.0005
    else:
        lam = 1
        xi = 0.0001       
    running_loss = 0.0
    for step, (batch_x, batch_y) in enumerate(trainloader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        # min_b = int(BATCH_SIZE/2)
        min_b = int(batch_y.size()[0]/2)
        pari_y = batch_y[0:min_b] - batch_y[min_b:]
        optimizer.zero_grad()
        output1 = net(batch_x[0:min_b,:])
        output2 = net(batch_x[min_b:,:])
        loss = criterion(output1, output2, pari_y, batch_y, xi, lam)
        loss.backward()
        optimizer.step()   
  
        # print statistics
        running_loss += loss.item()
        # print every 1000 mini-batches
        if step % 1000 == 999:    
            print('[%d, %5d] , progress %.4f% %.4f, mean loss: %.10f' %
                (epoch + 1, step + 1, step/len(trainloader), running_loss / 1000))
            running_loss = 0.0

def test(epoch):
    global best_acc
    net.eval()
    r2scorelist = []
    acclist = []
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(testloader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # min_b = int(BATCH_SIZE/2)
            min_b = int(batch_y.size()[0]/2)
            pari_y = batch_y[0:min_b] - batch_y[min_b:]
            pari_y[np.where(pari_y>0)[0]] = 1
            pari_y[np.where(pari_y<=0)[0]] = 1
            output = net(batch_x)
            pred = output.detach().numpy()
            y = batch_y.detach().numpy()
            r2score = r2_score(pred, y)
            pair_pred = pred[0:min_b] - pred[min_b:]
            pair_pred[np.where(pair_pred>0)[0]] = 1
            pair_pred[np.where(pair_pred<=0)[0]] = 1
            acc = accuracy_score(pair_pred, pari_y)
            r2scorelist.append(r2score)
            acclist.append(acc)

    mean_acc = np.mean(acclist)
    mean_r2 = np.mean(r2scorelist)
    # Save checkpoint.
    if mean_acc > best_acc:
        print('Saving..')        
        print('Current mean_rank_acc: %.10f'%mean_acc)
        print('Current mean r2 score: %.10f'%mean_r2)
        state = {
            'net': net.state_dict(),
            'mean_rank_acc': mean_acc,
            'mean_r2_score':mean_r2,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = mean_acc

for epoch in range(EPOCH_SIZE):
    train(epoch)
    test(epoch)


if __name__ == "__main__":

    # hyper parameters
    BATCH_SIZE = 128
    EPOCH_SIZE = 2

    LR = 1e-2
    MOMENTUM = 0.9
    WEIGHT_DECAY =  0.0005

    DATASET_SHUFFLE = False
    num_WORKERS = 3

    TASK = 'regression'

    LAMBDA = 1
    XI = 0.0005


    save_path = './'
    metadata_dir = 'E:/metadata数据集/new_bigmetadata/australian/query_time/'

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # net = Net(n_feature=396, first_n_hidden=100, second_n_hidden=20, n_output=2)
    # net_bn = Net(n_feature=418, first_n_hidden=200, second_n_hidden=100, n_output=1, batch_normalization=True)
    # net_no_bn = Net(n_feature=418, first_n_hidden=200, second_n_hidden=100, n_output=1, batch_normalization=False)
    net = Net_3hiddens(n_feature=418, first_n_hidden=200, second_n_hidden=100, third_n_hidden=50, n_output=1, batch_normalization=True)

    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    criterion = MyLoss(XI, LAMBDA)
    print(net)
    print(optimizer)
    print(criterion)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('The current device is : ', device)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)

    print('Preparing the dataset....')
    # trainloader = MetaDataSet(metadata_dir, BATCH_SIZE, num_WORKERS, DATASET_SHUFFLE, TASK)
    # testloader = MetaDataSet(metadata_dir, BATCH_SIZE, num_WORKERS, DATASET_SHUFFLE, TASK)

    metadata1 = np.load('D:/generate_metadata/new_bigmetadata/australian/query_time/2australian30_big_metadata30.npy')
    metadata2 = np.load('D:/generate_metadata/new_bigmetadata/australian/query_time/4australian30_big_metadata30.npy')
    metadata3 = np.load('D:/generate_metadata/new_bigmetadata/australian/query_time/10australian30_big_metadata30.npy')
    metadata4 = np.load('D:/generate_metadata/new_bigmetadata/australian/query_time/20australian30_big_metadata30.npy')
    metadata5 = np.load('D:/generate_metadata/new_bigmetadata/australian/query_time/30australian30_big_metadata30.npy')
    metadata6 = np.load('D:/generate_metadata/new_bigmetadata/australian/query_time/48australian30_big_metadata30.npy')

    metadata = np.vstack((metadata1, metadata2, metadata3, metadata4, metadata5, metadata6))

    X = metadata[:, 0:396]
    y = metadata[:, 396]

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    X_tensor = X_tensor.float()
    y_tensor = y_tensor.float()

    troch_dataset = Data.TensorDataset(X_tensor, y_tensor)
    trainloader = Data.DataLoader(dataset=troch_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=DATASET_SHUFFLE,
                            num_workers=num_WORKERS)

    test_X = metadata1[:, 0:396]
    test_y = metadata1[:, 396]

    test_X_tensor = torch.from_numpy(X)
    test_y_tensor = torch.from_numpy(y)
    test_X_tensor = X_tensor.float()
    test_y_tensor = y_tensor.float()

    test_troch_dataset = Data.TensorDataset(X_tensor, y_tensor)
    testloader = Data.DataLoader(dataset=test_troch_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=DATASET_SHUFFLE,
                            num_workers=num_WORKERS)

    print("Dataset is ready!")