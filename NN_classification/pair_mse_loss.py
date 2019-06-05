from __future__ import print_function
import os
import argparse
import numpy as np 

import torch
import torch.utils.data as Data
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import r2_score, accuracy_score

from MyNet import Net, MetaDataSet, MyLoss, Net_3hiddens

      
def train(args, net, device, train_loader, optimizer, scheduler, epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    # 
    if epoch < args.epochs * 0.3:
        lam = 0
        xi = 0.0005
    elif epoch < args.epochs * 0.6:
        lam = 0.5
        xi = 0.0005
    else:
        lam = 1
        xi = 0.0001       
    # running_loss = 0.0
    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
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
        
        # print('optimizer lr is: ', optimizer.state_dict()['param_groups'][0]['lr'])
        # # print statistics
        # running_loss += loss.item()
        # # print every 1000 mini-batches
        # if batch_idx % 1000 == 999:    
        #     print('[%d, %5d] , progress %.4f% %.4f, mean loss: %.10f' %
        #         (epoch + 1, batch_idx + 1, batch_idx/len(trainloader), running_loss / 1000))
        #     running_loss = 0.0

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(batch_x), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
    scheduler.step()

def test(args, net, device, test_loader):
    global best_acc
    global best_r2
    net.eval()
    r2scorelist = []
    acclist = []
    # with torch.no_grad():
    #     for step, (batch_x, batch_y) in enumerate(testloader):
    #         batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    #         # min_b = int(BATCH_SIZE/2)
    #         min_b = int(batch_y.size()[0]/2)
    #         pari_y = batch_y[0:min_b] - batch_y[min_b:]
    #         pari_y[np.where(pari_y>0)[0]] = 1
    #         pari_y[np.where(pari_y<=0)[0]] = 1
    #         output = net(batch_x)
    #         pred = output.detach().numpy()
    #         y = batch_y.detach().numpy()
    #         r2score = r2_score(pred, y)
    #         pair_pred = pred[0:min_b] - pred[min_b:]
    #         pair_pred[np.where(pair_pred>0)[0]] = 1
    #         pair_pred[np.where(pair_pred<=0)[0]] = 1
    #         acc = accuracy_score(pair_pred, pari_y)
    #         r2scorelist.append(r2score)
    #         acclist.append(acc)

    with torch.no_grad():
        for batch_x, batch_y in testloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
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
    if mean_acc > best_acc or mean_r2 > best_r2:
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

def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num-worker', type=int, default=8, metavar='N',
                        help='number of worker to load dataset (default: 8)')    
    parser.add_argument('--shuffle', action='store_false', default=False,
                        help='whether to shuffle the dataset (default: False)')
    parser.add_argument('--metadata-dir', type=str, default='E:/metadata数据集/new_bigmetadata/australian/query_time/',
                        help='the dir of the metadata')    
    parser.add_argument('--save-dir', type=str, default='./checkpoints/',
                        help='the dir of the metadata')  
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=0.0005, metavar='M',
                        help='Adam weight-decay (default: 0.0005)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')
    
    args = parser.parse_args()
    return args
    

if __name__ == "__main__":

    args = get_args()
    torch.manual_seed(args.seed)

    best_acc = 0  # best test accuracy
    best_r2 = 0
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # net = Net(n_feature=396, first_n_hidden=100, second_n_hidden=20, n_output=2)
    # net_bn = Net(n_feature=418, first_n_hidden=200, second_n_hidden=100, n_output=1, batch_normalization=True)
    # net_no_bn = Net(n_feature=418, first_n_hidden=200, second_n_hidden=100, n_output=1, batch_normalization=False)
    net = Net_3hiddens(n_feature=418, first_n_hidden=200, second_n_hidden=100, third_n_hidden=50, n_output=1, batch_normalization=True)

    print('Preparing the dataset....')
    # trainloader = MetaDataSet(metadata_dir, BATCH_SIZE, num_WORKERS, DATASET_SHUFFLE, TASK)
    # testloader = MetaDataSet(metadata_dir, BATCH_SIZE, num_WORKERS, DATASET_SHUFFLE, TASK)

    metadata1 = np.load('D:/generate_metadata/new_bigmetadata/australian/query_time/2australian30_big_metadata30.npy')
    metadata2 = np.load('D:/generate_metadata/new_bigmetadata/australian/query_time/4australian30_big_metadata30.npy')
    # metadata3 = np.load('D:/generate_metadata/new_bigmetadata/australian/query_time/10australian30_big_metadata30.npy')
    # metadata4 = np.load('D:/generate_metadata/new_bigmetadata/australian/query_time/20australian30_big_metadata30.npy')
    # metadata5 = np.load('D:/generate_metadata/new_bigmetadata/australian/query_time/30australian30_big_metadata30.npy')
    # metadata6 = np.load('D:/generate_metadata/new_bigmetadata/australian/query_time/48australian30_big_metadata30.npy')

    # metadata = np.vstack((metadata1, metadata2, metadata3, metadata4, metadata5, metadata6))
    metadata = np.vstack((metadata1, metadata2))

    X = metadata[:, 0:418]
    y = metadata[:, 418]

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    X_tensor = X_tensor.float()
    y_tensor = y_tensor.float()

    troch_dataset = Data.TensorDataset(X_tensor, y_tensor)
    trainloader = Data.DataLoader(dataset=troch_dataset,
                            batch_size=args.batch_size,
                            shuffle=args.shuffle,
                            num_workers=args.num_worker)

    test_X = metadata1[:, 0:418]
    test_y = metadata1[:, 418]

    test_X_tensor = torch.from_numpy(X)
    test_y_tensor = torch.from_numpy(y)
    test_X_tensor = X_tensor.float()
    test_y_tensor = y_tensor.float()

    test_troch_dataset = Data.TensorDataset(X_tensor, y_tensor)
    testloader = Data.DataLoader(dataset=test_troch_dataset,
                            batch_size=args.test_batch_size,
                            shuffle=args.shuffle,
                            num_workers=args.num_worker)

    print("Dataset is ready!")

    # optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    criterion = MyLoss()
    print(net)
    print(optimizer)
    print(criterion)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('The current device is : ', device)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)

    for epoch in range(1, args.epochs + 1):
        train(args, net, device, trainloader, optimizer, scheduler, epoch)
        test(args, net, device, testloader)

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    torch.save(net.state_dict(),args.save_dir+"pari_mse_3nn.pt")

