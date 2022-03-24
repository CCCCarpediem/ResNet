import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from tqdm import *
from torchsummary import summary
import argparse


def get_data():
  ROOT = '.data'

  transforms_train = torchvision.transforms.Compose([
                             torchvision.transforms.RandomCrop(32, padding=4),
                             torchvision.transforms.RandomHorizontalFlip(),
                             torchvision.transforms.ToTensor()])

  transforms_test = torchvision.transforms.Compose([
                             torchvision.transforms.ToTensor()])

  train_data = torchvision.datasets.CIFAR10(root = ROOT, 
                             train = True, 
                             download = True, 
                             transform = transforms_train)

  test_data = torchvision.datasets.CIFAR10(root = ROOT, 
                             train = False, 
                             download = True, 
                             transform = transforms_test)
  return train_data, test_data

  
########################################################
# set model structure
########################################################
class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.linear = nn.Linear(256, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def project1_model():
    return ResNet(BasicBlock, [8, 4, 8, 2])

########################################################
# define test function
########################################################
def model_test(net):
  net.eval()
  test_loss = 0
  acc_num = 0
  for i, (X_ts, y_ts) in enumerate(testloader):

    X_ts, y_ts = X_ts.to(device), y_ts.to(device)
    y_ts_hat = net(X_ts)
    l_ts = loss(y_ts_hat, y_ts)
    test_loss += l_ts.item()
      
    y_pred = torch.argmax(y_ts_hat,axis=1)
    acc_num += torch.sum(y_pred == y_ts)

  acc = acc_num/len(test_data)
  # return acc
  return np.array(acc.clone().detach().cpu())


if __name__ == '__main__':
  lr = 0.001
  batch_size = 64
  ngpu= 1

  parser = argparse.ArgumentParser()
  parser.add_argument('-epoch', default=500, type=int, help='Number of epochs')
  parser.add_argument('-train', action='store_true', help='Train mode')
  parser.add_argument('-test', action='store_true', help='Test mode')
  parser.add_argument('-summary', action='store_true', help='Show model summary')
  args = parser.parse_args()

  num_epochs = args.epoch
  print("number of epochs = " + str(num_epochs))

  device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
  print(device)

  train_data, test_data = get_data()
  trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
  testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

  if args.train:
    net = project1_model().to(device)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    loss = nn.CrossEntropyLoss().to(device)

    tr_loss = []
    ts_loss = []
    acc_list = []
    acc_max = 0

    for epoch in range(num_epochs):
      net.train()
      train_loss = 0
      for i, (X, y) in enumerate(tqdm(trainloader)):
      # for i, (X, y) in enumerate(trainloader):

        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()

        train_loss += l.item()

      net.eval()
      test_loss = 0
      acc_num = 0
      for i, (X_ts, y_ts) in enumerate(testloader):


        X_ts, y_ts = X_ts.to(device), y_ts.to(device)
        y_ts_hat = net(X_ts)
        l_ts = loss(y_ts_hat, y_ts)
        test_loss += l_ts.item()

        y_pred = torch.argmax(y_ts_hat,axis=1)
        acc_num += torch.sum(y_pred == y_ts)


      train_loss = train_loss*batch_size/len(train_data)
      test_loss = test_loss*batch_size/len(test_data)

      acc = acc_num/len(test_data)

      ts_loss.append(test_loss)
      tr_loss.append(train_loss)
      acc_list.append(acc)


      print('Epoch %d, training loss is %f, testing loss is %f, acc is %f'%(epoch,train_loss,test_loss,acc))
      if acc >= acc_max:
        accuracy = acc.cpu().numpy()
        filename = 'project1_model.pt'
        torch.save(net, filename)
        acc_max = acc;
      # if acc > 0.94:
        # break

    plt.plot(range(epoch+1),tr_loss,'-',linewidth=3,label='Train loss')
    plt.plot(range(epoch+1),ts_loss,'-',linewidth=3,label='Test loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.legend()

    plt.plot(range(epoch+1),np.array(torch.tensor(acc_list, device='cpu')),'-',linewidth=3,label='Acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid(True)
    plt.legend()  
  elif args.test:
    model_path = "project1_model.pt" 
    net=torch.load(model_path)
    best_acc = model_test(net)
    print("The best accuracy is " + str(best_acc))  
  elif args.summary:
    net = project1_model().to(device)
    summary(net,(3,32,32))
  else:
    print("please define a train/test mode")
