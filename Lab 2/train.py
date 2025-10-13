import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader
from torchsummary import summary
from model import snoutNet
from dataset import CustomDataset
import time

# Paths
save_file = 'snoutnet_weights.pth'
train_ann = "train_noses.txt"
test_ann = "test_noses.txt"
img_dir = "images"

n_epochs = 20
batch_size = 32
plot_file = 'snoutnet_plot.png'

def train(n_epochs, optimizer, model, loss_fn, train_loader, test_loader, scheduler, device, save_file=None, plot_file=None):
    model.train()
    losses_train = []
    losses_test = []
    print(f'Starting training for {n_epochs} epochs...')
    time_start = time.time()
    for epoch in range(1, n_epochs+1):
        loss_train = 0.0
        for data, labels in train_loader:
            imgs = data.to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        scheduler.step(loss_train)
        losses_train += [loss_train/len(train_loader)]

        for data, labels in test_loader:
            imgs = data.to(device=device)
            labels = labels.to(device=device)
            with torch.no_grad():
                outputs = model(imgs)
                loss = loss_fn(outputs, labels)
                loss_test += loss.item()
        losses_test += [loss_test/len(test_loader)]

        if save_file != None:
            torch.save(model.state_dict(), save_file)

        print(f"Epoch {epoch}/{n_epochs} | Train Loss: {losses_train[-1]:.4f} | Test Loss: {losses_test[-1]:.4f}")
        elapsed = time.time() - time_start
        avg_per_epoch = elapsed / (epoch + 1)
        remaining = avg_per_epoch * (n_epochs - (epoch + 1))

        def sec_to_hms(s):
          s = int(max(0, s))
          h = s // 3600
          m = (s % 3600) // 60
          sec = s % 60
          return f"{h:02d}:{m:02d}:{sec:02d}"

        print(f"Time Elapsed: {sec_to_hms(elapsed)} | Remaining: {sec_to_hms(remaining)}")

        if plot_file != None:
            plt.figure(2, figsize=(12, 7))
            plt.clf()
            plt.plot(losses_train, label='train')
            plt.plot(losses_test, label='test')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid()
            plt.savefig(plot_file)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def main():
    global bottleneck_size, save_file, n_epochs, batch_size

    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', metavar='state', type=str, help='parameter file (.pth)')
    argParser.add_argument('-z', metavar='bottleneck size', type=int, help='int [32]')
    argParser.add_argument('-e', metavar='epochs', type=int, help='# of epochs [30]')
    argParser.add_argument('-b', metavar='batch size', type=int, help='batch size [32]')
    argParser.add_argument('-p', metavar='plot', type=str, help='output loss plot file (.png)')

    args = argParser.parse_args()

    save_file = 'snoutnet_weights.pth'
    bottleneck_size = 32
    n_epochs = 30
    batch_size = 32
    plot_file = 'snoutnet_plot.png'


    if args.s != None:
        save_file = args.s
    if args.z != None:
        bottleneck_size = args.z
    if args.e != None:
        n_epochs = args.e
    if args.b != None:
        batch_size = args.b
    if args.p != None:
        plot_file = args.p

    print('\t\tbottleneck size = ', bottleneck_size)
    print('\t\tn epochs = ', n_epochs)
    print('\t\tbatch size = ', batch_size)
    print('\t\tsave file = ', save_file)
    print('\t\tplot file = ', plot_file)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        print("Using GPU for training")
    
    model = snoutNet()
    model = model.to(device)
    model.apply(init_weights)
    summary(model, (3, 96, 96))

    train_set = CustomDataset(train_ann, img_dir)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = CustomDataset(test_ann, img_dir)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    loss_fn = nn.MSELoss(size_average=None, reduce=None, reduction='mean')

    train(n_epochs, optimizer, model, loss_fn, train_loader, test_loader, scheduler, device, save_file, plot_file)

if __name__ == '__main__':
    main()