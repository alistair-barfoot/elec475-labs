import random
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# import torchvision.transforms as transforms
from torchvision.transforms import v2 as transforms
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
    losses_train = []
    losses_test = []
    print(f'Starting training for {n_epochs} epochs...')
    time_start = time.time()
    min_test_loss = float('inf')
    min_epoch = -1
    
    for epoch in range(1, n_epochs+1):
        # Training phase
        model.train()
        loss_train = 0.0
        for data, labels in train_loader:
            if random.random() < 0.001:
                img = cv2.cvtColor(np.ascontiguousarray((data[0].numpy().transpose(1, 2, 0) * 255).astype('uint8')), cv2.COLOR_RGB2BGR)
                nose = (int(labels[0][0].item()), int(labels[0][1].item()))
                if 0 <= nose[0] < img.shape[1] and 0 <= nose[1] < img.shape[0]:
                    cv2.circle(img, nose, 5, (0, 255, 0), -1)
                cv2.imwrite('example_debug.png', img)
            imgs = data.to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        
        # Testing phase
        model.eval()
        loss_test = 0.0
        with torch.no_grad():
            for data, labels in test_loader:
                imgs = data.to(device=device)
                labels = labels.to(device=device)
                outputs = model(imgs)
                loss = loss_fn(outputs, labels)
                loss_test += loss.item()

        # Early stopping check
        if loss_test/len(test_loader) < min_test_loss:
            min_test_loss = loss_test/len(test_loader)
            min_epoch = epoch
        elif epoch - min_epoch >= 5:
            print(f"Early stopping at epoch {epoch} with minimum test loss {min_test_loss:.4f} at epoch {min_epoch}")
            break
        
        # Store losses
        losses_train.append(loss_train/len(train_loader))
        losses_test.append(loss_test/len(test_loader))
        
        # Step scheduler with validation loss
        scheduler.step(losses_test[-1])

        if save_file != None:
            torch.save(model.state_dict(), save_file)

        print(f"Epoch {epoch}/{n_epochs} | Train Loss: {losses_train[-1]:.4f} | Test Loss: {losses_test[-1]:.4f}")
        elapsed = time.time() - time_start
        avg_per_epoch = elapsed / epoch
        remaining = avg_per_epoch * (n_epochs - epoch)

        def sec_to_hms(s):
          s = int(max(0, s))
          h = s // 3600
          m = (s % 3600) // 60
          sec = s % 60
          return f"{h:02d}:{m:02d}:{sec:02d}"

        print(f"Elapsed: {sec_to_hms(elapsed)} | Left: {sec_to_hms(remaining)} | Per Epoch: {sec_to_hms(avg_per_epoch)}")
        print("-"*75)

        if plot_file != None:
            plt.figure(figsize=(12, 7))
            plt.clf()
            plt.plot(losses_train, label='train')
            plt.plot(losses_test, label='test')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid()
            plt.savefig(plot_file)
            plt.close()

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)

def main():
    global save_file, n_epochs, batch_size

    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', metavar='state', type=str, help='parameter file (.pth)')
    argParser.add_argument('-e', metavar='epochs', type=int, help='# of epochs [30]')
    argParser.add_argument('-b', metavar='batch size', type=int, help='batch size [32]')
    argParser.add_argument('-p', metavar='plot', type=str, help='output loss plot file (.png)')

    argParser.add_argument('-r', '--reflection', action='store_true', help='use reflection augmentation [False]')
    argParser.add_argument('-f', '--flip', action='store_true', help='use random flip augmentation [False]')
    argParser.add_argument('-n', '--noise', action='store_true', help='use random noise augmentation [False]')

    args = argParser.parse_args()

    save_file = 'snoutnet_weights.pth'
    n_epochs = 30
    batch_size = 32
    plot_file = 'snoutnet_plot.png'


    if args.s != None:
        save_file = args.s
    if args.e != None:
        n_epochs = args.e
    if args.b != None:
        batch_size = args.b
    if args.p != None:
        plot_file = args.p

    print('\t\tn epochs = ', n_epochs)
    print('\t\tbatch size = ', batch_size)
    print('\t\tsave file = ', save_file)
    print('\t\tplot file = ', plot_file)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        print("Using GPU for training")
    else :
        print("Using CPU for training")
    
    model = snoutNet()
    model = model.to(device)
    model.apply(init_weights)
    summary(model, (3, 227, 227))

    transform = transforms.Compose([
        transforms.Resize((227, 227)),
    ])

    if args.flip:
        transform.transforms.append(transforms.RandomHorizontalFlip(p=0.1))
        transform.transforms.append(transforms.RandomVerticalFlip(p=0.1))
    if args.reflection:
        transform.transforms.append(transforms.RandomRotation(degrees=(-10, 10)))
    if args.noise:
        transform.transforms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
    transform.transforms.append(transforms.ToImage())
    transform.transforms.append(transforms.ToDtype(torch.float32, scale=True))

    print("Using transforms:")
    for t in transform.transforms:
        print(f" - {t}")

    train_set = CustomDataset(train_ann, img_dir, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])

    test_set = CustomDataset(test_ann, img_dir, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=2e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    loss_fn = nn.MSELoss(size_average=None, reduce=None, reduction='mean')

    train(
        n_epochs=n_epochs,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        test_loader=test_loader,
        scheduler=scheduler,
        device=device,
        save_file=save_file,
        plot_file=plot_file
    )

if __name__ == '__main__':
    main()