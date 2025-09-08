
#########################################################################################################
#
#   ELEC 475 - Lab 1, Step 1
#   Fall 2025
#

import torch
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import MNIST
from model import autoencoderMLP4Layer

def main():

    print('running main ...')

    #   read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', metavar='state', type=str, help='parameter file (.pth)')
    argParser.add_argument('-z', metavar='bottleneck size', type=int, help='int [32]')

    args = argParser.parse_args()

    save_file = None
    if args.s != None:
        save_file = args.s
    bottleneck_size = 0
    if args.z != None:
        bottleneck_size = args.z

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('\t\tusing device ', device)

    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_transform = train_transform

    train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)
    test_set = MNIST('./data/mnist', train=False, download=True, transform=test_transform)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    N_input = 28 * 28   # MNIST image size
    N_output = N_input
    model = autoencoderMLP4Layer(N_input=N_input, N_bottleneck=bottleneck_size, N_output=N_output)
    model.load_state_dict(torch.load(save_file))
    model.to(device)
    model.eval()

    idx = 0
    while idx >= 0:
        idx = input("Enter index > ")
        idx = int(idx)
        n = input("enter number of iterations > ")
        n = int(n)
        if 0 <= idx <= train_set.data.size()[0]:
            img = train_set.data[idx]
            img = (img - torch.min(img)) / torch.max(img)
            imgs = []
            imgs.append(img.clone())
            for i in range(n):
                if i>0:
                    img = imgs[i-1]
                img = img.type(torch.float32)                
                img = (img - torch.min(img)) / torch.max(img)

                # Add random noise to the image
                noise_factor = 0.1
                noise = torch.rand(img.shape) * noise_factor
                img = img + noise
                img = torch.clamp(img, 0., 1.)
                img = img.to(device=device)
                img = img.view(1, img.shape[0]*img.shape[1]).type(torch.FloatTensor)

                with torch.no_grad():
                    output = model(img.view(1, 28*28).to(device=device))
                output = output.view(28, 28).type(torch.FloatTensor)

                img = img.view(28, 28).type(torch.FloatTensor)
                imgs.append(output)
            
            f = plt.figure()
            for i in range(n):
                f.add_subplot(1,n,i+1)
                plt.imshow(imgs[i], cmap='gray')
            plt.show()






###################################################################

if __name__ == '__main__':
    main()



