# ELEC 475 Lab 1
# Fall 2025
# Alistair Barfoot and Luke Barry

import torch
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt 
from torchvision.datasets import MNIST
from model import autoencoderMLP4Layer

def main():
  # read arguments from command line
  argParser = argparse.ArgumentParser()
  argParser.add_argument('-l', metavar='state', type=str, help='parameter file (.pth)')
  argParser.add_argument('-z', metavar='bottleneck size', type=int, help='int [32]')

  args = argParser.parse_args()

  save_file = None
  if args.l != None:
      save_file = args.l
  bottleneck_size = 8
  if args.z != None:
      bottleneck_size = args.z
  print('bottleneck size = ', bottleneck_size)

  device = 'cpu'
  if torch.cuda.is_available():
      device = 'cuda'
  print('\t\tusing device ', device)

  train_transform = transforms.Compose([
      transforms.ToTensor()
  ])

  train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)

  N_input = 28 * 28   # MNIST image size 
  N_output = N_input
  model = autoencoderMLP4Layer(N_input=N_input, N_bottleneck=bottleneck_size, N_output=N_output) 
  model.load_state_dict(torch.load(save_file))
  model.to(device)
  model.eval()

  idx = 0
  while idx >= 0:
    # Get user input
    idx = input("Enter index > ")
    idx = int(idx)
    n = input("Enter number of iterations > ")
    n = int(n)
    
    # Part 4
    if 0 <= idx <= train_set.data.size()[0]:
      img = train_set.data[idx]
      img = img.type(torch.float32)
      img = (img - torch.min(img)) / torch.max(img)
      img = img.view(1, img.shape[0]*img.shape[1]).type(torch.FloatTensor)
      # Run through model
      with torch.no_grad():
        output = model(img)
      img = img.view(28, 28).type(torch.FloatTensor)
      output = output.view(28, 28).type(torch.FloatTensor)

      f=plt.figure()
      f.add_subplot(1,2,1)
      plt.title('Input')
      plt.xlabel(f"Index {idx}")
      plt.imshow(img, cmap='gray')
      f.add_subplot(1,2,2)
      plt.title('Part 4 Output')
      plt.xlabel(f"Index {idx}")
      plt.imshow(output, cmap='gray')
      plt.show()

    # Part 5
    if 0 <= idx <= train_set.data.size()[0]:
      img = train_set.data[idx]
      img = img.type(torch.float32)
      img = (img - torch.min(img)) / torch.max(img)

      # Add random noise to the image
      noise_factor = 0.25
      noise = torch.rand(img.shape) * noise_factor
      noisy_img = img + noise
      noisy_img = torch.clamp(noisy_img, 0., 1.)
      noisy_img = noisy_img.to(device=device)
      noisy_img = noisy_img.view(1, noisy_img.shape[0]*noisy_img.shape[1]).type(torch.FloatTensor)

      # Run through model
      with torch.no_grad():
        output = model(noisy_img)
      noisy_img = noisy_img.view(28, 28).type(torch.FloatTensor)
      output = output.view(28, 28).type(torch.FloatTensor)

      # Output images
      f=plt.figure()
      f.add_subplot(1,3,1)
      plt.title('Input')
      plt.xlabel(f"Index {idx}")
      plt.imshow(img, cmap='gray')
      f.add_subplot(1,3,2)
      plt.title(f'Noisy Image ({noise_factor})')
      plt.xlabel(f"Index {idx}")
      plt.imshow(noisy_img, cmap='gray')
      f.add_subplot(1,3,3)
      plt.title('Part 5 Output')
      plt.xlabel(f"Index {idx}")
      plt.imshow(output, cmap='gray')
      plt.show()

    # Part 6
    if 0 <= idx <= train_set.data.size()[0]:
      img = train_set.data[idx]
      img = (img - torch.min(img)) / torch.max(img)
      imgs = []
      imgs.append(img.clone())

      # Add noise 'n' times and pass through model
      for i in range(n):
        if i > 0:
          img = imgs[i-1]
        img = img.type(torch.float32)
        img = (img - torch.min(img)) / torch.max(img)

        # Add random noise to the image
        noise_factor = 0.25
        noise = torch.rand(img.shape) * noise_factor
        img = img + noise
        img = torch.clamp(img, 0., 1.)
        img = img.to(device=device)
        img = img.view(1, img.shape[0]*img.shape[1]).type(torch.FloatTensor)

        # Run through model
        with torch.no_grad():
          output = model(img.view(1, 28*28).to(device=device))
        output = output.view(28, 28).type(torch.FloatTensor)

        img = img.view(28, 28).type(torch.FloatTensor)
        # Add image to list
        imgs.append(output)
      f = plt.figure()
      # Plot each image
      for i in range(n):
        if i == 0:
          f.add_subplot(1,n,1)
          plt.title('Input')
        else:
          if i == (n/2) or i == (n/2)+0.5:
            plt.xlabel(f"Index {idx}, Noise Factor {noise_factor}")
          f.add_subplot(1,n,i+1)
          plt.title(f'{i}')
        plt.imshow(imgs[i], cmap='gray')
      plt.show()


if __name__ == "__main__":
  main()
