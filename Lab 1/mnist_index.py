from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

train_transform = transforms.Compose([transforms.ToTensor()])
train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)

try:
  while True:
    print("Enter a value between 0 and 59999. Ctrl+C to quit.")
    idx = int(input())
    if(idx >= 0 and idx <= 59999):
      plt.imshow(train_set.data[idx], cmap='gray')
      plt.show()
    else:
      print("Error. Not within bounds")
except KeyboardInterrupt:
  print("Exiting program now")

