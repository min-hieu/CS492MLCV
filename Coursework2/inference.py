import os
from torchvision.datasets import MNIST
from torchvision import transforms
import torch
import numpy as np
from model import Classifier

transform   = transforms.Compose([transforms.ToTensor(), 
                                  transforms.Normalize((0.1307,),(0.3081,))])
test        = MNIST(os.getcwd(), train=False, download=True,
                   transform=transform)

weight_path = './weight.ckpt'
inference   = Classifier.load_from_checkpoint(weight_path, lr_rate=1e-3)

with torch.no_grad():
    x = torch.cat((test[0][0],test[1][0],test[2][0]), 0).unsqueeze(1) # 3 image
    y = [test[0][1],test[1][1],test[2][1]]

    logits = inference(x)
    print('Prediction :',np.argmax(logits.detach().numpy(), axis=1))
    print('Real :', y)
