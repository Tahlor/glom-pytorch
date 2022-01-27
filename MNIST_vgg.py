# CONFIG
# CNN
# LM
import torch
import torchvision
import math
import torch.nn.functional as F
import numpy as np
from models import VGG, V1, VGGLinear
import argparse
from data import loaders
from torch import nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

TESTING = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./configs/stroke_config/baseline.yaml", help='Path to the config file.')
    parser.add_argument('--testing', action="store_true", default=False, help='Run testing version')
    parser.add_argument('--model', type=str, default="VGG", help='VGG, V1, VGGLinear')
    parser.add_argument('--data', type=str, default="Letters", help='Letters, Balanced')

    #parser.add_argument('--name', type=str, default="", help='Optional - special name for this run')
    opts = parser.parse_args()
    return opts

device = 'cuda'

# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main(num_epochs = 100,
         learning_rate = 0.005,
         momentum = 0.5,
         log_interval = 500,
         *args,
         **kwargs):

    args = parse_args()
    train_loader, test_loader = loaders.loader(batch_size_train = 100, batch_size_test = 1000, split=args.data.lower())
    if args.data.lower()=="balanced":
        alphabet_size = 47
    elif args.data.lower() == "letters":
        alphabet_size = 26
    # Train the model
    total_step = len(train_loader)
    curr_lr1 = learning_rate

    MODELS = {"VGG":VGG, "VGGLinear":VGGLinear, "V1":V1}
    model_type = MODELS[args.model]
    if TESTING:
        model1 = V1(alphabet_size).to(device)
    else:
        model1 = model_type().to(device)
    print(model1.__class__)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)

    scheduler = StepLR(optimizer1, step_size=20, gamma=0.5)

    # Train the model
    total_step = len(train_loader)

    best_accuracy1 = 0

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward
            outputs = model1(images)
            loss1 = criterion(outputs, labels)

            # Backward and optimize
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()


            if i == 499:
                print("Ordinary Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss1.item()))

        # Test the model
        model1.eval()

        with torch.no_grad():
            correct1 = 0
            total1 = 0

            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model1(images)
                _, predicted = torch.max(outputs.data, 1)
                total1 += labels.size(0)
                correct1 += (predicted == labels).sum().item()

            if best_accuracy1 >= correct1 / total1:
                curr_lr1 = learning_rate * np.asscalar(pow(np.random.rand(1), 3))
                update_lr(optimizer1, curr_lr1)
                print('Test Accuracy of NN: {} % Best: {} %'.format(100 * correct1 / total1, 100 * best_accuracy1))
            else:
                best_accuracy1 = correct1 / total1
                net_opt1 = model1
                print('Test Accuracy of NN: {} % (improvement)'.format(100 * correct1 / total1))

            model1.train()
            scheduler.step()

if __name__=='__main__':
    main()