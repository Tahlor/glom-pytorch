import torch
import torchvision
import math
import torch.nn.functional as F
import numpy as np
import argparse
from data import loaders
import torch
from glom_pytorch.glom_pytorch import Glom
from glom_pytorch.utils import GeometricMean, Mean, CNN
from glom_pytorch import stats
import torch.nn.functional as F
from torch import nn
from einops.layers.torch import Rearrange

## Sine embeddings
## Top level NN to create letter
## Reconstruction loss
## (extend to multiple letters)
    ## add random letter noise so the border between letters is noisey
## (transformer language model on top)
## Use CNN at the bottom -- ablation to test if CNN is doing all the learning
## Residual connections?
## Try using reversible network
# WHAT IS GOING ON
    # ARE THERE NEURAL NETWORKS BETWEEN LAYERS? Yes, 1D Conv

## Ablation
    # Compare after different number of epochs
    # NEED TO BEAT just CNN + Classifier
    # Show what different levels are learning

#### HOW DOES GLOM AGGREGATE HIGHER IN A HIERARCHY??
#### NEED TO AVERAGE TOP LEVEL TO FEED INTO NN
## How does backprop work? Is it backprop through time? Or just backprop 1 iteration at a time?
## Can we use the convolutional pose machine? CPM takes all initial segmentations, then re-segments using the same weights
# INVERSE TEMPERATURE PARAMETER

num_classes = 27
GLOM_DIM = 256 # 512
CHANNELS = 1 # 3
IMG_DIM = 28 # 224
P1 = P2 = 4 # 14
LEVELS = 4
USE_CNN = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./configs/stroke_config/baseline.yaml", help='Path to the config file.')
    parser.add_argument('--testing', action="store_true", default=False, help='Run testing version')
    #parser.add_argument('--name', type=str, default="", help='Optional - special name for this run')
    opts = parser.parse_args()
    return opts

device = 'cuda'

# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

patches_to_images = nn.Sequential(
    nn.Linear(GLOM_DIM, P1 * P2 * CHANNELS),
    Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=P1, p2=P2, h=(IMG_DIM // P1))
)

def denoise():
    top_level = all_levels[7, :, :, -1]  # get the top level embeddings after iteration 6
    recon_img = patches_to_images(top_level)

    # do self-supervised learning by denoising
    loss = F.mse_loss(img, recon_img)
    loss.backward()

def main(num_epochs = 200,
         learning_rate = 0.005,
         momentum = 0.5,
         log_interval = 500,
         *args,
         **kwargs):

    train_loader, test_loader = loaders.loader(batch_size_train = 100, batch_size_test = 1000)

    # Train the model
    total_step = len(train_loader)
    curr_lr1 = learning_rate

    my_cnn = None
    if USE_CNN:
        my_cnn = CNN().to(device)
        CHANNELS = 32

    model1 = Glom(
        dim=GLOM_DIM,
        levels=LEVELS,
        image_size=IMG_DIM,
        patch_size=P1,
        channels=CHANNELS,
    ).to(device)
    model = nn.Sequential(my_cnn, model1)


    # SOME KIND OF CONV THING HERE
    classifier = nn.Sequential(
        #Rearrange('b p dim -> b (p dim)'),
        #nn.ReLU(inplace=True),
        Mean(dim=1),
        nn.Dropout(p=0.5),
        nn.Linear(GLOM_DIM, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(512, num_classes),
        # Rearrange('(b p) dim -> b p dim')
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    best_accuracy1 = 0

    EPOCH_LENGTH = 100
    COUNTER = stats.Counter(instances_per_epoch=EPOCH_LENGTH)
    losses = stats.AutoStat(COUNTER, name="Loss1", x_plot="epoch_decimal")

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward
            outputs = model(images) # batch - patches^2 - levels - dimension
            letter_logits = classifier(outputs[:,:,-1]) # BATCH, PATCHES, DIM
            loss1 = criterion(letter_logits, labels)

            # Backward and optimize
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()
            losses.accumulate(loss1.item(), weight=images.shape[0])
            print(i, loss1.item())
            if i == EPOCH_LENGTH:
                losses.reset_accumulator()
                print("Ordinary Epoch [{}/{}], Step [{}/{}] Loss: {:.4f} {}"
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss1.item()), str(losses))
                break

        # Test the model
        model.eval()

        with torch.no_grad():
            correct1 = 0
            total1 = 0

            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total1 += labels.size(0)
                correct1 += (predicted == labels).sum().item()

            if best_accuracy1 >= correct1 / total1:
                curr_lr1 = learning_rate * np.asscalar(pow(np.random.rand(1), 3))
                update_lr(optimizer1, curr_lr1)
                print('Test Accuracy of NN: {} % Best: {} %'.format(100 * correct1 / total1, 100 * best_accuracy1))
            else:
                best_accuracy1 = correct1 / total1
                net_opt1 = model
                print('Test Accuracy of NN: {} % (improvement)'.format(100 * correct1 / total1))

            model.train()

if __name__=='__main__':
    main()