import itertools
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
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

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


## CONSIDER USING RESIDUAL CONNECTIONS
# 1 patch, 2 levels, batch=200, no cnn => 88.6% accuracy after 13 epochs
# 4 patch, 2 levels, batch=200, no cnn => 87.9% accuracy after 9 epochs
# 4 patch, 3 levels, batch=200, no cnn => 70.6% accuracy after 4 epochs
# 4 patch, 3 levels, batch=200, no cnn => 90.7% accuracy after 45 epochs, no improvement after
# 4 patch, 3 levels, batch=200, CNN-32 => 90.7% accuracy after 45 epochs, no improvement after
### CLASSIFIER WAS NOT BEING TRAINED, FIXED, CNN works now
# 4 patch, 4 levels, batch=200, no cnn => 90% accuracy after 50 epochs, no improvement after

### TO DO:
# Save the network
# Load from saved network
# backprop from each step after it has propagated up; pose machine style;

### ABLATIONS
# No feedback network
# What is going on inside?
# Just the forward prop network with no pooled attention (does the pooled-attention help?)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./configs/stroke_config/baseline.yaml", help='Path to the config file.')
    parser.add_argument('--testing', action="store_true", default=False, help='Run testing version')
    #parser.add_argument('--name', type=str, default="", help='Optional - special name for this run')

    parser.add_argument('--num_classes', type=int, default=27,  help='Number of classes')
    parser.add_argument('--glom_dim',    type=int, default=256, help='GLOM dimension')
    parser.add_argument('--channels',    type=int, default=1,   help='Channels')
    parser.add_argument('--patch_dim',   type=int, default=14,  help='Pixel width of patch')
    parser.add_argument('--use_cnn', action="store_true", default=False, help='Add CNN to first layer')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=400, help='Learning rate')
    parser.add_argument('--levels', type=int, default=3, help='How many GLOM levels?')
    parser.add_argument('--iterations', type=int, default=2, help='How many iterations through hierarchy (multiplied by levels)')
    parser.add_argument('--top_down_network', type=bool, default=True, help='Activate top down network')
    parser.add_argument('--attention_radius', type=int, default=True, help='Patch neighborhood')
    parser.add_argument('--advanced_classifier', type=bool, default=True, help='Advanced classifier')

    opts = parser.parse_args()
    return opts

opts = parse_args()
def parse_to_global():
    global num_classes, GLOM_DIM, CHANNELS, IMG_DIM, P1, P2, LEVELS, USE_CNN, BATCH_SIZE, LEARNING_RATE, RADIUS, TOP_DOWN, ITERATIONS, ADVANCED_CLASSIFIER
    num_classes = opts.num_classes
    GLOM_DIM = opts.glom_dim # 512
    CHANNELS = opts.channels # 3
    IMG_DIM = 28 # 224
    P1 = P2 = opts.patch_dim # 14
    LEVELS = opts.levels
    USE_CNN = opts.use_cnn
    BATCH_SIZE = opts.batch_size
    LEARNING_RATE = opts.learning_rate

    RADIUS = opts.attention_radius
    TOP_DOWN = opts.top_down_network
    ITERATIONS = opts.iterations
    ADVANCED_CLASSIFIER = opts.advanced_classifier
parse_to_global()

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
         learning_rate = LEARNING_RATE,
         momentum = 0.5,
         log_interval = 500,
         *args,
         **kwargs):

    train_loader, test_loader = loaders.loader(batch_size_train = BATCH_SIZE, batch_size_test = BATCH_SIZE*10)

    # Train the model
    total_step = len(train_loader)
    curr_lr1 = learning_rate

    my_cnn = torch.nn.Identity()
    if USE_CNN:
        global CHANNELS
        CHANNELS = 8
        my_cnn = CNN(f1=CHANNELS, f2=CHANNELS).to(device)


    model1 = Glom(
        dim=GLOM_DIM,
        levels=LEVELS,
        image_size=IMG_DIM,
        patch_size=P1,
        channels=CHANNELS,
        local_consensus_radius=RADIUS,
        top_down_network=TOP_DOWN,
    ).to(device)
    print(f"NUM PATCHES: {IMG_DIM/P1}")

    # model = model1

    # SOME KIND OF CONV THING HERE
    if ADVANCED_CLASSIFIER:
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
    else:
        classifier = nn.Sequential(
            Mean(dim=1),
            nn.Linear(GLOM_DIM, num_classes),
        ).to(device)

    model = nn.Sequential(my_cnn, model1)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    all_params = list(itertools.chain(model.parameters(), classifier.parameters()))
    optimizer1 = torch.optim.Adam(all_params, lr=learning_rate)
    scheduler = StepLR(optimizer1, step_size=10, gamma=0.8)

    # Print parameters - NOT WORKING
    parameters = sum(p.numel() for p in all_params if p.requires_grad)
    print(model.__class__)
    print("Parameters", parameters)


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
            outputs = model(images) # batch - num_patches - levels - dimension
            # outputs = model(images, return_all=True)# (timestep, batch, patches, levels, dimension)
            top_layer_output = outputs[:,:,-1] # BATCH, NUM_PATCHES, DIM
            letter_logits = classifier(top_layer_output)
            loss1 = criterion(letter_logits, labels)

            # Backward and optimize
            optimizer1.zero_grad()
            model1.parameters()
            loss1.backward()
            optimizer1.step()
            losses.accumulate(loss1.item(), weight=images.shape[0])
            print(f"{i} {loss1.item():.2f} {torch.max(top_layer_output):.2f}")
            if i == EPOCH_LENGTH:
                losses.reset_accumulator()
                print("Ordinary Epoch [{}/{}], Step [{}/{}] Loss: {:.4f} {}"
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss1.item(), str(losses)))
                break

        # Test the model
        model.eval()

        with torch.no_grad():
            correct1 = 0
            total1 = 0

            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = classifier(model(images)[:,:,-1])
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
            scheduler.step()


def get_params(model):
    params = [(name, torch.max(abs(param))) for name, param in model.named_parameters()]
    print(params)

if __name__=='__main__':
    main()