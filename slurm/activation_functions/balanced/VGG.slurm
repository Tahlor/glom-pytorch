/lustre/scratch/grp/fslg_internn/env/internn/bin/python
torch.Size([1000, 1, 28, 28])
max label 46
<class 'models.VGG'>
Parameters 3665135
<bound method VGG.two_conv_pool of VGG(
  (dropout): Dropout(p=0.5, inplace=False)
  (l1): Sequential(
    (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (l2): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (l3): Sequential(
    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (l4): Sequential(
    (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=256, out_features=512, bias=True)
    (2): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): ReLU(inplace=True)
    (4): Dropout(p=0.5, inplace=False)
    (5): Linear(in_features=512, out_features=47, bias=True)
  )
)> <bound method VGG.three_conv_pool of VGG(
  (dropout): Dropout(p=0.5, inplace=False)
  (l1): Sequential(
    (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (l2): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (l3): Sequential(
    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (l4): Sequential(
    (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=256, out_features=512, bias=True)
    (2): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): ReLU(inplace=True)
    (4): Dropout(p=0.5, inplace=False)
    (5): Linear(in_features=512, out_features=47, bias=True)
  )
)>
Ordinary Epoch [1/100], Step [500/1128] Loss: 1.2221
Test Accuracy of NN: 81.46808510638297 % (improvement)
Ordinary Epoch [2/100], Step [500/1128] Loss: 0.4901
Test Accuracy of NN: 85.58510638297872 % (improvement)
Ordinary Epoch [3/100], Step [500/1128] Loss: 0.4620
Test Accuracy of NN: 86.05319148936171 % (improvement)
Ordinary Epoch [4/100], Step [500/1128] Loss: 0.3811
Test Accuracy of NN: 87.86702127659575 % (improvement)
Ordinary Epoch [5/100], Step [500/1128] Loss: 0.4575
Test Accuracy of NN: 88.31382978723404 % (improvement)
Ordinary Epoch [6/100], Step [500/1128] Loss: 0.4869
/lustre/scratch/grp/fslg_internn/glom-pytorch/MNIST_vgg.py:121: DeprecationWarning: np.asscalar(a) is deprecated since NumPy v1.16, use a.item() instead
  curr_lr1 = learning_rate * np.asscalar(pow(np.random.rand(1), 3))
Test Accuracy of NN: 88.23404255319149 % Best: 88.31382978723404 %
Ordinary Epoch [7/100], Step [500/1128] Loss: 0.2227
Test Accuracy of NN: 89.49468085106383 % (improvement)
Ordinary Epoch [8/100], Step [500/1128] Loss: 0.3863
Test Accuracy of NN: 89.54255319148936 % (improvement)
Ordinary Epoch [9/100], Step [500/1128] Loss: 0.2173
Test Accuracy of NN: 89.64893617021276 % (improvement)
Ordinary Epoch [10/100], Step [500/1128] Loss: 0.4453
Test Accuracy of NN: 89.59574468085107 % Best: 89.64893617021276 %
Ordinary Epoch [11/100], Step [500/1128] Loss: 0.3070
Test Accuracy of NN: 89.04255319148936 % Best: 89.64893617021276 %
Ordinary Epoch [12/100], Step [500/1128] Loss: 0.3389
Test Accuracy of NN: 89.02659574468085 % Best: 89.64893617021276 %
Ordinary Epoch [13/100], Step [500/1128] Loss: 0.4855
Test Accuracy of NN: 89.47872340425532 % Best: 89.64893617021276 %
Ordinary Epoch [14/100], Step [500/1128] Loss: 0.6335
Test Accuracy of NN: 89.4627659574468 % Best: 89.64893617021276 %
Ordinary Epoch [15/100], Step [500/1128] Loss: 0.3519
Test Accuracy of NN: 89.77659574468085 % (improvement)
Ordinary Epoch [16/100], Step [500/1128] Loss: 0.2292
Test Accuracy of NN: 90.1063829787234 % (improvement)
Ordinary Epoch [17/100], Step [500/1128] Loss: 0.3523
Test Accuracy of NN: 89.87765957446808 % Best: 90.1063829787234 %
Ordinary Epoch [18/100], Step [500/1128] Loss: 0.3983
Test Accuracy of NN: 89.79787234042553 % Best: 90.1063829787234 %
Ordinary Epoch [19/100], Step [500/1128] Loss: 0.3962
Test Accuracy of NN: 89.60106382978724 % Best: 90.1063829787234 %
Ordinary Epoch [20/100], Step [500/1128] Loss: 0.3299
Test Accuracy of NN: 89.86170212765957 % Best: 90.1063829787234 %
Ordinary Epoch [21/100], Step [500/1128] Loss: 0.3971
Test Accuracy of NN: 90.11702127659575 % (improvement)
Ordinary Epoch [22/100], Step [500/1128] Loss: 0.2165
Test Accuracy of NN: 90.22872340425532 % (improvement)
Ordinary Epoch [23/100], Step [500/1128] Loss: 0.2309
Test Accuracy of NN: 90.40957446808511 % (improvement)
Ordinary Epoch [24/100], Step [500/1128] Loss: 0.2294
Test Accuracy of NN: 90.35106382978724 % Best: 90.40957446808511 %
Ordinary Epoch [25/100], Step [500/1128] Loss: 0.2738
Test Accuracy of NN: 90.30319148936171 % Best: 90.40957446808511 %
Ordinary Epoch [26/100], Step [500/1128] Loss: 0.2584
Test Accuracy of NN: 90.30851063829788 % Best: 90.40957446808511 %
Ordinary Epoch [27/100], Step [500/1128] Loss: 0.2742
Test Accuracy of NN: 89.96808510638297 % Best: 90.40957446808511 %
Ordinary Epoch [28/100], Step [500/1128] Loss: 0.2693
Test Accuracy of NN: 89.94680851063829 % Best: 90.40957446808511 %
Ordinary Epoch [29/100], Step [500/1128] Loss: 0.2674
Test Accuracy of NN: 90.23936170212765 % Best: 90.40957446808511 %
Ordinary Epoch [30/100], Step [500/1128] Loss: 0.1710
Test Accuracy of NN: 90.30851063829788 % Best: 90.40957446808511 %
Ordinary Epoch [31/100], Step [500/1128] Loss: 0.2375
Test Accuracy of NN: 89.67021276595744 % Best: 90.40957446808511 %
Ordinary Epoch [32/100], Step [500/1128] Loss: 0.3665
Test Accuracy of NN: 89.82446808510639 % Best: 90.40957446808511 %
Ordinary Epoch [33/100], Step [500/1128] Loss: 0.3758
Test Accuracy of NN: 89.81382978723404 % Best: 90.40957446808511 %
Ordinary Epoch [34/100], Step [500/1128] Loss: 0.2523
Test Accuracy of NN: 90.30851063829788 % Best: 90.40957446808511 %
Ordinary Epoch [35/100], Step [500/1128] Loss: 0.2577
Test Accuracy of NN: 90.57446808510639 % (improvement)
Ordinary Epoch [36/100], Step [500/1128] Loss: 0.1701
Test Accuracy of NN: 90.42021276595744 % Best: 90.57446808510639 %
Ordinary Epoch [37/100], Step [500/1128] Loss: 0.2311
Test Accuracy of NN: 89.06914893617021 % Best: 90.57446808510639 %
Ordinary Epoch [38/100], Step [500/1128] Loss: 0.2933
Test Accuracy of NN: 90.43617021276596 % Best: 90.57446808510639 %
Ordinary Epoch [39/100], Step [500/1128] Loss: 0.3241
Test Accuracy of NN: 90.48936170212765 % Best: 90.57446808510639 %
Ordinary Epoch [40/100], Step [500/1128] Loss: 0.3318
Test Accuracy of NN: 90.45212765957447 % Best: 90.57446808510639 %
Ordinary Epoch [41/100], Step [500/1128] Loss: 0.2235
Test Accuracy of NN: 90.42553191489361 % Best: 90.57446808510639 %
Ordinary Epoch [42/100], Step [500/1128] Loss: 0.2856
Test Accuracy of NN: 90.34574468085107 % Best: 90.57446808510639 %
Ordinary Epoch [43/100], Step [500/1128] Loss: 0.2006
Test Accuracy of NN: 90.56914893617021 % Best: 90.57446808510639 %
Ordinary Epoch [44/100], Step [500/1128] Loss: 0.3225
Test Accuracy of NN: 90.4627659574468 % Best: 90.57446808510639 %
Ordinary Epoch [45/100], Step [500/1128] Loss: 0.2396
Test Accuracy of NN: 90.65425531914893 % (improvement)
Ordinary Epoch [46/100], Step [500/1128] Loss: 0.2267
Test Accuracy of NN: 90.73936170212765 % (improvement)
Ordinary Epoch [47/100], Step [500/1128] Loss: 0.3551
Test Accuracy of NN: 90.7127659574468 % Best: 90.73936170212767 %
Ordinary Epoch [48/100], Step [500/1128] Loss: 0.1808
Test Accuracy of NN: 90.73936170212765 % Best: 90.73936170212767 %
Ordinary Epoch [49/100], Step [500/1128] Loss: 0.5091
Test Accuracy of NN: 89.06914893617021 % Best: 90.73936170212767 %
Ordinary Epoch [50/100], Step [500/1128] Loss: 0.1803
Test Accuracy of NN: 90.1436170212766 % Best: 90.73936170212767 %
Ordinary Epoch [51/100], Step [500/1128] Loss: 0.2729
Test Accuracy of NN: 90.17021276595744 % Best: 90.73936170212767 %
Ordinary Epoch [52/100], Step [500/1128] Loss: 0.3617
Test Accuracy of NN: 89.38297872340425 % Best: 90.73936170212767 %
Ordinary Epoch [53/100], Step [500/1128] Loss: 0.3109
Test Accuracy of NN: 90.47340425531915 % Best: 90.73936170212767 %
Ordinary Epoch [54/100], Step [500/1128] Loss: 0.2501
Test Accuracy of NN: 90.43085106382979 % Best: 90.73936170212767 %
Ordinary Epoch [55/100], Step [500/1128] Loss: 0.2832
Test Accuracy of NN: 90.51595744680851 % Best: 90.73936170212767 %
Ordinary Epoch [56/100], Step [500/1128] Loss: 0.2445
Test Accuracy of NN: 90.32446808510639 % Best: 90.73936170212767 %
Ordinary Epoch [57/100], Step [500/1128] Loss: 0.1390
Test Accuracy of NN: 90.38297872340425 % Best: 90.73936170212767 %
Ordinary Epoch [58/100], Step [500/1128] Loss: 0.2469
Test Accuracy of NN: 90.56382978723404 % Best: 90.73936170212767 %
Ordinary Epoch [59/100], Step [500/1128] Loss: 0.2190
Test Accuracy of NN: 90.32978723404256 % Best: 90.73936170212767 %
Ordinary Epoch [60/100], Step [500/1128] Loss: 0.3197
Test Accuracy of NN: 90.06382978723404 % Best: 90.73936170212767 %
Ordinary Epoch [61/100], Step [500/1128] Loss: 0.2873
Test Accuracy of NN: 90.36170212765957 % Best: 90.73936170212767 %
Ordinary Epoch [62/100], Step [500/1128] Loss: 0.2100
Test Accuracy of NN: 90.43085106382979 % Best: 90.73936170212767 %
Ordinary Epoch [63/100], Step [500/1128] Loss: 0.3361
Test Accuracy of NN: 90.45212765957447 % Best: 90.73936170212767 %
Ordinary Epoch [64/100], Step [500/1128] Loss: 0.1768
Test Accuracy of NN: 90.33510638297872 % Best: 90.73936170212767 %
Ordinary Epoch [65/100], Step [500/1128] Loss: 0.2602
Test Accuracy of NN: 90.70212765957447 % Best: 90.73936170212767 %
Ordinary Epoch [66/100], Step [500/1128] Loss: 0.3130
Test Accuracy of NN: 90.71808510638297 % Best: 90.73936170212767 %
Ordinary Epoch [67/100], Step [500/1128] Loss: 0.2849
Test Accuracy of NN: 90.53191489361703 % Best: 90.73936170212767 %
Ordinary Epoch [68/100], Step [500/1128] Loss: 0.3434
Test Accuracy of NN: 90.64893617021276 % Best: 90.73936170212767 %
Ordinary Epoch [69/100], Step [500/1128] Loss: 0.3144
Test Accuracy of NN: 89.98936170212765 % Best: 90.73936170212767 %
Ordinary Epoch [70/100], Step [500/1128] Loss: 0.2222
Test Accuracy of NN: 89.13829787234043 % Best: 90.73936170212767 %
Ordinary Epoch [71/100], Step [500/1128] Loss: 0.2983
Test Accuracy of NN: 90.59574468085107 % Best: 90.73936170212767 %
Ordinary Epoch [72/100], Step [500/1128] Loss: 0.3004
Test Accuracy of NN: 90.80851063829788 % (improvement)
Ordinary Epoch [73/100], Step [500/1128] Loss: 0.4445
Test Accuracy of NN: 90.7872340425532 % Best: 90.80851063829786 %
Ordinary Epoch [74/100], Step [500/1128] Loss: 0.2084
Test Accuracy of NN: 90.78191489361703 % Best: 90.80851063829786 %
Ordinary Epoch [75/100], Step [500/1128] Loss: 0.2079
Test Accuracy of NN: 90.81914893617021 % (improvement)
Ordinary Epoch [76/100], Step [500/1128] Loss: 0.3002
Test Accuracy of NN: 90.81914893617021 % Best: 90.81914893617021 %
Ordinary Epoch [77/100], Step [500/1128] Loss: 0.1769
Test Accuracy of NN: 90.8563829787234 % (improvement)
Ordinary Epoch [78/100], Step [500/1128] Loss: 0.2543
Test Accuracy of NN: 90.82978723404256 % Best: 90.8563829787234 %
Ordinary Epoch [79/100], Step [500/1128] Loss: 0.2496
Test Accuracy of NN: 90.80319148936171 % Best: 90.8563829787234 %
Ordinary Epoch [80/100], Step [500/1128] Loss: 0.1617
Test Accuracy of NN: 90.84574468085107 % Best: 90.8563829787234 %
Ordinary Epoch [81/100], Step [500/1128] Loss: 0.2042
Test Accuracy of NN: 90.81382978723404 % Best: 90.8563829787234 %
Ordinary Epoch [82/100], Step [500/1128] Loss: 0.2725
Test Accuracy of NN: 90.81914893617021 % Best: 90.8563829787234 %
Ordinary Epoch [83/100], Step [500/1128] Loss: 0.2067
Test Accuracy of NN: 90.75531914893617 % Best: 90.8563829787234 %
Ordinary Epoch [84/100], Step [500/1128] Loss: 0.2100
Test Accuracy of NN: 89.77659574468085 % Best: 90.8563829787234 %
Ordinary Epoch [85/100], Step [500/1128] Loss: 0.2613
Test Accuracy of NN: 90.74468085106383 % Best: 90.8563829787234 %
Ordinary Epoch [86/100], Step [500/1128] Loss: 0.3165
Test Accuracy of NN: 90.80851063829788 % Best: 90.8563829787234 %
Ordinary Epoch [87/100], Step [500/1128] Loss: 0.2253
Test Accuracy of NN: 90.8563829787234 % Best: 90.8563829787234 %
Ordinary Epoch [88/100], Step [500/1128] Loss: 0.2450
Test Accuracy of NN: 90.7872340425532 % Best: 90.8563829787234 %
Ordinary Epoch [89/100], Step [500/1128] Loss: 0.2909
Test Accuracy of NN: 90.77127659574468 % Best: 90.8563829787234 %
Ordinary Epoch [90/100], Step [500/1128] Loss: 0.2368
Test Accuracy of NN: 90.77659574468085 % Best: 90.8563829787234 %
Ordinary Epoch [91/100], Step [500/1128] Loss: 0.2456
Test Accuracy of NN: 90.65957446808511 % Best: 90.8563829787234 %
Ordinary Epoch [92/100], Step [500/1128] Loss: 0.1987
Test Accuracy of NN: 90.8563829787234 % Best: 90.8563829787234 %
Ordinary Epoch [93/100], Step [500/1128] Loss: 0.4254
Test Accuracy of NN: 90.23404255319149 % Best: 90.8563829787234 %
Ordinary Epoch [94/100], Step [500/1128] Loss: 0.1355
Test Accuracy of NN: 90.34574468085107 % Best: 90.8563829787234 %
Ordinary Epoch [95/100], Step [500/1128] Loss: 0.1249
Test Accuracy of NN: 90.40957446808511 % Best: 90.8563829787234 %
Ordinary Epoch [96/100], Step [500/1128] Loss: 0.2577
Test Accuracy of NN: 90.65425531914893 % Best: 90.8563829787234 %
Ordinary Epoch [97/100], Step [500/1128] Loss: 0.2552
Test Accuracy of NN: 90.74468085106383 % Best: 90.8563829787234 %
Ordinary Epoch [98/100], Step [500/1128] Loss: 0.2593
Test Accuracy of NN: 90.59574468085107 % Best: 90.8563829787234 %
Ordinary Epoch [99/100], Step [500/1128] Loss: 0.3057
Test Accuracy of NN: 90.17553191489361 % Best: 90.8563829787234 %
Ordinary Epoch [100/100], Step [500/1128] Loss: 0.2426
Test Accuracy of NN: 90.69680851063829 % Best: 90.8563829787234 %
