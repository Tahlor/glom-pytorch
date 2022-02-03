/lustre/scratch/grp/fslg_internn/env/internn/bin/python
torch.Size([1000, 1, 28, 28])
max label 46
<class 'models.VGG'>
Parameters 3665135
<bound method VGG.two_conv_pool of VGG(
  (dropout): Identity()
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
    (0): Identity()
    (1): Linear(in_features=256, out_features=512, bias=True)
    (2): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): ReLU(inplace=True)
    (4): Identity()
    (5): Linear(in_features=512, out_features=47, bias=True)
  )
)> <bound method VGG.three_conv_pool of VGG(
  (dropout): Identity()
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
    (0): Identity()
    (1): Linear(in_features=256, out_features=512, bias=True)
    (2): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): ReLU(inplace=True)
    (4): Identity()
    (5): Linear(in_features=512, out_features=47, bias=True)
  )
)>
Ordinary Epoch [1/100], Step [500/1128] Loss: 0.7545
Test Accuracy of NN: 82.8563829787234 % (improvement)
Ordinary Epoch [2/100], Step [500/1128] Loss: 0.5512
Test Accuracy of NN: 85.96808510638297 % (improvement)
Ordinary Epoch [3/100], Step [500/1128] Loss: 0.5569
Test Accuracy of NN: 86.48404255319149 % (improvement)
Ordinary Epoch [4/100], Step [500/1128] Loss: 0.4629
Test Accuracy of NN: 86.93085106382979 % (improvement)
Ordinary Epoch [5/100], Step [500/1128] Loss: 0.3773
Test Accuracy of NN: 87.72872340425532 % (improvement)
Ordinary Epoch [6/100], Step [500/1128] Loss: 0.3678
Test Accuracy of NN: 87.84574468085107 % (improvement)
Ordinary Epoch [7/100], Step [500/1128] Loss: 0.1898
Test Accuracy of NN: 88.37234042553192 % (improvement)
Ordinary Epoch [8/100], Step [500/1128] Loss: 0.3141
Test Accuracy of NN: 88.8936170212766 % (improvement)
Ordinary Epoch [9/100], Step [500/1128] Loss: 0.2777
/lustre/scratch/grp/fslg_internn/glom-pytorch/MNIST_vgg.py:121: DeprecationWarning: np.asscalar(a) is deprecated since NumPy v1.16, use a.item() instead
  curr_lr1 = learning_rate * np.asscalar(pow(np.random.rand(1), 3))
Test Accuracy of NN: 88.8936170212766 % Best: 88.8936170212766 %
Ordinary Epoch [10/100], Step [500/1128] Loss: 0.1814
Test Accuracy of NN: 90.15957446808511 % (improvement)
Ordinary Epoch [11/100], Step [500/1128] Loss: 0.1907
Test Accuracy of NN: 90.35106382978724 % (improvement)
Ordinary Epoch [12/100], Step [500/1128] Loss: 0.4287
Test Accuracy of NN: 90.47872340425532 % (improvement)
Ordinary Epoch [13/100], Step [500/1128] Loss: 0.3061
Test Accuracy of NN: 90.40425531914893 % Best: 90.47872340425532 %
Ordinary Epoch [14/100], Step [500/1128] Loss: 0.3930
Test Accuracy of NN: 90.10106382978724 % Best: 90.47872340425532 %
Ordinary Epoch [15/100], Step [500/1128] Loss: 0.3537
Test Accuracy of NN: 90.42021276595744 % Best: 90.47872340425532 %
Ordinary Epoch [16/100], Step [500/1128] Loss: 0.2167
Test Accuracy of NN: 90.40957446808511 % Best: 90.47872340425532 %
Ordinary Epoch [17/100], Step [500/1128] Loss: 0.1999
Test Accuracy of NN: 89.00531914893617 % Best: 90.47872340425532 %
Ordinary Epoch [18/100], Step [500/1128] Loss: 0.1827
Test Accuracy of NN: 89.97872340425532 % Best: 90.47872340425532 %
Ordinary Epoch [19/100], Step [500/1128] Loss: 0.2622
Test Accuracy of NN: 89.86170212765957 % Best: 90.47872340425532 %
Ordinary Epoch [20/100], Step [500/1128] Loss: 0.4067
Test Accuracy of NN: 90.34574468085107 % Best: 90.47872340425532 %
Ordinary Epoch [21/100], Step [500/1128] Loss: 0.3105
Test Accuracy of NN: 90.25 % Best: 90.47872340425532 %
Ordinary Epoch [22/100], Step [500/1128] Loss: 0.1967
Test Accuracy of NN: 90.30851063829788 % Best: 90.47872340425532 %
Ordinary Epoch [23/100], Step [500/1128] Loss: 0.2190
Test Accuracy of NN: 89.89893617021276 % Best: 90.47872340425532 %
Ordinary Epoch [24/100], Step [500/1128] Loss: 0.2616
Test Accuracy of NN: 90.35106382978724 % Best: 90.47872340425532 %
Ordinary Epoch [25/100], Step [500/1128] Loss: 0.2240
Test Accuracy of NN: 90.72340425531915 % (improvement)
Ordinary Epoch [26/100], Step [500/1128] Loss: 0.2308
Test Accuracy of NN: 90.65957446808511 % Best: 90.72340425531915 %
Ordinary Epoch [27/100], Step [500/1128] Loss: 0.2261
Test Accuracy of NN: 89.52127659574468 % Best: 90.72340425531915 %
Ordinary Epoch [28/100], Step [500/1128] Loss: 0.1905
Test Accuracy of NN: 90.52659574468085 % Best: 90.72340425531915 %
Ordinary Epoch [29/100], Step [500/1128] Loss: 0.1700
Test Accuracy of NN: 90.59042553191489 % Best: 90.72340425531915 %
Ordinary Epoch [30/100], Step [500/1128] Loss: 0.1873
Test Accuracy of NN: 90.54787234042553 % Best: 90.72340425531915 %
Ordinary Epoch [31/100], Step [500/1128] Loss: 0.1495
Test Accuracy of NN: 90.19148936170212 % Best: 90.72340425531915 %
Ordinary Epoch [32/100], Step [500/1128] Loss: 0.2338
Test Accuracy of NN: 90.76063829787235 % (improvement)
Ordinary Epoch [33/100], Step [500/1128] Loss: 0.2614
Test Accuracy of NN: 90.76595744680851 % (improvement)
Ordinary Epoch [34/100], Step [500/1128] Loss: 0.2262
Test Accuracy of NN: 90.75531914893617 % Best: 90.76595744680851 %
Ordinary Epoch [35/100], Step [500/1128] Loss: 0.3491
Test Accuracy of NN: 90.8936170212766 % (improvement)
Ordinary Epoch [36/100], Step [500/1128] Loss: 0.1775
Test Accuracy of NN: 90.81382978723404 % Best: 90.8936170212766 %
Ordinary Epoch [37/100], Step [500/1128] Loss: 0.2901
Test Accuracy of NN: 90.80851063829788 % Best: 90.8936170212766 %
Ordinary Epoch [38/100], Step [500/1128] Loss: 0.2421
Test Accuracy of NN: 90.35106382978724 % Best: 90.8936170212766 %
Ordinary Epoch [39/100], Step [500/1128] Loss: 0.2175
Test Accuracy of NN: 90.60106382978724 % Best: 90.8936170212766 %
Ordinary Epoch [40/100], Step [500/1128] Loss: 0.3932
Test Accuracy of NN: 90.24468085106383 % Best: 90.8936170212766 %
Ordinary Epoch [41/100], Step [500/1128] Loss: 0.1970
Test Accuracy of NN: 90.60106382978724 % Best: 90.8936170212766 %
Ordinary Epoch [42/100], Step [500/1128] Loss: 0.2638
Test Accuracy of NN: 90.69680851063829 % Best: 90.8936170212766 %
Ordinary Epoch [43/100], Step [500/1128] Loss: 0.3148
Test Accuracy of NN: 90.64893617021276 % Best: 90.8936170212766 %
Ordinary Epoch [44/100], Step [500/1128] Loss: 0.2400
Test Accuracy of NN: 90.69148936170212 % Best: 90.8936170212766 %
Ordinary Epoch [45/100], Step [500/1128] Loss: 0.2517
Test Accuracy of NN: 90.71808510638297 % Best: 90.8936170212766 %
Ordinary Epoch [46/100], Step [500/1128] Loss: 0.2020
Test Accuracy of NN: 90.82446808510639 % Best: 90.8936170212766 %
Ordinary Epoch [47/100], Step [500/1128] Loss: 0.1535
Test Accuracy of NN: 90.81914893617021 % Best: 90.8936170212766 %
Ordinary Epoch [48/100], Step [500/1128] Loss: 0.3016
Test Accuracy of NN: 90.05319148936171 % Best: 90.8936170212766 %
Ordinary Epoch [49/100], Step [500/1128] Loss: 0.1879
Test Accuracy of NN: 89.59574468085107 % Best: 90.8936170212766 %
Ordinary Epoch [50/100], Step [500/1128] Loss: 0.2396
Test Accuracy of NN: 90.63297872340425 % Best: 90.8936170212766 %
Ordinary Epoch [51/100], Step [500/1128] Loss: 0.1247
Test Accuracy of NN: 90.02659574468085 % Best: 90.8936170212766 %
Ordinary Epoch [52/100], Step [500/1128] Loss: 0.2412
Test Accuracy of NN: 90.52127659574468 % Best: 90.8936170212766 %
Ordinary Epoch [53/100], Step [500/1128] Loss: 0.3707
Test Accuracy of NN: 90.45744680851064 % Best: 90.8936170212766 %
Ordinary Epoch [54/100], Step [500/1128] Loss: 0.4023
Test Accuracy of NN: 90.56382978723404 % Best: 90.8936170212766 %
Ordinary Epoch [55/100], Step [500/1128] Loss: 0.2215
Test Accuracy of NN: 90.73936170212765 % Best: 90.8936170212766 %
Ordinary Epoch [56/100], Step [500/1128] Loss: 0.1469
Test Accuracy of NN: 90.38297872340425 % Best: 90.8936170212766 %
Ordinary Epoch [57/100], Step [500/1128] Loss: 0.2890
Test Accuracy of NN: 90.25 % Best: 90.8936170212766 %
Ordinary Epoch [58/100], Step [500/1128] Loss: 0.1233
Test Accuracy of NN: 90.79787234042553 % Best: 90.8936170212766 %
Ordinary Epoch [59/100], Step [500/1128] Loss: 0.3080
Test Accuracy of NN: 90.29787234042553 % Best: 90.8936170212766 %
Ordinary Epoch [60/100], Step [500/1128] Loss: 0.3277
Test Accuracy of NN: 90.8563829787234 % Best: 90.8936170212766 %
Ordinary Epoch [61/100], Step [500/1128] Loss: 0.2192
Test Accuracy of NN: 90.93085106382979 % (improvement)
Ordinary Epoch [62/100], Step [500/1128] Loss: 0.2034
Test Accuracy of NN: 90.94680851063829 % (improvement)
Ordinary Epoch [63/100], Step [500/1128] Loss: 0.2056
Test Accuracy of NN: 90.82978723404256 % Best: 90.9468085106383 %
Ordinary Epoch [64/100], Step [500/1128] Loss: 0.2437
Test Accuracy of NN: 90.82978723404256 % Best: 90.9468085106383 %
Ordinary Epoch [65/100], Step [500/1128] Loss: 0.2551
Test Accuracy of NN: 90.98936170212765 % (improvement)
Ordinary Epoch [66/100], Step [500/1128] Loss: 0.1648
Test Accuracy of NN: 90.92021276595744 % Best: 90.98936170212765 %
Ordinary Epoch [67/100], Step [500/1128] Loss: 0.1592
Test Accuracy of NN: 90.82978723404256 % Best: 90.98936170212765 %
Ordinary Epoch [68/100], Step [500/1128] Loss: 0.2576
Test Accuracy of NN: 89.93085106382979 % Best: 90.98936170212765 %
Ordinary Epoch [69/100], Step [500/1128] Loss: 0.3007
Test Accuracy of NN: 89.94148936170212 % Best: 90.98936170212765 %
Ordinary Epoch [70/100], Step [500/1128] Loss: 0.2195
Test Accuracy of NN: 90.13829787234043 % Best: 90.98936170212765 %
Ordinary Epoch [71/100], Step [500/1128] Loss: 0.1691
Test Accuracy of NN: 90.71808510638297 % Best: 90.98936170212765 %
Ordinary Epoch [72/100], Step [500/1128] Loss: 0.3647
Test Accuracy of NN: 90.84042553191489 % Best: 90.98936170212765 %
Ordinary Epoch [73/100], Step [500/1128] Loss: 0.1541
Test Accuracy of NN: 90.74468085106383 % Best: 90.98936170212765 %
Ordinary Epoch [74/100], Step [500/1128] Loss: 0.1599
Test Accuracy of NN: 90.51595744680851 % Best: 90.98936170212765 %
Ordinary Epoch [75/100], Step [500/1128] Loss: 0.2550
Test Accuracy of NN: 89.91489361702128 % Best: 90.98936170212765 %
Ordinary Epoch [76/100], Step [500/1128] Loss: 0.1750
Test Accuracy of NN: 90.73936170212765 % Best: 90.98936170212765 %
Ordinary Epoch [77/100], Step [500/1128] Loss: 0.1551
Test Accuracy of NN: 90.79787234042553 % Best: 90.98936170212765 %
Ordinary Epoch [78/100], Step [500/1128] Loss: 0.2834
Test Accuracy of NN: 89.97872340425532 % Best: 90.98936170212765 %
Ordinary Epoch [79/100], Step [500/1128] Loss: 0.2531
Test Accuracy of NN: 90.23404255319149 % Best: 90.98936170212765 %
Ordinary Epoch [80/100], Step [500/1128] Loss: 0.2757
Test Accuracy of NN: 90.20744680851064 % Best: 90.98936170212765 %
Ordinary Epoch [81/100], Step [500/1128] Loss: 0.1447
Test Accuracy of NN: 90.25 % Best: 90.98936170212765 %
Ordinary Epoch [82/100], Step [500/1128] Loss: 0.1340
Test Accuracy of NN: 90.55851063829788 % Best: 90.98936170212765 %
Ordinary Epoch [83/100], Step [500/1128] Loss: 0.1472
Test Accuracy of NN: 90.5372340425532 % Best: 90.98936170212765 %
Ordinary Epoch [84/100], Step [500/1128] Loss: 0.1399
Test Accuracy of NN: 90.92021276595744 % Best: 90.98936170212765 %
Ordinary Epoch [85/100], Step [500/1128] Loss: 0.2431
Test Accuracy of NN: 90.11170212765957 % Best: 90.98936170212765 %
Ordinary Epoch [86/100], Step [500/1128] Loss: 0.2429
Test Accuracy of NN: 90.70744680851064 % Best: 90.98936170212765 %
Ordinary Epoch [87/100], Step [500/1128] Loss: 0.2670
Test Accuracy of NN: 89.94148936170212 % Best: 90.98936170212765 %
Ordinary Epoch [88/100], Step [500/1128] Loss: 0.1793
Test Accuracy of NN: 90.52659574468085 % Best: 90.98936170212765 %
Ordinary Epoch [89/100], Step [500/1128] Loss: 0.1640
Test Accuracy of NN: 90.56914893617021 % Best: 90.98936170212765 %
Ordinary Epoch [90/100], Step [500/1128] Loss: 0.3598
Test Accuracy of NN: 89.96808510638297 % Best: 90.98936170212765 %
Ordinary Epoch [91/100], Step [500/1128] Loss: 0.1708
Test Accuracy of NN: 90.69680851063829 % Best: 90.98936170212765 %
Ordinary Epoch [92/100], Step [500/1128] Loss: 0.2343
Test Accuracy of NN: 90.76595744680851 % Best: 90.98936170212765 %
Ordinary Epoch [93/100], Step [500/1128] Loss: 0.0837
Test Accuracy of NN: 90.76063829787235 % Best: 90.98936170212765 %
Ordinary Epoch [94/100], Step [500/1128] Loss: 0.2253
Test Accuracy of NN: 90.57978723404256 % Best: 90.98936170212765 %
Ordinary Epoch [95/100], Step [500/1128] Loss: 0.1370
Test Accuracy of NN: 90.70744680851064 % Best: 90.98936170212765 %
Ordinary Epoch [96/100], Step [500/1128] Loss: 0.2196
Test Accuracy of NN: 90.80851063829788 % Best: 90.98936170212765 %
Ordinary Epoch [97/100], Step [500/1128] Loss: 0.2521
Test Accuracy of NN: 90.86170212765957 % Best: 90.98936170212765 %
Ordinary Epoch [98/100], Step [500/1128] Loss: 0.1192
Test Accuracy of NN: 90.47872340425532 % Best: 90.98936170212765 %
Ordinary Epoch [99/100], Step [500/1128] Loss: 0.1815
Test Accuracy of NN: 90.47872340425532 % Best: 90.98936170212765 %
Ordinary Epoch [100/100], Step [500/1128] Loss: 0.1340
Test Accuracy of NN: 90.63297872340425 % Best: 90.98936170212765 %
