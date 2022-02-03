/lustre/scratch/grp/fslg_internn/env/internn/bin/python
torch.Size([1000, 1, 28, 28])
max label 46
<class 'models.V1'>
Parameters 1117871
<bound method V1.two_conv_pool of V1(
  (dropout): Dropout(p=0.5, inplace=False)
  (l1): Sequential(
    (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (l2): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (l3): Sequential(
    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (l4): Sequential(
    (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=256, out_features=512, bias=True)
    (2): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=512, out_features=47, bias=True)
  )
)> <bound method V1.three_conv_pool of V1(
  (dropout): Dropout(p=0.5, inplace=False)
  (l1): Sequential(
    (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (l2): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (l3): Sequential(
    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (l4): Sequential(
    (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=256, out_features=512, bias=True)
    (2): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=512, out_features=47, bias=True)
  )
)>
Ordinary Epoch [1/100], Step [500/1128] Loss: 1.0045
Test Accuracy of NN: 84.24468085106383 % (improvement)
Ordinary Epoch [2/100], Step [500/1128] Loss: 0.7317
Test Accuracy of NN: 85.77659574468085 % (improvement)
Ordinary Epoch [3/100], Step [500/1128] Loss: 0.5805
Test Accuracy of NN: 86.96808510638297 % (improvement)
Ordinary Epoch [4/100], Step [500/1128] Loss: 0.6491
/lustre/scratch/grp/fslg_internn/glom-pytorch/MNIST_vgg.py:121: DeprecationWarning: np.asscalar(a) is deprecated since NumPy v1.16, use a.item() instead
  curr_lr1 = learning_rate * np.asscalar(pow(np.random.rand(1), 3))
Test Accuracy of NN: 86.73936170212765 % Best: 86.96808510638297 %
Ordinary Epoch [5/100], Step [500/1128] Loss: 0.2572
Test Accuracy of NN: 87.46808510638297 % (improvement)
Ordinary Epoch [6/100], Step [500/1128] Loss: 0.5149
Test Accuracy of NN: 87.47872340425532 % (improvement)
Ordinary Epoch [7/100], Step [500/1128] Loss: 0.3450
Test Accuracy of NN: 88.01595744680851 % (improvement)
Ordinary Epoch [8/100], Step [500/1128] Loss: 0.5690
Test Accuracy of NN: 88.04255319148936 % (improvement)
Ordinary Epoch [9/100], Step [500/1128] Loss: 0.6064
Test Accuracy of NN: 87.80319148936171 % Best: 88.04255319148936 %
Ordinary Epoch [10/100], Step [500/1128] Loss: 0.3769
Test Accuracy of NN: 89.00531914893617 % (improvement)
Ordinary Epoch [11/100], Step [500/1128] Loss: 0.3082
Test Accuracy of NN: 89.12234042553192 % (improvement)
Ordinary Epoch [12/100], Step [500/1128] Loss: 0.3783
Test Accuracy of NN: 89.24468085106383 % (improvement)
Ordinary Epoch [13/100], Step [500/1128] Loss: 0.3254
Test Accuracy of NN: 89.28191489361703 % (improvement)
Ordinary Epoch [14/100], Step [500/1128] Loss: 0.5448
Test Accuracy of NN: 89.29255319148936 % (improvement)
Ordinary Epoch [15/100], Step [500/1128] Loss: 0.2926
Test Accuracy of NN: 89.36170212765957 % (improvement)
Ordinary Epoch [16/100], Step [500/1128] Loss: 0.3767
Test Accuracy of NN: 89.33510638297872 % Best: 89.36170212765957 %
Ordinary Epoch [17/100], Step [500/1128] Loss: 0.3103
Test Accuracy of NN: 88.39893617021276 % Best: 89.36170212765957 %
Ordinary Epoch [18/100], Step [500/1128] Loss: 0.3504
Test Accuracy of NN: 88.92553191489361 % Best: 89.36170212765957 %
Ordinary Epoch [19/100], Step [500/1128] Loss: 0.3134
Test Accuracy of NN: 89.22340425531915 % Best: 89.36170212765957 %
Ordinary Epoch [20/100], Step [500/1128] Loss: 0.5531
Test Accuracy of NN: 89.52659574468085 % (improvement)
Ordinary Epoch [21/100], Step [500/1128] Loss: 0.2508
Test Accuracy of NN: 89.48404255319149 % Best: 89.52659574468085 %
Ordinary Epoch [22/100], Step [500/1128] Loss: 0.5878
Test Accuracy of NN: 88.2127659574468 % Best: 89.52659574468085 %
Ordinary Epoch [23/100], Step [500/1128] Loss: 0.5030
Test Accuracy of NN: 88.20212765957447 % Best: 89.52659574468085 %
Ordinary Epoch [24/100], Step [500/1128] Loss: 0.3917
Test Accuracy of NN: 89.00531914893617 % Best: 89.52659574468085 %
Ordinary Epoch [25/100], Step [500/1128] Loss: 0.5175
Test Accuracy of NN: 89.04787234042553 % Best: 89.52659574468085 %
Ordinary Epoch [26/100], Step [500/1128] Loss: 0.4107
Test Accuracy of NN: 89.09042553191489 % Best: 89.52659574468085 %
Ordinary Epoch [27/100], Step [500/1128] Loss: 0.4249
Test Accuracy of NN: 88.29787234042553 % Best: 89.52659574468085 %
Ordinary Epoch [28/100], Step [500/1128] Loss: 0.4160
Test Accuracy of NN: 88.28191489361703 % Best: 89.52659574468085 %
Ordinary Epoch [29/100], Step [500/1128] Loss: 0.3835
Test Accuracy of NN: 89.14893617021276 % Best: 89.52659574468085 %
Ordinary Epoch [30/100], Step [500/1128] Loss: 0.3787
Test Accuracy of NN: 89.30851063829788 % Best: 89.52659574468085 %
Ordinary Epoch [31/100], Step [500/1128] Loss: 0.3747
Test Accuracy of NN: 89.44680851063829 % Best: 89.52659574468085 %
Ordinary Epoch [32/100], Step [500/1128] Loss: 0.3723
Test Accuracy of NN: 89.02127659574468 % Best: 89.52659574468085 %
Ordinary Epoch [33/100], Step [500/1128] Loss: 0.4085
Test Accuracy of NN: 89.49468085106383 % Best: 89.52659574468085 %
Ordinary Epoch [34/100], Step [500/1128] Loss: 0.3649
Test Accuracy of NN: 89.48936170212765 % Best: 89.52659574468085 %
Ordinary Epoch [35/100], Step [500/1128] Loss: 0.3667
Test Accuracy of NN: 89.45212765957447 % Best: 89.52659574468085 %
Ordinary Epoch [36/100], Step [500/1128] Loss: 0.2783
Test Accuracy of NN: 89.56382978723404 % (improvement)
Ordinary Epoch [37/100], Step [500/1128] Loss: 0.3659
Test Accuracy of NN: 89.48936170212765 % Best: 89.56382978723404 %
Ordinary Epoch [38/100], Step [500/1128] Loss: 0.2907
Test Accuracy of NN: 88.63297872340425 % Best: 89.56382978723404 %
Ordinary Epoch [39/100], Step [500/1128] Loss: 0.4324
Test Accuracy of NN: 89.51063829787235 % Best: 89.56382978723404 %
Ordinary Epoch [40/100], Step [500/1128] Loss: 0.2442
Test Accuracy of NN: 89.18085106382979 % Best: 89.56382978723404 %
Ordinary Epoch [41/100], Step [500/1128] Loss: 0.4734
Test Accuracy of NN: 89.60106382978724 % (improvement)
Ordinary Epoch [42/100], Step [500/1128] Loss: 0.2877
Test Accuracy of NN: 89.65425531914893 % (improvement)
Ordinary Epoch [43/100], Step [500/1128] Loss: 0.3367
Test Accuracy of NN: 89.70744680851064 % (improvement)
Ordinary Epoch [44/100], Step [500/1128] Loss: 0.3776
Test Accuracy of NN: 89.62234042553192 % Best: 89.70744680851064 %
Ordinary Epoch [45/100], Step [500/1128] Loss: 0.3573
Test Accuracy of NN: 89.59574468085107 % Best: 89.70744680851064 %
Ordinary Epoch [46/100], Step [500/1128] Loss: 0.3696
Test Accuracy of NN: 89.74468085106383 % (improvement)
Ordinary Epoch [47/100], Step [500/1128] Loss: 0.4122
Test Accuracy of NN: 89.85106382978724 % (improvement)
Ordinary Epoch [48/100], Step [500/1128] Loss: 0.2477
Test Accuracy of NN: 89.73404255319149 % Best: 89.85106382978724 %
Ordinary Epoch [49/100], Step [500/1128] Loss: 0.2957
Test Accuracy of NN: 89.59042553191489 % Best: 89.85106382978724 %
Ordinary Epoch [50/100], Step [500/1128] Loss: 0.4038
Test Accuracy of NN: 89.61702127659575 % Best: 89.85106382978724 %
Ordinary Epoch [51/100], Step [500/1128] Loss: 0.3149
Test Accuracy of NN: 89.70212765957447 % Best: 89.85106382978724 %
Ordinary Epoch [52/100], Step [500/1128] Loss: 0.3189
Test Accuracy of NN: 89.86170212765957 % (improvement)
Ordinary Epoch [53/100], Step [500/1128] Loss: 0.2209
Test Accuracy of NN: 89.87234042553192 % (improvement)
Ordinary Epoch [54/100], Step [500/1128] Loss: 0.3544
Test Accuracy of NN: 89.8563829787234 % Best: 89.87234042553192 %
Ordinary Epoch [55/100], Step [500/1128] Loss: 0.2212
Test Accuracy of NN: 89.84574468085107 % Best: 89.87234042553192 %
Ordinary Epoch [56/100], Step [500/1128] Loss: 0.5072
Test Accuracy of NN: 89.92021276595744 % (improvement)
Ordinary Epoch [57/100], Step [500/1128] Loss: 0.3127
Test Accuracy of NN: 89.80851063829788 % Best: 89.92021276595744 %
Ordinary Epoch [58/100], Step [500/1128] Loss: 0.2678
Test Accuracy of NN: 89.75 % Best: 89.92021276595744 %
Ordinary Epoch [59/100], Step [500/1128] Loss: 0.3636
Test Accuracy of NN: 89.7127659574468 % Best: 89.92021276595744 %
Ordinary Epoch [60/100], Step [500/1128] Loss: 0.5504
Test Accuracy of NN: 89.26063829787235 % Best: 89.92021276595744 %
Ordinary Epoch [61/100], Step [500/1128] Loss: 0.2201
Test Accuracy of NN: 89.52659574468085 % Best: 89.92021276595744 %
Ordinary Epoch [62/100], Step [500/1128] Loss: 0.4487
Test Accuracy of NN: 89.48936170212765 % Best: 89.92021276595744 %
Ordinary Epoch [63/100], Step [500/1128] Loss: 0.4250
Test Accuracy of NN: 89.80851063829788 % Best: 89.92021276595744 %
Ordinary Epoch [64/100], Step [500/1128] Loss: 0.4290
Test Accuracy of NN: 89.2127659574468 % Best: 89.92021276595744 %
Ordinary Epoch [65/100], Step [500/1128] Loss: 0.1849
Test Accuracy of NN: 89.55851063829788 % Best: 89.92021276595744 %
Ordinary Epoch [66/100], Step [500/1128] Loss: 0.5251
Test Accuracy of NN: 88.43085106382979 % Best: 89.92021276595744 %
Ordinary Epoch [67/100], Step [500/1128] Loss: 0.3557
Test Accuracy of NN: 88.69148936170212 % Best: 89.92021276595744 %
Ordinary Epoch [68/100], Step [500/1128] Loss: 0.4293
Test Accuracy of NN: 88.66489361702128 % Best: 89.92021276595744 %
Ordinary Epoch [69/100], Step [500/1128] Loss: 0.3330
Test Accuracy of NN: 89.76063829787235 % Best: 89.92021276595744 %
Ordinary Epoch [70/100], Step [500/1128] Loss: 0.3876
Test Accuracy of NN: 89.77127659574468 % Best: 89.92021276595744 %
Ordinary Epoch [71/100], Step [500/1128] Loss: 0.4121
Test Accuracy of NN: 89.77659574468085 % Best: 89.92021276595744 %
Ordinary Epoch [72/100], Step [500/1128] Loss: 0.4402
Test Accuracy of NN: 89.80319148936171 % Best: 89.92021276595744 %
Ordinary Epoch [73/100], Step [500/1128] Loss: 0.3395
Test Accuracy of NN: 89.6436170212766 % Best: 89.92021276595744 %
Ordinary Epoch [74/100], Step [500/1128] Loss: 0.2269
Test Accuracy of NN: 89.6436170212766 % Best: 89.92021276595744 %
Ordinary Epoch [75/100], Step [500/1128] Loss: 0.3464
Test Accuracy of NN: 89.87765957446808 % Best: 89.92021276595744 %
Ordinary Epoch [76/100], Step [500/1128] Loss: 0.2890
Test Accuracy of NN: 89.8936170212766 % Best: 89.92021276595744 %
Ordinary Epoch [77/100], Step [500/1128] Loss: 0.3974
Test Accuracy of NN: 89.86702127659575 % Best: 89.92021276595744 %
Ordinary Epoch [78/100], Step [500/1128] Loss: 0.2431
Test Accuracy of NN: 89.87234042553192 % Best: 89.92021276595744 %
Ordinary Epoch [79/100], Step [500/1128] Loss: 0.4040
Test Accuracy of NN: 89.5 % Best: 89.92021276595744 %
Ordinary Epoch [80/100], Step [500/1128] Loss: 0.4003
Test Accuracy of NN: 89.8563829787234 % Best: 89.92021276595744 %
Ordinary Epoch [81/100], Step [500/1128] Loss: 0.2357
Test Accuracy of NN: 89.73936170212765 % Best: 89.92021276595744 %
Ordinary Epoch [82/100], Step [500/1128] Loss: 0.4325
Test Accuracy of NN: 89.88297872340425 % Best: 89.92021276595744 %
Ordinary Epoch [83/100], Step [500/1128] Loss: 0.3592
Test Accuracy of NN: 89.71808510638297 % Best: 89.92021276595744 %
Ordinary Epoch [84/100], Step [500/1128] Loss: 0.2814
Test Accuracy of NN: 89.94680851063829 % (improvement)
Ordinary Epoch [85/100], Step [500/1128] Loss: 0.4933
Test Accuracy of NN: 90.05319148936171 % (improvement)
Ordinary Epoch [86/100], Step [500/1128] Loss: 0.4095
Test Accuracy of NN: 89.93617021276596 % Best: 90.05319148936171 %
Ordinary Epoch [87/100], Step [500/1128] Loss: 0.3317
Test Accuracy of NN: 89.89893617021276 % Best: 90.05319148936171 %
Ordinary Epoch [88/100], Step [500/1128] Loss: 0.4359
Test Accuracy of NN: 88.62765957446808 % Best: 90.05319148936171 %
Ordinary Epoch [89/100], Step [500/1128] Loss: 0.4460
Test Accuracy of NN: 89.55851063829788 % Best: 90.05319148936171 %
Ordinary Epoch [90/100], Step [500/1128] Loss: 0.3072
Test Accuracy of NN: 89.70212765957447 % Best: 90.05319148936171 %
Ordinary Epoch [91/100], Step [500/1128] Loss: 0.3365
Test Accuracy of NN: 89.27659574468085 % Best: 90.05319148936171 %
Ordinary Epoch [92/100], Step [500/1128] Loss: 0.2148
Test Accuracy of NN: 89.76595744680851 % Best: 90.05319148936171 %
Ordinary Epoch [93/100], Step [500/1128] Loss: 0.3464
Test Accuracy of NN: 89.81382978723404 % Best: 90.05319148936171 %
Ordinary Epoch [94/100], Step [500/1128] Loss: 0.2395
Test Accuracy of NN: 89.53191489361703 % Best: 90.05319148936171 %
Ordinary Epoch [95/100], Step [500/1128] Loss: 0.2838
Test Accuracy of NN: 89.81914893617021 % Best: 90.05319148936171 %
Ordinary Epoch [96/100], Step [500/1128] Loss: 0.3054
Test Accuracy of NN: 89.33510638297872 % Best: 90.05319148936171 %
Ordinary Epoch [97/100], Step [500/1128] Loss: 0.2570
Test Accuracy of NN: 89.69680851063829 % Best: 90.05319148936171 %
Ordinary Epoch [98/100], Step [500/1128] Loss: 0.1992
Test Accuracy of NN: 89.74468085106383 % Best: 90.05319148936171 %
Ordinary Epoch [99/100], Step [500/1128] Loss: 0.2588
Test Accuracy of NN: 89.76063829787235 % Best: 90.05319148936171 %
Ordinary Epoch [100/100], Step [500/1128] Loss: 0.5021
Test Accuracy of NN: 89.20744680851064 % Best: 90.05319148936171 %
