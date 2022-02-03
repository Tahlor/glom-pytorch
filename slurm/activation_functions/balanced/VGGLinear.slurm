/lustre/scratch/grp/fslg_internn/env/internn/bin/python
torch.Size([1000, 1, 28, 28])
max label 46
<class 'models.VGGLinear'>
Parameters 3665135
<bound method VGGLinear.two_conv_pool of VGGLinear(
  (dropout): Dropout(p=0.5, inplace=False)
  (l1): Sequential(
    (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (l2): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (l3): Sequential(
    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (l4): Sequential(
    (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=256, out_features=512, bias=True)
    (2): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=512, out_features=47, bias=True)
  )
)> <bound method VGGLinear.three_conv_pool of VGGLinear(
  (dropout): Dropout(p=0.5, inplace=False)
  (l1): Sequential(
    (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (l2): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (l3): Sequential(
    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (l4): Sequential(
    (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=256, out_features=512, bias=True)
    (2): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=512, out_features=47, bias=True)
  )
)>
Ordinary Epoch [1/100], Step [500/1128] Loss: 0.9102
Test Accuracy of NN: 81.95212765957447 % (improvement)
Ordinary Epoch [2/100], Step [500/1128] Loss: 0.4369
Test Accuracy of NN: 83.40957446808511 % (improvement)
Ordinary Epoch [3/100], Step [500/1128] Loss: 0.5541
Test Accuracy of NN: 85.29787234042553 % (improvement)
Ordinary Epoch [4/100], Step [500/1128] Loss: 0.6048
Test Accuracy of NN: 85.60106382978724 % (improvement)
Ordinary Epoch [5/100], Step [500/1128] Loss: 0.4359
Test Accuracy of NN: 86.01595744680851 % (improvement)
Ordinary Epoch [6/100], Step [500/1128] Loss: 0.5031
Test Accuracy of NN: 86.59574468085107 % (improvement)
Ordinary Epoch [7/100], Step [500/1128] Loss: 0.5179
Test Accuracy of NN: 86.72340425531915 % (improvement)
Ordinary Epoch [8/100], Step [500/1128] Loss: 0.5743
Test Accuracy of NN: 87.10106382978724 % (improvement)
Ordinary Epoch [9/100], Step [500/1128] Loss: 0.3030
Test Accuracy of NN: 87.42553191489361 % (improvement)
Ordinary Epoch [10/100], Step [500/1128] Loss: 0.2846
Test Accuracy of NN: 87.93085106382979 % (improvement)
Ordinary Epoch [11/100], Step [500/1128] Loss: 0.6254
/lustre/scratch/grp/fslg_internn/glom-pytorch/MNIST_vgg.py:121: DeprecationWarning: np.asscalar(a) is deprecated since NumPy v1.16, use a.item() instead
  curr_lr1 = learning_rate * np.asscalar(pow(np.random.rand(1), 3))
Test Accuracy of NN: 86.48404255319149 % Best: 87.93085106382978 %
Ordinary Epoch [12/100], Step [500/1128] Loss: 0.4934
Test Accuracy of NN: 87.94680851063829 % (improvement)
Ordinary Epoch [13/100], Step [500/1128] Loss: 0.4992
Test Accuracy of NN: 87.87765957446808 % Best: 87.94680851063829 %
Ordinary Epoch [14/100], Step [500/1128] Loss: 0.3482
Test Accuracy of NN: 89.38297872340425 % (improvement)
Ordinary Epoch [15/100], Step [500/1128] Loss: 0.6115
Test Accuracy of NN: 89.28191489361703 % Best: 89.38297872340426 %
Ordinary Epoch [16/100], Step [500/1128] Loss: 0.4003
Test Accuracy of NN: 89.15425531914893 % Best: 89.38297872340426 %
Ordinary Epoch [17/100], Step [500/1128] Loss: 0.2343
Test Accuracy of NN: 89.33510638297872 % Best: 89.38297872340426 %
Ordinary Epoch [18/100], Step [500/1128] Loss: 0.6086
Test Accuracy of NN: 89.57978723404256 % (improvement)
Ordinary Epoch [19/100], Step [500/1128] Loss: 0.3353
Test Accuracy of NN: 89.5372340425532 % Best: 89.57978723404256 %
Ordinary Epoch [20/100], Step [500/1128] Loss: 0.2716
Test Accuracy of NN: 89.52127659574468 % Best: 89.57978723404256 %
Ordinary Epoch [21/100], Step [500/1128] Loss: 0.3742
Test Accuracy of NN: 88.72340425531915 % Best: 89.57978723404256 %
Ordinary Epoch [22/100], Step [500/1128] Loss: 0.3638
Test Accuracy of NN: 89.40425531914893 % Best: 89.57978723404256 %
Ordinary Epoch [23/100], Step [500/1128] Loss: 0.4612
Test Accuracy of NN: 89.41489361702128 % Best: 89.57978723404256 %
Ordinary Epoch [24/100], Step [500/1128] Loss: 0.4021
Test Accuracy of NN: 88.38297872340425 % Best: 89.57978723404256 %
Ordinary Epoch [25/100], Step [500/1128] Loss: 0.4024
Test Accuracy of NN: 89.31382978723404 % Best: 89.57978723404256 %
Ordinary Epoch [26/100], Step [500/1128] Loss: 0.3380
Test Accuracy of NN: 89.45744680851064 % Best: 89.57978723404256 %
Ordinary Epoch [27/100], Step [500/1128] Loss: 0.4185
Test Accuracy of NN: 88.3563829787234 % Best: 89.57978723404256 %
Ordinary Epoch [28/100], Step [500/1128] Loss: 0.3674
Test Accuracy of NN: 89.46808510638297 % Best: 89.57978723404256 %
Ordinary Epoch [29/100], Step [500/1128] Loss: 0.2999
Test Accuracy of NN: 89.62234042553192 % (improvement)
Ordinary Epoch [30/100], Step [500/1128] Loss: 0.3471
Test Accuracy of NN: 89.63297872340425 % (improvement)
Ordinary Epoch [31/100], Step [500/1128] Loss: 0.5981
Test Accuracy of NN: 89.70212765957447 % (improvement)
Ordinary Epoch [32/100], Step [500/1128] Loss: 0.4434
Test Accuracy of NN: 89.73936170212765 % (improvement)
Ordinary Epoch [33/100], Step [500/1128] Loss: 0.1994
Test Accuracy of NN: 89.76063829787235 % (improvement)
Ordinary Epoch [34/100], Step [500/1128] Loss: 0.4673
Test Accuracy of NN: 89.77659574468085 % (improvement)
Ordinary Epoch [35/100], Step [500/1128] Loss: 0.2689
Test Accuracy of NN: 89.77659574468085 % Best: 89.77659574468085 %
Ordinary Epoch [36/100], Step [500/1128] Loss: 0.2581
Test Accuracy of NN: 89.85106382978724 % (improvement)
Ordinary Epoch [37/100], Step [500/1128] Loss: 0.4583
Test Accuracy of NN: 89.81914893617021 % Best: 89.85106382978724 %
Ordinary Epoch [38/100], Step [500/1128] Loss: 0.4212
Test Accuracy of NN: 89.76595744680851 % Best: 89.85106382978724 %
Ordinary Epoch [39/100], Step [500/1128] Loss: 0.4675
Test Accuracy of NN: 89.68617021276596 % Best: 89.85106382978724 %
Ordinary Epoch [40/100], Step [500/1128] Loss: 0.3454
Test Accuracy of NN: 89.30319148936171 % Best: 89.85106382978724 %
Ordinary Epoch [41/100], Step [500/1128] Loss: 0.2625
Test Accuracy of NN: 89.76595744680851 % Best: 89.85106382978724 %
Ordinary Epoch [42/100], Step [500/1128] Loss: 0.4609
Test Accuracy of NN: 89.17021276595744 % Best: 89.85106382978724 %
Ordinary Epoch [43/100], Step [500/1128] Loss: 0.3517
Test Accuracy of NN: 89.61702127659575 % Best: 89.85106382978724 %
Ordinary Epoch [44/100], Step [500/1128] Loss: 0.2248
Test Accuracy of NN: 89.77127659574468 % Best: 89.85106382978724 %
Ordinary Epoch [45/100], Step [500/1128] Loss: 0.3530
Test Accuracy of NN: 88.97872340425532 % Best: 89.85106382978724 %
Ordinary Epoch [46/100], Step [500/1128] Loss: 0.3436
Test Accuracy of NN: 88.68085106382979 % Best: 89.85106382978724 %
Ordinary Epoch [47/100], Step [500/1128] Loss: 0.5065
Test Accuracy of NN: 89.43085106382979 % Best: 89.85106382978724 %
Ordinary Epoch [48/100], Step [500/1128] Loss: 0.3972
Test Accuracy of NN: 88.70744680851064 % Best: 89.85106382978724 %
Ordinary Epoch [49/100], Step [500/1128] Loss: 0.3314
Test Accuracy of NN: 89.77659574468085 % Best: 89.85106382978724 %
Ordinary Epoch [50/100], Step [500/1128] Loss: 0.5542
Test Accuracy of NN: 89.61702127659575 % Best: 89.85106382978724 %
Ordinary Epoch [51/100], Step [500/1128] Loss: 0.3008
Test Accuracy of NN: 89.98936170212765 % (improvement)
Ordinary Epoch [52/100], Step [500/1128] Loss: 0.3198
Test Accuracy of NN: 89.8936170212766 % Best: 89.98936170212765 %
Ordinary Epoch [53/100], Step [500/1128] Loss: 0.3968
Test Accuracy of NN: 89.07978723404256 % Best: 89.98936170212765 %
Ordinary Epoch [54/100], Step [500/1128] Loss: 0.5970
Test Accuracy of NN: 89.32978723404256 % Best: 89.98936170212765 %
Ordinary Epoch [55/100], Step [500/1128] Loss: 0.2509
Test Accuracy of NN: 89.63829787234043 % Best: 89.98936170212765 %
Ordinary Epoch [56/100], Step [500/1128] Loss: 0.4257
Test Accuracy of NN: 89.1436170212766 % Best: 89.98936170212765 %
Ordinary Epoch [57/100], Step [500/1128] Loss: 0.3719
Test Accuracy of NN: 89.98936170212765 % Best: 89.98936170212765 %
Ordinary Epoch [58/100], Step [500/1128] Loss: 0.2629
Test Accuracy of NN: 89.95744680851064 % Best: 89.98936170212765 %
Ordinary Epoch [59/100], Step [500/1128] Loss: 0.1888
Test Accuracy of NN: 89.89893617021276 % Best: 89.98936170212765 %
Ordinary Epoch [60/100], Step [500/1128] Loss: 0.4571
Test Accuracy of NN: 89.80851063829788 % Best: 89.98936170212765 %
Ordinary Epoch [61/100], Step [500/1128] Loss: 0.2404
Test Accuracy of NN: 89.88829787234043 % Best: 89.98936170212765 %
Ordinary Epoch [62/100], Step [500/1128] Loss: 0.4007
Test Accuracy of NN: 90.02659574468085 % (improvement)
Ordinary Epoch [63/100], Step [500/1128] Loss: 0.2517
Test Accuracy of NN: 90.03191489361703 % (improvement)
Ordinary Epoch [64/100], Step [500/1128] Loss: 0.4554
Test Accuracy of NN: 90.15957446808511 % (improvement)
Ordinary Epoch [65/100], Step [500/1128] Loss: 0.3711
Test Accuracy of NN: 90.07978723404256 % Best: 90.1595744680851 %
Ordinary Epoch [66/100], Step [500/1128] Loss: 0.4296
Test Accuracy of NN: 89.62234042553192 % Best: 90.1595744680851 %
Ordinary Epoch [67/100], Step [500/1128] Loss: 0.2755
Test Accuracy of NN: 89.63829787234043 % Best: 90.1595744680851 %
Ordinary Epoch [68/100], Step [500/1128] Loss: 0.3863
Test Accuracy of NN: 89.93617021276596 % Best: 90.1595744680851 %
Ordinary Epoch [69/100], Step [500/1128] Loss: 0.3737
Test Accuracy of NN: 90.01063829787235 % Best: 90.1595744680851 %
Ordinary Epoch [70/100], Step [500/1128] Loss: 0.4807
Test Accuracy of NN: 89.02659574468085 % Best: 90.1595744680851 %
Ordinary Epoch [71/100], Step [500/1128] Loss: 0.3172
Test Accuracy of NN: 89.8563829787234 % Best: 90.1595744680851 %
Ordinary Epoch [72/100], Step [500/1128] Loss: 0.1574
Test Accuracy of NN: 89.65425531914893 % Best: 90.1595744680851 %
Ordinary Epoch [73/100], Step [500/1128] Loss: 0.3279
Test Accuracy of NN: 90.01063829787235 % Best: 90.1595744680851 %
Ordinary Epoch [74/100], Step [500/1128] Loss: 0.4130
Test Accuracy of NN: 88.57446808510639 % Best: 90.1595744680851 %
Ordinary Epoch [75/100], Step [500/1128] Loss: 0.4341
Test Accuracy of NN: 89.55851063829788 % Best: 90.1595744680851 %
Ordinary Epoch [76/100], Step [500/1128] Loss: 0.2391
Test Accuracy of NN: 89.99468085106383 % Best: 90.1595744680851 %
Ordinary Epoch [77/100], Step [500/1128] Loss: 0.2799
Test Accuracy of NN: 90.07978723404256 % Best: 90.1595744680851 %
Ordinary Epoch [78/100], Step [500/1128] Loss: 0.3025
Test Accuracy of NN: 90.17021276595744 % (improvement)
Ordinary Epoch [79/100], Step [500/1128] Loss: 0.2487
Test Accuracy of NN: 90.11170212765957 % Best: 90.17021276595744 %
Ordinary Epoch [80/100], Step [500/1128] Loss: 0.2282
Test Accuracy of NN: 90.03191489361703 % Best: 90.17021276595744 %
Ordinary Epoch [81/100], Step [500/1128] Loss: 0.2979
Test Accuracy of NN: 90.01063829787235 % Best: 90.17021276595744 %
Ordinary Epoch [82/100], Step [500/1128] Loss: 0.4004
Test Accuracy of NN: 89.05851063829788 % Best: 90.17021276595744 %
Ordinary Epoch [83/100], Step [500/1128] Loss: 0.2335
Test Accuracy of NN: 89.74468085106383 % Best: 90.17021276595744 %
Ordinary Epoch [84/100], Step [500/1128] Loss: 0.3108
Test Accuracy of NN: 90.13297872340425 % Best: 90.17021276595744 %
Ordinary Epoch [85/100], Step [500/1128] Loss: 0.3779
Test Accuracy of NN: 89.27659574468085 % Best: 90.17021276595744 %
Ordinary Epoch [86/100], Step [500/1128] Loss: 0.3347
Test Accuracy of NN: 90.08510638297872 % Best: 90.17021276595744 %
Ordinary Epoch [87/100], Step [500/1128] Loss: 0.2455
Test Accuracy of NN: 90.07978723404256 % Best: 90.17021276595744 %
Ordinary Epoch [88/100], Step [500/1128] Loss: 0.2811
Test Accuracy of NN: 89.93617021276596 % Best: 90.17021276595744 %
Ordinary Epoch [89/100], Step [500/1128] Loss: 0.5298
Test Accuracy of NN: 90.12765957446808 % Best: 90.17021276595744 %
Ordinary Epoch [90/100], Step [500/1128] Loss: 0.3823
Test Accuracy of NN: 89.94148936170212 % Best: 90.17021276595744 %
Ordinary Epoch [91/100], Step [500/1128] Loss: 0.3432
Test Accuracy of NN: 89.2127659574468 % Best: 90.17021276595744 %
Ordinary Epoch [92/100], Step [500/1128] Loss: 0.4522
Test Accuracy of NN: 89.61702127659575 % Best: 90.17021276595744 %
Ordinary Epoch [93/100], Step [500/1128] Loss: 0.3390
Test Accuracy of NN: 90.1063829787234 % Best: 90.17021276595744 %
Ordinary Epoch [94/100], Step [500/1128] Loss: 0.3891
Test Accuracy of NN: 89.59042553191489 % Best: 90.17021276595744 %
Ordinary Epoch [95/100], Step [500/1128] Loss: 0.2102
Test Accuracy of NN: 89.68617021276596 % Best: 90.17021276595744 %
Ordinary Epoch [96/100], Step [500/1128] Loss: 0.3248
Test Accuracy of NN: 89.8936170212766 % Best: 90.17021276595744 %
Ordinary Epoch [97/100], Step [500/1128] Loss: 0.3950
Test Accuracy of NN: 90.09574468085107 % Best: 90.17021276595744 %
Ordinary Epoch [98/100], Step [500/1128] Loss: 0.3053
Test Accuracy of NN: 90.25 % (improvement)
Ordinary Epoch [99/100], Step [500/1128] Loss: 0.2372
Test Accuracy of NN: 90.09042553191489 % Best: 90.25 %
Ordinary Epoch [100/100], Step [500/1128] Loss: 0.3760
Test Accuracy of NN: 88.70744680851064 % Best: 90.25 %
