/lustre/scratch/grp/fslg_internn/env/internn/bin/python
Test Accuracy of NN: 94.76923076923077 % Best: 95.39423076923077 %
torch.Size([1000, 1, 28, 28])
max label 45
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
Ordinary Epoch [49/100], Step [500/1128] Loss: 2.2785
Ordinary Epoch [94/100], Step [500/1248] Loss: 0.2275
Ordinary Epoch [1/100], Step [500/1128] Loss: 1.0496
Test Accuracy of NN: 61.244680851063826 % Best: 62.175531914893625 %
Test Accuracy of NN: 95.125 % Best: 95.39423076923077 %
Test Accuracy of NN: 81.66489361702128 % (improvement)
Ordinary Epoch [50/100], Step [500/1128] Loss: 2.2065
Ordinary Epoch [95/100], Step [500/1248] Loss: 0.1411
Ordinary Epoch [2/100], Step [500/1128] Loss: 0.6884
Test Accuracy of NN: 62.11702127659574 % Best: 62.175531914893625 %
Test Accuracy of NN: 84.82978723404256 % (improvement)
Ordinary Epoch [51/100], Step [500/1128] Loss: 2.1121
Test Accuracy of NN: 95.36057692307692 % Best: 95.39423076923077 %
Ordinary Epoch [96/100], Step [500/1248] Loss: 0.1501
Ordinary Epoch [3/100], Step [500/1128] Loss: 0.7111
Test Accuracy of NN: 62.319148936170215 % (improvement)
Ordinary Epoch [52/100], Step [500/1128] Loss: 2.8163
Test Accuracy of NN: 95.05288461538461 % Best: 95.39423076923077 %
Test Accuracy of NN: 85.84574468085107 % (improvement)
Ordinary Epoch [97/100], Step [500/1248] Loss: 0.1473
Test Accuracy of NN: 62.11702127659574 % Best: 62.31914893617021 %
Ordinary Epoch [4/100], Step [500/1128] Loss: 0.3822
Ordinary Epoch [53/100], Step [500/1128] Loss: 2.0210
Test Accuracy of NN: 94.58653846153847 % Best: 95.39423076923077 %
/lustre/scratch/grp/fslg_internn/glom-pytorch/MNIST_vgg.py:121: DeprecationWarning: np.asscalar(a) is deprecated since NumPy v1.16, use a.item() instead
  curr_lr1 = learning_rate * np.asscalar(pow(np.random.rand(1), 3))
Test Accuracy of NN: 85.62765957446808 % Best: 85.84574468085107 %
Ordinary Epoch [98/100], Step [500/1248] Loss: 0.1761
Test Accuracy of NN: 60.76063829787234 % Best: 62.31914893617021 %
Ordinary Epoch [5/100], Step [500/1128] Loss: 0.6577
Ordinary Epoch [54/100], Step [500/1128] Loss: 2.5696
Test Accuracy of NN: 95.14903846153847 % Best: 95.39423076923077 %
Test Accuracy of NN: 87.61170212765957 % (improvement)
Test Accuracy of NN: 61.409574468085104 % Best: 62.31914893617021 %
Ordinary Epoch [99/100], Step [500/1248] Loss: 0.1962
Ordinary Epoch [55/100], Step [500/1128] Loss: 2.3023
Ordinary Epoch [6/100], Step [500/1128] Loss: 0.4617
Test Accuracy of NN: 95.19711538461539 % Best: 95.39423076923077 %
Test Accuracy of NN: 62.0531914893617 % Best: 62.31914893617021 %
Test Accuracy of NN: 87.93085106382979 % (improvement)
Ordinary Epoch [100/100], Step [500/1248] Loss: 0.1614
Ordinary Epoch [56/100], Step [500/1128] Loss: 2.0839
Ordinary Epoch [7/100], Step [500/1128] Loss: 0.4455
Test Accuracy of NN: 95.20192307692308 % Best: 95.39423076923077 %
Test Accuracy of NN: 61.81382978723404 % Best: 62.31914893617021 %
Test Accuracy of NN: 87.43085106382979 % Best: 87.93085106382978 %
Ordinary Epoch [57/100], Step [500/1128] Loss: 2.2870
Ordinary Epoch [8/100], Step [500/1128] Loss: 0.3812
Test Accuracy of NN: 62.16489361702128 % Best: 62.31914893617021 %
Test Accuracy of NN: 88.07446808510639 % (improvement)
Ordinary Epoch [58/100], Step [500/1128] Loss: 2.2500
Ordinary Epoch [9/100], Step [500/1128] Loss: 0.4751
Test Accuracy of NN: 62.159574468085104 % Best: 62.31914893617021 %
Test Accuracy of NN: 87.79255319148936 % Best: 88.07446808510639 %
Ordinary Epoch [59/100], Step [500/1128] Loss: 2.4804
Ordinary Epoch [10/100], Step [500/1128] Loss: 0.3104
Test Accuracy of NN: 62.4468085106383 % (improvement)
Test Accuracy of NN: 88.05851063829788 % Best: 88.07446808510639 %
Ordinary Epoch [60/100], Step [500/1128] Loss: 2.7364
Ordinary Epoch [11/100], Step [500/1128] Loss: 0.4075
Test Accuracy of NN: 62.43617021276596 % Best: 62.4468085106383 %
Test Accuracy of NN: 88.76063829787235 % (improvement)
Ordinary Epoch [61/100], Step [500/1128] Loss: 1.7561
Ordinary Epoch [12/100], Step [500/1128] Loss: 0.5176
Test Accuracy of NN: 62.452127659574465 % (improvement)
Test Accuracy of NN: 88.97340425531915 % (improvement)
Ordinary Epoch [62/100], Step [500/1128] Loss: 2.1183
Ordinary Epoch [13/100], Step [500/1128] Loss: 0.2418
Test Accuracy of NN: 62.3563829787234 % Best: 62.452127659574465 %
Test Accuracy of NN: 89.10106382978724 % (improvement)
Ordinary Epoch [63/100], Step [500/1128] Loss: 1.9986
Ordinary Epoch [14/100], Step [500/1128] Loss: 0.4055
Test Accuracy of NN: 62.4468085106383 % Best: 62.452127659574465 %
Test Accuracy of NN: 89.22340425531915 % (improvement)
Ordinary Epoch [64/100], Step [500/1128] Loss: 2.1309
Ordinary Epoch [15/100], Step [500/1128] Loss: 0.4558
Test Accuracy of NN: 59.16489361702128 % Best: 62.452127659574465 %
Ordinary Epoch [65/100], Step [500/1128] Loss: 2.3341
Test Accuracy of NN: 89.09042553191489 % Best: 89.22340425531915 %
Ordinary Epoch [16/100], Step [500/1128] Loss: 0.3180
Test Accuracy of NN: 61.819148936170215 % Best: 62.452127659574465 %
Ordinary Epoch [66/100], Step [500/1128] Loss: 2.4433
Test Accuracy of NN: 89.30851063829788 % (improvement)
Ordinary Epoch [17/100], Step [500/1128] Loss: 0.3314
Test Accuracy of NN: 61.361702127659576 % Best: 62.452127659574465 %
Ordinary Epoch [67/100], Step [500/1128] Loss: 2.2213
Test Accuracy of NN: 89.20744680851064 % Best: 89.30851063829788 %
Ordinary Epoch [18/100], Step [500/1128] Loss: 0.2682
Test Accuracy of NN: 62.12234042553192 % Best: 62.452127659574465 %
Ordinary Epoch [68/100], Step [500/1128] Loss: 2.0684
Test Accuracy of NN: 89.55851063829788 % (improvement)
Ordinary Epoch [19/100], Step [500/1128] Loss: 0.4658
Test Accuracy of NN: 62.191489361702125 % Best: 62.452127659574465 %
Ordinary Epoch [69/100], Step [500/1128] Loss: 2.2551
Test Accuracy of NN: 89.65957446808511 % (improvement)
Ordinary Epoch [20/100], Step [500/1128] Loss: 0.3591
Test Accuracy of NN: 59.12234042553192 % Best: 62.452127659574465 %
Ordinary Epoch [70/100], Step [500/1128] Loss: 2.0396
Test Accuracy of NN: 89.65957446808511 % Best: 89.6595744680851 %
Ordinary Epoch [21/100], Step [500/1128] Loss: 0.4594
Test Accuracy of NN: 62.159574468085104 % Best: 62.452127659574465 %
Ordinary Epoch [71/100], Step [500/1128] Loss: 2.2351
Test Accuracy of NN: 89.5372340425532 % Best: 89.6595744680851 %
Ordinary Epoch [22/100], Step [500/1128] Loss: 0.6031
Test Accuracy of NN: 62.180851063829785 % Best: 62.452127659574465 %
Ordinary Epoch [72/100], Step [500/1128] Loss: 2.2124
Test Accuracy of NN: 88.37765957446808 % Best: 89.6595744680851 %
Ordinary Epoch [23/100], Step [500/1128] Loss: 0.3346
Test Accuracy of NN: 62.090425531914896 % Best: 62.452127659574465 %
Ordinary Epoch [73/100], Step [500/1128] Loss: 2.1388
Test Accuracy of NN: 89.03191489361703 % Best: 89.6595744680851 %
Ordinary Epoch [24/100], Step [500/1128] Loss: 0.6023
Test Accuracy of NN: 62.27127659574468 % Best: 62.452127659574465 %
Ordinary Epoch [74/100], Step [500/1128] Loss: 2.1352
Test Accuracy of NN: 88.06914893617021 % Best: 89.6595744680851 %
Ordinary Epoch [25/100], Step [500/1128] Loss: 0.3419
Test Accuracy of NN: 60.75 % Best: 62.452127659574465 %
Ordinary Epoch [75/100], Step [500/1128] Loss: 2.2415
Test Accuracy of NN: 89.04255319148936 % Best: 89.6595744680851 %
Ordinary Epoch [26/100], Step [500/1128] Loss: 0.5192
Test Accuracy of NN: 62.04255319148936 % Best: 62.452127659574465 %
Ordinary Epoch [76/100], Step [500/1128] Loss: 2.3371
Test Accuracy of NN: 89.61702127659575 % Best: 89.6595744680851 %
Ordinary Epoch [27/100], Step [500/1128] Loss: 0.6880
Test Accuracy of NN: 62.111702127659576 % Best: 62.452127659574465 %
Ordinary Epoch [77/100], Step [500/1128] Loss: 2.4913
Test Accuracy of NN: 89.2127659574468 % Best: 89.6595744680851 %
Test Accuracy of NN: 62.26063829787234 % Best: 62.452127659574465 %
Ordinary Epoch [28/100], Step [500/1128] Loss: 0.3106
Ordinary Epoch [78/100], Step [500/1128] Loss: 2.2846
Test Accuracy of NN: 89.6436170212766 % Best: 89.6595744680851 %
Test Accuracy of NN: 60.54255319148936 % Best: 62.452127659574465 %
Ordinary Epoch [29/100], Step [500/1128] Loss: 0.3631
Ordinary Epoch [79/100], Step [500/1128] Loss: 2.4932
Test Accuracy of NN: 89.57446808510639 % Best: 89.6595744680851 %
Test Accuracy of NN: 62.31382978723404 % Best: 62.452127659574465 %
Ordinary Epoch [30/100], Step [500/1128] Loss: 0.2056
Ordinary Epoch [80/100], Step [500/1128] Loss: 2.3068
Test Accuracy of NN: 89.63297872340425 % Best: 89.6595744680851 %
Test Accuracy of NN: 62.6436170212766 % (improvement)
Ordinary Epoch [81/100], Step [500/1128] Loss: 2.1920
Ordinary Epoch [31/100], Step [500/1128] Loss: 0.4485
Test Accuracy of NN: 62.723404255319146 % (improvement)
Test Accuracy of NN: 88.79787234042553 % Best: 89.6595744680851 %
Ordinary Epoch [82/100], Step [500/1128] Loss: 2.3340
Ordinary Epoch [32/100], Step [500/1128] Loss: 0.3417
Test Accuracy of NN: 62.75 % (improvement)
Test Accuracy of NN: 89.47340425531915 % Best: 89.6595744680851 %
Ordinary Epoch [83/100], Step [500/1128] Loss: 2.1730
Ordinary Epoch [33/100], Step [500/1128] Loss: 0.3782
Test Accuracy of NN: 62.505319148936174 % Best: 62.74999999999999 %
Test Accuracy of NN: 89.67021276595744 % (improvement)
Ordinary Epoch [84/100], Step [500/1128] Loss: 2.3120
Ordinary Epoch [34/100], Step [500/1128] Loss: 0.3632
Test Accuracy of NN: 62.51063829787234 % Best: 62.74999999999999 %
Test Accuracy of NN: 89.75 % (improvement)
Ordinary Epoch [85/100], Step [500/1128] Loss: 2.1331
Ordinary Epoch [35/100], Step [500/1128] Loss: 0.4678
Test Accuracy of NN: 62.66489361702128 % Best: 62.74999999999999 %
Test Accuracy of NN: 89.75 % Best: 89.75 %
Ordinary Epoch [86/100], Step [500/1128] Loss: 2.7701
Ordinary Epoch [36/100], Step [500/1128] Loss: 0.4521
Test Accuracy of NN: 61.1436170212766 % Best: 62.74999999999999 %
Test Accuracy of NN: 89.72340425531915 % Best: 89.75 %
Ordinary Epoch [87/100], Step [500/1128] Loss: 2.1606
Ordinary Epoch [37/100], Step [500/1128] Loss: 0.2783
Test Accuracy of NN: 60.51063829787234 % Best: 62.74999999999999 %
Test Accuracy of NN: 89.51595744680851 % Best: 89.75 %
Ordinary Epoch [88/100], Step [500/1128] Loss: 2.3534
Ordinary Epoch [38/100], Step [500/1128] Loss: 0.4076
Test Accuracy of NN: 62.255319148936174 % Best: 62.74999999999999 %
Test Accuracy of NN: 89.63297872340425 % Best: 89.75 %
Ordinary Epoch [89/100], Step [500/1128] Loss: 2.3897
Ordinary Epoch [39/100], Step [500/1128] Loss: 0.4563
Test Accuracy of NN: 60.494680851063826 % Best: 62.74999999999999 %
Test Accuracy of NN: 87.60106382978724 % Best: 89.75 %
Ordinary Epoch [90/100], Step [500/1128] Loss: 2.3078
Ordinary Epoch [40/100], Step [500/1128] Loss: 0.3076
Test Accuracy of NN: 60.505319148936174 % Best: 62.74999999999999 %
Test Accuracy of NN: 89.25531914893617 % Best: 89.75 %
Ordinary Epoch [91/100], Step [500/1128] Loss: 2.2812
Ordinary Epoch [41/100], Step [500/1128] Loss: 0.3629
Test Accuracy of NN: 61.6063829787234 % Best: 62.74999999999999 %
Test Accuracy of NN: 89.21808510638297 % Best: 89.75 %
Ordinary Epoch [92/100], Step [500/1128] Loss: 2.5296
Ordinary Epoch [42/100], Step [500/1128] Loss: 0.2481
Test Accuracy of NN: 62.537234042553195 % Best: 62.74999999999999 %
Test Accuracy of NN: 89.28191489361703 % Best: 89.75 %
Ordinary Epoch [93/100], Step [500/1128] Loss: 2.2556
Ordinary Epoch [43/100], Step [500/1128] Loss: 0.3245
Test Accuracy of NN: 62.45744680851064 % Best: 62.74999999999999 %
Ordinary Epoch [94/100], Step [500/1128] Loss: 2.3764
Test Accuracy of NN: 89.58510638297872 % Best: 89.75 %
Ordinary Epoch [44/100], Step [500/1128] Loss: 0.2819
Test Accuracy of NN: 59.611702127659576 % Best: 62.74999999999999 %
Ordinary Epoch [95/100], Step [500/1128] Loss: 2.6038
Test Accuracy of NN: 89.63297872340425 % Best: 89.75 %
Ordinary Epoch [45/100], Step [500/1128] Loss: 0.2948
Test Accuracy of NN: 61.670212765957444 % Best: 62.74999999999999 %
Ordinary Epoch [96/100], Step [500/1128] Loss: 2.3037
Test Accuracy of NN: 89.65425531914893 % Best: 89.75 %
Ordinary Epoch [46/100], Step [500/1128] Loss: 0.4220
Test Accuracy of NN: 62.547872340425535 % Best: 62.74999999999999 %
Ordinary Epoch [97/100], Step [500/1128] Loss: 2.3139
Test Accuracy of NN: 89.54255319148936 % Best: 89.75 %
Ordinary Epoch [47/100], Step [500/1128] Loss: 0.3857
Test Accuracy of NN: 60.70744680851064 % Best: 62.74999999999999 %
Ordinary Epoch [98/100], Step [500/1128] Loss: 2.5656
Test Accuracy of NN: 89.66489361702128 % Best: 89.75 %
Ordinary Epoch [48/100], Step [500/1128] Loss: 0.4241
Test Accuracy of NN: 62.058510638297875 % Best: 62.74999999999999 %
Ordinary Epoch [99/100], Step [500/1128] Loss: 2.3949
Test Accuracy of NN: 89.76063829787235 % (improvement)
Ordinary Epoch [49/100], Step [500/1128] Loss: 0.3632
Test Accuracy of NN: 60.569148936170215 % Best: 62.74999999999999 %
Ordinary Epoch [100/100], Step [500/1128] Loss: 2.3032
Test Accuracy of NN: 89.74468085106383 % Best: 89.76063829787235 %
Ordinary Epoch [50/100], Step [500/1128] Loss: 0.4148
Test Accuracy of NN: 62.53191489361702 % Best: 62.74999999999999 %
Test Accuracy of NN: 89.72340425531915 % Best: 89.76063829787235 %
Ordinary Epoch [51/100], Step [500/1128] Loss: 0.5554
Test Accuracy of NN: 88.21808510638297 % Best: 89.76063829787235 %
Ordinary Epoch [52/100], Step [500/1128] Loss: 0.5584
Test Accuracy of NN: 89.27127659574468 % Best: 89.76063829787235 %
Ordinary Epoch [53/100], Step [500/1128] Loss: 0.2906
Test Accuracy of NN: 87.95212765957447 % Best: 89.76063829787235 %
Ordinary Epoch [54/100], Step [500/1128] Loss: 0.3956
Test Accuracy of NN: 88.98404255319149 % Best: 89.76063829787235 %
Ordinary Epoch [55/100], Step [500/1128] Loss: 0.4186
Test Accuracy of NN: 89.07446808510639 % Best: 89.76063829787235 %
Ordinary Epoch [56/100], Step [500/1128] Loss: 0.4584
Test Accuracy of NN: 89.62234042553192 % Best: 89.76063829787235 %
Ordinary Epoch [57/100], Step [500/1128] Loss: 0.1956
Test Accuracy of NN: 89.69680851063829 % Best: 89.76063829787235 %
Ordinary Epoch [58/100], Step [500/1128] Loss: 0.2223
Test Accuracy of NN: 89.69680851063829 % Best: 89.76063829787235 %
Ordinary Epoch [59/100], Step [500/1128] Loss: 0.2314
Test Accuracy of NN: 89.69680851063829 % Best: 89.76063829787235 %
Ordinary Epoch [60/100], Step [500/1128] Loss: 0.3952
Test Accuracy of NN: 89.17553191489361 % Best: 89.76063829787235 %
Ordinary Epoch [61/100], Step [500/1128] Loss: 0.3966
Test Accuracy of NN: 89.74468085106383 % Best: 89.76063829787235 %
Ordinary Epoch [62/100], Step [500/1128] Loss: 0.4027
Test Accuracy of NN: 88.97872340425532 % Best: 89.76063829787235 %
Ordinary Epoch [63/100], Step [500/1128] Loss: 0.3875
Test Accuracy of NN: 88.39893617021276 % Best: 89.76063829787235 %
Ordinary Epoch [64/100], Step [500/1128] Loss: 0.3945
Test Accuracy of NN: 89.30319148936171 % Best: 89.76063829787235 %
Ordinary Epoch [65/100], Step [500/1128] Loss: 0.3732
Test Accuracy of NN: 89.33510638297872 % Best: 89.76063829787235 %
Ordinary Epoch [66/100], Step [500/1128] Loss: 0.3938
Test Accuracy of NN: 89.31914893617021 % Best: 89.76063829787235 %
Ordinary Epoch [67/100], Step [500/1128] Loss: 0.2759
Test Accuracy of NN: 89.70212765957447 % Best: 89.76063829787235 %
Ordinary Epoch [68/100], Step [500/1128] Loss: 0.3159
Test Accuracy of NN: 89.69148936170212 % Best: 89.76063829787235 %
Ordinary Epoch [69/100], Step [500/1128] Loss: 0.7591
Test Accuracy of NN: 89.79787234042553 % (improvement)
Ordinary Epoch [70/100], Step [500/1128] Loss: 0.5397
Test Accuracy of NN: 89.81382978723404 % (improvement)
Ordinary Epoch [71/100], Step [500/1128] Loss: 0.2691
Test Accuracy of NN: 89.84574468085107 % (improvement)
Ordinary Epoch [72/100], Step [500/1128] Loss: 0.3326
Test Accuracy of NN: 89.81914893617021 % Best: 89.84574468085106 %
Ordinary Epoch [73/100], Step [500/1128] Loss: 0.3076
Test Accuracy of NN: 90.06914893617021 % (improvement)
Ordinary Epoch [74/100], Step [500/1128] Loss: 0.3067
Test Accuracy of NN: 90.13829787234043 % (improvement)
Ordinary Epoch [75/100], Step [500/1128] Loss: 0.1824
Test Accuracy of NN: 90.12234042553192 % Best: 90.13829787234042 %
Ordinary Epoch [76/100], Step [500/1128] Loss: 0.4943
Test Accuracy of NN: 89.67553191489361 % Best: 90.13829787234042 %
Ordinary Epoch [77/100], Step [500/1128] Loss: 0.2277
Test Accuracy of NN: 89.76595744680851 % Best: 90.13829787234042 %
Ordinary Epoch [78/100], Step [500/1128] Loss: 0.3015
Test Accuracy of NN: 89.70744680851064 % Best: 90.13829787234042 %
Ordinary Epoch [79/100], Step [500/1128] Loss: 0.3432
Test Accuracy of NN: 89.88829787234043 % Best: 90.13829787234042 %
Ordinary Epoch [80/100], Step [500/1128] Loss: 0.2915
Test Accuracy of NN: 89.75531914893617 % Best: 90.13829787234042 %
Ordinary Epoch [81/100], Step [500/1128] Loss: 0.3173
Test Accuracy of NN: 90.02127659574468 % Best: 90.13829787234042 %
Ordinary Epoch [82/100], Step [500/1128] Loss: 0.3045
Test Accuracy of NN: 89.99468085106383 % Best: 90.13829787234042 %
Ordinary Epoch [83/100], Step [500/1128] Loss: 0.4201
Test Accuracy of NN: 89.75 % Best: 90.13829787234042 %
Ordinary Epoch [84/100], Step [500/1128] Loss: 0.4043
Test Accuracy of NN: 89.79787234042553 % Best: 90.13829787234042 %
Ordinary Epoch [85/100], Step [500/1128] Loss: 0.5387
Test Accuracy of NN: 88.81914893617021 % Best: 90.13829787234042 %
Ordinary Epoch [86/100], Step [500/1128] Loss: 0.3104
Test Accuracy of NN: 89.88829787234043 % Best: 90.13829787234042 %
Ordinary Epoch [87/100], Step [500/1128] Loss: 0.4356
Test Accuracy of NN: 89.75 % Best: 90.13829787234042 %
Ordinary Epoch [88/100], Step [500/1128] Loss: 0.2556
Test Accuracy of NN: 90.01063829787235 % Best: 90.13829787234042 %
Ordinary Epoch [89/100], Step [500/1128] Loss: 0.3121
Test Accuracy of NN: 90.10106382978724 % Best: 90.13829787234042 %
Ordinary Epoch [90/100], Step [500/1128] Loss: 0.4578
Test Accuracy of NN: 88.9627659574468 % Best: 90.13829787234042 %
Ordinary Epoch [91/100], Step [500/1128] Loss: 0.3013
Test Accuracy of NN: 89.68617021276596 % Best: 90.13829787234042 %
Ordinary Epoch [92/100], Step [500/1128] Loss: 0.3411
Test Accuracy of NN: 88.82978723404256 % Best: 90.13829787234042 %
Ordinary Epoch [93/100], Step [500/1128] Loss: 0.4767
Test Accuracy of NN: 88.48936170212765 % Best: 90.13829787234042 %
Ordinary Epoch [94/100], Step [500/1128] Loss: 0.3336
Test Accuracy of NN: 89.81914893617021 % Best: 90.13829787234042 %
Ordinary Epoch [95/100], Step [500/1128] Loss: 0.4040
Test Accuracy of NN: 88.92021276595744 % Best: 90.13829787234042 %
Ordinary Epoch [96/100], Step [500/1128] Loss: 0.2840
Test Accuracy of NN: 89.00531914893617 % Best: 90.13829787234042 %
Ordinary Epoch [97/100], Step [500/1128] Loss: 0.2826
Test Accuracy of NN: 89.95212765957447 % Best: 90.13829787234042 %
Ordinary Epoch [98/100], Step [500/1128] Loss: 0.5417
Test Accuracy of NN: 88.15957446808511 % Best: 90.13829787234042 %
Ordinary Epoch [99/100], Step [500/1128] Loss: 0.4271
Test Accuracy of NN: 89.09042553191489 % Best: 90.13829787234042 %
Ordinary Epoch [100/100], Step [500/1128] Loss: 0.3478
Test Accuracy of NN: 89.7127659574468 % Best: 90.13829787234042 %
