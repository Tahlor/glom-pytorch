/lustre/scratch/grp/fslg_internn/env/internn/bin/python
Ordinary Epoch [36/100], Step [500/1128] Loss: 2.3194
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
Ordinary Epoch [1/100], Step [500/1128] Loss: 0.8436
Test Accuracy of NN: 60.5531914893617 % Best: 62.49468085106383 %
Ordinary Epoch [37/100], Step [500/1128] Loss: 2.1781
Test Accuracy of NN: 84.35106382978724 % (improvement)
Ordinary Epoch [2/100], Step [500/1128] Loss: 0.7198
Test Accuracy of NN: 62.526595744680854 % (improvement)
Ordinary Epoch [38/100], Step [500/1128] Loss: 2.4375
Test Accuracy of NN: 85.84042553191489 % (improvement)
Ordinary Epoch [3/100], Step [500/1128] Loss: 0.4567
Test Accuracy of NN: 62.53191489361702 % (improvement)
Test Accuracy of NN: 86.53191489361703 % (improvement)
Ordinary Epoch [39/100], Step [500/1128] Loss: 2.7258
Ordinary Epoch [4/100], Step [500/1128] Loss: 0.5802
Test Accuracy of NN: 62.70744680851064 % (improvement)
Test Accuracy of NN: 87.1063829787234 % (improvement)
Ordinary Epoch [40/100], Step [500/1128] Loss: 2.4599
Ordinary Epoch [5/100], Step [500/1128] Loss: 0.5937
Test Accuracy of NN: 62.755319148936174 % (improvement)
/lustre/scratch/grp/fslg_internn/glom-pytorch/MNIST_vgg.py:121: DeprecationWarning: np.asscalar(a) is deprecated since NumPy v1.16, use a.item() instead
  curr_lr1 = learning_rate * np.asscalar(pow(np.random.rand(1), 3))
Test Accuracy of NN: 86.93617021276596 % Best: 87.1063829787234 %
Ordinary Epoch [41/100], Step [500/1128] Loss: 2.2832
Ordinary Epoch [6/100], Step [500/1128] Loss: 0.4949
Test Accuracy of NN: 62.56382978723404 % Best: 62.755319148936174 %
Test Accuracy of NN: 88.40425531914893 % (improvement)
Ordinary Epoch [42/100], Step [500/1128] Loss: 1.9719
Ordinary Epoch [7/100], Step [500/1128] Loss: 0.3244
Test Accuracy of NN: 61.61702127659574 % Best: 62.755319148936174 %
Test Accuracy of NN: 88.62765957446808 % (improvement)
Ordinary Epoch [43/100], Step [500/1128] Loss: 2.4615
Ordinary Epoch [8/100], Step [500/1128] Loss: 0.6523
Test Accuracy of NN: 62.148936170212764 % Best: 62.755319148936174 %
Test Accuracy of NN: 88.86702127659575 % (improvement)
Ordinary Epoch [44/100], Step [500/1128] Loss: 2.2122
Ordinary Epoch [9/100], Step [500/1128] Loss: 0.4402
Test Accuracy of NN: 88.80851063829788 % Best: 88.86702127659575 %
Test Accuracy of NN: 62.5531914893617 % Best: 62.755319148936174 %
Ordinary Epoch [10/100], Step [500/1128] Loss: 0.5691
Ordinary Epoch [45/100], Step [500/1128] Loss: 2.0471
Test Accuracy of NN: 87.59574468085107 % Best: 88.86702127659575 %
Test Accuracy of NN: 62.43617021276596 % Best: 62.755319148936174 %
Ordinary Epoch [11/100], Step [500/1128] Loss: 0.6993
Ordinary Epoch [46/100], Step [500/1128] Loss: 2.5319
Test Accuracy of NN: 89.0 % (improvement)
Test Accuracy of NN: 62.61702127659574 % Best: 62.755319148936174 %
Ordinary Epoch [12/100], Step [500/1128] Loss: 0.3297
Ordinary Epoch [47/100], Step [500/1128] Loss: 2.1707
Test Accuracy of NN: 89.13297872340425 % (improvement)
Test Accuracy of NN: 62.638297872340424 % Best: 62.755319148936174 %
Ordinary Epoch [13/100], Step [500/1128] Loss: 0.5218
Ordinary Epoch [48/100], Step [500/1128] Loss: 2.2980
Test Accuracy of NN: 88.98404255319149 % Best: 89.13297872340425 %
Test Accuracy of NN: 61.808510638297875 % Best: 62.755319148936174 %
Ordinary Epoch [14/100], Step [500/1128] Loss: 0.4578
Ordinary Epoch [49/100], Step [500/1128] Loss: 2.2962
Test Accuracy of NN: 88.90957446808511 % Best: 89.13297872340425 %
Test Accuracy of NN: 61.744680851063826 % Best: 62.755319148936174 %
Ordinary Epoch [15/100], Step [500/1128] Loss: 0.3847
Ordinary Epoch [50/100], Step [500/1128] Loss: 2.5897
Test Accuracy of NN: 89.23404255319149 % (improvement)
Test Accuracy of NN: 62.579787234042556 % Best: 62.755319148936174 %
Ordinary Epoch [16/100], Step [500/1128] Loss: 0.5652
Ordinary Epoch [51/100], Step [500/1128] Loss: 2.5972
Test Accuracy of NN: 89.20212765957447 % Best: 89.23404255319149 %
Ordinary Epoch [17/100], Step [500/1128] Loss: 0.3266
Test Accuracy of NN: 60.62765957446808 % Best: 62.755319148936174 %
Ordinary Epoch [52/100], Step [500/1128] Loss: 2.4109
Test Accuracy of NN: 88.89893617021276 % Best: 89.23404255319149 %
Ordinary Epoch [18/100], Step [500/1128] Loss: 0.2258
Test Accuracy of NN: 62.058510638297875 % Best: 62.755319148936174 %
Ordinary Epoch [53/100], Step [500/1128] Loss: 2.2783
Test Accuracy of NN: 89.29787234042553 % (improvement)
Ordinary Epoch [19/100], Step [500/1128] Loss: 0.4461
Test Accuracy of NN: 62.319148936170215 % Best: 62.755319148936174 %
Test Accuracy of NN: 89.56914893617021 % (improvement)
Ordinary Epoch [54/100], Step [500/1128] Loss: 2.0425
Ordinary Epoch [20/100], Step [500/1128] Loss: 0.4520
Test Accuracy of NN: 62.015957446808514 % Best: 62.755319148936174 %
Test Accuracy of NN: 89.49468085106383 % Best: 89.56914893617022 %
Ordinary Epoch [55/100], Step [500/1128] Loss: 2.4611
Ordinary Epoch [21/100], Step [500/1128] Loss: 0.4055
Test Accuracy of NN: 62.6436170212766 % Best: 62.755319148936174 %
Test Accuracy of NN: 89.56914893617021 % Best: 89.56914893617022 %
Ordinary Epoch [56/100], Step [500/1128] Loss: 2.5202
Ordinary Epoch [22/100], Step [500/1128] Loss: 0.4401
Test Accuracy of NN: 62.66489361702128 % Best: 62.755319148936174 %
Test Accuracy of NN: 89.56382978723404 % Best: 89.56914893617022 %
Ordinary Epoch [57/100], Step [500/1128] Loss: 2.4595
Ordinary Epoch [23/100], Step [500/1128] Loss: 0.2538
Test Accuracy of NN: 62.648936170212764 % Best: 62.755319148936174 %
Test Accuracy of NN: 89.55851063829788 % Best: 89.56914893617022 %
Ordinary Epoch [58/100], Step [500/1128] Loss: 2.1208
Ordinary Epoch [24/100], Step [500/1128] Loss: 0.3839
Test Accuracy of NN: 62.1436170212766 % Best: 62.755319148936174 %
Test Accuracy of NN: 89.62765957446808 % (improvement)
Ordinary Epoch [59/100], Step [500/1128] Loss: 2.4416
Ordinary Epoch [25/100], Step [500/1128] Loss: 0.3197
Test Accuracy of NN: 62.72872340425532 % Best: 62.755319148936174 %
Test Accuracy of NN: 89.64893617021276 % (improvement)
Ordinary Epoch [60/100], Step [500/1128] Loss: 2.4697
Ordinary Epoch [26/100], Step [500/1128] Loss: 0.3306
Test Accuracy of NN: 89.60106382978724 % Best: 89.64893617021276 %
Test Accuracy of NN: 62.65425531914894 % Best: 62.755319148936174 %
Ordinary Epoch [27/100], Step [500/1128] Loss: 0.3488
Ordinary Epoch [61/100], Step [500/1128] Loss: 2.2093
Test Accuracy of NN: 89.67021276595744 % (improvement)
Test Accuracy of NN: 62.8563829787234 % (improvement)
Ordinary Epoch [28/100], Step [500/1128] Loss: 0.3087
Ordinary Epoch [62/100], Step [500/1128] Loss: 2.2801
Test Accuracy of NN: 89.61702127659575 % Best: 89.67021276595744 %
Test Accuracy of NN: 62.90425531914894 % (improvement)
Ordinary Epoch [29/100], Step [500/1128] Loss: 0.3715
Ordinary Epoch [63/100], Step [500/1128] Loss: 2.8630
Test Accuracy of NN: 89.0 % Best: 89.67021276595744 %
Test Accuracy of NN: 62.75 % Best: 62.90425531914894 %
Ordinary Epoch [30/100], Step [500/1128] Loss: 0.2962
Ordinary Epoch [64/100], Step [500/1128] Loss: 2.6744
Test Accuracy of NN: 89.59574468085107 % Best: 89.67021276595744 %
Test Accuracy of NN: 62.96808510638298 % (improvement)
Ordinary Epoch [31/100], Step [500/1128] Loss: 0.2990
Ordinary Epoch [65/100], Step [500/1128] Loss: 2.1261
Test Accuracy of NN: 88.88829787234043 % Best: 89.67021276595744 %
Test Accuracy of NN: 62.808510638297875 % Best: 62.96808510638298 %
Ordinary Epoch [32/100], Step [500/1128] Loss: 0.4340
Ordinary Epoch [66/100], Step [500/1128] Loss: 2.7798
Test Accuracy of NN: 89.69148936170212 % (improvement)
Test Accuracy of NN: 60.8563829787234 % Best: 62.96808510638298 %
Ordinary Epoch [33/100], Step [500/1128] Loss: 0.2999
Ordinary Epoch [67/100], Step [500/1128] Loss: 2.4840
Test Accuracy of NN: 89.63297872340425 % Best: 89.69148936170212 %
Ordinary Epoch [34/100], Step [500/1128] Loss: 0.4617
Test Accuracy of NN: 61.40425531914894 % Best: 62.96808510638298 %
Ordinary Epoch [68/100], Step [500/1128] Loss: 2.2433
Test Accuracy of NN: 89.7127659574468 % (improvement)
Ordinary Epoch [35/100], Step [500/1128] Loss: 0.4253
Test Accuracy of NN: 62.638297872340424 % Best: 62.96808510638298 %
Ordinary Epoch [69/100], Step [500/1128] Loss: 2.0176
Test Accuracy of NN: 89.7127659574468 % Best: 89.7127659574468 %
Ordinary Epoch [36/100], Step [500/1128] Loss: 0.5448
Test Accuracy of NN: 62.558510638297875 % Best: 62.96808510638298 %
Test Accuracy of NN: 89.75 % (improvement)
Ordinary Epoch [70/100], Step [500/1128] Loss: 2.4069
Ordinary Epoch [37/100], Step [500/1128] Loss: 0.3814
Test Accuracy of NN: 62.76063829787234 % Best: 62.96808510638298 %
Test Accuracy of NN: 89.76595744680851 % (improvement)
Ordinary Epoch [71/100], Step [500/1128] Loss: 2.4179
Ordinary Epoch [38/100], Step [500/1128] Loss: 0.3296
Test Accuracy of NN: 62.84574468085106 % Best: 62.96808510638298 %
Test Accuracy of NN: 89.80851063829788 % (improvement)
Ordinary Epoch [72/100], Step [500/1128] Loss: 1.9454
Ordinary Epoch [39/100], Step [500/1128] Loss: 0.4223
Test Accuracy of NN: 63.01063829787234 % (improvement)
Test Accuracy of NN: 89.83510638297872 % (improvement)
Ordinary Epoch [73/100], Step [500/1128] Loss: 2.4956
Ordinary Epoch [40/100], Step [500/1128] Loss: 0.3196
Test Accuracy of NN: 63.015957446808514 % (improvement)
Test Accuracy of NN: 89.82446808510639 % Best: 89.83510638297872 %
Ordinary Epoch [74/100], Step [500/1128] Loss: 2.4608
Ordinary Epoch [41/100], Step [500/1128] Loss: 0.3788
Test Accuracy of NN: 62.973404255319146 % Best: 63.015957446808514 %
Test Accuracy of NN: 89.80851063829788 % Best: 89.83510638297872 %
Ordinary Epoch [75/100], Step [500/1128] Loss: 2.2618
Ordinary Epoch [42/100], Step [500/1128] Loss: 0.2170
Test Accuracy of NN: 62.797872340425535 % Best: 63.015957446808514 %
Test Accuracy of NN: 89.85106382978724 % (improvement)
Ordinary Epoch [43/100], Step [500/1128] Loss: 0.3004
Ordinary Epoch [76/100], Step [500/1128] Loss: 2.2554
Test Accuracy of NN: 89.7127659574468 % Best: 89.85106382978724 %
Test Accuracy of NN: 62.81382978723404 % Best: 63.015957446808514 %
Ordinary Epoch [44/100], Step [500/1128] Loss: 0.3754
Ordinary Epoch [77/100], Step [500/1128] Loss: 2.0279
Test Accuracy of NN: 89.55319148936171 % Best: 89.85106382978724 %
Test Accuracy of NN: 61.96808510638298 % Best: 63.015957446808514 %
Ordinary Epoch [45/100], Step [500/1128] Loss: 0.4255
Ordinary Epoch [78/100], Step [500/1128] Loss: 2.2201
Test Accuracy of NN: 89.75 % Best: 89.85106382978724 %
Test Accuracy of NN: 62.20744680851064 % Best: 63.015957446808514 %
Ordinary Epoch [46/100], Step [500/1128] Loss: 0.4534
Ordinary Epoch [79/100], Step [500/1128] Loss: 2.2209
Test Accuracy of NN: 89.71808510638297 % Best: 89.85106382978724 %
Test Accuracy of NN: 61.9468085106383 % Best: 63.015957446808514 %
Ordinary Epoch [47/100], Step [500/1128] Loss: 0.2612
Ordinary Epoch [80/100], Step [500/1128] Loss: 2.3486
Test Accuracy of NN: 89.85106382978724 % Best: 89.85106382978724 %
Test Accuracy of NN: 62.702127659574465 % Best: 63.015957446808514 %
Ordinary Epoch [48/100], Step [500/1128] Loss: 0.2039
Ordinary Epoch [81/100], Step [500/1128] Loss: 2.0915
Test Accuracy of NN: 89.82978723404256 % Best: 89.85106382978724 %
Ordinary Epoch [49/100], Step [500/1128] Loss: 0.2796
Test Accuracy of NN: 62.829787234042556 % Best: 63.015957446808514 %
Ordinary Epoch [82/100], Step [500/1128] Loss: 2.0639
Test Accuracy of NN: 89.53191489361703 % Best: 89.85106382978724 %
Ordinary Epoch [50/100], Step [500/1128] Loss: 0.4749
Test Accuracy of NN: 62.91489361702128 % Best: 63.015957446808514 %
Test Accuracy of NN: 89.81914893617021 % Best: 89.85106382978724 %
Ordinary Epoch [83/100], Step [500/1128] Loss: 2.5300
Ordinary Epoch [51/100], Step [500/1128] Loss: 0.3840
Test Accuracy of NN: 62.88297872340426 % Best: 63.015957446808514 %
Test Accuracy of NN: 89.55851063829788 % Best: 89.85106382978724 %
Ordinary Epoch [84/100], Step [500/1128] Loss: 2.2467
Ordinary Epoch [52/100], Step [500/1128] Loss: 0.3102
Test Accuracy of NN: 62.79255319148936 % Best: 63.015957446808514 %
Test Accuracy of NN: 89.72872340425532 % Best: 89.85106382978724 %
Ordinary Epoch [85/100], Step [500/1128] Loss: 2.1350
Ordinary Epoch [53/100], Step [500/1128] Loss: 0.2451
Test Accuracy of NN: 62.941489361702125 % Best: 63.015957446808514 %
Test Accuracy of NN: 89.88829787234043 % (improvement)
Ordinary Epoch [86/100], Step [500/1128] Loss: 2.3707
Ordinary Epoch [54/100], Step [500/1128] Loss: 0.3138
Test Accuracy of NN: 62.6968085106383 % Best: 63.015957446808514 %
Test Accuracy of NN: 89.90957446808511 % (improvement)
Ordinary Epoch [87/100], Step [500/1128] Loss: 2.2571
Ordinary Epoch [55/100], Step [500/1128] Loss: 0.2400
Test Accuracy of NN: 62.079787234042556 % Best: 63.015957446808514 %
Test Accuracy of NN: 89.8563829787234 % Best: 89.90957446808511 %
Ordinary Epoch [56/100], Step [500/1128] Loss: 0.3257
Ordinary Epoch [88/100], Step [500/1128] Loss: 2.4312
Test Accuracy of NN: 89.6063829787234 % Best: 89.90957446808511 %
Test Accuracy of NN: 62.765957446808514 % Best: 63.015957446808514 %
Ordinary Epoch [57/100], Step [500/1128] Loss: 0.5174
Ordinary Epoch [89/100], Step [500/1128] Loss: 2.2920
Test Accuracy of NN: 88.91489361702128 % Best: 89.90957446808511 %
Test Accuracy of NN: 60.276595744680854 % Best: 63.015957446808514 %
Ordinary Epoch [58/100], Step [500/1128] Loss: 0.4463
Ordinary Epoch [90/100], Step [500/1128] Loss: 2.6157
Test Accuracy of NN: 88.98404255319149 % Best: 89.90957446808511 %
Test Accuracy of NN: 60.340425531914896 % Best: 63.015957446808514 %
Ordinary Epoch [59/100], Step [500/1128] Loss: 0.3824
Ordinary Epoch [91/100], Step [500/1128] Loss: 2.1869
Test Accuracy of NN: 89.47872340425532 % Best: 89.90957446808511 %
Test Accuracy of NN: 62.787234042553195 % Best: 63.015957446808514 %
Ordinary Epoch [60/100], Step [500/1128] Loss: 0.2657
Ordinary Epoch [92/100], Step [500/1128] Loss: 2.7506
Test Accuracy of NN: 89.70744680851064 % Best: 89.90957446808511 %
Ordinary Epoch [61/100], Step [500/1128] Loss: 0.4110
Test Accuracy of NN: 61.191489361702125 % Best: 63.015957446808514 %
Ordinary Epoch [93/100], Step [500/1128] Loss: 2.0749
Test Accuracy of NN: 89.09574468085107 % Best: 89.90957446808511 %
Ordinary Epoch [62/100], Step [500/1128] Loss: 0.5242
Test Accuracy of NN: 61.6968085106383 % Best: 63.015957446808514 %
Test Accuracy of NN: 89.15425531914893 % Best: 89.90957446808511 %
Ordinary Epoch [94/100], Step [500/1128] Loss: 2.0166
Ordinary Epoch [63/100], Step [500/1128] Loss: 0.4545
Test Accuracy of NN: 61.484042553191486 % Best: 63.015957446808514 %
Test Accuracy of NN: 89.70744680851064 % Best: 89.90957446808511 %
Ordinary Epoch [95/100], Step [500/1128] Loss: 2.3514
Ordinary Epoch [64/100], Step [500/1128] Loss: 0.3798
Test Accuracy of NN: 60.57446808510638 % Best: 63.015957446808514 %
Test Accuracy of NN: 88.82978723404256 % Best: 89.90957446808511 %
Ordinary Epoch [96/100], Step [500/1128] Loss: 2.4513
Ordinary Epoch [65/100], Step [500/1128] Loss: 0.3198
Test Accuracy of NN: 61.3031914893617 % Best: 63.015957446808514 %
Test Accuracy of NN: 89.54255319148936 % Best: 89.90957446808511 %
Ordinary Epoch [97/100], Step [500/1128] Loss: 2.1376
Ordinary Epoch [66/100], Step [500/1128] Loss: 0.2570
Test Accuracy of NN: 62.462765957446805 % Best: 63.015957446808514 %
Test Accuracy of NN: 89.73936170212765 % Best: 89.90957446808511 %
Ordinary Epoch [98/100], Step [500/1128] Loss: 2.3551
Ordinary Epoch [67/100], Step [500/1128] Loss: 0.3633
Test Accuracy of NN: 89.69680851063829 % Best: 89.90957446808511 %
Test Accuracy of NN: 62.659574468085104 % Best: 63.015957446808514 %
Ordinary Epoch [68/100], Step [500/1128] Loss: 0.4230
Ordinary Epoch [99/100], Step [500/1128] Loss: 2.4044
Test Accuracy of NN: 89.4627659574468 % Best: 89.90957446808511 %
Test Accuracy of NN: 62.441489361702125 % Best: 63.015957446808514 %
Ordinary Epoch [69/100], Step [500/1128] Loss: 0.4185
Ordinary Epoch [100/100], Step [500/1128] Loss: 2.0268
Test Accuracy of NN: 89.45744680851064 % Best: 89.90957446808511 %
Test Accuracy of NN: 62.484042553191486 % Best: 63.015957446808514 %
Ordinary Epoch [70/100], Step [500/1128] Loss: 0.2667
Test Accuracy of NN: 89.67021276595744 % Best: 89.90957446808511 %
Ordinary Epoch [71/100], Step [500/1128] Loss: 0.3970
Test Accuracy of NN: 89.75531914893617 % Best: 89.90957446808511 %
Ordinary Epoch [72/100], Step [500/1128] Loss: 0.2910
Test Accuracy of NN: 88.76595744680851 % Best: 89.90957446808511 %
Ordinary Epoch [73/100], Step [500/1128] Loss: 0.3733
Test Accuracy of NN: 88.06382978723404 % Best: 89.90957446808511 %
Ordinary Epoch [74/100], Step [500/1128] Loss: 0.3603
Test Accuracy of NN: 89.17553191489361 % Best: 89.90957446808511 %
Ordinary Epoch [75/100], Step [500/1128] Loss: 0.5860
Test Accuracy of NN: 88.57446808510639 % Best: 89.90957446808511 %
Ordinary Epoch [76/100], Step [500/1128] Loss: 0.3850
Test Accuracy of NN: 89.07446808510639 % Best: 89.90957446808511 %
Ordinary Epoch [77/100], Step [500/1128] Loss: 0.3850
Test Accuracy of NN: 89.3563829787234 % Best: 89.90957446808511 %
Ordinary Epoch [78/100], Step [500/1128] Loss: 0.5281
Test Accuracy of NN: 89.27127659574468 % Best: 89.90957446808511 %
Ordinary Epoch [79/100], Step [500/1128] Loss: 0.4258
Test Accuracy of NN: 89.50531914893617 % Best: 89.90957446808511 %
Ordinary Epoch [80/100], Step [500/1128] Loss: 0.4028
Test Accuracy of NN: 89.5 % Best: 89.90957446808511 %
Ordinary Epoch [81/100], Step [500/1128] Loss: 0.4777
Test Accuracy of NN: 89.61702127659575 % Best: 89.90957446808511 %
Ordinary Epoch [82/100], Step [500/1128] Loss: 0.4480
Test Accuracy of NN: 89.7127659574468 % Best: 89.90957446808511 %
Ordinary Epoch [83/100], Step [500/1128] Loss: 0.2949
Test Accuracy of NN: 89.75531914893617 % Best: 89.90957446808511 %
Ordinary Epoch [84/100], Step [500/1128] Loss: 0.4355
Test Accuracy of NN: 89.15425531914893 % Best: 89.90957446808511 %
Ordinary Epoch [85/100], Step [500/1128] Loss: 0.5624
Test Accuracy of NN: 89.6436170212766 % Best: 89.90957446808511 %
Ordinary Epoch [86/100], Step [500/1128] Loss: 0.4675
Test Accuracy of NN: 89.76595744680851 % Best: 89.90957446808511 %
Ordinary Epoch [87/100], Step [500/1128] Loss: 0.2506
Test Accuracy of NN: 89.44680851063829 % Best: 89.90957446808511 %
Ordinary Epoch [88/100], Step [500/1128] Loss: 0.4622
Test Accuracy of NN: 89.87765957446808 % Best: 89.90957446808511 %
Ordinary Epoch [89/100], Step [500/1128] Loss: 0.4153
Test Accuracy of NN: 89.88297872340425 % Best: 89.90957446808511 %
Ordinary Epoch [90/100], Step [500/1128] Loss: 0.3177
Test Accuracy of NN: 89.92553191489361 % (improvement)
Ordinary Epoch [91/100], Step [500/1128] Loss: 0.2438
Test Accuracy of NN: 89.90425531914893 % Best: 89.92553191489363 %
Ordinary Epoch [92/100], Step [500/1128] Loss: 0.5298
Test Accuracy of NN: 89.16489361702128 % Best: 89.92553191489363 %
Ordinary Epoch [93/100], Step [500/1128] Loss: 0.3665
Test Accuracy of NN: 89.69148936170212 % Best: 89.92553191489363 %
Ordinary Epoch [94/100], Step [500/1128] Loss: 0.2604
Test Accuracy of NN: 89.89893617021276 % Best: 89.92553191489363 %
Ordinary Epoch [95/100], Step [500/1128] Loss: 0.3593
Test Accuracy of NN: 89.97872340425532 % (improvement)
Ordinary Epoch [96/100], Step [500/1128] Loss: 0.4061
Test Accuracy of NN: 89.94680851063829 % Best: 89.97872340425532 %
Ordinary Epoch [97/100], Step [500/1128] Loss: 0.2999
Test Accuracy of NN: 89.76063829787235 % Best: 89.97872340425532 %
Ordinary Epoch [98/100], Step [500/1128] Loss: 0.1928
Test Accuracy of NN: 89.69680851063829 % Best: 89.97872340425532 %
Ordinary Epoch [99/100], Step [500/1128] Loss: 0.2949
Test Accuracy of NN: 89.72872340425532 % Best: 89.97872340425532 %
Ordinary Epoch [100/100], Step [500/1128] Loss: 0.2287
Test Accuracy of NN: 89.37765957446808 % Best: 89.97872340425532 %
