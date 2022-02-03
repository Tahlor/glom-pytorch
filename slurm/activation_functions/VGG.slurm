/lustre/scratch/grp/fslg_internn/env/internn/bin/python
torch.Size([1000, 1, 28, 28])
max label 26
<class 'models.VGG'>
Parameters 3654875
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
    (5): Linear(in_features=512, out_features=27, bias=True)
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
    (5): Linear(in_features=512, out_features=27, bias=True)
  )
)>
Ordinary Epoch [1/100], Step [500/1248] Loss: 0.6203
Test Accuracy of NN: 91.96153846153847 % (improvement)
Ordinary Epoch [2/100], Step [500/1248] Loss: 0.4725
Test Accuracy of NN: 93.00961538461539 % (improvement)
Ordinary Epoch [3/100], Step [500/1248] Loss: 0.4719
Test Accuracy of NN: 93.38942307692308 % (improvement)
Ordinary Epoch [4/100], Step [500/1248] Loss: 0.1599
Test Accuracy of NN: 93.92307692307692 % (improvement)
Ordinary Epoch [5/100], Step [500/1248] Loss: 0.2137
Test Accuracy of NN: 94.53846153846153 % (improvement)
Ordinary Epoch [6/100], Step [500/1248] Loss: 0.2094
/lustre/scratch/grp/fslg_internn/glom-pytorch/MNIST_vgg.py:121: DeprecationWarning: np.asscalar(a) is deprecated since NumPy v1.16, use a.item() instead
  curr_lr1 = learning_rate * np.asscalar(pow(np.random.rand(1), 3))
Test Accuracy of NN: 94.3125 % Best: 94.53846153846153 %
Ordinary Epoch [7/100], Step [500/1248] Loss: 0.2268
Test Accuracy of NN: 95.2548076923077 % (improvement)
Ordinary Epoch [8/100], Step [500/1248] Loss: 0.1390
Test Accuracy of NN: 95.36057692307692 % (improvement)
Ordinary Epoch [9/100], Step [500/1248] Loss: 0.1823
Test Accuracy of NN: 95.3798076923077 % (improvement)
Ordinary Epoch [10/100], Step [500/1248] Loss: 0.1487
Test Accuracy of NN: 95.36057692307692 % Best: 95.3798076923077 %
Ordinary Epoch [11/100], Step [500/1248] Loss: 0.1105
Test Accuracy of NN: 95.66346153846153 % (improvement)
Ordinary Epoch [12/100], Step [500/1248] Loss: 0.1219
Test Accuracy of NN: 95.51923076923077 % Best: 95.66346153846153 %
Ordinary Epoch [13/100], Step [500/1248] Loss: 0.1087
Test Accuracy of NN: 95.60096153846153 % Best: 95.66346153846153 %
Ordinary Epoch [14/100], Step [500/1248] Loss: 0.0761
Test Accuracy of NN: 94.99038461538461 % Best: 95.66346153846153 %
Ordinary Epoch [15/100], Step [500/1248] Loss: 0.3066
Test Accuracy of NN: 95.5576923076923 % Best: 95.66346153846153 %
Ordinary Epoch [16/100], Step [500/1248] Loss: 0.2239
Test Accuracy of NN: 95.3125 % Best: 95.66346153846153 %
Ordinary Epoch [17/100], Step [500/1248] Loss: 0.1348
Test Accuracy of NN: 95.39903846153847 % Best: 95.66346153846153 %
Ordinary Epoch [18/100], Step [500/1248] Loss: 0.1181
Test Accuracy of NN: 95.66826923076923 % (improvement)
Ordinary Epoch [19/100], Step [500/1248] Loss: 0.1516
Test Accuracy of NN: 95.64903846153847 % Best: 95.66826923076923 %
Ordinary Epoch [20/100], Step [500/1248] Loss: 0.0766
Test Accuracy of NN: 95.50961538461539 % Best: 95.66826923076923 %
Ordinary Epoch [21/100], Step [500/1248] Loss: 0.2843
Test Accuracy of NN: 95.61057692307692 % Best: 95.66826923076923 %
Ordinary Epoch [22/100], Step [500/1248] Loss: 0.1651
Test Accuracy of NN: 95.48076923076923 % Best: 95.66826923076923 %
Ordinary Epoch [23/100], Step [500/1248] Loss: 0.1729
Test Accuracy of NN: 95.6201923076923 % Best: 95.66826923076923 %
Ordinary Epoch [24/100], Step [500/1248] Loss: 0.0599
Test Accuracy of NN: 95.73076923076923 % (improvement)
Ordinary Epoch [25/100], Step [500/1248] Loss: 0.1012
Test Accuracy of NN: 95.67307692307692 % Best: 95.73076923076923 %
Ordinary Epoch [26/100], Step [500/1248] Loss: 0.0792
Test Accuracy of NN: 95.73076923076923 % Best: 95.73076923076923 %
Ordinary Epoch [27/100], Step [500/1248] Loss: 0.1438
Test Accuracy of NN: 95.29326923076923 % Best: 95.73076923076923 %
Ordinary Epoch [28/100], Step [500/1248] Loss: 0.0754
Test Accuracy of NN: 95.75 % (improvement)
Ordinary Epoch [29/100], Step [500/1248] Loss: 0.1305
Test Accuracy of NN: 95.6923076923077 % Best: 95.75 %
Ordinary Epoch [30/100], Step [500/1248] Loss: 0.1972
Test Accuracy of NN: 94.96153846153847 % Best: 95.75 %
Ordinary Epoch [31/100], Step [500/1248] Loss: 0.1514
Test Accuracy of NN: 95.64903846153847 % Best: 95.75 %
Ordinary Epoch [32/100], Step [500/1248] Loss: 0.1163
Test Accuracy of NN: 95.5673076923077 % Best: 95.75 %
Ordinary Epoch [33/100], Step [500/1248] Loss: 0.0710
Test Accuracy of NN: 95.76923076923077 % (improvement)
Ordinary Epoch [34/100], Step [500/1248] Loss: 0.1504
Test Accuracy of NN: 95.73076923076923 % Best: 95.76923076923077 %
Ordinary Epoch [35/100], Step [500/1248] Loss: 0.0675
Test Accuracy of NN: 95.79326923076923 % (improvement)
Ordinary Epoch [36/100], Step [500/1248] Loss: 0.1662
Test Accuracy of NN: 95.78846153846153 % Best: 95.79326923076923 %
Ordinary Epoch [37/100], Step [500/1248] Loss: 0.1220
Test Accuracy of NN: 95.1201923076923 % Best: 95.79326923076923 %
Ordinary Epoch [38/100], Step [500/1248] Loss: 0.0774
Test Accuracy of NN: 95.47115384615384 % Best: 95.79326923076923 %
Ordinary Epoch [39/100], Step [500/1248] Loss: 0.0963
Test Accuracy of NN: 95.57692307692308 % Best: 95.79326923076923 %
Ordinary Epoch [40/100], Step [500/1248] Loss: 0.1493
Test Accuracy of NN: 95.63461538461539 % Best: 95.79326923076923 %
Ordinary Epoch [41/100], Step [500/1248] Loss: 0.2072
Test Accuracy of NN: 95.63942307692308 % Best: 95.79326923076923 %
Ordinary Epoch [42/100], Step [500/1248] Loss: 0.1407
Test Accuracy of NN: 95.67788461538461 % Best: 95.79326923076923 %
Ordinary Epoch [43/100], Step [500/1248] Loss: 0.2116
Test Accuracy of NN: 95.66826923076923 % Best: 95.79326923076923 %
Ordinary Epoch [44/100], Step [500/1248] Loss: 0.2140
Test Accuracy of NN: 95.64423076923077 % Best: 95.79326923076923 %
Ordinary Epoch [45/100], Step [500/1248] Loss: 0.1143
Test Accuracy of NN: 95.71634615384616 % Best: 95.79326923076923 %
Ordinary Epoch [46/100], Step [500/1248] Loss: 0.1809
Test Accuracy of NN: 95.41826923076923 % Best: 95.79326923076923 %
Ordinary Epoch [47/100], Step [500/1248] Loss: 0.1164
Test Accuracy of NN: 95.71153846153847 % Best: 95.79326923076923 %
Ordinary Epoch [48/100], Step [500/1248] Loss: 0.1674
Test Accuracy of NN: 95.61057692307692 % Best: 95.79326923076923 %
Ordinary Epoch [49/100], Step [500/1248] Loss: 0.3020
Test Accuracy of NN: 95.67788461538461 % Best: 95.79326923076923 %
Ordinary Epoch [50/100], Step [500/1248] Loss: 0.0563
Test Accuracy of NN: 95.67788461538461 % Best: 95.79326923076923 %
Ordinary Epoch [51/100], Step [500/1248] Loss: 0.0518
Test Accuracy of NN: 95.6923076923077 % Best: 95.79326923076923 %
Ordinary Epoch [52/100], Step [500/1248] Loss: 0.0878
Test Accuracy of NN: 95.66346153846153 % Best: 95.79326923076923 %
Ordinary Epoch [53/100], Step [500/1248] Loss: 0.1585
Test Accuracy of NN: 95.34134615384616 % Best: 95.79326923076923 %
Ordinary Epoch [54/100], Step [500/1248] Loss: 0.1448
Test Accuracy of NN: 95.42788461538461 % Best: 95.79326923076923 %
Ordinary Epoch [55/100], Step [500/1248] Loss: 0.0994
Test Accuracy of NN: 95.33653846153847 % Best: 95.79326923076923 %
Ordinary Epoch [56/100], Step [500/1248] Loss: 0.0968
Test Accuracy of NN: 95.42788461538461 % Best: 95.79326923076923 %
Ordinary Epoch [57/100], Step [500/1248] Loss: 0.1458
Test Accuracy of NN: 95.38461538461539 % Best: 95.79326923076923 %
Ordinary Epoch [58/100], Step [500/1248] Loss: 0.0959
Test Accuracy of NN: 95.4375 % Best: 95.79326923076923 %
Ordinary Epoch [59/100], Step [500/1248] Loss: 0.1436
Test Accuracy of NN: 95.58173076923077 % Best: 95.79326923076923 %
Ordinary Epoch [60/100], Step [500/1248] Loss: 0.0534
Test Accuracy of NN: 95.79807692307692 % (improvement)
Ordinary Epoch [61/100], Step [500/1248] Loss: 0.0811
Test Accuracy of NN: 95.84134615384616 % (improvement)
Ordinary Epoch [62/100], Step [500/1248] Loss: 0.0545
Test Accuracy of NN: 95.84615384615384 % (improvement)
Ordinary Epoch [63/100], Step [500/1248] Loss: 0.0549
Test Accuracy of NN: 95.7451923076923 % Best: 95.84615384615385 %
Ordinary Epoch [64/100], Step [500/1248] Loss: 0.1908
Test Accuracy of NN: 95.375 % Best: 95.84615384615385 %
Ordinary Epoch [65/100], Step [500/1248] Loss: 0.1205
Test Accuracy of NN: 94.98076923076923 % Best: 95.84615384615385 %
Ordinary Epoch [66/100], Step [500/1248] Loss: 0.1316
Test Accuracy of NN: 95.65384615384616 % Best: 95.84615384615385 %
Ordinary Epoch [67/100], Step [500/1248] Loss: 0.0872
Test Accuracy of NN: 95.66826923076923 % Best: 95.84615384615385 %
Ordinary Epoch [68/100], Step [500/1248] Loss: 0.1030
Test Accuracy of NN: 95.75961538461539 % Best: 95.84615384615385 %
Ordinary Epoch [69/100], Step [500/1248] Loss: 0.0388
Test Accuracy of NN: 95.22115384615384 % Best: 95.84615384615385 %
Ordinary Epoch [70/100], Step [500/1248] Loss: 0.1195
Test Accuracy of NN: 95.44711538461539 % Best: 95.84615384615385 %
Ordinary Epoch [71/100], Step [500/1248] Loss: 0.0524
Test Accuracy of NN: 95.125 % Best: 95.84615384615385 %
Ordinary Epoch [72/100], Step [500/1248] Loss: 0.1904
Test Accuracy of NN: 95.71634615384616 % Best: 95.84615384615385 %
Ordinary Epoch [73/100], Step [500/1248] Loss: 0.1017
Test Accuracy of NN: 95.73076923076923 % Best: 95.84615384615385 %
Ordinary Epoch [74/100], Step [500/1248] Loss: 0.3513
Test Accuracy of NN: 95.7451923076923 % Best: 95.84615384615385 %
Ordinary Epoch [75/100], Step [500/1248] Loss: 0.1802
Test Accuracy of NN: 95.83173076923077 % Best: 95.84615384615385 %
Ordinary Epoch [76/100], Step [500/1248] Loss: 0.0867
Test Accuracy of NN: 95.83653846153847 % Best: 95.84615384615385 %
Ordinary Epoch [77/100], Step [500/1248] Loss: 0.1137
Test Accuracy of NN: 95.19711538461539 % Best: 95.84615384615385 %
Ordinary Epoch [78/100], Step [500/1248] Loss: 0.2953
Test Accuracy of NN: 95.52403846153847 % Best: 95.84615384615385 %
Ordinary Epoch [79/100], Step [500/1248] Loss: 0.0454
Test Accuracy of NN: 95.8125 % Best: 95.84615384615385 %
Ordinary Epoch [80/100], Step [500/1248] Loss: 0.0931
Test Accuracy of NN: 95.88461538461539 % (improvement)
Ordinary Epoch [81/100], Step [500/1248] Loss: 0.3705
Test Accuracy of NN: 95.83653846153847 % Best: 95.88461538461537 %
Ordinary Epoch [82/100], Step [500/1248] Loss: 0.1245
Test Accuracy of NN: 95.8798076923077 % Best: 95.88461538461537 %
Ordinary Epoch [83/100], Step [500/1248] Loss: 0.1277
Test Accuracy of NN: 95.71153846153847 % Best: 95.88461538461537 %
Ordinary Epoch [84/100], Step [500/1248] Loss: 0.0632
Test Accuracy of NN: 95.84134615384616 % Best: 95.88461538461537 %
Ordinary Epoch [85/100], Step [500/1248] Loss: 0.1317
Test Accuracy of NN: 95.86538461538461 % Best: 95.88461538461537 %
Ordinary Epoch [86/100], Step [500/1248] Loss: 0.1053
Test Accuracy of NN: 95.875 % Best: 95.88461538461537 %
Ordinary Epoch [87/100], Step [500/1248] Loss: 0.1331
Test Accuracy of NN: 95.89423076923077 % (improvement)
Ordinary Epoch [88/100], Step [500/1248] Loss: 0.1068
Test Accuracy of NN: 95.85576923076923 % Best: 95.89423076923077 %
Ordinary Epoch [89/100], Step [500/1248] Loss: 0.2631
Test Accuracy of NN: 95.45673076923077 % Best: 95.89423076923077 %
Ordinary Epoch [90/100], Step [500/1248] Loss: 0.1139
Test Accuracy of NN: 95.77884615384616 % Best: 95.89423076923077 %
Ordinary Epoch [91/100], Step [500/1248] Loss: 0.0734
Test Accuracy of NN: 95.8076923076923 % Best: 95.89423076923077 %
Ordinary Epoch [92/100], Step [500/1248] Loss: 0.2463
Test Accuracy of NN: 95.85576923076923 % Best: 95.89423076923077 %
Ordinary Epoch [93/100], Step [500/1248] Loss: 0.0724
Test Accuracy of NN: 95.875 % Best: 95.89423076923077 %
Ordinary Epoch [94/100], Step [500/1248] Loss: 0.0997
Test Accuracy of NN: 95.86057692307692 % Best: 95.89423076923077 %
Ordinary Epoch [95/100], Step [500/1248] Loss: 0.1573
Test Accuracy of NN: 95.46153846153847 % Best: 95.89423076923077 %
Ordinary Epoch [96/100], Step [500/1248] Loss: 0.0581
Test Accuracy of NN: 95.75961538461539 % Best: 95.89423076923077 %
Ordinary Epoch [97/100], Step [500/1248] Loss: 0.0873
Test Accuracy of NN: 95.78846153846153 % Best: 95.89423076923077 %
Ordinary Epoch [98/100], Step [500/1248] Loss: 0.1396
Test Accuracy of NN: 95.83653846153847 % Best: 95.89423076923077 %
Ordinary Epoch [99/100], Step [500/1248] Loss: 0.1601
Test Accuracy of NN: 95.89903846153847 % (improvement)
Ordinary Epoch [100/100], Step [500/1248] Loss: 0.0461
Test Accuracy of NN: 95.89423076923077 % Best: 95.89903846153847 %
