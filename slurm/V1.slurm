/lustre/scratch/grp/fslg_internn/env/internn/bin/python
torch.Size([1000, 1, 28, 28])
max label 26
<class 'models.V1'>
Parameters 1107611
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
    (4): Linear(in_features=512, out_features=27, bias=True)
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
    (4): Linear(in_features=512, out_features=27, bias=True)
  )
)>
Ordinary Epoch [1/100], Step [500/1248] Loss: 0.6600
Test Accuracy of NN: 91.94711538461539 % (improvement)
Ordinary Epoch [2/100], Step [500/1248] Loss: 0.5004
Test Accuracy of NN: 93.15384615384616 % (improvement)
Ordinary Epoch [3/100], Step [500/1248] Loss: 0.4467
Test Accuracy of NN: 93.2548076923077 % (improvement)
Ordinary Epoch [4/100], Step [500/1248] Loss: 0.2802
Test Accuracy of NN: 93.5 % (improvement)
Ordinary Epoch [5/100], Step [500/1248] Loss: 0.2986
Test Accuracy of NN: 93.86538461538461 % (improvement)
Ordinary Epoch [6/100], Step [500/1248] Loss: 0.2991
/lustre/scratch/grp/fslg_internn/glom-pytorch/MNIST_vgg.py:121: DeprecationWarning: np.asscalar(a) is deprecated since NumPy v1.16, use a.item() instead
  curr_lr1 = learning_rate * np.asscalar(pow(np.random.rand(1), 3))
Test Accuracy of NN: 93.4326923076923 % Best: 93.86538461538461 %
Ordinary Epoch [7/100], Step [500/1248] Loss: 0.2693
Test Accuracy of NN: 94.36057692307692 % (improvement)
Ordinary Epoch [8/100], Step [500/1248] Loss: 0.2151
Test Accuracy of NN: 94.42788461538461 % (improvement)
Ordinary Epoch [9/100], Step [500/1248] Loss: 0.3018
Test Accuracy of NN: 94.47115384615384 % (improvement)
Ordinary Epoch [10/100], Step [500/1248] Loss: 0.1953
Test Accuracy of NN: 94.52884615384616 % (improvement)
Ordinary Epoch [11/100], Step [500/1248] Loss: 0.2902
Test Accuracy of NN: 94.58653846153847 % (improvement)
Ordinary Epoch [12/100], Step [500/1248] Loss: 0.3475
Test Accuracy of NN: 94.61538461538461 % (improvement)
Ordinary Epoch [13/100], Step [500/1248] Loss: 0.3191
Test Accuracy of NN: 94.63942307692308 % (improvement)
Ordinary Epoch [14/100], Step [500/1248] Loss: 0.2396
Test Accuracy of NN: 94.6875 % (improvement)
Ordinary Epoch [15/100], Step [500/1248] Loss: 0.2025
Test Accuracy of NN: 94.72596153846153 % (improvement)
Ordinary Epoch [16/100], Step [500/1248] Loss: 0.2567
Test Accuracy of NN: 94.67788461538461 % Best: 94.72596153846153 %
Ordinary Epoch [17/100], Step [500/1248] Loss: 0.2934
Test Accuracy of NN: 94.64903846153847 % Best: 94.72596153846153 %
Ordinary Epoch [18/100], Step [500/1248] Loss: 0.2348
Test Accuracy of NN: 94.88461538461539 % (improvement)
Ordinary Epoch [19/100], Step [500/1248] Loss: 0.1992
Test Accuracy of NN: 95.04807692307692 % (improvement)
Ordinary Epoch [20/100], Step [500/1248] Loss: 0.2165
Test Accuracy of NN: 94.99038461538461 % Best: 95.04807692307692 %
Ordinary Epoch [21/100], Step [500/1248] Loss: 0.3236
Test Accuracy of NN: 95.01442307692308 % Best: 95.04807692307692 %
Ordinary Epoch [22/100], Step [500/1248] Loss: 0.2606
Test Accuracy of NN: 95.02403846153847 % Best: 95.04807692307692 %
Ordinary Epoch [23/100], Step [500/1248] Loss: 0.1430
Test Accuracy of NN: 94.75 % Best: 95.04807692307692 %
Ordinary Epoch [24/100], Step [500/1248] Loss: 0.1295
Test Accuracy of NN: 95.03365384615384 % Best: 95.04807692307692 %
Ordinary Epoch [25/100], Step [500/1248] Loss: 0.2136
Test Accuracy of NN: 95.07692307692308 % (improvement)
Ordinary Epoch [26/100], Step [500/1248] Loss: 0.1615
Test Accuracy of NN: 95.01923076923077 % Best: 95.07692307692308 %
Ordinary Epoch [27/100], Step [500/1248] Loss: 0.2619
Test Accuracy of NN: 95.09134615384616 % (improvement)
Ordinary Epoch [28/100], Step [500/1248] Loss: 0.1136
Test Accuracy of NN: 95.0673076923077 % Best: 95.09134615384616 %
Ordinary Epoch [29/100], Step [500/1248] Loss: 0.1841
Test Accuracy of NN: 94.84615384615384 % Best: 95.09134615384616 %
Ordinary Epoch [30/100], Step [500/1248] Loss: 0.2199
Test Accuracy of NN: 95.04807692307692 % Best: 95.09134615384616 %
Ordinary Epoch [31/100], Step [500/1248] Loss: 0.1942
Test Accuracy of NN: 95.0576923076923 % Best: 95.09134615384616 %
Ordinary Epoch [32/100], Step [500/1248] Loss: 0.3352
Test Accuracy of NN: 95.0673076923077 % Best: 95.09134615384616 %
Ordinary Epoch [33/100], Step [500/1248] Loss: 0.2430
Test Accuracy of NN: 94.67307692307692 % Best: 95.09134615384616 %
Ordinary Epoch [34/100], Step [500/1248] Loss: 0.2000
Test Accuracy of NN: 94.8798076923077 % Best: 95.09134615384616 %
Ordinary Epoch [35/100], Step [500/1248] Loss: 0.5608
Test Accuracy of NN: 95.03365384615384 % Best: 95.09134615384616 %
Ordinary Epoch [36/100], Step [500/1248] Loss: 0.2477
Test Accuracy of NN: 94.2548076923077 % Best: 95.09134615384616 %
Ordinary Epoch [37/100], Step [500/1248] Loss: 0.2007
Test Accuracy of NN: 94.86538461538461 % Best: 95.09134615384616 %
Ordinary Epoch [38/100], Step [500/1248] Loss: 0.2382
Test Accuracy of NN: 95.01923076923077 % Best: 95.09134615384616 %
Ordinary Epoch [39/100], Step [500/1248] Loss: 0.1575
Test Accuracy of NN: 95.0 % Best: 95.09134615384616 %
Ordinary Epoch [40/100], Step [500/1248] Loss: 0.2993
Test Accuracy of NN: 95.01442307692308 % Best: 95.09134615384616 %
Ordinary Epoch [41/100], Step [500/1248] Loss: 0.2820
Test Accuracy of NN: 95.08173076923077 % Best: 95.09134615384616 %
Ordinary Epoch [42/100], Step [500/1248] Loss: 0.2041
Test Accuracy of NN: 94.34134615384616 % Best: 95.09134615384616 %
Ordinary Epoch [43/100], Step [500/1248] Loss: 0.2181
Test Accuracy of NN: 94.75 % Best: 95.09134615384616 %
Ordinary Epoch [44/100], Step [500/1248] Loss: 0.1577
Test Accuracy of NN: 95.09615384615384 % (improvement)
Ordinary Epoch [45/100], Step [500/1248] Loss: 0.1712
Test Accuracy of NN: 95.16346153846153 % (improvement)
Ordinary Epoch [46/100], Step [500/1248] Loss: 0.2188
Test Accuracy of NN: 95.11538461538461 % Best: 95.16346153846153 %
Ordinary Epoch [47/100], Step [500/1248] Loss: 0.3552
Test Accuracy of NN: 94.65384615384616 % Best: 95.16346153846153 %
Ordinary Epoch [48/100], Step [500/1248] Loss: 0.2424
Test Accuracy of NN: 95.17788461538461 % (improvement)
Ordinary Epoch [49/100], Step [500/1248] Loss: 0.2350
Test Accuracy of NN: 95.1201923076923 % Best: 95.17788461538461 %
Ordinary Epoch [50/100], Step [500/1248] Loss: 0.1593
Test Accuracy of NN: 95.17307692307692 % Best: 95.17788461538461 %
Ordinary Epoch [51/100], Step [500/1248] Loss: 0.1355
Test Accuracy of NN: 94.88461538461539 % Best: 95.17788461538461 %
Ordinary Epoch [52/100], Step [500/1248] Loss: 0.2579
Test Accuracy of NN: 95.28365384615384 % (improvement)
Ordinary Epoch [53/100], Step [500/1248] Loss: 0.1730
Test Accuracy of NN: 94.94711538461539 % Best: 95.28365384615385 %
Ordinary Epoch [54/100], Step [500/1248] Loss: 0.1825
Test Accuracy of NN: 95.19711538461539 % Best: 95.28365384615385 %
Ordinary Epoch [55/100], Step [500/1248] Loss: 0.2055
Test Accuracy of NN: 95.22115384615384 % Best: 95.28365384615385 %
Ordinary Epoch [56/100], Step [500/1248] Loss: 0.2916
Test Accuracy of NN: 95.13461538461539 % Best: 95.28365384615385 %
Ordinary Epoch [57/100], Step [500/1248] Loss: 0.1993
Test Accuracy of NN: 95.19711538461539 % Best: 95.28365384615385 %
Ordinary Epoch [58/100], Step [500/1248] Loss: 0.1689
Test Accuracy of NN: 94.84615384615384 % Best: 95.28365384615385 %
Ordinary Epoch [59/100], Step [500/1248] Loss: 0.1559
Test Accuracy of NN: 95.20673076923077 % Best: 95.28365384615385 %
Ordinary Epoch [60/100], Step [500/1248] Loss: 0.1204
Test Accuracy of NN: 95.1923076923077 % Best: 95.28365384615385 %
Ordinary Epoch [61/100], Step [500/1248] Loss: 0.2073
Test Accuracy of NN: 95.22115384615384 % Best: 95.28365384615385 %
Ordinary Epoch [62/100], Step [500/1248] Loss: 0.3414
Test Accuracy of NN: 95.03846153846153 % Best: 95.28365384615385 %
Ordinary Epoch [63/100], Step [500/1248] Loss: 0.1768
Test Accuracy of NN: 95.25 % Best: 95.28365384615385 %
Ordinary Epoch [64/100], Step [500/1248] Loss: 0.2455
Test Accuracy of NN: 94.84615384615384 % Best: 95.28365384615385 %
Ordinary Epoch [65/100], Step [500/1248] Loss: 0.2201
Test Accuracy of NN: 95.13461538461539 % Best: 95.28365384615385 %
Ordinary Epoch [66/100], Step [500/1248] Loss: 0.1575
Test Accuracy of NN: 95.13461538461539 % Best: 95.28365384615385 %
Ordinary Epoch [67/100], Step [500/1248] Loss: 0.2273
Test Accuracy of NN: 94.6826923076923 % Best: 95.28365384615385 %
Ordinary Epoch [68/100], Step [500/1248] Loss: 0.2652
Test Accuracy of NN: 95.11538461538461 % Best: 95.28365384615385 %
Ordinary Epoch [69/100], Step [500/1248] Loss: 0.2125
Test Accuracy of NN: 94.98557692307692 % Best: 95.28365384615385 %
Ordinary Epoch [70/100], Step [500/1248] Loss: 0.3802
Test Accuracy of NN: 95.24038461538461 % Best: 95.28365384615385 %
Ordinary Epoch [71/100], Step [500/1248] Loss: 0.3213
Test Accuracy of NN: 95.00961538461539 % Best: 95.28365384615385 %
Ordinary Epoch [72/100], Step [500/1248] Loss: 0.2011
Test Accuracy of NN: 95.1923076923077 % Best: 95.28365384615385 %
Ordinary Epoch [73/100], Step [500/1248] Loss: 0.1907
Test Accuracy of NN: 95.21153846153847 % Best: 95.28365384615385 %
Ordinary Epoch [74/100], Step [500/1248] Loss: 0.2368
Test Accuracy of NN: 94.6826923076923 % Best: 95.28365384615385 %
Ordinary Epoch [75/100], Step [500/1248] Loss: 0.2531
Test Accuracy of NN: 94.80288461538461 % Best: 95.28365384615385 %
Ordinary Epoch [76/100], Step [500/1248] Loss: 0.1877
Test Accuracy of NN: 95.14423076923077 % Best: 95.28365384615385 %
Ordinary Epoch [77/100], Step [500/1248] Loss: 0.2192
Test Accuracy of NN: 95.17788461538461 % Best: 95.28365384615385 %
Ordinary Epoch [78/100], Step [500/1248] Loss: 0.2605
Test Accuracy of NN: 95.20192307692308 % Best: 95.28365384615385 %
Ordinary Epoch [79/100], Step [500/1248] Loss: 0.1259
Test Accuracy of NN: 95.00961538461539 % Best: 95.28365384615385 %
Ordinary Epoch [80/100], Step [500/1248] Loss: 0.1736
Test Accuracy of NN: 95.0576923076923 % Best: 95.28365384615385 %
Ordinary Epoch [81/100], Step [500/1248] Loss: 0.1883
Test Accuracy of NN: 95.07692307692308 % Best: 95.28365384615385 %
Ordinary Epoch [82/100], Step [500/1248] Loss: 0.1379
Test Accuracy of NN: 95.0625 % Best: 95.28365384615385 %
Ordinary Epoch [83/100], Step [500/1248] Loss: 0.1259
Test Accuracy of NN: 95.22115384615384 % Best: 95.28365384615385 %
Ordinary Epoch [84/100], Step [500/1248] Loss: 0.4031
Test Accuracy of NN: 95.23076923076923 % Best: 95.28365384615385 %
Ordinary Epoch [85/100], Step [500/1248] Loss: 0.1593
Test Accuracy of NN: 95.3125 % (improvement)
Ordinary Epoch [86/100], Step [500/1248] Loss: 0.1766
Test Accuracy of NN: 95.3076923076923 % Best: 95.3125 %
Ordinary Epoch [87/100], Step [500/1248] Loss: 0.1444
Test Accuracy of NN: 95.28846153846153 % Best: 95.3125 %
Ordinary Epoch [88/100], Step [500/1248] Loss: 0.1650
Test Accuracy of NN: 95.32211538461539 % (improvement)
Ordinary Epoch [89/100], Step [500/1248] Loss: 0.2194
Test Accuracy of NN: 95.32211538461539 % Best: 95.32211538461539 %
Ordinary Epoch [90/100], Step [500/1248] Loss: 0.1360
Test Accuracy of NN: 95.32211538461539 % Best: 95.32211538461539 %
Ordinary Epoch [91/100], Step [500/1248] Loss: 0.2270
Test Accuracy of NN: 94.79326923076923 % Best: 95.32211538461539 %
Ordinary Epoch [92/100], Step [500/1248] Loss: 0.1454
Test Accuracy of NN: 95.04807692307692 % Best: 95.32211538461539 %
Ordinary Epoch [93/100], Step [500/1248] Loss: 0.2023
Test Accuracy of NN: 95.08173076923077 % Best: 95.32211538461539 %
Ordinary Epoch [94/100], Step [500/1248] Loss: 0.1745
Test Accuracy of NN: 94.98557692307692 % Best: 95.32211538461539 %
Ordinary Epoch [95/100], Step [500/1248] Loss: 0.1638
Test Accuracy of NN: 95.23557692307692 % Best: 95.32211538461539 %
Ordinary Epoch [96/100], Step [500/1248] Loss: 0.2094
Test Accuracy of NN: 95.24038461538461 % Best: 95.32211538461539 %
Ordinary Epoch [97/100], Step [500/1248] Loss: 0.2158
Test Accuracy of NN: 95.23557692307692 % Best: 95.32211538461539 %
Ordinary Epoch [98/100], Step [500/1248] Loss: 0.2056
Test Accuracy of NN: 94.9375 % Best: 95.32211538461539 %
Ordinary Epoch [99/100], Step [500/1248] Loss: 0.1341
Test Accuracy of NN: 95.24038461538461 % Best: 95.32211538461539 %
Ordinary Epoch [100/100], Step [500/1248] Loss: 0.1723
Test Accuracy of NN: 95.28365384615384 % Best: 95.32211538461539 %
