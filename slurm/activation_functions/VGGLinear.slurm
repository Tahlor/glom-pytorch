/lustre/scratch/grp/fslg_internn/env/internn/bin/python
torch.Size([1000, 1, 28, 28])
max label 26
<class 'models.VGGLinear'>
Parameters 3654875
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
    (4): Linear(in_features=512, out_features=27, bias=True)
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
    (4): Linear(in_features=512, out_features=27, bias=True)
  )
)>
Ordinary Epoch [1/100], Step [500/1248] Loss: 0.6411
Test Accuracy of NN: 89.22596153846153 % (improvement)
Ordinary Epoch [2/100], Step [500/1248] Loss: 0.3399
Test Accuracy of NN: 91.95192307692308 % (improvement)
Ordinary Epoch [3/100], Step [500/1248] Loss: 0.2949
Test Accuracy of NN: 92.15865384615384 % (improvement)
Ordinary Epoch [4/100], Step [500/1248] Loss: 0.2872
Test Accuracy of NN: 92.6923076923077 % (improvement)
Ordinary Epoch [5/100], Step [500/1248] Loss: 0.4552
/lustre/scratch/grp/fslg_internn/glom-pytorch/MNIST_vgg.py:121: DeprecationWarning: np.asscalar(a) is deprecated since NumPy v1.16, use a.item() instead
  curr_lr1 = learning_rate * np.asscalar(pow(np.random.rand(1), 3))
Test Accuracy of NN: 91.47115384615384 % Best: 92.6923076923077 %
Ordinary Epoch [6/100], Step [500/1248] Loss: 0.3291
Test Accuracy of NN: 93.91346153846153 % (improvement)
Ordinary Epoch [7/100], Step [500/1248] Loss: 0.1938
Test Accuracy of NN: 94.25961538461539 % (improvement)
Ordinary Epoch [8/100], Step [500/1248] Loss: 0.2657
Test Accuracy of NN: 94.4423076923077 % (improvement)
Ordinary Epoch [9/100], Step [500/1248] Loss: 0.2649
Test Accuracy of NN: 94.4375 % Best: 94.4423076923077 %
Ordinary Epoch [10/100], Step [500/1248] Loss: 0.2849
Test Accuracy of NN: 94.67307692307692 % (improvement)
Ordinary Epoch [11/100], Step [500/1248] Loss: 0.2451
Test Accuracy of NN: 94.6298076923077 % Best: 94.67307692307692 %
Ordinary Epoch [12/100], Step [500/1248] Loss: 0.2438
Test Accuracy of NN: 93.74038461538461 % Best: 94.67307692307692 %
Ordinary Epoch [13/100], Step [500/1248] Loss: 0.2699
Test Accuracy of NN: 94.66826923076923 % Best: 94.67307692307692 %
Ordinary Epoch [14/100], Step [500/1248] Loss: 0.3010
Test Accuracy of NN: 94.76923076923077 % (improvement)
Ordinary Epoch [15/100], Step [500/1248] Loss: 0.1807
Test Accuracy of NN: 94.75961538461539 % Best: 94.76923076923077 %
Ordinary Epoch [16/100], Step [500/1248] Loss: 0.2629
Test Accuracy of NN: 93.74038461538461 % Best: 94.76923076923077 %
Ordinary Epoch [17/100], Step [500/1248] Loss: 0.2214
Test Accuracy of NN: 94.14903846153847 % Best: 94.76923076923077 %
Ordinary Epoch [18/100], Step [500/1248] Loss: 0.3640
Test Accuracy of NN: 94.66346153846153 % Best: 94.76923076923077 %
Ordinary Epoch [19/100], Step [500/1248] Loss: 0.2229
Test Accuracy of NN: 94.41346153846153 % Best: 94.76923076923077 %
Ordinary Epoch [20/100], Step [500/1248] Loss: 0.3494
Test Accuracy of NN: 94.84615384615384 % (improvement)
Ordinary Epoch [21/100], Step [500/1248] Loss: 0.1929
Test Accuracy of NN: 94.96153846153847 % (improvement)
Ordinary Epoch [22/100], Step [500/1248] Loss: 0.4934
Test Accuracy of NN: 95.03365384615384 % (improvement)
Ordinary Epoch [23/100], Step [500/1248] Loss: 0.1841
Test Accuracy of NN: 94.9375 % Best: 95.03365384615384 %
Ordinary Epoch [24/100], Step [500/1248] Loss: 0.2223
Test Accuracy of NN: 93.89423076923077 % Best: 95.03365384615384 %
Ordinary Epoch [25/100], Step [500/1248] Loss: 0.1428
Test Accuracy of NN: 94.07211538461539 % Best: 95.03365384615384 %
Ordinary Epoch [26/100], Step [500/1248] Loss: 0.2723
Test Accuracy of NN: 94.17788461538461 % Best: 95.03365384615384 %
Ordinary Epoch [27/100], Step [500/1248] Loss: 0.1744
Test Accuracy of NN: 94.76923076923077 % Best: 95.03365384615384 %
Ordinary Epoch [28/100], Step [500/1248] Loss: 0.1476
Test Accuracy of NN: 94.625 % Best: 95.03365384615384 %
Ordinary Epoch [29/100], Step [500/1248] Loss: 0.1475
Test Accuracy of NN: 94.96634615384616 % Best: 95.03365384615384 %
Ordinary Epoch [30/100], Step [500/1248] Loss: 0.2268
Test Accuracy of NN: 94.2451923076923 % Best: 95.03365384615384 %
Ordinary Epoch [31/100], Step [500/1248] Loss: 0.3024
Test Accuracy of NN: 95.03365384615384 % Best: 95.03365384615384 %
Ordinary Epoch [32/100], Step [500/1248] Loss: 0.1009
Test Accuracy of NN: 95.03846153846153 % (improvement)
Ordinary Epoch [33/100], Step [500/1248] Loss: 0.1739
Test Accuracy of NN: 95.10576923076923 % (improvement)
Ordinary Epoch [34/100], Step [500/1248] Loss: 0.2525
Test Accuracy of NN: 95.11057692307692 % (improvement)
Ordinary Epoch [35/100], Step [500/1248] Loss: 0.2072
Test Accuracy of NN: 95.11538461538461 % (improvement)
Ordinary Epoch [36/100], Step [500/1248] Loss: 0.1321
Test Accuracy of NN: 95.15865384615384 % (improvement)
Ordinary Epoch [37/100], Step [500/1248] Loss: 0.2458
Test Accuracy of NN: 95.1923076923077 % (improvement)
Ordinary Epoch [38/100], Step [500/1248] Loss: 0.1077
Test Accuracy of NN: 95.13461538461539 % Best: 95.1923076923077 %
Ordinary Epoch [39/100], Step [500/1248] Loss: 0.1385
Test Accuracy of NN: 95.09615384615384 % Best: 95.1923076923077 %
Ordinary Epoch [40/100], Step [500/1248] Loss: 0.1665
Test Accuracy of NN: 95.20192307692308 % (improvement)
Ordinary Epoch [41/100], Step [500/1248] Loss: 0.3147
Test Accuracy of NN: 95.19711538461539 % Best: 95.20192307692308 %
Ordinary Epoch [42/100], Step [500/1248] Loss: 0.1387
Test Accuracy of NN: 95.17307692307692 % Best: 95.20192307692308 %
Ordinary Epoch [43/100], Step [500/1248] Loss: 0.1781
Test Accuracy of NN: 95.1298076923077 % Best: 95.20192307692308 %
Ordinary Epoch [44/100], Step [500/1248] Loss: 0.1983
Test Accuracy of NN: 94.76923076923077 % Best: 95.20192307692308 %
Ordinary Epoch [45/100], Step [500/1248] Loss: 0.2745
Test Accuracy of NN: 95.08173076923077 % Best: 95.20192307692308 %
Ordinary Epoch [46/100], Step [500/1248] Loss: 0.3123
Test Accuracy of NN: 94.95192307692308 % Best: 95.20192307692308 %
Ordinary Epoch [47/100], Step [500/1248] Loss: 0.2107
Test Accuracy of NN: 93.86057692307692 % Best: 95.20192307692308 %
Ordinary Epoch [48/100], Step [500/1248] Loss: 0.1643
Test Accuracy of NN: 94.70192307692308 % Best: 95.20192307692308 %
Ordinary Epoch [49/100], Step [500/1248] Loss: 0.4619
Test Accuracy of NN: 94.9375 % Best: 95.20192307692308 %
Ordinary Epoch [50/100], Step [500/1248] Loss: 0.1813
Test Accuracy of NN: 94.97596153846153 % Best: 95.20192307692308 %
Ordinary Epoch [51/100], Step [500/1248] Loss: 0.2126
Test Accuracy of NN: 95.11057692307692 % Best: 95.20192307692308 %
Ordinary Epoch [52/100], Step [500/1248] Loss: 0.1884
Test Accuracy of NN: 94.90384615384616 % Best: 95.20192307692308 %
Ordinary Epoch [53/100], Step [500/1248] Loss: 0.1602
Test Accuracy of NN: 95.21634615384616 % (improvement)
Ordinary Epoch [54/100], Step [500/1248] Loss: 0.2348
Test Accuracy of NN: 95.23076923076923 % (improvement)
Ordinary Epoch [55/100], Step [500/1248] Loss: 0.1722
Test Accuracy of NN: 95.27884615384616 % (improvement)
Ordinary Epoch [56/100], Step [500/1248] Loss: 0.2564
Test Accuracy of NN: 95.20192307692308 % Best: 95.27884615384615 %
Ordinary Epoch [57/100], Step [500/1248] Loss: 0.1446
Test Accuracy of NN: 94.02403846153847 % Best: 95.27884615384615 %
Ordinary Epoch [58/100], Step [500/1248] Loss: 0.1735
Test Accuracy of NN: 94.95192307692308 % Best: 95.27884615384615 %
Ordinary Epoch [59/100], Step [500/1248] Loss: 0.2547
Test Accuracy of NN: 93.61538461538461 % Best: 95.27884615384615 %
Ordinary Epoch [60/100], Step [500/1248] Loss: 0.1938
Test Accuracy of NN: 94.96153846153847 % Best: 95.27884615384615 %
Ordinary Epoch [61/100], Step [500/1248] Loss: 0.1753
Test Accuracy of NN: 95.0625 % Best: 95.27884615384615 %
Ordinary Epoch [62/100], Step [500/1248] Loss: 0.0973
Test Accuracy of NN: 95.3173076923077 % (improvement)
Ordinary Epoch [63/100], Step [500/1248] Loss: 0.0903
Test Accuracy of NN: 95.3076923076923 % Best: 95.3173076923077 %
Ordinary Epoch [64/100], Step [500/1248] Loss: 0.1674
Test Accuracy of NN: 94.8076923076923 % Best: 95.3173076923077 %
Ordinary Epoch [65/100], Step [500/1248] Loss: 0.2184
Test Accuracy of NN: 94.7451923076923 % Best: 95.3173076923077 %
Ordinary Epoch [66/100], Step [500/1248] Loss: 0.2071
Test Accuracy of NN: 94.61057692307692 % Best: 95.3173076923077 %
Ordinary Epoch [67/100], Step [500/1248] Loss: 0.0552
Test Accuracy of NN: 95.1875 % Best: 95.3173076923077 %
Ordinary Epoch [68/100], Step [500/1248] Loss: 0.2305
Test Accuracy of NN: 94.73557692307692 % Best: 95.3173076923077 %
Ordinary Epoch [69/100], Step [500/1248] Loss: 0.1837
Test Accuracy of NN: 95.30288461538461 % Best: 95.3173076923077 %
Ordinary Epoch [70/100], Step [500/1248] Loss: 0.0823
Test Accuracy of NN: 95.32211538461539 % (improvement)
Ordinary Epoch [71/100], Step [500/1248] Loss: 0.0983
Test Accuracy of NN: 95.3173076923077 % Best: 95.32211538461539 %
Ordinary Epoch [72/100], Step [500/1248] Loss: 0.1720
Test Accuracy of NN: 95.32211538461539 % Best: 95.32211538461539 %
Ordinary Epoch [73/100], Step [500/1248] Loss: 0.1266
Test Accuracy of NN: 95.35576923076923 % (improvement)
Ordinary Epoch [74/100], Step [500/1248] Loss: 0.2316
Test Accuracy of NN: 95.29807692307692 % Best: 95.35576923076923 %
Ordinary Epoch [75/100], Step [500/1248] Loss: 0.1687
Test Accuracy of NN: 95.3076923076923 % Best: 95.35576923076923 %
Ordinary Epoch [76/100], Step [500/1248] Loss: 0.1470
Test Accuracy of NN: 95.14903846153847 % Best: 95.35576923076923 %
Ordinary Epoch [77/100], Step [500/1248] Loss: 0.2470
Test Accuracy of NN: 95.13942307692308 % Best: 95.35576923076923 %
Ordinary Epoch [78/100], Step [500/1248] Loss: 0.2221
Test Accuracy of NN: 95.3125 % Best: 95.35576923076923 %
Ordinary Epoch [79/100], Step [500/1248] Loss: 0.0769
Test Accuracy of NN: 95.3173076923077 % Best: 95.35576923076923 %
Ordinary Epoch [80/100], Step [500/1248] Loss: 0.0962
Test Accuracy of NN: 95.30288461538461 % Best: 95.35576923076923 %
Ordinary Epoch [81/100], Step [500/1248] Loss: 0.1519
Test Accuracy of NN: 95.30288461538461 % Best: 95.35576923076923 %
Ordinary Epoch [82/100], Step [500/1248] Loss: 0.0598
Test Accuracy of NN: 95.38942307692308 % (improvement)
Ordinary Epoch [83/100], Step [500/1248] Loss: 0.1678
Test Accuracy of NN: 95.28365384615384 % Best: 95.38942307692308 %
Ordinary Epoch [84/100], Step [500/1248] Loss: 0.0991
Test Accuracy of NN: 95.38461538461539 % Best: 95.38942307692308 %
Ordinary Epoch [85/100], Step [500/1248] Loss: 0.0895
Test Accuracy of NN: 95.36057692307692 % Best: 95.38942307692308 %
Ordinary Epoch [86/100], Step [500/1248] Loss: 0.1437
Test Accuracy of NN: 95.3701923076923 % Best: 95.38942307692308 %
Ordinary Epoch [87/100], Step [500/1248] Loss: 0.2059
Test Accuracy of NN: 95.0048076923077 % Best: 95.38942307692308 %
Ordinary Epoch [88/100], Step [500/1248] Loss: 0.3437
Test Accuracy of NN: 95.44711538461539 % (improvement)
Ordinary Epoch [89/100], Step [500/1248] Loss: 0.2321
Test Accuracy of NN: 95.375 % Best: 95.44711538461539 %
Ordinary Epoch [90/100], Step [500/1248] Loss: 0.2678
Test Accuracy of NN: 94.52884615384616 % Best: 95.44711538461539 %
Ordinary Epoch [91/100], Step [500/1248] Loss: 0.2205
Test Accuracy of NN: 94.8701923076923 % Best: 95.44711538461539 %
Ordinary Epoch [92/100], Step [500/1248] Loss: 0.1905
Test Accuracy of NN: 94.64423076923077 % Best: 95.44711538461539 %
Ordinary Epoch [93/100], Step [500/1248] Loss: 0.1210
Test Accuracy of NN: 95.27884615384616 % Best: 95.44711538461539 %
Ordinary Epoch [94/100], Step [500/1248] Loss: 0.1819
Test Accuracy of NN: 95.36057692307692 % Best: 95.44711538461539 %
Ordinary Epoch [95/100], Step [500/1248] Loss: 0.1659
Test Accuracy of NN: 95.40384615384616 % Best: 95.44711538461539 %
Ordinary Epoch [96/100], Step [500/1248] Loss: 0.2169
Test Accuracy of NN: 95.41346153846153 % Best: 95.44711538461539 %
Ordinary Epoch [97/100], Step [500/1248] Loss: 0.1019
Test Accuracy of NN: 95.46634615384616 % (improvement)
Ordinary Epoch [98/100], Step [500/1248] Loss: 0.2229
Test Accuracy of NN: 95.40865384615384 % Best: 95.46634615384615 %
Ordinary Epoch [99/100], Step [500/1248] Loss: 0.1340
Test Accuracy of NN: 95.41346153846153 % Best: 95.46634615384615 %
Ordinary Epoch [100/100], Step [500/1248] Loss: 0.1266
Test Accuracy of NN: 95.07692307692308 % Best: 95.46634615384615 %
