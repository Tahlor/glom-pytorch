/lustre/scratch/grp/fslg_internn/env/internn/bin/python
Test Accuracy of NN: 88.85106382978724 % Best: 89.79255319148936 %
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
Test Accuracy of NN: 90.23936170212765 % Best: 90.8936170212766 %
Ordinary Epoch [18/100], Step [500/1128] Loss: 0.2975
Ordinary Epoch [7/100], Step [500/1128] Loss: 0.4482
Ordinary Epoch [1/100], Step [500/1248] Loss: 0.8022
Ordinary Epoch [88/100], Step [500/1128] Loss: 0.2024
Test Accuracy of NN: 89.61702127659575 % Best: 89.79255319148936 %
Test Accuracy of NN: 92.27884615384616 % (improvement)
/lustre/scratch/grp/fslg_internn/glom-pytorch/MNIST_vgg.py:121: DeprecationWarning: np.asscalar(a) is deprecated since NumPy v1.16, use a.item() instead
  curr_lr1 = learning_rate * np.asscalar(pow(np.random.rand(1), 3))
Test Accuracy of NN: 88.47872340425532 % Best: 88.63829787234042 %
Test Accuracy of NN: 90.30319148936171 % Best: 90.8936170212766 %
Ordinary Epoch [19/100], Step [500/1128] Loss: 0.1990
Ordinary Epoch [2/100], Step [500/1248] Loss: 0.3209
Ordinary Epoch [89/100], Step [500/1128] Loss: 0.2847
Ordinary Epoch [8/100], Step [500/1128] Loss: 0.3649
Test Accuracy of NN: 89.68617021276596 % Best: 89.79255319148936 %
/lustre/scratch/grp/fslg_internn/glom-pytorch/MNIST_vgg.py:121: DeprecationWarning: np.asscalar(a) is deprecated since NumPy v1.16, use a.item() instead
  curr_lr1 = learning_rate * np.asscalar(pow(np.random.rand(1), 3))
Test Accuracy of NN: 91.6875 % Best: 92.27884615384615 %
Test Accuracy of NN: 90.76063829787235 % Best: 90.8936170212766 %
Ordinary Epoch [20/100], Step [500/1128] Loss: 0.2444
Test Accuracy of NN: 89.55851063829788 % (improvement)
Ordinary Epoch [3/100], Step [500/1248] Loss: 0.2167
Ordinary Epoch [90/100], Step [500/1128] Loss: 0.2128
Test Accuracy of NN: 89.90425531914893 % (improvement)
Ordinary Epoch [9/100], Step [500/1128] Loss: 0.2275
Test Accuracy of NN: 94.01923076923077 % (improvement)
Test Accuracy of NN: 90.41489361702128 % Best: 90.8936170212766 %
Ordinary Epoch [21/100], Step [500/1128] Loss: 0.4812
Ordinary Epoch [4/100], Step [500/1248] Loss: 0.3341
Ordinary Epoch [91/100], Step [500/1128] Loss: 0.2677
Test Accuracy of NN: 89.87234042553192 % (improvement)
Test Accuracy of NN: 89.80319148936171 % Best: 89.90425531914894 %
Test Accuracy of NN: 94.44711538461539 % (improvement)
Test Accuracy of NN: 90.51595744680851 % Best: 90.8936170212766 %
Ordinary Epoch [10/100], Step [500/1128] Loss: 0.3487
Ordinary Epoch [22/100], Step [500/1128] Loss: 0.2914
Ordinary Epoch [5/100], Step [500/1248] Loss: 0.2291
Ordinary Epoch [92/100], Step [500/1128] Loss: 0.3941
Test Accuracy of NN: 89.75531914893617 % Best: 89.90425531914894 %
Test Accuracy of NN: 89.90957446808511 % (improvement)
Test Accuracy of NN: 90.22872340425532 % Best: 90.8936170212766 %
Test Accuracy of NN: 94.33173076923077 % Best: 94.44711538461539 %
Ordinary Epoch [23/100], Step [500/1128] Loss: 0.2731
Ordinary Epoch [11/100], Step [500/1128] Loss: 0.4437
Ordinary Epoch [93/100], Step [500/1128] Loss: 0.1708
Ordinary Epoch [6/100], Step [500/1248] Loss: 0.1477
Test Accuracy of NN: 89.93617021276596 % (improvement)
Test Accuracy of NN: 89.94148936170212 % (improvement)
Test Accuracy of NN: 90.82446808510639 % Best: 90.8936170212766 %
Ordinary Epoch [24/100], Step [500/1128] Loss: 0.4351
Test Accuracy of NN: 95.22596153846153 % (improvement)
Ordinary Epoch [94/100], Step [500/1128] Loss: 0.1903
Ordinary Epoch [12/100], Step [500/1128] Loss: 0.3333
Ordinary Epoch [7/100], Step [500/1248] Loss: 0.1296
Test Accuracy of NN: 89.94148936170212 % (improvement)
Test Accuracy of NN: 90.83510638297872 % Best: 90.8936170212766 %
Ordinary Epoch [25/100], Step [500/1128] Loss: 0.2878
Test Accuracy of NN: 95.32692307692308 % (improvement)
Test Accuracy of NN: 89.93085106382979 % Best: 89.94148936170212 %
Ordinary Epoch [95/100], Step [500/1128] Loss: 0.2228
Ordinary Epoch [8/100], Step [500/1248] Loss: 0.2012
Test Accuracy of NN: 89.9627659574468 % (improvement)
Ordinary Epoch [13/100], Step [500/1128] Loss: 0.1726
Ordinary Epoch [26/100], Step [500/1128] Loss: 0.3791
Test Accuracy of NN: 90.86170212765957 % Best: 90.8936170212766 %
Test Accuracy of NN: 95.15384615384616 % Best: 95.32692307692308 %
Ordinary Epoch [96/100], Step [500/1128] Loss: 0.2483
Test Accuracy of NN: 90.02127659574468 % (improvement)
Ordinary Epoch [9/100], Step [500/1248] Loss: 0.1482
Test Accuracy of NN: 90.0 % (improvement)
Ordinary Epoch [27/100], Step [500/1128] Loss: 0.2022
Test Accuracy of NN: 90.65957446808511 % Best: 90.8936170212766 %
Ordinary Epoch [14/100], Step [500/1128] Loss: 0.2959
Test Accuracy of NN: 95.1826923076923 % Best: 95.32692307692308 %
Ordinary Epoch [97/100], Step [500/1128] Loss: 0.2411
Test Accuracy of NN: 89.98936170212765 % Best: 90.0 %
Ordinary Epoch [10/100], Step [500/1248] Loss: 0.1019
Test Accuracy of NN: 90.00531914893617 % Best: 90.02127659574468 %
Ordinary Epoch [28/100], Step [500/1128] Loss: 0.3788
Test Accuracy of NN: 90.43085106382979 % Best: 90.8936170212766 %
Test Accuracy of NN: 95.40384615384616 % (improvement)
Ordinary Epoch [15/100], Step [500/1128] Loss: 0.4201
Ordinary Epoch [98/100], Step [500/1128] Loss: 0.1339
Test Accuracy of NN: 89.12765957446808 % Best: 90.0 %
Ordinary Epoch [11/100], Step [500/1248] Loss: 0.1578
Ordinary Epoch [29/100], Step [500/1128] Loss: 0.4518
Test Accuracy of NN: 90.46808510638297 % Best: 90.8936170212766 %
Test Accuracy of NN: 88.79787234042553 % Best: 90.02127659574468 %
Test Accuracy of NN: 95.46153846153847 % (improvement)
Ordinary Epoch [99/100], Step [500/1128] Loss: 0.2639
Test Accuracy of NN: 89.74468085106383 % Best: 90.0 %
Ordinary Epoch [16/100], Step [500/1128] Loss: 0.2616
Ordinary Epoch [12/100], Step [500/1248] Loss: 0.1851
Ordinary Epoch [30/100], Step [500/1128] Loss: 0.3930
Test Accuracy of NN: 90.82446808510639 % Best: 90.8936170212766 %
Test Accuracy of NN: 95.52884615384616 % (improvement)
Ordinary Epoch [100/100], Step [500/1128] Loss: 0.1837
Test Accuracy of NN: 89.84042553191489 % Best: 90.02127659574468 %
Test Accuracy of NN: 89.98936170212765 % Best: 90.0 %
Ordinary Epoch [13/100], Step [500/1248] Loss: 0.1341
Ordinary Epoch [17/100], Step [500/1128] Loss: 0.4293
Ordinary Epoch [31/100], Step [500/1128] Loss: 0.3405
Test Accuracy of NN: 90.75531914893617 % Best: 90.8936170212766 %
Test Accuracy of NN: 95.57692307692308 % (improvement)
Test Accuracy of NN: 90.0372340425532 % (improvement)
Test Accuracy of NN: 89.63297872340425 % Best: 90.02127659574468 %
Ordinary Epoch [14/100], Step [500/1248] Loss: 0.2010
Ordinary Epoch [32/100], Step [500/1128] Loss: 0.3346
Ordinary Epoch [18/100], Step [500/1128] Loss: 0.3584
Test Accuracy of NN: 90.05851063829788 % (improvement)
Test Accuracy of NN: 95.52884615384616 % Best: 95.57692307692308 %
Ordinary Epoch [15/100], Step [500/1248] Loss: 0.2068
Ordinary Epoch [33/100], Step [500/1128] Loss: 0.2362
Test Accuracy of NN: 89.91489361702128 % Best: 90.02127659574468 %
Test Accuracy of NN: 90.04787234042553 % Best: 90.05851063829788 %
Ordinary Epoch [19/100], Step [500/1128] Loss: 0.2674
Test Accuracy of NN: 95.35096153846153 % Best: 95.57692307692308 %
Ordinary Epoch [34/100], Step [500/1128] Loss: 0.4519
Ordinary Epoch [16/100], Step [500/1248] Loss: 0.2674
Test Accuracy of NN: 90.14893617021276 % (improvement)
Test Accuracy of NN: 89.96808510638297 % Best: 90.05851063829788 %
Test Accuracy of NN: 95.54326923076923 % Best: 95.57692307692308 %
Ordinary Epoch [20/100], Step [500/1128] Loss: 0.3715
Ordinary Epoch [35/100], Step [500/1128] Loss: 0.4012
Ordinary Epoch [17/100], Step [500/1248] Loss: 0.2744
Test Accuracy of NN: 89.32446808510639 % Best: 90.05851063829788 %
Test Accuracy of NN: 90.19680851063829 % (improvement)
Test Accuracy of NN: 95.03846153846153 % Best: 95.57692307692308 %
Ordinary Epoch [36/100], Step [500/1128] Loss: 0.2697
Ordinary Epoch [18/100], Step [500/1248] Loss: 0.2507
Ordinary Epoch [21/100], Step [500/1128] Loss: 0.1833
Test Accuracy of NN: 90.0 % Best: 90.05851063829788 %
Test Accuracy of NN: 94.16826923076923 % Best: 95.57692307692308 %
Test Accuracy of NN: 90.22872340425532 % (improvement)
Ordinary Epoch [37/100], Step [500/1128] Loss: 0.2980
Ordinary Epoch [19/100], Step [500/1248] Loss: 0.2262
Ordinary Epoch [22/100], Step [500/1128] Loss: 0.1750
Test Accuracy of NN: 90.08510638297872 % (improvement)
Test Accuracy of NN: 94.39903846153847 % Best: 95.57692307692308 %
Ordinary Epoch [38/100], Step [500/1128] Loss: 0.3314
Test Accuracy of NN: 90.35106382978724 % (improvement)
Ordinary Epoch [20/100], Step [500/1248] Loss: 0.0847
Test Accuracy of NN: 90.09042553191489 % (improvement)
Ordinary Epoch [23/100], Step [500/1128] Loss: 0.2092
Ordinary Epoch [39/100], Step [500/1128] Loss: 0.3255
Test Accuracy of NN: 95.23557692307692 % Best: 95.57692307692308 %
Ordinary Epoch [21/100], Step [500/1248] Loss: 0.1188
Test Accuracy of NN: 90.38829787234043 % (improvement)
Test Accuracy of NN: 90.1436170212766 % (improvement)
Ordinary Epoch [40/100], Step [500/1128] Loss: 0.2764
Ordinary Epoch [24/100], Step [500/1128] Loss: 0.2507
Test Accuracy of NN: 95.44711538461539 % Best: 95.57692307692308 %
Ordinary Epoch [22/100], Step [500/1248] Loss: 0.1290
Test Accuracy of NN: 90.17021276595744 % (improvement)
Test Accuracy of NN: 90.35106382978724 % Best: 90.38829787234043 %
Ordinary Epoch [41/100], Step [500/1128] Loss: 0.2970
Test Accuracy of NN: 95.6201923076923 % (improvement)
Ordinary Epoch [25/100], Step [500/1128] Loss: 0.3914
Ordinary Epoch [23/100], Step [500/1248] Loss: 0.1572
Test Accuracy of NN: 90.17553191489361 % (improvement)
Ordinary Epoch [42/100], Step [500/1128] Loss: 0.3003
Test Accuracy of NN: 89.57978723404256 % Best: 90.38829787234043 %
Test Accuracy of NN: 95.57211538461539 % Best: 95.6201923076923 %
Ordinary Epoch [26/100], Step [500/1128] Loss: 0.3201
Test Accuracy of NN: 90.22872340425532 % (improvement)
Ordinary Epoch [24/100], Step [500/1248] Loss: 0.3551
Ordinary Epoch [43/100], Step [500/1128] Loss: 0.2393
Test Accuracy of NN: 89.84574468085107 % Best: 90.38829787234043 %
Test Accuracy of NN: 94.57211538461539 % Best: 95.6201923076923 %
Test Accuracy of NN: 90.34574468085107 % (improvement)
Ordinary Epoch [25/100], Step [500/1248] Loss: 0.1004
Ordinary Epoch [27/100], Step [500/1128] Loss: 0.4017
Ordinary Epoch [44/100], Step [500/1128] Loss: 0.2534
Test Accuracy of NN: 95.30288461538461 % Best: 95.6201923076923 %
Test Accuracy of NN: 89.75 % Best: 90.38829787234043 %
Test Accuracy of NN: 90.27127659574468 % Best: 90.34574468085106 %
Ordinary Epoch [26/100], Step [500/1248] Loss: 0.1176
Ordinary Epoch [28/100], Step [500/1128] Loss: 0.4041
Ordinary Epoch [45/100], Step [500/1128] Loss: 0.4427
Test Accuracy of NN: 95.38461538461539 % Best: 95.6201923076923 %
Test Accuracy of NN: 89.40425531914893 % Best: 90.34574468085106 %
Test Accuracy of NN: 90.09042553191489 % Best: 90.38829787234043 %
Ordinary Epoch [27/100], Step [500/1248] Loss: 0.1450
Ordinary Epoch [46/100], Step [500/1128] Loss: 0.3608
Ordinary Epoch [29/100], Step [500/1128] Loss: 0.2300
Test Accuracy of NN: 95.625 % (improvement)
Test Accuracy of NN: 89.36702127659575 % Best: 90.34574468085106 %
Test Accuracy of NN: 90.2872340425532 % Best: 90.38829787234043 %
Ordinary Epoch [28/100], Step [500/1248] Loss: 0.1145
Ordinary Epoch [47/100], Step [500/1128] Loss: 0.2826
Ordinary Epoch [30/100], Step [500/1128] Loss: 0.3332
Test Accuracy of NN: 95.6201923076923 % Best: 95.625 %
Test Accuracy of NN: 89.69148936170212 % Best: 90.34574468085106 %
Ordinary Epoch [29/100], Step [500/1248] Loss: 0.1905
Ordinary Epoch [48/100], Step [500/1128] Loss: 0.2012
Test Accuracy of NN: 90.19680851063829 % Best: 90.38829787234043 %
Ordinary Epoch [31/100], Step [500/1128] Loss: 0.2509
Test Accuracy of NN: 90.22872340425532 % Best: 90.34574468085106 %
Test Accuracy of NN: 95.64903846153847 % (improvement)
Ordinary Epoch [49/100], Step [500/1128] Loss: 0.4929
Ordinary Epoch [30/100], Step [500/1248] Loss: 0.2511
Test Accuracy of NN: 90.2872340425532 % Best: 90.38829787234043 %
Ordinary Epoch [32/100], Step [500/1128] Loss: 0.1078
Test Accuracy of NN: 89.94148936170212 % Best: 90.34574468085106 %
Test Accuracy of NN: 95.66346153846153 % (improvement)
Ordinary Epoch [50/100], Step [500/1128] Loss: 0.2788
Ordinary Epoch [31/100], Step [500/1248] Loss: 0.0989
Test Accuracy of NN: 90.17021276595744 % Best: 90.38829787234043 %
Test Accuracy of NN: 90.2872340425532 % Best: 90.34574468085106 %
Test Accuracy of NN: 95.66826923076923 % (improvement)
Ordinary Epoch [33/100], Step [500/1128] Loss: 0.1584
Ordinary Epoch [51/100], Step [500/1128] Loss: 0.1552
Ordinary Epoch [32/100], Step [500/1248] Loss: 0.1335
Test Accuracy of NN: 89.80319148936171 % Best: 90.38829787234043 %
Test Accuracy of NN: 90.44680851063829 % (improvement)
Test Accuracy of NN: 95.66346153846153 % Best: 95.66826923076923 %
Ordinary Epoch [52/100], Step [500/1128] Loss: 0.3651
Ordinary Epoch [34/100], Step [500/1128] Loss: 0.2375
Ordinary Epoch [33/100], Step [500/1248] Loss: 0.2584
Test Accuracy of NN: 90.47872340425532 % (improvement)
Test Accuracy of NN: 90.4627659574468 % (improvement)
Test Accuracy of NN: 95.29807692307692 % Best: 95.66826923076923 %
Ordinary Epoch [53/100], Step [500/1128] Loss: 0.2501
Ordinary Epoch [34/100], Step [500/1248] Loss: 0.0855
Ordinary Epoch [35/100], Step [500/1128] Loss: 0.2628
Test Accuracy of NN: 90.42553191489361 % Best: 90.47872340425532 %
Ordinary Epoch [54/100], Step [500/1128] Loss: 0.1664
Test Accuracy of NN: 95.49038461538461 % Best: 95.66826923076923 %
Test Accuracy of NN: 90.5372340425532 % (improvement)
Ordinary Epoch [35/100], Step [500/1248] Loss: 0.1592
Test Accuracy of NN: 90.03191489361703 % Best: 90.47872340425532 %
Ordinary Epoch [36/100], Step [500/1128] Loss: 0.2972
Ordinary Epoch [55/100], Step [500/1128] Loss: 0.3689
Test Accuracy of NN: 95.53846153846153 % Best: 95.66826923076923 %
Test Accuracy of NN: 90.32978723404256 % Best: 90.5372340425532 %
Ordinary Epoch [36/100], Step [500/1248] Loss: 0.2050
Test Accuracy of NN: 90.13829787234043 % Best: 90.47872340425532 %
Ordinary Epoch [37/100], Step [500/1128] Loss: 0.2166
Ordinary Epoch [56/100], Step [500/1128] Loss: 0.4298
Test Accuracy of NN: 95.54807692307692 % Best: 95.66826923076923 %
Test Accuracy of NN: 90.32446808510639 % Best: 90.5372340425532 %
Test Accuracy of NN: 90.48936170212765 % (improvement)
Ordinary Epoch [37/100], Step [500/1248] Loss: 0.1409
Ordinary Epoch [57/100], Step [500/1128] Loss: 0.3985
Ordinary Epoch [38/100], Step [500/1128] Loss: 0.2231
Test Accuracy of NN: 95.39423076923077 % Best: 95.66826923076923 %
Test Accuracy of NN: 90.60106382978724 % (improvement)
Ordinary Epoch [38/100], Step [500/1248] Loss: 0.2054
Test Accuracy of NN: 90.32446808510639 % Best: 90.5372340425532 %
Ordinary Epoch [58/100], Step [500/1128] Loss: 0.2031
Test Accuracy of NN: 94.48076923076923 % Best: 95.66826923076923 %
Ordinary Epoch [39/100], Step [500/1128] Loss: 0.2134
Test Accuracy of NN: 90.6436170212766 % (improvement)
Ordinary Epoch [39/100], Step [500/1248] Loss: 0.1290
Ordinary Epoch [59/100], Step [500/1128] Loss: 0.2789
Test Accuracy of NN: 90.45212765957447 % Best: 90.5372340425532 %
Test Accuracy of NN: 95.5673076923077 % Best: 95.66826923076923 %
Test Accuracy of NN: 90.62765957446808 % Best: 90.6436170212766 %
Ordinary Epoch [40/100], Step [500/1128] Loss: 0.1162
Ordinary Epoch [40/100], Step [500/1248] Loss: 0.1287
Ordinary Epoch [60/100], Step [500/1128] Loss: 0.2029
Test Accuracy of NN: 89.81914893617021 % Best: 90.5372340425532 %
Test Accuracy of NN: 95.53365384615384 % Best: 95.66826923076923 %
Test Accuracy of NN: 89.89893617021276 % Best: 90.6436170212766 %
Ordinary Epoch [41/100], Step [500/1128] Loss: 0.1849
Ordinary Epoch [41/100], Step [500/1248] Loss: 0.1275
Ordinary Epoch [61/100], Step [500/1128] Loss: 0.2868
Test Accuracy of NN: 95.58173076923077 % Best: 95.66826923076923 %
Test Accuracy of NN: 90.43617021276596 % Best: 90.6436170212766 %
Test Accuracy of NN: 90.33510638297872 % Best: 90.5372340425532 %
Ordinary Epoch [42/100], Step [500/1248] Loss: 0.0763
Ordinary Epoch [62/100], Step [500/1128] Loss: 0.2763
Ordinary Epoch [42/100], Step [500/1128] Loss: 0.2877
Test Accuracy of NN: 89.86702127659575 % Best: 90.6436170212766 %
Test Accuracy of NN: 95.6923076923077 % (improvement)
Test Accuracy of NN: 90.48404255319149 % Best: 90.5372340425532 %
Ordinary Epoch [63/100], Step [500/1128] Loss: 0.2056
Ordinary Epoch [43/100], Step [500/1248] Loss: 0.1074
Ordinary Epoch [43/100], Step [500/1128] Loss: 0.1834
Test Accuracy of NN: 90.32978723404256 % Best: 90.6436170212766 %
Test Accuracy of NN: 95.73557692307692 % (improvement)
Ordinary Epoch [64/100], Step [500/1128] Loss: 0.2981
Test Accuracy of NN: 90.56382978723404 % (improvement)
Ordinary Epoch [44/100], Step [500/1248] Loss: 0.1929
Test Accuracy of NN: 90.62765957446808 % Best: 90.6436170212766 %
Ordinary Epoch [44/100], Step [500/1128] Loss: 0.1834
Test Accuracy of NN: 95.66346153846153 % Best: 95.73557692307692 %
Ordinary Epoch [65/100], Step [500/1128] Loss: 0.4437
Ordinary Epoch [45/100], Step [500/1248] Loss: 0.0495
Test Accuracy of NN: 90.59042553191489 % (improvement)
Test Accuracy of NN: 89.01595744680851 % Best: 90.6436170212766 %
Ordinary Epoch [45/100], Step [500/1128] Loss: 0.2184
Test Accuracy of NN: 95.75 % (improvement)
Ordinary Epoch [66/100], Step [500/1128] Loss: 0.2205
Ordinary Epoch [46/100], Step [500/1248] Loss: 0.1432
Test Accuracy of NN: 90.53191489361703 % Best: 90.5904255319149 %
Test Accuracy of NN: 89.61170212765957 % Best: 90.6436170212766 %
Test Accuracy of NN: 95.73076923076923 % Best: 95.75 %
Ordinary Epoch [46/100], Step [500/1128] Loss: 0.2256
Ordinary Epoch [67/100], Step [500/1128] Loss: 0.2240
Ordinary Epoch [47/100], Step [500/1248] Loss: 0.0932
Test Accuracy of NN: 90.57978723404256 % Best: 90.6436170212766 %
Test Accuracy of NN: 90.7127659574468 % (improvement)
Ordinary Epoch [68/100], Step [500/1128] Loss: 0.4376
Test Accuracy of NN: 95.69711538461539 % Best: 95.75 %
Ordinary Epoch [47/100], Step [500/1128] Loss: 0.2528
Ordinary Epoch [48/100], Step [500/1248] Loss: 0.0690
Test Accuracy of NN: 90.7872340425532 % (improvement)
Test Accuracy of NN: 90.4627659574468 % Best: 90.7127659574468 %
Ordinary Epoch [69/100], Step [500/1128] Loss: 0.3416
Test Accuracy of NN: 95.73076923076923 % Best: 95.75 %
Ordinary Epoch [48/100], Step [500/1128] Loss: 0.2651
Ordinary Epoch [49/100], Step [500/1248] Loss: 0.3097
Test Accuracy of NN: 90.65957446808511 % Best: 90.7872340425532 %
Ordinary Epoch [70/100], Step [500/1128] Loss: 0.2771
Test Accuracy of NN: 90.1063829787234 % Best: 90.7127659574468 %
Test Accuracy of NN: 95.33173076923077 % Best: 95.75 %
Ordinary Epoch [50/100], Step [500/1248] Loss: 0.2896
Test Accuracy of NN: 90.80851063829788 % (improvement)
Ordinary Epoch [49/100], Step [500/1128] Loss: 0.2600
Ordinary Epoch [71/100], Step [500/1128] Loss: 0.3244
Test Accuracy of NN: 95.41346153846153 % Best: 95.75 %
Test Accuracy of NN: 90.38297872340425 % Best: 90.7127659574468 %
Test Accuracy of NN: 90.75 % Best: 90.80851063829786 %
Ordinary Epoch [51/100], Step [500/1248] Loss: 0.0753
Ordinary Epoch [50/100], Step [500/1128] Loss: 0.4839
Ordinary Epoch [72/100], Step [500/1128] Loss: 0.1275
Test Accuracy of NN: 95.73076923076923 % Best: 95.75 %
Test Accuracy of NN: 90.61170212765957 % Best: 90.7127659574468 %
Test Accuracy of NN: 90.54787234042553 % Best: 90.80851063829786 %
Ordinary Epoch [52/100], Step [500/1248] Loss: 0.1939
Ordinary Epoch [73/100], Step [500/1128] Loss: 0.3369
Ordinary Epoch [51/100], Step [500/1128] Loss: 0.5876
Test Accuracy of NN: 95.32211538461539 % Best: 95.75 %
Test Accuracy of NN: 90.79787234042553 % Best: 90.80851063829786 %
Test Accuracy of NN: 89.76595744680851 % Best: 90.7127659574468 %
Ordinary Epoch [53/100], Step [500/1248] Loss: 0.0825
Ordinary Epoch [74/100], Step [500/1128] Loss: 0.4291
Ordinary Epoch [52/100], Step [500/1128] Loss: 0.2237
Test Accuracy of NN: 95.69711538461539 % Best: 95.75 %
Test Accuracy of NN: 89.96808510638297 % Best: 90.80851063829786 %
Ordinary Epoch [54/100], Step [500/1248] Loss: 0.1683
Test Accuracy of NN: 90.37765957446808 % Best: 90.7127659574468 %
Ordinary Epoch [75/100], Step [500/1128] Loss: 0.2344
Ordinary Epoch [53/100], Step [500/1128] Loss: 0.2476
Test Accuracy of NN: 95.07692307692308 % Best: 95.75 %
Test Accuracy of NN: 90.47340425531915 % Best: 90.80851063829786 %
Ordinary Epoch [55/100], Step [500/1248] Loss: 0.1822
Ordinary Epoch [76/100], Step [500/1128] Loss: 0.5180
Test Accuracy of NN: 90.4627659574468 % Best: 90.7127659574468 %
Test Accuracy of NN: 90.45212765957447 % Best: 90.80851063829786 %
Test Accuracy of NN: 95.46153846153847 % Best: 95.75 %
Ordinary Epoch [54/100], Step [500/1128] Loss: 0.1980
Ordinary Epoch [77/100], Step [500/1128] Loss: 0.3338
Ordinary Epoch [56/100], Step [500/1248] Loss: 0.0882
Test Accuracy of NN: 90.57978723404256 % Best: 90.7127659574468 %
Test Accuracy of NN: 90.44148936170212 % Best: 90.80851063829786 %
Test Accuracy of NN: 95.78846153846153 % (improvement)
Ordinary Epoch [55/100], Step [500/1128] Loss: 0.3124
Ordinary Epoch [78/100], Step [500/1128] Loss: 0.3403
Ordinary Epoch [57/100], Step [500/1248] Loss: 0.1164
Test Accuracy of NN: 90.50531914893617 % Best: 90.80851063829786 %
Test Accuracy of NN: 90.5 % Best: 90.7127659574468 %
Test Accuracy of NN: 95.75961538461539 % Best: 95.78846153846153 %
Ordinary Epoch [79/100], Step [500/1128] Loss: 0.1427
Ordinary Epoch [58/100], Step [500/1248] Loss: 0.1367
Ordinary Epoch [56/100], Step [500/1128] Loss: 0.2091
Test Accuracy of NN: 90.72872340425532 % Best: 90.80851063829786 %
Test Accuracy of NN: 95.26442307692308 % Best: 95.78846153846153 %
Test Accuracy of NN: 90.52659574468085 % Best: 90.7127659574468 %
Ordinary Epoch [80/100], Step [500/1128] Loss: 0.4447
Ordinary Epoch [59/100], Step [500/1248] Loss: 0.1111
Ordinary Epoch [57/100], Step [500/1128] Loss: 0.1911
Test Accuracy of NN: 90.65957446808511 % Best: 90.80851063829786 %
Test Accuracy of NN: 95.60576923076923 % Best: 95.78846153846153 %
Ordinary Epoch [81/100], Step [500/1128] Loss: 0.1982
Test Accuracy of NN: 90.65425531914893 % Best: 90.7127659574468 %
Ordinary Epoch [60/100], Step [500/1248] Loss: 0.1088
Test Accuracy of NN: 90.30851063829788 % Best: 90.80851063829786 %
Ordinary Epoch [58/100], Step [500/1128] Loss: 0.2265
Test Accuracy of NN: 95.79326923076923 % (improvement)
Ordinary Epoch [82/100], Step [500/1128] Loss: 0.2397
Ordinary Epoch [61/100], Step [500/1248] Loss: 0.1670
Test Accuracy of NN: 90.2872340425532 % Best: 90.7127659574468 %
Test Accuracy of NN: 90.30851063829788 % Best: 90.80851063829786 %
Ordinary Epoch [59/100], Step [500/1128] Loss: 0.2350
Test Accuracy of NN: 95.85096153846153 % (improvement)
Ordinary Epoch [83/100], Step [500/1128] Loss: 0.2394
Ordinary Epoch [62/100], Step [500/1248] Loss: 0.2272
Test Accuracy of NN: 90.64893617021276 % Best: 90.80851063829786 %
Test Accuracy of NN: 90.48936170212765 % Best: 90.7127659574468 %
Test Accuracy of NN: 95.86057692307692 % (improvement)
Ordinary Epoch [84/100], Step [500/1128] Loss: 0.2927
Ordinary Epoch [60/100], Step [500/1128] Loss: 0.3903
Ordinary Epoch [63/100], Step [500/1248] Loss: 0.0655
Test Accuracy of NN: 90.61702127659575 % Best: 90.80851063829786 %
Test Accuracy of NN: 90.57446808510639 % Best: 90.7127659574468 %
Ordinary Epoch [85/100], Step [500/1128] Loss: 0.2427
Test Accuracy of NN: 95.82692307692308 % Best: 95.86057692307692 %
Ordinary Epoch [64/100], Step [500/1248] Loss: 0.1334
Ordinary Epoch [61/100], Step [500/1128] Loss: 0.2771
Test Accuracy of NN: 90.70744680851064 % Best: 90.80851063829786 %
Ordinary Epoch [86/100], Step [500/1128] Loss: 0.3152
Test Accuracy of NN: 95.85096153846153 % Best: 95.86057692307692 %
Test Accuracy of NN: 90.65957446808511 % Best: 90.7127659574468 %
Ordinary Epoch [65/100], Step [500/1248] Loss: 0.2008
Test Accuracy of NN: 90.73404255319149 % Best: 90.80851063829786 %
Ordinary Epoch [62/100], Step [500/1128] Loss: 0.1646
Ordinary Epoch [87/100], Step [500/1128] Loss: 0.2897
Test Accuracy of NN: 95.8076923076923 % Best: 95.86057692307692 %
Test Accuracy of NN: 90.80851063829788 % (improvement)
Ordinary Epoch [66/100], Step [500/1248] Loss: 0.1406
Test Accuracy of NN: 90.71808510638297 % Best: 90.80851063829786 %
Ordinary Epoch [63/100], Step [500/1128] Loss: 0.2536
Ordinary Epoch [88/100], Step [500/1128] Loss: 0.2254
Test Accuracy of NN: 95.8076923076923 % Best: 95.86057692307692 %
Ordinary Epoch [67/100], Step [500/1248] Loss: 0.0937
Test Accuracy of NN: 90.81382978723404 % (improvement)
Test Accuracy of NN: 90.80319148936171 % Best: 90.80851063829786 %
Ordinary Epoch [89/100], Step [500/1128] Loss: 0.2611
Ordinary Epoch [64/100], Step [500/1128] Loss: 0.3289
Test Accuracy of NN: 95.82692307692308 % Best: 95.86057692307692 %
Test Accuracy of NN: 90.87765957446808 % (improvement)
Ordinary Epoch [68/100], Step [500/1248] Loss: 0.1694
Test Accuracy of NN: 89.06914893617021 % Best: 90.80851063829786 %
Ordinary Epoch [90/100], Step [500/1128] Loss: 0.1433
Test Accuracy of NN: 95.51442307692308 % Best: 95.86057692307692 %
Ordinary Epoch [65/100], Step [500/1128] Loss: 0.2370
Test Accuracy of NN: 90.89893617021276 % (improvement)
Ordinary Epoch [69/100], Step [500/1248] Loss: 0.0951
Ordinary Epoch [91/100], Step [500/1128] Loss: 0.3372
Test Accuracy of NN: 90.5372340425532 % Best: 90.80851063829786 %
Test Accuracy of NN: 95.83173076923077 % Best: 95.86057692307692 %
Ordinary Epoch [66/100], Step [500/1128] Loss: 0.3447
Test Accuracy of NN: 90.7872340425532 % Best: 90.89893617021276 %
Ordinary Epoch [70/100], Step [500/1248] Loss: 0.2121
Ordinary Epoch [92/100], Step [500/1128] Loss: 0.2895
Test Accuracy of NN: 90.16489361702128 % Best: 90.80851063829786 %
Test Accuracy of NN: 95.79807692307692 % Best: 95.86057692307692 %
Test Accuracy of NN: 90.12234042553192 % Best: 90.89893617021276 %
Ordinary Epoch [71/100], Step [500/1248] Loss: 0.1127
Ordinary Epoch [67/100], Step [500/1128] Loss: 0.3118
Ordinary Epoch [93/100], Step [500/1128] Loss: 0.2789
Test Accuracy of NN: 95.84615384615384 % Best: 95.86057692307692 %
Test Accuracy of NN: 90.20212765957447 % Best: 90.89893617021276 %
Test Accuracy of NN: 90.15425531914893 % Best: 90.80851063829786 %
Ordinary Epoch [72/100], Step [500/1248] Loss: 0.1072
Ordinary Epoch [94/100], Step [500/1128] Loss: 0.3366
Ordinary Epoch [68/100], Step [500/1128] Loss: 0.1614
Test Accuracy of NN: 95.73076923076923 % Best: 95.86057692307692 %
Test Accuracy of NN: 89.84574468085107 % Best: 90.89893617021276 %
Test Accuracy of NN: 90.3563829787234 % Best: 90.80851063829786 %
Ordinary Epoch [73/100], Step [500/1248] Loss: 0.1540
Ordinary Epoch [95/100], Step [500/1128] Loss: 0.3316
Ordinary Epoch [69/100], Step [500/1128] Loss: 0.2696
Test Accuracy of NN: 89.95744680851064 % Best: 90.89893617021276 %
Test Accuracy of NN: 95.82692307692308 % Best: 95.86057692307692 %
Test Accuracy of NN: 90.06382978723404 % Best: 90.80851063829786 %
Ordinary Epoch [96/100], Step [500/1128] Loss: 0.2872
Ordinary Epoch [74/100], Step [500/1248] Loss: 0.1419
Ordinary Epoch [70/100], Step [500/1128] Loss: 0.1365
Test Accuracy of NN: 89.92553191489361 % Best: 90.89893617021276 %
Test Accuracy of NN: 95.82692307692308 % Best: 95.86057692307692 %
Ordinary Epoch [97/100], Step [500/1128] Loss: 0.2789
Ordinary Epoch [75/100], Step [500/1248] Loss: 0.0566
Test Accuracy of NN: 90.68617021276596 % Best: 90.80851063829786 %
Test Accuracy of NN: 89.74468085106383 % Best: 90.89893617021276 %
Ordinary Epoch [71/100], Step [500/1128] Loss: 0.3203
Test Accuracy of NN: 95.80288461538461 % Best: 95.86057692307692 %
Ordinary Epoch [98/100], Step [500/1128] Loss: 0.3141
Ordinary Epoch [76/100], Step [500/1248] Loss: 0.4799
Test Accuracy of NN: 90.14893617021276 % Best: 90.80851063829786 %
Test Accuracy of NN: 90.68617021276596 % Best: 90.89893617021276 %
Test Accuracy of NN: 95.17307692307692 % Best: 95.86057692307692 %
Ordinary Epoch [72/100], Step [500/1128] Loss: 0.2181
Ordinary Epoch [99/100], Step [500/1128] Loss: 0.3432
Ordinary Epoch [77/100], Step [500/1248] Loss: 0.0868
Test Accuracy of NN: 90.5 % Best: 90.89893617021276 %
Test Accuracy of NN: 90.32446808510639 % Best: 90.80851063829786 %
Test Accuracy of NN: 95.0625 % Best: 95.86057692307692 %
Ordinary Epoch [100/100], Step [500/1128] Loss: 0.2316
Ordinary Epoch [73/100], Step [500/1128] Loss: 0.2884
Ordinary Epoch [78/100], Step [500/1248] Loss: 0.0894
Test Accuracy of NN: 90.12234042553192 % Best: 90.89893617021276 %
Test Accuracy of NN: 90.55851063829788 % Best: 90.80851063829786 %
Test Accuracy of NN: 95.61057692307692 % Best: 95.86057692307692 %
Ordinary Epoch [79/100], Step [500/1248] Loss: 0.0920
Ordinary Epoch [74/100], Step [500/1128] Loss: 0.2265
Test Accuracy of NN: 95.76442307692308 % Best: 95.86057692307692 %
Test Accuracy of NN: 90.26063829787235 % Best: 90.80851063829786 %
Ordinary Epoch [80/100], Step [500/1248] Loss: 0.0404
Ordinary Epoch [75/100], Step [500/1128] Loss: 0.4236
Test Accuracy of NN: 95.625 % Best: 95.86057692307692 %
Test Accuracy of NN: 90.62765957446808 % Best: 90.80851063829786 %
Ordinary Epoch [81/100], Step [500/1248] Loss: 0.1086
Ordinary Epoch [76/100], Step [500/1128] Loss: 0.2842
Test Accuracy of NN: 95.64903846153847 % Best: 95.86057692307692 %
Ordinary Epoch [82/100], Step [500/1248] Loss: 0.0979
Test Accuracy of NN: 90.66489361702128 % Best: 90.80851063829786 %
Ordinary Epoch [77/100], Step [500/1128] Loss: 0.2232
Test Accuracy of NN: 95.51923076923077 % Best: 95.86057692307692 %
Ordinary Epoch [83/100], Step [500/1248] Loss: 0.1176
Test Accuracy of NN: 90.17021276595744 % Best: 90.80851063829786 %
Ordinary Epoch [78/100], Step [500/1128] Loss: 0.2766
Test Accuracy of NN: 95.27403846153847 % Best: 95.86057692307692 %
Ordinary Epoch [84/100], Step [500/1248] Loss: 0.0862
Test Accuracy of NN: 90.70212765957447 % Best: 90.80851063829786 %
Test Accuracy of NN: 95.42788461538461 % Best: 95.86057692307692 %
Ordinary Epoch [79/100], Step [500/1128] Loss: 0.1931
Ordinary Epoch [85/100], Step [500/1248] Loss: 0.1082
Test Accuracy of NN: 90.87234042553192 % (improvement)
Test Accuracy of NN: 95.6201923076923 % Best: 95.86057692307692 %
Ordinary Epoch [80/100], Step [500/1128] Loss: 0.2788
Ordinary Epoch [86/100], Step [500/1248] Loss: 0.0908
Test Accuracy of NN: 90.88297872340425 % (improvement)
Test Accuracy of NN: 95.79326923076923 % Best: 95.86057692307692 %
Ordinary Epoch [87/100], Step [500/1248] Loss: 0.1106
Ordinary Epoch [81/100], Step [500/1128] Loss: 0.3470
Test Accuracy of NN: 95.66826923076923 % Best: 95.86057692307692 %
Test Accuracy of NN: 90.8936170212766 % (improvement)
Ordinary Epoch [88/100], Step [500/1248] Loss: 0.0499
Ordinary Epoch [82/100], Step [500/1128] Loss: 0.2855
Test Accuracy of NN: 95.64423076923077 % Best: 95.86057692307692 %
Test Accuracy of NN: 90.89893617021276 % (improvement)
Ordinary Epoch [89/100], Step [500/1248] Loss: 0.0767
Ordinary Epoch [83/100], Step [500/1128] Loss: 0.3889
Test Accuracy of NN: 95.65865384615384 % Best: 95.86057692307692 %
Ordinary Epoch [90/100], Step [500/1248] Loss: 0.0756
Test Accuracy of NN: 90.98936170212765 % (improvement)
Test Accuracy of NN: 95.7451923076923 % Best: 95.86057692307692 %
Ordinary Epoch [84/100], Step [500/1128] Loss: 0.4724
Ordinary Epoch [91/100], Step [500/1248] Loss: 0.1327
Test Accuracy of NN: 90.94680851063829 % Best: 90.98936170212765 %
Test Accuracy of NN: 95.8173076923077 % Best: 95.86057692307692 %
Ordinary Epoch [85/100], Step [500/1128] Loss: 0.3055
Ordinary Epoch [92/100], Step [500/1248] Loss: 0.1361
Test Accuracy of NN: 90.83510638297872 % Best: 90.98936170212765 %
Test Accuracy of NN: 95.84134615384616 % Best: 95.86057692307692 %
Ordinary Epoch [93/100], Step [500/1248] Loss: 0.3224
Ordinary Epoch [86/100], Step [500/1128] Loss: 0.2097
Test Accuracy of NN: 95.1923076923077 % Best: 95.86057692307692 %
Test Accuracy of NN: 90.82446808510639 % Best: 90.98936170212765 %
Ordinary Epoch [94/100], Step [500/1248] Loss: 0.2323
Ordinary Epoch [87/100], Step [500/1128] Loss: 0.2768
Test Accuracy of NN: 95.38942307692308 % Best: 95.86057692307692 %
Test Accuracy of NN: 90.79255319148936 % Best: 90.98936170212765 %
Ordinary Epoch [95/100], Step [500/1248] Loss: 0.2449
Ordinary Epoch [88/100], Step [500/1128] Loss: 0.3263
Test Accuracy of NN: 95.20673076923077 % Best: 95.86057692307692 %
Test Accuracy of NN: 90.02127659574468 % Best: 90.98936170212765 %
Ordinary Epoch [96/100], Step [500/1248] Loss: 0.1403
Ordinary Epoch [89/100], Step [500/1128] Loss: 0.2725
Test Accuracy of NN: 94.88942307692308 % Best: 95.86057692307692 %
Ordinary Epoch [97/100], Step [500/1248] Loss: 0.0533
Test Accuracy of NN: 90.38297872340425 % Best: 90.98936170212765 %
Ordinary Epoch [90/100], Step [500/1128] Loss: 0.1906
Test Accuracy of NN: 95.58173076923077 % Best: 95.86057692307692 %
Ordinary Epoch [98/100], Step [500/1248] Loss: 0.0731
Test Accuracy of NN: 90.49468085106383 % Best: 90.98936170212765 %
Test Accuracy of NN: 95.65865384615384 % Best: 95.86057692307692 %
Ordinary Epoch [91/100], Step [500/1128] Loss: 0.2065
Ordinary Epoch [99/100], Step [500/1248] Loss: 0.0708
Test Accuracy of NN: 90.44148936170212 % Best: 90.98936170212765 %
Test Accuracy of NN: 95.625 % Best: 95.86057692307692 %
Ordinary Epoch [92/100], Step [500/1128] Loss: 0.1864
Ordinary Epoch [100/100], Step [500/1248] Loss: 0.1226
Test Accuracy of NN: 90.49468085106383 % Best: 90.98936170212765 %
Test Accuracy of NN: 95.5625 % Best: 95.86057692307692 %
Ordinary Epoch [93/100], Step [500/1128] Loss: 0.2125
Test Accuracy of NN: 90.75 % Best: 90.98936170212765 %
Ordinary Epoch [94/100], Step [500/1128] Loss: 0.2709
Test Accuracy of NN: 90.82446808510639 % Best: 90.98936170212765 %
Ordinary Epoch [95/100], Step [500/1128] Loss: 0.2578
Test Accuracy of NN: 90.94680851063829 % Best: 90.98936170212765 %
Ordinary Epoch [96/100], Step [500/1128] Loss: 0.2731
Test Accuracy of NN: 90.9627659574468 % Best: 90.98936170212765 %
Ordinary Epoch [97/100], Step [500/1128] Loss: 0.2895
Test Accuracy of NN: 90.8563829787234 % Best: 90.98936170212765 %
Ordinary Epoch [98/100], Step [500/1128] Loss: 0.1946
Test Accuracy of NN: 90.8563829787234 % Best: 90.98936170212765 %
Ordinary Epoch [99/100], Step [500/1128] Loss: 0.2600
Test Accuracy of NN: 90.85106382978724 % Best: 90.98936170212765 %
Ordinary Epoch [100/100], Step [500/1128] Loss: 0.3869
Test Accuracy of NN: 90.46808510638297 % Best: 90.98936170212765 %
