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
    (2): AvgPool2d(kernel_size=2, stride=2, padding=0)
  )
  (l2): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): AvgPool2d(kernel_size=2, stride=2, padding=0)
  )
  (l3): Sequential(
    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): AvgPool2d(kernel_size=2, stride=2, padding=0)
  )
  (l4): Sequential(
    (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): AvgPool2d(kernel_size=2, stride=2, padding=0)
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
    (2): AvgPool2d(kernel_size=2, stride=2, padding=0)
  )
  (l2): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): AvgPool2d(kernel_size=2, stride=2, padding=0)
  )
  (l3): Sequential(
    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): AvgPool2d(kernel_size=2, stride=2, padding=0)
  )
  (l4): Sequential(
    (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): AvgPool2d(kernel_size=2, stride=2, padding=0)
  )
  (classifier): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=256, out_features=512, bias=True)
    (2): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=512, out_features=47, bias=True)
  )
)>
Ordinary Epoch [1/100], Step [500/1128] Loss: 2.6747
Test Accuracy of NN: 54.170212765957444 % (improvement)
Ordinary Epoch [2/100], Step [500/1128] Loss: 2.5573
Test Accuracy of NN: 56.97872340425532 % (improvement)
Ordinary Epoch [3/100], Step [500/1128] Loss: 2.6142
/lustre/scratch/grp/fslg_internn/glom-pytorch/MNIST_vgg.py:121: DeprecationWarning: np.asscalar(a) is deprecated since NumPy v1.16, use a.item() instead
  curr_lr1 = learning_rate * np.asscalar(pow(np.random.rand(1), 3))
Test Accuracy of NN: 55.744680851063826 % Best: 56.97872340425532 %
Ordinary Epoch [4/100], Step [500/1128] Loss: 2.0348
Test Accuracy of NN: 57.65425531914894 % (improvement)
Ordinary Epoch [5/100], Step [500/1128] Loss: 2.4591
Test Accuracy of NN: 59.25 % (improvement)
Ordinary Epoch [6/100], Step [500/1128] Loss: 2.4237
Test Accuracy of NN: 59.34574468085106 % (improvement)
Ordinary Epoch [7/100], Step [500/1128] Loss: 1.9718
Test Accuracy of NN: 59.5531914893617 % (improvement)
Ordinary Epoch [8/100], Step [500/1128] Loss: 2.5724
Test Accuracy of NN: 58.75 % Best: 59.5531914893617 %
Ordinary Epoch [9/100], Step [500/1128] Loss: 2.3134
Test Accuracy of NN: 61.20744680851064 % (improvement)
Ordinary Epoch [10/100], Step [500/1128] Loss: 2.3462
Test Accuracy of NN: 61.340425531914896 % (improvement)
Ordinary Epoch [11/100], Step [500/1128] Loss: 2.2390
Test Accuracy of NN: 61.58510638297872 % (improvement)
Ordinary Epoch [12/100], Step [500/1128] Loss: 2.4770
Test Accuracy of NN: 61.62234042553192 % (improvement)
Ordinary Epoch [13/100], Step [500/1128] Loss: 2.0669
Test Accuracy of NN: 61.787234042553195 % (improvement)
Ordinary Epoch [14/100], Step [500/1128] Loss: 2.0353
Test Accuracy of NN: 61.829787234042556 % (improvement)
Ordinary Epoch [15/100], Step [500/1128] Loss: 2.2553
Test Accuracy of NN: 61.77127659574468 % Best: 61.829787234042556 %
Ordinary Epoch [16/100], Step [500/1128] Loss: 2.2217
Test Accuracy of NN: 61.66489361702128 % Best: 61.829787234042556 %
Ordinary Epoch [17/100], Step [500/1128] Loss: 2.5183
Test Accuracy of NN: 61.29255319148936 % Best: 61.829787234042556 %
Ordinary Epoch [18/100], Step [500/1128] Loss: 2.4111
Test Accuracy of NN: 61.82446808510638 % Best: 61.829787234042556 %
Ordinary Epoch [19/100], Step [500/1128] Loss: 2.2840
Test Accuracy of NN: 61.994680851063826 % (improvement)
Ordinary Epoch [20/100], Step [500/1128] Loss: 2.6714
Test Accuracy of NN: 61.888297872340424 % Best: 61.99468085106383 %
Ordinary Epoch [21/100], Step [500/1128] Loss: 2.6147
Test Accuracy of NN: 61.95744680851064 % Best: 61.99468085106383 %
Ordinary Epoch [22/100], Step [500/1128] Loss: 2.2910
Test Accuracy of NN: 61.63297872340426 % Best: 61.99468085106383 %
Ordinary Epoch [23/100], Step [500/1128] Loss: 2.1269
Test Accuracy of NN: 62.09574468085106 % (improvement)
Ordinary Epoch [24/100], Step [500/1128] Loss: 2.2428
Test Accuracy of NN: 62.18617021276596 % (improvement)
Ordinary Epoch [25/100], Step [500/1128] Loss: 2.4950
Test Accuracy of NN: 62.111702127659576 % Best: 62.18617021276596 %
Ordinary Epoch [26/100], Step [500/1128] Loss: 2.2135
Test Accuracy of NN: 62.1436170212766 % Best: 62.18617021276596 %
Ordinary Epoch [27/100], Step [500/1128] Loss: 2.2182
Test Accuracy of NN: 62.21808510638298 % (improvement)
Ordinary Epoch [28/100], Step [500/1128] Loss: 2.2708
Test Accuracy of NN: 62.265957446808514 % (improvement)
Ordinary Epoch [29/100], Step [500/1128] Loss: 2.2250
Test Accuracy of NN: 62.361702127659576 % (improvement)
Ordinary Epoch [30/100], Step [500/1128] Loss: 2.2146
Test Accuracy of NN: 62.4468085106383 % (improvement)
Ordinary Epoch [31/100], Step [500/1128] Loss: 2.3291
Test Accuracy of NN: 62.32446808510638 % Best: 62.4468085106383 %
Ordinary Epoch [32/100], Step [500/1128] Loss: 2.4600
Test Accuracy of NN: 60.47872340425532 % Best: 62.4468085106383 %
Ordinary Epoch [33/100], Step [500/1128] Loss: 2.5572
Test Accuracy of NN: 59.909574468085104 % Best: 62.4468085106383 %
Ordinary Epoch [34/100], Step [500/1128] Loss: 2.3225
Test Accuracy of NN: 61.297872340425535 % Best: 62.4468085106383 %
Ordinary Epoch [35/100], Step [500/1128] Loss: 2.3800
Test Accuracy of NN: 61.87765957446808 % Best: 62.4468085106383 %
Ordinary Epoch [36/100], Step [500/1128] Loss: 2.4981
Test Accuracy of NN: 61.71808510638298 % Best: 62.4468085106383 %
Ordinary Epoch [37/100], Step [500/1128] Loss: 2.3179
Test Accuracy of NN: 60.46808510638298 % Best: 62.4468085106383 %
Ordinary Epoch [38/100], Step [500/1128] Loss: 2.4172
Test Accuracy of NN: 60.73936170212766 % Best: 62.4468085106383 %
Ordinary Epoch [39/100], Step [500/1128] Loss: 2.2209
Test Accuracy of NN: 61.54255319148936 % Best: 62.4468085106383 %
Ordinary Epoch [40/100], Step [500/1128] Loss: 2.5162
Test Accuracy of NN: 61.90425531914894 % Best: 62.4468085106383 %
Ordinary Epoch [41/100], Step [500/1128] Loss: 2.0982
Test Accuracy of NN: 62.058510638297875 % Best: 62.4468085106383 %
Ordinary Epoch [42/100], Step [500/1128] Loss: 2.0835
Test Accuracy of NN: 60.984042553191486 % Best: 62.4468085106383 %
Ordinary Epoch [43/100], Step [500/1128] Loss: 2.4921
Test Accuracy of NN: 62.202127659574465 % Best: 62.4468085106383 %
Ordinary Epoch [44/100], Step [500/1128] Loss: 2.4474
Test Accuracy of NN: 62.4468085106383 % Best: 62.4468085106383 %
Ordinary Epoch [45/100], Step [500/1128] Loss: 2.4344
Test Accuracy of NN: 62.047872340425535 % Best: 62.4468085106383 %
Ordinary Epoch [46/100], Step [500/1128] Loss: 2.4620
Test Accuracy of NN: 61.420212765957444 % Best: 62.4468085106383 %
Ordinary Epoch [47/100], Step [500/1128] Loss: 2.4193
Test Accuracy of NN: 59.98936170212766 % Best: 62.4468085106383 %
Ordinary Epoch [48/100], Step [500/1128] Loss: 2.2084
Test Accuracy of NN: 62.111702127659576 % Best: 62.4468085106383 %
Ordinary Epoch [49/100], Step [500/1128] Loss: 2.4212
Test Accuracy of NN: 60.28191489361702 % Best: 62.4468085106383 %
Ordinary Epoch [50/100], Step [500/1128] Loss: 2.4841
Test Accuracy of NN: 61.9468085106383 % Best: 62.4468085106383 %
Ordinary Epoch [51/100], Step [500/1128] Loss: 2.1861
Test Accuracy of NN: 62.29255319148936 % Best: 62.4468085106383 %
Ordinary Epoch [52/100], Step [500/1128] Loss: 2.2788
Test Accuracy of NN: 62.6436170212766 % (improvement)
Ordinary Epoch [53/100], Step [500/1128] Loss: 2.2817
Test Accuracy of NN: 62.601063829787236 % Best: 62.6436170212766 %
Ordinary Epoch [54/100], Step [500/1128] Loss: 2.4635
Test Accuracy of NN: 62.54255319148936 % Best: 62.6436170212766 %
Ordinary Epoch [55/100], Step [500/1128] Loss: 2.1773
Test Accuracy of NN: 62.723404255319146 % (improvement)
Ordinary Epoch [56/100], Step [500/1128] Loss: 2.1162
Test Accuracy of NN: 62.73936170212766 % (improvement)
Ordinary Epoch [57/100], Step [500/1128] Loss: 2.1275
Test Accuracy of NN: 62.8563829787234 % (improvement)
Ordinary Epoch [58/100], Step [500/1128] Loss: 2.2703
Test Accuracy of NN: 62.73936170212766 % Best: 62.8563829787234 %
Ordinary Epoch [59/100], Step [500/1128] Loss: 1.8842
Test Accuracy of NN: 62.712765957446805 % Best: 62.8563829787234 %
Ordinary Epoch [60/100], Step [500/1128] Loss: 2.1623
Test Accuracy of NN: 61.537234042553195 % Best: 62.8563829787234 %
Ordinary Epoch [61/100], Step [500/1128] Loss: 2.4462
Test Accuracy of NN: 61.97872340425532 % Best: 62.8563829787234 %
Ordinary Epoch [62/100], Step [500/1128] Loss: 2.4375
Test Accuracy of NN: 61.13297872340426 % Best: 62.8563829787234 %
Ordinary Epoch [63/100], Step [500/1128] Loss: 2.1921
Test Accuracy of NN: 62.579787234042556 % Best: 62.8563829787234 %
Ordinary Epoch [64/100], Step [500/1128] Loss: 2.3155
Test Accuracy of NN: 62.68617021276596 % Best: 62.8563829787234 %
Ordinary Epoch [65/100], Step [500/1128] Loss: 2.3344
Test Accuracy of NN: 62.67553191489362 % Best: 62.8563829787234 %
Ordinary Epoch [66/100], Step [500/1128] Loss: 1.9695
Test Accuracy of NN: 62.52127659574468 % Best: 62.8563829787234 %
Ordinary Epoch [67/100], Step [500/1128] Loss: 2.3210
Test Accuracy of NN: 61.340425531914896 % Best: 62.8563829787234 %
Ordinary Epoch [68/100], Step [500/1128] Loss: 2.4704
Test Accuracy of NN: 61.851063829787236 % Best: 62.8563829787234 %
Ordinary Epoch [69/100], Step [500/1128] Loss: 2.4192
Test Accuracy of NN: 62.388297872340424 % Best: 62.8563829787234 %
Ordinary Epoch [70/100], Step [500/1128] Loss: 2.0740
Test Accuracy of NN: 61.61702127659574 % Best: 62.8563829787234 %
Ordinary Epoch [71/100], Step [500/1128] Loss: 2.2211
Test Accuracy of NN: 62.62234042553192 % Best: 62.8563829787234 %
Ordinary Epoch [72/100], Step [500/1128] Loss: 2.1762
Test Accuracy of NN: 62.65425531914894 % Best: 62.8563829787234 %
Ordinary Epoch [73/100], Step [500/1128] Loss: 2.4989
Test Accuracy of NN: 62.62234042553192 % Best: 62.8563829787234 %
Ordinary Epoch [74/100], Step [500/1128] Loss: 2.2771
Test Accuracy of NN: 62.755319148936174 % Best: 62.8563829787234 %
Ordinary Epoch [75/100], Step [500/1128] Loss: 2.4098
Test Accuracy of NN: 62.37765957446808 % Best: 62.8563829787234 %
Ordinary Epoch [76/100], Step [500/1128] Loss: 2.2398
Test Accuracy of NN: 62.648936170212764 % Best: 62.8563829787234 %
Ordinary Epoch [77/100], Step [500/1128] Loss: 2.3785
Test Accuracy of NN: 61.57446808510638 % Best: 62.8563829787234 %
Ordinary Epoch [78/100], Step [500/1128] Loss: 2.1826
Test Accuracy of NN: 62.5531914893617 % Best: 62.8563829787234 %
Ordinary Epoch [79/100], Step [500/1128] Loss: 2.5988
Test Accuracy of NN: 62.97872340425532 % (improvement)
Ordinary Epoch [80/100], Step [500/1128] Loss: 2.2376
Test Accuracy of NN: 62.819148936170215 % Best: 62.97872340425532 %
Ordinary Epoch [81/100], Step [500/1128] Loss: 2.7058
Test Accuracy of NN: 62.851063829787236 % Best: 62.97872340425532 %
Ordinary Epoch [82/100], Step [500/1128] Loss: 1.9514
Test Accuracy of NN: 62.3936170212766 % Best: 62.97872340425532 %
Ordinary Epoch [83/100], Step [500/1128] Loss: 2.3749
Test Accuracy of NN: 62.48936170212766 % Best: 62.97872340425532 %
Ordinary Epoch [84/100], Step [500/1128] Loss: 2.5392
Test Accuracy of NN: 59.81382978723404 % Best: 62.97872340425532 %
Ordinary Epoch [85/100], Step [500/1128] Loss: 2.2246
Test Accuracy of NN: 60.590425531914896 % Best: 62.97872340425532 %
Ordinary Epoch [86/100], Step [500/1128] Loss: 2.3277
Test Accuracy of NN: 62.13297872340426 % Best: 62.97872340425532 %
Ordinary Epoch [87/100], Step [500/1128] Loss: 2.3263
Test Accuracy of NN: 62.01063829787234 % Best: 62.97872340425532 %
Ordinary Epoch [88/100], Step [500/1128] Loss: 2.3601
Test Accuracy of NN: 60.888297872340424 % Best: 62.97872340425532 %
Ordinary Epoch [89/100], Step [500/1128] Loss: 2.4160
Test Accuracy of NN: 62.37765957446808 % Best: 62.97872340425532 %
Ordinary Epoch [90/100], Step [500/1128] Loss: 2.4905
Test Accuracy of NN: 62.5 % Best: 62.97872340425532 %
Ordinary Epoch [91/100], Step [500/1128] Loss: 2.4381
Test Accuracy of NN: 61.79255319148936 % Best: 62.97872340425532 %
Ordinary Epoch [92/100], Step [500/1128] Loss: 2.4243
Test Accuracy of NN: 62.648936170212764 % Best: 62.97872340425532 %
Ordinary Epoch [93/100], Step [500/1128] Loss: 2.1801
Test Accuracy of NN: 62.65425531914894 % Best: 62.97872340425532 %
Ordinary Epoch [94/100], Step [500/1128] Loss: 2.1804
Test Accuracy of NN: 62.930851063829785 % Best: 62.97872340425532 %
Ordinary Epoch [95/100], Step [500/1128] Loss: 2.1468
Test Accuracy of NN: 62.819148936170215 % Best: 62.97872340425532 %
Ordinary Epoch [96/100], Step [500/1128] Loss: 2.2573
Test Accuracy of NN: 62.52127659574468 % Best: 62.97872340425532 %
Ordinary Epoch [97/100], Step [500/1128] Loss: 2.1550
Test Accuracy of NN: 60.1968085106383 % Best: 62.97872340425532 %
Ordinary Epoch [98/100], Step [500/1128] Loss: 2.3642
Test Accuracy of NN: 62.462765957446805 % Best: 62.97872340425532 %
Ordinary Epoch [99/100], Step [500/1128] Loss: 2.1714
Test Accuracy of NN: 62.505319148936174 % Best: 62.97872340425532 %
Ordinary Epoch [100/100], Step [500/1128] Loss: 2.3565
Test Accuracy of NN: 60.6968085106383 % Best: 62.97872340425532 %
