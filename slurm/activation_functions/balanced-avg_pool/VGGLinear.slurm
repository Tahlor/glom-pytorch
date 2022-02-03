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
    (4): AvgPool2d(kernel_size=2, stride=2, padding=0)
  )
  (l2): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): AvgPool2d(kernel_size=2, stride=2, padding=0)
  )
  (l3): Sequential(
    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): AvgPool2d(kernel_size=2, stride=2, padding=0)
  )
  (l4): Sequential(
    (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): AvgPool2d(kernel_size=2, stride=2, padding=0)
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
    (4): AvgPool2d(kernel_size=2, stride=2, padding=0)
  )
  (l2): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): AvgPool2d(kernel_size=2, stride=2, padding=0)
  )
  (l3): Sequential(
    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): AvgPool2d(kernel_size=2, stride=2, padding=0)
  )
  (l4): Sequential(
    (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): AvgPool2d(kernel_size=2, stride=2, padding=0)
  )
  (classifier): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=256, out_features=512, bias=True)
    (2): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=512, out_features=47, bias=True)
  )
)>
Ordinary Epoch [1/100], Step [500/1128] Loss: 2.7269
Test Accuracy of NN: 51.03191489361702 % (improvement)
Ordinary Epoch [2/100], Step [500/1128] Loss: 2.6050
Test Accuracy of NN: 53.648936170212764 % (improvement)
Ordinary Epoch [3/100], Step [500/1128] Loss: 2.4119
Test Accuracy of NN: 54.92553191489362 % (improvement)
Ordinary Epoch [4/100], Step [500/1128] Loss: 2.6318
Test Accuracy of NN: 55.21808510638298 % (improvement)
Ordinary Epoch [5/100], Step [500/1128] Loss: 2.3494
Test Accuracy of NN: 55.59574468085106 % (improvement)
Ordinary Epoch [6/100], Step [500/1128] Loss: 2.2499
Test Accuracy of NN: 56.6968085106383 % (improvement)
Ordinary Epoch [7/100], Step [500/1128] Loss: 2.8883
Test Accuracy of NN: 56.70744680851064 % (improvement)
Ordinary Epoch [8/100], Step [500/1128] Loss: 2.2340
Test Accuracy of NN: 57.34574468085106 % (improvement)
Ordinary Epoch [9/100], Step [500/1128] Loss: 2.4362
Test Accuracy of NN: 58.026595744680854 % (improvement)
Ordinary Epoch [10/100], Step [500/1128] Loss: 2.3030
/lustre/scratch/grp/fslg_internn/glom-pytorch/MNIST_vgg.py:121: DeprecationWarning: np.asscalar(a) is deprecated since NumPy v1.16, use a.item() instead
  curr_lr1 = learning_rate * np.asscalar(pow(np.random.rand(1), 3))
Test Accuracy of NN: 57.462765957446805 % Best: 58.026595744680854 %
Ordinary Epoch [11/100], Step [500/1128] Loss: 2.6035
Test Accuracy of NN: 60.712765957446805 % (improvement)
Ordinary Epoch [12/100], Step [500/1128] Loss: 2.3492
Test Accuracy of NN: 60.920212765957444 % (improvement)
Ordinary Epoch [13/100], Step [500/1128] Loss: 2.1082
Test Accuracy of NN: 61.22872340425532 % (improvement)
Ordinary Epoch [14/100], Step [500/1128] Loss: 2.5125
Test Accuracy of NN: 61.388297872340424 % (improvement)
Ordinary Epoch [15/100], Step [500/1128] Loss: 2.2862
Test Accuracy of NN: 61.223404255319146 % Best: 61.388297872340424 %
Ordinary Epoch [16/100], Step [500/1128] Loss: 2.6409
Test Accuracy of NN: 59.180851063829785 % Best: 61.388297872340424 %
Ordinary Epoch [17/100], Step [500/1128] Loss: 2.6286
Test Accuracy of NN: 58.96808510638298 % Best: 61.388297872340424 %
Ordinary Epoch [18/100], Step [500/1128] Loss: 2.3152
Test Accuracy of NN: 59.079787234042556 % Best: 61.388297872340424 %
Ordinary Epoch [19/100], Step [500/1128] Loss: 1.8800
Test Accuracy of NN: 59.297872340425535 % Best: 61.388297872340424 %
Ordinary Epoch [20/100], Step [500/1128] Loss: 2.3667
Test Accuracy of NN: 61.398936170212764 % (improvement)
Ordinary Epoch [21/100], Step [500/1128] Loss: 2.1128
Test Accuracy of NN: 61.691489361702125 % (improvement)
Ordinary Epoch [22/100], Step [500/1128] Loss: 2.4722
Test Accuracy of NN: 61.787234042553195 % (improvement)
Ordinary Epoch [23/100], Step [500/1128] Loss: 2.4492
Test Accuracy of NN: 61.659574468085104 % Best: 61.787234042553195 %
Ordinary Epoch [24/100], Step [500/1128] Loss: 2.1951
Test Accuracy of NN: 58.691489361702125 % Best: 61.787234042553195 %
Ordinary Epoch [25/100], Step [500/1128] Loss: 2.1217
Test Accuracy of NN: 61.62765957446808 % Best: 61.787234042553195 %
Ordinary Epoch [26/100], Step [500/1128] Loss: 1.9739
Test Accuracy of NN: 61.51063829787234 % Best: 61.787234042553195 %
Ordinary Epoch [27/100], Step [500/1128] Loss: 2.4905
Test Accuracy of NN: 61.12234042553192 % Best: 61.787234042553195 %
Ordinary Epoch [28/100], Step [500/1128] Loss: 2.3557
Test Accuracy of NN: 59.20744680851064 % Best: 61.787234042553195 %
Ordinary Epoch [29/100], Step [500/1128] Loss: 2.3980
Test Accuracy of NN: 61.319148936170215 % Best: 61.787234042553195 %
Ordinary Epoch [30/100], Step [500/1128] Loss: 2.2146
Test Accuracy of NN: 60.71808510638298 % Best: 61.787234042553195 %
Ordinary Epoch [31/100], Step [500/1128] Loss: 2.6561
Test Accuracy of NN: 61.53191489361702 % Best: 61.787234042553195 %
Ordinary Epoch [32/100], Step [500/1128] Loss: 2.6868
Test Accuracy of NN: 61.829787234042556 % (improvement)
Ordinary Epoch [33/100], Step [500/1128] Loss: 2.1464
Test Accuracy of NN: 61.95744680851064 % (improvement)
Ordinary Epoch [34/100], Step [500/1128] Loss: 2.5204
Test Accuracy of NN: 62.037234042553195 % (improvement)
Ordinary Epoch [35/100], Step [500/1128] Loss: 2.3200
Test Accuracy of NN: 62.11702127659574 % (improvement)
Ordinary Epoch [36/100], Step [500/1128] Loss: 2.0650
Test Accuracy of NN: 61.98936170212766 % Best: 62.11702127659574 %
Ordinary Epoch [37/100], Step [500/1128] Loss: 2.4257
Test Accuracy of NN: 61.308510638297875 % Best: 62.11702127659574 %
Ordinary Epoch [38/100], Step [500/1128] Loss: 2.4439
Test Accuracy of NN: 61.91489361702128 % Best: 62.11702127659574 %
Ordinary Epoch [39/100], Step [500/1128] Loss: 2.6441
Test Accuracy of NN: 62.22872340425532 % (improvement)
Ordinary Epoch [40/100], Step [500/1128] Loss: 2.0193
Test Accuracy of NN: 62.234042553191486 % (improvement)
Ordinary Epoch [41/100], Step [500/1128] Loss: 2.1155
Test Accuracy of NN: 62.22872340425532 % Best: 62.23404255319149 %
Ordinary Epoch [42/100], Step [500/1128] Loss: 2.6571
Test Accuracy of NN: 62.170212765957444 % Best: 62.23404255319149 %
Ordinary Epoch [43/100], Step [500/1128] Loss: 2.2312
Test Accuracy of NN: 62.29255319148936 % (improvement)
Ordinary Epoch [44/100], Step [500/1128] Loss: 2.2297
Test Accuracy of NN: 62.12234042553192 % Best: 62.29255319148936 %
Ordinary Epoch [45/100], Step [500/1128] Loss: 2.4414
Test Accuracy of NN: 62.27127659574468 % Best: 62.29255319148936 %
Ordinary Epoch [46/100], Step [500/1128] Loss: 2.5685
Test Accuracy of NN: 60.13297872340426 % Best: 62.29255319148936 %
Ordinary Epoch [47/100], Step [500/1128] Loss: 2.2261
Test Accuracy of NN: 62.287234042553195 % Best: 62.29255319148936 %
Ordinary Epoch [48/100], Step [500/1128] Loss: 2.1883
Test Accuracy of NN: 62.33510638297872 % (improvement)
Ordinary Epoch [49/100], Step [500/1128] Loss: 2.3636
Test Accuracy of NN: 62.20744680851064 % Best: 62.33510638297872 %
Ordinary Epoch [50/100], Step [500/1128] Loss: 2.2188
Test Accuracy of NN: 62.12234042553192 % Best: 62.33510638297872 %
Ordinary Epoch [51/100], Step [500/1128] Loss: 2.2179
Test Accuracy of NN: 61.8031914893617 % Best: 62.33510638297872 %
Ordinary Epoch [52/100], Step [500/1128] Loss: 2.4763
Test Accuracy of NN: 62.03191489361702 % Best: 62.33510638297872 %
Ordinary Epoch [53/100], Step [500/1128] Loss: 2.2899
Test Accuracy of NN: 59.819148936170215 % Best: 62.33510638297872 %
Ordinary Epoch [54/100], Step [500/1128] Loss: 2.4685
Test Accuracy of NN: 60.1063829787234 % Best: 62.33510638297872 %
Ordinary Epoch [55/100], Step [500/1128] Loss: 2.4227
Test Accuracy of NN: 61.41489361702128 % Best: 62.33510638297872 %
Ordinary Epoch [56/100], Step [500/1128] Loss: 2.3425
Test Accuracy of NN: 61.81382978723404 % Best: 62.33510638297872 %
Ordinary Epoch [57/100], Step [500/1128] Loss: 2.2546
Test Accuracy of NN: 62.16489361702128 % Best: 62.33510638297872 %
Ordinary Epoch [58/100], Step [500/1128] Loss: 2.4682
Test Accuracy of NN: 62.31382978723404 % Best: 62.33510638297872 %
Ordinary Epoch [59/100], Step [500/1128] Loss: 2.4049
Test Accuracy of NN: 62.420212765957444 % (improvement)
Ordinary Epoch [60/100], Step [500/1128] Loss: 2.1705
Test Accuracy of NN: 62.223404255319146 % Best: 62.42021276595745 %
Ordinary Epoch [61/100], Step [500/1128] Loss: 2.3748
Test Accuracy of NN: 62.361702127659576 % Best: 62.42021276595745 %
Ordinary Epoch [62/100], Step [500/1128] Loss: 2.3262
Test Accuracy of NN: 60.01063829787234 % Best: 62.42021276595745 %
Ordinary Epoch [63/100], Step [500/1128] Loss: 2.2566
Test Accuracy of NN: 58.058510638297875 % Best: 62.42021276595745 %
Ordinary Epoch [64/100], Step [500/1128] Loss: 2.6955
Test Accuracy of NN: 62.494680851063826 % (improvement)
Ordinary Epoch [65/100], Step [500/1128] Loss: 2.1623
Test Accuracy of NN: 62.09574468085106 % Best: 62.49468085106383 %
Ordinary Epoch [66/100], Step [500/1128] Loss: 2.2884
Test Accuracy of NN: 62.473404255319146 % Best: 62.49468085106383 %
Ordinary Epoch [67/100], Step [500/1128] Loss: 2.2035
Test Accuracy of NN: 60.590425531914896 % Best: 62.49468085106383 %
Ordinary Epoch [68/100], Step [500/1128] Loss: 2.1836
Test Accuracy of NN: 61.170212765957444 % Best: 62.49468085106383 %
Ordinary Epoch [69/100], Step [500/1128] Loss: 2.2122
Test Accuracy of NN: 60.76063829787234 % Best: 62.49468085106383 %
Ordinary Epoch [70/100], Step [500/1128] Loss: 2.4301
Test Accuracy of NN: 62.388297872340424 % Best: 62.49468085106383 %
Ordinary Epoch [71/100], Step [500/1128] Loss: 2.3390
Test Accuracy of NN: 60.26063829787234 % Best: 62.49468085106383 %
Ordinary Epoch [72/100], Step [500/1128] Loss: 2.2674
Test Accuracy of NN: 60.98936170212766 % Best: 62.49468085106383 %
Ordinary Epoch [73/100], Step [500/1128] Loss: 2.2204
Test Accuracy of NN: 60.351063829787236 % Best: 62.49468085106383 %
Ordinary Epoch [74/100], Step [500/1128] Loss: 2.1422
Test Accuracy of NN: 61.72872340425532 % Best: 62.49468085106383 %
Ordinary Epoch [75/100], Step [500/1128] Loss: 2.2698
Test Accuracy of NN: 62.526595744680854 % (improvement)
Ordinary Epoch [76/100], Step [500/1128] Loss: 2.3657
Test Accuracy of NN: 62.223404255319146 % Best: 62.52659574468085 %
Ordinary Epoch [77/100], Step [500/1128] Loss: 2.4137
Test Accuracy of NN: 62.57446808510638 % (improvement)
Ordinary Epoch [78/100], Step [500/1128] Loss: 2.7006
Test Accuracy of NN: 62.84574468085106 % (improvement)
Ordinary Epoch [79/100], Step [500/1128] Loss: 2.0425
Test Accuracy of NN: 62.71808510638298 % Best: 62.84574468085107 %
Ordinary Epoch [80/100], Step [500/1128] Loss: 2.3725
Test Accuracy of NN: 62.441489361702125 % Best: 62.84574468085107 %
Ordinary Epoch [81/100], Step [500/1128] Loss: 2.2632
Test Accuracy of NN: 62.829787234042556 % Best: 62.84574468085107 %
Ordinary Epoch [82/100], Step [500/1128] Loss: 2.1449
Test Accuracy of NN: 62.819148936170215 % Best: 62.84574468085107 %
Ordinary Epoch [83/100], Step [500/1128] Loss: 2.3158
Test Accuracy of NN: 62.8936170212766 % (improvement)
Ordinary Epoch [84/100], Step [500/1128] Loss: 2.2890
Test Accuracy of NN: 62.909574468085104 % (improvement)
Ordinary Epoch [85/100], Step [500/1128] Loss: 2.2154
Test Accuracy of NN: 62.84574468085106 % Best: 62.909574468085104 %
Ordinary Epoch [86/100], Step [500/1128] Loss: 2.3149
Test Accuracy of NN: 62.87234042553192 % Best: 62.909574468085104 %
Ordinary Epoch [87/100], Step [500/1128] Loss: 2.1786
Test Accuracy of NN: 62.37765957446808 % Best: 62.909574468085104 %
Ordinary Epoch [88/100], Step [500/1128] Loss: 2.1769
Test Accuracy of NN: 59.840425531914896 % Best: 62.909574468085104 %
Ordinary Epoch [89/100], Step [500/1128] Loss: 2.0745
Test Accuracy of NN: 61.611702127659576 % Best: 62.909574468085104 %
Ordinary Epoch [90/100], Step [500/1128] Loss: 2.3245
Test Accuracy of NN: 61.70744680851064 % Best: 62.909574468085104 %
Ordinary Epoch [91/100], Step [500/1128] Loss: 2.3673
Test Accuracy of NN: 62.1968085106383 % Best: 62.909574468085104 %
Ordinary Epoch [92/100], Step [500/1128] Loss: 2.3099
Test Accuracy of NN: 62.579787234042556 % Best: 62.909574468085104 %
Ordinary Epoch [93/100], Step [500/1128] Loss: 2.6459
Test Accuracy of NN: 60.67553191489362 % Best: 62.909574468085104 %
Ordinary Epoch [94/100], Step [500/1128] Loss: 2.1141
Test Accuracy of NN: 61.851063829787236 % Best: 62.909574468085104 %
Ordinary Epoch [95/100], Step [500/1128] Loss: 2.1940
Test Accuracy of NN: 60.43617021276596 % Best: 62.909574468085104 %
Ordinary Epoch [96/100], Step [500/1128] Loss: 2.3576
Test Accuracy of NN: 61.079787234042556 % Best: 62.909574468085104 %
Ordinary Epoch [97/100], Step [500/1128] Loss: 2.2099
Test Accuracy of NN: 62.287234042553195 % Best: 62.909574468085104 %
Ordinary Epoch [98/100], Step [500/1128] Loss: 2.3267
Test Accuracy of NN: 62.32446808510638 % Best: 62.909574468085104 %
Ordinary Epoch [99/100], Step [500/1128] Loss: 1.9856
Test Accuracy of NN: 62.57446808510638 % Best: 62.909574468085104 %
Ordinary Epoch [100/100], Step [500/1128] Loss: 1.9687
Test Accuracy of NN: 62.57446808510638 % Best: 62.909574468085104 %
