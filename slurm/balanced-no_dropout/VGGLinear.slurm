/lustre/scratch/grp/fslg_internn/env/internn/bin/python
torch.Size([1000, 1, 28, 28])
max label 46
<class 'models.VGGLinear'>
Parameters 3665135
<bound method VGGLinear.two_conv_pool of VGGLinear(
  (dropout): Identity()
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
    (0): Identity()
    (1): Linear(in_features=256, out_features=512, bias=True)
    (2): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Identity()
    (4): Linear(in_features=512, out_features=47, bias=True)
  )
)> <bound method VGGLinear.three_conv_pool of VGGLinear(
  (dropout): Identity()
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
    (0): Identity()
    (1): Linear(in_features=256, out_features=512, bias=True)
    (2): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Identity()
    (4): Linear(in_features=512, out_features=47, bias=True)
  )
)>
Ordinary Epoch [1/100], Step [500/1128] Loss: 1.1351
Test Accuracy of NN: 82.76595744680851 % (improvement)
Ordinary Epoch [2/100], Step [500/1128] Loss: 0.8137
Test Accuracy of NN: 84.0 % (improvement)
Ordinary Epoch [3/100], Step [500/1128] Loss: 0.4215
Test Accuracy of NN: 85.31914893617021 % (improvement)
Ordinary Epoch [4/100], Step [500/1128] Loss: 0.4561
/lustre/scratch/grp/fslg_internn/glom-pytorch/MNIST_vgg.py:121: DeprecationWarning: np.asscalar(a) is deprecated since NumPy v1.16, use a.item() instead
  curr_lr1 = learning_rate * np.asscalar(pow(np.random.rand(1), 3))
Test Accuracy of NN: 83.73404255319149 % Best: 85.31914893617021 %
Ordinary Epoch [5/100], Step [500/1128] Loss: 0.3423
Test Accuracy of NN: 88.7872340425532 % (improvement)
Ordinary Epoch [6/100], Step [500/1128] Loss: 0.4252
Test Accuracy of NN: 88.95744680851064 % (improvement)
Ordinary Epoch [7/100], Step [500/1128] Loss: 0.3899
Test Accuracy of NN: 88.73936170212765 % Best: 88.95744680851064 %
Ordinary Epoch [8/100], Step [500/1128] Loss: 0.3256
Test Accuracy of NN: 88.79787234042553 % Best: 88.95744680851064 %
Ordinary Epoch [9/100], Step [500/1128] Loss: 0.5120
Test Accuracy of NN: 89.3563829787234 % (improvement)
Ordinary Epoch [10/100], Step [500/1128] Loss: 0.4309
Test Accuracy of NN: 89.50531914893617 % (improvement)
Ordinary Epoch [11/100], Step [500/1128] Loss: 0.3165
Test Accuracy of NN: 89.56914893617021 % (improvement)
Ordinary Epoch [12/100], Step [500/1128] Loss: 0.2817
Test Accuracy of NN: 89.59042553191489 % (improvement)
Ordinary Epoch [13/100], Step [500/1128] Loss: 0.2577
Test Accuracy of NN: 89.61170212765957 % (improvement)
Ordinary Epoch [14/100], Step [500/1128] Loss: 0.3948
Test Accuracy of NN: 89.73936170212765 % (improvement)
Ordinary Epoch [15/100], Step [500/1128] Loss: 0.3793
Test Accuracy of NN: 89.63829787234043 % Best: 89.73936170212767 %
Ordinary Epoch [16/100], Step [500/1128] Loss: 0.4872
Test Accuracy of NN: 87.20212765957447 % Best: 89.73936170212767 %
Ordinary Epoch [17/100], Step [500/1128] Loss: 0.2958
Test Accuracy of NN: 87.22872340425532 % Best: 89.73936170212767 %
Ordinary Epoch [18/100], Step [500/1128] Loss: 0.3557
Test Accuracy of NN: 87.85106382978724 % Best: 89.73936170212767 %
Ordinary Epoch [19/100], Step [500/1128] Loss: 0.3489
Test Accuracy of NN: 89.52659574468085 % Best: 89.73936170212767 %
Ordinary Epoch [20/100], Step [500/1128] Loss: 0.2971
Test Accuracy of NN: 89.53191489361703 % Best: 89.73936170212767 %
Ordinary Epoch [21/100], Step [500/1128] Loss: 0.4166
Test Accuracy of NN: 89.34042553191489 % Best: 89.73936170212767 %
Ordinary Epoch [22/100], Step [500/1128] Loss: 0.3985
Test Accuracy of NN: 89.58510638297872 % Best: 89.73936170212767 %
Ordinary Epoch [23/100], Step [500/1128] Loss: 0.4952
Test Accuracy of NN: 88.5 % Best: 89.73936170212767 %
Ordinary Epoch [24/100], Step [500/1128] Loss: 0.3571
Test Accuracy of NN: 88.08510638297872 % Best: 89.73936170212767 %
Ordinary Epoch [25/100], Step [500/1128] Loss: 0.1914
Test Accuracy of NN: 89.3563829787234 % Best: 89.73936170212767 %
Ordinary Epoch [26/100], Step [500/1128] Loss: 0.4086
Test Accuracy of NN: 88.42021276595744 % Best: 89.73936170212767 %
Ordinary Epoch [27/100], Step [500/1128] Loss: 0.2990
Test Accuracy of NN: 89.6063829787234 % Best: 89.73936170212767 %
Ordinary Epoch [28/100], Step [500/1128] Loss: 0.2716
Test Accuracy of NN: 87.76595744680851 % Best: 89.73936170212767 %
Ordinary Epoch [29/100], Step [500/1128] Loss: 0.3772
Test Accuracy of NN: 87.62765957446808 % Best: 89.73936170212767 %
Ordinary Epoch [30/100], Step [500/1128] Loss: 0.2848
Test Accuracy of NN: 89.71808510638297 % Best: 89.73936170212767 %
Ordinary Epoch [31/100], Step [500/1128] Loss: 0.2307
Test Accuracy of NN: 89.23404255319149 % Best: 89.73936170212767 %
Ordinary Epoch [32/100], Step [500/1128] Loss: 0.3683
Test Accuracy of NN: 89.60106382978724 % Best: 89.73936170212767 %
Ordinary Epoch [33/100], Step [500/1128] Loss: 0.2450
Test Accuracy of NN: 88.71808510638297 % Best: 89.73936170212767 %
Ordinary Epoch [34/100], Step [500/1128] Loss: 0.2936
Test Accuracy of NN: 89.67021276595744 % Best: 89.73936170212767 %
Ordinary Epoch [35/100], Step [500/1128] Loss: 0.3723
Test Accuracy of NN: 88.13829787234043 % Best: 89.73936170212767 %
Ordinary Epoch [36/100], Step [500/1128] Loss: 0.2410
Test Accuracy of NN: 89.18617021276596 % Best: 89.73936170212767 %
Ordinary Epoch [37/100], Step [500/1128] Loss: 0.3460
Test Accuracy of NN: 89.92553191489361 % (improvement)
Ordinary Epoch [38/100], Step [500/1128] Loss: 0.3771
Test Accuracy of NN: 89.90957446808511 % Best: 89.92553191489363 %
Ordinary Epoch [39/100], Step [500/1128] Loss: 0.3693
Test Accuracy of NN: 88.95744680851064 % Best: 89.92553191489363 %
Ordinary Epoch [40/100], Step [500/1128] Loss: 0.2565
Test Accuracy of NN: 89.86702127659575 % Best: 89.92553191489363 %
Ordinary Epoch [41/100], Step [500/1128] Loss: 0.1734
Test Accuracy of NN: 89.9627659574468 % (improvement)
Ordinary Epoch [42/100], Step [500/1128] Loss: 0.3151
Test Accuracy of NN: 90.1063829787234 % (improvement)
Ordinary Epoch [43/100], Step [500/1128] Loss: 0.1775
Test Accuracy of NN: 90.18085106382979 % (improvement)
Ordinary Epoch [44/100], Step [500/1128] Loss: 0.3204
Test Accuracy of NN: 90.16489361702128 % Best: 90.18085106382979 %
Ordinary Epoch [45/100], Step [500/1128] Loss: 0.3627
Test Accuracy of NN: 89.41489361702128 % Best: 90.18085106382979 %
Ordinary Epoch [46/100], Step [500/1128] Loss: 0.2374
Test Accuracy of NN: 89.93085106382979 % Best: 90.18085106382979 %
Ordinary Epoch [47/100], Step [500/1128] Loss: 0.3127
Test Accuracy of NN: 89.54255319148936 % Best: 90.18085106382979 %
Ordinary Epoch [48/100], Step [500/1128] Loss: 0.4109
Test Accuracy of NN: 89.71808510638297 % Best: 90.18085106382979 %
Ordinary Epoch [49/100], Step [500/1128] Loss: 0.2909
Test Accuracy of NN: 90.15957446808511 % Best: 90.18085106382979 %
Ordinary Epoch [50/100], Step [500/1128] Loss: 0.2474
Test Accuracy of NN: 89.66489361702128 % Best: 90.18085106382979 %
Ordinary Epoch [51/100], Step [500/1128] Loss: 0.3372
Test Accuracy of NN: 90.04787234042553 % Best: 90.18085106382979 %
Ordinary Epoch [52/100], Step [500/1128] Loss: 0.4805
Test Accuracy of NN: 90.26063829787235 % (improvement)
Ordinary Epoch [53/100], Step [500/1128] Loss: 0.1815
Test Accuracy of NN: 90.16489361702128 % Best: 90.26063829787235 %
Ordinary Epoch [54/100], Step [500/1128] Loss: 0.2815
Test Accuracy of NN: 90.01063829787235 % Best: 90.26063829787235 %
Ordinary Epoch [55/100], Step [500/1128] Loss: 0.2229
Test Accuracy of NN: 90.07978723404256 % Best: 90.26063829787235 %
Ordinary Epoch [56/100], Step [500/1128] Loss: 0.4210
Test Accuracy of NN: 89.04255319148936 % Best: 90.26063829787235 %
Ordinary Epoch [57/100], Step [500/1128] Loss: 0.2301
Test Accuracy of NN: 89.91489361702128 % Best: 90.26063829787235 %
Ordinary Epoch [58/100], Step [500/1128] Loss: 0.4419
Test Accuracy of NN: 90.17553191489361 % Best: 90.26063829787235 %
Ordinary Epoch [59/100], Step [500/1128] Loss: 0.1455
Test Accuracy of NN: 90.21808510638297 % Best: 90.26063829787235 %
Ordinary Epoch [60/100], Step [500/1128] Loss: 0.2192
Test Accuracy of NN: 90.24468085106383 % Best: 90.26063829787235 %
Ordinary Epoch [61/100], Step [500/1128] Loss: 0.2444
Test Accuracy of NN: 90.32978723404256 % (improvement)
Ordinary Epoch [62/100], Step [500/1128] Loss: 0.1388
Test Accuracy of NN: 90.32446808510639 % Best: 90.32978723404256 %
Ordinary Epoch [63/100], Step [500/1128] Loss: 0.1949
Test Accuracy of NN: 90.39893617021276 % (improvement)
Ordinary Epoch [64/100], Step [500/1128] Loss: 0.2417
Test Accuracy of NN: 90.45744680851064 % (improvement)
Ordinary Epoch [65/100], Step [500/1128] Loss: 0.1627
Test Accuracy of NN: 90.44680851063829 % Best: 90.45744680851064 %
Ordinary Epoch [66/100], Step [500/1128] Loss: 0.2835
Test Accuracy of NN: 90.18617021276596 % Best: 90.45744680851064 %
Ordinary Epoch [67/100], Step [500/1128] Loss: 0.3132
Test Accuracy of NN: 90.48936170212765 % (improvement)
Ordinary Epoch [68/100], Step [500/1128] Loss: 0.1641
Test Accuracy of NN: 90.35106382978724 % Best: 90.48936170212765 %
Ordinary Epoch [69/100], Step [500/1128] Loss: 0.2531
Test Accuracy of NN: 90.31914893617021 % Best: 90.48936170212765 %
Ordinary Epoch [70/100], Step [500/1128] Loss: 0.2603
Test Accuracy of NN: 89.39893617021276 % Best: 90.48936170212765 %
Ordinary Epoch [71/100], Step [500/1128] Loss: 0.3457
Test Accuracy of NN: 89.27127659574468 % Best: 90.48936170212765 %
Ordinary Epoch [72/100], Step [500/1128] Loss: 0.2232
Test Accuracy of NN: 90.03191489361703 % Best: 90.48936170212765 %
Ordinary Epoch [73/100], Step [500/1128] Loss: 0.2104
Test Accuracy of NN: 90.02127659574468 % Best: 90.48936170212765 %
Ordinary Epoch [74/100], Step [500/1128] Loss: 0.3221
Test Accuracy of NN: 88.92553191489361 % Best: 90.48936170212765 %
Ordinary Epoch [75/100], Step [500/1128] Loss: 0.4227
Test Accuracy of NN: 88.31382978723404 % Best: 90.48936170212765 %
Ordinary Epoch [76/100], Step [500/1128] Loss: 0.4057
Test Accuracy of NN: 89.91489361702128 % Best: 90.48936170212765 %
Ordinary Epoch [77/100], Step [500/1128] Loss: 0.2736
Test Accuracy of NN: 88.76063829787235 % Best: 90.48936170212765 %
Ordinary Epoch [78/100], Step [500/1128] Loss: 0.2481
Test Accuracy of NN: 89.11702127659575 % Best: 90.48936170212765 %
Ordinary Epoch [79/100], Step [500/1128] Loss: 0.2533
Test Accuracy of NN: 90.07978723404256 % Best: 90.48936170212765 %
Ordinary Epoch [80/100], Step [500/1128] Loss: 0.1624
Test Accuracy of NN: 90.22340425531915 % Best: 90.48936170212765 %
Ordinary Epoch [81/100], Step [500/1128] Loss: 0.2686
Test Accuracy of NN: 90.23936170212765 % Best: 90.48936170212765 %
Ordinary Epoch [82/100], Step [500/1128] Loss: 0.3024
Test Accuracy of NN: 90.12234042553192 % Best: 90.48936170212765 %
Ordinary Epoch [83/100], Step [500/1128] Loss: 0.3068
Test Accuracy of NN: 89.74468085106383 % Best: 90.48936170212765 %
Ordinary Epoch [84/100], Step [500/1128] Loss: 0.2743
Test Accuracy of NN: 89.35106382978724 % Best: 90.48936170212765 %
Ordinary Epoch [85/100], Step [500/1128] Loss: 0.1863
Test Accuracy of NN: 90.34042553191489 % Best: 90.48936170212765 %
Ordinary Epoch [86/100], Step [500/1128] Loss: 0.1920
Test Accuracy of NN: 89.51063829787235 % Best: 90.48936170212765 %
Ordinary Epoch [87/100], Step [500/1128] Loss: 0.3374
Test Accuracy of NN: 89.59574468085107 % Best: 90.48936170212765 %
Ordinary Epoch [88/100], Step [500/1128] Loss: 0.4033
Test Accuracy of NN: 90.15957446808511 % Best: 90.48936170212765 %
Ordinary Epoch [89/100], Step [500/1128] Loss: 0.2848
Test Accuracy of NN: 88.33510638297872 % Best: 90.48936170212765 %
Ordinary Epoch [90/100], Step [500/1128] Loss: 0.3403
Test Accuracy of NN: 90.06382978723404 % Best: 90.48936170212765 %
Ordinary Epoch [91/100], Step [500/1128] Loss: 0.2182
Test Accuracy of NN: 90.04255319148936 % Best: 90.48936170212765 %
Ordinary Epoch [92/100], Step [500/1128] Loss: 0.2163
Test Accuracy of NN: 90.24468085106383 % Best: 90.48936170212765 %
Ordinary Epoch [93/100], Step [500/1128] Loss: 0.2754
Test Accuracy of NN: 90.10106382978724 % Best: 90.48936170212765 %
Ordinary Epoch [94/100], Step [500/1128] Loss: 0.2464
Test Accuracy of NN: 90.16489361702128 % Best: 90.48936170212765 %
Ordinary Epoch [95/100], Step [500/1128] Loss: 0.4464
Test Accuracy of NN: 89.3936170212766 % Best: 90.48936170212765 %
Ordinary Epoch [96/100], Step [500/1128] Loss: 0.1516
Test Accuracy of NN: 89.87234042553192 % Best: 90.48936170212765 %
Ordinary Epoch [97/100], Step [500/1128] Loss: 0.2985
Test Accuracy of NN: 89.90957446808511 % Best: 90.48936170212765 %
Ordinary Epoch [98/100], Step [500/1128] Loss: 0.1413
Test Accuracy of NN: 90.05851063829788 % Best: 90.48936170212765 %
Ordinary Epoch [99/100], Step [500/1128] Loss: 0.1780
Test Accuracy of NN: 90.11702127659575 % Best: 90.48936170212765 %
Ordinary Epoch [100/100], Step [500/1128] Loss: 0.2582
Test Accuracy of NN: 90.41489361702128 % Best: 90.48936170212765 %
