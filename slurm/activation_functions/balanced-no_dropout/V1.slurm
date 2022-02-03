/lustre/scratch/grp/fslg_internn/env/internn/bin/python
torch.Size([1000, 1, 28, 28])
max label 46
<class 'models.V1'>
Parameters 1117871
<bound method V1.two_conv_pool of V1(
  (dropout): Identity()
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
    (0): Identity()
    (1): Linear(in_features=256, out_features=512, bias=True)
    (2): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Identity()
    (4): Linear(in_features=512, out_features=47, bias=True)
  )
)> <bound method V1.three_conv_pool of V1(
  (dropout): Identity()
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
    (0): Identity()
    (1): Linear(in_features=256, out_features=512, bias=True)
    (2): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Identity()
    (4): Linear(in_features=512, out_features=47, bias=True)
  )
)>
Ordinary Epoch [1/100], Step [500/1128] Loss: 0.7412
Test Accuracy of NN: 84.59574468085107 % (improvement)
Ordinary Epoch [2/100], Step [500/1128] Loss: 0.6857
Test Accuracy of NN: 85.09574468085107 % (improvement)
Ordinary Epoch [3/100], Step [500/1128] Loss: 0.3790
Test Accuracy of NN: 86.50531914893617 % (improvement)
Ordinary Epoch [4/100], Step [500/1128] Loss: 0.5111
Test Accuracy of NN: 86.62234042553192 % (improvement)
Ordinary Epoch [5/100], Step [500/1128] Loss: 0.3029
Test Accuracy of NN: 87.41489361702128 % (improvement)
Ordinary Epoch [6/100], Step [500/1128] Loss: 0.2282
Test Accuracy of NN: 88.17553191489361 % (improvement)
Ordinary Epoch [7/100], Step [500/1128] Loss: 0.2986
/lustre/scratch/grp/fslg_internn/glom-pytorch/MNIST_vgg.py:121: DeprecationWarning: np.asscalar(a) is deprecated since NumPy v1.16, use a.item() instead
  curr_lr1 = learning_rate * np.asscalar(pow(np.random.rand(1), 3))
Test Accuracy of NN: 87.86170212765957 % Best: 88.17553191489361 %
Ordinary Epoch [8/100], Step [500/1128] Loss: 0.2093
Test Accuracy of NN: 89.29787234042553 % (improvement)
Ordinary Epoch [9/100], Step [500/1128] Loss: 0.3841
Test Accuracy of NN: 89.2127659574468 % Best: 89.29787234042553 %
Ordinary Epoch [10/100], Step [500/1128] Loss: 0.1881
Test Accuracy of NN: 89.5372340425532 % (improvement)
Ordinary Epoch [11/100], Step [500/1128] Loss: 0.2722
Test Accuracy of NN: 89.58510638297872 % (improvement)
Ordinary Epoch [12/100], Step [500/1128] Loss: 0.3787
Test Accuracy of NN: 89.61702127659575 % (improvement)
Ordinary Epoch [13/100], Step [500/1128] Loss: 0.3504
Test Accuracy of NN: 89.73936170212765 % (improvement)
Ordinary Epoch [14/100], Step [500/1128] Loss: 0.2213
Test Accuracy of NN: 89.60106382978724 % Best: 89.73936170212767 %
Ordinary Epoch [15/100], Step [500/1128] Loss: 0.2068
Test Accuracy of NN: 89.30851063829788 % Best: 89.73936170212767 %
Ordinary Epoch [16/100], Step [500/1128] Loss: 0.3975
Test Accuracy of NN: 87.93085106382979 % Best: 89.73936170212767 %
Ordinary Epoch [17/100], Step [500/1128] Loss: 0.3715
Test Accuracy of NN: 88.96808510638297 % Best: 89.73936170212767 %
Ordinary Epoch [18/100], Step [500/1128] Loss: 0.3341
Test Accuracy of NN: 89.15957446808511 % Best: 89.73936170212767 %
Ordinary Epoch [19/100], Step [500/1128] Loss: 0.3933
Test Accuracy of NN: 88.59574468085107 % Best: 89.73936170212767 %
Ordinary Epoch [20/100], Step [500/1128] Loss: 0.4222
Test Accuracy of NN: 88.96808510638297 % Best: 89.73936170212767 %
Ordinary Epoch [21/100], Step [500/1128] Loss: 0.3142
Test Accuracy of NN: 89.74468085106383 % (improvement)
Ordinary Epoch [22/100], Step [500/1128] Loss: 0.2957
Test Accuracy of NN: 89.60106382978724 % Best: 89.74468085106383 %
Ordinary Epoch [23/100], Step [500/1128] Loss: 0.3179
Test Accuracy of NN: 89.59574468085107 % Best: 89.74468085106383 %
Ordinary Epoch [24/100], Step [500/1128] Loss: 0.3887
Test Accuracy of NN: 89.55851063829788 % Best: 89.74468085106383 %
Ordinary Epoch [25/100], Step [500/1128] Loss: 0.3614
Test Accuracy of NN: 89.77659574468085 % (improvement)
Ordinary Epoch [26/100], Step [500/1128] Loss: 0.2612
Test Accuracy of NN: 89.84574468085107 % (improvement)
Ordinary Epoch [27/100], Step [500/1128] Loss: 0.3586
Test Accuracy of NN: 89.93617021276596 % (improvement)
Ordinary Epoch [28/100], Step [500/1128] Loss: 0.2645
Test Accuracy of NN: 89.83510638297872 % Best: 89.93617021276596 %
Ordinary Epoch [29/100], Step [500/1128] Loss: 0.1827
Test Accuracy of NN: 89.87234042553192 % Best: 89.93617021276596 %
Ordinary Epoch [30/100], Step [500/1128] Loss: 0.3802
Test Accuracy of NN: 89.87765957446808 % Best: 89.93617021276596 %
Ordinary Epoch [31/100], Step [500/1128] Loss: 0.3287
Test Accuracy of NN: 89.93617021276596 % Best: 89.93617021276596 %
Ordinary Epoch [32/100], Step [500/1128] Loss: 0.3520
Test Accuracy of NN: 89.72340425531915 % Best: 89.93617021276596 %
Ordinary Epoch [33/100], Step [500/1128] Loss: 0.2961
Test Accuracy of NN: 89.7872340425532 % Best: 89.93617021276596 %
Ordinary Epoch [34/100], Step [500/1128] Loss: 0.1984
Test Accuracy of NN: 89.86170212765957 % Best: 89.93617021276596 %
Ordinary Epoch [35/100], Step [500/1128] Loss: 0.4477
Test Accuracy of NN: 89.7127659574468 % Best: 89.93617021276596 %
Ordinary Epoch [36/100], Step [500/1128] Loss: 0.2985
Test Accuracy of NN: 89.88829787234043 % Best: 89.93617021276596 %
Ordinary Epoch [37/100], Step [500/1128] Loss: 0.1859
Test Accuracy of NN: 89.81914893617021 % Best: 89.93617021276596 %
Ordinary Epoch [38/100], Step [500/1128] Loss: 0.3652
Test Accuracy of NN: 89.82446808510639 % Best: 89.93617021276596 %
Ordinary Epoch [39/100], Step [500/1128] Loss: 0.2237
Test Accuracy of NN: 89.55319148936171 % Best: 89.93617021276596 %
Ordinary Epoch [40/100], Step [500/1128] Loss: 0.1626
Test Accuracy of NN: 89.6436170212766 % Best: 89.93617021276596 %
Ordinary Epoch [41/100], Step [500/1128] Loss: 0.3110
Test Accuracy of NN: 89.97340425531915 % (improvement)
Ordinary Epoch [42/100], Step [500/1128] Loss: 0.1945
Test Accuracy of NN: 89.84574468085107 % Best: 89.97340425531914 %
Ordinary Epoch [43/100], Step [500/1128] Loss: 0.2191
Test Accuracy of NN: 89.81914893617021 % Best: 89.97340425531914 %
Ordinary Epoch [44/100], Step [500/1128] Loss: 0.1962
Test Accuracy of NN: 89.72340425531915 % Best: 89.97340425531914 %
Ordinary Epoch [45/100], Step [500/1128] Loss: 0.2731
Test Accuracy of NN: 89.76595744680851 % Best: 89.97340425531914 %
Ordinary Epoch [46/100], Step [500/1128] Loss: 0.2243
Test Accuracy of NN: 88.38297872340425 % Best: 89.97340425531914 %
Ordinary Epoch [47/100], Step [500/1128] Loss: 0.2640
Test Accuracy of NN: 88.21808510638297 % Best: 89.97340425531914 %
Ordinary Epoch [48/100], Step [500/1128] Loss: 0.3514
Test Accuracy of NN: 88.51063829787235 % Best: 89.97340425531914 %
Ordinary Epoch [49/100], Step [500/1128] Loss: 0.3084
Test Accuracy of NN: 89.63297872340425 % Best: 89.97340425531914 %
Ordinary Epoch [50/100], Step [500/1128] Loss: 0.2328
Test Accuracy of NN: 89.51595744680851 % Best: 89.97340425531914 %
Ordinary Epoch [51/100], Step [500/1128] Loss: 0.2058
Test Accuracy of NN: 89.68085106382979 % Best: 89.97340425531914 %
Ordinary Epoch [52/100], Step [500/1128] Loss: 0.2426
Test Accuracy of NN: 89.84042553191489 % Best: 89.97340425531914 %
Ordinary Epoch [53/100], Step [500/1128] Loss: 0.2178
Test Accuracy of NN: 89.88297872340425 % Best: 89.97340425531914 %
Ordinary Epoch [54/100], Step [500/1128] Loss: 0.2712
Test Accuracy of NN: 89.19148936170212 % Best: 89.97340425531914 %
Ordinary Epoch [55/100], Step [500/1128] Loss: 0.3954
Test Accuracy of NN: 89.52127659574468 % Best: 89.97340425531914 %
Ordinary Epoch [56/100], Step [500/1128] Loss: 0.1564
Test Accuracy of NN: 89.90425531914893 % Best: 89.97340425531914 %
Ordinary Epoch [57/100], Step [500/1128] Loss: 0.3079
Test Accuracy of NN: 89.98936170212765 % (improvement)
Ordinary Epoch [58/100], Step [500/1128] Loss: 0.1681
Test Accuracy of NN: 90.04255319148936 % (improvement)
Ordinary Epoch [59/100], Step [500/1128] Loss: 0.2250
Test Accuracy of NN: 90.0372340425532 % Best: 90.04255319148936 %
Ordinary Epoch [60/100], Step [500/1128] Loss: 0.2894
Test Accuracy of NN: 90.1063829787234 % (improvement)
Ordinary Epoch [61/100], Step [500/1128] Loss: 0.2233
Test Accuracy of NN: 89.95744680851064 % Best: 90.1063829787234 %
Ordinary Epoch [62/100], Step [500/1128] Loss: 0.3604
Test Accuracy of NN: 89.98404255319149 % Best: 90.1063829787234 %
Ordinary Epoch [63/100], Step [500/1128] Loss: 0.3256
Test Accuracy of NN: 89.96808510638297 % Best: 90.1063829787234 %
Ordinary Epoch [64/100], Step [500/1128] Loss: 0.2099
Test Accuracy of NN: 89.86170212765957 % Best: 90.1063829787234 %
Ordinary Epoch [65/100], Step [500/1128] Loss: 0.3532
Test Accuracy of NN: 89.02659574468085 % Best: 90.1063829787234 %
Ordinary Epoch [66/100], Step [500/1128] Loss: 0.3113
Test Accuracy of NN: 89.73936170212765 % Best: 90.1063829787234 %
Ordinary Epoch [67/100], Step [500/1128] Loss: 0.2346
Test Accuracy of NN: 90.01063829787235 % Best: 90.1063829787234 %
Ordinary Epoch [68/100], Step [500/1128] Loss: 0.3199
Test Accuracy of NN: 88.81382978723404 % Best: 90.1063829787234 %
Ordinary Epoch [69/100], Step [500/1128] Loss: 0.5212
Test Accuracy of NN: 88.58510638297872 % Best: 90.1063829787234 %
Ordinary Epoch [70/100], Step [500/1128] Loss: 0.3035
Test Accuracy of NN: 88.67021276595744 % Best: 90.1063829787234 %
Ordinary Epoch [71/100], Step [500/1128] Loss: 0.2901
Test Accuracy of NN: 89.30851063829788 % Best: 90.1063829787234 %
Ordinary Epoch [72/100], Step [500/1128] Loss: 0.2154
Test Accuracy of NN: 89.72340425531915 % Best: 90.1063829787234 %
Ordinary Epoch [73/100], Step [500/1128] Loss: 0.2887
Test Accuracy of NN: 89.05851063829788 % Best: 90.1063829787234 %
Ordinary Epoch [74/100], Step [500/1128] Loss: 0.3395
Test Accuracy of NN: 89.38297872340425 % Best: 90.1063829787234 %
Ordinary Epoch [75/100], Step [500/1128] Loss: 0.2425
Test Accuracy of NN: 89.73404255319149 % Best: 90.1063829787234 %
Ordinary Epoch [76/100], Step [500/1128] Loss: 0.1985
Test Accuracy of NN: 89.93085106382979 % Best: 90.1063829787234 %
Ordinary Epoch [77/100], Step [500/1128] Loss: 0.3358
Test Accuracy of NN: 89.81914893617021 % Best: 90.1063829787234 %
Ordinary Epoch [78/100], Step [500/1128] Loss: 0.2115
Test Accuracy of NN: 89.66489361702128 % Best: 90.1063829787234 %
Ordinary Epoch [79/100], Step [500/1128] Loss: 0.1640
Test Accuracy of NN: 90.00531914893617 % Best: 90.1063829787234 %
Ordinary Epoch [80/100], Step [500/1128] Loss: 0.2095
Test Accuracy of NN: 89.69680851063829 % Best: 90.1063829787234 %
Ordinary Epoch [81/100], Step [500/1128] Loss: 0.1517
Test Accuracy of NN: 89.84042553191489 % Best: 90.1063829787234 %
Ordinary Epoch [82/100], Step [500/1128] Loss: 0.3539
Test Accuracy of NN: 89.79255319148936 % Best: 90.1063829787234 %
Ordinary Epoch [83/100], Step [500/1128] Loss: 0.2817
Test Accuracy of NN: 89.89893617021276 % Best: 90.1063829787234 %
Ordinary Epoch [84/100], Step [500/1128] Loss: 0.1954
Test Accuracy of NN: 89.62234042553192 % Best: 90.1063829787234 %
Ordinary Epoch [85/100], Step [500/1128] Loss: 0.2694
Test Accuracy of NN: 89.64893617021276 % Best: 90.1063829787234 %
Ordinary Epoch [86/100], Step [500/1128] Loss: 0.3521
Test Accuracy of NN: 88.56914893617021 % Best: 90.1063829787234 %
Ordinary Epoch [87/100], Step [500/1128] Loss: 0.1813
Test Accuracy of NN: 89.95744680851064 % Best: 90.1063829787234 %
Ordinary Epoch [88/100], Step [500/1128] Loss: 0.2640
Test Accuracy of NN: 90.0 % Best: 90.1063829787234 %
Ordinary Epoch [89/100], Step [500/1128] Loss: 0.2319
Test Accuracy of NN: 89.93085106382979 % Best: 90.1063829787234 %
Ordinary Epoch [90/100], Step [500/1128] Loss: 0.2554
Test Accuracy of NN: 90.1063829787234 % Best: 90.1063829787234 %
Ordinary Epoch [91/100], Step [500/1128] Loss: 0.2227
Test Accuracy of NN: 90.02127659574468 % Best: 90.1063829787234 %
Ordinary Epoch [92/100], Step [500/1128] Loss: 0.4047
Test Accuracy of NN: 89.1063829787234 % Best: 90.1063829787234 %
Ordinary Epoch [93/100], Step [500/1128] Loss: 0.2003
Test Accuracy of NN: 89.91489361702128 % Best: 90.1063829787234 %
Ordinary Epoch [94/100], Step [500/1128] Loss: 0.4245
Test Accuracy of NN: 89.97872340425532 % Best: 90.1063829787234 %
Ordinary Epoch [95/100], Step [500/1128] Loss: 0.2902
Test Accuracy of NN: 89.54787234042553 % Best: 90.1063829787234 %
Ordinary Epoch [96/100], Step [500/1128] Loss: 0.3607
Test Accuracy of NN: 89.61702127659575 % Best: 90.1063829787234 %
Ordinary Epoch [97/100], Step [500/1128] Loss: 0.3361
Test Accuracy of NN: 89.85106382978724 % Best: 90.1063829787234 %
Ordinary Epoch [98/100], Step [500/1128] Loss: 0.2402
Test Accuracy of NN: 89.98404255319149 % Best: 90.1063829787234 %
Ordinary Epoch [99/100], Step [500/1128] Loss: 0.2835
Test Accuracy of NN: 88.97872340425532 % Best: 90.1063829787234 %
Ordinary Epoch [100/100], Step [500/1128] Loss: 0.1907
Test Accuracy of NN: 89.20744680851064 % Best: 90.1063829787234 %
