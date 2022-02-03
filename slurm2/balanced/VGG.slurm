/lustre/scratch/grp/fslg_internn/env/internn/bin/python
torch.Size([1000, 1, 28, 28])
Ordinary Epoch [30/100], Step [500/1128] Loss: 0.1512
max label 46
<class 'models_avg_pool.VGG'>
Parameters 3665135
<bound method VGG.two_conv_pool of VGG(
  (l1): Sequential(
    (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): AvgPool2d(kernel_size=2, stride=2, padding=0)
  )
  (l2): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): AvgPool2d(kernel_size=2, stride=2, padding=0)
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
    (9): AvgPool2d(kernel_size=2, stride=2, padding=0)
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
    (9): AvgPool2d(kernel_size=2, stride=2, padding=0)
  )
  (classifier): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=256, out_features=512, bias=True)
    (2): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): ReLU(inplace=True)
    (4): Dropout(p=0.5, inplace=False)
    (5): Linear(in_features=512, out_features=47, bias=True)
  )
)> <bound method VGG.three_conv_pool of VGG(
  (l1): Sequential(
    (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): AvgPool2d(kernel_size=2, stride=2, padding=0)
  )
  (l2): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): AvgPool2d(kernel_size=2, stride=2, padding=0)
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
    (9): AvgPool2d(kernel_size=2, stride=2, padding=0)
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
    (9): AvgPool2d(kernel_size=2, stride=2, padding=0)
  )
  (classifier): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=256, out_features=512, bias=True)
    (2): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): ReLU(inplace=True)
    (4): Dropout(p=0.5, inplace=False)
    (5): Linear(in_features=512, out_features=47, bias=True)
  )
)>
Ordinary Epoch [1/100], Step [500/1128] Loss: 1.1095
Test Accuracy of NN: 90.3563829787234 % Best: 90.42021276595744 %
Ordinary Epoch [31/100], Step [500/1128] Loss: 0.2340
Test Accuracy of NN: 82.36170212765957 % (improvement)
Ordinary Epoch [2/100], Step [500/1128] Loss: 1.0078
Test Accuracy of NN: 90.10106382978724 % Best: 90.42021276595744 %
Ordinary Epoch [32/100], Step [500/1128] Loss: 0.3412
Test Accuracy of NN: 85.44680851063829 % (improvement)
Test Accuracy of NN: 90.06382978723404 % Best: 90.42021276595744 %
Ordinary Epoch [3/100], Step [500/1128] Loss: 0.4913
Ordinary Epoch [33/100], Step [500/1128] Loss: 0.3221
Test Accuracy of NN: 87.60106382978724 % (improvement)
Test Accuracy of NN: 90.30319148936171 % Best: 90.42021276595744 %
Ordinary Epoch [4/100], Step [500/1128] Loss: 0.2752
Ordinary Epoch [34/100], Step [500/1128] Loss: 0.3408
/lustre/scratch/grp/fslg_internn/glom-pytorch/MNIST_vgg.py:116: DeprecationWarning: np.asscalar(a) is deprecated since NumPy v1.16, use a.item() instead
  curr_lr1 = learning_rate * np.asscalar(pow(np.random.rand(1), 3))
Test Accuracy of NN: 87.02659574468085 % Best: 87.60106382978722 %
Test Accuracy of NN: 90.44148936170212 % (improvement)
Ordinary Epoch [5/100], Step [500/1128] Loss: 0.2739
Ordinary Epoch [35/100], Step [500/1128] Loss: 0.2301
Test Accuracy of NN: 88.89893617021276 % (improvement)
Test Accuracy of NN: 90.40425531914893 % Best: 90.44148936170214 %
Ordinary Epoch [6/100], Step [500/1128] Loss: 0.2744
Ordinary Epoch [36/100], Step [500/1128] Loss: 0.2548
Test Accuracy of NN: 89.15957446808511 % (improvement)
Ordinary Epoch [7/100], Step [500/1128] Loss: 0.5149
Test Accuracy of NN: 90.37765957446808 % Best: 90.44148936170214 %
Ordinary Epoch [37/100], Step [500/1128] Loss: 0.2981
Test Accuracy of NN: 89.22340425531915 % (improvement)
Ordinary Epoch [8/100], Step [500/1128] Loss: 0.4259
Test Accuracy of NN: 89.67021276595744 % Best: 90.44148936170214 %
Ordinary Epoch [38/100], Step [500/1128] Loss: 0.3320
Test Accuracy of NN: 89.24468085106383 % (improvement)
Ordinary Epoch [9/100], Step [500/1128] Loss: 0.4234
Test Accuracy of NN: 90.32446808510639 % Best: 90.44148936170214 %
Ordinary Epoch [39/100], Step [500/1128] Loss: 0.1954
Test Accuracy of NN: 89.27659574468085 % (improvement)
Ordinary Epoch [10/100], Step [500/1128] Loss: 0.2820
Test Accuracy of NN: 90.32446808510639 % Best: 90.44148936170214 %
Ordinary Epoch [40/100], Step [500/1128] Loss: 0.4435
Test Accuracy of NN: 89.24468085106383 % Best: 89.27659574468085 %
Ordinary Epoch [11/100], Step [500/1128] Loss: 0.2633
Test Accuracy of NN: 89.54255319148936 % Best: 90.44148936170214 %
Ordinary Epoch [41/100], Step [500/1128] Loss: 0.2967
Test Accuracy of NN: 89.45744680851064 % (improvement)
Ordinary Epoch [12/100], Step [500/1128] Loss: 0.2443
Test Accuracy of NN: 89.8563829787234 % Best: 90.44148936170214 %
Ordinary Epoch [42/100], Step [500/1128] Loss: 0.2928
Test Accuracy of NN: 89.3936170212766 % Best: 89.45744680851064 %
Ordinary Epoch [13/100], Step [500/1128] Loss: 0.2920
Test Accuracy of NN: 90.08510638297872 % Best: 90.44148936170214 %
Ordinary Epoch [43/100], Step [500/1128] Loss: 0.1765
Test Accuracy of NN: 89.06382978723404 % Best: 89.45744680851064 %
slurmstepd: error: *** JOB 46310972 ON m9g-2-18 CANCELLED AT 2022-01-27T00:21:25 ***
slurmstepd: error: *** JOB 46311231 ON m9g-2-5 CANCELLED AT 2022-01-27T00:21:27 ***
