/lustre/scratch/grp/fslg_internn/env/internn/bin/python
torch.Size([1000, 1, 28, 28])
max label 44
<class 'models_avg_pool.V1'>
Parameters 1117871
<bound method V1.two_conv_pool of V1(
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
Ordinary Epoch [1/100], Step [500/1128] Loss: 2.6742
Test Accuracy of NN: 54.42553191489362 % (improvement)
Ordinary Epoch [2/100], Step [500/1128] Loss: 2.2458
Test Accuracy of NN: 56.22872340425532 % (improvement)
Ordinary Epoch [3/100], Step [500/1128] Loss: 2.4623
Test Accuracy of NN: 56.66489361702128 % (improvement)
Ordinary Epoch [4/100], Step [500/1128] Loss: 2.0326
Test Accuracy of NN: 57.63297872340426 % (improvement)
Ordinary Epoch [5/100], Step [500/1128] Loss: 2.3270
Test Accuracy of NN: 58.63297872340426 % (improvement)
Ordinary Epoch [6/100], Step [500/1128] Loss: 2.2351
Test Accuracy of NN: 58.994680851063826 % (improvement)
Ordinary Epoch [7/100], Step [500/1128] Loss: 2.7380
/lustre/scratch/grp/fslg_internn/glom-pytorch/MNIST_vgg.py:116: DeprecationWarning: np.asscalar(a) is deprecated since NumPy v1.16, use a.item() instead
  curr_lr1 = learning_rate * np.asscalar(pow(np.random.rand(1), 3))
Test Accuracy of NN: 57.84574468085106 % Best: 58.99468085106383 %
Ordinary Epoch [8/100], Step [500/1128] Loss: 2.3208
Test Accuracy of NN: 61.058510638297875 % (improvement)
Ordinary Epoch [9/100], Step [500/1128] Loss: 2.1972
Test Accuracy of NN: 61.638297872340424 % (improvement)
Ordinary Epoch [10/100], Step [500/1128] Loss: 2.6316
Test Accuracy of NN: 61.744680851063826 % (improvement)
Ordinary Epoch [11/100], Step [500/1128] Loss: 2.2082
Test Accuracy of NN: 62.06382978723404 % (improvement)
Ordinary Epoch [12/100], Step [500/1128] Loss: 2.4512
Test Accuracy of NN: 61.75 % Best: 62.063829787234035 %
Ordinary Epoch [13/100], Step [500/1128] Loss: 2.3284
Test Accuracy of NN: 62.212765957446805 % (improvement)
Ordinary Epoch [14/100], Step [500/1128] Loss: 2.0979
Test Accuracy of NN: 62.27127659574468 % (improvement)
Ordinary Epoch [15/100], Step [500/1128] Loss: 2.3293
Test Accuracy of NN: 62.18617021276596 % Best: 62.27127659574469 %
Ordinary Epoch [16/100], Step [500/1128] Loss: 2.4810
Test Accuracy of NN: 62.26063829787234 % Best: 62.27127659574469 %
Ordinary Epoch [17/100], Step [500/1128] Loss: 2.3401
Test Accuracy of NN: 59.87234042553192 % Best: 62.27127659574469 %
Ordinary Epoch [18/100], Step [500/1128] Loss: 2.3773
Test Accuracy of NN: 61.90425531914894 % Best: 62.27127659574469 %
Ordinary Epoch [19/100], Step [500/1128] Loss: 2.4101
Test Accuracy of NN: 62.22872340425532 % Best: 62.27127659574469 %
Ordinary Epoch [20/100], Step [500/1128] Loss: 2.5479
Test Accuracy of NN: 62.223404255319146 % Best: 62.27127659574469 %
Ordinary Epoch [21/100], Step [500/1128] Loss: 2.0035
Test Accuracy of NN: 62.3563829787234 % (improvement)
Ordinary Epoch [22/100], Step [500/1128] Loss: 2.4183
Test Accuracy of NN: 62.18617021276596 % Best: 62.3563829787234 %
Ordinary Epoch [23/100], Step [500/1128] Loss: 2.4590
Test Accuracy of NN: 62.4468085106383 % (improvement)
Ordinary Epoch [24/100], Step [500/1128] Loss: 2.3112
Test Accuracy of NN: 62.28191489361702 % Best: 62.4468085106383 %
Ordinary Epoch [25/100], Step [500/1128] Loss: 2.1943
Test Accuracy of NN: 62.308510638297875 % Best: 62.4468085106383 %
Ordinary Epoch [26/100], Step [500/1128] Loss: 2.5543
slurmstepd: error: *** JOB 46311229 ON m9g-1-5 CANCELLED AT 2022-01-27T00:21:25 ***
