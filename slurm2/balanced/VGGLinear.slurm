/lustre/scratch/grp/fslg_internn/env/internn/bin/python
torch.Size([1000, 1, 28, 28])
max label 46
<class 'models_avg_pool.VGGLinear'>
Parameters 3665135
<bound method VGGLinear.two_conv_pool of VGGLinear(
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
Ordinary Epoch [1/100], Step [500/1128] Loss: 2.4881
Test Accuracy of NN: 50.5531914893617 % (improvement)
Ordinary Epoch [2/100], Step [500/1128] Loss: 2.6989
Test Accuracy of NN: 54.12234042553192 % (improvement)
Ordinary Epoch [3/100], Step [500/1128] Loss: 2.1888
Test Accuracy of NN: 56.07446808510638 % (improvement)
Ordinary Epoch [4/100], Step [500/1128] Loss: 2.4802
/lustre/scratch/grp/fslg_internn/glom-pytorch/MNIST_vgg.py:116: DeprecationWarning: np.asscalar(a) is deprecated since NumPy v1.16, use a.item() instead
  curr_lr1 = learning_rate * np.asscalar(pow(np.random.rand(1), 3))
Test Accuracy of NN: 55.265957446808514 % Best: 56.07446808510639 %
Ordinary Epoch [5/100], Step [500/1128] Loss: 2.2651
Test Accuracy of NN: 59.16489361702128 % (improvement)
Ordinary Epoch [6/100], Step [500/1128] Loss: 2.0873
Test Accuracy of NN: 59.898936170212764 % (improvement)
Ordinary Epoch [7/100], Step [500/1128] Loss: 2.1242
Test Accuracy of NN: 59.88297872340426 % Best: 59.89893617021277 %
Ordinary Epoch [8/100], Step [500/1128] Loss: 2.2157
Test Accuracy of NN: 57.51063829787234 % Best: 59.89893617021277 %
Ordinary Epoch [9/100], Step [500/1128] Loss: 2.4433
Test Accuracy of NN: 57.851063829787236 % Best: 59.89893617021277 %
Ordinary Epoch [10/100], Step [500/1128] Loss: 2.5727
Test Accuracy of NN: 58.31382978723404 % Best: 59.89893617021277 %
Ordinary Epoch [11/100], Step [500/1128] Loss: 2.7552
Test Accuracy of NN: 59.797872340425535 % Best: 59.89893617021277 %
Ordinary Epoch [12/100], Step [500/1128] Loss: 2.2454
Test Accuracy of NN: 59.27127659574468 % Best: 59.89893617021277 %
Ordinary Epoch [13/100], Step [500/1128] Loss: 2.2672
Test Accuracy of NN: 56.6968085106383 % Best: 59.89893617021277 %
Ordinary Epoch [14/100], Step [500/1128] Loss: 2.1111
Test Accuracy of NN: 60.20744680851064 % (improvement)
Ordinary Epoch [15/100], Step [500/1128] Loss: 1.9713
Test Accuracy of NN: 59.96808510638298 % Best: 60.20744680851063 %
Ordinary Epoch [16/100], Step [500/1128] Loss: 2.6592
Test Accuracy of NN: 60.56382978723404 % (improvement)
Ordinary Epoch [17/100], Step [500/1128] Loss: 2.2214
Test Accuracy of NN: 60.648936170212764 % (improvement)
Ordinary Epoch [18/100], Step [500/1128] Loss: 1.9933
slurmstepd: error: *** JOB 46311230 ON m9g-2-2 CANCELLED AT 2022-01-27T00:21:27 ***
