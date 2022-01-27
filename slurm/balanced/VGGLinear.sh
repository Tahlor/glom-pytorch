#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -C 'rhel7&pascal'
#SBATCH --mem-per-cpu 5285MB
#SBATCH --ntasks 7
#SBATCH --nodes=1
#SBATCH --output="/lustre/scratch/grp/fslg_internn/glom-pytorch/slurm/balancedVGGLinear.slurm"
#SBATCH --time 4:00:00
#SBATCH --mail-user=taylornarchibald@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#%Module

module purge
module load cuda/10.1
module load cudnn/7.6

export PATH="/lustre/scratch/grp/fslg_internn/env/internn:$PATH"
eval "$(conda shell.bash hook)"
conda activate /lustre/scratch/grp/fslg_internn/env/internn

cd "/lustre/scratch/grp/fslg_internn/glom-pytorch"
which python
python -u /lustre/scratch/grp/fslg_internn/glom-pytorch/MNIST_vgg.py --model VGGLinear --data balanced