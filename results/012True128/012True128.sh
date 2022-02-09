#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -C 'rhel7&pascal'
#SBATCH --mem-per-cpu 6000
#SBATCH --ntasks 7
#SBATCH --nodes=1
#SBATCH --output="/lustre/scratch/grp/fslg_internn/glom-pytorch/./results/012True128/log.slurm"
#SBATCH --time 72:00:00
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
python MNIST_train.py  --attention_radius 0 --iterations 1 --levels 2 --use_cnn True --glom_dim 128 --save_path ./results/012True128/model.pt
