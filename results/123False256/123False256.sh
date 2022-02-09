#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -C 'rhel7&pascal'
#SBATCH --mem-per-cpu 6000
#SBATCH --ntasks 7
#SBATCH --nodes=1
#SBATCH --output="/lustre/scratch/grp/fslg_internn/glom-pytorch/./results/123False256/log.slurm"
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
python MNIST_train.py  --attention_radius 1 --iterations 2 --levels 3 --use_cnn False --glom_dim 256 --save_path ./results/123False256/model.pt
