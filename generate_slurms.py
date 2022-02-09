from itertools import product
from pathlib import Path

threads = 7
mem = 6000
root_sh_path = "./results/"
time = "72:00:00"
gpu = "pascal"
EMAIL = "taylornarchibald@gmail.com"
env = "/lustre/scratch/grp/fslg_internn/env/internn"
ROOT_PROJECT = "/lustre/scratch/grp/fslg_internn/glom-pytorch"


def cartesian_product(inp):
    return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))


### PATCH SIZE

variation_dict = {"attention_radius":[0,1,2],
            "iterations":[2,3,4],
            "levels":[2,3],
            "use_cnn":[True,False],
            "glom_dim":[128,256],
            }

# --attention_radius --iterations --levels --use_cnn --glom_dim --save_path


def write_out(command,
              name,
              result_path):
    Path(result_path).mkdir(parents=True,exist_ok=True)
    with open(f"{result_path}/{name}.sh", "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -C 'rhel7&pascal'
#SBATCH --mem-per-cpu {mem}
#SBATCH --ntasks {threads}
#SBATCH --nodes=1
#SBATCH --output="{ROOT_PROJECT}/{result_path}/log.slurm"
#SBATCH --time {time}
#SBATCH --mail-user={EMAIL}   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#%Module

module purge
module load cuda/10.1
module load cudnn/7.6

export PATH="{env}:$PATH"
eval "$(conda shell.bash hook)"
conda activate {env}

cd "{ROOT_PROJECT}"
which python
{command}
""")

# all_combinations = list(cartesian_product(variation_dict))

all_combinations = [
    {'attention_radius': 0, 'iterations': 1, 'levels': 2, 'use_cnn': True, 'glom_dim': 128},
    {'attention_radius': 0, 'iterations': 2, 'levels': 2, 'use_cnn': True, 'glom_dim': 128},
    {'attention_radius': 0, 'iterations': 2, 'levels': 2, 'use_cnn': True, 'glom_dim': 256},
    {'attention_radius': 0, 'iterations': 2, 'levels': 2, 'use_cnn': False, 'glom_dim': 128},
    {'attention_radius': 0, 'iterations': 2, 'levels': 2, 'use_cnn': False, 'glom_dim': 256},
    {'attention_radius': 0, 'iterations': 2, 'levels': 3, 'use_cnn': True, 'glom_dim': 128},
    {'attention_radius': 1, 'iterations': 2, 'levels': 2, 'use_cnn': False, 'glom_dim': 128},
    {'attention_radius': 1, 'iterations': 2, 'levels': 2, 'use_cnn': False, 'glom_dim': 256},
    {'attention_radius': 1, 'iterations': 2, 'levels': 3, 'use_cnn': False, 'glom_dim': 128},
    {'attention_radius': 1, 'iterations': 2, 'levels': 3, 'use_cnn': False, 'glom_dim': 256},
    {'attention_radius': 1, 'iterations': 3, 'levels': 2, 'use_cnn': False, 'glom_dim': 128},
    {'attention_radius': 1, 'iterations': 3, 'levels': 3, 'use_cnn': False, 'glom_dim': 256}
]

for variant in all_combinations:
    base = "python MNIST_train.py "
    path = f"./results/"
    name = ""
    for k,v in variant.items():
        base += f" --{k} {v}"
        name += f"{v}"
    base += f" --save_path {path}{name}/model.pt"
    print(base)
    result_path = f"{path}{name}"
    write_out(base, name, result_path)
