import socket

def get_computer():
    return socket.gethostname()

def is_galois():
    return get_computer() == "Galois"


import matplotlib.pyplot as plt
from pathlib import Path
import torch
from general_tools.utils import get_root

def save_model(path, model, optimizer=None, epoch=None, loss=None, scheduler=None):
    if not Path(path).suffix:
        path = Path(path) / "MODEL.pt"
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else {},
        'loss': loss,
        'scheduler':scheduler.state_dict() if scheduler else {}
    }, path)

def load_model(path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler'])
    return {"model":model,
            "optimizer":optimizer,
            "epoch":epoch,
            "loss":loss,
            "scheduler":scheduler}

def incrementer(root, base, make_new_folder=False, start_over=False, incrementer=True):
    """

    Args:
        root (folder):
        base (filename or subfolder):
        make_new_folder (make new subfolder if it doesn't exist): if you just want a new filename, set to false
        start_over (bool): start at original version again
    Returns:

    """
    if not incrementer:
        return Path(root) / base
    def _new_folder():
        increment_string = f"_{increment:03d}" if increment > 0 else ""
        new_folder = Path(root / (base.stem + increment_string + base.suffix))
        return new_folder
    increment = 0
    if "*" in base and not start_over:
        all_matches = list(Path(root).glob(base))
        all_matches.sort()
        if all_matches:
            try:
                increment = int(all_matches[-1].stem.split("_")[-1])
            except:
                pass

    base = base.replace("*", "")  # remove asterisk if it exists
    base = Path(base); root = Path(root)
    new_folder = _new_folder()
    while new_folder.exists():
        increment += 1
        new_folder = _new_folder()
    if make_new_folder:
        new_folder.mkdir(parents=True, exist_ok=True)
    return new_folder

def get_latest_file(folder, filename="*.pt"):
    """

    Args:
        folder:
        filename (str): A glob file string

    Returns:

    """
    list_of_paths = list(Path(folder).glob(filename))
    if list_of_paths:
        return max(list_of_paths, key=lambda p: p.stat().st_ctime)
    else:
        return False

def plot(y, x=None, save=False, ymax=.2):
    ax = plt.gca()
    ax.set_ylim([0, ymax])
    if not x:
        x = range(0,len(y))
    plt.plot(y,x)
    if save:
        plt.savefig(save, dpi=300)
        plt.close()
    else:
        plt.show()
    print(f"Plotting losses: {y,x}")


if __name__ == "__main__":
    x = incrementer("/media/data/GitHub/internn/data/embedding_datasets/embeddings_v2.1", "BERT_embedding*.pt")
    print(x)