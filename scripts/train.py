import os
print(os.path.abspath(os.curdir))

from tafberta.training.train_lightning import main
from tafberta.params import param2default

main(param2default)