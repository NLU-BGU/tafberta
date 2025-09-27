from tafberta.train_lightning import main, objective
from tafberta.params import param2default
import torch import os

# Disable parallelism in Hugging Face tokenizers to avoid the warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('medium')

# train with roberta parameters
param2default['batch_size'] = 1024
param2default['leave_unmasked_prob'] = 0.1
param2default['num_layers'] = 12
param2default['num_attention_heads'] = 12
param2default['hidden_size'] = 768
param2default['intermediate_size'] = 3072

main(param2default, patience=10000)