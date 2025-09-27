import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict
from tafberta import configs

def collate_fn(
    batch: List[Dict],
    tokenizer
    ) -> Dict[str, torch.Tensor]:
    # Extract inputs, labels, and attention masks from the batch
    input_ids = [item[0]['input_ids'].squeeze(0) for item in batch]  # Squeeze out extra dimension if present
    attention_masks = [item[0]['attention_mask'].squeeze(0) for item in batch]  # Ensure 1D tensors
    labels = [item[1].squeeze(0) for item in batch]  # Ensure 1D tensors for labels

    # Ensure that all input IDs and attention masks are 1D
    input_ids = [x if x.dim() == 1 else x.view(-1) for x in input_ids]
    attention_masks = [x if x.dim() == 1 else x.view(-1) for x in attention_masks]
    labels = [x if x.dim() == 1 else x.view(-1) for x in labels]

    # Pad sequences to the longest length in the batch
    pad_token_id = tokenizer.token_to_id(configs.Data.pad_symbol)
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    # Pad labels to the longest length in input_ids
    max_len = input_ids_padded.size(1)
    labels_padded = [torch.cat([label, torch.full((max_len - label.size(0),), -100)])
                     if label.size(0) < max_len
                     else label
                     for label in labels
                    ]
    labels_padded = torch.stack(labels_padded)

    return {
        'input_ids': input_ids_padded, 
        'attention_mask': attention_masks_padded
    }, labels_padded