import os
from pathlib import Path
from typing import List, Dict, Union
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    AutoTokenizer
)

from tafberta.evaluation.holistic.utils import calc_accuracy_from_scores, get_correct_vector_results_from_scores
from tafberta.utils.io import load_tokenizer


class ModelScorer:
    def __init__(
        self,
        model_path: str = None,  # None in case the class was defined in lightning training
        tokenizer_path: str = None,
        model: Union[AutoModelForMaskedLM, AutoModelForCausalLM] = None,
        tokenizer: Union[PreTrainedTokenizerFast] = None,
        max_length: int = 128,
        model_type: str = "mlm",  # "mlm" for Masked LM, "clm" for Causal LM
    ):
        """Initialize the model scorer with model and tokenizer."""
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model_type = model_type.lower()
        self.model = self.load_model() if self.model_path else model
        self.tokenizer = (
            self.load_PreTrainedTokenizerFast(self.tokenizer_path, max_length)
            if tokenizer_path
            else tokenizer
        )
        self.loss_fct = CrossEntropyLoss(reduction="none")

        # Ensure tokenizer has pad_token
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Resize model embeddings if tokenizer size has changed
        # if self.model.config.vocab_size != len(self.tokenizer):
        #     self.model.resize_token_embeddings(len(self.tokenizer))

    def load_PreTrainedTokenizerFast(
        self, tokenizer_path: str, max_length: int
    ) -> PreTrainedTokenizerFast:
        """Load and configure the tokenizer."""
        try:
            tokenizer = load_tokenizer(tokenizer_path, max_length)
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
            
        except:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            tokenizer.model_max_length = max_length
            
        # Ensure tokenizer has pad_token
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        return tokenizer

    def load_model(self):
        """Load the model for either masked language modeling or causal language modeling."""
        if self.model_type == "mlm":
            return AutoModelForMaskedLM.from_pretrained(self.model_path)
        elif self.model_type == "clm":
            return AutoModelForCausalLM.from_pretrained(self.model_path)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}. Use 'mlm' or 'clm'.")

    def compute_scores_with_model(
        self,
        model: torch.nn.Module,
        paradigm_paths: List[str],
        tokenizer: PreTrainedTokenizerFast = None,
        device: torch.device = None,
    ) -> Dict[str, float]:
        """
        Compute accuracy scores for multiple paradigms using a pre-loaded model and tokenizer.

        Args:
            model (torch.nn.Module): Pre-loaded model.
            tokenizer (PreTrainedTokenizerFast): Tokenizer object to tokenize the sentences.
            paradigm_paths (List[str]): List of file paths to paradigm sentence files.
            device (torch.device): Device index to select

        Returns:
            Dict[str, float]: A dictionary mapping paradigm names to accuracy scores.
        """
        if tokenizer:
            self.tokenizer = tokenizer
        if device:
            model.to(device)
        model.eval()  # Ensure the model is in evaluation mode
        accuracy_scores = {}

        with torch.no_grad():
            for paradigm_path in paradigm_paths:
                paradigm_name = os.path.basename(os.path.dirname(paradigm_path))
                print(f"Processing {paradigm_name}")
                sentences = Path(paradigm_path).read_text().splitlines()

                cross_entropies = self.compute_cross_entropies(
                    sentences, model, device
                )

                # Calculate accuracy from cross-entropies for the current paradigm
                accuracy_holistic = calc_accuracy_from_scores(
                    cross_entropies, "holistic"
                )
                accuracy_scores[paradigm_name] = accuracy_holistic

        return accuracy_scores
        
    def compute_results_vector_with_model(
        self,
        model: torch.nn.Module,
        paradigm_paths: List[str],
        tokenizer: PreTrainedTokenizerFast = None,
        device: torch.device = None,
    ) -> Dict[str, List]:
        """
        Compute correct model predictions for multiple paradigms using a pre-loaded model and tokenizer.

        Args:
            model (torch.nn.Module): Pre-loaded model.
            tokenizer (PreTrainedTokenizerFast): Tokenizer object to tokenize the sentences.
            paradigm_paths (List[str]): List of file paths to paradigm sentence files.
            device (torch.device): Device index to select

        Returns:
            Dict[str, List]: A dictionary mapping paradigm names to answers vector.
        """
        if tokenizer:
            self.tokenizer = tokenizer
        if device:
            model.to(device)
        model.eval()  # Ensure the model is in evaluation mode
        results = {}

        with torch.no_grad():
            for paradigm_path in paradigm_paths:
                paradigm_name = os.path.basename(os.path.dirname(paradigm_path))
                print(f"Processing {paradigm_name}")
                sentences = Path(paradigm_path).read_text().splitlines()

                cross_entropies = self.compute_cross_entropies(
                    sentences, model, device
                )

                result = get_correct_vector_results_from_scores(
                    cross_entropies, "holistic"
                )

                results[paradigm_name] = result

        return results

    def compute_cross_entropies(
        self,
        sentences: List[str],
        model: torch.nn.Module,
        device: torch.device = None,
        batch_size: int = 32,
    ) -> List[float]:
        """Compute the cross-entropy loss for the given sentences in batches."""
        cross_entropies = []

        if device:
            model.to(device)  # Move model to the specified device

        with torch.no_grad():
            # Process the sentences in batches
            for i in range(0, len(sentences), batch_size):
                batch_sentences = sentences[i : i + batch_size]

                # Tokenize the batch of sentences
                tokenized_input = self.tokenizer(
                    batch_sentences,
                    padding="longest",
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )

                input_ids = tokenized_input["input_ids"]
                attention_mask = tokenized_input["attention_mask"]

                # Move tensors to the device
                if device:
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)

                if self.model_type == "mlm":
                    # For MLM models
                    labels = input_ids.clone()
                    labels[attention_mask == 0] = -100  # Ignore padding tokens

                    # Forward pass to get logits
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits.permute(0, 2, 1)  # [batch_size, vocab_size, seq_len]

                    # Compute loss for each token
                    loss = self.loss_fct(logits, labels)
                
                    # Calculate the cross-entropy for each token where the mask is not applied
                    for loss_i, row_mask in zip(loss, attention_mask):
                        valid_token_loss = loss_i[row_mask==1].mean().item()
                        cross_entropies.append(valid_token_loss)

                elif self.model_type == "clm":
                    # For CLM models
                    labels = input_ids.clone()
                    labels[:, 0] = -100  # Cannot predict the first token
                    labels[attention_mask == 0] = -100  # Ignore padding tokens
    
                    # Forward pass to get logits
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits.permute(0, 2, 1)  # [batch_size, vocab_size, seq_len]
    
                    # Compute loss for each token
                    loss = self.loss_fct(logits, labels)
                    
                    # Create a mask for valid positions
                    valid_mask = (attention_mask == 1) & (labels != -100)
                    
                    # Use masked_select to get only the valid loss values for each row
                    # Replace NaN for rows where no valid tokens exist
                    valid_losses = torch.where(
                        valid_mask,
                        loss,
                        torch.tensor(float('nan'), device=loss.device)
                    )

                    # Compute the mean of valid losses along the sequence dimension
                    cross_entropies.extend(torch.nanmean(valid_losses, dim=1).tolist())
                        
                else:
                    raise ValueError(f"Unsupported model_type: {self.model_type}")

        return cross_entropies


def process_paradigms(
    model_path: str = None,
    tokenizer_path: str = None,
    model: Union[AutoModelForMaskedLM, AutoModelForCausalLM] = None,
    tokenizer: PreTrainedTokenizerFast = None,
    paradigm_paths: List[str] = None,
    model_type: str = "mlm",
    device: torch.device = None,
) -> Dict[str, float]:
    """
    Process multiple paradigm files by loading the model and tokenizer from paths,
    then compute accuracy scores for each paradigm.

    Returns:
        Dict[str, float]: A dictionary mapping paradigm names to accuracy scores.
    """
    # Initialize the scorer using paths for model and tokenizer
    scorer = ModelScorer(model_path, tokenizer_path, model_type=model_type)

    # Load model and tokenizer
    model = scorer.model if model_path else model
    tokenizer = scorer.tokenizer if tokenizer_path else tokenizer

    # Use the new function to compute scores with the pre-loaded model and tokenizer
    accuracy_scores = scorer.compute_scores_with_model(
        model=model, tokenizer=tokenizer, paradigm_paths=paradigm_paths, device=device,
    )

    return accuracy_scores
