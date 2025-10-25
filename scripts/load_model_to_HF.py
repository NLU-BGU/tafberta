# import mlflow.pytorch
# import torch

# # Path to the saved MLflow model directory
# model_path = "saved_models/best_model"

# # Load the model (returns a PyTorch nn.Module)
# model = mlflow.pytorch.load_model(model_path, map_location=torch.device('cpu'))
# print(type(model))

# # Example: Use the model for inference
# model.eval()
# # output = model(input_tensor)

from tafberta import configs

tokenizer_path = str(configs.Dirs.tokenizers / 'htberman_tokenizer.json')

import os
import json
import torch
import mlflow.pytorch
from transformers import RobertaConfig, RobertaForMaskedLM, PreTrainedTokenizerFast
from huggingface_hub import HfApi


def export_to_huggingface(
    mlflow_model_dir: str,
    hf_export_dir: str,
    tokenizer_path: str,
    config_path: str = None,
    hf_repo_id: str = None,
):
    """
    Load a trained MLflow model, convert it to Hugging Face format,
    and optionally upload it to the Hugging Face Hub.

    Args
    ----
    mlflow_model_dir : str
        Path to MLflow-logged model directory (e.g., "saved_models/best_model").
    hf_export_dir : str
        Output directory for Hugging Face model files.
    tokenizer_path : str
        Path to tokenizer directory or tokenizer.json file.
    config_path : str, optional
        Path to Hugging Face-compatible config.json file.
    hf_repo_id : str, optional
        Hugging Face repo_id (e.g. "geanita/TafBERTa") to upload to.
        Requires HF_TOKEN in environment.
    """

    os.makedirs(hf_export_dir, exist_ok=True)

    # 1. Load MLflow model
    print(f"🔹 Loading model from MLflow dir: {mlflow_model_dir}")
    mlflow_model: torch.nn.Module = mlflow.pytorch.load_model(mlflow_model_dir, map_location=torch.device('cpu'))
    mlflow_model.eval()

    # 2. Load or build RobertaConfig
    if config_path is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        hf_config = RobertaConfig(**cfg_dict)
    elif hasattr(mlflow_model, "config") and isinstance(mlflow_model.config, RobertaConfig):
        hf_config = mlflow_model.config
    else:
        raise ValueError(
            "Cannot infer model config. Provide a config_path to a valid config.json file."
        )

    # 3. Build RobertaForMaskedLM and load state dict
    print("🔹 Building Hugging Face model...")
    hf_model = RobertaForMaskedLM(hf_config)
    hf_model.load_state_dict(mlflow_model.state_dict(), strict=True)

    # 4. Load tokenizer
    print(f"🔹 Loading tokenizer from: {tokenizer_path}")
    if os.path.isdir(tokenizer_path):
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    else:
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
        )

    # 5. Save model + tokenizer
    print(f"💾 Saving model in Hugging Face format to: {hf_export_dir}")
    hf_model.save_pretrained(hf_export_dir)
    tokenizer.save_pretrained(hf_export_dir)

    print("✅ Local export completed successfully.")

    # 6. Optional upload to Hugging Face Hub
    if hf_repo_id is not None:
        print(f"☁️ Uploading to Hugging Face Hub: {hf_repo_id}")
        api = HfApi(token=os.getenv("HF_TOKEN"))
        api.upload_folder(
            folder_path=hf_export_dir,
            repo_id=hf_repo_id,
            repo_type="model",
        )
        print(f"🚀 Model uploaded to https://huggingface.co/{hf_repo_id}")


if __name__ == "__main__":
    export_to_huggingface(
        mlflow_model_dir="../saved_models/best_model",      # your MLflow model dir
        hf_export_dir="huggingface_export",              # local export dir
        tokenizer_path=tokenizer_path,  # your trained tokenizer
        config_path=None,               # path to config.json
        hf_repo_id="geanita/TafBERTa",                   # your Hugging Face repo
    )
