
import torch
from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import hf_hub_download
from langchain.llms import LlamaCpp

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
)
from constants import (
    CONTEXT_WINDOW_SIZE, 
    MAX_NEW_TOKENS
)

def load_quantized_model_gguf_ggml(model_id, model_basename, device_type, logging):
    """
    Load a GGUF/GGML quantized model using LlamaCpp.

    This function attempts to load a GGUF/GGML quantized model using the LlamaCpp library. 
    If the model is of type GGML, and newer version of LLAMA-CPP is used which does not support GGML, 
    it logs a message indicating that LLAMA-CPP has dropped support for GGML.

    Parameters:
    - model_id (str): The identifier for the model on HuggingFace Hub.
    - model_basename (str): The base name of the model file.
    - device_type (str): The type of device where the model will run, e.g., 'mps', 'cuda', etc.
    - logging (logging.Logger): Logger instance for logging messages.

    Returns:
    - LlamaCpp: An instance of the LlamaCpp model if successful, otherwise None.

    Notes:
    - The function uses the `hf_hub_download` function to download the model from the HuggingFace Hub.
    - The number of GPU layers is set based on the device type.
    """

    try:
        logging.info("Using Llamacpp for GGUF/GGML quantized models")
        model_path = hf_hub_download(
            repo_id=model_id, 
            filename=model_basename, 
            resume_download=True, 
            cache_dir="./models",
            )
        logging.info(f"Model : {model_id} is downloaded and ready to use");
        kwargs = {
            "model_path": model_path,
            "n_ctx": CONTEXT_WINDOW_SIZE,
            "max_tokens": MAX_NEW_TOKENS,
        }
        if device_type.lower() == "mps":
            kwargs["n_gpu_layers"] = 1
        if device_type.lower() == "cuda":
            kwargs["n_gpu_layers"] = 100 # set this based on your GPU
        logging.info(f"Model : {model_id} returend to its parent function");
        logging.info(f"Model path : {model_path}");
        llm = LlamaCpp(
            model_path=model_path,
        )
        return llm
    except:
        if 'ggml' in model_basename:
            logging.info("If you were using GGML model, LLAMA-CPP Dropped Support, Use GGUF Instead")
        return None