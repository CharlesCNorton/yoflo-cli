"""Model loading and management for Florence-2."""

import os
import logging

import torch
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.modeling_utils import PreTrainedModel

# Monkeypatch to fix Florence-2 compatibility with transformers 4.45+
_original_pretrained_getattr = PreTrainedModel.__getattr__


def _patched_pretrained_getattr(self, name):
    if name == '_supports_sdpa':
        return True
    return _original_pretrained_getattr(self, name)


PreTrainedModel.__getattr__ = _patched_pretrained_getattr


class ModelManager:
    """Loads and manages Florence-2 model and processor with optional quantization."""

    DEFAULT_REPO_ID = "microsoft/Florence-2-base-ft"
    LARGE_REPO_ID = "microsoft/Florence-2-large-ft"

    def __init__(self, device=None, quantization=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.quantization = quantization

    def _get_quant_config(self):
        """Return quantization config if enabled."""
        if self.quantization == "4bit":
            logging.info("Using 4-bit quantization.")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        return None

    def load_local_model(self, model_path):
        """
        Load model from a local directory.

        :param model_path: Path to the model directory.
        :return: True if successful, False otherwise.
        """
        if not os.path.exists(model_path):
            logging.error(f"Model path {os.path.abspath(model_path)} does not exist.")
            return False
        if not os.path.isdir(model_path):
            logging.error(f"Model path {os.path.abspath(model_path)} is not a directory.")
            return False

        try:
            logging.info(f"Loading model from {os.path.abspath(model_path)}")
            quant_config = self._get_quant_config()

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                quantization_config=quant_config,
            ).eval()

            if not self.quantization:
                self.model.to(self.device)
                if torch.cuda.is_available():
                    self.model = self.model.half()
                    logging.info("Using FP16 precision.")

            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            logging.info("Model loaded successfully.")
            return True

        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return False

    def download_and_load_model(self, repo_id=None):
        """
        Download model from HuggingFace Hub and load it.

        :param repo_id: HuggingFace repository ID (default: Florence-2-base-ft).
        :return: True if successful, False otherwise.
        """
        repo_id = repo_id or self.DEFAULT_REPO_ID

        try:
            local_dir = "model"
            logging.info(f"Downloading model from {repo_id}...")
            snapshot_download(repo_id=repo_id, local_dir=local_dir)

            if not os.path.exists(local_dir) or not os.path.isdir(local_dir):
                logging.error("Model download failed.")
                return False

            logging.info(f"Model downloaded to {os.path.abspath(local_dir)}")
            return self.load_local_model(local_dir)

        except Exception as e:
            logging.error(f"Error downloading model: {e}")
            return False

    @property
    def is_loaded(self):
        """Check if model is loaded."""
        return self.model is not None and self.processor is not None
