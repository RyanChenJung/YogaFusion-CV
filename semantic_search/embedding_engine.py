"""
Phase 1: The Embedding Engine
==============================
TFLite-based feature extraction engine for Yoga Pose Semantic Search.

Supports three model types:
  - a1  : YogaMaster_A1_Production.tflite  (Quantized CNN)
  - v2  : YogaMaster_V2_Production.tflite  (Quantized MobileNetV2)
  - hybrid: YogaMaster_Hybrid_Production.tflite (Quantized CNN+Pose)

Usage:
  # As a module
  from embedding_engine import FeatureExtractor
  extractor = FeatureExtractor(model_type="v2")
  vector = extractor.extract("path/to/image.jpg")

  # Standalone test
  python embedding_engine.py --model v2 --test
  python embedding_engine.py --model a1 --test
  python embedding_engine.py --model hybrid --test
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Union

import numpy as np
from PIL import Image

# Try to import TFLite Runtime, fall back to tensorflow
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# Load config
import yaml


def load_config(path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return a logger."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )
    return logging.getLogger(__name__)


# Model metadata: expected input shape and quantization info
# These are educated guesses based on model architecture names.
# The engine will auto-detect actual shapes from the TFLite model file.
MODEL_METADATA = {
    "a1": {
        "filename": "YogaMaster_A1_Production.tflite",
        "description": "Quantized Custom CNN model",
    },
    "v2": {
        "filename": "YogaMaster_V2_Production.tflite",
        "description": "Quantized MobileNetV2 model",
    },
    "hybrid": {
        "filename": "YogaMaster_Hybrid_Production.tflite",
        "description": "Quantized Multi-modal (CNN + Pose) model",
    },
}


class FeatureExtractor:
    """
    TFLite-based feature extractor for yoga pose semantic search.
    
    Loads a .tflite model and extracts fixed-dimensional feature vectors
    from input images using the TensorFlow Lite interpreter.
    
    Attributes:
        model_type: One of 'a1', 'v2', 'hybrid'
        model_path: Absolute path to the .tflite model file
        interpreter: TFLite Interpreter instance
        input_index: Input tensor index
        output_index: Output tensor index
        input_shape: Input tensor shape (batch, height, width, channels)
        output_shape: Output tensor shape
        is_quantized: Whether the model uses quantized inputs/outputs
    """

    def __init__(self, model_type: str = "v2", model_path: Optional[str] = None, base_dir: Optional[str] = None):
        """
        Initialize the FeatureExtractor.
        
        Args:
            model_type: Model type identifier ('a1', 'v2', or 'hybrid')
            model_path: Optional custom path to .tflite file. If None, uses config.
            base_dir: Optional custom base directory for models. If None, uses config.
        """
        self.logger = logging.getLogger(__name__)
        
        # Validate model type
        if model_type not in MODEL_METADATA:
            raise ValueError(
                f"Unknown model type '{model_type}'. "
                f"Valid types: {list(MODEL_METADATA.keys())}"
            )
        self.model_type = model_type
        
        # Load config to determine model path
        config = load_config()
        self.base_dir = Path(base_dir) if base_dir else Path(config["models"]["base_dir"])
        
        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = self.base_dir / config["models"][model_type]
        
        # Verify model file exists
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                f"Search path: {self.base_dir.absolute()}"
            )
        
        self.logger.info("Loading TFLite model: %s", self.model_path)
        
        # Load the TFLite interpreter
        self.interpreter = Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        
        # Get tensor metadata
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        
        self.input_index = input_details[0]["index"]
        self.output_index = output_details[0]["index"]
        self.input_shape = np.array(input_details[0]["shape"])
        self.output_shape = np.array(output_details[0]["shape"])
        
        # Detect quantization
        self.input_dtype = input_details[0]["dtype"]
        self.output_dtype = output_details[0]["dtype"]
        self.is_quantized = (
            input_details[0]["dtype"] == np.uint8 or
            input_details[0]["dtype"] == np.int8
        )
        
        # Store quantization parameters (for dequantizing outputs if needed)
        self.input_scale = input_details[0].get("scale", 1.0)
        self.input_zero_point = input_details[0].get("zero_point", 0)
        self.output_scale = output_details[0].get("scale", 1.0)
        self.output_zero_point = output_details[0].get("zero_point", 0)
        
        # Detect input resolution
        if len(self.input_shape) == 4:
            self.input_height = self.input_shape[1]
            self.input_width = self.input_shape[2]
            self.input_channels = self.input_shape[3]
        elif len(self.input_shape) == 2:
            # Some models have flattened input spec; default to 224x224
            self.input_height = 224
            self.input_width = 224
            self.input_channels = 3
        else:
            raise ValueError(f"Unexpected input shape: {self.input_shape}")
        
        # Get output vector dimension
        self.vector_dim = int(self.output_shape[1]) if len(self.output_shape) >= 2 else int(self.output_shape[0])
        
        self.logger.info(
            "Model loaded: input_shape=%s output_shape=%s dtype=%s quantized=%s",
            self.input_shape, self.output_shape, self.input_dtype, self.is_quantized
        )
    
    @property
    def vector_dimension(self) -> int:
        """Return the dimensionality of the output feature vector."""
        return self.vector_dim
    
    def preprocess_image(
        self, 
        image: Union[Image.Image, np.ndarray],
        target_size: Optional[tuple] = None
    ) -> np.ndarray:
        """
        Preprocess a single image for TFLite inference.
        
        Args:
            image: PIL Image or numpy array
            target_size: Optional (height, width) override. If None, uses model's input size.
        
        Returns:
            Preprocessed numpy array ready for TFLite input
        """
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        elif isinstance(image, np.ndarray):
            image_array = image
        else:
            raise TypeError(f"Expected PIL.Image or np.ndarray, got {type(image)}")
        
        # Convert to RGB if necessary
        if len(image_array.shape) == 2:
            # Grayscale -> RGB
            image_array = np.stack([image_array] * 3, axis=-1)
        elif image_array.shape[2] == 4:
            # RGBA -> RGB
            image_array = image_array[:, :, :3]
        
        # Resize to model input size
        size = target_size or (self.input_width, self.input_height)
        if isinstance(size, tuple):
            # PIL resize expects (width, height)
            pil_image = Image.fromarray(image_array)
            pil_image = pil_image.resize(size, Image.Resampling.LANCZOS)
            image_array = np.array(pil_image)
        
        # Normalize to float32
        img_float = image_array.astype(np.float32)
        
        # Apply model-specific preprocessing
        if self.is_quantized:
            # Scale to [0, 255] range for quantized models
            img_float = img_float * self.input_scale + self.input_zero_point
        else:
            # Apply ImageNet-style preprocessing if model is ~224x224
            if self.input_height >= 224:
                # TensorFlow preprocess_input: map [0, 255] to [-1, 1]
                img_float = img_float / 127.5 - 1.0
            else:
                # Simple normalization
                img_float = img_float / 255.0
        
        return img_float
    
    def extract(self, image_input: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Extract a feature vector from a single image.
        
        Args:
            image_input: File path (str), PIL Image, or numpy array
        
        Returns:
            1D numpy array of feature embeddings
        """
        # Load image if path given
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input
        
        # Preprocess
        processed = self.preprocess_image(image)
        
        # Ensure correct shape for batch inference
        if processed.ndim == 3:
            processed = np.expand_dims(processed, axis=0)
        
        # Run inference
        self.interpreter.set_tensor(self.input_index, processed)
        self.interpreter.invoke()
        
        # IMMEDIATELY copy output data to avoid internal reference issues
        # This is critical when called from ThreadPoolExecutor
        output_raw = self.interpreter.get_tensor(self.output_index)
        output = np.array(output_raw)  # Make a deep copy of the output data
        
        # Dequantize if needed
        if self.is_quantized:
            output = (output - self.output_zero_point) * self.output_scale
        
        # Flatten to 1D vector and return a copy to avoid internal references
        return output.flatten().copy()
    
    def extract_batch(
        self, 
        image_inputs: List[Union[str, Image.Image, np.ndarray]]
    ) -> np.ndarray:
        """
        Extract feature vectors from a batch of images.
        
        Args:
            image_inputs: List of file paths, PIL Images, or numpy arrays
        
        Returns:
            2D numpy array of shape (n_images, vector_dim)
        """
        embeddings = []
        for i, img_input in enumerate(image_inputs):
            try:
                emb = self.extract(img_input)
                embeddings.append(emb)
            except Exception as e:
                self.logger.warning(
                    "Failed to extract feature from input %d: %s", i, e
                )
        
        if not embeddings:
            raise ValueError("No successful extractions in batch")
        
        return np.array(embeddings)
    
    def get_model_info(self) -> dict:
        """Return metadata about the loaded model."""
        return {
            "model_type": self.model_type,
            "model_path": str(self.model_path),
            "input_shape": self.input_shape.tolist(),
            "output_shape": self.output_shape.tolist(),
            "input_dtype": str(self.input_dtype),
            "output_dtype": str(self.output_dtype),
            "vector_dim": self.vector_dim,
            "is_quantized": self.is_quantized,
            "input_size": (self.input_height, self.input_width),
        }
    
    def __repr__(self) -> str:
        info = self.get_model_info()
        return (
            f"FeatureExtractor(type='{self.model_type}', "
            f"vector_dim={info['vector_dim']}, "
            f"input_size={info['input_size']}, "
            f"quantized={info['is_quantized']})"
        )


# =============================================================================
# Standalone test / CLI interface
# =============================================================================

def test_extraction(model_type: str):
    """Run a self-test of the feature extractor."""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Phase 1 Self-Test: Model '%s'", model_type)
    logger.info("=" * 60)
    
    try:
        extractor = FeatureExtractor(model_type=model_type)
    except FileNotFoundError as e:
        logger.error("SKIP: %s", e)
        return
    
    info = extractor.get_model_info()
    logger.info("Model Info:")
    for key, value in info.items():
        logger.info("  %s: %s", key, value)
    
    # Test with yoga_samples if available
    sample_dir = Path("../yoga_samples")
    if not sample_dir.exists():
        sample_dir = Path("yoga_samples")
    
    if sample_dir.exists():
        sample_files = list(sample_dir.glob("*.jpg"))[:3]  # Test first 3
        if sample_files:
            logger.info("\nTesting with %d sample images...", len(sample_files))
            try:
                embeddings = extractor.extract_batch([str(f) for f in sample_files])
                logger.info(
                    "Batch extraction successful: shape=%s dtype=%s",
                    embeddings.shape, embeddings.dtype
                )
                for path, emb in zip(sample_files, embeddings):
                    norm = np.linalg.norm(emb)
                    logger.info(
                        "  %s: norm=%.4f, first_3=[%.4f, %.4f, %.4f]",
                        path.name, norm, emb[0], emb[1], emb[2]
                    )
                logger.info("PASS: Phase 1 self-test completed successfully.")
            except Exception as e:
                logger.error("FAIL: Batch extraction error: %s", e)
        else:
            logger.warning("No .jpg files found in %s", sample_dir)
    else:
        logger.warning("Sample directory not found at %s", sample_dir)
        logger.info("SKIP: No samples to test with.")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Yoga Pose Embedding Engine Self-Test"
    )
    parser.add_argument(
        "--model",
        choices=["a1", "v2", "hybrid"],
        default="v2",
        help="Model type to test (default: v2)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=True,
        help="Run self-test with sample images"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    setup_logging(args.log_level)
    
    if args.test:
        test_extraction(args.model)


if __name__ == "__main__":
    main()