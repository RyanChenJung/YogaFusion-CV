"""
Phase 2: Qdrant Database Initialization
=========================================
Initializes and populates a Qdrant vector database with yoga pose embeddings.

Features:
  - Docker container management for Qdrant
  - Collection creation with proper vector parameters
  - Batch embedding extraction using Phase 1's FeatureExtractor
  - Upsert with metadata (image_id, pose_name, model_type)

Usage:
  # Start Qdrant Docker container
  python db_initializer.py --docker start
  
  # Stop Qdrant Docker container
  python db_initializer.py --docker stop
  
  # Initialize collection for a specific model
  python db_initializer.py --model v2
  python db_initializer.py --model a1
  python db_initializer.py --model hybrid
  
  # Initialize all collections at once
  python db_initializer.py --model all
  
  # Full pipeline: start Docker + initialize all
  python db_initializer.py --docker start --model all
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

# Qdrant imports
from qdrant_client import QdrantClient, models as qdrant_models

# Load config
import yaml

# Lazy import placeholder for Phase 1 — imported inside functions to avoid
# segfaults when only running Docker commands (e.g., --docker start).
FeatureExtractor = None
MODEL_METADATA = None


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


# =============================================================================
# Docker Management
# =============================================================================

class QdrantDockerManager:
    """Manages the Qdrant Docker container lifecycle."""

    def __init__(self, compose_file: str = "docker-compose.yml", logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.compose_file = Path(compose_file)
    
    def _run_command(self, cmd: List[str]) -> Tuple[int, str, str]:
        """Run a shell command and return (returncode, stdout, stderr)."""
        import subprocess
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)
    
    def start(self) -> bool:
        """
        Start Qdrant Docker container using docker-compose.
        
        Returns:
            True if container started successfully
        """
        self.logger.info("Starting Qdrant Docker container...")
        
        if not self.compose_file.exists():
            self.logger.error("docker-compose.yml not found at: %s", self.compose_file)
            return False
        
        returncode, stdout, stderr = self._run_command(
            ["docker-compose", "up", "-d"]
        )
        
        if returncode == 0:
            self.logger.info("Qdrant container started successfully.")
            # Wait for Qdrant to be ready
            self.logger.info("Waiting for Qdrant to be ready...")
            import time
            time.sleep(3)
            return True
        else:
            self.logger.error("Failed to start Qdrant: %s", stderr)
            return False
    
    def stop(self) -> bool:
        """
        Stop and remove Qdrant Docker container.
        
        Returns:
            True if container stopped successfully
        """
        self.logger.info("Stopping Qdrant Docker container...")
        
        returncode, stdout, stderr = self._run_command(
            ["docker-compose", "down"]
        )
        
        if returncode == 0:
            self.logger.info("Qdrant container stopped successfully.")
            return True
        else:
            self.logger.error("Failed to stop Qdrant: %s", stderr)
            return False
    
    def status(self) -> bool:
        """
        Check Qdrant Docker container status.
        
        Returns:
            True if container is running
        """
        self.logger.info("Checking Qdrant container status...")
        
        returncode, stdout, stderr = self._run_command(
            ["docker-compose", "ps"]
        )
        
        if returncode == 0:
            self.logger.info(stdout)
            return "Up" in stdout
        else:
            self.logger.error("Failed to check status: %s", stderr)
            return False
    
    def wait_for_ready(self, timeout: int = 30) -> bool:
        """
        Wait for Qdrant REST API to become available.
        
        Args:
            timeout: Maximum seconds to wait
        
        Returns:
            True if Qdrant is ready
        """
        import time
        config = load_config()
        url = config["qdrant"]["url"]
        
        self.logger.info("Waiting for Qdrant at %s (timeout=%ds)...", url, timeout)
        
        from qdrant_client import QdrantClient
        client = QdrantClient(url)
        
        elapsed = 0
        while elapsed < timeout:
            try:
                # Try to get collections - will succeed when Qdrant is ready
                client.get_collections()
                self.logger.info("Qdrant is ready after %ds.", elapsed)
                return True
            except Exception:
                time.sleep(1)
                elapsed += 1
        
        self.logger.error("Qdrant did not become ready within %ds.", timeout)
        return False


# =============================================================================
# Collection Management
# =============================================================================

def create_collection(
    client: QdrantClient,
    collection_name: str,
    vector_dim: int,
    logger: logging.Logger
) -> bool:
    """
    Create (or recreate) a Qdrant collection with vector parameters.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection
        vector_dim: Dimensionality of feature vectors
        logger: Logger instance
    
    Returns:
        True if collection was created successfully
    """
    try:
        # Check if collection exists and delete it first
        try:
            client.delete_collection(collection_name)
            logger.info("Deleted existing collection '%s'.", collection_name)
        except Exception:
            pass  # Collection didn't exist, which is fine
        
        logger.info("Creating collection '%s' (dim=%d, distance=COSINE)...", 
                    collection_name, vector_dim)
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=vector_dim,
                distance=qdrant_models.Distance.COSINE,
            ),
        )
        logger.info("Collection '%s' created successfully.", collection_name)
        return True
    except Exception as e:
        logger.error("Failed to create collection '%s': %s", collection_name, e)
        return False


def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """Check if a collection already exists."""
    try:
        collections = client.get_collections()
        for col in collections.collections:
            if col.name == collection_name:
                return True
        return False
    except Exception:
        return False


# =============================================================================
# Image Processing & Database Population
# =============================================================================

def extract_pose_name(filename: str) -> str:
    """
    Extract pose name from image filename.
    
    Examples:
        "sample_397_parithasana.jpg" -> "Parithasana"
        "sample_533_ustrasana.jpg" -> "Ustrasana"
    
    Args:
        filename: Image filename without path or extension
    
    Returns:
        Cleaned pose name
    """
    # Remove "sample_" prefix
    name = filename.replace("sample_", "")
    
    # Remove numeric portions
    name = ''.join([c for c in name if not c.isdigit()]).strip()
    
    # Replace underscores and title-case
    name = name.replace("_", " ").title()
    
    return name


def get_image_files(images_dir: str, logger: logging.Logger) -> List[Path]:
    """
    Get all JPG image files from the images directory.
    
    Args:
        images_dir: Path to directory containing yoga sample images
    
    Returns:
        List of Path objects for .jpg files
    """
    img_dir = Path(images_dir)
    if not img_dir.exists():
        logger.warning("Images directory not found: %s", img_dir)
        return []
    
    image_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.jpeg"))
    logger.info("Found %d image files in %s", len(image_files), img_dir)
    return image_files


def process_single_image(
    image_path: Path,
    model_type: str,
    logger: logging.Logger
) -> Tuple[dict, bool]:
    """
    Process a single image: extract embedding and prepare payload.
    
    NOTE: Creates a fresh FeatureExtractor per call because TFLite
    Interpreter is NOT thread-safe. Sharing one interpreter across
    threads causes internal memory reference conflicts.
    
    Args:
        image_path: Path to the image file
        model_type: Model type identifier
        logger: Logger instance
    
    Returns:
        Tuple of (point_data dict, success bool)
    """
    try:
        filename = image_path.stem  # filename without extension
        image_id = filename
        pose_name = extract_pose_name(filename)
        
        # Create a fresh extractor for this thread (TFLite is not thread-safe)
        sys.path.insert(0, str(Path(__file__).parent))
        from embedding_engine import FeatureExtractor as FE
        extractor = FE(model_type=model_type)
        
        # Extract feature vector
        embedding = extractor.extract(str(image_path))
        
        point_data = {
            "id": hash(image_id) % (10 ** 15),  # Simple hash-based ID
            "payload": {
                "image_id": image_id,
                "image_path": str(image_path),
                "pose_name": pose_name,
                "model_type": model_type,
                "filename": image_path.name,
            },
            "vector": embedding.tolist(),
        }
        
        return point_data, True
    
    except Exception as e:
        logger.warning("Failed to process %s: %s", image_path.name, e)
        return None, False


def initialize_collection(
    model_type: str,
    images_dir: str,
    qdrant_url: str,
    logger: logging.Logger,
    batch_size: int = 8,
    max_workers: int = 4
) -> bool:
    """
    Initialize a Qdrant collection for a specific model type.
    
    Steps:
        1. Load FeatureExtractor for the model
        2. Connect to Qdrant
        3. Create/recreate the collection
        4. Extract embeddings and upsert to Qdrant
    
    Args:
        model_type: Model type ('a1', 'v2', 'hybrid')
        images_dir: Directory containing yoga sample images
        qdrant_url: Qdrant REST API URL
        logger: Logger instance
        batch_size: Number of images to process per batch
        max_workers: Thread pool size for parallel extraction
    
    Returns:
        True if initialization was successful
    """
    logger.info("=" * 60)
    logger.info("Phase 2: Initializing Qdrant collection for model '%s'", model_type)
    logger.info("=" * 60)
    
    # Step 1: Load FeatureExtractor (lazy import to avoid segfault on --docker commands)
    global FeatureExtractor
    if FeatureExtractor is None:
        sys.path.insert(0, str(Path(__file__).parent))
        from embedding_engine import FeatureExtractor as FE
        FeatureExtractor = FE
    
    try:
        extractor = FeatureExtractor(model_type=model_type)
    except FileNotFoundError as e:
        logger.error("SKIP: %s", e)
        return False
    except Exception as e:
        logger.error("Failed to load extractor: %s", e)
        return False
    
    vector_dim = extractor.vector_dimension
    logger.info("Feature vector dimension: %d", vector_dim)
    
    # Step 2: Connect to Qdrant
    try:
        client = QdrantClient(url=qdrant_url)
        # Verify connection
        client.get_collections()
        logger.info("Connected to Qdrant at %s", qdrant_url)
    except Exception as e:
        logger.error("Failed to connect to Qdrant at %s: %s", qdrant_url, e)
        logger.error("Make sure Qdrant is running: docker-compose up -d")
        return False
    
    # Step 3: Get collection name
    config = load_config()
    collection_name = config["qdrant"]["collections"].get(model_type, f"yoga_{model_type}")
    
    # Step 4: Create collection
    if not create_collection(client, collection_name, vector_dim, logger):
        return False
    
    # Step 5: Get image files
    image_files = get_image_files(images_dir, logger)
    if not image_files:
        logger.warning("No images to process. Collection created but empty.")
        return True
    
    # Step 6: Process and upsert images
    total = len(image_files)
    upserted = 0
    failed = 0
    
    logger.info("Processing %d images (batch_size=%d, workers=%d)...", 
                total, batch_size, max_workers)
    
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_files = image_files[batch_start:batch_end]
        
        # Process batch in parallel
        # NOTE: Each thread creates its own FeatureExtractor because
        # TFLite Interpreter is NOT thread-safe
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single_image, f, model_type, logger): f
                for f in batch_files
            }
            for future in as_completed(futures):
                result, success = future.result()
                if success and result:
                    results.append(result)
        
        # Upsert batch
        if results:
            try:
                client.upsert(
                    collection_name=collection_name,
                    points=results,
                    wait=True
                )
                upserted += len(results)
                logger.info(
                    "Progress: %d/%d images upserted (%d failed)",
                    upserted, total, failed
                )
            except Exception as e:
                logger.error("Upsert failed: %s", e)
                failed += (batch_end - batch_start - len(results))
    
    # Step 7: Summary
    logger.info("=" * 60)
    logger.info(
        "Collection '%s' initialized: %d upserted, %d failed out of %d total",
        collection_name, upserted, failed, total
    )
    
    # Verify
    try:
        collection_info = client.get_collection(collection_name)
        logger.info(
            "Collection stats: points=%s",
            collection_info.points_count
        )
    except Exception as e:
        logger.warning("Could not get collection info: %s", e)
    
    logger.info("Phase 2 complete for model '%s'.", model_type)
    return True


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Qdrant Database Initialization"
    )
    
    # Docker management
    docker_group = parser.add_argument_group("Docker Management")
    docker_group.add_argument(
        "--docker",
        choices=["start", "stop", "status"],
        default=None,
        help="Manage Qdrant Docker container"
    )
    
    # Database initialization
    db_group = parser.add_argument_group("Database Initialization")
    db_group.add_argument(
        "--model",
        choices=["a1", "v2", "hybrid", "all"],
        default="v2",
        help="Model type to initialize (default: v2)"
    )
    db_group.add_argument(
        "--images-dir",
        default=None,
        help="Override images directory from config"
    )
    db_group.add_argument(
        "--qdrant-url",
        default=None,
        help="Override Qdrant URL from config"
    )
    db_group.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for image processing (default: 8)"
    )
    db_group.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads (default: 4)"
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    logger = setup_logging(args.log_level)
    
    config = load_config()
    
    # Handle Docker commands
    if args.docker:
        docker_mgr = QdrantDockerManager(logger=logger)
        if args.docker == "start":
            if docker_mgr.start():
                docker_mgr.wait_for_ready()
        elif args.docker == "stop":
            docker_mgr.stop()
        elif args.docker == "status":
            docker_mgr.status()
        return
    
    # Handle database initialization
    images_dir = args.images_dir or config["app"]["images_dir"]
    qdrant_url = args.qdrant_url or config["qdrant"]["url"]
    
    if args.model == "all":
        # Initialize all three model collections
        for model_type in ["a1", "v2", "hybrid"]:
            success = initialize_collection(
                model_type=model_type,
                images_dir=images_dir,
                qdrant_url=qdrant_url,
                logger=logger,
                batch_size=args.batch_size,
                max_workers=args.workers
            )
            if not success:
                logger.warning("Model '%s' initialization had errors.", model_type)
    else:
        initialize_collection(
            model_type=args.model,
            images_dir=images_dir,
            qdrant_url=qdrant_url,
            logger=logger,
            batch_size=args.batch_size,
            max_workers=args.workers
        )


if __name__ == "__main__":
    main()