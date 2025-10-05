"""Image processing module for visual content."""

from typing import Dict, List, Any, Union
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
from datetime import datetime

import cv2
import pytesseract
from PIL import Image
from loguru import logger

from ..config import RAGConfig


class ImageProcessor:
    """Process image files and extract visual and text content."""
    
    def __init__(self, config: RAGConfig):
        """Initialize image processor."""
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def process_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process an image file and extract content.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Dict containing processed content and metadata
        """
        file_path = Path(file_path)
        
        # Extract visual features and OCR text
        image_data = await self._load_image(file_path)
        ocr_text = await self._extract_text_ocr(file_path) if self.config.ocr_enabled else ""
        visual_features = await self._extract_visual_features(image_data)
        
        # Generate metadata
        file_hash = await self._calculate_file_hash(file_path)
        
        result = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_type': 'image',
            'file_extension': file_path.suffix.lstrip('.'),
            'file_hash': file_hash,
            'file_size': file_path.stat().st_size,
            'processed_at': datetime.now().isoformat(),
            'ocr_text': ocr_text,
            'visual_features': visual_features,
            'content': ocr_text,  # Use OCR text as searchable content
            'chunks': [
                {
                    'chunk_id': 0,
                    'text': ocr_text,
                    'image_path': str(file_path),
                    'visual_features': visual_features,
                    'metadata': {
                        'source_file': str(file_path),
                        'chunk_index': 0,
                        'file_type': 'image'
                    }
                }
            ] if ocr_text or visual_features else []
        }
        
        logger.info(f"Processed image {file_path}")
        return result
    
    async def _load_image(self, file_path: Path) -> Any:
        """Load image file."""
        def load():
            return cv2.imread(str(file_path))
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, load
        )
    
    async def _extract_text_ocr(self, file_path: Path) -> str:
        """Extract text from image using OCR."""
        def extract_ocr():
            try:
                image = Image.open(file_path)
                text = pytesseract.image_to_string(image)
                return text.strip()
            except Exception as e:
                logger.error(f"OCR failed for {file_path}: {e}")
                return ""
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, extract_ocr
        )
    
    async def _extract_visual_features(self, image_data: Any) -> Dict[str, Any]:
        """Extract visual features from image (placeholder)."""
        # This would implement actual visual feature extraction
        # using models like CLIP, ResNet, etc.
        return {"placeholder": "visual_features"}
    
    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file."""
        def calculate_hash():
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, calculate_hash
        )
