"""Audio processing module for voice recordings."""

from typing import Dict, List, Any, Union
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
from datetime import datetime

import librosa
import whisper
from loguru import logger

from ..config import RAGConfig


class AudioProcessor:
    """Process audio files and extract speech content."""
    
    def __init__(self, config: RAGConfig):
        """Initialize audio processor."""
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=2)
        # Load Whisper model lazily
        self._whisper_model = None
    
    @property
    def whisper_model(self):
        """Lazy load Whisper model."""
        if self._whisper_model is None:
            self._whisper_model = whisper.load_model(self.config.whisper_model)
        return self._whisper_model
    
    async def process_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process an audio file and extract speech content.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dict containing processed content and metadata
        """
        file_path = Path(file_path)
        
        # Extract speech transcript
        transcript = await self._transcribe_audio(file_path)
        audio_features = await self._extract_audio_features(file_path)
        
        # Generate metadata
        file_hash = await self._calculate_file_hash(file_path)
        
        result = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_type': 'audio',
            'file_extension': file_path.suffix.lstrip('.'),
            'file_hash': file_hash,
            'file_size': file_path.stat().st_size,
            'processed_at': datetime.now().isoformat(),
            'transcript': transcript,
            'audio_features': audio_features,
            'content': transcript,  # Use transcript as searchable content
            'chunks': [
                {
                    'chunk_id': 0,
                    'text': transcript,
                    'audio_path': str(file_path),
                    'audio_features': audio_features,
                    'metadata': {
                        'source_file': str(file_path),
                        'chunk_index': 0,
                        'file_type': 'audio'
                    }
                }
            ] if transcript else []
        }
        
        logger.info(f"Processed audio {file_path}")
        return result
    
    async def _transcribe_audio(self, file_path: Path) -> str:
        """Transcribe audio using Whisper."""
        def transcribe():
            try:
                result = self.whisper_model.transcribe(str(file_path))
                return result["text"].strip()
            except Exception as e:
                logger.error(f"Transcription failed for {file_path}: {e}")
                return ""
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, transcribe
        )
    
    async def _extract_audio_features(self, file_path: Path) -> Dict[str, Any]:
        """Extract audio features (placeholder)."""
        def extract_features():
            try:
                y, sr = librosa.load(str(file_path))
                duration = librosa.get_duration(y=y, sr=sr)
                
                return {
                    "duration": duration,
                    "sample_rate": sr,
                    "length": len(y)
                }
            except Exception as e:
                logger.error(f"Feature extraction failed for {file_path}: {e}")
                return {}
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, extract_features
        )
    
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
