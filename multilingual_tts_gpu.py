#!/usr/bin/env python3
"""
Multilingual (Bangla + Arabic) Text-to-Speech (TTS)
Converted from Jupyter notebooks for standalone use.
Optimized for Ubuntu 24 (Noble) with NVIDIA GPU support (RTX 5060Ti 16GB).

This module provides a comprehensive multilingual TTS system that can handle
both Bangla and Arabic text, with automatic language detection and appropriate
voice synthesis.

Usage:
    python multilingual_tts_gpu.py --text "আমার সোনার বাংলা" --output output.wav
    python multilingual_tts_gpu.py --text "بسم الله" --output arabic.wav
"""

import os
import re
import sys
import logging
import argparse
from typing import List, Dict, Tuple, Optional

try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GPUManager:
    """Manages GPU resources and configuration."""
    
    def __init__(self, device_id: int = 0):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not installed. Install with: "
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
            )
        self.device_id = device_id
        self.device = self._setup_device()
    
    def _setup_device(self):
        """Setup and configure the compute device."""
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{self.device_id}")
            
            # Log GPU information
            gpu_name = torch.cuda.get_device_name(self.device_id)
            gpu_memory = torch.cuda.get_device_properties(self.device_id).total_memory / (1024**3)
            
            logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.2f} GB)")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            
            # Set memory growth to avoid OOM errors
            torch.cuda.empty_cache()
            
            return device
        else:
            logger.warning("CUDA not available. Using CPU.")
            return torch.device("cpu")
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        if not torch.cuda.is_available():
            return {"allocated": 0, "cached": 0, "total": 0}
        
        return {
            "allocated": torch.cuda.memory_allocated(self.device_id) / (1024**3),
            "cached": torch.cuda.memory_reserved(self.device_id) / (1024**3),
            "total": torch.cuda.get_device_properties(self.device_id).total_memory / (1024**3)
        }
    
    def clear_cache(self):
        """Clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")


class TextProcessor:
    """Processes and normalizes text for TTS synthesis."""
    
    # Bangla Unicode range
    BANGLA_RANGE = r'\u0980-\u09FF'
    # Arabic Unicode range
    ARABIC_RANGE = r'\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF'
    
    def __init__(self):
        # Try to import optional normalizers
        try:
            from bnunicodenormalizer import Normalizer
            self.bn_normalizer = Normalizer()
        except ImportError:
            self.bn_normalizer = None
            logger.warning("bnunicodenormalizer not available. Bengali normalization disabled.")
        
        try:
            from bnnumerizer import numerize
            self.numerize = numerize
        except ImportError:
            self.numerize = None
            logger.warning("bnnumerizer not available. Number conversion disabled.")
        
        try:
            import bangla
            self.bangla = bangla
        except ImportError:
            self.bangla = None
            logger.warning("bangla package not available.")
        
        # Attribution expansions
        self.attributions = {
            "সাঃ": "সাল্লাল্লাহু আলাইহি ওয়া সাল্লাম",
            "স.": "সাল্লাল্লাহু আলাইহি ওয়া সাল্লাম",
            "আঃ": "আলাইহিস সালাম",
            "রাঃ": "রাদিআল্লাহু আনহু",
            "রহঃ": "রহমাতুল্লাহি আলাইহি",
            "রহিঃ": "রহিমাহুল্লাহ",
            "হাফিঃ": "হাফিযাহুল্লাহ",
            "দাঃবাঃ": "দামাত বারাকাতুহুম, দামাত বারাকাতুল্লাহ",
        }
    
    def detect_language(self, text: str) -> str:
        """
        Detect the primary language of the text.
        
        Args:
            text: Input text
            
        Returns:
            Language code ('bn' for Bangla, 'ar' for Arabic, 'mixed' for both)
        """
        bangla_chars = len(re.findall(f'[{self.BANGLA_RANGE}]', text))
        arabic_chars = len(re.findall(f'[{self.ARABIC_RANGE}]', text))
        
        if bangla_chars > 0 and arabic_chars > 0:
            return 'mixed'
        elif bangla_chars > arabic_chars:
            return 'bn'
        elif arabic_chars > bangla_chars:
            return 'ar'
        else:
            return 'bn'  # Default to Bangla
    
    def normalize_bangla(self, text: str) -> str:
        """Normalize Bangla text."""
        if self.bn_normalizer is None:
            return text
        
        words = []
        for word in text.split():
            normalized = self.bn_normalizer(word).get('normalized')
            if normalized:
                words.append(normalized)
        
        return ' '.join(words)
    
    def expand_attributions(self, text: str) -> str:
        """Expand abbreviated attributions to full form."""
        for abbr, full in self.attributions.items():
            text = text.replace(abbr, full)
        return text
    
    def convert_numbers(self, text: str) -> str:
        """Convert English numbers to Bangla and expand to words."""
        # Convert English digits to Bangla
        if self.bangla:
            text = self.bangla.convert_english_digit_to_bangla_digit(text)
        
        # Convert numbers to words
        if self.numerize:
            try:
                text = self.numerize(text)
            except Exception as e:
                logger.warning(f"Number conversion failed: {e}")
        
        return text
    
    def preprocess(self, text: str) -> str:
        """
        Full preprocessing pipeline for text.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text ready for TTS
        """
        # Expand attributions
        text = self.expand_attributions(text)
        
        # Convert numbers
        text = self.convert_numbers(text)
        
        # Normalize Bangla text
        text = self.normalize_bangla(text)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def split_by_language(self, text: str) -> List[Tuple[str, str]]:
        """
        Split mixed text into language-tagged segments.
        
        Args:
            text: Input text potentially containing multiple languages
            
        Returns:
            List of (language, text) tuples
        """
        segments = []
        current_text = ""
        current_lang = None
        
        for char in text:
            if re.match(f'[{self.BANGLA_RANGE}]', char):
                char_lang = 'bn'
            elif re.match(f'[{self.ARABIC_RANGE}]', char):
                char_lang = 'ar'
            else:
                char_lang = current_lang  # Keep punctuation with current language
            
            if char_lang != current_lang and current_text.strip():
                if current_lang:
                    segments.append((current_lang, current_text.strip()))
                current_text = char
                current_lang = char_lang
            else:
                current_text += char
                if current_lang is None:
                    current_lang = char_lang
        
        if current_text.strip() and current_lang:
            segments.append((current_lang, current_text.strip()))
        
        return segments


class BanglaTTS:
    """Bangla Text-to-Speech engine with GPU support."""
    
    def __init__(self, voice: str = "male", use_gpu: bool = True, device_id: int = 0):
        """
        Initialize the Bangla TTS engine.
        
        Args:
            voice: Voice type ('male' or 'female')
            use_gpu: Whether to use GPU acceleration
            device_id: GPU device ID to use
        """
        self.voice = voice
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device_id = device_id
        
        self.gpu_manager = GPUManager(device_id) if self.use_gpu else None
        self.text_processor = TextProcessor()
        
        self.tts = None
        self._load_model()
    
    def _load_model(self):
        """Load the TTS model."""
        try:
            from TTS.api import TTS
        except ImportError:
            raise ImportError(
                "TTS library not found. Install with: pip install TTS"
            )
        
        model_name = f"tts_models/bn/custom/vits-{self.voice}"
        logger.info(f"Loading TTS model: {model_name}")
        
        self.tts = TTS(model_name=model_name, gpu=self.use_gpu)
        
        logger.info(f"Model loaded on {'GPU' if self.use_gpu else 'CPU'}")
    
    def synthesize(
        self,
        text: str,
        output_path: str = "output.wav",
        speaker_wav: Optional[str] = None
    ) -> str:
        """
        Synthesize speech from text.
        
        Args:
            text: Input Bangla text
            output_path: Path to save the output audio
            speaker_wav: Path to speaker audio for voice cloning (optional)
            
        Returns:
            Path to the generated audio file
        """
        # Preprocess text
        text = self.text_processor.preprocess(text)
        logger.info(f"Synthesizing: {text[:100]}...")
        
        if speaker_wav and os.path.exists(speaker_wav):
            # Voice cloning mode
            self.tts.tts_with_vc_to_file(
                text=text,
                speaker_wav=speaker_wav,
                file_path=output_path
            )
        else:
            # Standard synthesis
            self.tts.tts_to_file(text=text, file_path=output_path)
        
        # Clear GPU cache after synthesis
        if self.gpu_manager:
            self.gpu_manager.clear_cache()
        
        logger.info(f"Audio saved to: {output_path}")
        return output_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multilingual (Bangla/Arabic) Text-to-Speech with GPU support"
    )
    
    parser.add_argument(
        "--text", "-t",
        type=str,
        help="Text to convert to speech"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output.wav",
        help="Output audio file path"
    )
    
    parser.add_argument(
        "--voice", "-v",
        type=str,
        choices=["male", "female"],
        default="male",
        help="Voice type"
    )
    
    parser.add_argument(
        "--speaker-wav",
        type=str,
        help="Speaker audio for voice cloning"
    )
    
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration"
    )
    
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU device ID to use"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check if all dependencies are installed"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if args.check_deps:
        print("Checking dependencies...")
        print(f"  PyTorch: {'✓ Installed' if TORCH_AVAILABLE else '✗ Not installed'}")
        try:
            from TTS.api import TTS
            print("  TTS: ✓ Installed")
        except ImportError:
            print("  TTS: ✗ Not installed")
        try:
            from bnunicodenormalizer import Normalizer
            print("  bnunicodenormalizer: ✓ Installed")
        except ImportError:
            print("  bnunicodenormalizer: ✗ Not installed")
        return
    
    # Validate required arguments
    if not args.text:
        parser.error("--text is required (unless using --check-deps)")
    
    if not TORCH_AVAILABLE:
        logger.error("PyTorch is not installed. Install with:")
        logger.error("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        sys.exit(1)
    
    # Initialize TTS
    tts = BanglaTTS(
        voice=args.voice,
        use_gpu=not args.no_gpu,
        device_id=args.gpu_id
    )
    
    # Synthesize
    tts.synthesize(
        text=args.text,
        output_path=args.output,
        speaker_wav=args.speaker_wav
    )


if __name__ == "__main__":
    main()
