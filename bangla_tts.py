#!/usr/bin/env python3
"""
Bangla Text-to-Speech (TTS) - Standalone Python Script
Converted from Jupyter notebooks for standalone use.
Optimized for Ubuntu 24 (Noble) with NVIDIA GPU support (RTX 5060Ti 16GB).

Usage:
    python bangla_tts.py --text "আমার সোনার বাংলা" --output output.wav
    python bangla_tts.py --text "আমার সোনার বাংলা" --voice male --output output.wav
    python bangla_tts.py --text "আমার সোনার বাংলা" --voice female --output output.wav
    python bangla_tts.py --text "আমার সোনার বাংলা" --voice-clone speaker.wav --output output.wav
"""

import argparse
import os
import sys
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_gpu_availability():
    """Check and display GPU availability and configuration."""
    if not TORCH_AVAILABLE:
        logger.error("PyTorch is not installed. Please install it first:")
        logger.error("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        return None
    
    logger.info("=" * 60)
    logger.info("GPU Configuration Check")
    logger.info("=" * 60)
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"CUDA is available with {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
        
        # Set default device
        device = torch.device("cuda:0")
        logger.info(f"Using device: {device}")
        
        # Display CUDA version
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"PyTorch Version: {torch.__version__}")
        
        # Memory info
        if gpu_count > 0:
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            cached = torch.cuda.memory_reserved(0) / (1024**3)
            logger.info(f"GPU Memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
    else:
        logger.warning("CUDA is not available. Using CPU (this will be slower).")
        device = torch.device("cpu")
    
    logger.info("=" * 60)
    return device


def get_available_models():
    """Get list of available Bangla TTS models."""
    return {
        "male": "tts_models/bn/custom/vits-male",
        "female": "tts_models/bn/custom/vits-female",
    }


def synthesize_speech(
    text: str,
    output_path: str = "output.wav",
    voice: str = "male",
    speaker_wav: str = None,
    use_gpu: bool = True
):
    """
    Synthesize speech from Bangla text.
    
    Args:
        text: Bangla text to convert to speech
        output_path: Path to save the output audio file
        voice: Voice type ('male' or 'female')
        speaker_wav: Path to speaker audio for voice cloning (optional)
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        Path to the generated audio file
    """
    if not TORCH_AVAILABLE:
        logger.error("PyTorch is not installed. Please install it first:")
        logger.error("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        sys.exit(1)
    
    try:
        from TTS.api import TTS
    except ImportError:
        logger.error("TTS library not found. Please install it with: pip install TTS")
        sys.exit(1)
    
    # Check GPU availability
    device = check_gpu_availability()
    
    # Determine if we should use GPU
    use_cuda = use_gpu and torch.cuda.is_available()
    
    # Get model name
    models = get_available_models()
    if voice not in models:
        logger.error(f"Invalid voice '{voice}'. Available voices: {list(models.keys())}")
        sys.exit(1)
    
    model_name = models[voice]
    logger.info(f"Loading TTS model: {model_name}")
    
    # Initialize TTS with GPU support if available
    tts = TTS(model_name=model_name, gpu=use_cuda)
    
    logger.info(f"Synthesizing speech for text: {text[:50]}...")
    
    if speaker_wav and os.path.exists(speaker_wav):
        # Voice cloning mode
        logger.info(f"Using voice cloning with speaker: {speaker_wav}")
        tts.tts_with_vc_to_file(
            text=text,
            speaker_wav=speaker_wav,
            file_path=output_path
        )
    else:
        # Standard TTS mode
        tts.tts_to_file(text=text, file_path=output_path)
    
    logger.info(f"Audio saved to: {output_path}")
    return output_path


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Bangla Text-to-Speech Synthesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with male voice
    python bangla_tts.py --text "আমার সোনার বাংলা" --output output.wav

    # Use female voice
    python bangla_tts.py --text "আমার সোনার বাংলা" --voice female --output output.wav

    # Voice cloning
    python bangla_tts.py --text "আমার সোনার বাংলা" --voice-clone speaker.wav --output cloned.wav

    # Force CPU usage
    python bangla_tts.py --text "আমার সোনার বাংলা" --no-gpu --output output.wav

    # Check GPU status only
    python bangla_tts.py --check-gpu
        """
    )
    
    parser.add_argument(
        "--text", "-t",
        type=str,
        help="Bangla text to convert to speech"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output.wav",
        help="Output audio file path (default: output.wav)"
    )
    
    parser.add_argument(
        "--voice", "-v",
        type=str,
        choices=["male", "female"],
        default="male",
        help="Voice type (default: male)"
    )
    
    parser.add_argument(
        "--voice-clone", "-vc",
        type=str,
        dest="speaker_wav",
        help="Path to speaker audio file for voice cloning"
    )
    
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration (use CPU only)"
    )
    
    parser.add_argument(
        "--check-gpu",
        action="store_true",
        help="Check GPU availability and exit"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available TTS models and exit"
    )
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.check_gpu:
        check_gpu_availability()
        return
    
    if args.list_models:
        models = get_available_models()
        print("\nAvailable Bangla TTS Models:")
        print("-" * 40)
        for voice, model in models.items():
            print(f"  {voice}: {model}")
        print()
        return
    
    # Validate text input
    if not args.text:
        parser.error("--text is required for speech synthesis")
    
    # Run synthesis
    synthesize_speech(
        text=args.text,
        output_path=args.output,
        voice=args.voice,
        speaker_wav=args.speaker_wav,
        use_gpu=not args.no_gpu
    )


if __name__ == "__main__":
    main()
