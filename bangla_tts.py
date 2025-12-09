#!/usr/bin/env python3
"""
Bangla Text-to-Speech (TTS) - Standalone Python Script
Converted from Jupyter notebooks for standalone use.
Supports NVIDIA GPU acceleration.

Usage:
    python bangla_tts.py --text "আমার সোনার বাংলা" --output output.wav
    python bangla_tts.py --text "আমার সোনার বাংলা" --voice male --output output.wav
    python bangla_tts.py --text "আমার সোনার বাংলা" --voice female --output output.wav
    
    # Using custom local model weights (preserves trained voice and speech patterns)
    python bangla_tts.py --text "আমার নাম কি" --model-path checkpoint.pth --config-path config.json --output output.wav
    
    # Voice conversion with FreeVC (converts voice to match reference speaker)
    # Note: This loses the custom model's trained speech patterns
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


def setup_gpu_memory_allocation():
    """Configure GPU to allocate maximum VRAM for better performance."""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return
    
    try:
        # Set CUDA to maximize performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # Pre-allocate GPU memory to reserve the memory pool
        device = torch.device("cuda:0")
        total_memory = torch.cuda.get_device_properties(0).total_memory
        
        # Try to allocate ~90% of available VRAM to reserve the memory pool
        target_allocation = int(total_memory * 0.90)
        try:
            # Allocate in chunks to avoid OOM
            chunk_size = 256 * 1024 * 1024  # 256 MB chunks
            tensors = []
            allocated = 0
            while allocated < target_allocation:
                try:
                    t = torch.empty(chunk_size // 4, dtype=torch.float32, device=device)
                    tensors.append(t)
                    allocated += chunk_size
                except RuntimeError:
                    break
            
            # Keep track of allocated amount before freeing
            reserved_amount = allocated
            
            # Free the tensors - the memory pool remains reserved by PyTorch
            # Note: We do NOT call empty_cache() to keep the memory pool allocated
            del tensors
            
            logger.info(f"GPU memory pool initialized (~{reserved_amount / (1024**3):.2f} GB reserved)")
        except Exception as e:
            logger.warning(f"Could not pre-allocate GPU memory: {e}")
        
    except Exception as e:
        logger.warning(f"GPU memory optimization setup failed: {e}")


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
        
        # Setup GPU memory allocation for maximum utilization
        setup_gpu_memory_allocation()
        
        # Memory info after allocation
        if gpu_count > 0:
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            cached = torch.cuda.memory_reserved(0) / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {cached:.2f} GB, Total: {total:.2f} GB")
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
    use_gpu: bool = True,
    model_path: str = None,
    config_path: str = None
):
    """
    Synthesize speech from Bangla text.
    
    Args:
        text: Bangla text to convert to speech
        output_path: Path to save the output audio file
        voice: Voice type ('male' or 'female') - used only when model_path is not provided
        speaker_wav: Path to speaker audio for voice conversion using FreeVC (optional).
            Note: When using custom model (model_path), voice conversion will lose
            the trained speech patterns. Use this only to match a different speaker's voice.
        use_gpu: Whether to use GPU acceleration
        model_path: Path to custom model checkpoint file (optional)
        config_path: Path to custom model config file (optional)
    
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
    
    if use_cuda:
        logger.info("GPU acceleration ENABLED")
    else:
        logger.info("GPU acceleration DISABLED - using CPU")
    
    # Validate speaker_wav if provided
    if speaker_wav and not os.path.exists(speaker_wav):
        logger.error(f"Speaker wav file not found: {speaker_wav}")
        logger.error("Please provide a valid path to the speaker audio file for voice cloning.")
        sys.exit(1)
    
    # Warn if using voice conversion with custom model
    if speaker_wav and model_path:
        logger.warning("=" * 60)
        logger.warning("WARNING: Using voice conversion (--voice-clone) with custom model")
        logger.warning("This will apply FreeVC voice conversion, which may lose the")
        logger.warning("trained speech patterns from your custom model.")
        logger.warning("If you want to preserve your model's voice, remove --voice-clone")
        logger.warning("=" * 60)
    
    # Load TTS model
    if model_path and config_path:
        # Use custom local model weights
        if not os.path.exists(model_path):
            logger.error(f"Model checkpoint file not found: {model_path}")
            sys.exit(1)
        if not os.path.exists(config_path):
            logger.error(f"Model config file not found: {config_path}")
            sys.exit(1)
        
        logger.info(f"Loading custom TTS model from: {model_path}")
        logger.info(f"Using config: {config_path}")
        logger.info("Custom model will use its trained voice and speech patterns")
        tts = TTS(model_path=model_path, config_path=config_path, gpu=use_cuda)
    else:
        # Use pre-trained model from coqui-ai model zoo
        models = get_available_models()
        if voice not in models:
            logger.error(f"Invalid voice '{voice}'. Available voices: {list(models.keys())}")
            sys.exit(1)
        
        model_name = models[voice]
        logger.info(f"Loading TTS model: {model_name}")
        tts = TTS(model_name=model_name, gpu=use_cuda)
    
    logger.info(f"Synthesizing speech for text: {text[:50]}...")
    
    if speaker_wav:
        # Voice conversion mode using FreeVC - speaker_wav is validated to exist above
        logger.info(f"Using FreeVC voice conversion with speaker: {speaker_wav}")
        logger.info("Note: This converts voice to match reference speaker (may lose trained patterns)")
        tts.tts_with_vc_to_file(
            text=text,
            speaker_wav=speaker_wav,
            file_path=output_path
        )
    else:
        # Standard TTS mode - uses model's native trained voice
        logger.info("Using model's native voice (no voice conversion)")
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
    # Basic usage with male voice (uses pre-trained coqui-ai model)
    python bangla_tts.py --text "আমার সোনার বাংলা" --output output.wav

    # Use female voice
    python bangla_tts.py --text "আমার সোনার বাংলা" --voice female --output output.wav

    # Using custom local model weights (RECOMMENDED for trained models)
    # This preserves your model's trained voice and speech patterns
    python bangla_tts.py --text "আমার নাম কি" --model-path checkpoint.pth --config-path config.json --output output.wav

    # Voice conversion with FreeVC (converts to match reference speaker)
    # WARNING: This may lose trained speech patterns from custom models
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
        help="Path to speaker audio for FreeVC voice conversion. WARNING: This converts voice to match reference speaker and may lose trained speech patterns from custom models"
    )
    
    parser.add_argument(
        "--model-path", "-m",
        type=str,
        help="Path to custom model checkpoint file. Uses trained voice/speech patterns directly"
    )
    
    parser.add_argument(
        "--config-path", "-c",
        type=str,
        help="Path to custom model config file (required when using --model-path)"
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
    
    # Validate model path and config path
    if args.model_path and not args.config_path:
        parser.error("--config-path is required when using --model-path")
    if args.config_path and not args.model_path:
        parser.error("--model-path is required when using --config-path")
    
    # Run synthesis
    synthesize_speech(
        text=args.text,
        output_path=args.output,
        voice=args.voice,
        speaker_wav=args.speaker_wav,
        use_gpu=not args.no_gpu,
        model_path=args.model_path,
        config_path=args.config_path
    )


if __name__ == "__main__":
    main()
