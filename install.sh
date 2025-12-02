#!/bin/bash
# =============================================================================
# Comprehensive Bangla TTS - Installation Script
# For Ubuntu 24 (Noble) with NVIDIA GPU support (RTX 5060Ti 16GB)
# =============================================================================

set -e

echo "=============================================="
echo "Comprehensive Bangla TTS - Installation Script"
echo "Optimized for Ubuntu 24 with NVIDIA GPU"
echo "=============================================="

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "Warning: Running as root. Consider using a virtual environment."
fi

# Check Ubuntu version
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "Detected OS: $NAME $VERSION_ID"
fi

# Check for NVIDIA GPU
echo ""
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "NVIDIA driver detected!"
else
    echo "Warning: nvidia-smi not found. Make sure NVIDIA drivers are installed."
    echo "Install NVIDIA drivers with: sudo apt install nvidia-driver-550"
fi

# Install system dependencies
echo ""
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    git

# Create virtual environment if it doesn't exist
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "Creating Python virtual environment..."
    python3 -m venv $VENV_DIR
fi

# Activate virtual environment
source $VENV_DIR/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch with CUDA 12.4 support..."
echo "This is optimized for RTX 50 series GPUs"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install project dependencies
echo ""
echo "Installing project dependencies..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')
"

echo ""
echo "=============================================="
echo "Installation Complete!"
echo "=============================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To test the installation, run:"
echo "  python bangla_tts.py --check-gpu"
echo ""
echo "To synthesize speech, run:"
echo "  python bangla_tts.py --text \"আমার সোনার বাংলা\" --output test.wav"
echo ""
