#!/bin/bash
# Setup script for Face Similarity Search Project using UV

echo "======================================================================"
echo "Visual Face Similarity Search - Setup Script (UV)"
echo "======================================================================"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null
then
    echo "❌ UV is not installed!"
    echo ""
    echo "Please install UV first using one of these methods:"
    echo ""
    echo "macOS/Linux:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    echo "Or with pip:"
    echo "  pip install uv"
    echo ""
    echo "Or with Homebrew:"
    echo "  brew install uv"
    echo ""
    exit 1
fi

echo "✅ UV is installed: $(uv --version)"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"
echo ""

# Create virtual environment with UV
echo "Creating virtual environment with UV..."
uv venv --python 3.8

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Sync dependencies from pyproject.toml
echo ""
echo "Installing dependencies from pyproject.toml..."
uv pip sync

# Install the project in editable mode
echo ""
echo "Installing project in editable mode..."
uv pip install -e .

# Install CLIP from GitHub (not available on PyPI)
echo ""
echo "Installing CLIP from GitHub..."
uv pip install git+https://github.com/openai/CLIP.git

# Create directory structure
echo ""
echo "Creating directory structure..."
python3 -c "from config import Config; Config.create_directories()" 2>/dev/null || echo "⚠️  Note: Run directory creation after config.py is available"

# Print success message
echo ""
echo "======================================================================"
echo "✅ Setup complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Add face images to: data/raw/sample_faces/"
echo ""
echo "3. Run the pipeline:"
echo "   python main.py --step all --max-images 100"
echo ""
echo "For more information, see README.md and QUICKSTART.md"
echo "======================================================================"
