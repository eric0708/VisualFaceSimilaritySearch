#!/bin/bash
# Setup script for Face Similarity Search Project

echo "======================================================================"
echo "Visual Face Similarity Search - Setup Script"
echo "======================================================================"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Install CLIP
echo ""
echo "Installing CLIP..."
pip install git+https://github.com/openai/CLIP.git

# Create directory structure
echo ""
echo "Creating directory structure..."
python3 -c "from config import Config; Config.create_directories()"

# Print success message
echo ""
echo "======================================================================"
echo "✅ Setup complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Add face images to: data/raw/sample_faces/"
echo ""
echo "3. Run the pipeline:"
echo "   python main.py --step all --max-images 100"
echo ""
echo "For more information, see README.md and QUICKSTART.md"
echo "======================================================================"
