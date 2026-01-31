#!/bin/bash
# Quick start script for Version 1 (Phoenix Protocol Baseline)
# v1 is the ROOT directory - you're already there!

echo "🚀 Starting Phoenix Protocol v1 (Baseline Implementation)"
echo "=================================================="
echo ""
echo "📍 Note: v1 IS the root directory - no need to navigate!"
echo ""

# Navigate to v1 directory
cd "$(dirname "$0")/v1" || exit 1

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate || . venv/Scripts/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "📚 Quick Commands (run from root):"
echo "  • Train baseline model:    python one_click_train_test.py --mode train --model-type neurosnake_ca"
echo "  • Evaluate model:          python one_click_train_test.py --mode test"
echo "  • Run validation:          python validate_implementation.py"
echo "  • View documentation:      cat README.md"
echo ""
echo "🎯 Version: v1 - Phoenix Protocol Baseline (ROOT)"
echo "📊 Expected Accuracy: ~95.2%"
echo "💾 Parameters: 2.1M"
echo ""
echo "To deactivate the environment, run: deactivate"
echo ""
