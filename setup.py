"""
Setup script for the JioPay RAG Chatbot project.
"""
import subprocess
import sys
import os
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up JioPay RAG Chatbot project...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 9):
        print("❌ Python 3.9+ is required")
        sys.exit(1)
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Install Playwright browsers
    if not run_command("playwright install", "Installing Playwright browsers"):
        print("⚠️  Playwright installation failed - headless browser features may not work")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    env_example = Path("env.example")
    
    if not env_file.exists() and env_example.exists():
        print("📝 Creating .env file from template...")
        with open(env_example, 'r') as f:
            content = f.read()
        with open(env_file, 'w') as f:
            f.write(content)
        print("✅ .env file created. Please edit it with your API keys.")
    
    # Verify directory structure
    required_dirs = [
        "src/scraping", "src/processing", "src/embeddings", 
        "src/retrieval", "src/generation", "src/frontend", 
        "src/evaluation", "data/scraped", "data/processed", 
        "data/chroma_db", "data/faiss_index", "data/evaluation", 
        "tests", "reports"
    ]
    
    print("📁 Verifying directory structure...")
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ All directories created successfully")
    
    # Test imports
    print("🧪 Testing imports...")
    try:
        import fastapi
        import uvicorn
        import requests
        import beautifulsoup4
        import chromadb
        import openai
        import google.generativeai
        print("✅ All required packages imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .env file with your API keys")
    print("2. Run: python src/main.py (to start the API)")
    print("3. Run: streamlit run src/frontend/streamlit_app.py (for web UI)")
    print("\n📚 Documentation:")
    print("- README.md: Project overview and setup")
    print("- DATA_CARD.md: Data collection details")
    print("- src/config.py: Configuration settings")


if __name__ == "__main__":
    main()
