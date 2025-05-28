#!/usr/bin/env python3
"""
MLX Architecture UI Launcher
Simple script to launch the Streamlit interface with proper configuration
"""

import sys
import subprocess
import importlib

def check_dependencies():
    """Check if all required dependencies are installed"""
    
    required_packages = [
        'streamlit',
        'plotly', 
        'pandas',
        'mlx',
        'mlx_lm'
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} - MISSING")
    
    if missing:
        print(f"\n🚨 Missing dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    print(f"\n✅ All dependencies satisfied!")
    return True

def launch_ui():
    """Launch the Streamlit UI"""
    
    print("🚀 Launching MLX Architecture Lab UI...")
    print("🌐 The interface will open in your browser")
    print("📍 Default URL: http://localhost:8501")
    print("\n" + "="*50)
    
    # Launch streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "mlx_architecture_ui.py",
            "--server.port", "8501",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 UI stopped by user")
    except Exception as e:
        print(f"\n❌ Failed to launch UI: {e}")

def main():
    """Main launcher function"""
    
    print("🧠 MLX Architecture Lab - UI Launcher")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n💡 Install missing dependencies and try again")
        return 1
    
    # Launch UI
    launch_ui()
    return 0

if __name__ == "__main__":
    exit(main()) 