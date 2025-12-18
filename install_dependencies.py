import subprocess
import sys

def install_packages():  
    packages = [
        "streamlit==1.28.2",
        "librosa==0.10.1",
        "numpy==1.24.3",
        "matplotlib==3.7.2",
        "seaborn==0.12.2",
        "pydub==0.25.1",
        "sounddevice==0.4.6",
        "scipy==1.10.1",
        "scikit-learn==1.3.0", 
        "plotly==5.15.0",
        "pandas==2.0.3",
        "reportlab==4.0.4",
        "typing-extensions==4.8.0",
        "pyopenssl==23.2.0",
        "cryptography==41.0.8"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")

if __name__ == "__main__":
    print("Installing VoxSense dependencies...")
    install_packages()
    print("\n🎉 All packages installed successfully!")
    print("\nNow run: streamlit run main.py")