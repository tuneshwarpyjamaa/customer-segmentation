"""
Dataset Download Script

Downloads the Online Retail dataset from Kaggle.
"""

import os
from pathlib import Path
import sys
import urllib.request
import zipfile


def download_from_kaggle():
    """
    Download dataset using Kaggle API.
    Requires kaggle.json credentials file.
    """
    try:
        import kaggle
        
        print("Downloading dataset from Kaggle...")
        print("Dataset: vijayuv/onlineretail")
        
        # Download to data/raw directory
        project_root = Path(__file__).parent.parent
        download_path = project_root / "data" / "raw"
        download_path.mkdir(parents=True, exist_ok=True)
        
        # Download dataset
        kaggle.api.dataset_download_files(
            'vijayuv/onlineretail',
            path=str(download_path),
            unzip=True
        )
        
        print(f"\n✓ Dataset downloaded successfully to: {download_path}/")
        print("  File: OnlineRetail.xlsx")
        
        return True
        
    except ImportError:
        print("❌ Kaggle API not installed.")
        print("\nTo install: pip install kaggle")
        return False
    
    except Exception as e:
        print(f"❌ Error downloading from Kaggle: {e}")
        return False


def print_manual_instructions():
    """Print manual download instructions."""
    print("\n" + "="*80)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*80)
    print("\n📥 To download the dataset manually:")
    print("\n1. Visit: https://www.kaggle.com/datasets/vijayuv/onlineretail")
    print("2. Click the 'Download' button (you may need to sign in)")
    print("3. Extract the downloaded zip file")
    print("4. Move 'OnlineRetail.xlsx' to: data/raw/")
    print("\n" + "="*80 + "\n")


def check_kaggle_credentials():
    """Check if Kaggle credentials are configured."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    
    if not kaggle_json.exists():
        print("❌ Kaggle API credentials not found.")
        print("\n📝 To set up Kaggle API:")
        print("1. Go to: https://www.kaggle.com/account")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New API Token'")
        print("4. Move the downloaded kaggle.json to: ~/.kaggle/")
        print("5. On Linux/Mac: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    return True


def main():
    """Main function."""
    print("\n" + "="*80)
    print("ONLINE RETAIL DATASET DOWNLOADER")
    print("="*80 + "\n")
    
    # Check if file already exists
    project_root = Path(__file__).parent.parent
    data_file = project_root / "data" / "raw" / "OnlineRetail.xlsx"
    
    if data_file.exists():
        print(f"✓ Dataset already exists at: {data_file}")
        response = input("\nDo you want to re-download? (y/N): ").strip().lower()
        if response != 'y':
            print("Keeping existing file.")
            sys.exit(0)
    
    # Try automated download
    print("Attempting automated download via Kaggle API...\n")
    
    if not check_kaggle_credentials():
        print_manual_instructions()
        sys.exit(1)
    
    success = download_from_kaggle()
    
    if not success:
        print_manual_instructions()
        sys.exit(1)
    
    # Verify download
    if data_file.exists():
        file_size = data_file.stat().st_size / (1024 * 1024)  # MB
        print(f"\n✅ Download successful!")
        print(f"   File size: {file_size:.2f} MB")
        print(f"   Location: {data_file}")
        print("\n🚀 You can now run the analysis:")
        print("   python src/main.py")
    else:
        print("\n❌ Download failed. Please try manual download.")
        print_manual_instructions()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Download cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
