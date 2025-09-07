"""
Installation script for ComfyUI HunyuanVideo-Foley Custom Node
"""

import os
import sys
import subprocess
import pkg_resources
from pathlib import Path

def parse_requirements(file_path):
    """Parse requirements file and handle git dependencies."""
    requirements = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if line.startswith('git+'):
                    # For git repos, find the package name from the egg fragment
                    egg_name = None
                    if '#egg=' in line:
                        egg_name = line.split('#egg=')[-1]
                    
                    if egg_name:
                        requirements.append((egg_name, line))
                    else:
                        print(f"⚠️ Git requirement '{line}' is missing the '#egg=' part and cannot be checked. It will be installed regardless.")
                        # Fallback: We can't check it, so we'll just try to install it.
                        # The package name is passed as None to signal an install attempt.
                        requirements.append((None, line))
                else:
                    # Standard package
                    req = pkg_resources.Requirement.parse(line)
                    requirements.append((req.project_name, str(req)))
    return requirements

def check_and_install_requirements():
    """Check and install required packages without overriding existing ones."""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ Requirements file not found!")
        return False
    
    try:
        print("🚀 Checking and installing requirements...")
        
        # Get list of (package_name, requirement_string)
        requirements = parse_requirements(requirements_file)
        
        for pkg_name, requirement_str in requirements:
            # If pkg_name is None, it's a git URL we couldn't parse. Try installing.
            if pkg_name is None:
                print(f"Attempting to install from git: {requirement_str}")
                try:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', requirement_str])
                    print(f"✅ Successfully installed {requirement_str}")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed to install {requirement_str}: {e}")
                continue

            # Check if the package is already installed
            try:
                pkg_resources.require(requirement_str)
                print(f"✅ {pkg_name} is already installed and meets version requirements.")
            except pkg_resources.DistributionNotFound:
                print(f"Installing {pkg_name}...")
                try:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', requirement_str])
                    print(f"✅ Successfully installed {pkg_name}")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed to install {pkg_name}: {e}")
            except pkg_resources.VersionConflict as e:
                print(f"⚠️ Version conflict for {pkg_name}: {e.req} is required, but you have {e.dist}.")
                print("   Skipping upgrade to avoid conflicts with other nodes. If you encounter issues, please update this package manually.")
            except Exception as e:
                print(f"An unexpected error occurred while checking {pkg_name}: {e}")

        print("✅ All dependencies checked.")
        return True
        
    except Exception as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def setup_model_directories():
    """Create necessary model directories"""
    base_dir = Path(__file__).parent.parent.parent  # Go up to ComfyUI root
    
    # Create ComfyUI/models/foley directory for automatic downloads
    foley_models_dir = base_dir / "models" / "foley"
    foley_models_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created ComfyUI models directory: {foley_models_dir}")
    
    # Also create local fallback directories
    node_dir = Path(__file__).parent
    local_dirs = [
        node_dir / "pretrained_models",
        node_dir / "configs"
    ]
    
    for dir_path in local_dirs:
        dir_path.mkdir(exist_ok=True)
        print(f"✓ Created local directory: {dir_path}")

def main():
    """Main installation function"""
    print("🚀 Installing ComfyUI HunyuanVideo-Foley Custom Node...")
    
    # Install requirements
    if not check_and_install_requirements():
        print("❌ Failed to install requirements")
        return False
    
    # Setup directories
    setup_model_directories()
    
    print("📋 Installation completed!")
    print()
    print("📌 Next steps:")
    print("1. Restart ComfyUI to load the custom nodes")
    print("2. Models will be automatically downloaded when you first use the node")
    print("3. Alternatively, manually download models and place them in ComfyUI/models/foley/")
    print("4. Model URLs are configured in model_urls.py (can be updated as needed)")
    print()
    
    return True

if __name__ == "__main__":
    main()