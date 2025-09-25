#!/usr/bin/env python3
"""
Setup script for TGFL Market Scenario Simulator
Handles environment setup, dependency installation, and basic validation
"""

import subprocess
import sys
import os
from pathlib import Path
import platform
import venv as _venv

def run_command(cmd, check=True, cwd=None):
    """Run a shell command and return result"""
    print(f"Running: {cmd}")
    # Use shell invocation to allow paths with spaces and complex commands
    try:
        # Prefer passing a list when cmd is a sequence, otherwise run through the shell
        if isinstance(cmd, (list, tuple)):
            result = subprocess.run(
                cmd,
                check=check,
                capture_output=True,
                text=True,
                cwd=cwd,
            )
        else:
            # Use shell=True for complex commands and paths with spaces
            result = subprocess.run(
                cmd,
                check=check,
                capture_output=True,
                text=True,
                cwd=cwd,
                shell=True,
                executable=os.getenv("SHELL", "/bin/bash"),
            )

        if result.stdout:
            print(result.stdout)
        if result.stderr:
            # Print stderr only when present to help debug
            print(result.stderr)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if getattr(e, 'stderr', None):
            print(f"Error output: {e.stderr}")
        if check:
            sys.exit(1)
        return e

def check_python_version():
    """Ensure Python 3.10+ is available"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("ERROR: Python 3.10+ required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    print(f"Python {version.major}.{version.minor}.{version.micro} OK")

def check_node_version():
    """Check if Node.js 18+ is available"""
    try:
        result = run_command("node --version", check=False)
        if result.returncode == 0:
            version = result.stdout.strip().replace('v', '')
            major_version = int(version.split('.')[0])
            if major_version >= 18:
                print(f"Node.js {version} OK")
                return True
            else:
                print(f"Warning: Node.js {version} found, but 18+ recommended")
                return False
    except:
        pass
    
    print("ERROR: Node.js 18+ not found")
    print("Please install Node.js from https://nodejs.org/")
    return False

def setup_python_environment():
    """Set up Python virtual environment and install dependencies"""
    project_root = Path(__file__).parent.parent  # Go up from scripts/ to project root
    api_dir = project_root / "api"
    
    print("\nSetting up Python environment...")
    
    # Create virtual environment
    venv_path = api_dir / ".venv"
    if not venv_path.exists():
        print("Creating virtual environment via venv module...")
        try:
            _venv.EnvBuilder(with_pip=True).create(str(venv_path))
        except Exception as e:
            print(f"Warning: venv creation failed: {e}")
            # Try to detect existing venvs inside api/ (common in this repo)
            candidates = []
            for child in api_dir.iterdir():
                if (child / 'pyvenv.cfg').exists():
                    candidates.append(child)
            if candidates:
                chosen = candidates[0]
                print(f"Using existing venv at {chosen}")
                venv_path = chosen
            else:
                print("ERROR: Could not create a virtualenv and no existing venv found under api/")
                sys.exit(1)
    
    # Determine activation script and venv python based on OS
    if platform.system() == "Windows":
        activate_script = venv_path / "Scripts" / "activate"
        python_path = venv_path / "Scripts" / "python"
    else:
        activate_script = venv_path / "bin" / "activate"
        python_path = venv_path / "bin" / "python"
    
    # Install dependencies
    requirements_file = api_dir / "requirements.txt"
    if requirements_file.exists():
        print("Installing Python dependencies...")
        # Run pip via the venv python executable to avoid missing pip binary name
        run_command([str(python_path), "-m", "pip", "install", "--upgrade", "pip"]) 

    # Install some build tools first to allow building packages that require pyproject hooks
    print("Installing build tools (setuptools, wheel, build)...")
    run_command([str(python_path), "-m", "pip", "install", "setuptools", "wheel", "build"])        
    # Install torch separately using the PyTorch CPU index to avoid platform wheel issues
    torch_line = None
    other_lines = []
    with open(requirements_file, 'r') as f:
        for line in f:
            sline = line.strip()
            if not sline or sline.startswith('#'):
                continue
            if sline.startswith('torch'):
                torch_line = sline
            else:
                other_lines.append(sline)

    torch_installed = False
    if torch_line:
        # Extract version if present
        import re
        m = re.search(r'torch==([0-9\.]+)', torch_line)
        if m:
            version = m.group(1)
            torch_spec = f'torch=={version}+cpu'
        else:
            torch_spec = 'torch'

        print(f"Installing PyTorch CPU wheel: {torch_spec}")
        try:
            run_command([str(python_path), "-m", "pip", "install", torch_spec, "--index-url", "https://download.pytorch.org/whl/cpu" ])
            torch_installed = True
        except SystemExit:
            # pip install raised a SystemExit (run_command exits on check failures)
            torch_installed = False
        except Exception:
            torch_installed = False

    # Install the remaining requirements from a temp file
    import tempfile
    temp_req = None
    if other_lines:
        with tempfile.NamedTemporaryFile('w', delete=False) as tf:
            tf.write('\n'.join(other_lines))
            temp_req = tf.name
        try:
            run_command([str(python_path), "-m", "pip", "install", "-r", temp_req])
        finally:
            try:
                if temp_req and os.path.exists(temp_req):
                    os.unlink(temp_req)
            except Exception:
                pass

    if not torch_installed:
        print("\nWarning: PyTorch could not be installed automatically for this Python version.")
        print("Common solutions:")
        print("- Use Python 3.10 or 3.11 (recommended) and re-run setup")
        print("- Install PyTorch via conda/miniforge: https://pytorch.org/get-started/locally/")
        print("- Or install a compatible wheel manually from https://download.pytorch.org/whl/cpu")
    else:
        print("requirements.txt not found in api/ directory")
    
    print("Python environment setup complete")
    return str(activate_script)

def setup_node_environment():
    """Set up Node.js environment and install dependencies"""
    project_root = Path(__file__).parent.parent  # Go up from scripts/ to project root
    web_dir = project_root / "web"
    
    if not check_node_version():
        return False
    
    print("\nSetting up Node.js environment...")
    
    package_json = web_dir / "package.json"
    if package_json.exists():
        print("Installing Node.js dependencies...")
        # Use list form to avoid shell path parsing issues
        run_command(["npm", "install"], cwd=web_dir)
        print("Node.js environment setup complete")
        return True
    else:
        print("Warning: package.json not found in web/ directory")
        return False

def create_env_files():
    """Create .env files from examples"""
    project_root = Path(__file__).parent.parent  # Go up from scripts/ to project root
    
    print("\nSetting up environment files...")
    
    # API .env
    api_env_example = project_root / "api" / ".env.example"
    api_env = project_root / "api" / ".env.local"
    
    if api_env_example.exists() and not api_env.exists():
        print("Creating API .env.local from example...")
        with open(api_env_example) as src, open(api_env, 'w') as dst:
            dst.write(src.read())
    
    # Web .env
    web_env_example = project_root / "web" / ".env.example"
    web_env = project_root / "web" / ".env.local"
    
    if web_env_example.exists() and not web_env.exists():
        print("Creating Web .env.local from example...")
        with open(web_env_example) as src, open(web_env, 'w') as dst:
            dst.write(src.read())
    
    print("Environment files created")

def create_data_directories():
    """Create necessary data directories"""
    project_root = Path(__file__).parent.parent  # Go up from scripts/ to project root
    
    print("\nCreating data directories...")
    
    directories = [
        "data/cache",
        "data/models", 
        "data/results",
        "ml/checkpoints",
        "api/logs"
    ]
    
    for dir_path in directories:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {dir_path}")
    
    print("Data directories created")

def validate_setup():
    """Run basic validation tests"""
    project_root = Path(__file__).parent.parent  # Go up from scripts/ to project root
    
    print("\nValidating setup...")
    
    # Test Python imports
    try:
        import torch
        import fastapi
        import pandas
        print("Python dependencies import successfully")
    except ImportError as e:
        print(f"ERROR: Python import error: {e}")
        return False
    
    # Test PyTorch
    try:
        import torch
        x = torch.randn(2, 2)
        print(f"PyTorch working (CPU: {not torch.cuda.is_available()})")
    except Exception as e:
        print(f"PyTorch test failed: {e}")
        return False
    
    return True

def print_next_steps(activate_script):
    """Print instructions for next steps"""
    print("\nSetup complete!")
    print("\nNext steps:")
    print("1. Start the API server:")
    if platform.system() == "Windows":
        print(f"   cd api && {activate_script} && python main.py")
    else:
        print(f"   cd api && source {activate_script} && python main.py")
    
    print("\n2. In a new terminal, start the web server:")
    print("   cd web && npm run dev")
    
    print("\n3. Open http://localhost:3000 in your browser")
    print("\n4. API docs available at http://localhost:8000/docs")

def main():
    """Main setup function"""
    print("TGFL Market Scenario Simulator Setup")
    print("=" * 50)
    
    # Check requirements
    check_python_version()
    
    # Setup environments
    activate_script = setup_python_environment()
    setup_node_environment()
    
    # Setup configuration
    create_env_files()
    create_data_directories()
    
    # Validate
    if validate_setup():
        print_next_steps(activate_script)
    else:
        print("\nSetup validation failed. Please check errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()