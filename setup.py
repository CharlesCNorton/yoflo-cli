from setuptools import setup, find_packages
import os
import platform
from pathlib import Path
import sys

def find_cuda_path():
    system = platform.system()

    if system == 'Windows':
        default_path = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA'
        cuda_path = os.getenv('CUDA_PATH', default_path)
        if not Path(cuda_path).exists():
            possible_versions = [p for p in Path(default_path).iterdir() if p.is_dir()]
            if possible_versions:
                cuda_path = str(possible_versions[-1])
            else:
                raise FileNotFoundError(f"No CUDA versions found in {default_path}.")
    else:
        cuda_path = os.getenv('CUDA_PATH', '/usr/local/cuda')

    if not Path(cuda_path).exists():
        raise FileNotFoundError(f"CUDA not found at {cuda_path}. Please set the CUDA_PATH environment variable correctly.")

    return cuda_path

def set_environment_variables():
    cuda_home = os.getenv('CUDA_HOME')
    if cuda_home:
        print(f"CUDA_HOME is already set to {cuda_home}. Skipping environment variable setup.")
        return

    try:
        cuda_path = find_cuda_path()
        os.environ['CUDA_HOME'] = cuda_path
        os.environ['PATH'] = f"{cuda_path}/bin:{os.environ.get('PATH', '')}"
        os.environ['LD_LIBRARY_PATH'] = f"{cuda_path}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
        print(f"Environment variables set for {platform.system()}:")
        print(f"CUDA_HOME={os.environ['CUDA_HOME']}")
        print(f"PATH={os.environ['PATH']}")
        print(f"LD_LIBRARY_PATH={os.environ['LD_LIBRARY_PATH']}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during environment variable setup: {e}")
        sys.exit(1)

try:
    set_environment_variables()
except Exception as e:
    print(f"Critical error: {e}")
    sys.exit(1)

try:
    setup(
        name='yoflo',
        version='0.2.6',
        packages=find_packages(),
        include_package_data=True,
        install_requires=[
            'torch',
            'timm',
            'transformers>=4.38.0',
            'Pillow',
            'numpy',
            'opencv-python',
            'huggingface_hub',
            'datasets',
            'flash-attn',
            'scipy'
        ],
        entry_points={
            'console_scripts': [
                'yoflo=yoflo.yoflo:main',
            ],
        },
        author='Charles Norton',
        author_email='CharlesCornellNorton@gmail.com',
        description='YO-FLO: A proof-of-concept in using advanced vision models as a YOLO alternative.',
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        url='https://github.com/CharlesCNorton/yoflo-cli',
        classifiers=[
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
        ],
        python_requires='>=3.6',
    )
except FileNotFoundError as e:
    print(f"File not found: {e}")
    sys.exit(1)
except IOError as e:
    print(f"I/O error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(1)
