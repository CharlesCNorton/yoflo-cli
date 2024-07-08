from setuptools import setup, find_packages

setup(
    name='yoflo',
    version='0.2.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'torch',
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
