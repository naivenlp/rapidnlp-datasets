import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="naivenlp-datasets",
    version="0.1.0",
    description="Data pipelines for TensorFlow and PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/naivenlp/naivenlp-datasets",
    author="ZhouYang Luo",
    author_email="zhouyang.luo@gmail.com",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        "tokenizers",
    ],
    extras_require={
        "tf": ["tensorflow"],
        "pt": ["torch", "torchaudio", "torchvision"],
    },
    license="Apache Software License",
    classifiers=(
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ),
)
