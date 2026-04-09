from setuptools import setup, find_packages

setup(
    name="geno-storage",
    version="0.1.0",
    description="Provably Verifiable Genotype Storage with LMDB",
    packages=find_packages(),
    install_requires=[
        "lmdb>=1.0.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
    },
    python_requires=">=3.9",
)