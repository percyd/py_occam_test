from setuptools import setup, find_packages

setup(
    name="py-occam",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "scipy>=1.4.0",
        "pyyaml>=5.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=20.8b1",
            "isort>=5.6.0",
        ]
    }
)