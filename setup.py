"""Setup script for westminster_ground_truth_analysis package."""

from pathlib import Path
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="westminster_ground_truth_analysis",
    version="0.1.0",
    author="Your Name",
    description="Ground truth analysis for orthomosaic creation from DJI drone imagery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/westminster_ground_truth_analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)

