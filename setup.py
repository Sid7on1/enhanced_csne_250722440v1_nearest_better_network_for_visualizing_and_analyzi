import setuptools
from setuptools import find_packages
from pathlib import Path

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="enhanced-cs-ne-2507-22440v1-nearest-better-network",
    version="1.0.0",
    author="XR Eye Tracking Team",
    author_email="xr.eyetracking@example.com",
    description="Enhanced AI for computer vision-based eye tracking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/enhanced-cs-ne-2507-22440v1-nearest-better-network",  # Replace with your repository link
    project_urls={
        "Bug Reports": "https://github.com/example/enhanced-cs-ne-2507-22440v1-nearest-better-network/issues",  # Replace with your repository issue link
        "Funding": "https://donate.example.org",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=["torch", "numpy", "pandas"],
    include_package_data=True,
    package_data={"": ["*.json", "*.yaml", "*.yml"]},
    entry_points={
        "console_scripts": [
            "enhanced_cs_ne_2507_22440v1_nbn = enhanced_cs_ne_2507_22440v1_nbn.cli:main",
        ]
    },
)

This setup.py script is designed for a Python package named 'enhanced-cs-ne-2507-22440v1-nearest-better-network', which is part of an enterprise XR eye-tracking system project. It includes essential metadata and specifies the package's dependencies, entry points, and other configurations.

The script starts by importing the necessary modules from setuptools and pathlib. The long_description is read from the README.md file using a context manager.

The setuptools.setup() function is then called to define the package's metadata, including the package name, version, author, description, long description, URL, project URLs, classifiers, package directory, and more.

The packages are discovered using the find_packages() function, which searches in the "src" directory. The script specifies the Python version requirement (Python 3.7 or higher) and lists the required dependencies, including torch, numpy, and pandas.

The include_package_data option is set to True to include optional package data files, and package_data is used to specify wildcard patterns for including specific types of files.

Additionally, the script defines a console script entry point named "enhanced_cs_ne_2507_22440v1_nbn", which allows the package to be executed from the command line. The entry point points to the main function in the enhanced_cs_ne_2507_22440v1_nbn.cli module.

This setup.py script adheres to the specified requirements and serves as a comprehensive package installation setup for the enhanced-cs-ne-2507-22440v1-nearest-better-network project.