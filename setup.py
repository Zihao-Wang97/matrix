from setuptools import setup, find_packages

setup(
    name="hawp-laq",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pyyaml>=6.0",
    ],
)
