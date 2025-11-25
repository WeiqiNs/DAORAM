from pathlib import Path

from setuptools import find_packages, setup


def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with (Path(__file__).parent / filename).open('r', encoding='utf-8') as file:
        lines = file.read().splitlines()
    return [line for line in lines if line and not line.startswith("#")]


setup(
    name="pyoblivlib",
    version="0.1.0",
    author="Weiqi Feng, Xinle Cao",
    author_email="weltch1997@gmail.com, xinlecao72@gmail.com",
    description="A Python library of oblivious data-structure algorithms (ORAM, OMAP, and oblivious graphs).",
    long_description=(Path(__file__).parent / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/Weiqi97/DAORAM",
    packages=find_packages(include=["pyoblivlib", "pyoblivlib.*"]),
    install_requires=parse_requirements("requirements.txt"),  # Dynamically read dependencies
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    license="Apache 2.0"
)
