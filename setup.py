from setuptools import setup, find_packages


def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename, 'r') as file:
        lines = file.read().splitlines()
    return [line for line in lines if line and not line.startswith("#")]


setup(
    name="daoram",
    version="0.1.0",
    author="Weiqi Feng, Xinle Cao",
    author_email="weltch1997@gmail.com, xinlecao72@gmail.com",
    description="A python library for ORAMs and OMAPs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Weiqi97/DAORAM",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),  # Dynamically read dependencies
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    license="Apache 2.0"
)