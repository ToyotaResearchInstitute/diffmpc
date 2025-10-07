from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

requirements = [line.strip() for line in open("./requirements.txt", "r")]

setup(
    name="diffmpc",
    version="1.0.0",
    author="diffmpc",
    author_email="thomas.lew@tri.global",
    description="Differentiable Model Predictive Control on the GPU",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ToyotaResearchInstitute",
    install_requires=requirements,
    packages=find_packages(),
    python_requires=">=3.10",
)
