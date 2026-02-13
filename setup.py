from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()


setup(
    name="music-data-pipeline",
    version="0.1",
    author="atsigman",
    packages=find_packages(where="./src"),
    package_dir={"": "./src"},
    install_requires=requirements,
)
