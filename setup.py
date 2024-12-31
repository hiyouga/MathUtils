from setuptools import find_packages, setup


def get_requires():
    with open("requirements.txt", encoding="utf-8") as f:
        file_content = f.read()
        lines = [line.strip() for line in file_content.strip().split("\n") if not line.startswith("#")]
        return lines


setup(
    name="math_utils",
    version="0.0.1",
    packages=find_packages(),
    install_requires=get_requires(),
    entry_points={"console_scripts": ["math_utils = prm800k.interface:main"]},
)
