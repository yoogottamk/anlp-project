from setuptools import setup, find_packages
from pathlib import Path

current_dir = Path(__file__).resolve().parent
with open(str(current_dir / "requirements.txt")) as reqs_file:
    reqs = reqs_file.read().split()

setup(
    name="anlp_project",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    version="v0.0.0",
    install_requires=reqs,
    entry_points={
        "console_scripts": [
            "anlp_project=anlp_project.main:anlp_project",
        ]
    },
    include_package_data=True,
)
