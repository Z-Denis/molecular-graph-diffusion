"""Setup script for the mgd (molecular-graph-diffusion) package."""

from pathlib import Path

from setuptools import find_packages, setup


def read_requirements() -> list[str]:
    reqs_path = Path("requirements.txt")
    if not reqs_path.exists():
        return []
    return [
        line.strip()
        for line in reqs_path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


setup(
    name="mgd",
    version="0.0.1",
    description="Molecular graph diffusion models in Flax.",
    author="",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=read_requirements(),
    include_package_data=True,
)
