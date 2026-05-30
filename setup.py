from setuptools import setup, find_packages

setup(
    name="patternbloom",
    version="0.1.0",
    description="Reference implementation of the PatternBloom RL framework for agentic RAG.",
    packages=find_packages(include=["patternbloom", "patternbloom.*"]),
    python_requires=">=3.10",
)
