from setuptools import find_packages, setup

setup(
    name="legalqa",
    version="1.0.0",
    description="OOP Vietnamese Legal QA pipeline: retriever/reranker fine-tuning, dense chunk cache, tuned RAG QA.",
    packages=find_packages(include=["legalqa", "legalqa.*"]),
    python_requires=">=3.10",
)