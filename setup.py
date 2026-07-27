import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dibo_benchmark",
    version="0.1.0",
    author="Bernd Schuscha",
    author_email="bernd.schuscha@mcl.at",
    description=(
        "DIBOB: a benchmark for knowledge-integration strategies in "
        "multi-objective Bayesian optimization for materials design"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/<user>/PIBOB",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(include=["dibo_benchmark", "dibo_benchmark.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "botorch",
        "gpytorch",
        "pyro-ppl",
        "pandas",
        "numpy",
        "joblib",
        "matplotlib",
    ],
)
