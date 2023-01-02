from setuptools import find_packages, setup

setup(
    name="jax_xenet",
    version="0.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "jax",
        "jaxlib",
        "flax",
    ],
)
