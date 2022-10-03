import setuptools

with open("README.md") as fh:
    long_description = fh.read()

install_requires = [
    "numpy >= 1.19.4",
    "scipy >= 1.5.4",
    "scikit-learn",
    "fbpca >= 1.0",
]

# extras_require["complete"] = sorted({v for req in extras_require.values() for v in req})

packages = [
    "unipy",
]

tests = [p + ".tests" for p in packages]

setuptools.setup(
    name="unipy",
    version="0.0.1",
    author="R.A.R. MOENS",
    author_email="r.a.r.moens@tudelft.nl",
    description="A wrapper package unifying numpy, scipy and cupy.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vandeplas/unipy",
    packages=packages + tests,
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
