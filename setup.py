import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="NanoGrad",
    version="0.1.3",
    author="Rivera.ai/Fredy",
    author_email="riveraaai200678@gmail.com",
    description="A small Autograd project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Rivera-ai/NanoGrad",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
