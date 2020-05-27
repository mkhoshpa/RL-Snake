import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="RL-Snake-mkhoshpa", # Replace with your own username
    version="0.0.7",
    author="Mehrgan Khoshpasand Foumani",
    author_email="me.khoshpasand@gmail.com",
    description="Snake game environment for reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mkhoshpa/RL-Snake",
    install_requires = ['matplotlib','torch','numpy'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)