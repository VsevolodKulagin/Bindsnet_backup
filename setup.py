from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

version = "0.2.5"

setup(
    name="bindsnet",
    version=version,
    description="Spiking neural networks for ML in Python",
    license="AGPL-3.0",
    long_description=long_description,
    long_description_content_type="text/markdown",  # This is important!
    url="http://github.com/Hananel-Hazan/bindsnet",
    author="Daniel Saunders, Hananel Hazan, Darpan Sanghavi, Hassaan Khan",
    author_email="danjsaund@gmail.com",
    packages=find_packages(),
    zip_safe=False,
    download_url="https://github.com/Hananel-Hazan/bindsnet/archive/%s.tar.gz"
    % version,
    install_requires=[
        "numpy==1.20.1",
        "torch==1.7.1+cpu",
        "torchvision==0.8.2+cpu",
        "tensorboardX==2.6.1",
        "tqdm==4.59.0",
        "matplotlib==3.3.4",
        "gym==0.18.3",
        "scikit_image==0.18.1",
        "scikit_learn==0.24.1",
        "opencv-python==4.5.2.54",
        "pytest==6.2.3",
        "scipy==1.6.2",
        "cython==0.29.23",
        "pandas==1.2.4",
    ],
)
