import setuptools

setuptools.setup(
    name="AFOS",
    version='1.0.0',
    author="Abdur Rahman",
    author_email="ar2806@msstate.edu",
    description="Code for Activation Function Optimization Scheme for Image Classification",
    url="https://github.com/abdurrahman1828/AFOS",
    keywords=["Activation function", "Evolutionary approach", "Ex-ponential Error Linear Unit (EELU)", "Genetic Algorithm"],
    packages=setuptools.find_packages(exclude=('tests',)),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn'
        'tensorflow',
    ],
    requires_python='>=3.8',
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: MIT License",
        "Operating System :: OS Independent",
    ],
)