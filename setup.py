from setuptools import setup, find_packages

setup(
    name="Intelligent Book",
    version="0.1.0",
    description="Intelligent Book API.",
    long_description=open("README.md").read(),
    url="https://github.com/yourusername/your-package-name",
    author="Lalithkumar",
    author_email="lalithcse@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="intelligent_book",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
    ],
    extras_require={
        "dev": ["check-manifest"],
        "test": ["coverage"],
    },
    entry_points={
        "console_scripts": [
            "myprogram=myproject.myprogram:main",
        ],
    },
)
