from setuptools import setup, find_packages

setup(
    name="lx_anonymizer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "spacy>=3.0.0",
        "pdfplumber",
        "faker",
        "gender-guesser",
    ],
    python_requires=">=3.8",
    author="Max Hild",
    author_email="Maxhild10@gmail.com",
    description="A tool for anonymizing medical reports and images",
)
