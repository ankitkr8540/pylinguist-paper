from setuptools import setup, find_packages

setup(
    name='pylinguist',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'torch',
        'transformers',
        'deep-translator',
        'nltk',
        'sacrebleu',
        'openai',
        'datasets',
    ],
)
