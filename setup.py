from setuptools import setup, find_packages

setup(
    name="APO2MolFlow",
    version="0.0.0",
    packages=find_packages(),
    # packages=[
    #     'openfold',
    #     'pepflow',
    #     'data',
    # ],
    package_dir={
        'APO2MolFlow': './model',
        'openfold': './openfold'
    },
)
