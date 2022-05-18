from distutils.core import setup
from pathlib import Path

from setuptools import find_packages


README = Path('README.md')


def get_readme():
    if README.exists():
        return README.read_text()
    return ""


setup(
    name='Boruta',
    version='0.1',
    author='Ivan Reveguk', author_email='ivan.reveguk@gmail.com',
    description='Flexible sklearn-compatible tested python Boruta implementation',
    long_description=get_readme(), long_description_content_type='text/markdown',
    url='https://github.com/edikedik/Boruta', license='MIT',
    classifiers=[
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
        ],
    # package_dir={'': 'Boruta'},
    python_requires='>=3.8',
    packages=['Boruta'],
    package_dir={'Boruta': 'Boruta'},
    install_requires=[
        'scikit-learn>=1.0.2',
        'numpy',
        'pandas',
        'statsmodels',
        'tqdm',
        'scipy',
        'shap>=0.40.0'
    ]
)
