import os.path as pt
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    dependencies = [l.strip(' \n') for l in f]
    # Pillow-simd==9.0.0.post1 ?

setup(
    name='eoe',
    version='0.1',
    classifiers=[
        'Development Status :: 4 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.10'
    ],
    keywords='deep-learning anomaly-detection one-class-classification outlier-exposure',
    package_dir={"": "src"},
    packages=find_packages(where="src", include=["eoe", "eoe.*"]),
    install_requires=dependencies,
)
