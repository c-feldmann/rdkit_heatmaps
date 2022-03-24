from setuptools import setup

setup(
    name='rdkit_heatmap',
    version='0.1',
    author='Christian Feldmann',
    license="BSD",
    packages=['rdkit_heatmaps', ],
    author_email='cfeldmann@bit.uni-bonn.de',
    description='Toolkit for more custom heatmaps in RDKit',
    install_requires=['numpy', 'matplotlib', "pillow"]
)
