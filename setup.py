from setuptools import setup, find_packages

setup(
    name='outlier_explainer',
    version='2',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'umap-learn',
        'lime',
    ],
    author='Kai',
    description='Outlier detection and explanation tool',
    url='https://github.com/Alyxx-The-Sniper/outlier_explainer',
)
