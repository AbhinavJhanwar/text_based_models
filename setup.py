from setuptools import setup, find_packages

base_packages = [
        "scikit-learn>=0.22.2",
        "numpy>=1.18.5",
        "pandas>=1.3.5",
        "groq==0.10.0",
        "nltk>=3.6.7",
        "plotly>=5.5.0",
        "tqdm>=4.62.3",
        "spacy==3.7.6",
        "pyarrow>=6.0.1",
        "en_core_web_sm@https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl",
        "yake",
        "hdbscan",
        "umap-learn",
        "keybert"
]

setup(
    name='text_analysis_models',
    packages=find_packages(exclude=["notebooks", "docs"]),
    version='0.2.0',    
    description='A text semantics api for text keyword extraction and insights, sentiment analysis and topic modeling',
    url='https://github.com/AbhinavJhanwar/text_analysis_models.git',
    author='Abhinav Jhanwar',
    author_email='abhij.1994@gmail.com',
    install_requires=base_packages,
    classifiers=[
        "Programming Language :: Python:: 3.10",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Operating System :: MacOS"
    ],
)