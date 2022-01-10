from setuptools import setup, find_packages

setup(
    name             = 'maist-model-core',
    version          = '1.5',
    description      = 'main core of machine learning module by Yonsei MAIST',
    author           = 'Chanwoo Gwon',
    author_email     = 'arknell@yonsei.ac.kr',
    url              = 'https://github.com/Yonsei-Maist/ABR-image-processor.git',
    install_requires = [],
    packages         = find_packages(exclude = ['docs', 'tests*']),
    keywords         = ['ai', 'tensorflow'],
    python_requires  = '>=3',
    zip_safe=False,
    classifiers      = [
        'Programming Language :: Python :: 3.7'
    ]
)