from setuptools import setup, find_packages

setup(
    name             = 'maist-model-core',
    version          = '1.0',
    description      = 'main core of machine learning module by Yonsei MAIST',
    author           = 'Chanwoo Gwon',
    author_email     = 'arknell@yonsei.ac.kr',
    url              = 'https://github.com/Yonsei-Maist/ABR-image-processor.git',
    install_requires = [
        "tensorflow 2.4"
    ],
    packages         = find_packages(exclude = ['docs', 'tests*']),
    keywords         = ['ai', 'tensorflow'],
    python_requires  = '>=3',
    zip_safe=False,
    classifiers      = [
        'Programming Language :: Python :: 3.7'
    ]
)