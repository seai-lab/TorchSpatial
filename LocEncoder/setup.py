from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'space2vec Python package'
LONG_DESCRIPTION = 'First space2vec Python package'

setup(
    name="space2vec",
    version=VERSION,
    author="Jason Dsouza",
    author_email="<youremail@email.com>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],  # add any additional packages that
    # 需要和你的包一起安装，例如：'caer'

    keywords=['python', 'first package'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
