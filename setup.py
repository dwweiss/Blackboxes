from setuptools import setup

setup(
    author='Dietmar Wilhelm Weiss',
    author_email='dietmar.wilhelm.weiss@gmail.com',
    description='Unified interface to popular neural network libraries',
    install_requires = [numpy, matplotlib, scipy, keras, torch, neurolab],
    keywords='neural network, black box, empirical model',
    license='GNU lesser license',
    long_description="\n\n".join([README, CHANGES]),
    name='blackboxes',
    packages=['blackboxes'],
    url='https://github.com/blackboxes/blackboxes',
    version = '03.24',
)

