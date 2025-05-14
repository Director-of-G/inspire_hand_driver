from setuptools import setup, find_packages

setup(name='inspire_sdkpy',
      version='1.0.0',
      author='Unitree',
      author_email='unitree@unitree.com',
      license="BSD-3-Clause",
      packages=find_packages(include=['inspire_sdkpy','inspire_sdkpy.*']),
      description='Inspire Hand sdk for python',
      python_requires='>=3.8',
      install_requires=[
            "cyclonedds==0.10.2",
            "numpy",
            "PyQt5",
            "pyqtgraph",
            "colorcet",
            "pymodbus==3.6.9",
            "pyserial"
      ],
      )
