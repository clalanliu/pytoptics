from setuptools import setup, find_packages

setup(
    name='pytoptics',
    install_requires=['pyvista','PyVTK','numpy','scipy','KrakenOS','torch','matplotlib'],
    packages=find_packages()
)