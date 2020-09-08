from setuptools import setup

setup(
   name='pydmc',
   version='0.1',
   description='Simple DMC module in python',
   author='Jesse van Rhijn',
   author_email='jesse.v.rhijn@gmail.com',
   packages=['pydmc'],  #same as name
   install_requires=['numpy', 'scipy', 'matplotlib'], #external packages as dependencies
)
