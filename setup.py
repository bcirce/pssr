from setuptools import setup, find_packages


VERSION = '0.0.1' 
DESCRIPTION = 'My first Python package'
LONG_DESCRIPTION = 'My first Python package with a slightly longer description'


setup(
   name='pssr',
   version='0.1.0',
   author='Bárbara Circe',
   author_email='barbara.circe@eac.ufsm.br',
   packages=['pssr'],  
   #scripts=['bin/script1','bin/script2'],
   #url='http://pypi.python.org/pypi/PackageName/',
   license='LICENSE.txt',
   description='An awesome package that does something',
   long_description=open('README.md').read(),
   install_requires=[
       "SpeechRecognition",
       "arlpy",
       "pyaudio==0.2.11"#;python_version<='3.6.9'" #pyaudio doesn't have the wheels for python 3.7 or higher
   ],
)