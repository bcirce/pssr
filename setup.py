from setuptools import setup, find_packages


VERSION = '0.0.1' 
DESCRIPTION = 'My first Python package'
LONG_DESCRIPTION = 'My first Python package with a slightly longer description'


setup(
   name='pssr',
   version=VERSION,
   author='Bárbara Circe',
   author_email='barbara.circe@eac.ufsm.br',
   packages=['pssr'],  
   #scripts=['bin/script1','bin/script2'],
   #url='http://pypi.python.org/pypi/PackageName/',
   license='LICENSE.txt',
   description=DESCRIPTION,
   long_description=open('README.md').read(),
   install_requires=[
       "SpeechRecognition",
       "arlpy",
       "PyAudio"#;python_version<='3.6.9'" #pyaudio doesn't have the wheels for python 3.7 or higher
   ],
)