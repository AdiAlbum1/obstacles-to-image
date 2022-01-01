from distutils.core import setup
from os import system
import platform
import requests

print("installing bindings")

package_data = {}

if platform.system() == 'Windows':
      url = 'https://www.cs.tau.ac.il/~cgl/python_rmp/libgmp-10.dll'
      r = requests.get(url, allow_redirects=True)
      with open('libgmp-10.dll', 'wb') as f:
            f.write(r.content)

      url = 'https://www.cs.tau.ac.il/~cgl/python_rmp/libmpfr-4.dll'
      r = requests.get(url, allow_redirects=True)
      with open('libmpfr-4.dll', 'wb') as f:
            f.write(r.content)

      url = 'https://www.cs.tau.ac.il/~cgl/python_rmp/CGALPY.pyd'
      r = requests.get(url, allow_redirects=True)
      with open('CGALPY.pyd', 'wb') as f:
            f.write(r.content)

      package_data = {'': ['CGALPY.pyd', 'libgmp-10.dll', 'libmpfr-4.dll']}

elif platform.system() == 'Linux':

      url = 'https://www.cs.tau.ac.il/~cgl/python_rmp/libgmp.so.10'
      r = requests.get(url, allow_redirects=True)
      with open('libgmp.so.10', 'wb') as f:
            f.write(r.content)

      url = 'https://www.cs.tau.ac.il/~cgl/python_rmp/libmpfr.so.6'
      r = requests.get(url, allow_redirects=True)
      with open('libmpfr.so.6', 'wb') as f:
            f.write(r.content)

      url = 'https://www.cs.tau.ac.il/~cgl/python_rmp/libboost_python39.so.1.77.0'
      r = requests.get(url, allow_redirects=True)
      with open('libboost_python39.so.1.77.0', 'wb') as f:
            f.write(r.content)

      url = 'https://www.cs.tau.ac.il/~cgl/python_rmp/CGALPY.so'
      r = requests.get(url, allow_redirects=True)
      with open('CGALPY.so', 'wb') as f:
            f.write(r.content)

      package_data = {'': ['CGALPY.so', 'libgmp.so.10', 'libmpfr.so.6', 'libboost_python39.so.1.77.0']}

setup(name='CGALPY',
      version='1.0',
      py_modules=['CGALPY'],
      packages=[''],
      package_data=package_data
)