This is the place to deploy the 64 bit version of the
fftw3 library under VC++ which can be found here:

 http://www.fftw.org/install/windows.html

More precisely, these are the files (e.g. version 3.3)
  fftw-3.3.3-dll64.zip
which you will have to install here.

At least you need the three files
fftw3.h
libfftw3-3.dll
libfftw3-3.def

If you need to distribute ngspice.exe to another directory,
copy libfftw3-3.dll into the same directory or to a place
which is in your PATH environment.
