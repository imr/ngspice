#!/bin/sh
# MacOS build script for NGSPICE

# Considering a MacOS 10.12 system, there are some prerequisites to be satisfied:
# 1) Install an X11 system of your choice. XQuartz from SourceForge is fine: https://www.xquartz.org
# 2) Install automake, autoconf, libtool and an updated version of bison by using the method you prefer.
#    From sources, from 'brew' or from 'MacPorts' are the known ones and I prefer using MacPorts,
#    available at this address: https://www.macports.org .
#    You can install from a tarball or you can use the installer, which will also configure the PATH.
#
# Said that, the script is quite linear and simple.


# Build

./autogen.sh

./configure \
  --enable-xspice \
  --enable-cider \
  --enable-pss \
  --disable-debug \
  --prefix=/Applications/ngspice

make
make DESTDIR="$(pwd)/root-tree" install

# Package
pkgbuild \
  --root "$(pwd)/root-tree" \
  --identifier ngspice.pkg \
  --install-location / \
  ngspice.pkg
