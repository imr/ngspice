#!/bin/sh

#  ngconfig.sh
#     configure options for ngspice with numparam add-on
#     run this in  ngspice's top-level directory 

# specify your Numparam directory
HACK=/home/post/spice3f5/hack

# over-write the original subckt.c
cp -biv $HACK/ngsubckt.c  src/frontend/subckt.c  

# my box needs CFLAGS on 1st run, else 'terminal.c' wont find 'termcap.h' ?

CFLAGS=-I/usr/include/ncurses \
LIBS=$HACK/libnupa.a \
./configure --without-x --prefix=/usr/local/ngsp

####  end of sample script  ####

