# Process this file with automake
CFLAGS = -g -O2 -Wall
CC = gcc 
COMPILE = $(CC) $(DEFS) $(INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS) $(AM_CFLAGS) $(CFLAGS)

EXTRA_DIST = README

SUBDIRS = mif cm enh evt ipc idn

INCLUDES = -I$(top_srcdir)/src/include -I$(top_srcdir)

MAINTAINERCLEANFILES = Makefile.in

all: xspice.o
xspice.o:
	$(COMPILE) -c xspice.c
