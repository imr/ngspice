## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libind.la
else
noinst_LTLIBRARIES = libind.la
endif

libind_la_SOURCES =	\
	ind.c		\
	indacld.c	\
	indask.c	\
	inddefs.h	\
	inddel.c	\
	inddest.c	\
	indext.h	\
	indinit.c	\
	indinit.h	\
	inditf.h	\
	indload.c	\
	indmask.c	\
	indmdel.c	\
	indmpar.c	\
	indparam.c	\
	indpzld.c	\
	indsacl.c	\
	indsetup.c	\
	indsload.c	\
	indsprt.c	\
	indsset.c	\
	indsupd.c	\
	indtemp.c	\
	indtrunc.c	\
	mutacld.c	\
	mutask.c	\
	mutdel.c	\
	mutdest.c	\
	mutmdel.c	\
	mutparam.c	\
	mutpzld.c	\
	mutsetup.c	\
	mutsprt.c	\
	mutsset.c	\
	muttemp.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in
