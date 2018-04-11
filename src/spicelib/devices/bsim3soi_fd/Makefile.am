## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libbsim3soifd.la

libbsim3soifd_la_SOURCES = \
	b3soifd.c	\
	b3soifdacld.c	\
	b3soifdask.c	\
	b3soifdcheck.c	\
	b3soifdcvtest.c	\
	b3soifdgetic.c	\
	b3soifdld.c	\
	b3soifdmask.c	\
	b3soifdmpar.c	\
	b3soifdnoi.c	\
	b3soifdpar.c	\
	b3soifdpzld.c	\
	b3soifdset.c	\
	b3soifdtemp.c	\
	b3soifdtrunc.c	\
	b3soifddef.h	\
	b3soifdext.h	\
	b3soifdinit.c	\
	b3soifdinit.h	\
	b3soifditf.h



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in

EXTRA_DIST = BsimTerms_use
