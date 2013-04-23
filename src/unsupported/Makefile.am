## Process this file with automake to produce Makefile.in

EXTRA_DIST = readme spiced.c

noinst_LTLIBRARIES = libunsupported.la

libunsupported_la_SOURCES = \
	cktsenac.c		\
	cktsenup.c		\
	cktsncom.c		\
	cktsndct.c		\
	cktsnset.c		\
	cktsnld.c		\
	cktsnprt.c		\
	sen2dest.c		\
	sen2setp.c		\
	snaskq.c		\
	snstart.c


AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)
MAINTAINERCLEANFILES = Makefile.in
