## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libtxl.la
else
noinst_LTLIBRARIES = libtxl.la
endif

libtxl_la_SOURCES = \
	txl.c      \
	txlacct.c  \
	txlask.c   \
	txldefs.h  \
	txldest.c  \
	txlext.h   \
	txlfbr.c   \
	txlinit.h  \
	txlitf.h   \
	txlload.c  \
	txlmdel.c  \
	txlparam.c \
	txldel.c   \
	txlmask.c  \
	txlmpar.c  \
	txlsetup.c \
	txlinit.c

AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in
