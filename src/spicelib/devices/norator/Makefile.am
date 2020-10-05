## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libnora.la

libnora_la_SOURCES =	\
	nora.c		\
	noraask.c	\
	noradefs.h	\
	noraext.h	\
	norafbr.c	\
	norainit.c	\
	norainit.h	\
	noraitf.h	\
	noraload.c	\
	norapar.c	\
	norapzld.c	\
	noraset.c	
	
#	norasacl.c	\
#	norasld.c	\
#	norasprt.c	\
#	norasset.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)
MAINTAINERCLEANFILES = Makefile.in
