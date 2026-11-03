## Process this file with automake to produce Makefile.in
## Enhancement-242: native n-port rational-model device

noinst_LTLIBRARIES = libnport.la

libnport_la_SOURCES =	\
	nport.c		\
	nportacload.c	\
	nportask.c	\
	nportbindCSC.c	\
	nportdefs.h	\
	nportdel.c	\
	nportext.h	\
	nportinit.c	\
	nportinit.h	\
	nportitf.h	\
	nportload.c	\
	nportmask.c	\
	nportmpar.c	\
	nportparam.c	\
	nportread.c	\
	nportsetup.c	\
	nporttemp.c

AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
