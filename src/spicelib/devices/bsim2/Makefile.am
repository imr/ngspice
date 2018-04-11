## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libbsim2.la

libbsim2_la_SOURCES =	\
	b2.c		\
	b2acld.c	\
	b2ask.c		\
	b2cvtest.c	\
	b2eval.c	\
	b2getic.c	\
	b2ld.c		\
	b2mask.c	\
	b2moscap.c	\
	b2mpar.c	\
	b2noi.c		\
	b2par.c		\
	b2pzld.c	\
	b2set.c		\
	b2temp.c	\
	b2trunc.c	\
	bsim2def.h	\
	bsim2ext.h	\
	bsim2init.c	\
	bsim2init.h	\
	bsim2itf.h



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
