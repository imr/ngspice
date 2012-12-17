## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libhisim2.la
else
noinst_LTLIBRARIES = libhisim2.la
endif

libhisim2_la_SOURCES =	hisim2.h	\
			hsm2.c		\
			hsm2acld.c	\
			hsm2ask.c	\
			hsm2cvtest.c	\
			hsm2def.h	\
			hsm2del.c	\
			hsm2dest.c	\
			hsm2eval.c	\
			hsm2evalenv.h	\
			hsm2ext.h	\
			hsm2getic.c	\
			hsm2init.c	\
			hsm2init.h	\
			hsm2itf.h	\
			hsm2ld.c	\
			hsm2mask.c	\
			hsm2mdel.c	\
			hsm2mpar.c	\
			hsm2noi.c	\
			hsm2par.c	\
			hsm2pzld.c	\
			hsm2set.c	\
			hsm2temp.c	\
			hsm2trunc.c


AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in
