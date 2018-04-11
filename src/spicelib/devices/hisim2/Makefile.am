## Process this file with automake to produce Makefile.in

EXTRA_DIST =

noinst_LTLIBRARIES = libhisim2.la

libhisim2_la_SOURCES =	hisim2.h	\
			hsm2.c		\
			hsm2acld.c	\
			hsm2ask.c	\
			hsm2cvtest.c	\
			hsm2def.h	\
			hsm2eval.c	\
			hsm2evalenv.h	\
			hsm2eval_dep.h	\
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
			hsm2soachk.c	\
			hsm2temp.c	\
			hsm2trunc.c


AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
