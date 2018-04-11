## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libhisimhv2.la

libhisimhv2_la_SOURCES = hisimhv2.h	\
			hsmhv2.c		\
			hsmhv2acld.c	\
			hsmhv2ask.c 	\
			hsmhv2cvtest.c 	\
			hsmhv2def.h	\
			hsmhv2eval.c	\
			hsmhv2eval_dep.h	\
			hsmhv2eval_dio.c	\
			hsmhv2eval_qover.h	\
			hsmhv2eval_rdrift.c	\
			hsmhv2evalenv.h	\
			hsmhv2ext.h	\
			hsmhv2getic.c 	\
			hsmhv2init.c	\
			hsmhv2init.h	\
			hsmhv2itf.h	\
			hsmhv2ld.c 	\
			hsmhv2ld_info_eval.h 	\
			hsmhv2mask.c 	\
			hsmhv2mpar.c 	\
			hsmhv2noi.c 	\
			hsmhv2par.c 	\
			hsmhv2pzld.c 	\
			hsmhv2set.c 	\
			hsmhv2soachk.c 	\
			hsmhv2temp.c 	\
			hsmhv2temp_eval.h \
			hsmhv2temp_eval_dio.h \
			hsmhv2temp_eval_rdri.h \
			hsmhv2trunc.c

AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
