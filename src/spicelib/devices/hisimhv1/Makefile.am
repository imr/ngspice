## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libhisimhv1.la

libhisimhv1_la_SOURCES = hisimhv.h	\
			hsmhv.c		\
			hsmhvacld.c	\
			hsmhvask.c 	\
			hsmhvcvtest.c 	\
			hsmhvdef.h	\
			hsmhveval.c	\
			hsmhveval_qover.h	\
			hsmhvevalenv.h	\
			hsmhvext.h	\
			hsmhvgetic.c 	\
			hsmhvinit.c	\
			hsmhvinit.h	\
			hsmhvitf.h	\
			hsmhvld.c 	\
			hsmhvld_info_eval.h 	\
			hsmhvmask.c 	\
			hsmhvmpar.c 	\
			hsmhvnoi.c 	\
			hsmhvpar.c 	\
			hsmhvpzld.c 	\
			hsmhvset.c 	\
			hsmhvsoachk.c 	\
			hsmhvtemp.c 	\
			hsmhvtemp_eval.h \
			hsmhvtrunc.c

AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
