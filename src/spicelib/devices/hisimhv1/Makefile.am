## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libhisimhv1.la
else
noinst_LTLIBRARIES = libhisimhv1.la
endif

libhisimhv1_la_SOURCES = hisimhv.h	\
			hsmhv.c		\
			hsmhvacld.c	\
			hsmhvask.c 	\
			hsmhvcvtest.c 	\
			hsmhvdef.h	\
			hsmhvdel.c 	\
			hsmhvdest.c 	\
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
			hsmhvmdel.c 	\
			hsmhvmpar.c 	\
			hsmhvnoi.c 	\
			hsmhvpar.c 	\
			hsmhvpzld.c 	\
			hsmhvset.c 	\
			hsmhvtemp.c 	\
			hsmhvtemp_eval.h \
			hsmhvtrunc.c

AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in
