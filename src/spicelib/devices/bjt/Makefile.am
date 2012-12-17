## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libbjt.la
else
noinst_LTLIBRARIES = libbjt.la
endif

libbjt_la_SOURCES =	\
	bjt.c		\
	bjtacld.c	\
	bjtask.c	\
	bjtconv.c	\
	bjtdefs.h	\
	bjtdel.c	\
	bjtdest.c	\
	bjtdisto.c	\
	bjtdset.c	\
	bjtdset.h	\
	bjtext.h	\
	bjtgetic.c	\
	bjtinit.c	\
	bjtinit.h	\
	bjtitf.h	\
	bjtload.c	\
	bjtmask.c	\
	bjtmdel.c	\
	bjtmpar.c	\
	bjtnoise.c	\
	bjtparam.c	\
	bjtpzld.c	\
	bjtsacl.c	\
	bjtsetup.c	\
	bjtsload.c	\
	bjtsprt.c	\
	bjtsset.c	\
	bjtsupd.c	\
	bjttemp.c	\
	bjttrunc.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in
