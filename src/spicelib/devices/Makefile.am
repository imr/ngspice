## Process this file with automake to produce Makefile.in

SUBDIRS = \
	bjt	\
	cap	\
	dio	\
	ind	\
	isrc	\
	mos1	\
	res	\
	vsrc	\
	@VLADEVDIR@

DIST_SUBDIRS = \
	bjt	\
	cap	\
	dio	\
	ind	\
	isrc	\
	mos1	\
	res	\
	vsrc



noinst_LTLIBRARIES = libdev.la

libdev_la_SOURCES =	\
	dev.c		\
	dev.h		\
	devsup.c	\
	cktaccept.c	\
	cktaccept.h	\
	cktask.c	\
	cktbindnode.c	\
	cktcrte.c	\
	cktfinddev.c	\
	cktinit.c	\
	cktsoachk.c	\
	limit.c

EXTRA_DIST = adms/admst

AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include -I$(top_srcdir)/src/spicelib/devices
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
