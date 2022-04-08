## Process this file with automake to produce Makefile.in
#
# JW 3/9/01 - had a go and makeing an   autoconf script.

noinst_LTLIBRARIES = libevtxsp.la

libevtxsp_la_SOURCES = \
	evtaccept.c  \
	evtcall_hybrids.c  \
	evtdump.c  \
	evtiter.c  \
	evtnext_time.c \
	evtop.c     \
	evtprint.c  \
	evtsetup.c  \
	evtdest.c  \
	evtbackup.c  \
	evtdeque.c  \
	evtinit.c  \
	evtload.c  \
	evtnode_copy.c  \
	evtplot.c  \
	evtqueue.c  \
	evttermi.c  \
	evtshared.c \
	evtcheck_nodes.c

AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
