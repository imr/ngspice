## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libcmaths.la

libcmaths_la_SOURCES = \
		cmath.h		\
		cmath1.c	\
		cmath1.h	\
		cmath2.c	\
		cmath2.h	\
		cmath3.c	\
		cmath3.h	\
		cmath4.c	\
		cmath4.h

if !TCL_MODULE
if !WINDOWS

noinst_PROGRAMS = test_cx_mag test_cx_j test_cx_ph test_cx_cph

test_cx_ph_SOURCES = \
		test_cx_ph.c

test_cx_ph_LDADD = \
		libcmaths.la \
		../../misc/libmisc.la \
		$(TCL_LIB_SPEC)

test_cx_cph_SOURCES = \
               test_cx_cph.c

test_cx_cph_LDADD = \
                libcmaths.la \
                ../../misc/libmisc.la \
                $(TCL_LIB_SPEC)

test_cx_mag_SOURCES = \
		test_cx_mag.c

test_cx_mag_LDADD = \
		libcmaths.la \
		../../misc/libmisc.la \
		$(TCL_LIB_SPEC)

test_cx_j_SOURCES = \
		test_cx_j.c

test_cx_j_LDADD = \
		libcmaths.la \
		../../misc/libmisc.la \
		$(TCL_LIB_SPEC)

TESTS = test_cx_mag test_cx_j test_cx_ph test_cx_cph

endif !WINDOWS
endif !TCL_MODULE

AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include -I$(top_srcdir)/src/maths/poly

MAINTAINERCLEANFILES = Makefile.in
