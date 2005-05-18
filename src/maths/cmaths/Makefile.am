## Process this file with automake to produce Makefile.in

noinst_LIBRARIES = libcmaths.a

libcmaths_a_SOURCES = \
		cmath.h		\
		cmath1.c	\
		cmath1.h	\
		cmath2.c	\
		cmath2.h	\
		cmath3.c	\
		cmath3.h	\
		cmath4.c	\
		cmath4.h

## Test programs fail to build on Windows
if !WINDOWS 

noinst_PROGRAMS = test_cx_mag test_cx_j test_cx_ph
 
test_cx_ph_SOURCES = \
		test_cx_ph.c

test_cx_ph_LDADD = \
		libcmaths.a \
		../../misc/libmisc.a

test_cx_mag_SOURCES = \
		test_cx_mag.c

test_cx_mag_LDADD = \
		libcmaths.a \
		../../misc/libmisc.a

test_cx_j_SOURCES = \
		test_cx_j.c

test_cx_j_LDADD = \
		libcmaths.a \
		../../misc/libmisc.a

TESTS = test_cx_mag test_cx_j test_cx_ph

endif !WINDOWS


INCLUDES = -I$(top_srcdir)/src/include -I$(top_srcdir)/src/maths/poly

MAINTAINERCLEANFILES = Makefile.in
