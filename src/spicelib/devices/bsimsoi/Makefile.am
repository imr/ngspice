## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libbsim4soi.la
else
noinst_LTLIBRARIES = libbsim4soi.la
endif

libbsim4soi_la_SOURCES = \
        b4soi.c                \
        b4soiacld.c        \
        b4soiask.c        \
        b4soicheck.c        \
        b4soicvtest.c        \
        b4soidel.c        \
        b4soidest.c        \
        b4soigetic.c        \
        b4soild.c        \
        b4soimask.c        \
        b4soimdel.c        \
        b4soimpar.c        \
        b4soinoi.c        \
        b4soipar.c        \
        b4soipzld.c        \
        b4soiset.c        \
        b4soitemp.c        \
        b4soitrunc.c        \
        b4soidef.h        \
        b4soiext.h        \
        b4soiinit.c        \
        b4soiinit.h        \
        b4soiitf.h



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in

EXTRA_DIST = BsimTerms_use
