## Process this file with automake to produce Makefile.in

SUBDIRS =		\
	bsim3		\
	bsim4		\
	bsimsoi		\
	hisim		\
	hisimhv

if XSPICE_WANTED
SUBDIRS += xspice
endif


DIST_SUBDIRS =		\
	bsim1		\
	bsim2		\
	bsim3		\
	bsim3soidd	\
	bsim3soifd	\
	bsim3soipd	\
	bsim4		\
	bsimsoi		\
	filters		\
	general		\
	hfet		\
	hisim		\
	hisimhv		\
	jfet		\
	mes		\
	mesa		\
	mos6		\
	polezero	\
	resistance	\
	sensitivity	\
	transient	\
	transmission	\
	vbic		\
	xspice


TESTS_ENVIRONMENT = \
	$(SHELL) $(top_srcdir)/tests/bin/check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	README bin .gitignore

MAINTAINERCLEANFILES = Makefile.in
