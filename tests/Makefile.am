## Process this file with automake to produce Makefile.in
## regression first, then xspice, then the model qua tests

SUBDIRS = regression

if XSPICE_WANTED
SUBDIRS += xspice
endif

SUBDIRS +=		\
	bsim3		\
	bsim4		\
	bsimsoi		\
	hisim		\
	hisimhv1	\
	hisimhv2

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
	hisimhv1	\
	hisimhv2	\
	jfet		\
	mes		\
	mesa		\
	mos6		\
	polezero	\
	regression	\
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
