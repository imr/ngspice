## Process this file with automake to produce Makefile.in

SUBDIRS = bsim1			\
	bsim2			\
	bsim3			\
	bsim3soidd		\
	bsim3soifd		\
	bsim3soipd		\
	bsim4			\
	filters			\
	general			\
	hfet			\
	hisim			\
	jfet			\
	mes			\
	mesa			\
	mos6			\
	polezero		\
	resistance		\
	sensitivity		\
	transient		\
	transmission		\
	vbic
		

DIST_SUBDIRS = 	bsim1	\
	bsim2		\
	bsim3		\
	bsim3soidd	\
	bsim3soifd	\
	bsim3soipd	\
	bsim4		\
	filters		\
	general		\
	hfet		\
	hisim		\
	jfet		\
	mes		\
	mesa		\
	mos6		\
	polezero	\
	resistance	\
	sensitivity	\
	transient	\
	transmission	\
	vbic

TESTS_ENVIRONMENT = $(SHELL) $(top_srcdir)/tests/bin/check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	README bin

MAINTAINERCLEANFILES = Makefile.in
