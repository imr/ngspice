## Process this file with automake to produce Makefile.in

SUBDIRS = TransImpedanceAmp	\
	bsim1			\
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
	proc2mod		\
	resistance		\
	transmission		\
	vbic			\
	@CIDERTESTS@ 

DIST_SUBDIRS = 	bsim1	\
	bsim1		\
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
	proc2mod	\
	resistance	\
	TransImpedanceAmp	\
	transmission	\
	vbic		\
	cider

TESTS = \
	diffpair.cir		\
	fourbitadder.cir

TESTS_ENVIRONMENT = $(SHELL) $(srcdir)/check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	README			\
	check.sh		\
	maketest.sh		\
	$(TESTS)		\
	$(TESTS:.cir=.out)

MAINTAINERCLEANFILES = Makefile.in
