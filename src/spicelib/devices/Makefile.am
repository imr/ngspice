## Process this file with automake to produce Makefile.in

SUBDIRS = \
	asrc	\
	bjt	\
	bjt2	\
	bsim1	\
	bsim2	\
	bsim3	\
	bsim3soi	\
	bsim4	\
	bsim3v0	\
	bsim3v1 \
	bsim3v1s	\
	bsim3v1a	\
	bsim3soi_pd	\
	bsim3soi_fd	\
	bsim3soi_dd	\
	cap	\
	cccs	\
	ccvs	\
	cpl	\
	csw	\
	dio	\
	@EKVDIR@	\
	ind	\
	isrc	\
	hfet1	\
	hfet2	\
	hisim	\
	jfet	\
	jfet2	\
	ltra	\
	mes	\
	mesa	\
	mos1	\
	mos2	\
	mos3	\
	mos6	\
	mos9 	\
	res	\
	soi3	\
	sw	\
	tra	\
	txl	\
	urc	\
	vbic	\
	vccs	\
	vcvs	\
	vsrc	\
	@NUMDEVDIR@

DIST_SUBDIRS = \
	asrc	\
	bjt	\
	bjt2	\
	bsim1	\
	bsim2	\
	bsim3	\
	bsim3soi	\
	bsim4	\
	bsim3v0	\
	bsim3v1 \
	bsim3v1s	\
	bsim3v1a	\
	bsim3soi_pd	\
	bsim3soi_fd	\
	bsim3soi_dd	\
	cap	\
	cccs	\
	ccvs	\
	cpl	\
	csw	\
	dio	\
	@EKVDIR@	\
	ind	\
	isrc	\
	hfet1	\
	hfet2	\
	hisim	\
	jfet	\
	jfet2	\
	ltra	\
	mes	\
	mesa	\
	mos1	\
	mos2	\
	mos3	\
	mos6	\
	mos9 	\
	res	\
	soi3	\
	sw	\
	tra	\
	txl	\
	urc	\
	vbic	\
	vccs	\
	vcvs	\
	vsrc	\
	nbjt	\
	nbjt2	\
	numd	\
	numd2	\
	numos


noinst_LIBRARIES = libdev.a

libdev_a_SOURCES = 	\
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
	limit.c

INCLUDES = -I$(top_srcdir)/src/include -I$(top_srcdir)/src/spicelib/devices

MAINTAINERCLEANFILES = Makefile.in
