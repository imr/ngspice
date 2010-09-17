## Process this file with automake to produce Makefile.in

SUBDIRS = \
	asrc	\
	bjt	\
	bjt2	\
	bsim1	\
	bsim2	\
	bsim3	\
	bsim3soi	\
	bsim4		\
	bsim4v2		\
	bsim4v3		\
	bsim4v4		\
	bsim4v5		\
	bsim3v0		\
	bsim3v1		\
	bsim3v1a	\
	bsim3v1s	\
	bsim3v32	\
	bsim3soi_pd	\
	bsim3soi_fd	\
	bsim3soi_dd	\
	cap	\
	cccs	\
	ccvs	\
	cpl	\
	csw	\
	dio	\
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
	mos9	\
	@NDEV_DIR@ \
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
	@VLADEVDIR@ \
	@NUMDEVDIR@

DIST_SUBDIRS = \
	@VLADEVDIR@ \
	asrc	\
	bjt	\
	bjt2	\
	bsim1	\
	bsim2	\
	bsim3	\
	bsim3soi	\
	bsim4		\
	bsim4v2		\
	bsim4v3		\
	bsim4v4		\
	bsim4v5		\
	bsim3v0		\
	bsim3v1		\
	bsim3v1a	\
	bsim3v1s	\
	bsim3v32	\
	bsim3soi_pd	\
	bsim3soi_fd	\
	bsim3soi_dd	\
	cap	\
	cccs	\
	ccvs	\
	cpl	\
	csw	\
	dio	\
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
	mos9	\
	ndev	\
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
	limit.c

EXTRA_DIST = @NOTVLADEVDIR@

AM_CPPFLAGS = -I$(top_srcdir)/src/include -I$(top_srcdir)/src/spicelib/devices

MAINTAINERCLEANFILES = Makefile.in
