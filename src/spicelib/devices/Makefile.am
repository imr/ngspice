## Process this file with automake to produce Makefile.in

SUBDIRS = \
	asrc	\
	bjt	\
	bsim1	\
	bsim2	\
	bsim3	\
	bsim4	\
	bsim3v1 \
	bsim3v2 \
	cap	\
	cccs	\
	ccvs	\
	csw	\
	dio	\
	ind	\
	isrc	\
	jfet	\
	jfet2	\
	ltra	\
	mes	\
	mos1	\
	mos2	\
	mos3	\
	mos6	\
	res	\
	sw	\
	tra	\
	urc	\
	vccs	\
	vcvs	\
	vsrc	\
	parser

lib_LIBRARIES = libdev.a

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
	cktinit.c

EXTRA_DIST = README

INCLUDES = -I$(top_srcdir)/src/include

MAINTAINERCLEANFILES = Makefile.in
