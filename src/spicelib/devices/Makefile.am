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
	vccs	\
	vcvs	\
	vsrc

##  This line move to here 'cause it was causing automake to choke 
##  when it was in the list of subdirs
##	bjt2	\


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
	cktinit.c       \
	names.c

INCLUDES = -I$(top_srcdir)/src/include -I$(top_srcdir)/src/spicelib/devices

MAINTAINERCLEANFILES = Makefile.in
