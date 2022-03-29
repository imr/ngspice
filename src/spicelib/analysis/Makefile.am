## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libckt.la

libckt_la_SOURCES = \
		acan.c		\
		acaskq.c	\
		acsetp.c	\
		analysis.c	\
		analysis.h	\
		cktacct.c	\
		cktacdum.c	\
		cktaskaq.c	\
		cktasknq.c	\
		cktbkdum.c	\
		cktclrbk.c	\
		cktdelt.c	\
		cktdest.c	\
		cktdisto.c	\
		cktdlti.c	\
		cktdltm.c	\
		cktdltn.c	\
		cktdojob.c	\
		cktdump.c	\
		cktncdump.c	\
		cktfbran.c	\
		cktfnda.c	\
		cktfndm.c	\
		cktfnode.c	\
		cktftask.c	\
		cktgrnd.c	\
		ckti2nod.c	\
		cktic.c		\
		cktlnkeq.c	\
		cktload.c	\
		cktmapn.c	\
		cktmask.c	\
		cktmcrt.c	\
		cktmkcur.c	\
		cktmknod.c	\
		cktmkvol.c	\
		cktmpar.c	\
		cktnames.c	\
		cktnewan.c	\
		cktneweq.c	\
		cktnewn.c	\
		cktnodn.c	\
		cktnoise.c	\
		cktntask.c	\
		cktnum2n.c	\
		cktop.c		\
		cktparam.c	\
		cktpmnam.c	\
		cktpname.c	\
		cktpzld.c	\
		cktpzset.c	\
		cktpzstr.c	\
		cktsens.c	\
		cktsetap.c	\
		cktsetbk.c	\
		cktsetnp.c	\
		cktsetup.c	\
		cktsgen.c	\
		cktsopt.c	\
		cktspnoise.c	\
		ckttemp.c	\
		cktterr.c	\
		ckttroub.c	\
		ckttrunc.c	\
		ckttyplk.c	\
		daskq.c		\
		dcoaskq.c	\
		dcop.c		\
		dcosetp.c	\
		dctaskq.c	\
		dctran.c	\
		dctrcurv.c	\
		dctsetp.c	\
		distoan.c	\
		dkerproc.c	\
		dloadfns.c	\
		dsetparm.c	\
		naskq.c		\
		nevalsrc.c	\
		ninteg.c	\
		noisean.c	\
		noisesp.c	\
		nsetparm.c	\
		optran.c	\
        com_optran.h    \
		pzan.c		\
		pzaskq.c	\
		pzsetp.c	\
		sensaskq.c	\
		senssetp.c	\
		tfanal.c	\
		tfaskq.c	\
		tfsetp.c	\
		tranaskq.c	\
		traninit.c	\
		transetp.c	\
		cluster.c


if PSS_WANTED
libckt_la_SOURCES += \
		dcpss.c		\
		pssaskq.c	\
		pssinit.c	\
		psssetp.c
endif

if SP_WANTED
libckt_la_SOURCES += \
		cktspdum.c		\
		span.c	\
		spaskq.c	\
		spsetp.c
endif


AM_CPPFLAGS = @AM_CPPFLAGS@  -I$(top_srcdir)/src/include -I$(top_srcdir)/src/spicelib/devices
AM_CFLAGS = $(STATIC)
MAINTAINERCLEANFILES = Makefile.in
