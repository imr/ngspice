## Process this file with automake to produce Makefile.in

EXTRA_DIST = inpptree-parser.y
## For Windows with Visual Studio
EXTRA_DIST += inpptree-parser.c inpptree-parser.h

noinst_LTLIBRARIES = libinp.la

libinp_la_SOURCES = \
		ifeval.c	\
		ifnewuid.c	\
		inp2b.c		\
		inp2c.c		\
		inp2d.c		\
		inp2dot.c	\
		inp2e.c		\
		inp2f.c		\
		inp2g.c		\
		inp2h.c		\
		inp2i.c		\
		inp2j.c		\
		inp2k.c		\
		inp2l.c		\
		inp2m.c		\
		inp2n.c		\
		inp2o.c		\
		inp2p.c		\
		inp2q.c		\
		inp2r.c		\
		inp2s.c		\
		inp2t.c		\
		inp2u.c		\
		inp2v.c		\
		inp2w.c		\
		inp2y.c		\
		inp2z.c		\
		inpaname.c	\
		inpapnam.c	\
		inpcfix.c	\
		inpdomod.c	\
		inpdoopt.c	\
		inpdpar.c	\
		inperrc.c	\
		inperror.c	\
		inpeval.c	\
		inpfindl.c	\
		inpfindv.c	\
		inpgmod.c	\
		inpgstr.c	\
		inpgtitl.c	\
		inpgtok.c	\
		inpgval.c	\
		inpkmods.c	\
		inplist.c	\
		inplkmod.c	\
		inpmkmod.c	\
		inpmktmp.c	\
		inppas1.c	\
		inppas1.h	\
		inppas2.c	\
		inppas2.h	\
		inppas3.c	\
		inppas3.h	\
		inppname.c	\
		inpptree.c	\
		inpsymt.c	\
		inptyplk.c	\
		ptfuncs.c	\
		sperror.c	\
		inp.h

$(srcdir)/inpptree.c : inpptree-parser.c
inpptree-parser.c inpptree-parser.h : inpptree-parser.y
	bison $(srcdir)/inpptree-parser.y

AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include -I$(top_srcdir)/src/frontend
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
