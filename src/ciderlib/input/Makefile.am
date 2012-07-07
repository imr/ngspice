## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libciderinput.la

libciderinput_la_SOURCES =	\
		bdryset.c	\
		boundary.c	\
		cards.c		\
		contact.c	\
		contset.c	\
		domain.c	\
		domnset.c	\
		doping.c	\
		dopset.c	\
		elctset.c	\
		electrod.c	\
		material.c	\
		matlset.c	\
		mesh.c		\
		meshset.c	\
		method.c	\
		mobility.c	\
		mobset.c	\
		models.c	\
		modlset.c	\
		optionsc.c	\
		outpset.c	\
		output.c


EXTRA_DIST = readme
AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)
MAINTAINERCLEANFILES = Makefile.in
