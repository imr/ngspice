## Process this file with automake to produce Makefile.in

# Copyright (C) 1999 by John Heidemann <johnh@isi.edu>

## shallow
AUTOMAKE_OPTIONS = 1.0 foreign

bin_PROGRAMS = xgraph
xgraph_c_sources = \
		xgraph.c xgX.c hard_devices.c dialog.c hpgl.c \
		  ps.c idraw.c xtb.c st.c params.c alloc.c draw.c \
		  init.c read.c tgif.c derivative.c \
		copyright.h \
		general.h \
		hard_devices.h \
		params.h \
		plotter.h \
		st.h \
		xgout.h \
		xgraph.h \
		xtb.h
xgraph_SOURCES = $(xgraph_c_sources)

man_MANS = xgraph.man

EXTRA_DIST=\
	$(man_MANS) \
	README.ANNOUNCE \
	README.GENERAL \
	README.HARDCOPY \
	README.INSTALL \
	README.NEWFEATURES \
	bitmaps/dot.11 \
	bitmaps/gray \
	bitmaps/mark1.11 \
	bitmaps/mark2.11 \
	bitmaps/mark3.11 \
	bitmaps/mark4.11 \
	bitmaps/mark5.11 \
	bitmaps/mark6.11 \
	bitmaps/mark7.11 \
	bitmaps/mark8.11 \
	examples/Animation.xg \
	examples/bar.xg \
	examples/surface.xg \
	examples/surface2.xg \
	examples/surface3.xg \
	examples/xgtest.xg


# C_WARNINGS = -Wall
C_WARNINGS = 
CFLAGS=$(ADDITIONAL_INCLUDES)  $(C_WARNINGS) -g  $(X_CFLAGS) 

xgraph_LDADD= $(ADDITIONAL_LIBS) $(X_LIBS) $(X_PRE_LIBS) $(X_EXTRA_LIBS) -lX11 -lm


