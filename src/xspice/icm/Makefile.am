## Process this file with automake to produce Makefile.in
#
# Hacked on 5.23.2003 by SDB


SUBDIRS = icm_spice2poly

INCLUDES = -I$(top_srcdir)/src/include

MAINTAINERCLEANFILES = Makefile.in

EXTRA_DIST = README modpath.lst udnpath.lst spice2poly.cm dlmain.c


# Edit these to point to your Opus installation
CMPPDIR = $(top_srcdir)/src/xspice/cmpp
OPUS_INCLUDE_DIR = $(top_srcdir)/src/include


##------ automake definitions below. ------
##

## Yes, listing all this stuff is ugly, but I wasn't clever enough to
## find a better way to make it work.

SYSTEM_LIBS = -lm -lncurses -L/usr/X11R6/lib -lX11 -lXt -lXext -lXmu -lXaw  -lSM -lICE

LOCAL_LIBS = -lcmxsp,-levtxsp,-lidnxsp,-lipcxsp,-lenhxsp,-lmifxsp,-ldev,-lckt,-linp,-lparser,-lhlp,-linp,-lderiv,-lcmaths,-lpoly,-lni,-lsparse,-lmisc,-lfte,-lplotting

DEVICE_LIBS = -lasrc,-lbjt,-lbsim1,-lbsim2,-lbsim3,-lbsim3v1,-lbsim3v2,-lbsim4,-lcap,-lbsim3soipd,-lbsim3soifd,-lbsim3soidd,-lcccs,-lccvs,-lccvs,-lcpl,-lcsw,-ldio,-lind,-lisrc,-lhfet,-lhfet2,-ljfet,-ljfet2,-lltra,-lmes,-lmesa,-lmos1,-lmos2,-lmos3,-lmos6,-lmos9,-lres,-lsoi3,-lsw,-ltxl,-ltra,-lurc,-lvccs,-lvcvs,-lvsrc



LOCAL_LIB_DIRS = \
	-L$(top_srcdir)/src/xspice/cm/ 			\
	-L$(top_srcdir)/src/xspice/enh/ 		\
	-L$(top_srcdir)/src/xspice/evt/ 		\
	-L$(top_srcdir)/src/xspice/idn/ 		\
	-L$(top_srcdir)/src/xspice/ipc/ 		\
	-L$(top_srcdir)/src/xspice/mif/ 		\
	-L$(top_srcdir)/src/spicelib/devices/ 		\
	-L$(top_srcdir)/src/spicelib/analysis/ 		\
	-L$(top_srcdir)/src/spicelib/parser/ 		\
					   		\
	-L$(top_srcdir)/src/frontend/parser/ 		\
	-L$(top_srcdir)/src/frontend/help/ 		\
	-L$(top_srcdir)/src/maths/deriv/ 		\
	-L$(top_srcdir)/src/maths/cmaths/ 		\
	-L$(top_srcdir)/src/maths/poly/ 		\
	-L$(top_srcdir)/src/maths/ni/ 			\
	-L$(top_srcdir)/src/maths/sparse/ 		\
	-L$(top_srcdir)/src/misc/ 			\
							\
	-L$(top_srcdir)/src/frontend/ 			\
	-L$(top_srcdir)/src/frontend/plotting/	 	\
							\
	-L$(top_srcdir)/src/spicelib/devices/asrc/	\
	-L$(top_srcdir)/src/spicelib/devices/bjt/	 \
	-L$(top_srcdir)/src/spicelib/devices/bsim1/      	\
	-L$(top_srcdir)/src/spicelib/devices/bsim2/      	\
	-L$(top_srcdir)/src/spicelib/devices/bsim3/      	\
	-L$(top_srcdir)/src/spicelib/devices/bsim3v1/  	\
	-L$(top_srcdir)/src/spicelib/devices/bsim3v2/ 	\
	-L$(top_srcdir)/src/spicelib/devices/bsim4/      	\
	-L$(top_srcdir)/src/spicelib/devices/cap/		 \
	-L$(top_srcdir)/src/spicelib/devices/bsim3soi_pd/   \
	-L$(top_srcdir)/src/spicelib/devices/bsim3soi_fd/   \
	-L$(top_srcdir)/src/spicelib/devices/bsim3soi_dd/   \
	-L$(top_srcdir)/src/spicelib/devices/cccs/		\
	-L$(top_srcdir)/src/spicelib/devices/ccvs/		\
	-L$(top_srcdir)/src/spicelib/devices/cpl/		  \
	-L$(top_srcdir)/src/spicelib/devices/csw/		  \
	-L$(top_srcdir)/src/spicelib/devices/dio/		  \
									\
	-L$(top_srcdir)/src/spicelib/devices/ind/		  \
	-L$(top_srcdir)/src/spicelib/devices/isrc/		\
	-L$(top_srcdir)/src/spicelib/devices/hfet1/	       \
	-L$(top_srcdir)/src/spicelib/devices/hfet2/	      \
	-L$(top_srcdir)/src/spicelib/devices/jfet/		\
	-L$(top_srcdir)/src/spicelib/devices/jfet2/	      \
	-L$(top_srcdir)/src/spicelib/devices/ltra/		\
	-L$(top_srcdir)/src/spicelib/devices/mes/		  \
	-L$(top_srcdir)/src/spicelib/devices/mesa/	  	\
	-L$(top_srcdir)/src/spicelib/devices/mos1/		\
	-L$(top_srcdir)/src/spicelib/devices/mos2/		\
	-L$(top_srcdir)/src/spicelib/devices/mos3/		\
	-L$(top_srcdir)/src/spicelib/devices/mos6/		\
	-L$(top_srcdir)/src/spicelib/devices/mos9/		\
	-L$(top_srcdir)/src/spicelib/devices/res/		  \
	-L$(top_srcdir)/src/spicelib/devices/soi3/		\
	-L$(top_srcdir)/src/spicelib/devices/sw/		    \
	-L$(top_srcdir)/src/spicelib/devices/txl/	   	\
	-L$(top_srcdir)/src/spicelib/devices/tra/		  \
	-L$(top_srcdir)/src/spicelib/devices/urc/		  \
	-L$(top_srcdir)/src/spicelib/devices/vccs/		\
	-L$(top_srcdir)/src/spicelib/devices/vcvs/		\
	-L$(top_srcdir)/src/spicelib/devices/vsrc/		


EXTRA_LIB_DIRS = \
	-L$(top_srcdir)/src/xspice/cm/.libs/ 			\
	-L$(top_srcdir)/src/xspice/enh/.libs/ 		\
	-L$(top_srcdir)/src/xspice/evt/.libs/ 		\
	-L$(top_srcdir)/src/xspice/idn/.libs/ 		\
	-L$(top_srcdir)/src/xspice/ipc/.libs/ 		\
	-L$(top_srcdir)/src/xspice/mif/.libs/ 		\
	-L$(top_srcdir)/src/spicelib/devices/.libs/ 		\
	-L$(top_srcdir)/src/spicelib/analysis/.libs/	\
	-L$(top_srcdir)/src/spicelib/parser/.libs/	\
					   		\
	-L$(top_srcdir)/src/frontend/parser/.libs/	\
	-L$(top_srcdir)/src/frontend/help/.libs/		\
	-L$(top_srcdir)/src/maths/deriv/.libs/		\
	-L$(top_srcdir)/src/maths/cmaths/.libs/		\
	-L$(top_srcdir)/src/maths/poly/.libs/		\
	-L$(top_srcdir)/src/maths/ni/.libs/		\
	-L$(top_srcdir)/src/maths/sparse/.libs		\
	-L$(top_srcdir)/src/misc/.libs/		\
							\
	-L$(top_srcdir)/src/frontend/.libs/			\
	-L$(top_srcdir)/src/frontend/plotting/.libs/	 	\
							\
	-L$(top_srcdir)/src/spicelib/devices/asrc/.libs/	\
	-L$(top_srcdir)/src/spicelib/devices/bjt/.libs/    \
	-L$(top_srcdir)/src/spicelib/devices/bsim1/.libs/  	\
	-L$(top_srcdir)/src/spicelib/devices/bsim2/.libs/ 	\
	-L$(top_srcdir)/src/spicelib/devices/bsim3/.libs/ 	\
	-L$(top_srcdir)/src/spicelib/devices/bsim3v1/.libs/  	\
	-L$(top_srcdir)/src/spicelib/devices/bsim3v2/.libs/ 	\
	-L$(top_srcdir)/src/spicelib/devices/bsim4/.libs/   	\
	-L$(top_srcdir)/src/spicelib/devices/cap/.libs/ 	 \
	-L$(top_srcdir)/src/spicelib/devices/bsim3soi_pd/.libs/   \
	-L$(top_srcdir)/src/spicelib/devices/bsim3soi_fd/.libs/   \
	-L$(top_srcdir)/src/spicelib/devices/bsim3soi_dd/.libs/   \
	-L$(top_srcdir)/src/spicelib/devices/cccs/.libs/	       \
	-L$(top_srcdir)/src/spicelib/devices/ccvs/.libs/	   \
	-L$(top_srcdir)/src/spicelib/devices/cpl/.libs/	   \
	-L$(top_srcdir)/src/spicelib/devices/csw/.libs/	    \
	-L$(top_srcdir)/src/spicelib/devices/dio/.libs/	    \
									\
	-L$(top_srcdir)/src/spicelib/devices/ind/.libs/	    \
	-L$(top_srcdir)/src/spicelib/devices/isrc/.libs/	   \
	-L$(top_srcdir)/src/spicelib/devices/hfet1/.libs/	  \
	-L$(top_srcdir)/src/spicelib/devices/hfet2/.libs/	\
	-L$(top_srcdir)/src/spicelib/devices/jfet/.libs/	 \
	-L$(top_srcdir)/src/spicelib/devices/jfet2/.libs/	 \
	-L$(top_srcdir)/src/spicelib/devices/ltra/.libs/	 \
	-L$(top_srcdir)/src/spicelib/devices/mes/.libs/	   \
	-L$(top_srcdir)/src/spicelib/devices/mesa/.libs/     	\
	-L$(top_srcdir)/src/spicelib/devices/mos1/.libs/	  \
	-L$(top_srcdir)/src/spicelib/devices/mos2/.libs/	  \
	-L$(top_srcdir)/src/spicelib/devices/mos3/.libs/	  \
	-L$(top_srcdir)/src/spicelib/devices/mos6/.libs/	  \
	-L$(top_srcdir)/src/spicelib/devices/mos9/.libs/	  \
	-L$(top_srcdir)/src/spicelib/devices/res/.libs/	   \
	-L$(top_srcdir)/src/spicelib/devices/soi3/.libs/	   \
	-L$(top_srcdir)/src/spicelib/devices/sw/.libs/	    \
	-L$(top_srcdir)/src/spicelib/devices/txl/.libs/      	\
	-L$(top_srcdir)/src/spicelib/devices/tra/.libs/	    \
	-L$(top_srcdir)/src/spicelib/devices/urc/.libs/	    \
	-L$(top_srcdir)/src/spicelib/devices/vccs/.libs/	   \
	-L$(top_srcdir)/src/spicelib/devices/vcvs/.libs/	  \
	-L$(top_srcdir)/src/spicelib/devices/vsrc/.libs/	

clean: clean-recursive
	rm -f cmextrn.h cminfo.h udnextrn.h udninfo.h *.o


cmextrn.h: modpath.lst $(SUBDIRS)/cfunc.c $(SUBDIRS)/ifspec.c
	$(CMPPDIR)/cmpp -lst

cminfo.h: modpath.lst $(SUBDIRS)/cfunc.c $(SUBDIRS)/ifspec.c
	$(CMPPDIR)/cmpp -lst

udnextrn.h: udnpath.lst $(SUBDIRS)/cfunc.c $(SUBDIRS)/ifspec.c
	$(CMPPDIR)/cmpp -lst

udninfo.h: udnpath.lst  $(SUBDIRS)/cfunc.c $(SUBDIRS)/ifspec.c
	$(CMPPDIR)/cmpp -lst

all: spice2poly.cm

spice2poly.cm: $(SUBDIRS)/cfunc.o $(SUBDIRS)/ifspec.o dlmain.o
	$(top_srcdir)/libtool --mode=link $(CC) -module -shared -fPIC -o spice2poly.cm \
		$(SUBDIRS)/cfunc.o $(SUBDIRS)/ifspec.o dlmain.o \
		$(SYSTEM_LIBS) $(LOCAL_LIB_DIRS) \
		-Wl,--start-group -Wl,$(LOCAL_LIBS) -Wl,--end-group -ldl \
		-Wl,--rpath,/usr/local/geda/lib/ng-spice-rework

dlmain.o: dlmain.c cmextrn.h udnextrn.h cminfo.h udninfo.h
	$(CC) -fPIC -I$(OPUS_INCLUDE_DIR) $(INCLUDES) $(CFLAGS) -DCM_LINUX -c dlmain.c


distclean: clean

check:
