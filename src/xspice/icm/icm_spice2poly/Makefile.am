## Process this file with automake to produce Makefile.in
#
#  Makefile.am hand-crafted by SDB on 5.25.2003 to create POLY
#  codemodel shared object.  To enable simulation of POLY controlled sources,
#  do the following:
#

# Edit this to point to cmpp in your installation.
CMPPDIR = /usr/local/opus/bin

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
        -L$(top_srcdir)/src/spicelib/devices/asrc/        \
        -L$(top_srcdir)/src/spicelib/devices/bjt/         \
        -L$(top_srcdir)/src/spicelib/devices/bsim1/      	\
        -L$(top_srcdir)/src/spicelib/devices/bsim2/      	\
        -L$(top_srcdir)/src/spicelib/devices/bsim3/      	\
        -L$(top_srcdir)/src/spicelib/devices/bsim3v1/  	\
        -L$(top_srcdir)/src/spicelib/devices/bsim3v2/ 	\
        -L$(top_srcdir)/src/spicelib/devices/bsim4/      	\
        -L$(top_srcdir)/src/spicelib/devices/cap/        	 \
        -L$(top_srcdir)/src/spicelib/devices/bsim3soi_pd/   \
        -L$(top_srcdir)/src/spicelib/devices/bsim3soi_fd/   \
        -L$(top_srcdir)/src/spicelib/devices/bsim3soi_dd/   \
        -L$(top_srcdir)/src/spicelib/devices/cccs/                \
        -L$(top_srcdir)/src/spicelib/devices/ccvs/                \
        -L$(top_srcdir)/src/spicelib/devices/ccvs/                \
        -L$(top_srcdir)/src/spicelib/devices/cpl/                  \
        -L$(top_srcdir)/src/spicelib/devices/csw/                  \
        -L$(top_srcdir)/src/spicelib/devices/dio/                  \
									\
        -L$(top_srcdir)/src/spicelib/devices/ind/                  \
        -L$(top_srcdir)/src/spicelib/devices/isrc/                \
        -L$(top_srcdir)/src/spicelib/devices/hfet1/               \
        -L$(top_srcdir)/src/spicelib/devices/hfet2/              \
        -L$(top_srcdir)/src/spicelib/devices/jfet/                \
        -L$(top_srcdir)/src/spicelib/devices/jfet2/              \
        -L$(top_srcdir)/src/spicelib/devices/ltra/                \
        -L$(top_srcdir)/src/spicelib/devices/mes/                  \
        -L$(top_srcdir)/src/spicelib/devices/mesa/          	\
        -L$(top_srcdir)/src/spicelib/devices/mos1/                \
        -L$(top_srcdir)/src/spicelib/devices/mos2/                \
        -L$(top_srcdir)/src/spicelib/devices/mos3/                \
        -L$(top_srcdir)/src/spicelib/devices/mos6/                \
        -L$(top_srcdir)/src/spicelib/devices/mos9/                \
        -L$(top_srcdir)/src/spicelib/devices/res/                  \
        -L$(top_srcdir)/src/spicelib/devices/soi3/                \
        -L$(top_srcdir)/src/spicelib/devices/sw/                    \
        -L$(top_srcdir)/src/spicelib/devices/txl/           	\
        -L$(top_srcdir)/src/spicelib/devices/tra/                  \
        -L$(top_srcdir)/src/spicelib/devices/urc/                  \
        -L$(top_srcdir)/src/spicelib/devices/vccs/                \
        -L$(top_srcdir)/src/spicelib/devices/vcvs/                \
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
        -L$(top_srcdir)/src/spicelib/devices/asrc/.libs/        \
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
        -L$(top_srcdir)/src/spicelib/devices/cccs/.libs/               \
        -L$(top_srcdir)/src/spicelib/devices/ccvs/.libs/           \
        -L$(top_srcdir)/src/spicelib/devices/ccvs/.libs/          \
        -L$(top_srcdir)/src/spicelib/devices/cpl/.libs/           \
        -L$(top_srcdir)/src/spicelib/devices/csw/.libs/            \
        -L$(top_srcdir)/src/spicelib/devices/dio/.libs/            \
									\
        -L$(top_srcdir)/src/spicelib/devices/ind/.libs/            \
        -L$(top_srcdir)/src/spicelib/devices/isrc/.libs/           \
        -L$(top_srcdir)/src/spicelib/devices/hfet1/.libs/          \
        -L$(top_srcdir)/src/spicelib/devices/hfet2/.libs/        \
        -L$(top_srcdir)/src/spicelib/devices/jfet/.libs/         \
        -L$(top_srcdir)/src/spicelib/devices/jfet2/.libs/         \
        -L$(top_srcdir)/src/spicelib/devices/ltra/.libs/         \
        -L$(top_srcdir)/src/spicelib/devices/mes/.libs/           \
        -L$(top_srcdir)/src/spicelib/devices/mesa/.libs/     	\
        -L$(top_srcdir)/src/spicelib/devices/mos1/.libs/          \
        -L$(top_srcdir)/src/spicelib/devices/mos2/.libs/          \
        -L$(top_srcdir)/src/spicelib/devices/mos3/.libs/          \
        -L$(top_srcdir)/src/spicelib/devices/mos6/.libs/          \
        -L$(top_srcdir)/src/spicelib/devices/mos9/.libs/          \
        -L$(top_srcdir)/src/spicelib/devices/res/.libs/           \
        -L$(top_srcdir)/src/spicelib/devices/soi3/.libs/           \
        -L$(top_srcdir)/src/spicelib/devices/sw/.libs/            \
        -L$(top_srcdir)/src/spicelib/devices/txl/.libs/      	\
        -L$(top_srcdir)/src/spicelib/devices/tra/.libs/            \
        -L$(top_srcdir)/src/spicelib/devices/urc/.libs/            \
        -L$(top_srcdir)/src/spicelib/devices/vccs/.libs/           \
        -L$(top_srcdir)/src/spicelib/devices/vcvs/.libs/          \
        -L$(top_srcdir)/src/spicelib/devices/vsrc/.libs/	


INCLUDES = -I$(top_srcdir)/src/include -I$(top_srcdir)

EXTRA_DIST = cfunc.mod ifspec.ifs README cfunc.c ifspec.c README

MAINTAINERCLEANFILES = Makefile.in

##------ Make targets listed below. ------

all: cfunc.o ifspec.o

clean:
	rm -f *.o *.cm *.la
	rm -fR .libs

cfunc.o: cfunc.c
	$(CC) -fPIC $(INCLUDES) $(CFLAGS) -c cfunc.c

ifspec.o: ifspec.c
	$(CC) -fPIC $(INCLUDES) $(CFLAGS) -c ifspec.c

# Don't use the below targets unless you know what you are doing!
#
#  Hint:  if you create ifspec.c using cmpp, you need to replace MIFunsetup 
#  with NULL in ifspec.c before making ifspec.o (compiling with gcc) because
#  MIFunsetup is apparently unimplemented.
#
#cclean: clean
#	rm -f cfunc.c ifspec.c
#
#cfunc.c: cfunc.mod
#	$(CMPPDIR)/cmpp -mod cfunc.mod
#
#ifspec.c: ifspec.ifs
#	$(CMPPDIR)/cmpp -ifs 


