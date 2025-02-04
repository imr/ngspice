 ## Process this file with automake to produce Makefile.in

MAINTAINERCLEANFILES = Makefile.in

# Libs for Ngspice are irrelevant here.

LIBS =

# Verilator support: files installed to script directory and below.

initdatadir = $(pkgdatadir)/scripts
initdata_DATA = vlnggen

initdata1dir = $(pkgdatadir)/scripts/src
initdata1_DATA = verilator_shim.cpp verilator_main.cpp

initdata2dir = $(pkgdatadir)/scripts/src/ngspice
initdata2_DATA = ../../include/ngspice/cosim.h \
		 ../../include/ngspice/miftypes.h \
                 ../../include/ngspice/cmtypes.h \
		 ./coroutine.h ./coroutine_cosim.h ./coroutine_shim.h

# Icarus Verilog support: build two shared libraries.

pkglib_LTLIBRARIES = ivlng.la ivlngvpi.la
ivlng_la_SOURCES = icarus_shim.c icarus_shim.h coroutine_shim.h
ivlng_la_CFLAGS = -I$(top_srcdir)/src/include
ivlng_la_LDFLAGS = -module -shared -avoid-version

ivlngvpi_la_SOURCES = vpi.c icarus_shim.h vpi_user_dummy.h coroutine.h coroutine_cosim.h
ivlngvpi_la_CFLAGS = -I$(top_srcdir)/src/include
ivlngvpi_la_LDFLAGS = -module -shared  -avoid-version

# On Windows, symbols in DLLs must be fully resolved.
# Create a dummy libvvp.DLL so that Icarus Verilog need not be installed
# for building.

if DLIBS_FULLY_RESOLVED
pkglib_LTLIBRARIES += libvvp.la
libvvp_la_SOURCES = vpi_dummy.c vpi_user_dummy.h coroutine.h coroutine_cosim.h
libvvp_la_LDFLAGS = -no-undefined -module -shared -avoid-version
ivlng_la_LDFLAGS += -no-undefined
ivlngvpi_la_LIBADD = libvvp.la ivlng.la
ivlngvpi_la_LDFLAGS += -no-undefined
endif

# Libtool installs unwanted libraries, remove them after installation.
# On Windows, the dummy libvvp.* files are removed also.

install-exec-hook:
	cd $(DESTDIR)$(pkglibdir); \
	rm -f ivlng*a libvvp* ; \
	mv ivlngvpi.* ivlng.vpi

uninstall-hook:
	rm -f $(DESTDIR)$(pkglibdir)/ivlng.vpi $(DESTDIR)$(pkglibdir)/ivlng.so
