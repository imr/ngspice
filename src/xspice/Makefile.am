# Process this file with automake

EXTRA_DIST = README examples icm xspice.c .gitignore \
             verilog/vlnggen \
             verilog/verilator_shim.cpp verilog/verilator_main.cpp \
             verilog/libvvp.def verilog/MSVC.CMD verilog/README.txt \
             vhdl/ghnggen \
             vhdl/ghdl_shim.h vhdl/ghdl_shim.c vhdl/ghdl_vpi.c \
             tlines/msline_common.c tlines/msline_common.h \
             tlines/tline_common.c tlines/tline_common.h

SUBDIRS = mif cm enh evt ipc idn cmpp icm verilog vhdl

dist-hook:
	rm -f "$(distdir)/icm/makedefs"
	rm -f "$(distdir)/icm/GNUmakefile"
	rm -rf `find $(distdir)/icm -name .deps`
	rm -rf `find $(distdir)/icm -name *.o`
	rm -rf `find $(distdir)/icm -name *.cm`
	for sub in analog digital spice2poly xtradev xtraevt table ; do \
	    rm -rf `find $(distdir)/icm/$$sub -name cmextrn.h`; \
	    rm -rf `find $(distdir)/icm/$$sub -name cminfo.h`; \
	    rm -rf `find $(distdir)/icm/$$sub -name udnextrn.h`; \
	    rm -rf `find $(distdir)/icm/$$sub -name udninfo.h`; \
	    rm -rf `find $(distdir)/icm/$$sub -name ifspec.c`; \
	    rm -rf `find $(distdir)/icm/$$sub -name cfunc.c`; \
	    rm -rf `find $(distdir)/icm/$$sub -name dlmain.c`; \
	    rm -rf `find $(distdir)/icm/$$sub -name objects.inc`; \
	done

MAINTAINERCLEANFILES = Makefile.in
