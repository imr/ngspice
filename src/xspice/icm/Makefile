# The master makefile to make spiceopuse (TM) like codemodels
# Under the GPLV2 or later license 
# 2003 - Stefan Jones <stefan.jones@multigig.com>

include $(TOPDIR)makedefs

DEPS_MAGIC := $(shell mkdir .deps > /dev/null 2>&1 || :)

-include .deps/ifspec.P
-include .deps/cfunc.P
-include .deps/udnfunc.P
-include .deps/cm.P
-include .deps/dlmain.P

UPMAKE = make -f $(TOPDIR)../Makefile TOPDIR=$(TOPDIR)../

MAKE = make -f $(TOPDIR)Makefile TOPDIR=$(TOPDIR)

COMPILE = $(CC) $(INCLUDES) $(CFLAGS)

INSTALL_DATA = ${INSTALL} -m 644

all: 
	@for x in $(CMDIRS) ; do \
	( cd $$x && $(UPMAKE) $$x-mods ) \
	|| exit 1; \
	done

install: all
	$(mkinstalldirs) $(DESTDIR)$(libdir)/spice
	@for x in $(CMDIRS) ; do \
	echo "$(INSTALL_DATA) $$x/$$x.cm $(DESTDIR)$(libdir)/spice"; \
	$(INSTALL_DATA) $$x/$$x.cm $(DESTDIR)$(libdir)/spice \
	|| exit 1; \
	done 

clean:
	@for x in $(CMDIRS) ; do \
	( cd $$x && $(UPMAKE) $$x-mods-clean ) \
	|| exit 1; \
	done
	-rm -rf .deps

ifspec.c: ifspec.ifs
	-rm -f $@
	$(CMPP) -ifs

cfunc.c: cfunc.mod
	-rm -f $@
	$(CMPP) -mod

dlmain.c: $(TOPDIR)/dlmain.c 
	-cp $(TOPDIR)/dlmain.c .

objects.inc cmextrn.h cminfo.h udnextrn.h udninfo.h: modpath.lst  udnpath.lst
	-rm -f cmextrn.h cminfo.h objects.inc udnextrn.h udninfo.h 
	$(CMPP) -lst

dlmain.o: cmextrn.h cminfo.h udnextrn.h udninfo.h

%.cm: dlmain.o objects.inc
	@echo $@: objects.inc dlmain.o \\ > .deps/cm.P
	@for x in `cat modpath.lst` ; do \
	echo $$x/cfunc.o $$x/ifspec.o \\ >> .deps/cm.P ; done
	@for x in `cat udnpath.lst` ; do \
	echo $$x/udnfunc.o \\ >> .deps/cm.P ; done
	@echo "" >> .deps/cm.P
	$(COMPILE) $(LDFLAGS) -o $@ `awk '{ print $$1 }' objects.inc` dlmain.o

%-mods: modpath.lst udnpath.lst
	@for x in `cat modpath.lst` ; do \
	( cd $$x && $(UPMAKE) objs ) \
	|| exit 1; \
	done
	@for x in `cat udnpath.lst` ; do \
	( cd $$x && $(UPMAKE) uobjs ) \
	|| exit 1; \
	done
	@target=`echo $@ | sed s/-mods//`; $(MAKE) $$target.cm

%-mods-clean:
	@target=`echo $@ | sed s/-mods-clean//` && rm -f $$target.cm
	@for x in `cat modpath.lst` ; do \
	( cd $$x && $(UPMAKE) objs-clean ) \
	|| exit 1; \
	done
	@for x in `cat udnpath.lst` ; do \
	( cd $$x && $(UPMAKE) uobjs-clean ) \
	|| exit 1; \
	done
	-rm -f cmextrn.h cminfo.h objects.inc udnextrn.h udninfo.h \
	dlmain.c dlmain.o
	-rm -rf .deps
	

objs: ifspec.o cfunc.o

objs-clean:
	-rm -f cfunc.c ifspec.c cfunc.o ifspec.o
	-rm -rf .deps

uobjs: udnfunc.o

uobjs-clean:
	-rm -f udnfunc.o
	-rm -rf .deps

%.o: %.c
	@echo '$(COMPILE) -c $<'; \
	$(COMPILE) -Wp,-MD,.deps/$(*F).pp -c $<
	@-cp .deps/$(*F).pp .deps/$(*F).P; \
	tr ' ' '\012' < .deps/$(*F).pp \
	| sed -e 's/^\\$$//' -e '/^$$/ d' -e '/:$$/ d' -e 's/$$/ :/' \
	>> .deps/$(*F).P; \
	rm .deps/$(*F).pp

makedefs: $(srcdir)/makedefs.in $(top_builddir)/config.status
	cd $(top_builddir) \
	&& CONFIG_FILES=$(subdir)/$@ CONFIG_HEADERS= $(SHELL) ./config.status

# Hack by SDB to enable "make mrproper" from top level.
#  Since there is no Makefile.am for this directory, we need to 
#  incorporate this stuff by hand.
MAINTAINERCLEANFILES = 

maintainer-clean: maintainer-clean-generic
	rm -f `find . -type f -name "*~" -print`
	rm -f `find . -type f -name "*.orig" -print`
	rm -f `find . -type f -name "*.rej" -print`
	rm -fR `find . -type d -name ".deps" -print`

maintainer-clean-generic: clean
	-test -z "$(MAINTAINERCLEANFILES)" || rm -f $(MAINTAINERCLEANFILES)


distdir:


