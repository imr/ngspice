# The master makefile to make spiceopuse (TM) like codemodels
# Under the GPLV2 or later license 
# 2003 - Stefan Jones <stefan.jones@multigig.com>

include $(TOPDIR)makedefs

DEPS_MAGIC := $(shell mkdir .deps > /dev/null 2>&1 || :)

-include .deps/ifspec.P 
-include .deps/cfunc.P 
-include .deps/cm.P 
-include .deps/dlmain.P

UPMAKE = make -f $(TOPDIR)../Makefile TOPDIR=$(TOPDIR)../

MAKE = make -f $(TOPDIR)Makefile TOPDIR=$(TOPDIR)

COMPILE = $(CC) $(INCLUDES) $(CFLAGS)

all: 
	@for x in $(CMDIRS) ; do \
	( cd $$x && $(UPMAKE) $$x-mods ) ; done

clean:
	@for x in $(CMDIRS) ; do \
	( cd $$x && $(UPMAKE) $$x-mods-clean ) ; done
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
	@echo $@: objects.inc dlmain.o \\ > .deps/cm.P ; \
	for x in `cat udnpath.lst` `cat modpath.lst ` ; do \
	echo $$x/cfunc.o $$x/ifspec.o \\ >> .deps/cm.P ; done ; \
	echo "" >> .deps/cm.P
	$(COMPILE) $(LDFLAGS) -o $@ `cat objects.inc ` dlmain.o

%-mods: modpath.lst udnpath.lst
	@for x in `cat udnpath.lst` `cat modpath.lst ` ; do \
	( cd $$x && $(UPMAKE) objs ) ; done
	@target=`echo $@ | sed s/-mods//`; $(MAKE) $$target.cm

%-mods-clean:
	@target=`echo $@ | sed s/-mods-clean//` && rm -f $$target.cm && \
	for x in `cat udnpath.lst` `cat modpath.lst ` ; do \
	( cd $$x && $(UPMAKE) objs-clean ) ; done
	-rm -f cmextrn.h cminfo.h objects.inc udnextrn.h udninfo.h \
	dlmain.c dlmain.o
	-rm -rf .deps
	

objs: ifspec.o cfunc.o

objs-clean:
	-rm -f cfunc.c ifspec.c cfunc.o ifspec.o
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

