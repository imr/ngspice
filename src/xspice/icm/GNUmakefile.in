# The master makefile to make spiceopus (TM) like codemodels
# Under the GPLV2 or later license
# 2003 - Stefan Jones <stefan.jones@multigig.com>


include makedefs


# The codemodels to make
CMDIRS = spice2poly digital analog xtradev xtraevt table


#Invoke $(MAKE) for each of the CMDDIRS
all: dstring.o # One common dstring object file for all code modules
	for cm in $(CMDIRS) ; do \
		$(MAKE) cm=$$cm $$cm/$$cm.cm \
		|| exit 1; \
	done


install: all
	$(MKDIR_P) "$(DESTDIR)$(pkglibdir)"
	$(MKDIR_P) "$(DESTDIR)$(pkgdatadir)"
	for cm in $(CMDIRS) ; do \
		$(INSTALL_PROGRAM) $$cm/$$cm.cm "$(DESTDIR)$(pkglibdir)" \
		|| exit 1; \
	done
#	$(INSTALL_DATA) $(srcdir)/dlmain.c "$(DESTDIR)$(pkgdatadir)" || exit $$?;


uninstall:
	for cm in $(CMDIRS) ; do \
		rm -f "$(DESTDIR)$(pkglibdir)/$$cm.cm"; \
	done


clean:
	rm -f dstring.o
	for cm in $(CMDIRS) ; do \
		$(MAKE) cm=$$cm cm-clean \
		|| exit 1; \
	done


#-----------------------------------------------------------------------------
NGSRCDIR = $(srcdir)/../..
NGSRCBUILDDIR = $(CURDIR)/../..

dstring.o: $(NGSRCDIR)/misc/dstring.c $(NGSRCDIR)/include/ngspice/dstring.h
	$(CC) $(CFLAGS) -I$(NGSRCDIR)/include -I$(NGSRCBUILDDIR)/include -fPIC -o $@ -c $<

ifdef cm

ifeq ($(OS),Windows_NT)
    cmpp = ../cmpp/cmpp.exe
else
    cmpp = ../cmpp/cmpp
endif
modlst := $(shell $(cmpp) -p $(srcdir)/$(cm)/modpath.lst)
udnlst := $(shell $(cmpp) -p $(srcdir)/$(cm)/udnpath.lst)

cm-dirs := $(cm) $(udnlst:%=$(cm)/%) $(modlst:%=$(cm)/%)
cm-dep-dirs := $(cm-dirs:%=%/.deps)


cm-gens := \
	$(modlst:%=$(cm)/%/cfunc.c) \
	$(modlst:%=$(cm)/%/ifspec.c)

cm-objs := \
	$(cm)/dlmain.o \
	dstring.o \
	$(modlst:%=$(cm)/%/cfunc.o) \
	$(modlst:%=$(cm)/%/ifspec.o) \
	$(udnlst:%=$(cm)/%/udnfunc.o)

cm-deps := \
	$(cm)/.deps/dlmain.P \
	$(modlst:%=$(cm)/%/.deps/cfunc.P) \
	$(modlst:%=$(cm)/%/.deps/ifspec.P) \
	$(udnlst:%=$(cm)/%/.deps/udnfunc.P)

cm-descr := \
	$(cm)/cmextrn.h \
	$(cm)/cminfo.h \
	$(cm)/cminfo2.h \
	$(cm)/udnextrn.h \
	$(cm)/udninfo.h \
	$(cm)/udninfo2.h \
	$(cm)/objects.inc


-include $(cm-deps)


.SECONDARY : $(cm-gens)


cm-clean :
	-rm -f $(cm)/$(cm).cm
	-rm -f $(cm-descr) $(cm-objs) $(cm-gens)
	-rm -f $(cm-deps)

cm-distclean :
	-rm -rf $(cm-dep-dirs)

#-----

$(cm)/$(cm).cm : $(cm-objs)
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) $(VIS_CFLAGS) $(LDFLAGS) $^ $(LIBS) -o $@

# "order-only" prerequisites
#   (info "(make) Prerequisite Types")

$(cm-gens) $(cm-objs) $(cm-descr) : | cm-dirs

cm-dirs : $(cm-dirs) $(cm-dep-dirs)

$(cm-dirs) $(cm-dep-dirs) :
	$(MKDIR_P) $(cm-dirs) $(cm-dep-dirs)

#-----
# cancel builtin rules

%.o : %.c
%.o : %.mod

# multiple targets in a pattern group ...
#   (info "(make) Pattern Intro")

%/cmextrn.h %/cminfo.h %/udnextrn.h %/udninfo.h %/objects.inc : $(srcdir)/%/modpath.lst $(srcdir)/%/udnpath.lst
	CMPP_IDIR=$(srcdir)/$(@D) CMPP_ODIR=$(@D) $(CMPP) -lst

%/ifspec.c : $(srcdir)/%/ifspec.ifs
	CMPP_IDIR=$(srcdir)/$(@D) CMPP_ODIR=$(@D) $(CMPP) -ifs

%/cfunc.c : $(srcdir)/%/cfunc.mod
	CMPP_IDIR=$(srcdir)/$(@D) CMPP_ODIR=$(@D) $(CMPP) -mod


gen_pp = $(DEPFLAGS) $(@D)/.deps/$(basename $(<F)).pp

define do-deps
	@x=$(@D)/.deps/$(basename $(<F)) ; \
	cp $$x.pp $$x.P; \
	tr ' ' '\012' < $$x.pp \
	| sed -e 's/^\\$$//' -e '/^$$/ d' -e '/:$$/ d' -e 's/$$/ :/' \
	>> $$x.P; \
	rm $$x.pp
endef

COMPILE = $(CC) $(INCLUDES) -I$(cm) -I$(srcdir)/$(cm) $(CFLAGS) $(EXTRA_CFLAGS) $(VIS_CFLAGS)

$(cm)/dlmain.o : $(srcdir)/dlmain.c $(cm-descr)
	$(COMPILE) $(gen_pp)  -o $@ -c $<
	$(do-deps)

$(cm)/%/cfunc.o : $(cm)/%/cfunc.c
	$(COMPILE) $(gen_pp) -I$(srcdir)/$(<D) -o $@ -c $<
	$(do-deps)

$(cm)/%/ifspec.o : $(cm)/%/ifspec.c
	$(COMPILE) $(gen_pp) -I$(srcdir)/$(<D) -o $@ -c $<
	$(do-deps)

$(cm)/%/udnfunc.o : $(srcdir)/$(cm)/%/udnfunc.c
	$(COMPILE) $(gen_pp) -I$(srcdir)/$(<D) -o $@ -c $<
	$(do-deps)

endif

#-----------------------------------------------------------------------------

GNUmakefile makedefs : % : $(srcdir)/%.in $(top_builddir)/config.status
	cd $(top_builddir) \
	&& CONFIG_FILES=$(subdir)/$@ CONFIG_HEADERS= $(SHELL) ./config.status

maintainer-clean: distclean

distclean: clean
	for cm in $(CMDIRS) ; do \
		$(MAKE) cm=$$cm cm-distclean \
		|| exit 1; \
	done
	-rm -f GNUmakefile makedefs

distdir:

check:

pdf:

ps:

dvi:

html:

info:
