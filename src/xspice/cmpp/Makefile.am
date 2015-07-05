## Process this file with automake to produce Makefile.in

BUILT_SOURCES = ifs_yacc.h mod_yacc.h

EXTRA_DIST = .gitignore

MAINTAINERCLEANFILES = Makefile.in

bin_PROGRAMS = cmpp

AM_CPPFLAGS = -I. -I$(srcdir)
AM_YFLAGS = -d

cmpp_SOURCES = main.c cmpp.h \
	pp_ifs.c pp_lst.c pp_mod.c read_ifs.c writ_ifs.c util.c \
	ifs_lex.l ifs_yacc.y ifs_yacc_y.h \
	mod_lex.l mod_yacc.y mod_yacc_y.h

mod_lex.c : mod_lex.l
	$(am__skiplex) $(LEXCOMPILE) -o $@ $<

mod_yacc.c : mod_yacc.y
	$(am__skipyacc) $(YACCCOMPILE) -o $@ $<

ifs_lex.c : ifs_lex.l
	$(am__skiplex) $(LEXCOMPILE) -o $@ $<

ifs_yacc.c : ifs_yacc.y
	$(am__skipyacc) $(YACCCOMPILE) -o $@ $<
