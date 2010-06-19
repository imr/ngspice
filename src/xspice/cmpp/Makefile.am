## Process this file with automake to produce Makefile.in

BUILT_SOURCES = ifs_lex.c ifs_yacc.c ifs_yacc.h mod_lex.c mod_yacc.c mod_yacc.h

EXTRA_DIST = ifs_lex.l mod_lex.l ifs_yacc.h mod_yacc.h

MAINTAINERCLEANFILES = Makefile.in

LEX = @LEX@ -t
BISON = bison

bin_PROGRAMS = cmpp

cmpp_SOURCES = cmpp.h main.c pp_ifs.c pp_lst.c pp_mod.c read_ifs.c util.c \
	writ_ifs.c ifs_yacc_y.h ifs_yacc.y ifs_lex.c mod_yacc_y.h mod_yacc.y \
	mod_lex.c

mod_lex.o : mod_yacc.h
mod_lex.c : mod_lex.l
	$(LEX)  -Pmod_yy $< > $@

ifs_lex.o : ifs_yacc.h
ifs_lex.c : ifs_lex.l
	$(LEX)  -i -Pifs_yy $< > $@

ifs_yacc.c ifs_yacc.h : ifs_yacc.y
	$(BISON) $(BISONFLAGS) -d -o ifs_yacc.c $<

mod_yacc.c mod_yacc.h : mod_yacc.y
	$(BISON) $(BISONFLAGS) -d -o mod_yacc.c $<
