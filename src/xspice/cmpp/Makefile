
cmpp_OBJS=main.o  pp_ifs.o  pp_lst.o  pp_mod.o  read_ifs.o  util.o  writ_ifs.o \
	ifs_yacc.o ifs_lex.o mod_yacc.o mod_lex.o

cmpp_GEN=ifs_yacc.c ifs_tok.h \
	ifs_lex.c ifs_lex.h \
	mod_lex.c mod_lex.h \
	mod_yacc.c 	mod_tok.h

YACC=bison -d --yacc
#YACC=yacc -d

CC=gcc -O2 -g

LEX=flex -t

all: cmpp

cmpp: $(cmpp_OBJS)
	$(CC) -o cmpp $(cmpp_OBJS)

%.c : %.y
	$(YACC) -p $(*:yacc=)yy $<
	mv -f y.tab.c $*.c
	mv -f y.tab.h $(*:yacc=)tok.h

%.c : %.l
	$(LEX) -P$(*:lex=)yy $< > $@
	
ifs_lex.c : ifs_lex.l
	$(LEX) -i -P$(*:lex=)yy $< > $@



clean:
	rm -f $(cmpp_OBJS) $(cmpp_GEN) cmpp 
