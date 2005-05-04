/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1986 Wayne A. Christopher, U. C. Berkeley CAD Group
Modified: 2000  AlansFixes
**********/

/*
 * Definitions for all external symbols in CP.
 */

#ifndef CPEXTERN_H
#define CPEXTERN_H

#include "wordlist.h"
#include "bool.h"


/* alias.c */

extern struct alias *cp_aliases;
extern void com_alias();
extern void com_unalias();
extern void cp_paliases();
extern void cp_setalias();
extern void cp_unalias();

extern wordlist *cp_doalias();

/* backquote.c */

extern char cp_back;
extern wordlist *cp_bquote();

/* complete.c */

extern bool cp_nocc;
extern bool cp_comlook(char *word);
extern char *cp_kwswitch(int class, char *tree);
extern void cp_addcomm(char *word, long int bits0, long int bits1, long int bits2, long int bits3);
extern void cp_addkword(int class, char *word);
extern void cp_ccom(wordlist *wlist, char *buf, bool esc);
extern void cp_ccon(bool on);
extern void cp_ccrestart(bool kwords);
extern void cp_remcomm(char *word);
extern void cp_remkword(int class, char *word);
extern wordlist *cp_cctowl(char *stuff);

/* cshpar.c */

extern FILE *cp_in;
extern FILE *cp_out;
extern FILE *cp_err;
extern FILE *cp_curin;
extern FILE *cp_curout;
extern FILE *cp_curerr;
extern bool cp_debug;
extern char cp_amp;
extern char cp_gt;
extern char cp_lt;
extern void com_chdir();
extern void com_echo();
extern void com_strcmp();
extern void com_rehash();
extern void com_shell();
extern void cp_ioreset();
extern wordlist *cp_redirect();
extern wordlist *cp_parse();

/* control.c */

extern bool cp_cwait;
extern bool cp_dounixcom;
extern char *cp_csep;
extern char * get_alt_prompt(void);
extern int cp_evloop(char *string);
extern void cp_resetcontrol(void);
extern void cp_toplevel(void);
extern void cp_popcontrol(void);
extern void cp_pushcontrol(void);

/* com_cdump.c */

extern void com_cdump(wordlist *wl);

/* glob.c */

extern bool cp_globmatch(char *p, char *s);
extern char *cp_tildexpand(char *string);
extern char cp_cbrac;
extern char cp_ccurl;
extern char cp_comma;
extern char cp_huh;
extern char cp_obrac;
extern char cp_ocurl;
extern char cp_star;
extern char cp_til;
extern wordlist *cp_doglob(wordlist *wlist);

/* history.c */

extern bool cp_didhsubst;
extern char cp_bang;
extern char cp_hat;
extern int cp_maxhistlength;
extern struct histent *cp_lastone;
extern void com_history();
extern void cp_addhistent();
void cp_hprint(int eventhi, int eventlo, bool rev);
extern wordlist *cp_histsubst();

/* lexical.c */

extern FILE *cp_inp_cur;
extern bool cp_bqflag;
extern bool cp_interactive;
extern char *cp_altprompt;
extern char *cp_promptstring;
extern char cp_hash;
extern int cp_event;
extern wordlist *cp_lexer(char *string);
extern int inchar(FILE *fp);

/* modify.c */

extern char cp_chars[];
extern void cp_init(void);

/* output.c */

extern char out_pbuf[];
extern bool out_moremode;
extern bool out_isatty;
extern void out_init();
#ifndef out_printf
/* don't want to declare it if we have #define'ed it */

extern void out_printf();
#endif
extern void out_send();

/* quote.c */

extern char *cp_unquote(char *string);
extern void cp_quoteword(char *str);
extern void cp_striplist(wordlist *wlist);
extern void cp_wstrip(char *str);
extern void cp_printword(char *string, FILE *fp); 



/* unixcom.c */

extern bool cp_unixcom();
extern void cp_hstat();
void cp_rehash(char *pathlist, bool docc);

/* variable.c */


extern bool cp_ignoreeof;
extern bool cp_noclobber;
extern bool cp_noglob;
extern bool cp_nonomatch;
extern char cp_dol;
extern void cp_remvar(char *varname);
extern void cp_vset(char *varname, char type, char *value);
extern struct variable *cp_setparse(wordlist *wl);

/* var2.c */
extern void cp_vprint(void);
extern void com_set(wordlist *wl);
extern void com_option(wordlist *wl);
extern void com_state(wordlist *wl);
extern void com_unset(wordlist *wl);
extern void com_shift(wordlist *wl);
extern bool cp_getvar(char *name, int type, void *retval);

/* cpinterface.c etc -- stuff CP needs from FTE */

extern bool cp_istrue(wordlist *wl);
extern bool cp_oddcomm();
extern void cp_doquit();
extern void cp_periodic();
extern void ft_cpinit();
extern struct comm *cp_coms;
extern char *cp_program;
extern bool ft_nutmeg;
extern struct variable *cp_enqvar();
extern void cp_usrvars();
int cp_usrset(struct variable *var, bool isset);
extern void fatal();

#endif
