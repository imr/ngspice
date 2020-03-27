/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1986 Wayne A. Christopher, U. C. Berkeley CAD Group
Modified: 2000  AlansFixes
**********/

/*
 * Definitions for all external symbols in CP.
 */

#ifndef ngspice_CPEXTERN_H
#define ngspice_CPEXTERN_H

#include "ngspice/wordlist.h"
#include "ngspice/bool.h"

#include <stdarg.h>

struct ccom;

/* com_alias.c */

extern struct alias *cp_aliases;
extern void cp_paliases(char *word);
extern void cp_setalias(char *word, wordlist *wlist);
extern void cp_unalias(char *word);

extern wordlist *cp_doalias(wordlist *wlist);

/* backquote.c */

extern char cp_back;
extern wordlist *cp_bquote(wordlist *wlist);

/* complete.c */

extern bool cp_nocc;
extern bool cp_comlook(char *word);
extern struct ccom *cp_kwswitch(int kw_class, struct ccom *tree);
extern void cp_addcomm(char *word, long int bits0, long int bits1, long int bits2, long int bits3);
extern void cp_addkword(int kw_class, char *word);
extern void cp_ccom(wordlist *wlist, char *buf, bool esc);
extern void cp_ccon(bool on);
extern void cp_ccrestart(bool kwords);
extern void cp_remcomm(char *word);
extern void cp_remkword(int kw_class, const char *word);
extern void cp_destroy_keywords(void);

extern wordlist *cp_cctowl(struct ccom *stuff);

/* cshpar.c */

extern FILE *cp_in;
extern FILE *cp_out;
extern FILE *cp_err;
extern FILE *cp_curin;
extern FILE *cp_curout;
extern FILE *cp_curerr;
extern bool cp_debug;
extern bool cp_no_histsubst; /* controlled by "no_histsubst" true/false */
extern char cp_amp;
extern char cp_gt;
extern char cp_lt;
extern void cp_ioreset(void);
extern wordlist *cp_redirect(wordlist *wlist);
extern wordlist *cp_parse(char *string);

/* control.c */

extern bool cp_cwait;
extern bool cp_dounixcom;
extern char *cp_csep;
extern char * get_alt_prompt(void);
extern int cp_evloop(char *string);
extern void cp_resetcontrol(bool warn);
extern void cp_toplevel(void);
extern void cp_popcontrol(void);
extern void cp_pushcontrol(void);

/* glob.c */

extern char *cp_tildexpand(const char *string);
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
extern void cp_addhistent(int event, wordlist *wlist);
extern wordlist *cp_histsubst(wordlist *wlist);

/* lexical.c */

extern FILE *cp_inp_cur;
extern bool cp_bqflag;
extern bool cp_interactive;
extern char *cp_altprompt;
extern char *cp_promptstring;
extern int cp_event;
extern wordlist *cp_lexer(char *string);
extern int inchar(FILE *fp);

/* modify.c */

extern char cp_chars[];
extern void cp_init(void);

/* output.c */

extern bool out_moremode;
extern bool out_isatty;
extern void out_init(void);

#ifdef __GNUC__
extern void out_printf(char *fmt, ...) __attribute__ ((format (__printf__, 1, 2)));
#else
extern void out_printf(char *fmt, ...);
#endif

extern void out_vprintf(const char *fmt, va_list ap);
extern void out_send(char *string);

/* quote.c */

extern char *cp_unquote(const char *string);

/* unixcom.c */

extern bool cp_unixcom(wordlist *wlist);
extern void cp_hstat(void);
void cp_rehash(char *pathlist, bool docc);

/* variable.c */

enum cp_types {
  CP_BOOL,
  CP_NUM,
  CP_REAL,
  CP_STRING,
  CP_LIST
};

extern bool cp_ignoreeof;
extern bool cp_noclobber;
extern bool cp_noglob;
extern bool cp_nonomatch;
extern char cp_dol;
extern void cp_remvar(char *varname);
void cp_vset(const char *varname, enum cp_types type, const void *value);
extern struct variable *cp_setparse(wordlist *wl);
extern wordlist *vareval(char *string);
extern char *span_var_expr(char *t);

/* var2.c */
extern void cp_vprint(void);
extern bool cp_getvar(char *name, enum cp_types type, void *retval, size_t rsize);

/* cpinterface.c etc -- stuff CP needs from FTE */

extern bool cp_istrue(wordlist *wl);
extern bool cp_oddcomm(char *s, wordlist *wlist);
extern void cp_doquit(void);
extern void cp_periodic(void);
extern void ft_cpinit(void);
extern struct comm *cp_coms;
extern char *cp_program;
extern struct variable *cp_enqvar(const char *word, int *tbfreed);
extern struct variable *cp_usrvars(void);
int cp_usrset(struct variable *var, bool isset);
extern void fatal(void);

#endif
