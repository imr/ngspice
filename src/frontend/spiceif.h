/*************
 * Header file for spiceif.c
 * 1999 E. Rouat
 * $Id$
 ************/

#ifndef SPICEIF_H_INCLUDED
#define SPICEIF_H_INCLUDED

CKTcircuit * if_inpdeck(struct line *deck, INPtables **tab);
int if_run(CKTcircuit *t, char *what, wordlist *args, INPtables *tab);
int if_option(CKTcircuit *ckt, char *name, int type, char *value);
void if_dump(CKTcircuit *ckt, FILE *file);
void if_cktfree(CKTcircuit *ckt, INPtables *tab);
char * if_errstring(int code);
struct variable * spif_getparam(CKTcircuit *ckt, char **name, char *param, int ind, int do_model);
struct variable * spif_getparam_special(CKTcircuit *ckt,char **name,char *param,int ind,int do_model);
void if_setparam_model(CKTcircuit *ckt, char **name, char *val);
void if_setparam(CKTcircuit *ckt, char **name, char *param, struct dvec *val, int do_model);
int  if_analQbyName(CKTcircuit *ckt, int which, void *anal, char *name, IFvalue *parm);
bool if_tranparams(struct circ *ci, double *start, double *stop, double *step);
struct variable * if_getstat(CKTcircuit *ckt, char *name);

#ifdef EXPERIMENTAL_CODE
void com_loadsnap(wordlist *wl);
void com_savesnap(wordlist *wl);
#endif

#endif /* SPICEIF_H_INCLUDED */
