/*************
 * Header file for spiceif.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_SPICEIF_H
#define ngspice_SPICEIF_H

CKTcircuit * if_inpdeck(struct card *deck, INPtables **tab);
int if_run(CKTcircuit *t, char *what, wordlist *args, INPtables *tab);
int if_option(CKTcircuit *ckt, char *name, enum cp_types type, void *value);
void if_dump(CKTcircuit *ckt, FILE *file);
void if_cktfree(CKTcircuit *ckt, INPtables *tab);
int  if_analQbyName(CKTcircuit *ckt, int which, JOB *anal, char *name, IFvalue *parm);

void com_snload(wordlist *wl);
void com_snsave(wordlist *wl);

#endif
