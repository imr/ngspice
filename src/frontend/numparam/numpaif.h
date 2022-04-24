/*
 * numpaif.h
 * external interface to spice frontend  subckt.c
 */

#ifndef ngspice_NUMPAIF_H
#define ngspice_NUMPAIF_H

#define  NUPADECKCOPY 0
#define  NUPASUBSTART 1
#define  NUPASUBDONE  2
#define  NUPAEVALDONE 3

struct card;

extern char  *nupa_copy(struct card *c);
extern int    nupa_eval(struct card *card);
extern void   nupa_signal(int sig);
extern void   nupa_scan(const struct card *card);
extern void   nupa_list_params(FILE *cp_out);
extern double nupa_get_param(const char *param_name, int *found);
extern const char *nupa_get_string_param(const char *param_name);
extern void   nupa_add_param(char *param_name, double value);
extern void   nupa_copy_inst_dico(void);
extern void   nupa_del_dicoS(void);
extern int    nupa_add_dicoslist(void);
extern void   nupa_rem_dicoslist(int);
extern void   nupa_set_dicoslist(int);

extern int dynMaxckt; /* number of lines in deck after expansion */

#endif
