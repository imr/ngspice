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

struct nscope;

extern char  *nupa_copy(char *s, int linenum);
extern int    nupa_eval(char *s, int linenum, int orig_linenum);
extern int    nupa_signal(int sig, char *info);
extern void   nupa_scan(char * s, int linenum, int is_subckt, struct nscope *level);
extern void   nupa_list_params(FILE *cp_out);
extern double nupa_get_param(char *param_name, int *found);
extern void   nupa_add_param(char *param_name, double value);
extern void   nupa_add_inst_param(char *param_name, double value);
extern void   nupa_copy_inst_dico(void);
extern void   nupa_del_dicoS(void);
extern int    nupa_add_dicoslist(void);
extern void   nupa_rem_dicoslist(int);
extern void   nupa_set_dicoslist(int);

extern int dynMaxckt; /* number of lines in deck after expansion */

#endif
