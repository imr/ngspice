/*
 * numpaif.h
 * external interface to spice frontend  subckt.c
 */

#ifndef NUMPAIF_H
#define NUMPAIF_H

#define  NUPADECKCOPY 0
#define  NUPASUBSTART 1
#define  NUPASUBDONE  2
#define  NUPAEVALDONE 3

extern char  *nupa_copy(char *s, int linenum);
extern int    nupa_eval(char *s, int linenum, int orig_linenum);
extern int    nupa_signal(int sig, char *info);
extern void   nupa_scan(char * s, int linenum, int is_subckt);
extern void   nupa_list_params(FILE *cp_out);
extern double nupa_get_param(char *param_name, int *found);
extern void   nupa_add_param(char *param_name, double value);
extern void   nupa_add_inst_param(char *param_name, double value);
extern void   nupa_copy_inst_dico(void);

extern int dynMaxckt; /* number of lines in deck after expansion */

#endif
