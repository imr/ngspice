/*
 * numpaif.h
 * external interface to spice frontend  subckt.c 
 * $Id$
 */

#ifndef NUMPAIF_H
#define NUMPAIF_H

#define  NUPADECKCOPY 0
#define  NUPASUBSTART 1
#define  NUPASUBDONE  2
#define  NUPAEVALDONE 3

extern char * nupa_copy(char *s, int linenum);
extern int    nupa_eval(char *s, int linenum);
extern int    nupa_signal(int sig, char *info);
extern void   nupa_scan(char * s, int linenum);

#endif /* NUMPAIF_H */
