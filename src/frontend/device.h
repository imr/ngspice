/*************
 * Header file for device.c
 * 1999 E. Rouat
 * Modified: 2000 AlansFixes
 ************/

#ifndef DEVICE_H_INCLUDED
#define DEVICE_H_INCLUDED

#define LEFT_WIDTH 11
#define DEV_WIDTH 21

void com_showmod(wordlist *wl);
void com_show(wordlist *wl);
int printstr(dgen *dg, char *name);
void  param_forall(dgen *dg, int flags);
void  listparam(wordlist *p, dgen *dg);
int bogus1(dgen *dg);
int bogus2(dgen *dg);
int printvals(dgen *dg, IFparm *p, int i);
void old_show(wordlist *wl);
void com_alter(wordlist *wl);
void com_altermod(wordlist *wl);
void  com_alter_common(wordlist *wl, int do_model);




#endif
