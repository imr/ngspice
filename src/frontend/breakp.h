/*************
 * Header file for breakp.c
 * 1999 E. Rouat
 ************/

#ifndef BREAKP_H_INCLUDED
#define BREAKP_H_INCLUDED

void com_stop(wordlist *wl);
void com_trce(wordlist *wl);
void com_iplot(wordlist *wl);
void com_step(wordlist *wl);
void com_sttus(wordlist *wl);
void dbfree(struct dbcomm *db);
void com_delete(wordlist *wl);
bool ft_bpcheck(struct plot *runplot, int iteration);
void ft_trquery(void);




#endif
