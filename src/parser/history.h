/*************
 * Header file for history.c
 * 1999 E. Rouat
 ************/

#ifndef HISTORY_H_INCLUDED
#define HISTORY_H_INCLUDED

wordlist * cp_histsubst(wordlist *wlist);
void cp_addhistent(int event, wordlist *wlist);
void cp_hprint(int eventhi, int eventlo, bool rev);
void com_history(wordlist *wl);



#endif
