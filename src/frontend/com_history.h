/*************
 * Header file for history.c
 * 1999 E. Rouat
 ************/

#ifndef _COM_HISTORY_H
#define _COM_HISTORY_H

wordlist * cp_histsubst(wordlist *wlist);
void cp_addhistent(int event, wordlist *wlist);
void cp_hprint(int eventhi, int eventlo, bool rev);
void com_history(wordlist *wl);



#endif
