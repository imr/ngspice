/*************
 * Header file for inp.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_INP_H
#define ngspice_INP_H

void com_listing(wordlist *wl);
void com_edit(wordlist *wl);
void com_source(wordlist *wl);
void com_mc_source(wordlist *wl);
void com_circbyline(wordlist *wl);

void line_free_x(struct card *deck, bool recurse);
#define line_free(line, flag)                   \
    do {                                        \
        line_free_x(line, flag);                \
        line = NULL;                            \
    } while(0)

#endif
