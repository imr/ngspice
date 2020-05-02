/*************
 * Header file for postcoms.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_POSTCOMS_H
#define ngspice_POSTCOMS_H

void com_unlet(wordlist *wl);
void com_load(wordlist *wl);
void com_print(wordlist *wl);
void com_write(wordlist *wl);
void com_write_sparam(wordlist *wl);
void com_transpose(wordlist *wl);
void com_cross(wordlist *wl);
void com_destroy(wordlist *wl);
void com_splot(wordlist *wl);
void com_remzerovec(wordlist* wl);

void destroy_const_plot(void);


#endif
