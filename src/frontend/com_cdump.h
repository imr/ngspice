#ifndef ngspice_COM_CDUMP_H
#define ngspice_COM_CDUMP_H

void com_cdump(wordlist *wl);
void com_mdump(wordlist *wl);
void com_rdump(wordlist *wl);
#define TABINDENT 2 /* CDHW */  /* The orginal value was 8 */

#endif
