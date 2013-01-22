#ifndef ngspice_COM_PLOT_H
#define ngspice_COM_PLOT_H

void com_plot(wordlist *wl);
#ifdef TCL_MODULE
void com_bltplot(wordlist *wl);
#endif
#endif
