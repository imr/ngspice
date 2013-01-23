#ifndef ngspice_COM_DL_H
#define ngspice_COM_DL_H

#ifdef XSPICE
void com_codemodel(wordlist *wl);
#endif

#ifdef DEVLIB
void com_use(wordlist *wl);
#endif

#endif
