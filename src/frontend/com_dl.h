#ifndef _COM_DL_H
#define _COM_DL_H 1 

#ifdef XSPICE
void com_codemodel(wordlist *wl);
#endif

#ifdef DEVLIB
void com_use(wordlist *wl);
#endif

#endif
