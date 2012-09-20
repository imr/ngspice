/*************
 * Header file for spec.c
 * 1999 E. Rouat
 ************/

#ifndef SPEC_H_INCLUDED
#define SPEC_H_INCLUDED

#ifdef HAS_WINDOWS
extern void SetAnalyse(char *Analyse, int Percent);
#endif

void com_spec(wordlist *wl);

#endif
