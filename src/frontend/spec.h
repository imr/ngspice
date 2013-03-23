/*************
 * Header file for spec.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_SPEC_H
#define ngspice_SPEC_H

#ifdef HAS_WINGUI
extern void SetAnalyse(char *Analyse, int Percent);
#endif

void com_spec(wordlist *wl);

#endif
