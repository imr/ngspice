/*************
 * Header file for com_fft.c
 * 2008 H. Vogt
 ************/

#ifndef FFT_H_INCLUDED
#define FFT_H_INCLUDED

extern void free_pnode_o(struct pnode *t);

void com_fft(wordlist *wl);

static void fftext(float*, float*, long int, int);

#endif
