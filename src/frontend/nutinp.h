/*************
 * Header file for nutinp.c
 * 1999 E. Rouat
 ************/

#ifndef NUTINP_H_INCLUDED
#define NUTINP_H_INCLUDED

void inp_nutsource(FILE *fp, bool comfile, char *filename);
void nutcom_source(wordlist *wl);
void nutinp_source(char *file);
void nutinp_dodeck(struct line *deck, char *tt, wordlist *end, bool reuse, 
		   struct line *options, char *filename);



#endif
