/*************
 * Header file for inp.c
 * 1999 E. Rouat
 ************/

#ifndef INP_H_INCLUDED
#define INP_H_INCLUDED

void com_listing(wordlist *wl);
void inp_list(FILE *file, struct line *deck, struct line *extras, int type);
void inp_spsource(FILE *fp, bool comfile, char *filename);
void inp_dodeck(struct line *deck, char *tt, wordlist *end, bool reuse, 
		struct line *options, char *filename);
void com_edit(wordlist *wl);
void com_source(wordlist *wl);
void inp_source(char *file);


#endif
