/*************
 * Header file for unixcom.c
 * 1999 E. Rouat
 ************/

#ifndef UNIXCOM_H_INCLUDED
#define UNIXCOM_H_INCLUDED


void cp_rehash(char *pathlist, bool docc);
bool cp_unixcom(wordlist *wl);
void cp_hstat(void);



#endif
