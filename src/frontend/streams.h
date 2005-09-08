/*************
* Header file for streams.c
* $Id$
************/

#ifndef STREAMS_H
#define STREAMS_H

#include <bool.h>
#include <wordlist.h>

extern bool cp_debug;
extern char cp_amp;
extern char cp_gt;
extern char cp_lt;
FILE *cp_in;
FILE *cp_out;
FILE *cp_err;
FILE *cp_curin;
FILE *cp_curout;
FILE *cp_curerr;

void cp_ioreset(void);
void fixdescriptors(void);
wordlist * cp_redirect(wordlist *wl);

#endif /* STREAMS_H */
