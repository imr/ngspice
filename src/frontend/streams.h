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
extern FILE *cp_in;
extern FILE *cp_out;
extern FILE *cp_err;
extern FILE *cp_curin;
extern FILE *cp_curout;
extern FILE *cp_curerr;

void cp_ioreset(void);
void fixdescriptors(void);
wordlist * cp_redirect(wordlist *wl);

#endif /* STREAMS_H */
