/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/* Initialize stuff. */

#include <ngspice.h>
#include <cpdefs.h>

#include "init.h"
#include "variable.h"


char cp_chars[128];

static char *singlec = "<>;&";

void
cp_init(void)
{
    char *s, *getenv(const char *);

    bzero(cp_chars, 128);
    for (s = singlec; *s; s++)
        cp_chars[(int) *s] = (CPC_BRR | CPC_BRL);
    cp_vset("history", VT_NUM, (char *) &cp_maxhistlength);

    cp_curin = stdin;
    cp_curout = stdout;
    cp_curerr = stderr;

    cp_ioreset();

    return;
}
