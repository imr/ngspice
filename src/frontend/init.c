/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/* Initialize io, cp_chars[], variable "history". */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"

#include "init.h"
#include "variable.h"


char cp_chars[128]; /* used in fcn cp_lexer() from lexical.c */

static char *singlec = "<>;&";

void
cp_init(void)
/* called from ft_cpinit() in cpitf.c.
   Uses global variables:
   cp_chars[128]
   cp_maxhistlength (set to 10000 in com_history.c)
   cp_curin, cp_curout, cp_curerr (defined in streams.c)
*/
{
    char *s;

    bzero(cp_chars, 128);
    for (s = singlec; *s; s++)
        /* break word to right or left of characters <>;&*/
        cp_chars[(int) *s] = (CPC_BRR | CPC_BRL);

    cp_vset("history", CP_NUM, &cp_maxhistlength);

    cp_curin = stdin;
    cp_curout = stdout;
    cp_curerr = stderr;

    /* io redirection in streams.c:
       cp_in set to cp_curin etc. */
    cp_ioreset();
}
