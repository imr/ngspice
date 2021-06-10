/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/

/*
 * A more portable version of the standard "mktemp( )" function
 *
 * FIXME: remove smktemp() and adjust all callers to use tmpfile(3).
 */

#include "ngspice/ngspice.h"
#include "mktemp.h"

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifndef TEMPFORMAT
#define TEMPFORMAT "temp%s%d"
#endif
#ifndef TEMPFORMAT2
#define TEMPFORMAT "temp%s%d_%d"
#endif


char *
smktemp(char *id)
{
    if (!id)
        id = "sp";
    const char* const home = getenv("HOME");
    if (home) {
        return tprintf("%s"TEMPFORMAT, home, id, getpid());
    }
    const char* const usr = getenv("USERPROFILE");
    if (usr) {
        return tprintf("%s\\"TEMPFORMAT, usr, id, getpid());
    }
    return tprintf(TEMPFORMAT, id, getpid());
}


char*
smktemp2(char* id, int n)
{
    if (!id)
        id = "sp";
    const char* const home = getenv("HOME");
    if (home) {
        return tprintf("%s"TEMPFORMAT2, home, id, getpid(), n);
    }
    const char* const usr = getenv("USERPROFILE");
    if (usr) {
        return tprintf("%s\\"TEMPFORMAT2, usr, id, getpid(), n);
    }
    return tprintf(TEMPFORMAT2, id, getpid(), n);
}

