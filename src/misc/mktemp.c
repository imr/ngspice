/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/

/*
 * A more portable version of the standard "mktemp( )" function
 *
 * FIXME: remove smktemp() and adjust all callers to use tmpfile(3).
 */

#include "ngspice.h"
#include <stdio.h>
#include "mktemp.h"

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifndef TEMPFORMAT
#define TEMPFORMAT "temp%s%d"
#endif

char *
smktemp(char *id)
{
    char	rbuf[513];
    char	*nbuf;
    int		num;


    num = getpid( );


    if (!id)
	id = "sp";

    sprintf(rbuf, TEMPFORMAT, id, num);
    nbuf = (char *) tmalloc(strlen(rbuf) + 1);
    strcpy(nbuf, rbuf);

    return nbuf;
}
