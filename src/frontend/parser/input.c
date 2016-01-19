/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Jeffrey M. Hsu
**********/

/*
 * Stand-alone input routine.
 */
#include "ngspice/ngspice.h"

#include <errno.h>

#include "ngspice/fteinput.h"
#include "input.h"
#include "ngspice/cpextern.h"
#include "../display.h"
#ifdef _MSC_VER
#include "BaseTsd.h" /* for SSIZE_T */
#define ssize_t SSIZE_T
#ifndef HAS_WINGUI
#define read _read /* only for console */
#endif
#endif


/* A special 'getc' so that we can deal with ^D properly. There is no way for
 * stdio to know if we have typed a ^D after some other characters, so
 * don't use buffering at all
 */

int
inchar(FILE *fp)
{

#ifndef HAS_WINGUI
    if (cp_interactive && !cp_nocc) {
        char c;
        ssize_t i;

        do
            i = read(fileno(fp), &c, 1);
        while (i == -1 && errno == EINTR);

#if _MSC_VER == 1900
        /* There is a bug in MS VS2015, in that _read returns 0 instead of 1
           if '\n' is read from stdin in text mode */
        if (i == 0 && c == '\n')
            i = 1;
#endif

        if (i == 0 || c == '\004')
            return EOF;

        if (i == -1) {
            perror("read");
            return EOF;
        }

        return (int) c;
    }
#endif

    return getc(fp);
}


int
input(FILE *fp)
{
    REQUEST request;
    RESPONSE response;

    request.option = char_option;
    request.fp = fp;

    Input(&request, &response);

    return (inchar(fp));
}
