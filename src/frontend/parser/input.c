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


/* A special 'getc' so that we can deal with ^D properly. There is no way for
 * stdio to know if we have typed a ^D after some other characters, so
 * don't use buffering at all
 */

int
inchar(FILE *fp)
{

#if !(defined(HAS_WINGUI) || defined(_MSC_VER) || defined(__MINGW32__))
    if (cp_interactive && !cp_nocc) {
        char c;
        ssize_t i;

        do
            i = read(fileno(fp), &c, 1);
        while (i == -1 && errno == EINTR);

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
