/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Jeffrey M. Hsu
**********/

/*
 * Stand-alone input routine.
 */

#include "ngspice.h"
#include "fteinput.h"
#include "input.h"


extern int inchar(FILE *fp);

void
Input(REQUEST *request, RESPONSE *response)
{

    switch (request->option) {
      case char_option:
        response->reply.ch = inchar(request->fp);
        response->option = request->option;
        break;
      default:
        /* just ignore, since we don't want a million error messages */
        response->option = error_option;
        break;
    }
    return;

}
