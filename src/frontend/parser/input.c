/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Jeffrey M. Hsu
**********/

/*
 * Stand-alone input routine.
 */
#include <config.h>
#include <ngspice.h>

#include <errno.h>

#include "fteinput.h"
#include "input.h"
#include "cpextern.h"

/* A special 'getc' so that we can deal with ^D properly. There is no way for
 * stdio to know if we have typed a ^D after some other characters, so
 * don't use buffering at all
 */
int
inchar(FILE *fp)
{

    char c;
#ifndef HAS_WINDOWS    
    int i;

    if (cp_interactive && !cp_nocc) {
      do {
	i = read((int) fileno(fp), &c, 1);
	} while (i == -1 && errno == EINTR);
      if (i == 0 || c == '\004')
        return (EOF);
      else if (i == -1) {
        perror("read");
        return (EOF);
      } else
        return ((int) c);
    } else
#endif    
    c = getc(fp);
    return ((int) c);
}


extern void Input(REQUEST *req, RESPONSE *resp);

int 
input(FILE *fp)
{

    REQUEST request;
    RESPONSE response;

    request.option = char_option;
    request.fp = fp;
    Input(&request, &response);
    return(response.reply.ch);

}
