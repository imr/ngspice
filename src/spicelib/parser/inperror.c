/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/* 
 *  provide the error message appropriate for the given error code
 */

#include "ngspice.h"
#include <stdio.h>
#ifdef HAVE_STRING_H
#include <string.h>
#endif
#include "fteext.h"
#include "ifsim.h"
#include "iferrmsg.h"
#include "sperror.h"
#include "inp.h"

char *INPerror(int type)
{
    char *val;
    char *ebuf;

    val = SPerror(type);

    if (!val)
	return (val);

#ifdef HAVE_ASPRINTF
    if (errRtn)
	asprintf(&ebuf, "%s detected in routine \"%s\"\n", val, errRtn);
    else
	asprintf(&ebuf, "%s\n", val);
#else /* ~ HAVE_ASPRINTF */
    if (errRtn){
      if ( (ebuf = (char *) malloc(strlen(val) +
				   strlen(errRtn) + 25)) == NULL){
        fprintf(stderr,"malloc failed\n");
        exit(1);
      }
      sprintf(ebuf, "%s detected in routine \"%s\"\n", val, errRtn);
    }
    else{
      if ( (ebuf = (char *) malloc(strlen(val) + 2)) == NULL){
        fprintf(stderr,"malloc failed\n");
        exit(1);
      }
      sprintf(ebuf, "%s\n", val);
    }
#endif /* HAVE_ASPRINTF */
    return ebuf;
}
