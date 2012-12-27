/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/*
 *  provide the error message appropriate for the given error code
 */

#include "ngspice/ngspice.h"

#ifdef HAVE_ASPRINTF
#ifdef HAVE_LIBIBERTY_H /* asprintf */
#include <libiberty.h>
#elif defined(__MINGW32__) || defined(__SUNPRO_C) /* we have asprintf, but not libiberty.h */
#include <stdarg.h>
extern int asprintf(char **out, const char *fmt, ...);
extern int vasprintf(char **out, const char *fmt, va_list ap);
#endif
#endif

#include "ngspice/fteext.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/sperror.h"
#include "inpxx.h"

char *INPerror(int type)
{
    char *val;
    char *ebuf;

    /*CDHW Lots of things set errMsg but it is never used  so let's hack it in CDHW*/
    if ( errMsg ) {
        val = errMsg;
        errMsg = NULL;
    } else
        /*CDHW end of hack CDHW*/
        val = copy(SPerror(type));

    if (!val)
        return NULL;

#ifdef HAVE_ASPRINTF
    if (errRtn)
        asprintf(&ebuf, "%s detected in routine \"%s\"\n", val, errRtn);
    else
        asprintf(&ebuf, "%s\n", val);
#else /* ~ HAVE_ASPRINTF */
    if (errRtn) {
        ebuf = TMALLOC(char, strlen(val) + strlen(errRtn) + 25);
        sprintf(ebuf, "%s detected in routine \"%s\"\n", val, errRtn);
    } else {
        ebuf = TMALLOC(char, strlen(val) + 2);
        sprintf(ebuf, "%s\n", val);
    }
#endif /* HAVE_ASPRINTF */
    tfree(val);
    return ebuf;
}
