/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Thomas L. Quarles
**********/

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

#include "ngspice/wordlist.h"
#include "ngspice/bool.h"
#include "ngspice/inpdefs.h"
#include <circuits.h>
#include "ngspice/cpdefs.h"

#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/fteext.h"
#include "inpxx.h"


/* va: we should use tmalloc, whith also makes failure test */
int
IFnewUid(CKTcircuit *ckt, IFuid * newuid, IFuid olduid, char *suffix, int type,
	 CKTnode **nodedata)
{
    char *newname;
    int error;

    if (olduid) {
#ifdef HAVE_ASPRINTF    	
	asprintf(&newname, "%s#%s", olduid, suffix);
#else /* ~ HAVE_ASPRINTF */   
      newname = TMALLOC(char, strlen(olduid) + strlen(suffix) + 2); /* 2 = strlen("#\0") */ 
      sprintf(newname, "%s#%s", olduid, suffix);
#endif /* HAVE_ASPRINTF */			 

    } else {
    	
#ifdef HAVE_ASPRINTF    	
	asprintf(&newname, "%s", suffix);
#else /* ~ HAVE_ASPRINTF */
      newname = TMALLOC(char, strlen(suffix) + 1);
      sprintf(newname, "%s", suffix);
#endif /* HAVE_ASPRINTF */ 
    }

    switch (type) {
    case UID_ANALYSIS:
    case UID_TASK:
    case UID_INSTANCE:
    case UID_OTHER:
    case UID_MODEL:
	error = INPinsert(&newname, ft_curckt->ci_symtab);
	if (error && error != E_EXISTS)
	    return (error);
	*newuid = newname;
	break;

    case UID_SIGNAL:
	error = INPmkTerm(ckt, &newname,
			  ft_curckt->ci_symtab, nodedata);
	if (error && error != E_EXISTS)
	    return (error);
	*newuid = newname;
	break;

    default:
	return (E_BADPARM);
    }
    return (OK);
}

int IFdelUid(CKTcircuit *ckt, IFuid uid, int type)
{
    int error;

    NG_IGNORE(ckt);

    switch (type) {
    case UID_ANALYSIS:
    case UID_TASK:
    case UID_INSTANCE:
    case UID_OTHER:
    case UID_MODEL:
	error = INPremove(uid, ft_curckt->ci_symtab);
	if (error && error != E_EXISTS)
	    return (error);
	break;

    case UID_SIGNAL:
	error = INPremTerm(uid, ft_curckt->ci_symtab);
	if (error && error != E_EXISTS)
	    return (error);
	break;

    default:
	return (E_BADPARM);
    }
    return (OK);
}
