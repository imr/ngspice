/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Thomas L. Quarles
**********/

#include "ngspice.h"
#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include <wordlist.h>
#include <bool.h>
#include <inpdefs.h>
#include <circuits.h>
#include <cpdefs.h>

#include "ifsim.h"
#include "iferrmsg.h"
#include "fteext.h"
#include "inp.h"


/* va: we should use tmalloc, whith also makes failure test */
int
IFnewUid(void *ckt, IFuid * newuid, IFuid olduid, char *suffix, int type,
	 void **nodedata)
{
    char *newname;
    int error;

    if (olduid) {
#ifdef HAVE_ASPRINTF    	
	asprintf(&newname, "%s#%s", (char *) olduid, suffix);
#else /* ~ HAVE_ASPRINTF */   
      newname = (char *) tmalloc(strlen((char *) olduid) +
				 strlen(suffix) + 2); /* 2 = strlen("#\0") */ 
      sprintf(newname, "%s#%s", (char *) olduid, suffix);
#endif /* HAVE_ASPRINTF */			 

    } else {
    	
#ifdef HAVE_ASPRINTF    	
	asprintf(&newname, "%s", suffix);
#else /* ~ HAVE_ASPRINTF */
      newname = (char *) tmalloc(strlen(suffix) + 1 );
      sprintf(newname, "%s", suffix);
#endif /* HAVE_ASPRINTF */ 
    }

    switch (type) {
    case UID_ANALYSIS:
    case UID_TASK:
    case UID_INSTANCE:
    case UID_OTHER:
    case UID_MODEL:
	error = INPinsert(&newname, (INPtables *) ft_curckt->ci_symtab);
	if (error && error != E_EXISTS)
	    return (error);
	*newuid = (IFuid) newname;
	break;

    case UID_SIGNAL:
	error = INPmkTerm(ckt, &newname,
			  (INPtables *) ft_curckt->ci_symtab, nodedata);
	if (error && error != E_EXISTS)
	    return (error);
	*newuid = (IFuid) newname;
	break;

    default:
	return (E_BADPARM);
    }
    return (OK);
}

int IFdelUid(void *ckt, IFuid uid, int type)
{
    int error;

    switch (type) {
    case UID_ANALYSIS:
    case UID_TASK:
    case UID_INSTANCE:
    case UID_OTHER:
    case UID_MODEL:
	error = INPremove(uid, (INPtables *) ft_curckt->ci_symtab);
	if (error && error != E_EXISTS)
	    return (error);
	break;

    case UID_SIGNAL:
	error = INPremTerm(uid, (INPtables *) ft_curckt->ci_symtab);
	if (error && error != E_EXISTS)
	    return (error);
	break;

    default:
	return (E_BADPARM);
    }
    return (OK);
}
