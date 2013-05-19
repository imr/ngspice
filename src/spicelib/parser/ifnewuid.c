/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"

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

    if (olduid)
        newname = tprintf("%s#%s", olduid, suffix);
    else
        newname = tprintf("%s", suffix);

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
