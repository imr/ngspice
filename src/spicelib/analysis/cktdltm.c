/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/* CKTdltMod
 *  delete the specified model - not yet supported in spice 
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "ifsim.h"
#include "sperror.h"



/* ARGSUSED */
int
CKTdltMod(void *cktp, void *modPtr)
{
    CKTcircuit *ckt = (CKTcircuit *) cktp;
    GENmodel *m = (GENmodel *) modPtr, *mod, **prevp;
    GENinstance *h, *next_i;
    int	error;

    prevp = &ckt->CKThead[m->GENmodType];
    for (mod = *prevp; m && mod != m; mod = mod->GENnextModel)
	prevp = &mod->GENnextModel;

    if (!mod)
	return OK;

    *prevp = m->GENnextModel;

    for (h = m->GENinstances; h; h = next_i) {
	    next_i = h->GENnextInstance;
	    error = (*(SPfrontEnd->IFdelUid))((void *)ckt,h->GENname,
		    UID_INSTANCE);
	    tfree(h);
    }
    error = (*(SPfrontEnd->IFdelUid))((void *)ckt,m->GENmodName, UID_MODEL);
    tfree(m);
    return(OK);
}
