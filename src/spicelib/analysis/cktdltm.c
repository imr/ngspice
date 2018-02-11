/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/* CKTdltMod
 *  delete the specified model - not yet supported in spice 
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"



/* ARGSUSED */
int
CKTdltMod(CKTcircuit *ckt, GENmodel *m)
{
    GENmodel *mod, **prevp;
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
            if (h != nghash_delete(ckt->DEVnameHash, h->GENname))
                fprintf(stderr, "ERROR, ouch nasal daemons ...\n");
	    error = SPfrontEnd->IFdelUid (ckt, h->GENname,
		    UID_INSTANCE);
	    GENinstanceFree(h);
    }
    if (m != nghash_delete(ckt->MODnameHash, m->GENmodName))
        fprintf(stderr, "ERROR, ouch nasal daemons ...\n");
    error = SPfrontEnd->IFdelUid (ckt, m->GENmodName, UID_MODEL);
    GENmodelFree(m);
    return(OK);
}
