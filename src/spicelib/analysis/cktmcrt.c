/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

    /* CKTmodCrt(type,name,ckt,fast)
     *  Create a device model of the specified type, with the given name
     *  in the named circuit.
     */

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/sperror.h"



int
CKTmodCrt(CKTcircuit *ckt, int type, GENmodel **modfast, IFuid name)
{
    GENmodel *mymodfast = NULL;

    // assert(second)
    // assert(third && *third == NULL)
    mymodfast = CKTfndMod(ckt, name);
    if(!mymodfast) {
        mymodfast = (GENmodel *) tmalloc((size_t) *(DEVices[type]->DEVmodSize));
        if(mymodfast == NULL) return(E_NOMEM);
        mymodfast->GENmodType = type;
        mymodfast->GENmodName = name;
        mymodfast->GENnextModel = ckt->CKThead[type];
        ckt->CKThead[type] = mymodfast;
        if(modfast) *modfast=mymodfast;
        nghash_insert(ckt->MODnameHash, name, mymodfast);
        return(OK);
    } else {
        if(modfast) *modfast=mymodfast;
        return(E_EXISTS);
    }
    /*NOTREACHED*/
}
