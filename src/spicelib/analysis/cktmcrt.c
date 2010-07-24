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

#include "ngspice.h"
#include "devdefs.h"
#include "cktdefs.h"
#include "sperror.h"



int
CKTmodCrt(CKTcircuit *ckt, int type, GENmodel **modfast, IFuid name)
{
    GENmodel *mymodfast = NULL;
    int error;

    error = CKTfndMod(ckt, &type, &mymodfast, name);
    if(error == E_NOMOD) {
        mymodfast = (GENmodel *)MALLOC(*(DEVices[type]->DEVmodSize));
        if(mymodfast == (GENmodel *)NULL) return(E_NOMEM);
        mymodfast->GENmodType = type;
        mymodfast->GENmodName = name;
        mymodfast->GENnextModel =(GENmodel *)(ckt->CKThead[type]);
        ckt->CKThead[type]=(GENmodel *)mymodfast;
        if(modfast) *modfast=mymodfast;
        return(OK);
    } else if (error==0) {
        if(modfast) *modfast=mymodfast;
        return(E_EXISTS);
    } else {
        return(error);
    }
    /*NOTREACHED*/
}
