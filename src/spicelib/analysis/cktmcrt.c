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
CKTmodCrt(void *ckt, int type, void **modfast, IFuid name)
{
    extern SPICEdev **DEVices;
    GENmodel *mymodfast[1] = { NULL };
    int error;

    error = CKTfndMod (ckt, &type, (void**)mymodfast, name);
    if(error == E_NOMOD) {
        mymodfast[0] = (GENmodel *)MALLOC(*(DEVices[type]->DEVmodSize));
        if(mymodfast[0] == (GENmodel *)NULL) return(E_NOMEM);
        mymodfast[0]->GENmodType = type;
        mymodfast[0]->GENmodName = name;
        mymodfast[0]->GENnextModel =(GENmodel *)((CKTcircuit *)ckt)->CKThead[type];
        ((CKTcircuit *)ckt)->CKThead[type]=(GENmodel *)mymodfast[0];
        if(modfast) *modfast=(void *)mymodfast[0];
        return(OK);
    } else if (error==0) {
        if(modfast) *modfast=(void *)mymodfast[0];
        return(E_EXISTS);
    } else {
        return(error);
    }
    /*NOTREACHED*/
}
