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
    GENmodel *model = CKTfndMod(ckt, name);

    if (model) {
        *modfast = model;
        return E_EXISTS;
    }

    model = (GENmodel *) tmalloc((size_t) *(DEVices[type]->DEVmodSize));
    if (!model)
        return E_NOMEM;

    model->GENmodType = type;
    model->GENmodName = name;
    model->GENnextModel = ckt->CKThead[type];
    ckt->CKThead[type] = model;

    nghash_insert(ckt->MODnameHash, name, model);

    *modfast = model;

    return OK;
}
