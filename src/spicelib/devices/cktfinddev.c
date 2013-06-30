/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/config.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"
#include "string.h"


static GENinstance *
find_instance(GENinstance *here, IFuid name)
{
    for (; here; here = here->GENnextInstance)
        if (here->GENname == name)
            return here;

    return NULL;
}


int
CKTfndDev(CKTcircuit *ckt, int *type, GENinstance **fast, IFuid name, GENmodel *modfast)
{
    GENinstance *here;
    GENmodel *mods;

    /* we know the device instance `fast' */
    if (fast && *fast) {
        if (type)
            *type = (*fast)->GENmodPtr->GENmodType;
        return OK;
    }

    /* we know the model `modfast', but need to find the device instance */
    if (modfast) {
        here = find_instance(modfast->GENinstances, name);
        if (here) {
            if (fast)
                *fast = here;

            if (type)
                *type = modfast->GENmodType;

            return OK;
        }
        return E_NODEV;
    }

    /* we know device `type', but need to find model and device instance */
    if (*type >= 0 && *type < DEVmaxnum) {
        /* look through all models */
        for (mods = ckt->CKThead[*type]; mods ; mods = mods->GENnextModel) {
            /* and all instances */
                here = find_instance(mods->GENinstances, name);
                if (here) {
                    if (fast)
                        *fast = here;
                    return OK;
                }
                if (mods->GENmodName == NULL)
                    return E_NODEV;
        }
        return E_NOMOD;
    }

    /* we don't even know `type', search all of them */
    if (*type == -1) {
        for (*type = 0; *type < DEVmaxnum; (*type)++) {
            /* look through all models */
            for (mods = ckt->CKThead[*type]; mods; mods = mods->GENnextModel) {
                /* and all instances */
                    here = find_instance(mods->GENinstances, name);
                    if (here) {
                        if (fast)
                            *fast = here;
                        return OK;
                    }
                    if (mods->GENmodName == NULL)
                        return E_NODEV;
            }
        }
        *type = -1;
        return E_NODEV;
    }

    return E_BADPARM;
}
