#include <string.h>

#include "ngspice/ifsim.h"
#include "ngspice/devdefs.h"
#include "ngspice/gendefs.h"

/* Modified copy from spiceif.c.  FIX ME! */

static IFparm *parmlookup(IFdevice *dev, char *param)
{
    IFparm *p;
    int     i;

    if (dev->numInstanceParms) {
        for (i = 0; i < *(dev->numInstanceParms); i++) {
            p = dev->instanceParms + i;
            if ((p->dataType & IF_SET) && !strcmp(p->keyword, param)) {
                while ((p->dataType & IF_REDUNDANT) && i > 0)
                    i--;
                if ((p->dataType & IF_VARTYPES) != IF_REAL)
                    return NULL;
                return p;
            }
        }
    }

    if (dev->numModelParms) {
        for (i = 0; i < *(dev->numModelParms); i++) {
            p = dev->modelParms + i;
            if ((p->dataType & IF_SET) && !strcmp(p->keyword, param)) {
                while ((p->dataType & IF_REDUNDANT) && i > 0)
                    i--;
                if ((p->dataType & IF_VARTYPES) != IF_REAL)
                    return NULL;
                return p;
            }
        }
    }
    return NULL;
}

void ucm_spider(ARGS)
{
    CKTcircuit  *ckt;
    SPICEdev    *sdev;
    IFdevice    *ifdev;
    GENmodel    *mod;
    GENinstance *dev;
    IFparm      *p;
    char        *what;
    IFvalue      nval;
    double       v;
    int          i;

    if (INIT) {
//        cm_irreversible(1);
        return;
    }
    ckt = mif_private->ckt;
    what = PARAM(parameter);
    nval.rValue = INPUT(value);

    if (!strcmp(what, "temp")) {
        /* Set the overall circuit temperature. */

        ckt->CKTtemp = nval.rValue;
    }

    for (i = 0; i < mif_private->devices; ++i) {
        ifdev = mif_private->ifdevs[i];
        p = parmlookup(ifdev, what); // Cache me!
        if (!p)
            continue;
        sdev = mif_private->sdevs[i];
        for (mod = ckt->CKThead[i]; mod; mod = mod->GENnextModel) {
            for (dev = mod->GENinstances; dev; dev = dev->GENnextInstance) {
                /* Try setting the named parameter on this device.
                 * Modelled on cktparam.c.
                 */

                sdev->DEVparam(p->id, &nval, dev, NULL);
            }
        }
    }
}




