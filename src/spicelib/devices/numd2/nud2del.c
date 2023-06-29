/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author: 1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "numd2def.h"
#include "../../../ciderlib/twod/twoddefs.h"
#include "../../../ciderlib/twod/twodext.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/carddefs.h"


int
NUMD2delete(GENinstance *gen_inst)
{
    NUMD2instance *inst = (NUMD2instance *) gen_inst;

    TWOdestroy(inst->NUMD2pDevice);

    return OK;
}

int NUMD2modDelete(GENmodel *gen_model)
{
    NUMD2model *model = (NUMD2model *)gen_model;
    MESHcard *xmesh = model->NUMD2xMeshes;	/* list of xmesh cards */
    MESHcard *ymesh = model->NUMD2yMeshes;	/* list of ymesh cards */
    DOMNcard *domains = model->NUMD2domains;	/* list of domain cards */
    BDRYcard *boundaries = model->NUMD2boundaries;	/* list of boundary cards */
    DOPcard *dopings = model->NUMD2dopings;	/* list of doping cards */
    ELCTcard *electrodes = model->NUMD2electrodes;	/* list of electrode cards */
    CONTcard *contacts = model->NUMD2contacts;	/* list of contact cards */
    MODLcard *models = model->NUMD2models;	/* list of model cards */
    MATLcard *materials = model->NUMD2materials;	/* list of material cards */
    MOBcard *mobility = model->NUMD2mobility;	/* list of mobility cards */
    METHcard *methods = model->NUMD2methods;	/* list of method cards */
    OPTNcard *options = model->NUMD2options;	/* list of option cards */
    OUTPcard *outputs = model->NUMD2outputs;	/* list of output cards */
    TWOtranInfo *pInfo = model->NUMD2pInfo;	/* transient analysis information */
    DOPprofile *profiles = model->NUMD2profiles;	/* expanded list of doping profiles */
    DOPtable *dopTables = model->NUMD2dopTables;	/* list of tables used by profiles */
    TWOmaterial *matlInfo = model->NUMD2matlInfo;	/* list of material info structures */
    {
        MESHcard *next = NULL, *this = NULL;
        next = xmesh;
        while (next) {
            this = next;
            next = next->MESHnextCard;
            FREE(this);
        }
    }
    {
        MESHcard *next = NULL, *this = NULL;
        next = ymesh;
        while (next) {
            this = next;
            next = next->MESHnextCard;
            FREE(this);
        }
    }
    {
        DOMNcard *next = NULL, *this = NULL;
        next = domains;
        while (next) {
            this = next;
            next = next->DOMNnextCard;
            FREE(this);
        }
    }
    {
        BDRYcard *next = NULL, *this = NULL;
        next = boundaries;
        while (next) {
            this = next;
            next = next->BDRYnextCard;
            FREE(this);
        }
    }
    {
        DOPcard *next = NULL, *this = NULL;
        next = dopings;
        while (next) {
            this = next;
            next = next->DOPnextCard;
            if (this->DOPdomains) {
                FREE(this->DOPdomains);
            }
            if (this->DOPinFile) {
                FREE(this->DOPinFile);
            }
            FREE(this);
        }
    }
    {
        ELCTcard *next = NULL, *this = NULL;
        next = electrodes;
        while (next) {
            this = next;
            next = next->ELCTnextCard;
            FREE(this);
        }
    }
    {
        CONTcard *next = NULL, *this = NULL;
        next = contacts;
        while (next) {
            this = next;
            next = next->CONTnextCard;
            FREE(this);
        }
    }
    {
        MODLcard *next = NULL, *this = NULL;
        next = models;
        while (next) {
            this = next;
            next = next->MODLnextCard;
            FREE(this);
        }
    }
    {
        MATLcard *next = NULL, *this = NULL;
        next = materials;
        while (next) {
            this = next;
            next = next->MATLnextCard;
            FREE(this);
        }
    }
    {
        MOBcard *next = NULL, *this = NULL;
        next = mobility;
        while (next) {
            this = next;
            next = next->MOBnextCard;
            FREE(this);
        }
    }
    {
        METHcard *next = NULL, *this = NULL;
        next = methods;
        while (next) {
            this = next;
            next = next->METHnextCard;
            FREE(this);
        }
    }
    {
        OPTNcard *next = NULL, *this = NULL;
        next = options;
        while (next) {
            this = next;
            next = next->OPTNnextCard;
            FREE(this);
        }
    }
    {
        OUTPcard *next = NULL, *this = NULL;
        next = outputs;
        while (next) {
            this = next;
            next = next->OUTPnextCard;
            if (this->OUTProotFile) {
                FREE(this->OUTProotFile);
            }
            FREE(this);
        }
    }
    {
        if (pInfo) {
            FREE(pInfo);
        }
    }
    {
        DOPprofile *next = NULL, *this = NULL;
        next = profiles;
        while (next) {
            this = next;
            next = next->next;
            FREE(this);
        }
    }
    (void)dopTables;
    {
        TWOmaterial *next = NULL, *this = NULL;
        next = matlInfo;
        while (next) {
            this = next;
            next = next->next;
            FREE(this);
        }
    }

    return OK;
}
