/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author: 1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine deletes a NBJT2 instance from the circuit and frees the
 * storage it was using.
 */

#include "ngspice/ngspice.h"
#include "nbjt2def.h"
#include "../../../ciderlib/twod/twoddefs.h"
#include "../../../ciderlib/twod/twodext.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/carddefs.h"


int
NBJT2delete(GENinstance *gen_inst)
{
    NBJT2instance *inst = (NBJT2instance *) gen_inst;

    TWOdestroy(inst->NBJT2pDevice);

    return OK;
}

int NBJT2modDelete(GENmodel *gen_model)
{
    NBJT2model *model = (NBJT2model *)gen_model;
    MESHcard *xmesh = model->NBJT2xMeshes;	/* list of xmesh cards */
    MESHcard *ymesh = model->NBJT2yMeshes;	/* list of ymesh cards */
    DOMNcard *domains = model->NBJT2domains;	/* list of domain cards */
    BDRYcard *boundaries = model->NBJT2boundaries;	/* list of boundary cards */
    DOPcard *dopings = model->NBJT2dopings;	/* list of doping cards */
    ELCTcard *electrodes = model->NBJT2electrodes;	/* list of electrode cards */
    CONTcard *contacts = model->NBJT2contacts;	/* list of contact cards */
    MODLcard *models = model->NBJT2models;	/* list of model cards */
    MATLcard *materials = model->NBJT2materials;	/* list of material cards */
    MOBcard *mobility = model->NBJT2mobility;	/* list of mobility cards */
    METHcard *methods = model->NBJT2methods;	/* list of method cards */
    OPTNcard *options = model->NBJT2options;	/* list of option cards */
    OUTPcard *outputs = model->NBJT2outputs;	/* list of output cards */
    TWOtranInfo *pInfo = model->NBJT2pInfo;	/* transient analysis information */
    DOPprofile *profiles = model->NBJT2profiles;	/* expanded list of doping profiles */
    DOPtable *dopTables = model->NBJT2dopTables;	/* list of tables used by profiles */
    TWOmaterial *matlInfo = model->NBJT2matlInfo;	/* list of material info structures */
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
