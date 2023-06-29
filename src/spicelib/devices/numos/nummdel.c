/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author: 1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine deletes a NUMOS instance from the circuit and frees the
 * storage it was using.
 */

#include "ngspice/ngspice.h"
#include "numosdef.h"
#include "../../../ciderlib/twod/twoddefs.h"
#include "../../../ciderlib/twod/twodext.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/carddefs.h"


int
NUMOSdelete(GENinstance *gen_inst)
{
    NUMOSinstance *inst = (NUMOSinstance *) gen_inst;

    TWOdestroy(inst->NUMOSpDevice);

    return OK;
}

int NUMOSmodDelete(GENmodel *gen_model)
{
    NUMOSmodel *model = (NUMOSmodel *)gen_model;
    MESHcard *xmesh = model->NUMOSxMeshes;	/* list of xmesh cards */
    MESHcard *ymesh = model->NUMOSyMeshes;	/* list of ymesh cards */
    DOMNcard *domains = model->NUMOSdomains;	/* list of domain cards */
    BDRYcard *boundaries = model->NUMOSboundaries;	/* list of boundary cards */
    DOPcard *dopings = model->NUMOSdopings;	/* list of doping cards */
    ELCTcard *electrodes = model->NUMOSelectrodes;	/* list of electrode cards */
    CONTcard *contacts = model->NUMOScontacts;	/* list of contact cards */
    MODLcard *models = model->NUMOSmodels;	/* list of model cards */
    MATLcard *materials = model->NUMOSmaterials;	/* list of material cards */
    MOBcard *mobility = model->NUMOSmobility;	/* list of mobility cards */
    METHcard *methods = model->NUMOSmethods;	/* list of method cards */
    OPTNcard *options = model->NUMOSoptions;	/* list of option cards */
    OUTPcard *outputs = model->NUMOSoutputs;	/* list of output cards */
    TWOtranInfo *pInfo = model->NUMOSpInfo;	/* transient analysis information */
    DOPprofile *profiles = model->NUMOSprofiles;	/* expanded list of doping profiles */
    DOPtable *dopTables = model->NUMOSdopTables;	/* list of tables used by profiles */
    TWOmaterial *matlInfo = model->NUMOSmatlInfo;	/* list of material info structures */
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

