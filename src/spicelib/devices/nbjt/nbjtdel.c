/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author: 1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine deletes a NBJT instance from the circuit and frees the
 * storage it was using.
 */

#include "ngspice/ngspice.h"
#include "nbjtdefs.h"
#include "../../../ciderlib/oned/onedext.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
NBJTdelete(GENinstance *gen_inst)
{
    NBJTinstance *inst = (NBJTinstance *) gen_inst;

    ONEdestroy(inst->NBJTpDevice);

    return OK;
}

int NBJTmodDelete(GENmodel *gen_model)
{
    NBJTmodel *model = (NBJTmodel *)gen_model;
    MESHcard *xmesh = model->NBJTxMeshes;	/* list of xmesh cards */
    MESHcard *ymesh = model->NBJTyMeshes;	/* list of ymesh cards */
    DOMNcard *domains = model->NBJTdomains;	/* list of domain cards */
    BDRYcard *boundaries = model->NBJTboundaries;	/* list of boundary cards */
    DOPcard *dopings = model->NBJTdopings;	/* list of doping cards */
    ELCTcard *electrodes = model->NBJTelectrodes;	/* list of electrode cards */
    CONTcard *contacts = model->NBJTcontacts;	/* list of contact cards */
    MODLcard *models = model->NBJTmodels;	/* list of model cards */
    MATLcard *materials = model->NBJTmaterials;	/* list of material cards */
    MOBcard *mobility = model->NBJTmobility;	/* list of mobility cards */
    METHcard *methods = model->NBJTmethods;	/* list of method cards */
    OPTNcard *options = model->NBJToptions;	/* list of option cards */
    OUTPcard *outputs = model->NBJToutputs;	/* list of output cards */
    TWOtranInfo *pInfo = model->NBJTpInfo;	/* transient analysis information */
    DOPprofile *profiles = model->NBJTprofiles;	/* expanded list of doping profiles */
    DOPtable *dopTables = model->NBJTdopTables;	/* list of tables used by profiles */
    TWOmaterial *matlInfo = model->NBJTmatlInfo;	/* list of material info structures */
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
