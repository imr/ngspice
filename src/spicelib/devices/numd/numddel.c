/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author: 1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "numddefs.h"
#include "../../../ciderlib/oned/onedext.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/carddefs.h"


int
NUMDdelete(GENinstance *gen_inst)
{
    NUMDinstance *inst = (NUMDinstance *) gen_inst;

    ONEdestroy(inst->NUMDpDevice);

    return OK;
}

int NUMDmodDelete(GENmodel *gen_model)
{
    NUMDmodel *model = (NUMDmodel *)gen_model;
    MESHcard *xmesh = model->NUMDxMeshes;	/* list of xmesh cards */
    MESHcard *ymesh = model->NUMDyMeshes;	/* list of ymesh cards */
    DOMNcard *domains = model->NUMDdomains;	/* list of domain cards */
    BDRYcard *boundaries = model->NUMDboundaries;	/* list of boundary cards */
    DOPcard *dopings = model->NUMDdopings;		/* list of doping cards */
    ELCTcard *electrodes = model->NUMDelectrodes;	/* list of electrode cards */
    CONTcard *contacts = model->NUMDcontacts;	/* list of contact cards */
    MODLcard *models = model->NUMDmodels;		/* list of model cards */
    MATLcard *materials = model->NUMDmaterials;	/* list of material cards */
    MOBcard *mobility = model->NUMDmobility;	/* list of mobility cards */
    METHcard *methods = model->NUMDmethods;	/* list of method cards */
    OPTNcard *options = model->NUMDoptions;	/* list of option cards */
    OUTPcard *outputs = model->NUMDoutputs;	/* list of output cards */
    ONEtranInfo *pInfo = model->NUMDpInfo;	/* transient analysis information */
    DOPprofile *profiles = model->NUMDprofiles;	/* expanded list of doping profiles */
    DOPtable *dopTables = model->NUMDdopTables;	/* list of tables used by profiles */
    ONEmaterial *matlInfo = model->NUMDmatlInfo;	/* list of material info structures */
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
