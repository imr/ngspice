/**********
Copyright 1991 Regents of the University of California. All rights reserved.
Authors: 1991 David Gates
**********/

/* Member of CIDER device simulator
 * Version: 1b1
 */

/*
 * Generic Numerical Model Structure
 */
#ifndef NUMGEN_H
#define NUMGEN_H

#include "ifsim.h"
#include "cktdefs.h"
#include "gendefs.h"
#include "carddefs.h"

typedef struct sGENnumModel {	/* model structure for a numerical device */
    int GENmodType;		/* type index of this device type */
    struct sGENnumModel *GENnextModel;	/* pointer to next model in list */
    GENinstance *GENinstances;	/* pointer to list of instances */
    IFuid GENmodName;		/* pointer to string naming this model */
/* Everything below here is numerical-device-specific */
    MESHcard *GENxMeshes;	/* list of xmesh cards */
    MESHcard *GENyMeshes;	/* list of ymesh cards */
    DOMNcard *GENdomains;	/* list of domain cards */
    BDRYcard *GENboundaries;	/* list of boundary cards */
    DOPcard  *GENdopings;	/* list of doping cards */
    ELCTcard *GENelectrodes;	/* list of electrode cards */
    CONTcard *GENcontacts;	/* list of contact cards */
    MODLcard *GENmodels;	/* list of model cards */
    MATLcard *GENmaterials;	/* list of material cards */
    MOBcard  *GENmobility;	/* list of mobility cards */
    METHcard *GENmethods;	/* list of method cards */
    OPTNcard *GENoptions;	/* list of option cards */
    OUTPcard *GENoutputs;	/* list of output cards */
} GENnumModel;
    
#endif /* NUMGEN_H */
