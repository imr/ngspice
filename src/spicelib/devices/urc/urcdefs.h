/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifndef URC
#define URC


#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/gendefs.h"
#include "ngspice/complex.h"

    /*
     *  structures used to describe uniform RC lines
     */

/* information needed for each instance */

typedef struct sURCinstance {

    struct GENinstance gen;

#define URCmodPtr(inst) ((struct sURCmodel *)((inst)->gen.GENmodPtr))
#define URCnextInstance(inst) ((struct sURCinstance *)((inst)->gen.GENnextInstance))
#define URCname gen.GENname
#define URCstate gen.GENstate

    const int URCposNode;   /* number of positive node of URC */
    const int URCnegNode;   /* number of negative node of URC */
    const int URCgndNode;   /* number of the "ground" node of the URC */

    double URClength;   /* length of line */
    int URClumps;   /* number of lumps in line */
    unsigned URClenGiven : 1;   /* flag to indicate length was specified */
    unsigned URClumpsGiven : 1; /* flag to indicate lumps was specified */
} URCinstance ;

/* per model data */

typedef struct sURCmodel {       /* model structure for a resistor */

    struct GENmodel gen;

#define URCmodType gen.GENmodType
#define URCnextModel(inst) ((struct sURCmodel *)((inst)->gen.GENnextModel))
#define URCinstances(inst) ((URCinstance *)((inst)->gen.GENinstances))
#define URCmodName gen.GENmodName

    double URCk;        /* propagation constant for URC */
    double URCfmax;     /* max frequence of interest */
    double URCrPerL;    /* resistance per unit length */
    double URCcPerL;    /* capacitance per unit length */
    double URCisPerL;   /* diode saturation current per unit length */
    double URCrsPerL;   /* diode resistance per unit length */
    unsigned URCkGiven : 1;     /* flag to indicate k was specified */
    unsigned URCfmaxGiven : 1;  /* flag to indicate fmax was specified */
    unsigned URCrPerLGiven : 1; /* flag to indicate rPerL was specified */
    unsigned URCcPerLGiven : 1; /* flag to indicate cPerL was specified */
    unsigned URCisPerLGiven : 1; /* flag to indicate isPerL was specified */
    unsigned URCrsPerLGiven : 1; /* flag to indicate rsPerL was specified */
} URCmodel;

/* device parameters */
enum {
    URC_LEN = 1,
    URC_LUMPS,
    URC_POS_NODE,
    URC_NEG_NODE,
    URC_GND_NODE,
};

/* model parameters */
enum {
    URC_MOD_K = 101,
    URC_MOD_FMAX,
    URC_MOD_RPERL,
    URC_MOD_CPERL,
    URC_MOD_ISPERL,
    URC_MOD_RSPERL,
    URC_MOD_URC,
};

/* device questions */

/* model questions */

#include "urcext.h"

#endif /*URC*/
