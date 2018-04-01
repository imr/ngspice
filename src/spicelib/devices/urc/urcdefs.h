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
#define URC_LEN 1
#define URC_LUMPS 2
#define URC_POS_NODE 3
#define URC_NEG_NODE 4
#define URC_GND_NODE 5

/* model parameters */
#define URC_MOD_K 101
#define URC_MOD_FMAX 102
#define URC_MOD_RPERL 103
#define URC_MOD_CPERL 104
#define URC_MOD_ISPERL 105
#define URC_MOD_RSPERL 106
#define URC_MOD_URC 107

/* device questions */

/* model questions */

#include "urcext.h"

#endif /*URC*/
