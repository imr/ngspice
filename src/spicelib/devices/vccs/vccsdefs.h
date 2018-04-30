/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifndef VCCS
#define VCCS

#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/gendefs.h"
#include "ngspice/complex.h"

        /*
         * structures to describe Voltage Controlled Current Sources
         */

/* information to describe a single instance */

typedef struct sVCCSinstance {

    struct GENinstance gen;

#define VCCSmodPtr(inst) ((struct sVCCSmodel *)((inst)->gen.GENmodPtr))
#define VCCSnextInstance(inst) ((struct sVCCSinstance *)((inst)->gen.GENnextInstance))
#define VCCSname gen.GENname
#define VCCSstates gen.GENstate

    const int VCCSposNode;    /* number of positive node of source */
    const int VCCSnegNode;    /* number of negative node of source */
    const int VCCScontPosNode;    /* number of positive node of controlling source */
    const int VCCScontNegNode;    /* number of negative node of controlling source */

    double VCCSinitCond;    /* initial condition (of controlling source) */
    double VCCScoeff;       /* coefficient */
    double VCCSmValue;      /* Parallel multiplier */

    double *VCCSposContPosPtr;  /* pointer to sparse matrix element at 
                                 * (positive node, control positive node) */
    double *VCCSposContNegPtr;  /* pointer to sparse matrix element at 
                                 * (negative node, control negative node) */
    double *VCCSnegContPosPtr;  /* pointer to sparse matrix element at 
                                 * (positive node, control positive node) */
    double *VCCSnegContNegPtr;  /* pointer to sparse matrix element at 
                                 * (negative node, control negative node) */
    unsigned VCCScoeffGiven :1 ;/* flag to indicate function coeffs given */
    unsigned VCCSmGiven     :1 ;/* flag to indicate multiplier given */

    int  VCCSsenParmNo;   /* parameter # for sensitivity use;
            set equal to  0 if not a design parameter*/

} VCCSinstance ;

#define VCCSvOld VCCSstates
#define VCCScontVOld VCCSstates + 1

/* per model data */

typedef struct sVCCSmodel {       /* model structure for a source */

    struct GENmodel gen;

#define VCCSmodType gen.GENmodType
#define VCCSnextModel(inst) ((struct sVCCSmodel *)((inst)->gen.GENnextModel))
#define VCCSinstances(inst) ((VCCSinstance *)((inst)->gen.GENinstances))
#define VCCSmodName gen.GENmodName

} VCCSmodel;

/* device parameters */
enum {
    VCCS_TRANS = 1,
    VCCS_IC,
    VCCS_POS_NODE,
    VCCS_NEG_NODE,
    VCCS_CONT_P_NODE,
    VCCS_CONT_N_NODE,
    VCCS_CONT_V_OLD,
    VCCS_TRANS_SENS,
    VCCS_CURRENT,
    VCCS_POWER,
    VCCS_VOLTS,
    VCCS_M,
};

/* model parameters */

/* device questions */
enum {
    VCCS_QUEST_SENS_REAL = 201,
    VCCS_QUEST_SENS_IMAG,
    VCCS_QUEST_SENS_MAG,
    VCCS_QUEST_SENS_PH,
    VCCS_QUEST_SENS_CPLX,
    VCCS_QUEST_SENS_DC,
};

/* model questions */

#include "vccsext.h"

#endif /*VCCS*/
