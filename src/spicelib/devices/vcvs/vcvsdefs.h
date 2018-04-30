/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifndef VCVS
#define VCVS


#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/gendefs.h"
#include "ngspice/complex.h"

        /*
         * structures to describe Voltage Controlled Voltage Sources
         */

/* information to describe a single instance */

typedef struct sVCVSinstance {

    struct GENinstance gen;

#define VCVSmodPtr(inst) ((struct sVCVSmodel *)((inst)->gen.GENmodPtr))
#define VCVSnextInstance(inst) ((struct sVCVSinstance *)((inst)->gen.GENnextInstance))
#define VCVSname gen.GENname
#define VCVSstates gen.GENstate

    const int VCVSposNode;    /* number of positive node of source */
    const int VCVSnegNode;    /* number of negative node of source */
    const int VCVScontPosNode;    /* number of positive node of controlling source */
    const int VCVScontNegNode;    /* number of negative node of controlling source */
    int VCVSbranch; /* equation number of branch equation added for v source */

    double VCVSinitCond;    /* initial condition (of controlling source) */
    double VCVScoeff;   /* coefficient */

    double *VCVSposIbrPtr;  /* pointer to sparse matrix element at 
                             * (positive node, branch equation) */
    double *VCVSnegIbrPtr;  /* pointer to sparse matrix element at 
                             * (negative node, branch equation) */
    double *VCVSibrPosPtr;  /* pointer to sparse matrix element at 
                             * (branch equation, positive node) */
    double *VCVSibrNegPtr;  /* pointer to sparse matrix element at 
                             * (branch equation, negative node) */
    double *VCVSibrContPosPtr;  /* pointer to sparse matrix element at 
                                 *(branch equation, control positive node)*/
    double *VCVSibrContNegPtr;  /* pointer to sparse matrix element at 
                                 *(branch equation, control negative node)*/
    unsigned VCVScoeffGiven :1 ;/* flag to indicate function coeffs given */

    int  VCVSsenParmNo;   /* parameter # for sensitivity use;
            set equal to  0 if not a design parameter*/

} VCVSinstance ;

#define VCVSvOld VCVSstates
#define VCVScontVOld VCVSstates + 1

/* per model data */

typedef struct sVCVSmodel {       /* model structure for a source */

    struct GENmodel gen;

#define VCVSmodType gen.GENmodType
#define VCVSnextModel(inst) ((struct sVCVSmodel *)((inst)->gen.GENnextModel))
#define VCVSinstances(inst) ((VCVSinstance *)((inst)->gen.GENinstances))
#define VCVSmodName gen.GENmodName

} VCVSmodel;

/* device parameters */
enum {
    VCVS_GAIN = 1,
    VCVS_POS_NODE,
    VCVS_NEG_NODE,
    VCVS_CONT_P_NODE,
    VCVS_CONT_N_NODE,
    VCVS_BR,
    VCVS_IC,
    VCVS_CONT_V_OLD,
    VCVS_GAIN_SENS,
    VCVS_CURRENT,
    VCVS_POWER,
    VCVS_VOLTS,
};

/* model parameters */

/* device questions */
enum {
    VCVS_QUEST_SENS_REAL = 201,
    VCVS_QUEST_SENS_IMAG,
    VCVS_QUEST_SENS_MAG,
    VCVS_QUEST_SENS_PH,
    VCVS_QUEST_SENS_CPLX,
    VCVS_QUEST_SENS_DC,
};

/* model questions */

#include "vcvsext.h"

#endif /*VCVS*/
