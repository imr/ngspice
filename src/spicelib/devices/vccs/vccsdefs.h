/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifndef VCCS
#define VCCS

#include "ifsim.h"
#include "cktdefs.h"
#include "gendefs.h"
#include "complex.h"

        /*
         * structures to describe Voltage Controlled Current Sources
         */

/* information to describe a single instance */

typedef struct sVCCSinstance {
    struct sVCCSmodel *VCCSmodPtr;  /* backpointer to model */
    struct sVCCSinstance *VCCSnextInstance;  /* pointer to next instance of 
                                              *current model*/
    IFuid VCCSname; /* pointer to character string naming this instance */
    int VCCSowner;  /* number of owner process */
    int VCCSstates;         /* state info */

    int VCCSposNode;    /* number of positive node of source */
    int VCCSnegNode;    /* number of negative node of source */
    int VCCScontPosNode;    /* number of positive node of controlling source */
    int VCCScontNegNode;    /* number of negative node of controlling source */

    double VCCSinitCond;    /* initial condition (of controlling source) */
    double VCCScoeff;       /* coefficient */

    double *VCCSposContPosptr;  /* pointer to sparse matrix element at 
                                 * (positive node, control positive node) */
    double *VCCSposContNegptr;  /* pointer to sparse matrix element at 
                                 * (negative node, control negative node) */
    double *VCCSnegContPosptr;  /* pointer to sparse matrix element at 
                                 * (positive node, control positive node) */
    double *VCCSnegContNegptr;  /* pointer to sparse matrix element at 
                                 * (negative node, control negative node) */
    unsigned VCCScoeffGiven :1 ;/* flag to indicate function coeffs given */

    int  VCCSsenParmNo;   /* parameter # for sensitivity use;
            set equal to  0 if not a design parameter*/

} VCCSinstance ;

#define VCCSvOld VCCSstates
#define VCCScontVOld VCCSstates + 1

/* per model data */

typedef struct sVCCSmodel {       /* model structure for a source */
    int VCCSmodType;    /* type index of this device type */
    struct sVCCSmodel *VCCSnextModel;    /* pointer to next possible model 
                                          * in linked list */
    VCCSinstance * VCCSinstances;    /* pointer to list of instances 
                                      * that have this model */
    IFuid VCCSmodName;       /* pointer to character string naming this model */
} VCCSmodel;

/* device parameters */
#define VCCS_TRANS 1
#define VCCS_IC 2
#define VCCS_POS_NODE 3
#define VCCS_NEG_NODE 4
#define VCCS_CONT_P_NODE 5
#define VCCS_CONT_N_NODE 6
#define VCCS_CONT_V_OLD 7
#define VCCS_TRANS_SENS 8
#define VCCS_CURRENT 9
#define VCCS_POWER 10
#define VCCS_VOLTS 11

/* model parameters */

/* device questions */
#define VCCS_QUEST_SENS_REAL         201
#define VCCS_QUEST_SENS_IMAG         202
#define VCCS_QUEST_SENS_MAG      203
#define VCCS_QUEST_SENS_PH       204
#define VCCS_QUEST_SENS_CPLX         205
#define VCCS_QUEST_SENS_DC       206

/* model questions */

#endif /*VCCS*/
