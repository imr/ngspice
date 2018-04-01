/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifndef CCVS
#define CCVS

#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/gendefs.h"
#include "ngspice/complex.h"

    /* structures used to describe current controlled voltage sources */

/* information used to describe a single instance */

typedef struct sCCVSinstance {

    struct GENinstance gen;

#define CCVSmodPtr(inst) ((struct sCCVSmodel *)((inst)->gen.GENmodPtr))
#define CCVSnextInstance(inst) ((struct sCCVSinstance *)((inst)->gen.GENnextInstance))
#define CCVSname gen.GENname
#define CCVSstate gen.GENstate

    const int CCVSposNode;    /* number of positive node of source */
    const int CCVSnegNode;    /* number of negative node of source */

    IFuid CCVScontName; /* pointer to name of controlling instance */
    int CCVSbranch; /* equation number of branch equation added for v source */
    int CCVScontBranch;    /* number of branch eq of controlling source */

    double CCVScoeff;   /* coefficient */

    double *CCVSposIbrPtr;  /* pointer to sparse matrix element at 
                                     * (positive node, branch equation) */
    double *CCVSnegIbrPtr;  /* pointer to sparse matrix element at 
                                     * (negative node, branch equation) */
    double *CCVSibrPosPtr;  /* pointer to sparse matrix element at 
                                     * (branch equation, positive node) */
    double *CCVSibrNegPtr;  /* pointer to sparse matrix element at 
                                     * (branch equation, negative node) */
    double *CCVSibrContBrPtr;  /* pointer to sparse matrix element at 
                                     *(branch equation, control branch eq)*/
    unsigned CCVScoeffGiven :1 ;   /* flag to indicate coeff given */

    int  CCVSsenParmNo;   /* parameter # for sensitivity use;
            set equal to  0 if not a design parameter*/

#ifdef KLU
    BindElement *CCVSposIbrBinding ;
    BindElement *CCVSnegIbrBinding ;
    BindElement *CCVSibrNegBinding ;
    BindElement *CCVSibrPosBinding ;
    BindElement *CCVSibrContBrBinding ;
#endif

} CCVSinstance ;

/* per model data */

typedef struct sCCVSmodel {       /* model structure for a CCVsource */

    struct GENmodel gen;

#define CCVSmodType gen.GENmodType
#define CCVSnextModel(inst) ((struct sCCVSmodel *)((inst)->gen.GENnextModel))
#define CCVSinstances(inst) ((CCVSinstance *)((inst)->gen.GENinstances))
#define CCVSmodName gen.GENmodName

} CCVSmodel;

/* device parameters */
#define CCVS_TRANS 1
#define CCVS_CONTROL 2
#define CCVS_POS_NODE 3
#define CCVS_NEG_NODE 4
#define CCVS_BR 5
#define CCVS_CONT_BR 6
#define CCVS_TRANS_SENS 7
#define CCVS_CURRENT 8
#define CCVS_POWER 9
#define CCVS_VOLTS 10

/* model parameters */

/* device questions */
#define CCVS_QUEST_SENS_REAL         201
#define CCVS_QUEST_SENS_IMAG         202
#define CCVS_QUEST_SENS_MAG      203
#define CCVS_QUEST_SENS_PH       204
#define CCVS_QUEST_SENS_CPLX         205
#define CCVS_QUEST_SENS_DC       206

/* model questions */

#include "ccvsext.h"

#endif /*CCVS*/
