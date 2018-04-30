/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifndef CCCS
#define CCCS

#include "ngspice/ifsim.h"
#include "ngspice/gendefs.h"
#include "ngspice/complex.h"
#include "ngspice/cktdefs.h"

    /* structures used to describe Current Controlled Current Sources */

/* information needed for each instance */

typedef struct sCCCSinstance {

    struct GENinstance gen;

#define CCCSmodPtr(inst) ((struct sCCCSmodel *)((inst)->gen.GENmodPtr))
#define CCCSnextInstance(inst) ((struct sCCCSinstance *)((inst)->gen.GENnextInstance))
#define CCCSname gen.GENname
#define CCCSstate gen.GENstate

    const int CCCSposNode; /* number of positive node of source */
    const int CCCSnegNode; /* number of negative node of source */
    int CCCScontBranch;    /* number of branch eq of controlling source */

    char *CCCScontName; /* pointer to name of controlling instance */

    double CCCScoeff;   /* coefficient */

    double CCCSmValue;  /* Parallel multiplier */

    double *CCCSposContBrPtr;  /* pointer to sparse matrix element at 
                                     *(positive node, control branch eq)*/
    double *CCCSnegContBrPtr;  /* pointer to sparse matrix element at 
                                     *(negative node, control branch eq)*/
    unsigned CCCScoeffGiven :1 ;   /* flag to indicate coeff given */
    unsigned CCCSmGiven     :1 ;  /* flag to indicate multiplier given */

    int  CCCSsenParmNo;   /* parameter # for sensitivity use;
            set equal to  0 if not a design parameter*/

} CCCSinstance ;

/* per model data */

typedef struct sCCCSmodel {       /* model structure for a source */

    struct GENmodel gen;

#define CCCSmodType gen.GENmodType
#define CCCSnextModel(inst) ((struct sCCCSmodel *)((inst)->gen.GENnextModel))
#define CCCSinstances(inst) ((CCCSinstance *)((inst)->gen.GENinstances))
#define CCCSmodName gen.GENmodName

} CCCSmodel;

/* device parameters */
enum {
    CCCS_GAIN = 1,
    CCCS_CONTROL,
    CCCS_POS_NODE,
    CCCS_NEG_NODE,
    CCCS_CONT_BR,
    CCCS_GAIN_SENS,
    CCCS_CURRENT,
    CCCS_POWER,
    CCCS_VOLTS,
    CCCS_M,
};

/* model parameters */

/* device questions */
enum {
    CCCS_QUEST_SENS_REAL = 201,
    CCCS_QUEST_SENS_IMAG,
    CCCS_QUEST_SENS_MAG,
    CCCS_QUEST_SENS_PH,
    CCCS_QUEST_SENS_CPLX,
    CCCS_QUEST_SENS_DC,
};

/* model questions */

#include "cccsext.h"

#endif /*CCCS*/
