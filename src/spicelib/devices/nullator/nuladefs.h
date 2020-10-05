/**********
Author: Florian Ballenegger 2020
Adapted from VCVS device code.
**********/
/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifndef NULA
#define NULA


#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/gendefs.h"
#include "ngspice/complex.h"

        /*
         * structures to describe Voltage Controlled Voltage Sources
         */

/* information to describe a single instance */

typedef struct sNULAinstance {

    struct GENinstance gen;

#define NULAmodPtr(inst) ((struct sNULAmodel *)((inst)->gen.GENmodPtr))
#define NULAnextInstance(inst) ((struct sNULAinstance *)((inst)->gen.GENnextInstance))
#define NULAname gen.GENname
#define NULAstates gen.GENstate

    const int NULAcontPosNode;    /* number of positive node of controlling source */
    const int NULAcontNegNode;    /* number of negative node of controlling source */
    int NULAbranch; /* equation number of branch equation added for v source */

    double NULAoffset;  /* control offset */

    double *NULAibrContPosPtr;  /* pointer to sparse matrix element at 
                                 *(branch equation, control positive node)*/
    double *NULAibrContNegPtr;  /* pointer to sparse matrix element at 
                                 *(branch equation, control negative node)*/
    unsigned NULAoffsetGiven :1 ;/* flag to indicate offset given */

} NULAinstance ;

#define NULAvOld NULAstates
#define NULAcontVOld NULAstates + 1

/* per model data */

typedef struct sNULAmodel {       /* model structure for a source */

    struct GENmodel gen;

#define NULAmodType gen.GENmodType
#define NULAnextModel(inst) ((struct sNULAmodel *)((inst)->gen.GENnextModel))
#define NULAinstances(inst) ((NULAinstance *)((inst)->gen.GENinstances))
#define NULAmodName gen.GENmodName

} NULAmodel;

/* device parameters */
enum {
    NULA_OFFSET = 1,
    NULA_CONT_P_NODE,
    NULA_CONT_N_NODE,
};

/* model parameters */

/* device questions */

/* model questions */

#include "nulaext.h"

#endif /*NULA*/
