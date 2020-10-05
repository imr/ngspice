/**********
Author: Florian Ballenegger 2020
Adapted from VCVS device code.
**********/
/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifndef NORA
#define NORA


#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/gendefs.h"
#include "ngspice/complex.h"

        /*
         * structures to describe Voltage Controlled Voltage Sources
         */

/* information to describe a single instance */

typedef struct sNORAinstance {

    struct GENinstance gen;

#define NORAmodPtr(inst) ((struct sNORAmodel *)((inst)->gen.GENmodPtr))
#define NORAnextInstance(inst) ((struct sNORAinstance *)((inst)->gen.GENnextInstance))
#define NORAname gen.GENname
#define NORAstates gen.GENstate

    const int NORAposNode;    /* number of positive node of source */
    const int NORAnegNode;    /* number of negative node of source */
    int NORAbranch; /* equation number of branch equation added for v source */

    double NORAinitCond;    /* initial condition (of branch current) */

    double *NORAposIbrPtr;  /* pointer to sparse matrix element at 
                             * (positive node, branch equation) */
    double *NORAnegIbrPtr;  /* pointer to sparse matrix element at 
                             * (negative node, branch equation) */

} NORAinstance ;

#define NORAvOld NORAstates
#define NORAcontVOld NORAstates + 1

/* per model data */

typedef struct sNORAmodel {       /* model structure for a source */

    struct GENmodel gen;

#define NORAmodType gen.GENmodType
#define NORAnextModel(inst) ((struct sNORAmodel *)((inst)->gen.GENnextModel))
#define NORAinstances(inst) ((NORAinstance *)((inst)->gen.GENinstances))
#define NORAmodName gen.GENmodName

} NORAmodel;

/* device parameters */
enum {
    NORA_POS_NODE = 1,
    NORA_NEG_NODE,
    NORA_BR,
    NORA_IC,
    NORA_CURRENT,
    NORA_POWER,
    NORA_VOLTS,
};

/* model parameters */


/* model questions */

#include "noraext.h"

#endif /*NORA*/
