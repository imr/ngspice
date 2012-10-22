/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifndef ASRC
#define ASRC

#include "ngspice/cktdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/complex.h"

/*
 * structures to describe Arbitrary sources
*/

/* information to describe a single instance */

typedef struct sASRCinstance {
    struct sASRCmodel *ARRCmodPtr;  /* backpointer to model */
    struct sASRCinstance *ASRCnextInstance;  /* pointer to next instance of 
                                              * current model */
    IFuid ASRCname; /* pointer to character string naming this instance */
    int ASRCstates; /* state info */
    int ASRCposNode;    /* number of positive node of source */
    int ASRCnegNode;    /* number of negative node of source */
    int ASRCtype;       /* Whether source is voltage or current */
    int ASRCbranch;     /* number of branch equation added for v source */
    IFparseTree *ASRCtree; /* The parse tree */
    double ASRCtemp;     /* temperature at which this resistor operates */
    double ASRCdtemp;    /* delta-temperature of a particular instance  */
    double ASRCtc1;      /* first temperature coefficient of resistors */
    double ASRCtc2;      /* second temperature coefficient of resistors */
    int ASRCreciproctc;  /* Flag to calculate reciprocal temperature behaviour */
    double **ASRCposptr;  /* pointer to pointers of the elements
                           * in the sparce matrix */
    double ASRCprev_value; /* Previous value for the convergence test */
    double *ASRCacValues;  /* Store rhs and derivatives for ac anal */
    int ASRCcont_br;    /* Temporary store for controlling current branch */
    unsigned ASRCtempGiven   : 1;    /* indicates temperature specified */
    unsigned ASRCdtempGiven  : 1;    /* indicates delta-temp specified  */
    unsigned ASRCtc1Given    : 1;    /* indicates tc1 parameter specified */
    unsigned ASRCtc2Given    : 1;    /* indicates tc2 parameter specified */
    unsigned ASRCreciproctcGiven : 1; /* indicates reciproctc flag parameter specified */
} ASRCinstance ;

#define ASRCvOld ASRCstates
#define ASRCcontVOld ASRCstates + 1

/* per model data */

typedef struct sASRCmodel {       /* model structure for a source */
    int ASRCmodType;    /* type index of this device */
    struct sASRCmodel *ASRCnextModel;   /* pointer to next possible model 
                                         *in linked list */
    ASRCinstance * ASRCinstances;    /* pointer to list of instances 
                                      * that have this model */
    IFuid ASRCmodName;       /* pointer to character string naming this model */
} ASRCmodel;

/* device parameters */
#define ASRC_VOLTAGE 1
#define ASRC_CURRENT 2
#define ASRC_POS_NODE 3
#define ASRC_NEG_NODE 4
#define ASRC_PARSE_TREE 5
#define ASRC_OUTPUTVOLTAGE 6
#define ASRC_OUTPUTCURRENT 7
#define ASRC_TEMP 8
#define ASRC_DTEMP 9
#define ASRC_TC1 10
#define ASRC_TC2 11
#define ASRC_RTC 12

/* module-wide variables */

extern double *asrc_vals, *asrc_derivs;
extern int asrc_nvals;

/* model parameters */

/* device questions */

/* model questions */

#include "asrcext.h"
#endif /*ASRC*/
