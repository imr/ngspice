/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifndef ASRC
#define ASRC


#include "cktdefs.h"
#include "ifsim.h"
#include "complex.h"

        /*
         * structures to describe Arbitrary sources
         */

/* information to describe a single instance */

typedef struct sASRCinstance {
    struct sASRCmodel *ARRCmodPtr;  /* backpointer to model */
    struct sASRCinstance *ASRCnextInstance;  /* pointer to next instance of 
                                              *current model*/
    IFuid ASRCname; /* pointer to character string naming this instance */
    int ASRCowner;  /* number of owner process */
    int ASRCstates; /* state info */
    int ASRCposNode;    /* number of positive node of source */
    int ASRCnegNode;    /* number of negative node of source */
    int ASRCtype;   /* Whether source is voltage or current */
    int ASRCbranch;     /* number of branch equation added for v source */
    IFparseTree *ASRCtree; /* The parse tree */
    double **ASRCposptr;  /* pointer to pointers of the elements
               * in the sparce matrix */
    double ASRCprev_value; /* Previous value for the convergence test */
    double *ASRCacValues;  /* Store rhs and derivatives for ac anal */
    int ASRCcont_br;    /* Temporary store for controlling current branch */

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

/* module-wide variables */

extern double *asrc_vals, *asrc_derivs;
extern int asrc_nvals;

/* model parameters */

/* device questions */

/* model questions */

#include "asrcext.h"
#endif /*ASRC*/
