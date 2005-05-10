/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifndef VSRC
#define VSRC

#include "ifsim.h"
#include "cktdefs.h"
#include "gendefs.h"
#include "complex.h"

        /*
         * structures to describe independent voltage sources
         */


/* information needed for each instance */

typedef struct sVSRCinstance {
    struct sVSRCmodel *VSRCmodPtr;  /* backpointer to model */
    struct sVSRCinstance *VSRCnextInstance;  /* pointer to next instance of 
                                              *current model */
    IFuid VSRCname; /* pointer to character string naming this instance */
    int VSRCowner;  /* number of owner process */
    int VSRCstate;	/* not used */

    int VSRCposNode;    /* number of positive node of resistor */
    int VSRCnegNode;    /* number of negative node of resistor */

    int VSRCbranch; /* equation number of branch equation added for source */

    int VSRCfunctionType;   /* code number of function type for source */
    int VSRCfunctionOrder;  /* order of the function for the source */
    double *VSRCcoeffs; /* pointer to array of coefficients */

    double VSRCdcValue; /* DC and TRANSIENT value of source */

    double VSRCacPhase; /* AC phase angle */
    double VSRCacMag; /* AC magnitude */

    double VSRCacReal; /* AC real component */
    double VSRCacImag; /* AC imaginary component */

    double VSRCdF1mag; /* distortion f1 magnitude */
    double VSRCdF2mag; /* distortion f2 magnitude */
    double VSRCdF1phase; /* distortion f1 phase */
    double VSRCdF2phase; /* distortion f2 phase */

    double *VSRCposIbrptr;  /* pointer to sparse matrix element at 
                             * (positive node, branch equation) */
    double *VSRCnegIbrptr;  /* pointer to sparse matrix element at 
                             * (negative node, branch equation) */
    double *VSRCibrPosptr;  /* pointer to sparse matrix element at 
                             * (branch equation, positive node) */
    double *VSRCibrNegptr;  /* pointer to sparse matrix element at 
                             * (branch equation, negative node) */
    double *VSRCibrIbrptr;  /* pointer to sparse matrix element at 
                             * (branch equation, branch equation) */
    unsigned VSRCdcGiven     :1 ;   /* flag to indicate dc value given */
    unsigned VSRCacGiven     :1 ;   /* flag to indicate ac keyword given */
    unsigned VSRCacMGiven    :1 ;   /* flag to indicate ac magnitude given */
    unsigned VSRCacPGiven    :1 ;   /* flag to indicate ac phase given */
    unsigned VSRCfuncTGiven  :1 ;   /* flag to indicate function type given */
    unsigned VSRCcoeffsGiven :1 ;   /* flag to indicate function coeffs given */
    unsigned VSRCdGiven  :1 ; /* flag to indicate source is a disto input */
    unsigned VSRCdF1given :1; /* flag to indicate source is an f1 dist input */
    unsigned VSRCdF2given :1; /* flag to indicate source is an f2 dist input */
} VSRCinstance ;


/* per model data */

typedef struct sVSRCmodel {
    int VSRCmodType;    /* type index of this device type */
    struct sVSRCmodel *VSRCnextModel;    /* pointer to next possible model 
                                          *in linked list */
    VSRCinstance * VSRCinstances;    /* pointer to list of instances 
                                      * that have this model */
    IFuid VSRCmodName;       /* pointer to character string naming this model */
} VSRCmodel;

/* source function types (shared with current sources) */
#ifndef PULSE
#define PULSE 1
#define SINE 2
#define EXP 3
#define SFFM 4
#define PWL 5
#define AM 6
#endif /*PULSE*/

/* device parameters */
#define VSRC_DC 1
#define VSRC_AC 2
#define VSRC_AC_MAG 3
#define VSRC_AC_PHASE 4
#define VSRC_PULSE 5
#define VSRC_SINE 6
#define VSRC_EXP 7
#define VSRC_PWL 8
#define VSRC_SFFM 9
#define VSRC_BR 10
#define VSRC_FCN_TYPE 11
#define VSRC_FCN_ORDER 12
#define VSRC_FCN_COEFFS 13
#define VSRC_AC_REAL 14
#define VSRC_AC_IMAG 15
#define VSRC_POS_NODE 16
#define VSRC_NEG_NODE 17
#define VSRC_CURRENT 18
#define VSRC_POWER 19
#define VSRC_D_F1 20
#define VSRC_D_F2 21

#define VSRC_AM 22

/* model parameters */

/* device questions */

/* model questions */

#include "vsrcext.h"

#endif /*VSRC*/
