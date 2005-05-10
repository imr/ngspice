/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifndef ISRC
#define ISRC

#include "ifsim.h"
#include "complex.h"
#include "cktdefs.h"
#include "gendefs.h"

    /* structures used to describe current sources */


/* information needed per source instance */

typedef struct sISRCinstance {
    struct sISRCmodel *ISRCmodPtr;  /* backpointer to model */
    struct sISRCinstance *ISRCnextInstance;  /* pointer to next instance of 
                                              *current model*/
    IFuid ISRCname; /* pointer to character string naming this instance */
    int ISRCowner;  /* number of owner process */
    int ISRCstate; /* not used */

    int ISRCnegNode;    /* number of negative node of source */
    int ISRCposNode;    /* number of positive node of source */

    int ISRCfunctionType;   /* code number of function type for source */
    int ISRCfunctionOrder;  /* order of the function for the source */
    double *ISRCcoeffs; /* pointer to array of coefficients */

    double ISRCdcValue; /* DC and TRANSIENT value of source */
    double ISRCacPhase; /* AC phase angle */
    double ISRCacMag; /* AC magnitude */
    double ISRCacReal; /* AC real part */
    double ISRCacImag; /* AC imaginary */

    double ISRCdF1mag; /* distortion f1 magnitude */
    double ISRCdF2mag; /* distortion f2 magnitude */
    double ISRCdF1phase; /* distortion f1 phase */
    double ISRCdF2phase; /* distortion f2 phase */

    /* gtri - begin - add member to hold current source value */
#ifdef XSPICE
    /* needed for outputting results */
    double ISRCcurrent; /* current value */
#endif
    /* gtri - end - add member to hold current source value */

    unsigned ISRCdcGiven     :1 ;   /* flag to indicate dc value given */
    unsigned ISRCacGiven     :1 ;   /* flag to indicate ac keyword given */
    unsigned ISRCacMGiven    :1 ;   /* flag to indicate ac magnitude given */
    unsigned ISRCacPGiven    :1 ;   /* flag to indicate ac phase given */
    unsigned ISRCfuncTGiven  :1 ;   /* flag to indicate function type given */
    unsigned ISRCcoeffsGiven :1 ;   /* flag to indicate function coeffs given */
    unsigned ISRCdGiven  :1 ; /* flag to indicate source is a distortion input */
    unsigned ISRCdF1given :1; /* flag to indicate source is an f1 distortion input */
    unsigned ISRCdF2given :1; /* flag to indicate source is an f2 distortion input */
} ISRCinstance ;


/* per model data */

typedef struct sISRCmodel {       /* model structure for a resistor */
    int ISRCmodType;    /* type index of this device type */
    struct sISRCmodel *ISRCnextModel;    /* pointer to next possible model 
                                          *in linked list */
    ISRCinstance * ISRCinstances;    /* pointer to list of instances 
                                             * that have this model */
    IFuid ISRCmodName;       /* pointer to character string naming this model */
} ISRCmodel;


/* source types */

#ifndef PULSE
#define PULSE 1
#define SINE 2
#define EXP 3
#define SFFM 4
#define PWL 5
#define AM 6
#endif /*PULSE*/

/* device parameters */
#define ISRC_DC 1
#define ISRC_AC_MAG 2
#define ISRC_AC_PHASE 3
#define ISRC_AC 4
#define ISRC_PULSE 5
#define ISRC_SINE 6
#define ISRC_EXP 7
#define ISRC_PWL 8
#define ISRC_SFFM 9
#define ISRC_NEG_NODE 10
#define ISRC_POS_NODE 11
#define ISRC_AC_REAL 12
#define ISRC_AC_IMAG 13
#define ISRC_FCN_TYPE 14
#define ISRC_FCN_ORDER 15
#define ISRC_FCN_COEFFS 16
#define ISRC_POWER 17
#define ISRC_D_F1 18
#define ISRC_D_F2 19
#define ISRC_VOLTS 20

#define ISRC_AM 21
/* gtri - begin - add define for current source value */
#ifdef XSPICE
/* needed for outputting results */
#define ISRC_CURRENT 22
#endif
/* gtri - end - add define for current source value */

/* model parameters */

/* device questions */

/* model questions */

#include "isrcext.h"

#endif /*ISRC*/
