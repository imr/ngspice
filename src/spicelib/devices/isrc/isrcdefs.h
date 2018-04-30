/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifndef ISRC
#define ISRC

#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/gendefs.h"
#include "ngspice/complex.h"

/*
 * structures to describe independent current sources
 */


/* information needed for each instance */

typedef struct sISRCinstance {

    struct GENinstance gen;

#define ISRCmodPtr(inst) ((struct sISRCmodel *)((inst)->gen.GENmodPtr))
#define ISRCnextInstance(inst) ((struct sISRCinstance *)((inst)->gen.GENnextInstance))
#define ISRCname gen.GENname
#define ISRCstate gen.GENstate

    const int ISRCnegNode;    /* number of negative node of source */
    const int ISRCposNode;    /* number of positive node of source */

    int ISRCfunctionType;   /* code number of function type for source */
    int ISRCfunctionOrder;  /* order of the function for the source */
    double *ISRCcoeffs; /* pointer to array of coefficients */

    double ISRCdcValue; /* DC and TRANSIENT value of source */
    double ISRCmValue;  /* Parallel multiplier */

    double ISRCacPhase; /* AC phase angle */
    double ISRCacMag; /* AC magnitude */

    double ISRCacReal; /* AC real component */
    double ISRCacImag; /* AC imaginary component */

    double ISRCdF1mag; /* distortion f1 magnitude */
    double ISRCdF2mag; /* distortion f2 magnitude */
    double ISRCdF1phase; /* distortion f1 phase */
    double ISRCdF2phase; /* distortion f2 phase */

    struct trnoise_state *ISRCtrnoise_state; /* transient noise */
    struct trrandom_state *ISRCtrrandom_state; /* transient random source */

/* gtri - begin - add member to hold current source value */
#ifdef XSPICE
    /* needed for outputting results */
    double ISRCcurrent; /* current value */
#endif
/* gtri - end - add member to hold current source value */

    unsigned ISRCdcGiven     :1 ;  /* flag to indicate dc value given */
    unsigned ISRCmGiven      :1 ;  /* flag to indicate multiplier given */
    unsigned ISRCacGiven     :1 ;  /* flag to indicate ac keyword given */
    unsigned ISRCacMGiven    :1 ;  /* flag to indicate ac magnitude given */
    unsigned ISRCacPGiven    :1 ;  /* flag to indicate ac phase given */
    unsigned ISRCfuncTGiven  :1 ;  /* flag to indicate function type given */
    unsigned ISRCcoeffsGiven :1 ;  /* flag to indicate function coeffs given */
    unsigned ISRCdGiven      :1 ;  /* flag to indicate source is a distortion input */
    unsigned ISRCdF1given    :1 ;  /* flag to indicate source is an f1 distortion input */
    unsigned ISRCdF2given    :1 ;  /* flag to indicate source is an f2 distortion input */
} ISRCinstance ;


/* per model data */

typedef struct sISRCmodel {

    struct GENmodel gen;

#define ISRCmodType gen.GENmodType
#define ISRCnextModel(inst) ((struct sISRCmodel *)((inst)->gen.GENnextModel))
#define ISRCinstances(inst) ((ISRCinstance *)((inst)->gen.GENinstances))
#define ISRCmodName gen.GENmodName

} ISRCmodel;


/* source types */

#ifndef PULSE_FUN_TYPES
#define PULSE_FUN_TYPES
#define PULSE 1
#define SINE 2
#define EXP 3
#define SFFM 4
#define PWL 5
#define AM 6
#define TRNOISE 7
#define TRRANDOM 8
#define EXTERNAL 9
#endif

/* device parameters */
#define ISRC_DC 1
#define ISRC_M 2
#define ISRC_AC_MAG 3
#define ISRC_AC_PHASE 4
#define ISRC_AC 5
#define ISRC_PULSE 6
#define ISRC_SINE 7
#define ISRC_EXP 8
#define ISRC_PWL 9
#define ISRC_SFFM 10
#define ISRC_NEG_NODE 11
#define ISRC_POS_NODE 12
#define ISRC_AC_REAL 13
#define ISRC_AC_IMAG 14
#define ISRC_FCN_TYPE 15
#define ISRC_FCN_ORDER 16
#define ISRC_FCN_COEFFS 17
#define ISRC_POWER 18
#define ISRC_D_F1 19
#define ISRC_D_F2 20
#define ISRC_VOLTS 21

#define ISRC_AM 22
#define ISRC_CURRENT 23
#define ISRC_TRNOISE 25
#define ISRC_TRRANDOM 26
#define ISRC_EXTERNAL 27

/* model parameters */

/* device questions */

/* model questions */

#include "isrcext.h"

#endif /*ISRC*/
