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

    /* needed for outputting results */
    double ISRCcurrent; /* current value */

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
enum {
    PULSE = 1,
    SINE,
    EXP,
    SFFM,
    PWL,
    AM,
    TRNOISE,
    TRRANDOM,
    EXTERNAL,
};

#endif

/* device parameters */
enum {
    ISRC_DC = 1,
    ISRC_M,
    ISRC_AC_MAG,
    ISRC_AC_PHASE,
    ISRC_AC,
    ISRC_PULSE,
    ISRC_SINE,
    ISRC_EXP,
    ISRC_PWL,
    ISRC_SFFM,
    ISRC_NEG_NODE,
    ISRC_POS_NODE,
    ISRC_AC_REAL,
    ISRC_AC_IMAG,
    ISRC_FCN_TYPE,
    ISRC_FCN_ORDER,
    ISRC_FCN_COEFFS,
    ISRC_POWER,
    ISRC_D_F1,
    ISRC_D_F2,
    ISRC_VOLTS,
    ISRC_AM,
    ISRC_CURRENT,
};

enum {
    ISRC_TRNOISE = 25,
    ISRC_TRRANDOM,
    ISRC_EXTERNAL,
};

/* model parameters */

/* device questions */

/* model questions */

#include "isrcext.h"

#endif /*ISRC*/
