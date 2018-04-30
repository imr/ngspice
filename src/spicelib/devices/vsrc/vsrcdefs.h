/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifndef VSRC
#define VSRC

#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/gendefs.h"
#include "ngspice/complex.h"

struct trnoise_state;

/*
 * structures to describe independent voltage sources
 */


/* information needed for each instance */

typedef struct sVSRCinstance {

    struct GENinstance gen;

#define VSRCmodPtr(inst) ((struct sVSRCmodel *)((inst)->gen.GENmodPtr))
#define VSRCnextInstance(inst) ((struct sVSRCinstance *)((inst)->gen.GENnextInstance))
#define VSRCname gen.GENname
#define VSRCstate gen.GENstate

    const int VSRCposNode;    /* number of positive node of source */
    const int VSRCnegNode;    /* number of negative node of source */

    int VSRCbranch; /* equation number of branch equation added for source */

    int VSRCfunctionType;   /* code number of function type for source */
    int VSRCfunctionOrder;  /* order of the function for the source */
    int VSRCrBreakpt;       /* pwl repeat breakpoint index */
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

    struct trnoise_state *VSRCtrnoise_state; /* transient noise */
    struct trrandom_state *VSRCtrrandom_state; /* transient random source */

    double VSRCr;           /* pwl repeat */
    double VSRCrdelay;     /* pwl delay period */
    double *VSRCposIbrPtr;  /* pointer to sparse matrix element at
                             * (positive node, branch equation) */
    double *VSRCnegIbrPtr;  /* pointer to sparse matrix element at
                             * (negative node, branch equation) */
    double *VSRCibrPosPtr;  /* pointer to sparse matrix element at
                             * (branch equation, positive node) */
    double *VSRCibrNegPtr;  /* pointer to sparse matrix element at
                             * (branch equation, negative node) */
    double *VSRCibrIbrPtr;  /* pointer to sparse matrix element at
                             * (branch equation, branch equation) */
    unsigned VSRCdcGiven     :1 ;  /* flag to indicate dc value given */
    unsigned VSRCacGiven     :1 ;  /* flag to indicate ac keyword given */
    unsigned VSRCacMGiven    :1 ;  /* flag to indicate ac magnitude given */
    unsigned VSRCacPGiven    :1 ;  /* flag to indicate ac phase given */
    unsigned VSRCfuncTGiven  :1 ;  /* flag to indicate function type given */
    unsigned VSRCcoeffsGiven :1 ;  /* flag to indicate function coeffs given */
    unsigned VSRCdGiven      :1 ;  /* flag to indicate source is a distortion input */
    unsigned VSRCdF1given    :1 ;  /* flag to indicate source is an f1 distortion input */
    unsigned VSRCdF2given    :1 ;  /* flag to indicate source is an f2 distortion input */
    unsigned VSRCrGiven      :1 ;  /* flag to indicate repeating pwl */
} VSRCinstance ;


/* per model data */

typedef struct sVSRCmodel {

    struct GENmodel gen;

#define VSRCmodType gen.GENmodType
#define VSRCnextModel(inst) ((struct sVSRCmodel *)((inst)->gen.GENnextModel))
#define VSRCinstances(inst) ((VSRCinstance *)((inst)->gen.GENinstances))
#define VSRCmodName gen.GENmodName

} VSRCmodel;

/* source function types (shared with current sources) */
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
    VSRC_DC = 1,
    VSRC_AC,
    VSRC_AC_MAG,
    VSRC_AC_PHASE,
    VSRC_PULSE,
    VSRC_SINE,
    VSRC_EXP,
    VSRC_PWL,
    VSRC_SFFM,
    VSRC_BR,
    VSRC_FCN_TYPE,
    VSRC_FCN_ORDER,
    VSRC_FCN_COEFFS,
    VSRC_AC_REAL,
    VSRC_AC_IMAG,
    VSRC_POS_NODE,
    VSRC_NEG_NODE,
    VSRC_CURRENT,
    VSRC_POWER,
    VSRC_D_F1,
    VSRC_D_F2,
    VSRC_AM,
    VSRC_R,
    VSRC_TD,
    VSRC_TRNOISE,
    VSRC_TRRANDOM,
    VSRC_EXTERNAL,
};

/* model parameters */

/* device questions */

/* model questions */

#include "vsrcext.h"

#endif /*VSRC*/
