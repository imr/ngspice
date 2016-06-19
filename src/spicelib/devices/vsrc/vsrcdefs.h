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

#ifdef KLU
    BindElement *VSRCposIbrBinding ;
    BindElement *VSRCnegIbrBinding ;
    BindElement *VSRCibrNegBinding ;
    BindElement *VSRCibrPosBinding ;
    BindElement *VSRCibrIbrBinding ;
#endif

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
#ifndef PULSE
#define PULSE 1
#define SINE 2
#define EXP 3
#define SFFM 4
#define PWL 5
#define AM 6
#define TRNOISE 7
#define TRRANDOM 8
#define EXTERNAL 9
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
#define VSRC_R 23
#define VSRC_TD 24
#define VSRC_TRNOISE 25
#define VSRC_TRRANDOM 26
#define VSRC_EXTERNAL 27

/* model parameters */

/* device questions */

/* model questions */

#include "vsrcext.h"

#endif /*VSRC*/
