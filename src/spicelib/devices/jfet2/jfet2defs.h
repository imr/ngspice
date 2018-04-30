/**********
Based on jfetdefs.h
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

Modified to add PS model and new parameter definitions ( Anthony E. Parker )
   Copyright 1994  Macquarie University, Sydney Australia.
   10 Feb 1994: Added xiwoo, d3 and alpha to JFET2instance
                JFET2pave, JFET2vtrap ad JFET2_STATE_COUNT
                Changed model to call jfetparm.h, added JFET2za to model struct
                Defined JFET2_VTRAP and JFET2_PAVE
**********/

#ifndef JFET2
#define JFET2

#include "ngspice/ifsim.h"
#include "ngspice/gendefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/noisedef.h"

    /* structures used to describe Junction Field Effect Transistors */

/* indices to an array of JFET2 noise sources */

enum {
    JFET2RDNOIZ = 0,
    JFET2RSNOIZ,
    JFET2IDNOIZ,
    JFET2FLNOIZ,
    JFET2TOTNOIZ,
    /* finally, the number of noise sources */
    JFET2NSRCS
};

/* information used to describe a single instance */

typedef struct sJFET2instance {

    struct GENinstance gen;

#define JFET2modPtr(inst) ((struct sJFET2model *)((inst)->gen.GENmodPtr))
#define JFET2nextInstance(inst) ((struct sJFET2instance *)((inst)->gen.GENnextInstance))
#define JFET2name gen.GENname
#define JFET2state gen.GENstate

    const int JFET2drainNode;  /* number of drain node of jfet */
    const int JFET2gateNode;   /* number of gate node of jfet */
    const int JFET2sourceNode; /* number of source node of jfet */
    int JFET2drainPrimeNode; /* number of internal drain node of jfet */
    int JFET2sourcePrimeNode;    /* number of internal source node of jfet */

    double *JFET2drainDrainPrimePtr; /* pointer to sparse matrix at 
                                     * (drain,drain prime) */
    double *JFET2gateDrainPrimePtr;  /* pointer to sparse matrix at 
                                     * (gate,drain prime) */
    double *JFET2gateSourcePrimePtr; /* pointer to sparse matrix at 
                                     * (gate,source prime) */
    double *JFET2sourceSourcePrimePtr;   /* pointer to sparse matrix at 
                                         * (source,source prime) */
    double *JFET2drainPrimeDrainPtr; /* pointer to sparse matrix at 
                                     * (drain prime,drain) */
    double *JFET2drainPrimeGatePtr;  /* pointer to sparse matrix at 
                                     * (drain prime,gate) */
    double *JFET2drainPrimeSourcePrimePtr;   /* pointer to sparse matrix
                                             * (drain prime,source prime) */
    double *JFET2sourcePrimeGatePtr; /* pointer to sparse matrix at 
                                     * (source prime,gate) */
    double *JFET2sourcePrimeSourcePtr;   /* pointer to sparse matrix at 
                                         * (source prime,source) */
    double *JFET2sourcePrimeDrainPrimePtr;   /* pointer to sparse matrix
                                             * (source prime,drain prime) */
    double *JFET2drainDrainPtr;  /* pointer to sparse matrix at 
                                 * (drain,drain) */
    double *JFET2gateGatePtr;    /* pointer to sparse matrix at 
                                 * (gate,gate) */
    double *JFET2sourceSourcePtr;    /* pointer to sparse matrix at 
                                     * (source,source) */
    double *JFET2drainPrimeDrainPrimePtr;    /* pointer to sparse matrix
                                             * (drain prime,drain prime) */
    double *JFET2sourcePrimeSourcePrimePtr;  /* pointer to sparse matrix
                                             * (source prime,source prime) */

	int JFET2mode;
	/* distortion analysis Taylor coeffs. */

/*
 * naming convention:
 * x = vgs
 * y = vds
 * cdr = cdrain
 */

#define JFET2NDCOEFFS	21

#ifndef NODISTO
	double JFET2dCoeffs[JFET2NDCOEFFS];
#else /* NODISTO */
	double *JFET2dCoeffs;
#endif /* NODISTO */

#ifndef CONFIG

#define	cdr_x		JFET2dCoeffs[0]
#define	cdr_y		JFET2dCoeffs[1]
#define	cdr_x2		JFET2dCoeffs[2]
#define	cdr_y2		JFET2dCoeffs[3]
#define	cdr_xy		JFET2dCoeffs[4]
#define	cdr_x3		JFET2dCoeffs[5]
#define	cdr_y3		JFET2dCoeffs[6]
#define	cdr_x2y		JFET2dCoeffs[7]
#define	cdr_xy2		JFET2dCoeffs[8]

#define	ggs1		JFET2dCoeffs[9]
#define	ggd1		JFET2dCoeffs[10]
#define	ggs2		JFET2dCoeffs[11]
#define	ggd2		JFET2dCoeffs[12]
#define	ggs3		JFET2dCoeffs[13]
#define	ggd3		JFET2dCoeffs[14]
#define	capgs1		JFET2dCoeffs[15]
#define	capgd1		JFET2dCoeffs[16]
#define	capgs2		JFET2dCoeffs[17]
#define	capgd2		JFET2dCoeffs[18]
#define	capgs3		JFET2dCoeffs[19]
#define	capgd3		JFET2dCoeffs[20]

#endif

#ifndef NONOISE
    double JFET2nVar[NSTATVARS][JFET2NSRCS];
#else /* NONOISE */
	double **JFET2nVar;
#endif /* NONOISE */

    unsigned JFET2off :1;            /* 'off' flag for jfet */
    unsigned JFET2areaGiven  : 1;    /* flag to indicate area was specified */
    unsigned JFET2mGiven     : 1;    /* flag to indicate multiplier given */
    unsigned JFET2icVDSGiven : 1;    /* initial condition given flag for V D-S*/
    unsigned JFET2icVGSGiven : 1;    /* initial condition given flag for V G-S*/
    unsigned JFET2tempGiven  : 1;    /* flag to indicate instance temp given */
    unsigned JFET2dtempGiven : 1;    /* flag to indicate temperature difference given */


    double JFET2area;    /* area factor for the jfet */
    double JFET2m;       /* parallel multiplier for the diode */
    double JFET2icVDS;   /* initial condition voltage D-S*/
    double JFET2icVGS;   /* initial condition voltage G-S*/
    double JFET2temp;    /* operating temperature */
    double JFET2dtemp;   /* Instance temperature difference */
    double JFET2tSatCur; /* temperature adjusted saturation current */
    double JFET2tGatePot;    /* temperature adjusted gate potential */
    double JFET2tCGS;    /* temperature corrected G-S capacitance */
    double JFET2tCGD;    /* temperature corrected G-D capacitance */
    double JFET2corDepCap;   /* joining point of the fwd bias dep. cap eq.s */
    double JFET2vcrit;   /* critical voltage for the instance */
    double JFET2f1;      /* coefficient of capacitance polynomial exp */
    double JFET2xiwoo;       /* velocity saturation potential */
    double JFET2d3;          /* Dual Power-law parameter */
    double JFET2alpha;       /* capacitance model transition parameter */

} JFET2instance ;

#define JFET2vgs      JFET2state 
#define JFET2vgd      JFET2state+1 
#define JFET2cg       JFET2state+2 
#define JFET2cd       JFET2state+3 
#define JFET2cgd      JFET2state+4 
#define JFET2gm       JFET2state+5 
#define JFET2gds      JFET2state+6 
#define JFET2ggs      JFET2state+7 
#define JFET2ggd      JFET2state+8 
#define JFET2qgs      JFET2state+9 
#define JFET2cqgs     JFET2state+10 
#define JFET2qgd      JFET2state+11 
#define JFET2cqgd     JFET2state+12 
#define JFET2qds      JFET2state+13
#define JFET2cqds     JFET2state+14
#define JFET2pave     JFET2state+15
#define JFET2vtrap    JFET2state+16
#define JFET2vgstrap  JFET2state+17
#define JFET2unknown  JFET2state+18

#define JFET2numStates 19

/* per model data */

typedef struct sJFET2model {       /* model structure for a jfet */

    struct GENmodel gen;

#define JFET2modType gen.GENmodType
#define JFET2nextModel(inst) ((struct sJFET2model *)((inst)->gen.GENnextModel))
#define JFET2instances(inst) ((JFET2instance *)((inst)->gen.GENinstances))
#define JFET2modName gen.GENmodName

    int JFET2type;

#define  PARAM(code,id,flag,ref,default,descrip) double ref;
#include "jfet2parm.h"

    double JFET2drainConduct;
    double JFET2sourceConduct;
    double JFET2f2;
    double JFET2f3;
    double JFET2za;      /* saturation index parameter */
    double JFET2tnom;    /* temperature at which parameters were measured */

#define  PARAM(code,id,flag,ref,default,descrip) unsigned flag : 1;
#include "jfet2parm.h"
    unsigned JFET2tnomGiven : 1; /* user specified Tnom for model */

} JFET2model;

#ifndef NJF

#define NJF 1
#define PJF -1

#endif /*NJF*/

/* device parameters */
enum {
    JFET2_AREA = 1,
    JFET2_IC_VDS,
    JFET2_IC_VGS,
    JFET2_IC,
    JFET2_OFF,
    JFET2_TEMP,
    JFET2_DTEMP,
    JFET2_M,
};

/* device questions */
enum {
    JFET2_DRAINNODE = 301,
    JFET2_GATENODE,
    JFET2_SOURCENODE,
    JFET2_DRAINPRIMENODE,
    JFET2_SOURCEPRIMENODE,
    JFET2_VGS,
    JFET2_VGD,
    JFET2_CG,
    JFET2_CD,
    JFET2_CGD,
    JFET2_GM,
    JFET2_GDS,
    JFET2_GGS,
    JFET2_GGD,
    JFET2_QGS,
    JFET2_CQGS,
    JFET2_QGD,
    JFET2_CQGD,
    JFET2_CS,
    JFET2_POWER,
    JFET2_VTRAP,
    JFET2_PAVE,
};

/* model questions */
enum {
    JFET2_MOD_DRAINCONDUCT = 301,
    JFET2_MOD_SOURCECONDUCT,
    JFET2_MOD_DEPLETIONCAP,
    JFET2_MOD_VCRIT,
    JFET2_MOD_TYPE,
};

/* function definitions */

#include "jfet2ext.h"

#endif /*JFET2*/
