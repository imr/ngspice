/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Sydney University mods Copyright(c) 1989 Anthony E. Parker, David J. Skellern
	Laboratory for Communication Science Engineering
	Sydney University Department of Electrical Engineering, Australia
**********/

#ifndef JFET
#define JFET

#include "ifsim.h"
#include "gendefs.h"
#include "cktdefs.h"
#include "complex.h"
#include "noisedef.h"

    /* structures used to describe Junction Field Effect Transistors */


/* information used to describe a single instance */

typedef struct sJFETinstance {
    struct sJFETmodel *JFETmodPtr;  /* backpointer to model */
    struct sJFETinstance *JFETnextInstance; /* pointer to next instance of 
                                             * current model*/
    IFuid JFETname; /* pointer to character string naming this instance */
    int JFETowner;  /* number of owner process */
    int JFETstate; /* pointer to start of state vector for jfet */
    int JFETdrainNode;  /* number of drain node of jfet */
    int JFETgateNode;   /* number of gate node of jfet */
    int JFETsourceNode; /* number of source node of jfet */
    int JFETdrainPrimeNode; /* number of internal drain node of jfet */
    int JFETsourcePrimeNode;    /* number of internal source node of jfet */

    double *JFETdrainDrainPrimePtr; /* pointer to sparse matrix at 
                                     * (drain,drain prime) */
    double *JFETgateDrainPrimePtr;  /* pointer to sparse matrix at 
                                     * (gate,drain prime) */
    double *JFETgateSourcePrimePtr; /* pointer to sparse matrix at 
                                     * (gate,source prime) */
    double *JFETsourceSourcePrimePtr;   /* pointer to sparse matrix at 
                                         * (source,source prime) */
    double *JFETdrainPrimeDrainPtr; /* pointer to sparse matrix at 
                                     * (drain prime,drain) */
    double *JFETdrainPrimeGatePtr;  /* pointer to sparse matrix at 
                                     * (drain prime,gate) */
    double *JFETdrainPrimeSourcePrimePtr;   /* pointer to sparse matrix
                                             * (drain prime,source prime) */
    double *JFETsourcePrimeGatePtr; /* pointer to sparse matrix at 
                                     * (source prime,gate) */
    double *JFETsourcePrimeSourcePtr;   /* pointer to sparse matrix at 
                                         * (source prime,source) */
    double *JFETsourcePrimeDrainPrimePtr;   /* pointer to sparse matrix
                                             * (source prime,drain prime) */
    double *JFETdrainDrainPtr;  /* pointer to sparse matrix at 
                                 * (drain,drain) */
    double *JFETgateGatePtr;    /* pointer to sparse matrix at 
                                 * (gate,gate) */
    double *JFETsourceSourcePtr;    /* pointer to sparse matrix at 
                                     * (source,source) */
    double *JFETdrainPrimeDrainPrimePtr;    /* pointer to sparse matrix
                                             * (drain prime,drain prime) */
    double *JFETsourcePrimeSourcePrimePtr;  /* pointer to sparse matrix
                                             * (source prime,source prime) */

	int JFETmode;
	/* distortion analysis Taylor coeffs. */

/*
 * naming convention:
 * x = vgs
 * y = vds
 * cdr = cdrain
 */

#define JFETNDCOEFFS	21

#ifndef NODISTO
	double JFETdCoeffs[JFETNDCOEFFS];
#else /* NODISTO */
	double *JFETdCoeffs;
#endif /* NODISTO */

#ifndef CONFIG

#define	cdr_x		JFETdCoeffs[0]
#define	cdr_y		JFETdCoeffs[1]
#define	cdr_x2		JFETdCoeffs[2]
#define	cdr_y2		JFETdCoeffs[3]
#define	cdr_xy		JFETdCoeffs[4]
#define	cdr_x3		JFETdCoeffs[5]
#define	cdr_y3		JFETdCoeffs[6]
#define	cdr_x2y		JFETdCoeffs[7]
#define	cdr_xy2		JFETdCoeffs[8]

#define	ggs1		JFETdCoeffs[9]
#define	ggd1		JFETdCoeffs[10]
#define	ggs2		JFETdCoeffs[11]
#define	ggd2		JFETdCoeffs[12]
#define	ggs3		JFETdCoeffs[13]
#define	ggd3		JFETdCoeffs[14]
#define	capgs1		JFETdCoeffs[15]
#define	capgd1		JFETdCoeffs[16]
#define	capgs2		JFETdCoeffs[17]
#define	capgd2		JFETdCoeffs[18]
#define	capgs3		JFETdCoeffs[19]
#define	capgd3		JFETdCoeffs[20]

#endif

/* indices to an array of JFET noise sources */

#define JFETRDNOIZ       0
#define JFETRSNOIZ       1
#define JFETIDNOIZ       2
#define JFETFLNOIZ 3
#define JFETTOTNOIZ    4

#define JFETNSRCS     5

#ifndef NONOISE
    double JFETnVar[NSTATVARS][JFETNSRCS];
#else /* NONOISE */
	double **JFETnVar;
#endif /* NONOISE */

    unsigned JFEToff :1;            /* 'off' flag for jfet */
    unsigned JFETareaGiven  : 1;    /* flag to indicate area was specified */
    unsigned JFETmGiven     : 1;    /* flag to indicate parallel multiplier given */
    unsigned JFETicVDSGiven : 1;    /* initial condition given flag for V D-S*/
    unsigned JFETicVGSGiven : 1;    /* initial condition given flag for V G-S*/
    unsigned JFETtempGiven  : 1;    /* flag to indicate instance temp given */
    unsigned JFETdtempGiven : 1;    /* flag to indicate instance dtemp given */


    double JFETarea;    /* area factor for the jfet */
    double JFETm;       /* Parallel multiplier */
    double JFETicVDS;   /* initial condition voltage D-S*/
    double JFETicVGS;   /* initial condition voltage G-S*/
    double JFETtemp;    /* operating temperature */
    double JFETdtemp;   /* instance temperature difference */
    double JFETtSatCur; /* temperature adjusted saturation current */
    double JFETtGatePot;    /* temperature adjusted gate potential */
    double JFETtCGS;    /* temperature corrected G-S capacitance */
    double JFETtCGD;    /* temperature corrected G-D capacitance */
    double JFETcorDepCap;   /* joining point of the fwd bias dep. cap eq.s */
    double JFETvcrit;   /* critical voltage for the instance */
    double JFETf1;      /* coefficient of capacitance polynomial exp */


} JFETinstance ;

#define JFETvgs JFETstate 
#define JFETvgd JFETstate+1 
#define JFETcg JFETstate+2 
#define JFETcd JFETstate+3 
#define JFETcgd JFETstate+4 
#define JFETgm JFETstate+5 
#define JFETgds JFETstate+6 
#define JFETggs JFETstate+7 
#define JFETggd JFETstate+8 
#define JFETqgs JFETstate+9 
#define JFETcqgs JFETstate+10 
#define JFETqgd JFETstate+11 
#define JFETcqgd JFETstate+12 

/* per model data */

typedef struct sJFETmodel {       /* model structure for a jfet */
    int JFETmodType;    /* type index of this device type */
    struct sJFETmodel *JFETnextModel;   /* pointer to next possible model in 
                                         * linked list */
    JFETinstance * JFETinstances; /* pointer to list of instances 
                                   * that have this model */
    IFuid JFETmodName; /* pointer to character string naming this model */
    int JFETtype;

    double JFETthreshold;
    double JFETbeta;
    double JFETlModulation;
    double JFETdrainResist;
    double JFETsourceResist;
    double JFETcapGS;
    double JFETcapGD;
    double JFETgatePotential;
    double JFETgateSatCurrent;
    double JFETdepletionCapCoeff;
    double JFETfNcoef;
    double JFETfNexp;


    double JFETdrainConduct;
    double JFETsourceConduct;
    double JFETf2;
    double JFETf3;
    /* Modification for Sydney University JFET model */
    double JFETb;     /* doping profile parameter */
    double JFETbFac;  /* internal derived doping profile parameter */
    /* end Sydney University mod */
    double JFETtnom;    /* temperature at which parameters were measured */

    unsigned JFETthresholdGiven : 1;
    unsigned JFETbetaGiven : 1;
    unsigned JFETlModulationGiven : 1;
    unsigned JFETdrainResistGiven : 1;
    unsigned JFETsourceResistGiven : 1;
    unsigned JFETcapGSGiven : 1;
    unsigned JFETcapGDGiven : 1;
    unsigned JFETgatePotentialGiven : 1;
    unsigned JFETgateSatCurrentGiven : 1;
    unsigned JFETdepletionCapCoeffGiven : 1;
    /* Modification for Sydney University JFET model */
    unsigned JFETbGiven : 1;
    /* end Sydney University mod */
    unsigned JFETtnomGiven : 1; /* user specified Tnom for model */
    unsigned JFETfNcoefGiven : 1;
    unsigned JFETfNexpGiven : 1;


} JFETmodel;

#ifndef NJF

#define NJF 1
#define PJF -1

#endif /*NJF*/

/* device parameters */
#define JFET_AREA 1
#define JFET_IC_VDS 2
#define JFET_IC_VGS 3
#define JFET_IC 4
#define JFET_OFF 5
#define JFET_TEMP 6
#define JFET_DTEMP 7
#define JFET_M 8

/* model parameters */
#define JFET_MOD_VTO 101
#define JFET_MOD_BETA 102
#define JFET_MOD_LAMBDA 103
#define JFET_MOD_RD 104
#define JFET_MOD_RS 105
#define JFET_MOD_CGS 106
#define JFET_MOD_CGD 107
#define JFET_MOD_PB 108
#define JFET_MOD_IS 109
#define JFET_MOD_FC 110 
#define JFET_MOD_NJF 111
#define JFET_MOD_PJF 112
#define JFET_MOD_TNOM 113
#define JFET_MOD_KF 114
#define JFET_MOD_AF 115
/* Modification for Sydney University JFET model */
#define JFET_MOD_B 116
/* end Sydney University mod */

/* device questions */
#define JFET_DRAINNODE        301
#define JFET_GATENODE         302
#define JFET_SOURCENODE       303
#define JFET_DRAINPRIMENODE   304
#define JFET_SOURCEPRIMENODE  305
#define JFET_VGS              306
#define JFET_VGD              307
#define JFET_CG               308
#define JFET_CD               309
#define JFET_CGD              310
#define JFET_GM               311
#define JFET_GDS              312
#define JFET_GGS              313
#define JFET_GGD              314
#define JFET_QGS              315
#define JFET_CQGS             316
#define JFET_QGD              317
#define JFET_CQGD             318
#define JFET_CS               319
#define JFET_POWER            320

/* model questions */
#define JFET_MOD_DRAINCONDUCT   301
#define JFET_MOD_SOURCECONDUCT  302
#define JFET_MOD_DEPLETIONCAP   303
#define JFET_MOD_VCRIT          304
#define JFET_MOD_TYPE           305

/* function definitions */

#include "jfetext.h"

#endif /*JFET*/
