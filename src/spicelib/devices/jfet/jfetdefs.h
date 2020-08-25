/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Sydney University mods Copyright(c) 1989 Anthony E. Parker, David J. Skellern
	Laboratory for Communication Science Engineering
	Sydney University Department of Electrical Engineering, Australia
**********/

#ifndef JFET
#define JFET

#include "ngspice/ifsim.h"
#include "ngspice/gendefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/noisedef.h"

    /* structures used to describe Junction Field Effect Transistors */

/* indices to an array of JFET noise sources */

enum {
    JFETRDNOIZ = 0,
    JFETRSNOIZ,
    JFETIDNOIZ,
    JFETFLNOIZ,
    JFETTOTNOIZ,
    /* finally, the number of noise sources */
    JFETNSRCS
};

/* information used to describe a single instance */

typedef struct sJFETinstance {

    struct GENinstance gen;

#define JFETmodPtr(inst) ((struct sJFETmodel *)((inst)->gen.GENmodPtr))
#define JFETnextInstance(inst) ((struct sJFETinstance *)((inst)->gen.GENnextInstance))
#define JFETname gen.GENname
#define JFETstate gen.GENstate

    const int JFETdrainNode;  /* number of drain node of jfet */
    const int JFETgateNode;   /* number of gate node of jfet */
    const int JFETsourceNode; /* number of source node of jfet */
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
    double JFETtThreshold;    /* temperature adjusted threshold voltage */
    double JFETtBeta;   /* temperature adjusted beta */

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

#define JFETnumStates 13

/* per model data */

typedef struct sJFETmodel {       /* model structure for a jfet */

    struct GENmodel gen;

#define JFETmodType gen.GENmodType
#define JFETnextModel(inst) ((struct sJFETmodel *)((inst)->gen.GENnextModel))
#define JFETinstances(inst) ((JFETinstance *)((inst)->gen.GENinstances))
#define JFETmodName gen.GENmodName

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
    int JFETnlev;
    double JFETgdsnoi;

    double JFETdrainConduct;
    double JFETsourceConduct;
    double JFETf2;
    double JFETf3;
    /* Modification for Sydney University JFET model */
    double JFETb;     /* doping profile parameter */
    double JFETbFac;  /* internal derived doping profile parameter */
    /* end Sydney University mod */
    double JFETtnom;  /* temperature at which parameters were measured */
    double JFETtcv;
    double JFETvtotc;
    double JFETbex;
    double JFETbetatce;
    double JFETxti;
    double JFETeg;

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
    unsigned JFETtcvGiven : 1;
    unsigned JFETvtotcGiven : 1;
    unsigned JFETbexGiven : 1;
    unsigned JFETbetatceGiven : 1;
    unsigned JFETxtiGiven : 1;
    unsigned JFETegGiven : 1;
    unsigned JFETfNcoefGiven : 1;
    unsigned JFETfNexpGiven : 1;
    unsigned JFETnlevGiven : 1;
    unsigned JFETgdsnoiGiven : 1;

} JFETmodel;

#ifndef NJF

#define NJF 1
#define PJF -1

#endif /*NJF*/

/* device parameters */
enum {
    JFET_AREA = 1,
    JFET_IC_VDS,
    JFET_IC_VGS,
    JFET_IC,
    JFET_OFF,
    JFET_TEMP,
    JFET_DTEMP,
    JFET_M,
};

/* model parameters */
enum {
    JFET_MOD_VTO = 101,
    JFET_MOD_BETA,
    JFET_MOD_LAMBDA,
    JFET_MOD_RD,
    JFET_MOD_RS,
    JFET_MOD_CGS,
    JFET_MOD_CGD,
    JFET_MOD_PB,
    JFET_MOD_IS,
    JFET_MOD_FC,
    JFET_MOD_NJF,
    JFET_MOD_PJF,
    JFET_MOD_TNOM,
    JFET_MOD_B,  /* Modification for Sydney University JFET model */
    JFET_MOD_TCV,
    JFET_MOD_VTOTC,
    JFET_MOD_BEX,
    JFET_MOD_BETATCE,
    JFET_MOD_XTI,
    JFET_MOD_EG,
    JFET_MOD_KF,
    JFET_MOD_AF,
    JFET_MOD_NLEV,
    JFET_MOD_GDSNOI,
};

/* device questions */
enum {
    JFET_DRAINNODE = 301,
    JFET_GATENODE,
    JFET_SOURCENODE,
    JFET_DRAINPRIMENODE,
    JFET_SOURCEPRIMENODE,
    JFET_VGS,
    JFET_VGD,
    JFET_CG,
    JFET_CD,
    JFET_CGD,
    JFET_GM,
    JFET_GDS,
    JFET_GGS,
    JFET_GGD,
    JFET_QGS,
    JFET_CQGS,
    JFET_QGD,
    JFET_CQGD,
    JFET_CS,
    JFET_POWER,
};

/* model questions */
enum {
    JFET_MOD_DRAINCONDUCT = 301,
    JFET_MOD_SOURCECONDUCT,
    JFET_MOD_DEPLETIONCAP,
    JFET_MOD_VCRIT,
    JFET_MOD_TYPE,
};

/* function definitions */

#include "jfetext.h"

#endif /*JFET*/
