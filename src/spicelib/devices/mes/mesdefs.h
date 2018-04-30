/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 S. Hwang
**********/

#ifndef MES
#define MES

#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/gendefs.h"
#include "ngspice/complex.h"
#include "ngspice/noisedef.h"

#define MESnumStates	13

    /* structures used to describe MESFET Transistors */

/* indices to the array of MESFET noise sources */

enum {
    MESRDNOIZ = 0,
    MESRSNOIZ,
    MESIDNOIZ,
    MESFLNOIZ,
    MESTOTNOIZ,
    /* finally, the number of noise sources */
    MESNSRCS
};

/* information used to describe a single instance */

typedef struct sMESinstance {

    struct GENinstance gen;

#define MESmodPtr(inst) ((struct sMESmodel *)((inst)->gen.GENmodPtr))
#define MESnextInstance(inst) ((struct sMESinstance *)((inst)->gen.GENnextInstance))
#define MESname gen.GENname
#define MESstate gen.GENstate

    const int MESdrainNode;  /* number of drain node of mesfet */
    const int MESgateNode;   /* number of gate node of mesfet */
    const int MESsourceNode; /* number of source node of mesfet */
    int MESdrainPrimeNode; /* number of internal drain node of mesfet */
    int MESsourcePrimeNode;    /* number of internal source node of mesfet */

    double MESarea;    /* area factor for the mesfet */
    double MESm;       /* Parallel multiplier */
    double MESicVDS;   /* initial condition voltage D-S*/
    double MESicVGS;   /* initial condition voltage G-S*/
    double *MESdrainDrainPrimePtr; /* pointer to sparse matrix at 
                                     * (drain,drain prime) */
    double *MESgateDrainPrimePtr;  /* pointer to sparse matrix at 
                                     * (gate,drain prime) */
    double *MESgateSourcePrimePtr; /* pointer to sparse matrix at 
                                     * (gate,source prime) */
    double *MESsourceSourcePrimePtr;   /* pointer to sparse matrix at 
                                         * (source,source prime) */
    double *MESdrainPrimeDrainPtr; /* pointer to sparse matrix at 
                                     * (drain prime,drain) */
    double *MESdrainPrimeGatePtr;  /* pointer to sparse matrix at 
                                     * (drain prime,gate) */
    double *MESdrainPrimeSourcePrimePtr;   /* pointer to sparse matrix
                                             * (drain prime,source prime) */
    double *MESsourcePrimeGatePtr; /* pointer to sparse matrix at 
                                     * (source prime,gate) */
    double *MESsourcePrimeSourcePtr;   /* pointer to sparse matrix at 
                                         * (source prime,source) */
    double *MESsourcePrimeDrainPrimePtr;   /* pointer to sparse matrix
                                             * (source prime,drain prime) */
    double *MESdrainDrainPtr;  /* pointer to sparse matrix at 
                                 * (drain,drain) */
    double *MESgateGatePtr;    /* pointer to sparse matrix at 
                                 * (gate,gate) */
    double *MESsourceSourcePtr;    /* pointer to sparse matrix at 
                                     * (source,source) */
    double *MESdrainPrimeDrainPrimePtr;    /* pointer to sparse matrix
                                             * (drain prime,drain prime) */
    double *MESsourcePrimeSourcePrimePtr;  /* pointer to sparse matrix
                                             * (source prime,source prime) */

    int MESoff;   /* 'off' flag for mesfet */
    unsigned MESareaGiven  : 1;   /* flag to indicate area was specified */
    unsigned MESmGiven     : 1;   /* flag to indicate multiplier specified*/
    unsigned MESicVDSGiven : 1;   /* initial condition given flag for V D-S*/
    unsigned MESicVGSGiven : 1;   /* initial condition given flag for V G-S*/

int MESmode;
	
/*
 * naming convention:
 * x = vgs
 * y = vgd
 * z = vds
 * cdr = cdrain
 */

#define MESNDCOEFFS	27

#ifndef NODISTO
	double MESdCoeffs[MESNDCOEFFS];
#else /* NODISTO */
	double *MESdCoeffs;
#endif /* NODISTO */

#ifndef CONFIG

#define	cdr_x		MESdCoeffs[0]
#define	cdr_z		MESdCoeffs[1]
#define	cdr_x2		MESdCoeffs[2]
#define	cdr_z2		MESdCoeffs[3]
#define	cdr_xz		MESdCoeffs[4]
#define	cdr_x3		MESdCoeffs[5]
#define	cdr_z3		MESdCoeffs[6]
#define	cdr_x2z		MESdCoeffs[7]
#define	cdr_xz2		MESdCoeffs[8]

#define	ggs3		MESdCoeffs[9]
#define	ggd3		MESdCoeffs[10]
#define	ggs2		MESdCoeffs[11]
#define	ggd2		MESdCoeffs[12]

#define	qgs_x2		MESdCoeffs[13]
#define	qgs_y2		MESdCoeffs[14]
#define	qgs_xy		MESdCoeffs[15]
#define	qgs_x3		MESdCoeffs[16]
#define	qgs_y3		MESdCoeffs[17]
#define	qgs_x2y		MESdCoeffs[18]
#define	qgs_xy2		MESdCoeffs[19]

#define	qgd_x2		MESdCoeffs[20]
#define	qgd_y2		MESdCoeffs[21]
#define	qgd_xy		MESdCoeffs[22]
#define	qgd_x3		MESdCoeffs[23]
#define	qgd_y3		MESdCoeffs[24]
#define	qgd_x2y		MESdCoeffs[25]
#define	qgd_xy2		MESdCoeffs[26]

#endif

#ifndef NONOISE
    double MESnVar[NSTATVARS][MESNSRCS];
#else /* NONOISE */
	double **MESnVar;
#endif /* NONOISE */

} MESinstance ;

#define MESvgs MESstate 
#define MESvgd MESstate+1 
#define MEScg MESstate+2 
#define MEScd MESstate+3 
#define MEScgd MESstate+4 
#define MESgm MESstate+5 
#define MESgds MESstate+6 
#define MESggs MESstate+7 
#define MESggd MESstate+8 
#define MESqgs MESstate+9 
#define MEScqgs MESstate+10 
#define MESqgd MESstate+11 
#define MEScqgd MESstate+12 


/* per model data */

typedef struct sMESmodel {       /* model structure for a mesfet */

    struct GENmodel gen;

#define MESmodType gen.GENmodType
#define MESnextModel(inst) ((struct sMESmodel *)((inst)->gen.GENnextModel))
#define MESinstances(inst) ((MESinstance *)((inst)->gen.GENinstances))
#define MESmodName gen.GENmodName

    int MEStype;

    double MESthreshold;
    double MESalpha;
    double MESbeta;
    double MESlModulation;
    double MESb;
    double MESdrainResist;
    double MESsourceResist;
    double MEScapGS;
    double MEScapGD;
    double MESgatePotential;
    double MESgateSatCurrent;
    double MESdepletionCapCoeff;
    double MESfNcoef;
    double MESfNexp;

    double MESdrainConduct;
    double MESsourceConduct;
    double MESdepletionCap;
    double MESf1;
    double MESf2;
    double MESf3;
    double MESvcrit;

    unsigned MESthresholdGiven : 1;
    unsigned MESalphaGiven : 1;
    unsigned MESbetaGiven : 1;
    unsigned MESlModulationGiven : 1;
    unsigned MESbGiven : 1;
    unsigned MESdrainResistGiven : 1;
    unsigned MESsourceResistGiven : 1;
    unsigned MEScapGSGiven : 1;
    unsigned MEScapGDGiven : 1;
    unsigned MESgatePotentialGiven : 1;
    unsigned MESgateSatCurrentGiven : 1;
    unsigned MESdepletionCapCoeffGiven : 1;
    unsigned MESfNcoefGiven : 1;
    unsigned MESfNexpGiven : 1;


} MESmodel;

#ifndef NMF

#define NMF 1
#define PMF -1

#endif /*NMF*/

/* device parameters */
enum {
    MES_AREA = 1,
    MES_IC_VDS,
    MES_IC_VGS,
    MES_IC,
    MES_OFF,
    MES_CS,
    MES_POWER,
    MES_M,
};

/* model parameters */
enum {
    MES_MOD_VTO = 101,
    MES_MOD_ALPHA,
    MES_MOD_BETA,
    MES_MOD_LAMBDA,
    MES_MOD_B,
    MES_MOD_RD,
    MES_MOD_RS,
    MES_MOD_CGS,
    MES_MOD_CGD,
    MES_MOD_PB,
    MES_MOD_IS,
    MES_MOD_FC,
    MES_MOD_NMF,
    MES_MOD_PMF,
    MES_MOD_KF,
    MES_MOD_AF,
};

/* device questions */

enum {
    MES_DRAINNODE = 201,
    MES_GATENODE,
    MES_SOURCENODE,
    MES_DRAINPRIMENODE,
    MES_SOURCEPRIMENODE,
    MES_VGS,
    MES_VGD,
    MES_CG,
    MES_CD,
    MES_CGD,
    MES_GM,
    MES_GDS,
    MES_GGS,
    MES_GGD,
    MES_QGS,
    MES_CQGS,
    MES_QGD,
    MES_CQGD,
};

/* model questions */

enum {
    MES_MOD_DRAINCONDUCT = 301,
    MES_MOD_SOURCECONDUCT,
    MES_MOD_DEPLETIONCAP,
    MES_MOD_VCRIT,
    MES_MOD_TYPE,
};

#include "mesext.h"

#endif /*MES*/
