/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 S. Hwang
**********/

#ifndef MES
#define MES

#include "ifsim.h"
#include "cktdefs.h"
#include "gendefs.h"
#include "complex.h"
#include "noisedef.h"

#define MESnumStates	13

    /* structures used to describe MESFET Transistors */


/* information used to describe a single instance */

typedef struct sMESinstance {
    struct sMESmodel *MESmodPtr;    /* backpointer to model */
    struct sMESinstance *MESnextInstance; /* pointer to next instance of 
                                             * current model*/
    IFuid MESname; /* pointer to character string naming this instance */
    int MESowner;  /* number of owner process */
    int MESstate; /* pointer to start of state vector for mesfet */
    int MESdrainNode;  /* number of drain node of mesfet */
    int MESgateNode;   /* number of gate node of mesfet */
    int MESsourceNode; /* number of source node of mesfet */
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

/* indices to the array of MESFET noise sources */

#define MESRDNOIZ       0
#define MESRSNOIZ       1
#define MESIDNOIZ       2
#define MESFLNOIZ 3
#define MESTOTNOIZ    4

#define MESNSRCS     5     /* the number of MESFET noise sources */

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
    int MESmodType; /* type index of this device type */
    struct sMESmodel *MESnextModel;   /* pointer to next possible model in 
                                         * linked list */
    MESinstance * MESinstances; /* pointer to list of instances 
                                   * that have this model */
    IFuid MESmodName; /* pointer to character string naming this model */
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
#define MES_AREA 1
#define MES_IC_VDS 2
#define MES_IC_VGS 3
#define MES_IC 4
#define MES_OFF 5
#define MES_CS 6
#define MES_POWER 7
#define MES_M 8

/* model parameters */
#define MES_MOD_VTO 101
#define MES_MOD_ALPHA 102
#define MES_MOD_BETA 103
#define MES_MOD_LAMBDA 104
#define MES_MOD_B 105
#define MES_MOD_RD 106
#define MES_MOD_RS 107
#define MES_MOD_CGS 108
#define MES_MOD_CGD 109
#define MES_MOD_PB 110
#define MES_MOD_IS 111
#define MES_MOD_FC 112
#define MES_MOD_NMF 113
#define MES_MOD_PMF 114
#define MES_MOD_KF 115
#define MES_MOD_AF 116

/* device questions */

#define MES_DRAINNODE       201
#define MES_GATENODE        202
#define MES_SOURCENODE      203
#define MES_DRAINPRIMENODE  204
#define MES_SOURCEPRIMENODE 205

#define MES_VGS         206
#define MES_VGD         207
#define MES_CG          208
#define MES_CD          209
#define MES_CGD         210
#define MES_GM          211
#define MES_GDS         212
#define MES_GGS         213
#define MES_GGD         214
#define MES_QGS         215
#define MES_CQGS        216
#define MES_QGD         217
#define MES_CQGD        218

/* model questions */

#define MES_MOD_DRAINCONDUCT    301
#define MES_MOD_SOURCECONDUCT   302 
#define MES_MOD_DEPLETIONCAP    303
#define MES_MOD_VCRIT       304
#define MES_MOD_TYPE       305

#include "mesext.h"

#endif /*MES*/
