/**********
Copyright 1993: T. Ytterdal, K. Lee, M. Shur and T. A. Fjeldly. All rights reserved.
Author: Trond Ytterdal
**********/

#ifndef MESA
#define MESA

#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/gendefs.h"
#include "ngspice/complex.h"
#include "ngspice/noisedef.h"

    /* structures used to describe MESFET Transistors */

/* indices to the array of MESAFET noise sources */

enum {
    MESARDNOIZ = 0,
    MESARSNOIZ,
    MESAIDNOIZ,
    MESAFLNOIZ,
    MESATOTNOIZ,
    /* finally, the number of noise sources */
    MESANSRCS
};

/* information used to describe a single instance */

typedef struct sMESAinstance {

    struct GENinstance gen;

#define MESAmodPtr(inst) ((struct sMESAmodel *)((inst)->gen.GENmodPtr))
#define MESAnextInstance(inst) ((struct sMESAinstance *)((inst)->gen.GENnextInstance))
#define MESAname gen.GENname
#define MESAstate gen.GENstate

    const int MESAdrainNode;  /* number of drain node of MESAfet */
    const int MESAgateNode;   /* number of gate node of MESAfet */
    const int MESAsourceNode; /* number of source node of MESAfet */
    int MESAdrainPrimeNode; /* number of internal drain node of MESAfet */
    int MESAgatePrimeNode; /* number of internal gate node of MESAfet */
    int MESAsourcePrimeNode;    /* number of internal source node of MESAfet */
    int MESAsourcePrmPrmNode;
    int MESAdrainPrmPrmNode;
    double MESAlength;    /* length of MESAfet */
    double MESAwidth;     /* width of MESAfet */
    double MESAm;         /* Parallel Multiplier */
    double MESAicVDS;     /* initial condition voltage D-S*/
    double MESAicVGS;     /* initial condition voltage G-S*/
    double MESAtd;        /* drain temperature */
    double MESAts;        /* source temperature */
    double MESAdtemp;     /* Instance temperature difference */
    double MESAtVto;
    double MESAtLambda;
    double MESAtLambdahf;
    double MESAtEta;
    double MESAtMu;
    double MESAtPhib;
    double MESAtTheta;
    double MESAtRsi;
    double MESAtRdi;
    double MESAtRs;
    double MESAtRd;
    double MESAtRg;
    double MESAtRi;
    double MESAtRf;
    double MESAtGi;
    double MESAtGf;
    double MESAdrainConduct;
    double MESAsourceConduct;
    double MESAgateConduct;
    double *MESAdrainDrainPrimePtr;
    double *MESAgatePrimeDrainPrimePtr;
    double *MESAgatePrimeSourcePrimePtr;
    double *MESAsourceSourcePrimePtr;
    double *MESAdrainPrimeDrainPtr;
    double *MESAdrainPrimeGatePrimePtr;
    double *MESAdrainPrimeSourcePrimePtr;
    double *MESAsourcePrimeGatePrimePtr;
    double *MESAsourcePrimeSourcePtr;
    double *MESAsourcePrimeDrainPrimePtr;
    double *MESAdrainDrainPtr;
    double *MESAgatePrimeGatePrimePtr;
    double *MESAsourceSourcePtr;
    double *MESAdrainPrimeDrainPrimePtr;
    double *MESAsourcePrimeSourcePrimePtr;
    double *MESAgateGatePrimePtr;
    double *MESAgatePrimeGatePtr;
    double *MESAgateGatePtr;
    double *MESAsourcePrmPrmSourcePrmPrmPtr;
    double *MESAsourcePrmPrmSourcePrimePtr;
    double *MESAsourcePrimeSourcePrmPrmPtr;
    double *MESAsourcePrmPrmGatePrimePtr;
    double *MESAgatePrimeSourcePrmPrmPtr;
    double *MESAdrainPrmPrmDrainPrmPrmPtr;
    double *MESAdrainPrmPrmDrainPrimePtr;
    double *MESAdrainPrimeDrainPrmPrmPtr;
    double *MESAdrainPrmPrmGatePrimePtr;
    double *MESAgatePrimeDrainPrmPrmPtr;
    
#define MESAvgs   MESAstate
#define MESAvgd   MESAstate+1
#define MESAcg    MESAstate+2
#define MESAcd    MESAstate+3
#define MESAcgd   MESAstate+4
#define MESAcgs   MESAstate+5
#define MESAgm    MESAstate+6
#define MESAgds   MESAstate+7
#define MESAggs   MESAstate+8
#define MESAggd   MESAstate+9
#define MESAqgs   MESAstate+10
#define MESAcqgs  MESAstate+11
#define MESAqgd   MESAstate+12
#define MESAcqgd  MESAstate+13
#define MESAvgspp MESAstate+14
#define MESAggspp MESAstate+15
#define MESAcgspp MESAstate+16
#define MESAvgdpp MESAstate+17
#define MESAggdpp MESAstate+18
#define MESAcgdpp MESAstate+19

#define MESAnumStates 20

    int MESAoff;
    unsigned MESAlengthGiven : 1;
    unsigned MESAwidthGiven  : 1;
    unsigned MESAmGiven      : 1;
    unsigned MESAicVDSGiven  : 1;
    unsigned MESAicVGSGiven  : 1;
    unsigned MESAtdGiven     : 1;
    unsigned MESAtsGiven     : 1;
    unsigned MESAdtempGiven  : 1;

int MESAmode;
    
/*
 * naming convention:
 * x = vgs
 * y = vgd
 * z = vds
 * cdr = cdrain
 */

#define MESANDCOEFFS   27

#ifndef NODISTO
    double MESAdCoeffs[MESANDCOEFFS];
#else /* NODISTO */
    double *MESAdCoeffs;
#endif /* NODISTO */

#ifndef CONFIG

#define cdr_x       MESAdCoeffs[0]
#define cdr_z       MESAdCoeffs[1]
#define cdr_x2      MESAdCoeffs[2]
#define cdr_z2      MESAdCoeffs[3]
#define cdr_xz      MESAdCoeffs[4]
#define cdr_x3      MESAdCoeffs[5]
#define cdr_z3      MESAdCoeffs[6]
#define cdr_x2z     MESAdCoeffs[7]
#define cdr_xz2     MESAdCoeffs[8]

#define ggs3        MESAdCoeffs[9]
#define ggd3        MESAdCoeffs[10]
#define ggs2        MESAdCoeffs[11]
#define ggd2        MESAdCoeffs[12]

#define qgs_x2      MESAdCoeffs[13]
#define qgs_y2      MESAdCoeffs[14]
#define qgs_xy      MESAdCoeffs[15]
#define qgs_x3      MESAdCoeffs[16]
#define qgs_y3      MESAdCoeffs[17]
#define qgs_x2y     MESAdCoeffs[18]
#define qgs_xy2     MESAdCoeffs[19]

#define qgd_x2      MESAdCoeffs[20]
#define qgd_y2      MESAdCoeffs[21]
#define qgd_xy      MESAdCoeffs[22]
#define qgd_x3      MESAdCoeffs[23]
#define qgd_y3      MESAdCoeffs[24]
#define qgd_x2y     MESAdCoeffs[25]
#define qgd_xy2     MESAdCoeffs[26]

#endif

#ifndef NONOISE
    double MESAnVar[NSTATVARS][MESANSRCS];
#else /* NONOISE */
    double **MESAnVar;
#endif /* NONOISE */
    double MESAcsatfs;
    double MESAcsatfd;
    double MESAggrwl;
    double MESAgchi0;
    double MESAbeta;
    double MESAisatb0;
    double MESAimax;
    double MESAcf;
    double MESAfl;
    double MESAdelf;
    double MESAgds0;
    double MESAgm0;
    double MESAgm1;
    double MESAgm2;
    double MESAdelidvds0;
    double MESAdelidvds1;
    double MESAdelidgch0;
    double MESAn0;
    double MESAnsb0;
    double MESAvcrits;
    double MESAvcritd;
} MESAinstance ;


/* per model data */

typedef struct sMESAmodel {       /* model structure for a MESAfet */

    struct GENmodel gen;

#define MESAmodType gen.GENmodType
#define MESAnextModel(inst) ((struct sMESAmodel *)((inst)->gen.GENnextModel))
#define MESAinstances(inst) ((MESAinstance *)((inst)->gen.GENinstances))
#define MESAmodName gen.GENmodName

    int MESAtype;
    
    double MESAthreshold;
    double MESAlambda;
    double MESAbeta;
    double MESAvs;
    double MESAeta;
    double MESAm;
    double MESAmc;
    double MESAalpha;
    double MESAsigma0;
    double MESAvsigmat;
    double MESAvsigma;
    double MESAmu;
    double MESAtheta;
    double MESAmu1;
    double MESAmu2;
    double MESAd;
    double MESAnd;
    double MESAdu;
    double MESAndu;
    double MESAth;    
    double MESAndelta;
    double MESAdelta;
    double MESAtc;
    double MESArdi;
    double MESArsi;
    double MESAdrainResist;
    double MESAsourceResist;
    double MESAdrainConduct;
    double MESAsourceConduct;
    double MESAgateResist;
    double MESAri;
    double MESArf;
    double MESAphib;
    double MESAphib1;
    double MESAastar;
    double MESAggr;
    double MESAdel;
    double MESAxchi;    
    double MESAn;
    double MESAtvto;
    double MESAtlambda;
    double MESAteta0;
    double MESAteta1;
    double MESAtmu;
    double MESAxtm0;
    double MESAxtm1;
    double MESAxtm2;
    double MESAks;
    double MESAvsg;
    double MESAlambdahf;
    double MESAtf;
    double MESAflo;
    double MESAdelfo;
    double MESAag;
    double MESAtc1;
    double MESAtc2;
    double MESAzeta;                
    double MESAlevel;
    double MESAnmax;
    double MESAgamma;
    double MESAepsi;
    double MESAcbs;
    double MESAcas;
    double MESAvcrit;

    double MESAsigma;
    double MESAvpo;
    double MESAvpou;
    double MESAvpod;
    double MESAdeltaSqr;

    unsigned MESAthresholdGiven:1;
    unsigned MESAlambdaGiven:1;
    unsigned MESAbetaGiven:1;
    unsigned MESAvsGiven:1;
    unsigned MESAetaGiven:1;
    unsigned MESAmGiven:1;
    unsigned MESAmcGiven:1;
    unsigned MESAalphaGiven:1;
    unsigned MESAsigma0Given:1;
    unsigned MESAvsigmatGiven:1;
    unsigned MESAvsigmaGiven:1;
    unsigned MESAmuGiven:1;
    unsigned MESAthetaGiven:1;
    unsigned MESAmu1Given:1;
    unsigned MESAmu2Given:1;
    unsigned MESAdGiven:1;
    unsigned MESAndGiven:1;
    unsigned MESAduGiven:1;
    unsigned MESAnduGiven:1;
    unsigned MESAthGiven:1;
    unsigned MESAndeltaGiven:1;
    unsigned MESAdeltaGiven:1;
    unsigned MESAtcGiven:1;
    unsigned MESArdiGiven:1;
    unsigned MESArsiGiven:1;
    unsigned MESAdrainResistGiven:1;
    unsigned MESAsourceResistGiven:1;
    unsigned MESAgateResistGiven:1;
    unsigned MESAriGiven:1;
    unsigned MESArfGiven:1;
    unsigned MESAphibGiven:1;
    unsigned MESAphib1Given:1;
    unsigned MESAastarGiven:1;
    unsigned MESAggrGiven:1;
    unsigned MESAdelGiven:1;
    unsigned MESAxchiGiven:1;    
    unsigned MESAnGiven:1;
    unsigned MESAtvtoGiven:1;
    unsigned MESAtlambdaGiven:1;
    unsigned MESAteta0Given:1;
    unsigned MESAteta1Given:1;
    unsigned MESAtmuGiven:1;
    unsigned MESAxtm0Given:1;
    unsigned MESAxtm1Given:1;
    unsigned MESAxtm2Given:1;
    unsigned MESAksGiven:1;
    unsigned MESAvsgGiven:1;
    unsigned MESAlambdahfGiven:1;
    unsigned MESAtfGiven:1;
    unsigned MESAfloGiven:1;
    unsigned MESAdelfoGiven:1;
    unsigned MESAagGiven:1;
    unsigned MESAtc1Given:1;
    unsigned MESAtc2Given:1;
    unsigned MESAzetaGiven:1;
    unsigned MESAlevelGiven:1;
    unsigned MESAnmaxGiven:1;
    unsigned MESAgammaGiven:1;
    unsigned MESAepsiGiven:1;
    unsigned MESAcbsGiven:1;
    unsigned MESAcasGiven:1;
    
} MESAmodel;

#ifndef NMF

#define NMF 1
#define PMF -1
#endif

/* device parameters */
enum {
    MESA_LENGTH = 1,
    MESA_WIDTH,
    MESA_IC_VDS,
    MESA_IC_VGS,
    MESA_TD,
    MESA_TS,
    MESA_IC,
    MESA_OFF,
    MESA_CS,
    MESA_POWER,
    MESA_DTEMP,
    MESA_M,
};

/* model parameters */
enum {
    MESA_MOD_VTO = 101,
    MESA_MOD_VS,
    MESA_MOD_LAMBDA,
    MESA_MOD_RD,
    MESA_MOD_RS,
    MESA_MOD_RG,
    MESA_MOD_RI,
    MESA_MOD_RF,
    MESA_MOD_RDI,
    MESA_MOD_RSI,
    MESA_MOD_PHIB,
    MESA_MOD_PHIB1,
    MESA_MOD_ASTAR,
    MESA_MOD_GGR,
    MESA_MOD_DEL,
    MESA_MOD_XCHI,
    MESA_MOD_N,
    MESA_MOD_ETA,
    MESA_MOD_M,
    MESA_MOD_MC,
    MESA_MOD_SIGMA0,
    MESA_MOD_VSIGMAT,
    MESA_MOD_VSIGMA,
    MESA_MOD_MU,
    MESA_MOD_MU1,
    MESA_MOD_MU2,
    MESA_MOD_D,
    MESA_MOD_ND,
    MESA_MOD_DELTA,
    MESA_MOD_TC,
    MESA_MOD_NMF,
    MESA_MOD_TVTO,
};

enum {
    MESA_MOD_TLAMBDA = 134,
    MESA_MOD_TETA0,
    MESA_MOD_TETA1,
    MESA_MOD_TMU,
    MESA_MOD_XTM0,
    MESA_MOD_XTM1,
    MESA_MOD_XTM2,
    MESA_MOD_KS,
    MESA_MOD_VSG,
    MESA_MOD_LAMBDAHF,
    MESA_MOD_TF,
    MESA_MOD_FLO,
    MESA_MOD_DELFO,
    MESA_MOD_AG,
    MESA_MOD_THETA,
    MESA_MOD_ALPHA,
    MESA_MOD_TC1,
    MESA_MOD_TC2,
    MESA_MOD_ZETA,
    MESA_MOD_BETA,
    MESA_MOD_DU,
    MESA_MOD_NDU,
    MESA_MOD_TH,
    MESA_MOD_NDELTA,
    MESA_MOD_LEVEL,
    MESA_MOD_NMAX,
    MESA_MOD_GAMMA,
    MESA_MOD_EPSI,
    MESA_MOD_CBS,
    MESA_MOD_CAS,
    MESA_MOD_PMF,
    MESA_MOD_TYPE,
};

/* device questions */
enum {
    MESA_DRAINNODE = 201,
    MESA_GATENODE,
    MESA_SOURCENODE,
    MESA_DRAINPRIMENODE,
    MESA_SOURCEPRIMENODE,
    MESA_GATEPRIMENODE,
    MESA_VGS,
    MESA_VGD,
    MESA_CG,
    MESA_CD,
    MESA_CGD,
    MESA_GM,
    MESA_GDS,
    MESA_GGS,
    MESA_GGD,
    MESA_QGS,
    MESA_CQGS,
    MESA_QGD,
    MESA_CQGD,
};

/* model questions */
enum {
    MESA_MOD_DRAINCONDUCT = 301,
    MESA_MOD_SOURCECONDUCT,
    MESA_MOD_GATECONDUCT,
    MESA_MOD_DEPLETIONCAP,
    MESA_MOD_VCRIT,
};

#include "mesaext.h"

#endif /*MESA*/
