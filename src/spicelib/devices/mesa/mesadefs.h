/**********
Copyright 1993: T. Ytterdal, K. Lee, M. Shur and T. A. Fjeldly. All rights reserved.
Author: Trond Ytterdal
**********/

#ifndef MESA
#define MESA

#include "ifsim.h"
#include "cktdefs.h"
#include "gendefs.h"
#include "complex.h"
#include "noisedef.h"

    /* structures used to describe MESFET Transistors */


/* information used to describe a single instance */

typedef struct sMESAinstance {
    struct sMESAmodel *MESAmodPtr;    /* backpointer to model */
    struct sMESAinstance *MESAnextInstance; /* pointer to next instance of 
                                             * current model*/
    IFuid MESAname; /* pointer to character string naming this instance */
    int MESAowner;  /* number of owner process */
    int MESAstate; /* pointer to start of state vector for MESAfet */
    
    int MESAdrainNode;  /* number of drain node of MESAfet */
    int MESAgateNode;   /* number of gate node of MESAfet */
    int MESAsourceNode; /* number of source node of MESAfet */
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

/* indices to the array of MESAFET noise sources */

#define MESARDNOIZ       0
#define MESARSNOIZ       1
#define MESAIDNOIZ       2
#define MESAFLNOIZ       3
#define MESATOTNOIZ      4

#define MESANSRCS        5     /* the number of MESAFET noise sources */

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
    int MESAmodType; /* type index of this device type */
    struct sMESAmodel *MESAnextModel;   /* pointer to next possible model in 
                                         * linked list */
    MESAinstance * MESAinstances; /* pointer to list of instances 
                                   * that have this model */
    IFuid MESAmodName; /* pointer to character string naming this model */
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
#define MESA_LENGTH   1
#define MESA_WIDTH    2
#define MESA_IC_VDS   3
#define MESA_IC_VGS   4
#define MESA_TD       5
#define MESA_TS       6
#define MESA_IC       7
#define MESA_OFF      8
#define MESA_CS       9
#define MESA_POWER   10
#define MESA_DTEMP   11
#define MESA_M       12   

/* model parameters */
#define MESA_MOD_VTO      101
#define MESA_MOD_VS       102
#define MESA_MOD_LAMBDA   103
#define MESA_MOD_RD       104
#define MESA_MOD_RS       105
#define MESA_MOD_RG       106
#define MESA_MOD_RI       107
#define MESA_MOD_RF       108
#define MESA_MOD_RDI      109
#define MESA_MOD_RSI      110
#define MESA_MOD_PHIB     111
#define MESA_MOD_PHIB1    112
#define MESA_MOD_ASTAR    113
#define MESA_MOD_GGR      114
#define MESA_MOD_DEL      115
#define MESA_MOD_XCHI     116
#define MESA_MOD_N        117
#define MESA_MOD_ETA      118
#define MESA_MOD_M        119
#define MESA_MOD_MC       120
#define MESA_MOD_SIGMA0   121
#define MESA_MOD_VSIGMAT  122
#define MESA_MOD_VSIGMA   123
#define MESA_MOD_MU       124
#define MESA_MOD_MU1      125
#define MESA_MOD_MU2      126
#define MESA_MOD_D        127
#define MESA_MOD_ND       128
#define MESA_MOD_DELTA    129
#define MESA_MOD_TC       130
#define MESA_MOD_NMF      131
#define MESA_MOD_TVTO     132
#define MESA_MOD_TLAMBDA  134
#define MESA_MOD_TETA0    135
#define MESA_MOD_TETA1    136
#define MESA_MOD_TMU      137
#define MESA_MOD_XTM0     138
#define MESA_MOD_XTM1     139
#define MESA_MOD_XTM2     140
#define MESA_MOD_KS       141
#define MESA_MOD_VSG      142
#define MESA_MOD_LAMBDAHF 143
#define MESA_MOD_TF       144
#define MESA_MOD_FLO      145
#define MESA_MOD_DELFO    146
#define MESA_MOD_AG       147
#define MESA_MOD_THETA    148
#define MESA_MOD_ALPHA    149
#define MESA_MOD_TC1      150
#define MESA_MOD_TC2      151
#define MESA_MOD_ZETA     152
#define MESA_MOD_BETA     153
#define MESA_MOD_DU       154
#define MESA_MOD_NDU      155
#define MESA_MOD_TH       156
#define MESA_MOD_NDELTA   157
#define MESA_MOD_LEVEL    158
#define MESA_MOD_NMAX     159
#define MESA_MOD_GAMMA    160
#define MESA_MOD_EPSI     161
#define MESA_MOD_CBS      162
#define MESA_MOD_CAS      163
#define MESA_MOD_PMF      164
#define MESA_MOD_TYPE     165

#define MESA_DRAINNODE       201
#define MESA_GATENODE        202
#define MESA_SOURCENODE      203
#define MESA_DRAINPRIMENODE  204
#define MESA_SOURCEPRIMENODE 205
#define MESA_GATEPRIMENODE   206

#define MESA_VGS         207
#define MESA_VGD         208
#define MESA_CG          209
#define MESA_CD          210
#define MESA_CGD         211
#define MESA_GM          212
#define MESA_GDS         213
#define MESA_GGS         214
#define MESA_GGD         215
#define MESA_QGS         216
#define MESA_CQGS        217
#define MESA_QGD         218
#define MESA_CQGD        219

#define MESA_MOD_DRAINCONDUCT    301
#define MESA_MOD_SOURCECONDUCT   302
#define MESA_MOD_GATECONDUCT     303
#define MESA_MOD_DEPLETIONCAP    304
#define MESA_MOD_VCRIT           305

#include "mesaext.h"

#endif /*MESA*/
