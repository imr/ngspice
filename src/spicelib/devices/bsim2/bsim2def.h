/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Min-Chie Jeng, Hong June Park, Thomas L. Quarles
**********/

#ifndef BSIM2
#define BSIM2

#include "ngspice/ifsim.h"
#include "ngspice/gendefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"

    /* declarations for B2 MOSFETs */

/* information needed for each instance */

typedef struct sBSIM2instance {

    struct GENinstance gen;

#define B2modPtr(inst) ((struct sBSIM2model *)((inst)->gen.GENmodPtr))
#define B2nextInstance(inst) ((struct sBSIM2instance *)((inst)->gen.GENnextInstance))
#define B2name gen.GENname
#define B2states gen.GENstate

    const int B2dNode;  /* number of the gate node of the mosfet */
    const int B2gNode;  /* number of the gate node of the mosfet */
    const int B2sNode;  /* number of the source node of the mosfet */
    const int B2bNode;  /* number of the bulk node of the mosfet */
    int B2dNodePrime; /* number of the internal drain node of the mosfet */
    int B2sNodePrime; /* number of the internal source node of the mosfet */

    double B2l;   /* the length of the channel region */
    double B2w;   /* the width of the channel region */
    double B2m;   /* the parallel multiplier */
    double B2drainArea;   /* the area of the drain diffusion */
    double B2sourceArea;  /* the area of the source diffusion */
    double B2drainSquares;    /* the length of the drain in squares */
    double B2sourceSquares;   /* the length of the source in squares */
    double B2drainPerimeter;
    double B2sourcePerimeter;
    double B2sourceConductance;   /* cond. of source (or 0): set in setup */
    double B2drainConductance;    /* cond. of drain (or 0): set in setup */

    double B2icVBS;   /* initial condition B-S voltage */
    double B2icVDS;   /* initial condition D-S voltage */
    double B2icVGS;   /* initial condition G-S voltage */
    double B2von;
    double B2vdsat;
    int B2off;        /* non-zero to indicate device is off for dc analysis*/
    int B2mode;       /* device mode : 1 = normal, -1 = inverse */

    struct bsim2SizeDependParam  *pParam;


    unsigned B2lGiven :1;
    unsigned B2wGiven :1;
    unsigned B2mGiven :1;
    unsigned B2drainAreaGiven :1;
    unsigned B2sourceAreaGiven    :1;
    unsigned B2drainSquaresGiven  :1;
    unsigned B2sourceSquaresGiven :1;
    unsigned B2drainPerimeterGiven    :1;
    unsigned B2sourcePerimeterGiven   :1;
    unsigned B2dNodePrimeSet  :1;
    unsigned B2sNodePrimeSet  :1;
    unsigned B2icVBSGiven :1;
    unsigned B2icVDSGiven :1;
    unsigned B2icVGSGiven :1;
    unsigned B2vonGiven   :1;
    unsigned B2vdsatGiven :1;


    double *B2DdPtr;      /* pointer to sparse matrix element at
                                     * (Drain node,drain node) */
    double *B2GgPtr;      /* pointer to sparse matrix element at
                                     * (gate node,gate node) */
    double *B2SsPtr;      /* pointer to sparse matrix element at
                                     * (source node,source node) */
    double *B2BbPtr;      /* pointer to sparse matrix element at
                                     * (bulk node,bulk node) */
    double *B2DPdpPtr;    /* pointer to sparse matrix element at
                                     * (drain prime node,drain prime node) */
    double *B2SPspPtr;    /* pointer to sparse matrix element at
                                     * (source prime node,source prime node) */
    double *B2DdpPtr;     /* pointer to sparse matrix element at
                                     * (drain node,drain prime node) */
    double *B2GbPtr;      /* pointer to sparse matrix element at
                                     * (gate node,bulk node) */
    double *B2GdpPtr;     /* pointer to sparse matrix element at
                                     * (gate node,drain prime node) */
    double *B2GspPtr;     /* pointer to sparse matrix element at
                                     * (gate node,source prime node) */
    double *B2SspPtr;     /* pointer to sparse matrix element at
                                     * (source node,source prime node) */
    double *B2BdpPtr;     /* pointer to sparse matrix element at
                                     * (bulk node,drain prime node) */
    double *B2BspPtr;     /* pointer to sparse matrix element at
                                     * (bulk node,source prime node) */
    double *B2DPspPtr;    /* pointer to sparse matrix element at
                                     * (drain prime node,source prime node) */
    double *B2DPdPtr;     /* pointer to sparse matrix element at
                                     * (drain prime node,drain node) */
    double *B2BgPtr;      /* pointer to sparse matrix element at
                                     * (bulk node,gate node) */
    double *B2DPgPtr;     /* pointer to sparse matrix element at
                                     * (drain prime node,gate node) */

    double *B2SPgPtr;     /* pointer to sparse matrix element at
                                     * (source prime node,gate node) */
    double *B2SPsPtr;     /* pointer to sparse matrix element at
                                     * (source prime node,source node) */
    double *B2DPbPtr;     /* pointer to sparse matrix element at
                                     * (drain prime node,bulk node) */
    double *B2SPbPtr;     /* pointer to sparse matrix element at
                                     * (source prime node,bulk node) */
    double *B2SPdpPtr;    /* pointer to sparse matrix element at
                                     * (source prime node,drain prime node) */

#define B2RDNOIZ       0
#define B2RSNOIZ       1
#define B2IDNOIZ       2
#define B2FLNOIZ       3
#define B2TOTNOIZ      4

#define B2NSRCS        5     /* the number of BSIM2 noise sources */

#ifndef NONOISE
    double B2nVar[NSTATVARS][B2NSRCS];
#else /* NONOISE */
	double **B2nVar;
#endif /* NONOISE */

#define B2vbd B2states+ 0
#define B2vbs B2states+ 1
#define B2vgs B2states+ 2
#define B2vds B2states+ 3
#define B2cd B2states+ 4
#define B2id B2states+ 4
#define B2cbs B2states+ 5
#define B2ibs B2states+ 5
#define B2cbd B2states+ 6
#define B2ibd B2states+ 6
#define B2gm B2states+ 7
#define B2gds B2states+ 8
#define B2gmbs B2states+ 9
#define B2gbd B2states+ 10
#define B2gbs B2states+ 11
#define B2qb B2states+ 12
#define B2cqb B2states+ 13
#define B2iqb B2states+ 13
#define B2qg B2states+ 14
#define B2cqg B2states+ 15
#define B2iqg B2states+ 15
#define B2qd B2states+ 16
#define B2cqd B2states+ 17
#define B2iqd B2states+ 17
#define B2cggb B2states+ 18
#define B2cgdb B2states+ 19
#define B2cgsb B2states+ 20
#define B2cbgb B2states+ 21
#define B2cbdb B2states+ 22
#define B2cbsb B2states+ 23
#define B2capbd B2states+ 24
#define B2iqbd B2states+ 25
#define B2cqbd B2states+ 25
#define B2capbs B2states+ 26
#define B2iqbs B2states+ 27
#define B2cqbs B2states+ 27
#define B2cdgb B2states+ 28
#define B2cddb B2states+ 29
#define B2cdsb B2states+ 30
#define B2vono B2states+ 31
#define B2vdsato B2states+ 32
#define B2qbs  B2states+ 33
#define B2qbd  B2states+ 34

#define B2numStates 35           

} B2instance ;

struct bsim2SizeDependParam
{
    double Width;
    double Length;
    double B2vfb;      /* flat band voltage at given L and W */
    double B2phi;      /* surface potential at strong inversion */
    double B2k1;       /* bulk effect coefficient 1             */
    double B2k2;       /* bulk effect coefficient 2             */
    double B2eta0;      /* drain induced barrier lowering        */
    double B2etaB;     /* Vbs dependence of Eta                 */
    double B2beta0;    /* Beta at Vds = 0 and Vgs = Vth         */
    double B2beta0B;   /* Vbs dependence of Beta0               */
    double B2betas0;    /* Beta at Vds=Vdd and Vgs=Vth           */
    double B2betasB;   /* Vbs dependence of Betas               */
    double B2beta20;   /* Vds dependence of Beta in tanh term   */
    double B2beta2B;   /* Vbs dependence of Beta2               */
    double B2beta2G;   /* Vgs dependence of Beta2               */
    double B2beta30;   /* Vds dependence of Beta in linear term */
    double B2beta3B;   /* Vbs dependence of Beta3               */
    double B2beta3G;   /* Vgs dependence of Beta3               */
    double B2beta40;   /* Vds dependence of Beta in quadra term */
    double B2beta4B;   /* Vbs dependence of Beta4               */
    double B2beta4G;   /* Vgs dependence of Beta4               */
    double B2ua0;      /* Linear Vgs dependence of Mobility     */
    double B2uaB;      /* Vbs dependence of Ua                  */
    double B2ub0;      /* Quadratic Vgs dependence of Mobility  */
    double B2ubB;      /* Vbs dependence of Ub                  */
    double B2u10;      /* Drift Velocity Saturation due to Vds  */
    double B2u1B;      /* Vbs dependence of U1                  */
    double B2u1D;      /* Vds dependence of U1                  */
    double B2n0;       /* Subthreshold slope at Vds=0, Vbs=0    */
    double B2nB;       /* Vbs dependence of n                   */
    double B2nD;       /* Vds dependence of n                   */
    double B2vof0;     /* Vth offset at Vds=0, Vbs=0            */
    double B2vofB;     /* Vbs dependence of Vof                 */
    double B2vofD;     /* Vds dependence of Vof                 */
    double B2ai0;      /* Pre-factor in hot-electron effects    */
    double B2aiB;      /* Vbs dependence of Ai                  */
    double B2bi0;      /* Exp-factor in hot-electron effects    */
    double B2biB;      /* Vbs dependence of Bi                  */
    double B2vghigh;   /* Upper bound of cubic spline function  */
    double B2vglow;    /* Lower bound of cubic spline function  */
    double B2GDoverlapCap;/* Gate Drain Overlap Capacitance     */
    double B2GSoverlapCap;/* Gate Source Overlap Capacitance    */
    double B2GBoverlapCap;/* Gate Bulk Overlap Capacitance      */
    double SqrtPhi;
    double Phis3;
    double CoxWL;
    double One_Third_CoxWL;
    double Two_Third_CoxWL;
    double Arg;
    double B2vt0;
    struct bsim2SizeDependParam  *pNext;
};


/* per model data */

typedef struct sBSIM2model {       	/* model structure for a resistor */

    struct GENmodel gen;

#define B2modType gen.GENmodType
#define B2nextModel(inst) ((struct sBSIM2model *)((inst)->gen.GENnextModel))
#define B2instances(inst) ((B2instance *)((inst)->gen.GENinstances))
#define B2modName gen.GENmodName

    int B2type;       		/* device type: 1 = nmos,  -1 = pmos */
    int pad;

    double B2vfb0;
    double B2vfbL;
    double B2vfbW;
    double B2phi0;
    double B2phiL;
    double B2phiW;
    double B2k10;
    double B2k1L;
    double B2k1W;
    double B2k20;
    double B2k2L;
    double B2k2W;
    double B2eta00;
    double B2eta0L;
    double B2eta0W;
    double B2etaB0;
    double B2etaBL;
    double B2etaBW;
    double B2deltaL;
    double B2deltaW;
    double B2mob00;
    double B2mob0B0;
    double B2mob0BL;
    double B2mob0BW ;
    double B2mobs00;
    double B2mobs0L;
    double B2mobs0W;
    double B2mobsB0;
    double B2mobsBL;
    double B2mobsBW;
    double B2mob200;
    double B2mob20L;
    double B2mob20W;
    double B2mob2B0;
    double B2mob2BL;
    double B2mob2BW;
    double B2mob2G0;
    double B2mob2GL;
    double B2mob2GW;
    double B2mob300;
    double B2mob30L;
    double B2mob30W;
    double B2mob3B0;
    double B2mob3BL;
    double B2mob3BW;
    double B2mob3G0;
    double B2mob3GL;
    double B2mob3GW;
    double B2mob400;
    double B2mob40L;
    double B2mob40W;
    double B2mob4B0;
    double B2mob4BL;
    double B2mob4BW;
    double B2mob4G0;
    double B2mob4GL;
    double B2mob4GW;
    double B2ua00;
    double B2ua0L;
    double B2ua0W;
    double B2uaB0;
    double B2uaBL;
    double B2uaBW;
    double B2ub00;
    double B2ub0L;
    double B2ub0W;
    double B2ubB0;
    double B2ubBL;
    double B2ubBW;
    double B2u100;
    double B2u10L;
    double B2u10W;
    double B2u1B0;
    double B2u1BL;
    double B2u1BW;
    double B2u1D0;
    double B2u1DL;
    double B2u1DW;
    double B2n00;
    double B2n0L;
    double B2n0W;
    double B2nB0;
    double B2nBL;
    double B2nBW;
    double B2nD0;
    double B2nDL;
    double B2nDW;
    double B2vof00;
    double B2vof0L;
    double B2vof0W;
    double B2vofB0;
    double B2vofBL;
    double B2vofBW;
    double B2vofD0;
    double B2vofDL;
    double B2vofDW;
    double B2ai00;
    double B2ai0L;
    double B2ai0W;
    double B2aiB0;
    double B2aiBL;
    double B2aiBW;
    double B2bi00;
    double B2bi0L;
    double B2bi0W;
    double B2biB0;
    double B2biBL;
    double B2biBW;
    double B2vghigh0;
    double B2vghighL;
    double B2vghighW;
    double B2vglow0;
    double B2vglowL;
    double B2vglowW;
    double B2tox;              /* unit: micron  */
    double B2Cox;                         /* unit: F/cm**2 */
    double B2temp;
    double B2vdd;
    double B2vdd2;
    double B2vgg;
    double B2vgg2;
    double B2vbb;
    double B2vbb2;
    double B2gateSourceOverlapCap;
    double B2gateDrainOverlapCap;
    double B2gateBulkOverlapCap;
    double B2Vtm;

    double B2sheetResistance;
    double B2jctSatCurDensity;
    double B2bulkJctPotential;
    double B2bulkJctBotGradingCoeff;
    double B2bulkJctSideGradingCoeff;
    double B2sidewallJctPotential;
    double B2unitAreaJctCap;
    double B2unitLengthSidewallJctCap;
    double B2defaultWidth;
    double B2deltaLength; 
    double B2fNcoef;
    double B2fNexp;
    int B2channelChargePartitionFlag;


    struct bsim2SizeDependParam  *pSizeDependParamKnot;


    unsigned  B2vfb0Given   :1;
    unsigned  B2vfbLGiven   :1;
    unsigned  B2vfbWGiven   :1;
    unsigned  B2phi0Given   :1;
    unsigned  B2phiLGiven   :1;
    unsigned  B2phiWGiven   :1;
    unsigned  B2k10Given   :1;
    unsigned  B2k1LGiven   :1;
    unsigned  B2k1WGiven   :1;
    unsigned  B2k20Given   :1;
    unsigned  B2k2LGiven   :1;
    unsigned  B2k2WGiven   :1;
    unsigned  B2eta00Given   :1;
    unsigned  B2eta0LGiven   :1;
    unsigned  B2eta0WGiven   :1;
    unsigned  B2etaB0Given   :1;
    unsigned  B2etaBLGiven   :1;
    unsigned  B2etaBWGiven   :1;
    unsigned  B2deltaLGiven   :1;
    unsigned  B2deltaWGiven   :1;
    unsigned  B2mob00Given   :1;
    unsigned  B2mob0B0Given   :1;
    unsigned  B2mob0BLGiven   :1;
    unsigned  B2mob0BWGiven   :1;
    unsigned  B2mobs00Given   :1;
    unsigned  B2mobs0LGiven   :1;
    unsigned  B2mobs0WGiven   :1;
    unsigned  B2mobsB0Given   :1;
    unsigned  B2mobsBLGiven   :1;
    unsigned  B2mobsBWGiven   :1;
    unsigned  B2mob200Given   :1;
    unsigned  B2mob20LGiven   :1;
    unsigned  B2mob20WGiven   :1;
    unsigned  B2mob2B0Given   :1;
    unsigned  B2mob2BLGiven   :1;
    unsigned  B2mob2BWGiven   :1;
    unsigned  B2mob2G0Given   :1;
    unsigned  B2mob2GLGiven   :1;
    unsigned  B2mob2GWGiven   :1;
    unsigned  B2mob300Given   :1;
    unsigned  B2mob30LGiven   :1;
    unsigned  B2mob30WGiven   :1;
    unsigned  B2mob3B0Given   :1;
    unsigned  B2mob3BLGiven   :1;
    unsigned  B2mob3BWGiven   :1;
    unsigned  B2mob3G0Given   :1;
    unsigned  B2mob3GLGiven   :1;
    unsigned  B2mob3GWGiven   :1;
    unsigned  B2mob400Given   :1;
    unsigned  B2mob40LGiven   :1;
    unsigned  B2mob40WGiven   :1;
    unsigned  B2mob4B0Given   :1;
    unsigned  B2mob4BLGiven   :1;
    unsigned  B2mob4BWGiven   :1;
    unsigned  B2mob4G0Given   :1;
    unsigned  B2mob4GLGiven   :1;
    unsigned  B2mob4GWGiven   :1;
    unsigned  B2ua00Given   :1;
    unsigned  B2ua0LGiven   :1;
    unsigned  B2ua0WGiven   :1;
    unsigned  B2uaB0Given   :1;
    unsigned  B2uaBLGiven   :1;
    unsigned  B2uaBWGiven   :1;
    unsigned  B2ub00Given   :1;
    unsigned  B2ub0LGiven   :1;
    unsigned  B2ub0WGiven   :1;
    unsigned  B2ubB0Given   :1;
    unsigned  B2ubBLGiven   :1;
    unsigned  B2ubBWGiven   :1;
    unsigned  B2u100Given   :1;
    unsigned  B2u10LGiven   :1;
    unsigned  B2u10WGiven   :1;
    unsigned  B2u1B0Given   :1;
    unsigned  B2u1BLGiven   :1;
    unsigned  B2u1BWGiven   :1;
    unsigned  B2u1D0Given   :1;
    unsigned  B2u1DLGiven   :1;
    unsigned  B2u1DWGiven   :1;
    unsigned  B2n00Given   :1;
    unsigned  B2n0LGiven   :1;
    unsigned  B2n0WGiven   :1;
    unsigned  B2nB0Given   :1;
    unsigned  B2nBLGiven   :1;
    unsigned  B2nBWGiven   :1;
    unsigned  B2nD0Given   :1;
    unsigned  B2nDLGiven   :1;
    unsigned  B2nDWGiven   :1;
    unsigned  B2vof00Given   :1;
    unsigned  B2vof0LGiven   :1;
    unsigned  B2vof0WGiven   :1;
    unsigned  B2vofB0Given   :1;
    unsigned  B2vofBLGiven   :1;
    unsigned  B2vofBWGiven   :1;
    unsigned  B2vofD0Given   :1;
    unsigned  B2vofDLGiven   :1;
    unsigned  B2vofDWGiven   :1;
    unsigned  B2ai00Given   :1;
    unsigned  B2ai0LGiven   :1;
    unsigned  B2ai0WGiven   :1;
    unsigned  B2aiB0Given   :1;
    unsigned  B2aiBLGiven   :1;
    unsigned  B2aiBWGiven   :1;
    unsigned  B2bi00Given   :1;
    unsigned  B2bi0LGiven   :1;
    unsigned  B2bi0WGiven   :1;
    unsigned  B2biB0Given   :1;
    unsigned  B2biBLGiven   :1;
    unsigned  B2biBWGiven   :1;
    unsigned  B2vghigh0Given    :1;
    unsigned  B2vghighLGiven    :1;
    unsigned  B2vghighWGiven    :1;
    unsigned  B2vglow0Given     :1;
    unsigned  B2vglowLGiven     :1;
    unsigned  B2vglowWGiven     :1;
    unsigned  B2toxGiven   :1;
    unsigned  B2tempGiven   :1;
    unsigned  B2vddGiven   :1;
    unsigned  B2vggGiven   :1;
    unsigned  B2vbbGiven   :1;
    unsigned  B2gateSourceOverlapCapGiven   :1;
    unsigned  B2gateDrainOverlapCapGiven   :1;
    unsigned  B2gateBulkOverlapCapGiven   :1;
    unsigned  B2channelChargePartitionFlagGiven   :1;
    unsigned  B2sheetResistanceGiven   :1;
    unsigned  B2jctSatCurDensityGiven   :1;
    unsigned  B2bulkJctPotentialGiven   :1;
    unsigned  B2bulkJctBotGradingCoeffGiven   :1;
    unsigned  B2sidewallJctPotentialGiven   :1;
    unsigned  B2bulkJctSideGradingCoeffGiven   :1;
    unsigned  B2unitAreaJctCapGiven   :1;
    unsigned  B2unitLengthSidewallJctCapGiven   :1;
    unsigned  B2defaultWidthGiven   :1;
    unsigned  B2deltaLengthGiven   :1;
    unsigned  B2fNcoefGiven :1;
    unsigned  B2fNexpGiven :1;
    unsigned  B2typeGiven   :1;

} B2model;


#ifndef NMOS
#define NMOS 1
#define PMOS -1
#endif /*NMOS*/


/* device parameters */
#define BSIM2_W 1
#define BSIM2_L 2
#define BSIM2_AS 3
#define BSIM2_AD 4
#define BSIM2_PS 5
#define BSIM2_PD 6
#define BSIM2_NRS 7
#define BSIM2_NRD 8
#define BSIM2_OFF 9
#define BSIM2_IC_VBS 10
#define BSIM2_IC_VDS 11
#define BSIM2_IC_VGS 12
#define BSIM2_IC 13
#define BSIM2_M 14

/* model parameters */
#define BSIM2_MOD_VFB0 101
#define BSIM2_MOD_VFBL 102
#define BSIM2_MOD_VFBW 103
#define BSIM2_MOD_PHI0 104
#define BSIM2_MOD_PHIL 105
#define BSIM2_MOD_PHIW 106
#define BSIM2_MOD_K10 107
#define BSIM2_MOD_K1L 108
#define BSIM2_MOD_K1W 109
#define BSIM2_MOD_K20 110
#define BSIM2_MOD_K2L 111
#define BSIM2_MOD_K2W 112
#define BSIM2_MOD_ETA00 113
#define BSIM2_MOD_ETA0L 114
#define BSIM2_MOD_ETA0W 115
#define BSIM2_MOD_ETAB0 116
#define BSIM2_MOD_ETABL 117
#define BSIM2_MOD_ETABW 118
#define BSIM2_MOD_DELTAL 119
#define BSIM2_MOD_DELTAW 120
#define BSIM2_MOD_MOB00 121
#define BSIM2_MOD_MOB0B0 122
#define BSIM2_MOD_MOB0BL 123
#define BSIM2_MOD_MOB0BW 124
#define BSIM2_MOD_MOBS00 125
#define BSIM2_MOD_MOBS0L 126
#define BSIM2_MOD_MOBS0W 127
#define BSIM2_MOD_MOBSB0 128
#define BSIM2_MOD_MOBSBL 129
#define BSIM2_MOD_MOBSBW 130
#define BSIM2_MOD_MOB200 131
#define BSIM2_MOD_MOB20L 132
#define BSIM2_MOD_MOB20W 133
#define BSIM2_MOD_MOB2B0 134
#define BSIM2_MOD_MOB2BL 135
#define BSIM2_MOD_MOB2BW 136
#define BSIM2_MOD_MOB2G0 137
#define BSIM2_MOD_MOB2GL 138
#define BSIM2_MOD_MOB2GW 139
#define BSIM2_MOD_MOB300 140
#define BSIM2_MOD_MOB30L 141
#define BSIM2_MOD_MOB30W 142
#define BSIM2_MOD_MOB3B0 143
#define BSIM2_MOD_MOB3BL 144
#define BSIM2_MOD_MOB3BW 145
#define BSIM2_MOD_MOB3G0 146
#define BSIM2_MOD_MOB3GL 147
#define BSIM2_MOD_MOB3GW 148
#define BSIM2_MOD_MOB400 149
#define BSIM2_MOD_MOB40L 150
#define BSIM2_MOD_MOB40W 151
#define BSIM2_MOD_MOB4B0 152
#define BSIM2_MOD_MOB4BL 153
#define BSIM2_MOD_MOB4BW 154
#define BSIM2_MOD_MOB4G0 155
#define BSIM2_MOD_MOB4GL 156
#define BSIM2_MOD_MOB4GW 157
#define BSIM2_MOD_UA00 158
#define BSIM2_MOD_UA0L 159
#define BSIM2_MOD_UA0W 160
#define BSIM2_MOD_UAB0 161
#define BSIM2_MOD_UABL 162
#define BSIM2_MOD_UABW 163
#define BSIM2_MOD_UB00 164
#define BSIM2_MOD_UB0L 165
#define BSIM2_MOD_UB0W 166
#define BSIM2_MOD_UBB0 167
#define BSIM2_MOD_UBBL 168
#define BSIM2_MOD_UBBW 169
#define BSIM2_MOD_U100 170
#define BSIM2_MOD_U10L 171
#define BSIM2_MOD_U10W 172
#define BSIM2_MOD_U1B0 173
#define BSIM2_MOD_U1BL 174
#define BSIM2_MOD_U1BW 175
#define BSIM2_MOD_U1D0 176
#define BSIM2_MOD_U1DL 177
#define BSIM2_MOD_U1DW 178
#define BSIM2_MOD_N00 179
#define BSIM2_MOD_N0L 180
#define BSIM2_MOD_N0W 181
#define BSIM2_MOD_NB0 182
#define BSIM2_MOD_NBL 183
#define BSIM2_MOD_NBW 184
#define BSIM2_MOD_ND0 185
#define BSIM2_MOD_NDL 186
#define BSIM2_MOD_NDW 187
#define BSIM2_MOD_VOF00 188
#define BSIM2_MOD_VOF0L 189
#define BSIM2_MOD_VOF0W 190
#define BSIM2_MOD_VOFB0 191
#define BSIM2_MOD_VOFBL 192
#define BSIM2_MOD_VOFBW 193
#define BSIM2_MOD_VOFD0 194
#define BSIM2_MOD_VOFDL 195
#define BSIM2_MOD_VOFDW 196
#define BSIM2_MOD_AI00 197
#define BSIM2_MOD_AI0L 198
#define BSIM2_MOD_AI0W 199
#define BSIM2_MOD_AIB0 200
#define BSIM2_MOD_AIBL 201
#define BSIM2_MOD_AIBW 202
#define BSIM2_MOD_BI00 203
#define BSIM2_MOD_BI0L 204
#define BSIM2_MOD_BI0W 205
#define BSIM2_MOD_BIB0 206
#define BSIM2_MOD_BIBL 207
#define BSIM2_MOD_BIBW 208
#define BSIM2_MOD_VGHIGH0 209
#define BSIM2_MOD_VGHIGHL 210
#define BSIM2_MOD_VGHIGHW 211
#define BSIM2_MOD_VGLOW0 212
#define BSIM2_MOD_VGLOWL 213
#define BSIM2_MOD_VGLOWW 214
#define BSIM2_MOD_TOX 215
#define BSIM2_MOD_TEMP 216
#define BSIM2_MOD_VDD 217
#define BSIM2_MOD_VGG 218
#define BSIM2_MOD_VBB 219
#define BSIM2_MOD_CGSO 220
#define BSIM2_MOD_CGDO 221
#define BSIM2_MOD_CGBO 222
#define BSIM2_MOD_XPART 223
#define BSIM2_MOD_RSH 224
#define BSIM2_MOD_JS 225
#define BSIM2_MOD_PB 226
#define BSIM2_MOD_MJ 227
#define BSIM2_MOD_PBSW 228
#define BSIM2_MOD_MJSW 229
#define BSIM2_MOD_CJ 230
#define BSIM2_MOD_CJSW 231
#define BSIM2_MOD_DEFWIDTH 232
#define BSIM2_MOD_DELLENGTH 233
#define BSIM2_MOD_NMOS 234
#define BSIM2_MOD_PMOS 235

#define BSIM2_MOD_KF 236
#define BSIM2_MOD_AF 237

/* device questions */
#define BSIM2_DNODE              241
#define BSIM2_GNODE              242
#define BSIM2_SNODE              243
#define BSIM2_BNODE              244
#define BSIM2_DNODEPRIME         245
#define BSIM2_SNODEPRIME         246
#define BSIM2_VBD                247
#define BSIM2_VBS                248
#define BSIM2_VGS                249
#define BSIM2_VDS                250
#define BSIM2_CD         251
#define BSIM2_CBS        252
#define BSIM2_CBD        253
#define BSIM2_GM         254
#define BSIM2_GDS        255
#define BSIM2_GMBS       256
#define BSIM2_GBD        257
#define BSIM2_GBS        258
#define BSIM2_QB         259
#define BSIM2_CQB        260
#define BSIM2_QG         261
#define BSIM2_CQG        262
#define BSIM2_QD         263
#define BSIM2_CQD        264
#define BSIM2_CGG        265
#define BSIM2_CGD        266
#define BSIM2_CGS        267
#define BSIM2_CBG        268
#define BSIM2_CAPBD      269
#define BSIM2_CQBD       270
#define BSIM2_CAPBS      271
#define BSIM2_CQBS       272
#define BSIM2_CDG        273
#define BSIM2_CDD        274
#define BSIM2_CDS        275
#define BSIM2_VON        276
#define BSIM2_QBS        277
#define BSIM2_QBD        278
#define BSIM2_SOURCECONDUCT      279
#define BSIM2_DRAINCONDUCT       280

/* model questions */

#include "bsim2ext.h"

extern void B2evaluate(double,double,double,B2instance*,B2model*,
        double*,double*,double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, CKTcircuit*);


#endif /*B2*/

