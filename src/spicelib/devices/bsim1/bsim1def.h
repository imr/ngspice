/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985      Hong June Park, Thomas L. Quarles
**********/

#ifndef BSIM1
#define BSIM1

#include "ngspice/ifsim.h"
#include "ngspice/gendefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/noisedef.h"

    /* declarations for B1 MOSFETs */

#define B1numStates 35           
#define B1NDCOEFFS	82

/* information needed for each instance */

typedef struct sBSIM1instance {

    struct GENinstance gen;

#define B1modPtr(inst) ((struct sBSIM1model *)((inst)->gen.GENmodPtr))
#define B1nextInstance(inst) ((struct sBSIM1instance *)((inst)->gen.GENnextInstance))
#define B1name gen.GENname
#define B1states gen.GENstate

    const int B1dNode;  /* number of the gate node of the mosfet */
    const int B1gNode;  /* number of the gate node of the mosfet */
    const int B1sNode;  /* number of the source node of the mosfet */
    const int B1bNode;  /* number of the bulk node of the mosfet */
    int B1dNodePrime; /* number of the internal drain node of the mosfet */
    int B1sNodePrime; /* number of the internal source node of the mosfet */

    double B1l;   /* the length of the channel region */
    double B1w;   /* the width of the channel region  */ 
    double B1m;   /* the parallel multiplier          */
    double B1drainArea;   /* the area of the drain diffusion */
    double B1sourceArea;  /* the area of the source diffusion */
    double B1drainSquares;    /* the length of the drain in squares */
    double B1sourceSquares;   /* the length of the source in squares */
    double B1drainPerimeter;
    double B1sourcePerimeter;
    double B1sourceConductance;   /*conductance of source(or 0):set in setup*/
    double B1drainConductance;    /*conductance of drain(or 0):set in setup*/

    double B1icVBS;   /* initial condition B-S voltage */
    double B1icVDS;   /* initial condition D-S voltage */
    double B1icVGS;   /* initial condition G-S voltage */
    double B1von;
    double B1vdsat;
    int B1off;        /* non-zero to indicate device is off for dc analysis*/
    int B1mode;       /* device mode : 1 = normal, -1 = inverse */

    double B1vfb;      /* flat band voltage at given L and W */
    double B1phi;      /* surface potential at strong inversion */
    double B1K1;       /* bulk effect coefficient 1             */
    double B1K2;       /* bulk effect coefficient 2             */
    double B1eta;      /* drain induced barrier lowering        */
    double B1etaB;     /* Vbs dependence of Eta                 */
    double B1etaD;     /* Vds dependence of Eta                 */
    double B1betaZero; /* Beta at vds = 0 and vgs = Vth         */
    double B1betaZeroB; /* Vbs dependence of BetaZero           */
    double B1betaVdd;  /* Beta at vds=Vdd and vgs=Vth           */
    double B1betaVddB; /* Vbs dependence of BVdd             */
    double B1betaVddD; /* Vds dependence of BVdd             */
    double B1ugs;      /* Mobility degradation due to gate field*/
    double B1ugsB;     /* Vbs dependence of Ugs                 */
    double B1uds;      /* Drift Velocity Saturation due to Vds  */
    double B1udsB;     /* Vbs dependence of Uds                 */
    double B1udsD;     /* Vds dependence of Uds                 */
    double B1subthSlope; /* slope of subthreshold current with Vgs*/
    double B1subthSlopeB; /* Vbs dependence of Subthreshold Slope */
    double B1subthSlopeD; /* Vds dependence of Subthreshold Slope */
    double B1GDoverlapCap;/* Gate Drain Overlap Capacitance       */
    double B1GSoverlapCap;/* Gate Source Overlap Capacitance      */
    double B1GBoverlapCap;/* Gate Bulk Overlap Capacitance        */
    double B1vt0;
    double B1vdd;         /* Supply Voltage                       */
    double B1temp;
    double B1oxideThickness;
    unsigned B1channelChargePartitionFlag :1;
    unsigned B1lGiven :1;
    unsigned B1wGiven :1;
    unsigned B1mGiven :1;
    unsigned B1drainAreaGiven :1;
    unsigned B1sourceAreaGiven    :1;
    unsigned B1drainSquaresGiven  :1;
    unsigned B1sourceSquaresGiven :1;
    unsigned B1drainPerimeterGiven    :1;
    unsigned B1sourcePerimeterGiven   :1;
    unsigned B1dNodePrimeSet  :1;
    unsigned B1sNodePrimeSet  :1;
    unsigned B1icVBSGiven :1;
    unsigned B1icVDSGiven :1;
    unsigned B1icVGSGiven :1;
    unsigned B1vonGiven   :1;
    unsigned B1vdsatGiven :1;


    double *B1DdPtr;      /* pointer to sparse matrix element at
                                     * (Drain node,drain node) */
    double *B1GgPtr;      /* pointer to sparse matrix element at
                                     * (gate node,gate node) */
    double *B1SsPtr;      /* pointer to sparse matrix element at
                                     * (source node,source node) */
    double *B1BbPtr;      /* pointer to sparse matrix element at
                                     * (bulk node,bulk node) */
    double *B1DPdpPtr;    /* pointer to sparse matrix element at
                                     * (drain prime node,drain prime node) */
    double *B1SPspPtr;    /* pointer to sparse matrix element at
                                     * (source prime node,source prime node) */
    double *B1DdpPtr;     /* pointer to sparse matrix element at
                                     * (drain node,drain prime node) */
    double *B1GbPtr;      /* pointer to sparse matrix element at
                                     * (gate node,bulk node) */
    double *B1GdpPtr;     /* pointer to sparse matrix element at
                                     * (gate node,drain prime node) */
    double *B1GspPtr;     /* pointer to sparse matrix element at
                                     * (gate node,source prime node) */
    double *B1SspPtr;     /* pointer to sparse matrix element at
                                     * (source node,source prime node) */
    double *B1BdpPtr;     /* pointer to sparse matrix element at
                                     * (bulk node,drain prime node) */
    double *B1BspPtr;     /* pointer to sparse matrix element at
                                     * (bulk node,source prime node) */
    double *B1DPspPtr;    /* pointer to sparse matrix element at
                                     * (drain prime node,source prime node) */
    double *B1DPdPtr;     /* pointer to sparse matrix element at
                                     * (drain prime node,drain node) */
    double *B1BgPtr;      /* pointer to sparse matrix element at
                                     * (bulk node,gate node) */
    double *B1DPgPtr;     /* pointer to sparse matrix element at
                                     * (drain prime node,gate node) */

    double *B1SPgPtr;     /* pointer to sparse matrix element at
                                     * (source prime node,gate node) */
    double *B1SPsPtr;     /* pointer to sparse matrix element at
                                     * (source prime node,source node) */
    double *B1DPbPtr;     /* pointer to sparse matrix element at
                                     * (drain prime node,bulk node) */
    double *B1SPbPtr;     /* pointer to sparse matrix element at
                                     * (source prime node,bulk node) */
    double *B1SPdpPtr;    /* pointer to sparse matrix element at
                                     * (source prime node,drain prime node) */

#ifndef NODISTO
	double B1dCoeffs[B1NDCOEFFS];
#else /* NODISTO */
	double *B1dCoeffs;
#endif /* NODISTO */
/* indices to the array of BSIM1 noise sources */

#define B1RDNOIZ       0
#define B1RSNOIZ       1
#define B1IDNOIZ       2
#define B1FLNOIZ 3
#define B1TOTNOIZ    4

#define B1NSRCS     5     /* the number of BSIM1 noise sources */

#ifndef NONOISE
    double B1nVar[NSTATVARS][B1NSRCS];
#else /* NONOISE */
	double **B1nVar;
#endif /* NONOISE */


} B1instance ;

#ifndef CONFIG

#define B1vbd B1states+ 0
#define B1vbs B1states+ 1
#define B1vgs B1states+ 2
#define B1vds B1states+ 3
#define B1cd B1states+ 4
#define B1id B1states+ 4
#define B1cbs B1states+ 5
#define B1ibs B1states+ 5
#define B1cbd B1states+ 6
#define B1ibd B1states+ 6
#define B1gm B1states+ 7
#define B1gds B1states+ 8
#define B1gmbs B1states+ 9
#define B1gbd B1states+ 10
#define B1gbs B1states+ 11
#define B1qb B1states+ 12
#define B1cqb B1states+ 13
#define B1iqb B1states+ 13
#define B1qg B1states+ 14
#define B1cqg B1states+ 15
#define B1iqg B1states+ 15
#define B1qd B1states+ 16
#define B1cqd B1states+ 17
#define B1iqd B1states+ 17
#define B1cggb B1states+ 18
#define B1cgdb B1states+ 19
#define B1cgsb B1states+ 20
#define B1cbgb B1states+ 21
#define B1cbdb B1states+ 22
#define B1cbsb B1states+ 23
#define B1capbd B1states+ 24
#define B1iqbd B1states+ 25
#define B1cqbd B1states+ 25
#define B1capbs B1states+ 26
#define B1iqbs B1states+ 27
#define B1cqbs B1states+ 27
#define B1cdgb B1states+ 28
#define B1cddb B1states+ 29
#define B1cdsb B1states+ 30
#define B1vono B1states+ 31
#define B1vdsato B1states+ 32
#define B1qbs  B1states+ 33
#define B1qbd  B1states+ 34

/*
 * the following naming convention is used:
 * x = vgs
 * y = vbs
 * z = vds
 * DrC is the DrCur
 * therefore qg_xyz stands for the coefficient of the vgs*vbs*vds
 * term in the multidimensional Taylor expansion for qg; and DrC_x2y
 * for the coeff. of the vgs*vgs*vbs term in the DrC expansion.
 */

#define	qg_x		B1dCoeffs[0]
#define	qg_y		B1dCoeffs[1]
#define	qg_z		B1dCoeffs[2]
#define	qg_x2		B1dCoeffs[3]
#define	qg_y2		B1dCoeffs[4]
#define	qg_z2		B1dCoeffs[5]
#define	qg_xy		B1dCoeffs[6]
#define	qg_yz		B1dCoeffs[7]
#define	qg_xz		B1dCoeffs[8]
#define	qg_x3		B1dCoeffs[9]
#define	qg_y3		B1dCoeffs[10]
#define	qg_z3		B1dCoeffs[11]
#define	qg_x2z		B1dCoeffs[12]
#define	qg_x2y		B1dCoeffs[13]
#define	qg_y2z		B1dCoeffs[14]
#define	qg_xy2		B1dCoeffs[15]
#define	qg_xz2		B1dCoeffs[16]
#define	qg_yz2		B1dCoeffs[17]
#define	qg_xyz		B1dCoeffs[18]
#define	qb_x		B1dCoeffs[19]
#define	qb_y		B1dCoeffs[20]
#define	qb_z		B1dCoeffs[21]
#define	qb_x2		B1dCoeffs[22]
#define	qb_y2		B1dCoeffs[23]
#define	qb_z2		B1dCoeffs[24]
#define	qb_xy		B1dCoeffs[25]
#define	qb_yz		B1dCoeffs[26]
#define	qb_xz		B1dCoeffs[27]
#define	qb_x3		B1dCoeffs[28]
#define	qb_y3		B1dCoeffs[29]
#define	qb_z3		B1dCoeffs[30]
#define	qb_x2z		B1dCoeffs[31]
#define	qb_x2y		B1dCoeffs[32]
#define	qb_y2z		B1dCoeffs[33]
#define	qb_xy2		B1dCoeffs[34]
#define	qb_xz2		B1dCoeffs[35]
#define	qb_yz2		B1dCoeffs[36]
#define	qb_xyz		B1dCoeffs[37]
#define	qd_x		B1dCoeffs[38]
#define	qd_y		B1dCoeffs[39]
#define	qd_z		B1dCoeffs[40]
#define	qd_x2		B1dCoeffs[41]
#define	qd_y2		B1dCoeffs[42]
#define	qd_z2		B1dCoeffs[43]
#define	qd_xy		B1dCoeffs[44]
#define	qd_yz		B1dCoeffs[45]
#define	qd_xz		B1dCoeffs[46]
#define	qd_x3		B1dCoeffs[47]
#define	qd_y3		B1dCoeffs[48]
#define	qd_z3		B1dCoeffs[49]
#define	qd_x2z		B1dCoeffs[50]
#define	qd_x2y		B1dCoeffs[51]
#define	qd_y2z		B1dCoeffs[52]
#define	qd_xy2		B1dCoeffs[53]
#define	qd_xz2		B1dCoeffs[54]
#define	qd_yz2		B1dCoeffs[55]
#define	qd_xyz		B1dCoeffs[56]
#define	DrC_x		B1dCoeffs[57]
#define	DrC_y		B1dCoeffs[58]
#define	DrC_z		B1dCoeffs[59]
#define	DrC_x2		B1dCoeffs[60]
#define	DrC_y2		B1dCoeffs[61]
#define	DrC_z2		B1dCoeffs[62]
#define	DrC_xy		B1dCoeffs[63]
#define	DrC_yz		B1dCoeffs[64]
#define	DrC_xz		B1dCoeffs[65]
#define	DrC_x3		B1dCoeffs[66]
#define	DrC_y3		B1dCoeffs[67]
#define	DrC_z3		B1dCoeffs[68]
#define	DrC_x2z		B1dCoeffs[69]
#define	DrC_x2y		B1dCoeffs[70]
#define	DrC_y2z		B1dCoeffs[71]
#define	DrC_xy2		B1dCoeffs[72]
#define	DrC_xz2		B1dCoeffs[73]
#define	DrC_yz2		B1dCoeffs[74]
#define	DrC_xyz		B1dCoeffs[75]
#define	gbs1		B1dCoeffs[76]
#define	gbs2		B1dCoeffs[77]
#define	gbs3		B1dCoeffs[78]
#define	gbd1		B1dCoeffs[79]
#define	gbd2		B1dCoeffs[80]
#define	gbd3		B1dCoeffs[81]

#endif


/* per model data */

typedef struct sBSIM1model {       /* model structure for a resistor */

    struct GENmodel gen;

#define B1modType gen.GENmodType
#define B1nextModel(inst) ((struct sBSIM1model *)((inst)->gen.GENnextModel))
#define B1instances(inst) ((B1instance *)((inst)->gen.GENinstances))
#define B1modName gen.GENmodName

    int B1type;       /* device type : 1 = nmos,  -1 = pmos */

    double B1vfb0;
    double B1vfbL;
    double B1vfbW;
    double B1phi0;
    double B1phiL;
    double B1phiW;
    double B1K10;
    double B1K1L;
    double B1K1W;
    double B1K20;
    double B1K2L;
    double B1K2W;
    double B1eta0;
    double B1etaL;
    double B1etaW;
    double B1etaB0;
    double B1etaBl;
    double B1etaBw;
    double B1etaD0;
    double B1etaDl;
    double B1etaDw;
    double B1deltaL;
    double B1deltaW;
    double B1mobZero;
    double B1mobZeroB0;
    double B1mobZeroBl;
    double B1mobZeroBw ;
    double B1mobVdd0;
    double B1mobVddl;
    double B1mobVddw;
    double B1mobVddB0;
    double B1mobVddBl;
    double B1mobVddBw;
    double B1mobVddD0;
    double B1mobVddDl;
    double B1mobVddDw;
    double B1ugs0;
    double B1ugsL;
    double B1ugsW;
    double B1ugsB0;
    double B1ugsBL;
    double B1ugsBW;
    double B1uds0;
    double B1udsL;
    double B1udsW;
    double B1udsB0;
    double B1udsBL;
    double B1udsBW;
    double B1udsD0;
    double B1udsDL;
    double B1udsDW;
    double B1subthSlope0;
    double B1subthSlopeL;
    double B1subthSlopeW;
    double B1subthSlopeB0;
    double B1subthSlopeBL;
    double B1subthSlopeBW;
    double B1subthSlopeD0;
    double B1subthSlopeDL;
    double B1subthSlopeDW;
    double B1oxideThickness;              /* unit: micron  */
    double B1Cox;                         /* unit: F/cm**2 */
    double B1temp;
    double B1vdd;
    double B1gateSourceOverlapCap;
    double B1gateDrainOverlapCap;
    double B1gateBulkOverlapCap;
    unsigned B1channelChargePartitionFlag :1;

    double B1sheetResistance;
    double B1jctSatCurDensity;
    double B1bulkJctPotential;
    double B1bulkJctBotGradingCoeff;
    double B1bulkJctSideGradingCoeff;
    double B1sidewallJctPotential;
    double B1unitAreaJctCap;
    double B1unitLengthSidewallJctCap;
    double B1defaultWidth;
    double B1deltaLength;

    double B1fNcoef;
    double B1fNexp;



    unsigned  B1vfb0Given   :1;
    unsigned  B1vfbLGiven   :1;
    unsigned  B1vfbWGiven   :1;
    unsigned  B1phi0Given   :1;
    unsigned  B1phiLGiven   :1;
    unsigned  B1phiWGiven   :1;
    unsigned  B1K10Given   :1;
    unsigned  B1K1LGiven   :1;
    unsigned  B1K1WGiven   :1;
    unsigned  B1K20Given   :1;
    unsigned  B1K2LGiven   :1;
    unsigned  B1K2WGiven   :1;
    unsigned  B1eta0Given   :1;
    unsigned  B1etaLGiven   :1;
    unsigned  B1etaWGiven   :1;
    unsigned  B1etaB0Given   :1;
    unsigned  B1etaBlGiven   :1;
    unsigned  B1etaBwGiven   :1;
    unsigned  B1etaD0Given   :1;
    unsigned  B1etaDlGiven   :1;
    unsigned  B1etaDwGiven   :1;
    unsigned  B1deltaLGiven   :1;
    unsigned  B1deltaWGiven   :1;
    unsigned  B1mobZeroGiven   :1;
    unsigned  B1mobZeroB0Given   :1;
    unsigned  B1mobZeroBlGiven   :1;
    unsigned  B1mobZeroBwGiven   :1;
    unsigned  B1mobVdd0Given   :1;
    unsigned  B1mobVddlGiven   :1;
    unsigned  B1mobVddwGiven   :1;
    unsigned  B1mobVddB0Given   :1;
    unsigned  B1mobVddBlGiven   :1;
    unsigned  B1mobVddBwGiven   :1;
    unsigned  B1mobVddD0Given   :1;
    unsigned  B1mobVddDlGiven   :1;
    unsigned  B1mobVddDwGiven   :1;
    unsigned  B1ugs0Given   :1;
    unsigned  B1ugsLGiven   :1;
    unsigned  B1ugsWGiven   :1;
    unsigned  B1ugsB0Given   :1;
    unsigned  B1ugsBLGiven   :1;
    unsigned  B1ugsBWGiven   :1;
    unsigned  B1uds0Given   :1;
    unsigned  B1udsLGiven   :1;
    unsigned  B1udsWGiven   :1;
    unsigned  B1udsB0Given   :1;
    unsigned  B1udsBLGiven   :1;
    unsigned  B1udsBWGiven   :1;
    unsigned  B1udsD0Given   :1;
    unsigned  B1udsDLGiven   :1;
    unsigned  B1udsDWGiven   :1;
    unsigned  B1subthSlope0Given   :1;
    unsigned  B1subthSlopeLGiven   :1;
    unsigned  B1subthSlopeWGiven   :1;
    unsigned  B1subthSlopeB0Given   :1;
    unsigned  B1subthSlopeBLGiven   :1;
    unsigned  B1subthSlopeBWGiven   :1;
    unsigned  B1subthSlopeD0Given   :1;
    unsigned  B1subthSlopeDLGiven   :1;
    unsigned  B1subthSlopeDWGiven   :1;
    unsigned  B1oxideThicknessGiven   :1;
    unsigned  B1tempGiven   :1;
    unsigned  B1vddGiven   :1;
    unsigned  B1gateSourceOverlapCapGiven   :1;
    unsigned  B1gateDrainOverlapCapGiven   :1;
    unsigned  B1gateBulkOverlapCapGiven   :1;
    unsigned  B1channelChargePartitionFlagGiven   :1;
    unsigned  B1sheetResistanceGiven   :1;
    unsigned  B1jctSatCurDensityGiven   :1;
    unsigned  B1bulkJctPotentialGiven   :1;
    unsigned  B1bulkJctBotGradingCoeffGiven   :1;
    unsigned  B1sidewallJctPotentialGiven   :1;
    unsigned  B1bulkJctSideGradingCoeffGiven   :1;
    unsigned  B1unitAreaJctCapGiven   :1;
    unsigned  B1unitLengthSidewallJctCapGiven   :1;
    unsigned  B1defaultWidthGiven   :1;
    unsigned  B1deltaLengthGiven   :1;
    
    unsigned  B1fNcoefGiven :1;
    unsigned  B1fNexpGiven :1;

    unsigned  B1typeGiven   :1;

} B1model;


#ifndef NMOS
#define NMOS 1
#define PMOS -1
#endif /*NMOS*/


/* device parameters */
#define BSIM1_W 1
#define BSIM1_L 2
#define BSIM1_AS 3
#define BSIM1_AD 4
#define BSIM1_PS 5
#define BSIM1_PD 6
#define BSIM1_NRS 7
#define BSIM1_NRD 8
#define BSIM1_OFF 9
#define BSIM1_IC_VBS 10
#define BSIM1_IC_VDS 11
#define BSIM1_IC_VGS 12
#define BSIM1_IC 13
#define BSIM1_M 14

/* model parameters */
#define BSIM1_MOD_VFB0 101
#define BSIM1_MOD_VFBL 102
#define BSIM1_MOD_VFBW 103
#define BSIM1_MOD_PHI0 104
#define BSIM1_MOD_PHIL 105
#define BSIM1_MOD_PHIW 106
#define BSIM1_MOD_K10 107
#define BSIM1_MOD_K1L 108
#define BSIM1_MOD_K1W 109
#define BSIM1_MOD_K20 110
#define BSIM1_MOD_K2L 111
#define BSIM1_MOD_K2W 112
#define BSIM1_MOD_ETA0 113
#define BSIM1_MOD_ETAL 114
#define BSIM1_MOD_ETAW 115
#define BSIM1_MOD_ETAB0 116
#define BSIM1_MOD_ETABL 117
#define BSIM1_MOD_ETABW 118
#define BSIM1_MOD_ETAD0 119
#define BSIM1_MOD_ETADL 120
#define BSIM1_MOD_ETADW 121
#define BSIM1_MOD_DELTAL 122
#define BSIM1_MOD_DELTAW 123
#define BSIM1_MOD_MOBZERO 124
#define BSIM1_MOD_MOBZEROB0 125
#define BSIM1_MOD_MOBZEROBL 126
#define BSIM1_MOD_MOBZEROBW 127
#define BSIM1_MOD_MOBVDD0 128
#define BSIM1_MOD_MOBVDDL 129
#define BSIM1_MOD_MOBVDDW 130
#define BSIM1_MOD_MOBVDDB0 131
#define BSIM1_MOD_MOBVDDBL 132
#define BSIM1_MOD_MOBVDDBW 133
#define BSIM1_MOD_MOBVDDD0 134
#define BSIM1_MOD_MOBVDDDL 135
#define BSIM1_MOD_MOBVDDDW 136
#define BSIM1_MOD_UGS0 137
#define BSIM1_MOD_UGSL 138
#define BSIM1_MOD_UGSW 139
#define BSIM1_MOD_UGSB0 140
#define BSIM1_MOD_UGSBL 141
#define BSIM1_MOD_UGSBW 142
#define BSIM1_MOD_UDS0 143
#define BSIM1_MOD_UDSL 144
#define BSIM1_MOD_UDSW 145
#define BSIM1_MOD_UDSB0 146
#define BSIM1_MOD_UDSBL 147
#define BSIM1_MOD_UDSBW 148
#define BSIM1_MOD_UDSD0 149
#define BSIM1_MOD_UDSDL 150
#define BSIM1_MOD_UDSDW 151
#define BSIM1_MOD_N00 152
#define BSIM1_MOD_N0L 153
#define BSIM1_MOD_N0W 154
#define BSIM1_MOD_NB0 155
#define BSIM1_MOD_NBL 156
#define BSIM1_MOD_NBW 157
#define BSIM1_MOD_ND0 158
#define BSIM1_MOD_NDL 159
#define BSIM1_MOD_NDW 160
#define BSIM1_MOD_TOX 161
#define BSIM1_MOD_TEMP 162
#define BSIM1_MOD_VDD 163
#define BSIM1_MOD_CGSO 164
#define BSIM1_MOD_CGDO 165
#define BSIM1_MOD_CGBO 166
#define BSIM1_MOD_XPART 167
#define BSIM1_MOD_RSH 168
#define BSIM1_MOD_JS 169
#define BSIM1_MOD_PB 170
#define BSIM1_MOD_MJ 171
#define BSIM1_MOD_PBSW 172
#define BSIM1_MOD_MJSW 173
#define BSIM1_MOD_CJ 174
#define BSIM1_MOD_CJSW 175
#define BSIM1_MOD_DEFWIDTH 176
#define BSIM1_MOD_DELLENGTH 177
#define BSIM1_MOD_NMOS 178
#define BSIM1_MOD_PMOS 179

#define BSIM1_MOD_KF 180
#define BSIM1_MOD_AF 181

/* device questions */
#define BSIM1_DNODE              201
#define BSIM1_GNODE              202
#define BSIM1_SNODE              203
#define BSIM1_BNODE              204
#define BSIM1_DNODEPRIME         205
#define BSIM1_SNODEPRIME         206
#define BSIM1_VBD                207
#define BSIM1_VBS                208
#define BSIM1_VGS                209
#define BSIM1_VDS                210
#define BSIM1_CD         211
#define BSIM1_CBS        212
#define BSIM1_CBD        213
#define BSIM1_GM         214
#define BSIM1_GDS        215
#define BSIM1_GMBS       216
#define BSIM1_GBD        217
#define BSIM1_GBS        218
#define BSIM1_QB         219
#define BSIM1_CQB        220
#define BSIM1_QG         221
#define BSIM1_CQG        222
#define BSIM1_QD         223
#define BSIM1_CQD        224
#define BSIM1_CGG        225
#define BSIM1_CGD        226
#define BSIM1_CGS        227
#define BSIM1_CBG        228
#define BSIM1_CAPBD      231
#define BSIM1_CQBD       232
#define BSIM1_CAPBS      233
#define BSIM1_CQBS       234
#define BSIM1_CDG        235
#define BSIM1_CDD        236
#define BSIM1_CDS        237
#define BSIM1_VON        238
#define BSIM1_QBS        239
#define BSIM1_QBD        240
#define BSIM1_SOURCECONDUCT      241
#define BSIM1_DRAINCONDUCT       242

/* model questions */

#include "bsim1ext.h"

extern void B1evaluate(double,double,double,B1instance*,B1model*,
	double*,double*,double*, double*, double*, double*, double*,
	double*, double*, double*, double*, double*, double*, double*,
	double*, double*, double*, double*, CKTcircuit*);

#endif /*B1*/
