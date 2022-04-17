/*
 * Author: 2000 Wladek Grabinski; EKV v2.6 Model Upgrade
 * Author: 1997 Eckhard Brass;    EKV v2.5 Model Implementation
 *     (C) 1990 Regents of the University of California. Spice3 Format
 */

#ifndef EKV 
#define EKV 

#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/gendefs.h"
#include "ngspice/complex.h"
#include "ngspice/noisedef.h"

/* declarations for EKV V2.6 MOSFETs */
/*
 * wg 17-SEP-2K  common definitions for EKV v2.6 rev.XII   
 */
#define  epso           8.854214871e-12
#define  epssil         1.035943140e-10
#define  boltz          1.3806226e-23
#define  charge         1.6021918e-19

#define  konq           boltz/charge

#define  twopi          6.2831853072e0
#define  reftmp         300.15e0
#define  tnomref        273.15e0
#define  exp40          2.35385266837019946e17
#define  expm5          6.73794699909e-3
#define  two3rds        2.0/3.0
#define  four15th       4.0/15.0
#define  twentyfive16th 25.0/16.0

#define  c_a            0.28
#define  c_ee           1.936e-3


/* information needed for each instance */

typedef struct sEKVinstance {
	struct sEKVmodel *sEKVmodPtr; /* backpointer to model */
	struct sEKVinstance *EKVnextInstance;  /* pointer to next instance of
	                                              *current model*/
	IFuid EKVname; /* pointer to character string naming this instance */
	int EKVstates;     /* index into state table for this device */

	int EKVdNode;  /* number of the gate node of the mosfet */
	int EKVgNode;  /* number of the gate node of the mosfet */
	int EKVsNode;  /* number of the source node of the mosfet */
	int EKVbNode;  /* number of the bulk node of the mosfet */
	int EKVdNodePrime; /* number of the internal drain node of the mosfet */
	int EKVsNodePrime; /* number of the internal source node of the mosfet */

	double EKVl;   /* the length of the channel region */
	double EKVw;   /* the width of the channel region */
	double EKVdrainArea;   /* the area of the drain diffusion */
	double EKVsourceArea;  /* the area of the source diffusion */
	double EKVdrainSquares;    /* the length of the drain in squares */
	double EKVsourceSquares;   /* the length of the source in squares */
	double EKVdrainPerimiter;
	double EKVsourcePerimiter;
	double EKVsourceConductance;   /*conductance of source(or 0):set in setup*/
	double EKVdrainConductance;    /*conductance of drain(or 0):set in setup*/
	double EKVtemp;                /* operating temperature of this instance */

	double EKVtkp;                 /* temperature corrected transconductance*/
	double EKVtPhi;                /* temperature corrected Phi */
	double EKVtVto;                /* temperature corrected Vto */
	double EKVtSatCur;             /* temperature corrected saturation Cur. */
	double EKVtSatCurDens; /* temperature corrected saturation Cur. density*/
	double EKVtCbd;                /* temperature corrected B-D Capacitance */
	double EKVtCbs;                /* temperature corrected B-S Capacitance */
	double EKVtCj;         /* temperature corrected Bulk bottom Capacitance */
	double EKVtCjsw;       /* temperature corrected Bulk side Capacitance */
	double EKVtBulkPot;    /* temperature corrected Bulk potential */
	double EKVtpbsw;       /* temperature corrected pbsw */
	double EKVtDepCap;     /* temperature adjusted transition point in */
	/* the cureve matching Fc * Vj */
	double EKVtucrit;      /* temperature adjusted ucrit */
	double EKVtibb;        /* temperature adjusted ibb */
	double EKVtrs;         /* temperature adjusted rs */
	double EKVtrd;         /* temperature adjusted rd */
	double EKVtrsh;        /* temperature adjusted rsh */
	double EKVtrsc;        /* temperature adjusted rsc */
	double EKVtrdc;        /* temperature adjusted rdc */
	double EKVtjsw;        /* temperature adjusted jsw */
	double EKVtaf;         /* temperature adjusted af  */

	double EKVicVBS;   /* initial condition B-S voltage */
	double EKVicVDS;   /* initial condition D-S voltage */
	double EKVicVGS;   /* initial condition G-S voltage */
	double EKVvon;
	double EKVvdsat;
	double EKVvgeff;
	double EKVvgprime;
	double EKVvgstar;
	double EKVsourceVcrit; /* Vcrit for pos. vds */
	double EKVdrainVcrit;  /* Vcrit for pos. vds */
	double EKVcd;
	double EKVisub;
	double EKVvp;
	double EKVslope;
	double EKVif;
	double EKVir;
	double EKVirprime;
	double EKVtau;
	double EKVcbs;
	double EKVcbd;
	double EKVgmbs;
	double EKVgm;
	double EKVgms;
	double EKVgds;
	double EKVgbd;
	double EKVgbs;
	double EKVcapbd;
	double EKVcapbs;
	double EKVCbd;
	double EKVCbdsw;
	double EKVCbs;
	double EKVCbssw;
	double EKVf2d;
	double EKVf3d;
	double EKVf4d;
	double EKVf2s;
	double EKVf3s;
	double EKVf4s;

	/*
 * naming convention:
 * x = vgs
 * y = vbs
 * z = vds
 * cdr = cdrain
 */

#define	EKVNDCOEFFS	30

#ifndef NODISTO
	double EKVdCoeffs[EKVNDCOEFFS];
#else /* NODISTO */
	double *EKVdCoeffs;
#endif /* NODISTO */

#ifndef CONFIG

#define	capbs2		EKVdCoeffs[0]
#define	capbs3		EKVdCoeffs[1]
#define	capbd2		EKVdCoeffs[2]
#define	capbd3		EKVdCoeffs[3]
#define	gbs2		EKVdCoeffs[4]
#define	gbs3		EKVdCoeffs[5]
#define	gbd2		EKVdCoeffs[6]
#define	gbd3		EKVdCoeffs[7]
#define	capgb2		EKVdCoeffs[8]
#define	capgb3		EKVdCoeffs[9]
#define	cdr_x2		EKVdCoeffs[10]
#define	cdr_y2		EKVdCoeffs[11]
#define	cdr_z2		EKVdCoeffs[12]
#define	cdr_xy		EKVdCoeffs[13]
#define	cdr_yz		EKVdCoeffs[14]
#define	cdr_xz		EKVdCoeffs[15]
#define	cdr_x3		EKVdCoeffs[16]
#define	cdr_y3		EKVdCoeffs[17]
#define	cdr_z3		EKVdCoeffs[18]
#define	cdr_x2z		EKVdCoeffs[19]
#define	cdr_x2y		EKVdCoeffs[20]
#define	cdr_y2z		EKVdCoeffs[21]
#define	cdr_xy2		EKVdCoeffs[22]
#define	cdr_xz2		EKVdCoeffs[23]
#define	cdr_yz2		EKVdCoeffs[24]
#define	cdr_xyz		EKVdCoeffs[25]
#define	capgs2		EKVdCoeffs[26]
#define	capgs3		EKVdCoeffs[27]
#define	capgd2		EKVdCoeffs[28]
#define	capgd3		EKVdCoeffs[29]

#endif

#define EKVRDNOIZ   0
#define EKVRSNOIZ   1
#define EKVIDNOIZ   2
#define EKVFLNOIZ   3
#define EKVTOTNOIZ  4

#define EKVNSRCS    5     /* the number of EKVFET noise sources*/

#ifndef NONOISE
	double EKVnVar[NSTATVARS][EKVNSRCS];
#else /* NONOISE */
	double **EKVnVar;
#endif /* NONOISE */

	int EKVmode;       /* device mode : 1 = normal, -1 = inverse */


	unsigned EKVoff:1;  /* non-zero to indicate device is off for dc analysis*/
	unsigned EKVtempGiven :1;  /* instance temperature specified */
	unsigned EKVlGiven :1;
	unsigned EKVwGiven :1;
	unsigned EKVdrainAreaGiven :1;
	unsigned EKVsourceAreaGiven    :1;
	unsigned EKVdrainSquaresGiven  :1;
	unsigned EKVsourceSquaresGiven :1;
	unsigned EKVdrainPerimiterGiven    :1;
	unsigned EKVsourcePerimiterGiven   :1;
	unsigned EKVdNodePrimeSet  :1;
	unsigned EKVsNodePrimeSet  :1;
	unsigned EKVicVBSGiven :1;
	unsigned EKVicVDSGiven :1;
	unsigned EKVicVGSGiven :1;
	unsigned EKVvonGiven   :1;
	unsigned EKVvdsatGiven :1;
	unsigned EKVmodeGiven  :1;


	double *EKVDdPtr;      /* pointer to sparse matrix element at
	                                     * (Drain node,drain node) */
	double *EKVGgPtr;      /* pointer to sparse matrix element at
	                                     * (gate node,gate node) */
	double *EKVSsPtr;      /* pointer to sparse matrix element at
	                                     * (source node,source node) */
	double *EKVBbPtr;      /* pointer to sparse matrix element at
	                                     * (bulk node,bulk node) */
	double *EKVDPdpPtr;    /* pointer to sparse matrix element at
	                                     * (drain prime node,drain prime node) */
	double *EKVSPspPtr;    /* pointer to sparse matrix element at
	                                     * (source prime node,source prime node) */
	double *EKVDdpPtr;     /* pointer to sparse matrix element at
	                                     * (drain node,drain prime node) */
	double *EKVGbPtr;      /* pointer to sparse matrix element at
	                                     * (gate node,bulk node) */
	double *EKVGdpPtr;     /* pointer to sparse matrix element at
	                                     * (gate node,drain prime node) */
	double *EKVGspPtr;     /* pointer to sparse matrix element at
	                                     * (gate node,source prime node) */
	double *EKVSspPtr;     /* pointer to sparse matrix element at
	                                     * (source node,source prime node) */
	double *EKVBdpPtr;     /* pointer to sparse matrix element at
	                                     * (bulk node,drain prime node) */
	double *EKVBspPtr;     /* pointer to sparse matrix element at
	                                     * (bulk node,source prime node) */
	double *EKVDPspPtr;    /* pointer to sparse matrix element at
	                                     * (drain prime node,source prime node) */
	double *EKVDPdPtr;     /* pointer to sparse matrix element at
	                                     * (drain prime node,drain node) */
	double *EKVBgPtr;      /* pointer to sparse matrix element at
	                                     * (bulk node,gate node) */
	double *EKVDPgPtr;     /* pointer to sparse matrix element at
	                                     * (drain prime node,gate node) */

	double *EKVSPgPtr;     /* pointer to sparse matrix element at
	                                     * (source prime node,gate node) */
	double *EKVSPsPtr;     /* pointer to sparse matrix element at
	                                     * (source prime node,source node) */
	double *EKVDPbPtr;     /* pointer to sparse matrix element at
	                                     * (drain prime node,bulk node) */
	double *EKVSPbPtr;     /* pointer to sparse matrix element at
	                                     * (source prime node,bulk node) */
	double *EKVSPdpPtr;    /* pointer to sparse matrix element at
	                                     * (source prime node,drain prime node) */

	int  EKVsenParmNo;   /* parameter # for sensitivity use;
            set equal to 0  if  neither length
            nor width of the mosfet is a design
	            parameter */
	unsigned EKVsens_l :1;   /* field which indicates whether  
            length of the mosfet is a design
	            parameter or not */
	unsigned EKVsens_w :1;   /* field which indicates whether  
            width of the mosfet is a design
	            parameter or not */
	unsigned EKVsenPertFlag :1; /* indictes whether the the 
            parameter of the particular instance is 
	            to be perturbed */
	double EKVcgs;
	double EKVcgd;
	double EKVcgb;
	double *EKVsens;

#define EKVsenCgs EKVsens /* contains pertured values of cgs */
#define EKVsenCgd EKVsens + 6 /* contains perturbed values of cgd*/
#define EKVsenCgb EKVsens + 12 /* contains perturbed values of cgb*/
#define EKVsenCbd EKVsens + 18 /* contains perturbed values of cbd*/
#define EKVsenCbs EKVsens + 24 /* contains perturbed values of cbs*/
#define EKVsenGds EKVsens + 30 /* contains perturbed values of gds*/
#define EKVsenGbs EKVsens + 36 /* contains perturbed values of gbs*/
#define EKVsenGbd EKVsens + 42 /* contains perturbed values of gbd*/
#define EKVsenGm EKVsens + 48 /* contains perturbed values of gm*/
#define EKVsenGmbs EKVsens + 54 /* contains perturbed values of gmbs*/
#define EKVdphigs_dl EKVsens + 60
#define EKVdphigd_dl EKVsens + 61
#define EKVdphigb_dl EKVsens + 62
#define EKVdphibs_dl EKVsens + 63
#define EKVdphibd_dl EKVsens + 64
#define EKVdphigs_dw EKVsens + 65
#define EKVdphigd_dw EKVsens + 66
#define EKVdphigb_dw EKVsens + 67
#define EKVdphibs_dw EKVsens + 68
#define EKVdphibd_dw EKVsens + 69

} EKVinstance ;

#define EKVvbd EKVstates+ 0   /* bulk-drain voltage */
#define EKVvbs EKVstates+ 1   /* bulk-source voltage */
#define EKVvgs EKVstates+ 2   /* gate-source voltage */
#define EKVvds EKVstates+ 3   /* drain-source voltage */

#define EKVcapgs EKVstates+ 4  /* gate-source capacitor value */
#define EKVqgs   EKVstates+ 5  /* gate-source capacitor charge */
#define EKVcqgs  EKVstates+ 6  /* gate-source capacitor current */

#define EKVcapgd EKVstates+ 7 /* gate-drain capacitor value */
#define EKVqgd EKVstates+ 8   /* gate-drain capacitor charge */
#define EKVcqgd EKVstates+ 9  /* gate-drain capacitor current */

#define EKVcapgb EKVstates+10 /* gate-bulk capacitor value */
#define EKVqgb EKVstates+ 11  /* gate-bulk capacitor charge */
#define EKVcqgb EKVstates+ 12 /* gate-bulk capacitor current */

#define EKVqbd EKVstates+ 13  /* bulk-drain capacitor charge */
#define EKVcqbd EKVstates+ 14 /* bulk-drain capacitor current */

#define EKVqbs EKVstates+ 15  /* bulk-source capacitor charge */
#define EKVcqbs EKVstates+ 16 /* bulk-source capacitor current */

#define EKVnumStates 17

#define EKVsensxpgs EKVstates+17 /* charge sensitivities and 
their derivatives.  +18 for the derivatives:
pointer to the beginning of the array */
#define EKVsensxpgd  EKVstates+19
#define EKVsensxpgb  EKVstates+21
#define EKVsensxpbs  EKVstates+23
#define EKVsensxpbd  EKVstates+25

#define EKVnumSenStates 10


/* per model data */

/* NOTE:  parameters marked 'input - use xxxx' are paramters for
     * which a temperature correction is applied in EKVtemp, thus
     * the EKVxxxx value in the per-instance structure should be used
     * instead in all calculations 
     */


typedef struct sEKVmodel {          /* model structure for a resistor */
	int EKVmodType;                 /* type index to this device type */
	struct sEKVmodel *EKVnextModel; /* pointer to next possible model 
	                                     *in linked list */
	EKVinstance * EKVinstances;     /* pointer to list of instances 
	                                     * that have this model */
	IFuid EKVmodName;               /* pointer to character string naming this model */
	int EKVtype;       /* device type : 1 = nmos,  -1 = pmos */
	double EKVtnom;    /* temperature at which parameters measured */
	double EKVekvint;
	double EKVvt0;     /* input - use tvt0 */
	double EKVkp;      /* input - use tkp */
	double EKVgamma;
	double EKVphi;     /* input - use tphi */
	double EKVcox;
	double EKVxj;
	double EKVtheta;
	double EKVucrit;   /* input - use tucrit */
	double EKVdw;
	double EKVdl;
	double EKVlambda;
	double EKVweta;
	double EKVleta;
	double EKViba;
	double EKVibb;     /* input - use tibb */
	double EKVibn;
	double EKVq0;
	double EKVlk;
	double EKVtcv;
	double EKVbex;
	double EKVucex;
	double EKVibbt;
	double EKVnqs;
	double EKVsatlim;
	double EKVfNcoef;
	double EKVfNexp;
	double EKVjctSatCur;           /* input - use tSatCur */
	double EKVjctSatCurDensity;    /* input - use tSatCurDens */
	double EKVjsw;
	double EKVn;
	double EKVcapBD;               /* input - use tCbd */
	double EKVcapBS;               /* input - use tCbs */
	double EKVbulkCapFactor;       /* input - use tCj */
	double EKVsideWallCapFactor;   /* input - use tCjsw */
	double EKVbulkJctBotGradingCoeff;
	double EKVbulkJctSideGradingCoeff;
	double EKVfwdCapDepCoeff;
	double EKVbulkJctPotential;
	double EKVpbsw;
	double EKVtt;
	double EKVgateSourceOverlapCapFactor;
	double EKVgateDrainOverlapCapFactor;
	double EKVgateBulkOverlapCapFactor;
	double EKVdrainResistance;     /* input - use trd */
	double EKVsourceResistance;    /* input - use trs */
	double EKVsheetResistance;     /* input - use trsh */
	double EKVrsc;                 /* input - use trsc */
	double EKVrdc;                 /* input - use trdc */
	double EKVxti;
	double EKVtr1;
	double EKVtr2;
	double EKVnlevel;
	double EKVe0;                  /* wg 17-SEP-2K */
	int EKVgateType;

	unsigned EKVtypeGiven  :1;
	unsigned EKVekvintGiven :1;
	unsigned EKVtnomGiven  :1;
	unsigned EKVvt0Given   :1;
	unsigned EKVkpGiven  :1;
	unsigned EKVgammaGiven :1;
	unsigned EKVphiGiven   :1;
	unsigned EKVcoxGiven :1;
	unsigned EKVxjGiven :1;
	unsigned EKVthetaGiven :1;
	unsigned EKVucritGiven :1;
	unsigned EKVdwGiven :1;
	unsigned EKVdlGiven :1;
	unsigned EKVlambdaGiven    :1;
	unsigned EKVwetaGiven :1;
	unsigned EKVletaGiven :1;
	unsigned EKVibaGiven :1;
	unsigned EKVibbGiven :1;
	unsigned EKVibnGiven :1;
	unsigned EKVq0Given :1;
	unsigned EKVlkGiven :1;
	unsigned EKVtcvGiven :1;
	unsigned EKVbexGiven :1;
	unsigned EKVucexGiven :1;
	unsigned EKVibbtGiven :1;
	unsigned EKVnqsGiven :1;
	unsigned EKVsatlimGiven :1;
	unsigned EKVfNcoefGiven  :1;
	unsigned EKVfNexpGiven   :1;
	unsigned EKVjctSatCurGiven :1;
	unsigned EKVjctSatCurDensityGiven  :1;
	unsigned EKVjswGiven :1;
	unsigned EKVnGiven :1;
	unsigned EKVcapBDGiven :1;
	unsigned EKVcapBSGiven :1;
	unsigned EKVbulkCapFactorGiven :1;
	unsigned EKVsideWallCapFactorGiven   :1;
	unsigned EKVbulkJctBotGradingCoeffGiven    :1;
	unsigned EKVbulkJctSideGradingCoeffGiven   :1;
	unsigned EKVfwdCapDepCoeffGiven    :1;
	unsigned EKVbulkJctPotentialGiven  :1;
	unsigned EKVpbswGiven :1;
	unsigned EKVttGiven :1;
	unsigned EKVgateSourceOverlapCapFactorGiven    :1;
	unsigned EKVgateDrainOverlapCapFactorGiven :1;
	unsigned EKVgateBulkOverlapCapFactorGiven  :1;
	unsigned EKVdrainResistanceGiven   :1;
	unsigned EKVsourceResistanceGiven  :1;
	unsigned EKVsheetResistanceGiven   :1;
	unsigned EKVrscGiven :1;
	unsigned EKVrdcGiven :1;
	unsigned EKVxtiGiven :1;
	unsigned EKVtr1Given :1;
	unsigned EKVtr2Given :1;
	unsigned EKVnlevelGiven :1;
	unsigned EKVe0Given :1;                  /* wg 17-SEP-2K */

} EKVmodel;

#ifndef NMOS
#define NMOS 1
#define PMOS -1
#endif /*NMOS*/

/* device parameters */
#define EKV_W       1
#define EKV_L       2
#define EKV_AS      3
#define EKV_AD      4
#define EKV_PS      5
#define EKV_PD      6
#define EKV_NRS     7
#define EKV_NRD     8
#define EKV_OFF     9
#define EKV_IC     10
#define EKV_IC_VBS 11
#define EKV_IC_VDS 12
#define EKV_IC_VGS 13
#define EKV_W_SENS 14
#define EKV_L_SENS 15
#define EKV_CB     16
#define EKV_CG     17
#define EKV_CS     18
#define EKV_POWER  19
#define EKV_TEMP   20

#define EKV_TVTO    21
#define EKV_TPHI    22
#define EKV_TKP     23
#define EKV_TUCRIT  24
#define EKV_TIBB    25
#define EKV_TRS     26
#define EKV_TRD     27
#define EKV_TRSH    28
#define EKV_TRSC    29
#define EKV_TRDC    30 
#define EKV_TIS     31
#define EKV_TJS     32
#define EKV_TJSW    33
#define EKV_TPB     34
#define EKV_TPBSW   35
#define EKV_TCBD    36
#define EKV_TCBS    37
#define EKV_TCJ     38
#define EKV_TCJSW   39
#define EKV_TAF     40
#define EKV_ISUB    41
#define EKV_VP      42
#define EKV_SLOPE   43
#define EKV_IF      44
#define EKV_IR      45
#define EKV_IRPRIME 46
#define EKV_TAU     47

/* model paramerers */
#define EKV_MOD_TNOM   101
#define EKV_MOD_VTO    102
#define EKV_MOD_KP     103
#define EKV_MOD_GAMMA  104
#define EKV_MOD_PHI    105
#define EKV_MOD_COX    106
#define EKV_MOD_XJ     107
#define EKV_MOD_THETA  108
#define EKV_MOD_UCRIT  109
#define EKV_MOD_DW     110
#define EKV_MOD_DL     111
#define EKV_MOD_LAMBDA 112
#define EKV_MOD_WETA   113
#define EKV_MOD_LETA   114
#define EKV_MOD_IBA    115
#define EKV_MOD_IBB    116
#define EKV_MOD_IBN    117
#define EKV_MOD_TCV    118
#define EKV_MOD_BEX    119
#define EKV_MOD_UCEX   120
#define EKV_MOD_IBBT   121
#define EKV_MOD_NQS    122
#define EKV_MOD_SATLIM 123
#define EKV_MOD_KF     124
#define EKV_MOD_AF     125 
#define EKV_MOD_IS     126 
#define EKV_MOD_JS     127 
#define EKV_MOD_JSW    128 
#define EKV_MOD_N      129
#define EKV_MOD_CBD    130
#define EKV_MOD_CBS    131
#define EKV_MOD_CJ     132 
#define EKV_MOD_CJSW   133 
#define EKV_MOD_MJ     134 
#define EKV_MOD_MJSW   135 
#define EKV_MOD_FC     136 
#define EKV_MOD_PB     137
#define EKV_MOD_PBSW   138
#define EKV_MOD_TT     139
#define EKV_MOD_CGSO   140 
#define EKV_MOD_CGDO   141 
#define EKV_MOD_CGBO   142 
#define EKV_MOD_RD     143
#define EKV_MOD_RS     144
#define EKV_MOD_RSH    145
#define EKV_MOD_RSC    146
#define EKV_MOD_RDC    147
#define EKV_MOD_XTI    148
#define EKV_MOD_TR1    149
#define EKV_MOD_TR2    150
#define EKV_MOD_NLEVEL 151
#define EKV_MOD_EKVINT 152 
#define EKV_MOD_Q0     153 
#define EKV_MOD_LK     154 
#define EKV_MOD_NMOS   155 
#define EKV_MOD_PMOS   156 
#define EKV_MOD_TYPE   157 
#define EKV_MOD_E0     158                   /* wg 17-SEP-2K */

/* device questions */
#define EKV_CGS                201
#define EKV_CGD                202
#define EKV_DNODE              203
#define EKV_GNODE              204
#define EKV_SNODE              205
#define EKV_BNODE              206
#define EKV_DNODEPRIME         207
#define EKV_SNODEPRIME         208
#define EKV_SOURCECONDUCT      209
#define EKV_DRAINCONDUCT       210
#define EKV_VON                211
#define EKV_VDSAT              212
#define EKV_VGEFF              213
#define EKV_VGPRIME            214
#define EKV_SOURCEVCRIT        215
#define EKV_DRAINVCRIT         216
#define EKV_CD                 217
#define EKV_CBS                218
#define EKV_CBD                219
#define EKV_GMBS               220
#define EKV_GM                 221
#define EKV_GMS                222
#define EKV_GDS                223
#define EKV_GBD                224
#define EKV_GBS                225
#define EKV_CAPBD              226
#define EKV_CAPBS              227
#define EKV_CAPZEROBIASBD      228
#define EKV_CAPZEROBIASBDSW    229
#define EKV_CAPZEROBIASBS      230
#define EKV_CAPZEROBIASBSSW    231
#define EKV_VBD                232
#define EKV_VBS                233
#define EKV_VGS                234
#define EKV_VDS                235
#define EKV_CAPGS              236
#define EKV_QGS                237
#define EKV_CQGS               238
#define EKV_CAPGD              239
#define EKV_QGD                240
#define EKV_CQGD               241
#define EKV_CAPGB              242
#define EKV_QGB                243
#define EKV_CQGB               244
#define EKV_QBD                245
#define EKV_CQBD               246
#define EKV_QBS                247
#define EKV_CQBS               248
#define EKV_L_SENS_REAL        249
#define EKV_L_SENS_IMAG        250
#define EKV_L_SENS_MAG         251 
#define EKV_L_SENS_PH          252 
#define EKV_L_SENS_CPLX        253
#define EKV_W_SENS_REAL        254
#define EKV_W_SENS_IMAG        255
#define EKV_W_SENS_MAG         256 
#define EKV_W_SENS_PH          257 
#define EKV_W_SENS_CPLX        258
#define EKV_L_SENS_DC          259
#define EKV_W_SENS_DC          260
#define EKV_SOURCERESIST       261
#define EKV_DRAINRESIST        262
#define EKV_VGSTAR             263

/* model questions */

#include "ekvext.h"

#endif /*EKV*/

