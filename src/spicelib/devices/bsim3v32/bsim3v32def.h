/**********
Copyright 2001 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Author: 1997-1999 Weidong Liu.
Author: 2001 Xuemei Xi
Modified by Paolo Nenzi 2002 and Dietmar Warning 2003
File: bsim3v32def.h
**********/

#ifndef BSIM3v32
#define BSIM3v32

#include "ngspice/ifsim.h"
#include "ngspice/gendefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/noisedef.h"

typedef struct sBSIM3v32instance
{

    struct GENinstance gen;

#define BSIM3v32modPtr(inst) ((struct sBSIM3v32model *)((inst)->gen.GENmodPtr))
#define BSIM3v32nextInstance(inst) ((struct sBSIM3v32instance *)((inst)->gen.GENnextInstance))
#define BSIM3v32name gen.GENname
#define BSIM3v32states gen.GENstate

    const int BSIM3v32dNode;
    const int BSIM3v32gNode;
    const int BSIM3v32sNode;
    const int BSIM3v32bNode;
    int BSIM3v32dNodePrime;
    int BSIM3v32sNodePrime;
    int BSIM3v32qNode; /* MCJ */

    /* MCJ */
    double BSIM3v32ueff;
    double BSIM3v32thetavth;
    double BSIM3v32von;
    double BSIM3v32vdsat;
    double BSIM3v32cgdo;
    double BSIM3v32cgso;
    double BSIM3v32vjsm;
    double BSIM3v32IsEvjsm;
    double BSIM3v32vjdm;
    double BSIM3v32IsEvjdm;

    double BSIM3v32l;
    double BSIM3v32w;
    double BSIM3v32m;
    double BSIM3v32drainArea;
    double BSIM3v32sourceArea;
    double BSIM3v32drainSquares;
    double BSIM3v32sourceSquares;
    double BSIM3v32drainPerimeter;
    double BSIM3v32sourcePerimeter;
    double BSIM3v32sourceConductance;
    double BSIM3v32drainConductance;
    double BSIM3v32delvto;
    double BSIM3v32mulu0;
    double BSIM3v32vth0;
    double BSIM3v32vfb;
    double BSIM3v32vfbzb;
    double BSIM3v32u0temp;
    double BSIM3v32tconst;

    double BSIM3v32icVBS;
    double BSIM3v32icVDS;
    double BSIM3v32icVGS;
    int BSIM3v32off;
    int BSIM3v32mode;
    int BSIM3v32nqsMod;
    int BSIM3v32geo;

    /* OP point */
    double BSIM3v32qinv;
    double BSIM3v32cd;
    double BSIM3v32cbs;
    double BSIM3v32cbd;
    double BSIM3v32csub;
    double BSIM3v32gm;
    double BSIM3v32gds;
    double BSIM3v32gmbs;
    double BSIM3v32gbd;
    double BSIM3v32gbs;

    double BSIM3v32gbbs;
    double BSIM3v32gbgs;
    double BSIM3v32gbds;

    double BSIM3v32cggb;
    double BSIM3v32cgdb;
    double BSIM3v32cgsb;
    double BSIM3v32cbgb;
    double BSIM3v32cbdb;
    double BSIM3v32cbsb;
    double BSIM3v32cdgb;
    double BSIM3v32cddb;
    double BSIM3v32cdsb;
    double BSIM3v32capbd;
    double BSIM3v32capbs;

    double BSIM3v32cqgb;
    double BSIM3v32cqdb;
    double BSIM3v32cqsb;
    double BSIM3v32cqbb;

    double BSIM3v32qgate;
    double BSIM3v32qbulk;
    double BSIM3v32qdrn;

    double BSIM3v32gtau;
    double BSIM3v32gtg;
    double BSIM3v32gtd;
    double BSIM3v32gts;
    double BSIM3v32gtb;
    double BSIM3v32rds;  /* Noise bugfix */
    double BSIM3v32Vgsteff;
    double BSIM3v32Vdseff;
    double BSIM3v32Abulk;
    double BSIM3v32AbovVgst2Vtm;

    struct bsim3v32SizeDependParam  *pParam;

    unsigned BSIM3v32lGiven :1;
    unsigned BSIM3v32wGiven :1;
    unsigned BSIM3v32mGiven :1;
    unsigned BSIM3v32drainAreaGiven :1;
    unsigned BSIM3v32sourceAreaGiven    :1;
    unsigned BSIM3v32drainSquaresGiven  :1;
    unsigned BSIM3v32sourceSquaresGiven :1;
    unsigned BSIM3v32drainPerimeterGiven    :1;
    unsigned BSIM3v32sourcePerimeterGiven   :1;
    unsigned BSIM3v32delvtoGiven   :1;
    unsigned BSIM3v32mulu0Given   :1;
    unsigned BSIM3v32dNodePrimeSet  :1;
    unsigned BSIM3v32sNodePrimeSet  :1;
    unsigned BSIM3v32icVBSGiven :1;
    unsigned BSIM3v32icVDSGiven :1;
    unsigned BSIM3v32icVGSGiven :1;
    unsigned BSIM3v32nqsModGiven :1;
    unsigned BSIM3v32geoGiven :1;

    double *BSIM3v32DdPtr;
    double *BSIM3v32GgPtr;
    double *BSIM3v32SsPtr;
    double *BSIM3v32BbPtr;
    double *BSIM3v32DPdpPtr;
    double *BSIM3v32SPspPtr;
    double *BSIM3v32DdpPtr;
    double *BSIM3v32GbPtr;
    double *BSIM3v32GdpPtr;
    double *BSIM3v32GspPtr;
    double *BSIM3v32SspPtr;
    double *BSIM3v32BdpPtr;
    double *BSIM3v32BspPtr;
    double *BSIM3v32DPspPtr;
    double *BSIM3v32DPdPtr;
    double *BSIM3v32BgPtr;
    double *BSIM3v32DPgPtr;
    double *BSIM3v32SPgPtr;
    double *BSIM3v32SPsPtr;
    double *BSIM3v32DPbPtr;
    double *BSIM3v32SPbPtr;
    double *BSIM3v32SPdpPtr;

    double *BSIM3v32QqPtr;
    double *BSIM3v32QdpPtr;
    double *BSIM3v32QgPtr;
    double *BSIM3v32QspPtr;
    double *BSIM3v32QbPtr;
    double *BSIM3v32DPqPtr;
    double *BSIM3v32GqPtr;
    double *BSIM3v32SPqPtr;
    double *BSIM3v32BqPtr;

#ifdef USE_OMP
    /* per instance storage of results, to update matrix at a later stge */
    double BSIM3v32rhsG;
    double BSIM3v32rhsB;
    double BSIM3v32rhsD;
    double BSIM3v32rhsS;
    double BSIM3v32rhsQ;

    double BSIM3v32DdPt;
    double BSIM3v32GgPt;
    double BSIM3v32SsPt;
    double BSIM3v32BbPt;
    double BSIM3v32DPdpPt;
    double BSIM3v32SPspPt;
    double BSIM3v32DdpPt;
    double BSIM3v32GbPt;
    double BSIM3v32GdpPt;
    double BSIM3v32GspPt;
    double BSIM3v32SspPt;
    double BSIM3v32BdpPt;
    double BSIM3v32BspPt;
    double BSIM3v32DPspPt;
    double BSIM3v32DPdPt;
    double BSIM3v32BgPt;
    double BSIM3v32DPgPt;
    double BSIM3v32SPgPt;
    double BSIM3v32SPsPt;
    double BSIM3v32DPbPt;
    double BSIM3v32SPbPt;
    double BSIM3v32SPdpPt;

    double BSIM3v32QqPt;
    double BSIM3v32QdpPt;
    double BSIM3v32QgPt;
    double BSIM3v32QspPt;
    double BSIM3v32QbPt;
    double BSIM3v32DPqPt;
    double BSIM3v32GqPt;
    double BSIM3v32SPqPt;
    double BSIM3v32BqPt;
#endif

#define BSIM3v32vbd BSIM3v32states+ 0
#define BSIM3v32vbs BSIM3v32states+ 1
#define BSIM3v32vgs BSIM3v32states+ 2
#define BSIM3v32vds BSIM3v32states+ 3

#define BSIM3v32qb BSIM3v32states+ 4
#define BSIM3v32cqb BSIM3v32states+ 5
#define BSIM3v32qg BSIM3v32states+ 6
#define BSIM3v32cqg BSIM3v32states+ 7
#define BSIM3v32qd BSIM3v32states+ 8
#define BSIM3v32cqd BSIM3v32states+ 9

#define BSIM3v32qbs  BSIM3v32states+ 10
#define BSIM3v32qbd  BSIM3v32states+ 11

#define BSIM3v32qcheq BSIM3v32states+ 12
#define BSIM3v32cqcheq BSIM3v32states+ 13
#define BSIM3v32qcdump BSIM3v32states+ 14
#define BSIM3v32cqcdump BSIM3v32states+ 15

#define BSIM3v32qdef BSIM3v32states+ 16

#define BSIM3v32numStates 17


/* indices to the array of BSIM3v32 NOISE SOURCES */

#define BSIM3v32RDNOIZ       0
#define BSIM3v32RSNOIZ       1
#define BSIM3v32IDNOIZ       2
#define BSIM3v32FLNOIZ       3
#define BSIM3v32TOTNOIZ      4

#define BSIM3v32NSRCS        5  /* the number of BSIM3v32 MOSFET noise sources */

#ifndef NONOISE
    double BSIM3v32nVar[NSTATVARS][BSIM3v32NSRCS];
#else /* NONOISE */
        double **BSIM3v32nVar;
#endif /* NONOISE */

} BSIM3v32instance ;

struct bsim3v32SizeDependParam
{
    double Width;
    double Length;

    double BSIM3v32cdsc;
    double BSIM3v32cdscb;
    double BSIM3v32cdscd;
    double BSIM3v32cit;
    double BSIM3v32nfactor;
    double BSIM3v32xj;
    double BSIM3v32vsat;
    double BSIM3v32at;
    double BSIM3v32a0;
    double BSIM3v32ags;
    double BSIM3v32a1;
    double BSIM3v32a2;
    double BSIM3v32keta;
    double BSIM3v32nsub;
    double BSIM3v32npeak;
    double BSIM3v32ngate;
    double BSIM3v32gamma1;
    double BSIM3v32gamma2;
    double BSIM3v32vbx;
    double BSIM3v32vbi;
    double BSIM3v32vbm;
    double BSIM3v32vbsc;
    double BSIM3v32xt;
    double BSIM3v32phi;
    double BSIM3v32litl;
    double BSIM3v32k1;
    double BSIM3v32kt1;
    double BSIM3v32kt1l;
    double BSIM3v32kt2;
    double BSIM3v32k2;
    double BSIM3v32k3;
    double BSIM3v32k3b;
    double BSIM3v32w0;
    double BSIM3v32nlx;
    double BSIM3v32dvt0;
    double BSIM3v32dvt1;
    double BSIM3v32dvt2;
    double BSIM3v32dvt0w;
    double BSIM3v32dvt1w;
    double BSIM3v32dvt2w;
    double BSIM3v32drout;
    double BSIM3v32dsub;
    double BSIM3v32vth0;
    double BSIM3v32ua;
    double BSIM3v32ua1;
    double BSIM3v32ub;
    double BSIM3v32ub1;
    double BSIM3v32uc;
    double BSIM3v32uc1;
    double BSIM3v32u0;
    double BSIM3v32ute;
    double BSIM3v32voff;
    double BSIM3v32vfb;
    double BSIM3v32delta;
    double BSIM3v32rdsw;
    double BSIM3v32rds0;
    double BSIM3v32prwg;
    double BSIM3v32prwb;
    double BSIM3v32prt;
    double BSIM3v32eta0;
    double BSIM3v32etab;
    double BSIM3v32pclm;
    double BSIM3v32pdibl1;
    double BSIM3v32pdibl2;
    double BSIM3v32pdiblb;
    double BSIM3v32pscbe1;
    double BSIM3v32pscbe2;
    double BSIM3v32pvag;
    double BSIM3v32wr;
    double BSIM3v32dwg;
    double BSIM3v32dwb;
    double BSIM3v32b0;
    double BSIM3v32b1;
    double BSIM3v32alpha0;
    double BSIM3v32alpha1;
    double BSIM3v32beta0;


    /* CV model */
    double BSIM3v32elm;
    double BSIM3v32cgsl;
    double BSIM3v32cgdl;
    double BSIM3v32ckappa;
    double BSIM3v32cf;
    double BSIM3v32clc;
    double BSIM3v32cle;
    double BSIM3v32vfbcv;
    double BSIM3v32noff;
    double BSIM3v32voffcv;
    double BSIM3v32acde;
    double BSIM3v32moin;


/* Pre-calculated constants */

    double BSIM3v32dw;
    double BSIM3v32dl;
    double BSIM3v32leff;
    double BSIM3v32weff;

    double BSIM3v32dwc;
    double BSIM3v32dlc;
    double BSIM3v32leffCV;
    double BSIM3v32weffCV;
    double BSIM3v32abulkCVfactor;
    double BSIM3v32cgso;
    double BSIM3v32cgdo;
    double BSIM3v32cgbo;
    double BSIM3v32tconst;

    double BSIM3v32u0temp;
    double BSIM3v32vsattemp;
    double BSIM3v32sqrtPhi;
    double BSIM3v32phis3;
    double BSIM3v32Xdep0;
    double BSIM3v32sqrtXdep0;
    double BSIM3v32theta0vb0;
    double BSIM3v32thetaRout;

    double BSIM3v32cof1;
    double BSIM3v32cof2;
    double BSIM3v32cof3;
    double BSIM3v32cof4;
    double BSIM3v32cdep0;
    double BSIM3v32vfbzb;
    double BSIM3v32ldeb;
    double BSIM3v32k1ox;
    double BSIM3v32k2ox;

    struct bsim3v32SizeDependParam  *pNext;
};


typedef struct sBSIM3v32model
{

    struct GENmodel gen;

#define BSIM3v32modType gen.GENmodType
#define BSIM3v32nextModel(inst) ((struct sBSIM3v32model *)((inst)->gen.GENnextModel))
#define BSIM3v32instances(inst) ((BSIM3v32instance *)((inst)->gen.GENinstances))
#define BSIM3v32modName gen.GENmodName

    int BSIM3v32type;

    int    BSIM3v32mobMod;
    int    BSIM3v32capMod;
    int    BSIM3v32acmMod;
    int    BSIM3v32calcacm;
    int    BSIM3v32noiMod;
    int    BSIM3v32nqsMod;
    int    BSIM3v32binUnit;
    int    BSIM3v32paramChk;
    char   *BSIM3v32version;
    /* The following field is an integer coding
     * of BSIM3v32version.
     */
    int    BSIM3v32intVersion;
#define BSIM3v32V324  324       /* BSIM3v32 V3.2.4 */
#define BSIM3v32V323  323       /* BSIM3v32 V3.2.3 */
#define BSIM3v32V322  322       /* BSIM3v32 V3.2.2 */
#define BSIM3v32V32   32        /* BSIM3v32 V3.2   */
#define BSIM3v32V3OLD 0         /* Old model    */
    double BSIM3v32tox;
    double BSIM3v32toxm;
    double BSIM3v32cdsc;
    double BSIM3v32cdscb;
    double BSIM3v32cdscd;
    double BSIM3v32cit;
    double BSIM3v32nfactor;
    double BSIM3v32xj;
    double BSIM3v32vsat;
    double BSIM3v32at;
    double BSIM3v32a0;
    double BSIM3v32ags;
    double BSIM3v32a1;
    double BSIM3v32a2;
    double BSIM3v32keta;
    double BSIM3v32nsub;
    double BSIM3v32npeak;
    double BSIM3v32ngate;
    double BSIM3v32gamma1;
    double BSIM3v32gamma2;
    double BSIM3v32vbx;
    double BSIM3v32vbm;
    double BSIM3v32xt;
    double BSIM3v32k1;
    double BSIM3v32kt1;
    double BSIM3v32kt1l;
    double BSIM3v32kt2;
    double BSIM3v32k2;
    double BSIM3v32k3;
    double BSIM3v32k3b;
    double BSIM3v32w0;
    double BSIM3v32nlx;
    double BSIM3v32dvt0;
    double BSIM3v32dvt1;
    double BSIM3v32dvt2;
    double BSIM3v32dvt0w;
    double BSIM3v32dvt1w;
    double BSIM3v32dvt2w;
    double BSIM3v32drout;
    double BSIM3v32dsub;
    double BSIM3v32vth0;
    double BSIM3v32ua;
    double BSIM3v32ua1;
    double BSIM3v32ub;
    double BSIM3v32ub1;
    double BSIM3v32uc;
    double BSIM3v32uc1;
    double BSIM3v32u0;
    double BSIM3v32ute;
    double BSIM3v32voff;
    double BSIM3v32delta;
    double BSIM3v32rdsw;
    double BSIM3v32prwg;
    double BSIM3v32prwb;
    double BSIM3v32prt;
    double BSIM3v32eta0;
    double BSIM3v32etab;
    double BSIM3v32pclm;
    double BSIM3v32pdibl1;
    double BSIM3v32pdibl2;
    double BSIM3v32pdiblb;
    double BSIM3v32pscbe1;
    double BSIM3v32pscbe2;
    double BSIM3v32pvag;
    double BSIM3v32wr;
    double BSIM3v32dwg;
    double BSIM3v32dwb;
    double BSIM3v32b0;
    double BSIM3v32b1;
    double BSIM3v32alpha0;
    double BSIM3v32alpha1;
    double BSIM3v32beta0;
    double BSIM3v32ijth;
    double BSIM3v32vfb;

    /* CV model */
    double BSIM3v32elm;
    double BSIM3v32cgsl;
    double BSIM3v32cgdl;
    double BSIM3v32ckappa;
    double BSIM3v32cf;
    double BSIM3v32vfbcv;
    double BSIM3v32clc;
    double BSIM3v32cle;
    double BSIM3v32dwc;
    double BSIM3v32dlc;
    double BSIM3v32noff;
    double BSIM3v32voffcv;
    double BSIM3v32acde;
    double BSIM3v32moin;
    double BSIM3v32tcj;
    double BSIM3v32tcjsw;
    double BSIM3v32tcjswg;
    double BSIM3v32tpb;
    double BSIM3v32tpbsw;
    double BSIM3v32tpbswg;

    /* ACM model */
    double BSIM3v32xl;
    double BSIM3v32xw;
    double BSIM3v32hdif;
    double BSIM3v32ldif;
    double BSIM3v32ld;
    double BSIM3v32rd;
    double BSIM3v32rs;
    double BSIM3v32rdc;
    double BSIM3v32rsc;
    double BSIM3v32wmlt;

    double BSIM3v32lmlt;

    /* Length Dependence */
    double BSIM3v32lcdsc;
    double BSIM3v32lcdscb;
    double BSIM3v32lcdscd;
    double BSIM3v32lcit;
    double BSIM3v32lnfactor;
    double BSIM3v32lxj;
    double BSIM3v32lvsat;
    double BSIM3v32lat;
    double BSIM3v32la0;
    double BSIM3v32lags;
    double BSIM3v32la1;
    double BSIM3v32la2;
    double BSIM3v32lketa;
    double BSIM3v32lnsub;
    double BSIM3v32lnpeak;
    double BSIM3v32lngate;
    double BSIM3v32lgamma1;
    double BSIM3v32lgamma2;
    double BSIM3v32lvbx;
    double BSIM3v32lvbm;
    double BSIM3v32lxt;
    double BSIM3v32lk1;
    double BSIM3v32lkt1;
    double BSIM3v32lkt1l;
    double BSIM3v32lkt2;
    double BSIM3v32lk2;
    double BSIM3v32lk3;
    double BSIM3v32lk3b;
    double BSIM3v32lw0;
    double BSIM3v32lnlx;
    double BSIM3v32ldvt0;
    double BSIM3v32ldvt1;
    double BSIM3v32ldvt2;
    double BSIM3v32ldvt0w;
    double BSIM3v32ldvt1w;
    double BSIM3v32ldvt2w;
    double BSIM3v32ldrout;
    double BSIM3v32ldsub;
    double BSIM3v32lvth0;
    double BSIM3v32lua;
    double BSIM3v32lua1;
    double BSIM3v32lub;
    double BSIM3v32lub1;
    double BSIM3v32luc;
    double BSIM3v32luc1;
    double BSIM3v32lu0;
    double BSIM3v32lute;
    double BSIM3v32lvoff;
    double BSIM3v32ldelta;
    double BSIM3v32lrdsw;
    double BSIM3v32lprwg;
    double BSIM3v32lprwb;
    double BSIM3v32lprt;
    double BSIM3v32leta0;
    double BSIM3v32letab;
    double BSIM3v32lpclm;
    double BSIM3v32lpdibl1;
    double BSIM3v32lpdibl2;
    double BSIM3v32lpdiblb;
    double BSIM3v32lpscbe1;
    double BSIM3v32lpscbe2;
    double BSIM3v32lpvag;
    double BSIM3v32lwr;
    double BSIM3v32ldwg;
    double BSIM3v32ldwb;
    double BSIM3v32lb0;
    double BSIM3v32lb1;
    double BSIM3v32lalpha0;
    double BSIM3v32lalpha1;
    double BSIM3v32lbeta0;
    double BSIM3v32lvfb;

    /* CV model */
    double BSIM3v32lelm;
    double BSIM3v32lcgsl;
    double BSIM3v32lcgdl;
    double BSIM3v32lckappa;
    double BSIM3v32lcf;
    double BSIM3v32lclc;
    double BSIM3v32lcle;
    double BSIM3v32lvfbcv;
    double BSIM3v32lnoff;
    double BSIM3v32lvoffcv;
    double BSIM3v32lacde;
    double BSIM3v32lmoin;

    /* Width Dependence */
    double BSIM3v32wcdsc;
    double BSIM3v32wcdscb;
    double BSIM3v32wcdscd;
    double BSIM3v32wcit;
    double BSIM3v32wnfactor;
    double BSIM3v32wxj;
    double BSIM3v32wvsat;
    double BSIM3v32wat;
    double BSIM3v32wa0;
    double BSIM3v32wags;
    double BSIM3v32wa1;
    double BSIM3v32wa2;
    double BSIM3v32wketa;
    double BSIM3v32wnsub;
    double BSIM3v32wnpeak;
    double BSIM3v32wngate;
    double BSIM3v32wgamma1;
    double BSIM3v32wgamma2;
    double BSIM3v32wvbx;
    double BSIM3v32wvbm;
    double BSIM3v32wxt;
    double BSIM3v32wk1;
    double BSIM3v32wkt1;
    double BSIM3v32wkt1l;
    double BSIM3v32wkt2;
    double BSIM3v32wk2;
    double BSIM3v32wk3;
    double BSIM3v32wk3b;
    double BSIM3v32ww0;
    double BSIM3v32wnlx;
    double BSIM3v32wdvt0;
    double BSIM3v32wdvt1;
    double BSIM3v32wdvt2;
    double BSIM3v32wdvt0w;
    double BSIM3v32wdvt1w;
    double BSIM3v32wdvt2w;
    double BSIM3v32wdrout;
    double BSIM3v32wdsub;
    double BSIM3v32wvth0;
    double BSIM3v32wua;
    double BSIM3v32wua1;
    double BSIM3v32wub;
    double BSIM3v32wub1;
    double BSIM3v32wuc;
    double BSIM3v32wuc1;
    double BSIM3v32wu0;
    double BSIM3v32wute;
    double BSIM3v32wvoff;
    double BSIM3v32wdelta;
    double BSIM3v32wrdsw;
    double BSIM3v32wprwg;
    double BSIM3v32wprwb;
    double BSIM3v32wprt;
    double BSIM3v32weta0;
    double BSIM3v32wetab;
    double BSIM3v32wpclm;
    double BSIM3v32wpdibl1;
    double BSIM3v32wpdibl2;
    double BSIM3v32wpdiblb;
    double BSIM3v32wpscbe1;
    double BSIM3v32wpscbe2;
    double BSIM3v32wpvag;
    double BSIM3v32wwr;
    double BSIM3v32wdwg;
    double BSIM3v32wdwb;
    double BSIM3v32wb0;
    double BSIM3v32wb1;
    double BSIM3v32walpha0;
    double BSIM3v32walpha1;
    double BSIM3v32wbeta0;
    double BSIM3v32wvfb;

    /* CV model */
    double BSIM3v32welm;
    double BSIM3v32wcgsl;
    double BSIM3v32wcgdl;
    double BSIM3v32wckappa;
    double BSIM3v32wcf;
    double BSIM3v32wclc;
    double BSIM3v32wcle;
    double BSIM3v32wvfbcv;
    double BSIM3v32wnoff;
    double BSIM3v32wvoffcv;
    double BSIM3v32wacde;
    double BSIM3v32wmoin;

    /* Cross-term Dependence */
    double BSIM3v32pcdsc;
    double BSIM3v32pcdscb;
    double BSIM3v32pcdscd;
    double BSIM3v32pcit;
    double BSIM3v32pnfactor;
    double BSIM3v32pxj;
    double BSIM3v32pvsat;
    double BSIM3v32pat;
    double BSIM3v32pa0;
    double BSIM3v32pags;
    double BSIM3v32pa1;
    double BSIM3v32pa2;
    double BSIM3v32pketa;
    double BSIM3v32pnsub;
    double BSIM3v32pnpeak;
    double BSIM3v32pngate;
    double BSIM3v32pgamma1;
    double BSIM3v32pgamma2;
    double BSIM3v32pvbx;
    double BSIM3v32pvbm;
    double BSIM3v32pxt;
    double BSIM3v32pk1;
    double BSIM3v32pkt1;
    double BSIM3v32pkt1l;
    double BSIM3v32pkt2;
    double BSIM3v32pk2;
    double BSIM3v32pk3;
    double BSIM3v32pk3b;
    double BSIM3v32pw0;
    double BSIM3v32pnlx;
    double BSIM3v32pdvt0;
    double BSIM3v32pdvt1;
    double BSIM3v32pdvt2;
    double BSIM3v32pdvt0w;
    double BSIM3v32pdvt1w;
    double BSIM3v32pdvt2w;
    double BSIM3v32pdrout;
    double BSIM3v32pdsub;
    double BSIM3v32pvth0;
    double BSIM3v32pua;
    double BSIM3v32pua1;
    double BSIM3v32pub;
    double BSIM3v32pub1;
    double BSIM3v32puc;
    double BSIM3v32puc1;
    double BSIM3v32pu0;
    double BSIM3v32pute;
    double BSIM3v32pvoff;
    double BSIM3v32pdelta;
    double BSIM3v32prdsw;
    double BSIM3v32pprwg;
    double BSIM3v32pprwb;
    double BSIM3v32pprt;
    double BSIM3v32peta0;
    double BSIM3v32petab;
    double BSIM3v32ppclm;
    double BSIM3v32ppdibl1;
    double BSIM3v32ppdibl2;
    double BSIM3v32ppdiblb;
    double BSIM3v32ppscbe1;
    double BSIM3v32ppscbe2;
    double BSIM3v32ppvag;
    double BSIM3v32pwr;
    double BSIM3v32pdwg;
    double BSIM3v32pdwb;
    double BSIM3v32pb0;
    double BSIM3v32pb1;
    double BSIM3v32palpha0;
    double BSIM3v32palpha1;
    double BSIM3v32pbeta0;
    double BSIM3v32pvfb;

    /* CV model */
    double BSIM3v32pelm;
    double BSIM3v32pcgsl;
    double BSIM3v32pcgdl;
    double BSIM3v32pckappa;
    double BSIM3v32pcf;
    double BSIM3v32pclc;
    double BSIM3v32pcle;
    double BSIM3v32pvfbcv;
    double BSIM3v32pnoff;
    double BSIM3v32pvoffcv;
    double BSIM3v32pacde;
    double BSIM3v32pmoin;

    double BSIM3v32tnom;
    double BSIM3v32cgso;
    double BSIM3v32cgdo;
    double BSIM3v32cgbo;
    double BSIM3v32xpart;
    double BSIM3v32cFringOut;
    double BSIM3v32cFringMax;

    double BSIM3v32sheetResistance;
    double BSIM3v32jctSatCurDensity;
    double BSIM3v32jctSidewallSatCurDensity;
    double BSIM3v32bulkJctPotential;
    double BSIM3v32bulkJctBotGradingCoeff;
    double BSIM3v32bulkJctSideGradingCoeff;
    double BSIM3v32bulkJctGateSideGradingCoeff;
    double BSIM3v32sidewallJctPotential;
    double BSIM3v32GatesidewallJctPotential;
    double BSIM3v32unitAreaJctCap;
    double BSIM3v32unitLengthSidewallJctCap;
    double BSIM3v32unitLengthGateSidewallJctCap;
    double BSIM3v32jctEmissionCoeff;
    double BSIM3v32jctTempExponent;

    double BSIM3v32Lint;
    double BSIM3v32Ll;
    double BSIM3v32Llc;
    double BSIM3v32Lln;
    double BSIM3v32Lw;
    double BSIM3v32Lwc;
    double BSIM3v32Lwn;
    double BSIM3v32Lwl;
    double BSIM3v32Lwlc;
    double BSIM3v32Lmin;
    double BSIM3v32Lmax;

    double BSIM3v32Wint;
    double BSIM3v32Wl;
    double BSIM3v32Wlc;
    double BSIM3v32Wln;
    double BSIM3v32Ww;
    double BSIM3v32Wwc;
    double BSIM3v32Wwn;
    double BSIM3v32Wwl;
    double BSIM3v32Wwlc;
    double BSIM3v32Wmin;
    double BSIM3v32Wmax;

/* Pre-calculated constants */
    /* MCJ: move to size-dependent param. */
    double BSIM3v32vtm;
    double BSIM3v32cox;
    double BSIM3v32cof1;
    double BSIM3v32cof2;
    double BSIM3v32cof3;
    double BSIM3v32cof4;
    double BSIM3v32vcrit;
    double BSIM3v32factor1;
    double BSIM3v32PhiB;
    double BSIM3v32PhiBSW;
    double BSIM3v32PhiBSWG;
    double BSIM3v32jctTempSatCurDensity;
    double BSIM3v32jctSidewallTempSatCurDensity;
    double BSIM3v32unitAreaTempJctCap;
    double BSIM3v32unitLengthSidewallTempJctCap;
    double BSIM3v32unitLengthGateSidewallTempJctCap;

    double BSIM3v32oxideTrapDensityA;
    double BSIM3v32oxideTrapDensityB;
    double BSIM3v32oxideTrapDensityC;
    double BSIM3v32em;
    double BSIM3v32ef;
    double BSIM3v32af;
    double BSIM3v32kf;

    double BSIM3v32vgsMax;
    double BSIM3v32vgdMax;
    double BSIM3v32vgbMax;
    double BSIM3v32vdsMax;
    double BSIM3v32vbsMax;
    double BSIM3v32vbdMax;
    double BSIM3v32vgsrMax;
    double BSIM3v32vgdrMax;
    double BSIM3v32vgbrMax;
    double BSIM3v32vbsrMax;
    double BSIM3v32vbdrMax;

    struct bsim3v32SizeDependParam *pSizeDependParamKnot;

#ifdef USE_OMP
    int BSIM3v32InstCount;
    struct sBSIM3v32instance **BSIM3v32InstanceArray;
#endif

    /* Flags */
    unsigned  BSIM3v32mobModGiven :1;
    unsigned  BSIM3v32binUnitGiven :1;
    unsigned  BSIM3v32capModGiven :1;
    unsigned  BSIM3v32acmModGiven :1;
    unsigned  BSIM3v32calcacmGiven :1;
    unsigned  BSIM3v32paramChkGiven :1;
    unsigned  BSIM3v32noiModGiven :1;
    unsigned  BSIM3v32nqsModGiven :1;
    unsigned  BSIM3v32typeGiven   :1;
    unsigned  BSIM3v32toxGiven   :1;
    unsigned  BSIM3v32versionGiven   :1;
    unsigned  BSIM3v32toxmGiven   :1;
    unsigned  BSIM3v32cdscGiven   :1;
    unsigned  BSIM3v32cdscbGiven   :1;
    unsigned  BSIM3v32cdscdGiven   :1;
    unsigned  BSIM3v32citGiven   :1;
    unsigned  BSIM3v32nfactorGiven   :1;
    unsigned  BSIM3v32xjGiven   :1;
    unsigned  BSIM3v32vsatGiven   :1;
    unsigned  BSIM3v32atGiven   :1;
    unsigned  BSIM3v32a0Given   :1;
    unsigned  BSIM3v32agsGiven   :1;
    unsigned  BSIM3v32a1Given   :1;
    unsigned  BSIM3v32a2Given   :1;
    unsigned  BSIM3v32ketaGiven   :1;
    unsigned  BSIM3v32nsubGiven   :1;
    unsigned  BSIM3v32npeakGiven   :1;
    unsigned  BSIM3v32ngateGiven   :1;
    unsigned  BSIM3v32gamma1Given   :1;
    unsigned  BSIM3v32gamma2Given   :1;
    unsigned  BSIM3v32vbxGiven   :1;
    unsigned  BSIM3v32vbmGiven   :1;
    unsigned  BSIM3v32xtGiven   :1;
    unsigned  BSIM3v32k1Given   :1;
    unsigned  BSIM3v32kt1Given   :1;
    unsigned  BSIM3v32kt1lGiven   :1;
    unsigned  BSIM3v32kt2Given   :1;
    unsigned  BSIM3v32k2Given   :1;
    unsigned  BSIM3v32k3Given   :1;
    unsigned  BSIM3v32k3bGiven   :1;
    unsigned  BSIM3v32w0Given   :1;
    unsigned  BSIM3v32nlxGiven   :1;
    unsigned  BSIM3v32dvt0Given   :1;
    unsigned  BSIM3v32dvt1Given   :1;
    unsigned  BSIM3v32dvt2Given   :1;
    unsigned  BSIM3v32dvt0wGiven   :1;
    unsigned  BSIM3v32dvt1wGiven   :1;
    unsigned  BSIM3v32dvt2wGiven   :1;
    unsigned  BSIM3v32droutGiven   :1;
    unsigned  BSIM3v32dsubGiven   :1;
    unsigned  BSIM3v32vth0Given   :1;
    unsigned  BSIM3v32uaGiven   :1;
    unsigned  BSIM3v32ua1Given   :1;
    unsigned  BSIM3v32ubGiven   :1;
    unsigned  BSIM3v32ub1Given   :1;
    unsigned  BSIM3v32ucGiven   :1;
    unsigned  BSIM3v32uc1Given   :1;
    unsigned  BSIM3v32u0Given   :1;
    unsigned  BSIM3v32uteGiven   :1;
    unsigned  BSIM3v32voffGiven   :1;
    unsigned  BSIM3v32rdswGiven   :1;
    unsigned  BSIM3v32prwgGiven   :1;
    unsigned  BSIM3v32prwbGiven   :1;
    unsigned  BSIM3v32prtGiven   :1;
    unsigned  BSIM3v32eta0Given   :1;
    unsigned  BSIM3v32etabGiven   :1;
    unsigned  BSIM3v32pclmGiven   :1;
    unsigned  BSIM3v32pdibl1Given   :1;
    unsigned  BSIM3v32pdibl2Given   :1;
    unsigned  BSIM3v32pdiblbGiven   :1;
    unsigned  BSIM3v32pscbe1Given   :1;
    unsigned  BSIM3v32pscbe2Given   :1;
    unsigned  BSIM3v32pvagGiven   :1;
    unsigned  BSIM3v32deltaGiven  :1;
    unsigned  BSIM3v32wrGiven   :1;
    unsigned  BSIM3v32dwgGiven   :1;
    unsigned  BSIM3v32dwbGiven   :1;
    unsigned  BSIM3v32b0Given   :1;
    unsigned  BSIM3v32b1Given   :1;
    unsigned  BSIM3v32alpha0Given   :1;
    unsigned  BSIM3v32alpha1Given   :1;
    unsigned  BSIM3v32beta0Given   :1;
    unsigned  BSIM3v32ijthGiven   :1;
    unsigned  BSIM3v32vfbGiven   :1;

    /* CV model */
    unsigned  BSIM3v32elmGiven  :1;
    unsigned  BSIM3v32cgslGiven   :1;
    unsigned  BSIM3v32cgdlGiven   :1;
    unsigned  BSIM3v32ckappaGiven   :1;
    unsigned  BSIM3v32cfGiven   :1;
    unsigned  BSIM3v32vfbcvGiven   :1;
    unsigned  BSIM3v32clcGiven   :1;
    unsigned  BSIM3v32cleGiven   :1;
    unsigned  BSIM3v32dwcGiven   :1;
    unsigned  BSIM3v32dlcGiven   :1;
    unsigned  BSIM3v32noffGiven  :1;
    unsigned  BSIM3v32voffcvGiven :1;
    unsigned  BSIM3v32acdeGiven  :1;
    unsigned  BSIM3v32moinGiven  :1;
    unsigned  BSIM3v32tcjGiven   :1;
    unsigned  BSIM3v32tcjswGiven :1;
    unsigned  BSIM3v32tcjswgGiven :1;
    unsigned  BSIM3v32tpbGiven    :1;
    unsigned  BSIM3v32tpbswGiven  :1;
    unsigned  BSIM3v32tpbswgGiven :1;

    /* ACM model */
    unsigned  BSIM3v32xlGiven   :1;
    unsigned  BSIM3v32xwGiven   :1;
    unsigned  BSIM3v32hdifGiven  :1;
    unsigned  BSIM3v32ldifGiven   :1;
    unsigned  BSIM3v32ldGiven   :1;
    unsigned  BSIM3v32rdGiven   :1;
    unsigned  BSIM3v32rsGiven   :1;
    unsigned  BSIM3v32rdcGiven   :1;
    unsigned  BSIM3v32rscGiven   :1;
    unsigned  BSIM3v32wmltGiven   :1;

    unsigned  BSIM3v32lmltGiven :1;

    /* Length dependence */
    unsigned  BSIM3v32lcdscGiven   :1;
    unsigned  BSIM3v32lcdscbGiven   :1;
    unsigned  BSIM3v32lcdscdGiven   :1;
    unsigned  BSIM3v32lcitGiven   :1;
    unsigned  BSIM3v32lnfactorGiven   :1;
    unsigned  BSIM3v32lxjGiven   :1;
    unsigned  BSIM3v32lvsatGiven   :1;
    unsigned  BSIM3v32latGiven   :1;
    unsigned  BSIM3v32la0Given   :1;
    unsigned  BSIM3v32lagsGiven   :1;
    unsigned  BSIM3v32la1Given   :1;
    unsigned  BSIM3v32la2Given   :1;
    unsigned  BSIM3v32lketaGiven   :1;
    unsigned  BSIM3v32lnsubGiven   :1;
    unsigned  BSIM3v32lnpeakGiven   :1;
    unsigned  BSIM3v32lngateGiven   :1;
    unsigned  BSIM3v32lgamma1Given   :1;
    unsigned  BSIM3v32lgamma2Given   :1;
    unsigned  BSIM3v32lvbxGiven   :1;
    unsigned  BSIM3v32lvbmGiven   :1;
    unsigned  BSIM3v32lxtGiven   :1;
    unsigned  BSIM3v32lk1Given   :1;
    unsigned  BSIM3v32lkt1Given   :1;
    unsigned  BSIM3v32lkt1lGiven   :1;
    unsigned  BSIM3v32lkt2Given   :1;
    unsigned  BSIM3v32lk2Given   :1;
    unsigned  BSIM3v32lk3Given   :1;
    unsigned  BSIM3v32lk3bGiven   :1;
    unsigned  BSIM3v32lw0Given   :1;
    unsigned  BSIM3v32lnlxGiven   :1;
    unsigned  BSIM3v32ldvt0Given   :1;
    unsigned  BSIM3v32ldvt1Given   :1;
    unsigned  BSIM3v32ldvt2Given   :1;
    unsigned  BSIM3v32ldvt0wGiven   :1;
    unsigned  BSIM3v32ldvt1wGiven   :1;
    unsigned  BSIM3v32ldvt2wGiven   :1;
    unsigned  BSIM3v32ldroutGiven   :1;
    unsigned  BSIM3v32ldsubGiven   :1;
    unsigned  BSIM3v32lvth0Given   :1;
    unsigned  BSIM3v32luaGiven   :1;
    unsigned  BSIM3v32lua1Given   :1;
    unsigned  BSIM3v32lubGiven   :1;
    unsigned  BSIM3v32lub1Given   :1;
    unsigned  BSIM3v32lucGiven   :1;
    unsigned  BSIM3v32luc1Given   :1;
    unsigned  BSIM3v32lu0Given   :1;
    unsigned  BSIM3v32luteGiven   :1;
    unsigned  BSIM3v32lvoffGiven   :1;
    unsigned  BSIM3v32lrdswGiven   :1;
    unsigned  BSIM3v32lprwgGiven   :1;
    unsigned  BSIM3v32lprwbGiven   :1;
    unsigned  BSIM3v32lprtGiven   :1;
    unsigned  BSIM3v32leta0Given   :1;
    unsigned  BSIM3v32letabGiven   :1;
    unsigned  BSIM3v32lpclmGiven   :1;
    unsigned  BSIM3v32lpdibl1Given   :1;
    unsigned  BSIM3v32lpdibl2Given   :1;
    unsigned  BSIM3v32lpdiblbGiven   :1;
    unsigned  BSIM3v32lpscbe1Given   :1;
    unsigned  BSIM3v32lpscbe2Given   :1;
    unsigned  BSIM3v32lpvagGiven   :1;
    unsigned  BSIM3v32ldeltaGiven  :1;
    unsigned  BSIM3v32lwrGiven   :1;
    unsigned  BSIM3v32ldwgGiven   :1;
    unsigned  BSIM3v32ldwbGiven   :1;
    unsigned  BSIM3v32lb0Given   :1;
    unsigned  BSIM3v32lb1Given   :1;
    unsigned  BSIM3v32lalpha0Given   :1;
    unsigned  BSIM3v32lalpha1Given   :1;
    unsigned  BSIM3v32lbeta0Given   :1;
    unsigned  BSIM3v32lvfbGiven   :1;

    /* CV model */
    unsigned  BSIM3v32lelmGiven  :1;
    unsigned  BSIM3v32lcgslGiven   :1;
    unsigned  BSIM3v32lcgdlGiven   :1;
    unsigned  BSIM3v32lckappaGiven   :1;
    unsigned  BSIM3v32lcfGiven   :1;
    unsigned  BSIM3v32lclcGiven   :1;
    unsigned  BSIM3v32lcleGiven   :1;
    unsigned  BSIM3v32lvfbcvGiven   :1;
    unsigned  BSIM3v32lnoffGiven   :1;
    unsigned  BSIM3v32lvoffcvGiven :1;
    unsigned  BSIM3v32lacdeGiven   :1;
    unsigned  BSIM3v32lmoinGiven   :1;

    /* Width dependence */
    unsigned  BSIM3v32wcdscGiven   :1;
    unsigned  BSIM3v32wcdscbGiven   :1;
    unsigned  BSIM3v32wcdscdGiven   :1;
    unsigned  BSIM3v32wcitGiven   :1;
    unsigned  BSIM3v32wnfactorGiven   :1;
    unsigned  BSIM3v32wxjGiven   :1;
    unsigned  BSIM3v32wvsatGiven   :1;
    unsigned  BSIM3v32watGiven   :1;
    unsigned  BSIM3v32wa0Given   :1;
    unsigned  BSIM3v32wagsGiven   :1;
    unsigned  BSIM3v32wa1Given   :1;
    unsigned  BSIM3v32wa2Given   :1;
    unsigned  BSIM3v32wketaGiven   :1;
    unsigned  BSIM3v32wnsubGiven   :1;
    unsigned  BSIM3v32wnpeakGiven   :1;
    unsigned  BSIM3v32wngateGiven   :1;
    unsigned  BSIM3v32wgamma1Given   :1;
    unsigned  BSIM3v32wgamma2Given   :1;
    unsigned  BSIM3v32wvbxGiven   :1;
    unsigned  BSIM3v32wvbmGiven   :1;
    unsigned  BSIM3v32wxtGiven   :1;
    unsigned  BSIM3v32wk1Given   :1;
    unsigned  BSIM3v32wkt1Given   :1;
    unsigned  BSIM3v32wkt1lGiven   :1;
    unsigned  BSIM3v32wkt2Given   :1;
    unsigned  BSIM3v32wk2Given   :1;
    unsigned  BSIM3v32wk3Given   :1;
    unsigned  BSIM3v32wk3bGiven   :1;
    unsigned  BSIM3v32ww0Given   :1;
    unsigned  BSIM3v32wnlxGiven   :1;
    unsigned  BSIM3v32wdvt0Given   :1;
    unsigned  BSIM3v32wdvt1Given   :1;
    unsigned  BSIM3v32wdvt2Given   :1;
    unsigned  BSIM3v32wdvt0wGiven   :1;
    unsigned  BSIM3v32wdvt1wGiven   :1;
    unsigned  BSIM3v32wdvt2wGiven   :1;
    unsigned  BSIM3v32wdroutGiven   :1;
    unsigned  BSIM3v32wdsubGiven   :1;
    unsigned  BSIM3v32wvth0Given   :1;
    unsigned  BSIM3v32wuaGiven   :1;
    unsigned  BSIM3v32wua1Given   :1;
    unsigned  BSIM3v32wubGiven   :1;
    unsigned  BSIM3v32wub1Given   :1;
    unsigned  BSIM3v32wucGiven   :1;
    unsigned  BSIM3v32wuc1Given   :1;
    unsigned  BSIM3v32wu0Given   :1;
    unsigned  BSIM3v32wuteGiven   :1;
    unsigned  BSIM3v32wvoffGiven   :1;
    unsigned  BSIM3v32wrdswGiven   :1;
    unsigned  BSIM3v32wprwgGiven   :1;
    unsigned  BSIM3v32wprwbGiven   :1;
    unsigned  BSIM3v32wprtGiven   :1;
    unsigned  BSIM3v32weta0Given   :1;
    unsigned  BSIM3v32wetabGiven   :1;
    unsigned  BSIM3v32wpclmGiven   :1;
    unsigned  BSIM3v32wpdibl1Given   :1;
    unsigned  BSIM3v32wpdibl2Given   :1;
    unsigned  BSIM3v32wpdiblbGiven   :1;
    unsigned  BSIM3v32wpscbe1Given   :1;
    unsigned  BSIM3v32wpscbe2Given   :1;
    unsigned  BSIM3v32wpvagGiven   :1;
    unsigned  BSIM3v32wdeltaGiven  :1;
    unsigned  BSIM3v32wwrGiven   :1;
    unsigned  BSIM3v32wdwgGiven   :1;
    unsigned  BSIM3v32wdwbGiven   :1;
    unsigned  BSIM3v32wb0Given   :1;
    unsigned  BSIM3v32wb1Given   :1;
    unsigned  BSIM3v32walpha0Given   :1;
    unsigned  BSIM3v32walpha1Given   :1;
    unsigned  BSIM3v32wbeta0Given   :1;
    unsigned  BSIM3v32wvfbGiven   :1;

    /* CV model */
    unsigned  BSIM3v32welmGiven  :1;
    unsigned  BSIM3v32wcgslGiven   :1;
    unsigned  BSIM3v32wcgdlGiven   :1;
    unsigned  BSIM3v32wckappaGiven   :1;
    unsigned  BSIM3v32wcfGiven   :1;
    unsigned  BSIM3v32wclcGiven   :1;
    unsigned  BSIM3v32wcleGiven   :1;
    unsigned  BSIM3v32wvfbcvGiven   :1;
    unsigned  BSIM3v32wnoffGiven   :1;
    unsigned  BSIM3v32wvoffcvGiven :1;
    unsigned  BSIM3v32wacdeGiven   :1;
    unsigned  BSIM3v32wmoinGiven   :1;

    /* Cross-term dependence */
    unsigned  BSIM3v32pcdscGiven   :1;
    unsigned  BSIM3v32pcdscbGiven   :1;
    unsigned  BSIM3v32pcdscdGiven   :1;
    unsigned  BSIM3v32pcitGiven   :1;
    unsigned  BSIM3v32pnfactorGiven   :1;
    unsigned  BSIM3v32pxjGiven   :1;
    unsigned  BSIM3v32pvsatGiven   :1;
    unsigned  BSIM3v32patGiven   :1;
    unsigned  BSIM3v32pa0Given   :1;
    unsigned  BSIM3v32pagsGiven   :1;
    unsigned  BSIM3v32pa1Given   :1;
    unsigned  BSIM3v32pa2Given   :1;
    unsigned  BSIM3v32pketaGiven   :1;
    unsigned  BSIM3v32pnsubGiven   :1;
    unsigned  BSIM3v32pnpeakGiven   :1;
    unsigned  BSIM3v32pngateGiven   :1;
    unsigned  BSIM3v32pgamma1Given   :1;
    unsigned  BSIM3v32pgamma2Given   :1;
    unsigned  BSIM3v32pvbxGiven   :1;
    unsigned  BSIM3v32pvbmGiven   :1;
    unsigned  BSIM3v32pxtGiven   :1;
    unsigned  BSIM3v32pk1Given   :1;
    unsigned  BSIM3v32pkt1Given   :1;
    unsigned  BSIM3v32pkt1lGiven   :1;
    unsigned  BSIM3v32pkt2Given   :1;
    unsigned  BSIM3v32pk2Given   :1;
    unsigned  BSIM3v32pk3Given   :1;
    unsigned  BSIM3v32pk3bGiven   :1;
    unsigned  BSIM3v32pw0Given   :1;
    unsigned  BSIM3v32pnlxGiven   :1;
    unsigned  BSIM3v32pdvt0Given   :1;
    unsigned  BSIM3v32pdvt1Given   :1;
    unsigned  BSIM3v32pdvt2Given   :1;
    unsigned  BSIM3v32pdvt0wGiven   :1;
    unsigned  BSIM3v32pdvt1wGiven   :1;
    unsigned  BSIM3v32pdvt2wGiven   :1;
    unsigned  BSIM3v32pdroutGiven   :1;
    unsigned  BSIM3v32pdsubGiven   :1;
    unsigned  BSIM3v32pvth0Given   :1;
    unsigned  BSIM3v32puaGiven   :1;
    unsigned  BSIM3v32pua1Given   :1;
    unsigned  BSIM3v32pubGiven   :1;
    unsigned  BSIM3v32pub1Given   :1;
    unsigned  BSIM3v32pucGiven   :1;
    unsigned  BSIM3v32puc1Given   :1;
    unsigned  BSIM3v32pu0Given   :1;
    unsigned  BSIM3v32puteGiven   :1;
    unsigned  BSIM3v32pvoffGiven   :1;
    unsigned  BSIM3v32prdswGiven   :1;
    unsigned  BSIM3v32pprwgGiven   :1;
    unsigned  BSIM3v32pprwbGiven   :1;
    unsigned  BSIM3v32pprtGiven   :1;
    unsigned  BSIM3v32peta0Given   :1;
    unsigned  BSIM3v32petabGiven   :1;
    unsigned  BSIM3v32ppclmGiven   :1;
    unsigned  BSIM3v32ppdibl1Given   :1;
    unsigned  BSIM3v32ppdibl2Given   :1;
    unsigned  BSIM3v32ppdiblbGiven   :1;
    unsigned  BSIM3v32ppscbe1Given   :1;
    unsigned  BSIM3v32ppscbe2Given   :1;
    unsigned  BSIM3v32ppvagGiven   :1;
    unsigned  BSIM3v32pdeltaGiven  :1;
    unsigned  BSIM3v32pwrGiven   :1;
    unsigned  BSIM3v32pdwgGiven   :1;
    unsigned  BSIM3v32pdwbGiven   :1;
    unsigned  BSIM3v32pb0Given   :1;
    unsigned  BSIM3v32pb1Given   :1;
    unsigned  BSIM3v32palpha0Given   :1;
    unsigned  BSIM3v32palpha1Given   :1;
    unsigned  BSIM3v32pbeta0Given   :1;
    unsigned  BSIM3v32pvfbGiven   :1;

    /* CV model */
    unsigned  BSIM3v32pelmGiven  :1;
    unsigned  BSIM3v32pcgslGiven   :1;
    unsigned  BSIM3v32pcgdlGiven   :1;
    unsigned  BSIM3v32pckappaGiven   :1;
    unsigned  BSIM3v32pcfGiven   :1;
    unsigned  BSIM3v32pclcGiven   :1;
    unsigned  BSIM3v32pcleGiven   :1;
    unsigned  BSIM3v32pvfbcvGiven   :1;
    unsigned  BSIM3v32pnoffGiven   :1;
    unsigned  BSIM3v32pvoffcvGiven :1;
    unsigned  BSIM3v32pacdeGiven   :1;
    unsigned  BSIM3v32pmoinGiven   :1;

    unsigned  BSIM3v32useFringeGiven   :1;

    unsigned  BSIM3v32tnomGiven   :1;
    unsigned  BSIM3v32cgsoGiven   :1;
    unsigned  BSIM3v32cgdoGiven   :1;
    unsigned  BSIM3v32cgboGiven   :1;
    unsigned  BSIM3v32xpartGiven   :1;
    unsigned  BSIM3v32sheetResistanceGiven   :1;
    unsigned  BSIM3v32jctSatCurDensityGiven   :1;
    unsigned  BSIM3v32jctSidewallSatCurDensityGiven   :1;
    unsigned  BSIM3v32bulkJctPotentialGiven   :1;
    unsigned  BSIM3v32bulkJctBotGradingCoeffGiven   :1;
    unsigned  BSIM3v32sidewallJctPotentialGiven   :1;
    unsigned  BSIM3v32GatesidewallJctPotentialGiven   :1;
    unsigned  BSIM3v32bulkJctSideGradingCoeffGiven   :1;
    unsigned  BSIM3v32unitAreaJctCapGiven   :1;
    unsigned  BSIM3v32unitLengthSidewallJctCapGiven   :1;
    unsigned  BSIM3v32bulkJctGateSideGradingCoeffGiven   :1;
    unsigned  BSIM3v32unitLengthGateSidewallJctCapGiven   :1;
    unsigned  BSIM3v32jctEmissionCoeffGiven :1;
    unsigned  BSIM3v32jctTempExponentGiven        :1;

    unsigned  BSIM3v32oxideTrapDensityAGiven  :1;
    unsigned  BSIM3v32oxideTrapDensityBGiven  :1;
    unsigned  BSIM3v32oxideTrapDensityCGiven  :1;
    unsigned  BSIM3v32emGiven  :1;
    unsigned  BSIM3v32efGiven  :1;
    unsigned  BSIM3v32afGiven  :1;
    unsigned  BSIM3v32kfGiven  :1;

    unsigned  BSIM3v32LintGiven   :1;
    unsigned  BSIM3v32LlGiven   :1;
    unsigned  BSIM3v32LlcGiven   :1;
    unsigned  BSIM3v32LlnGiven   :1;
    unsigned  BSIM3v32LwGiven   :1;
    unsigned  BSIM3v32LwcGiven   :1;
    unsigned  BSIM3v32LwnGiven   :1;
    unsigned  BSIM3v32LwlGiven   :1;
    unsigned  BSIM3v32LwlcGiven   :1;
    unsigned  BSIM3v32LminGiven   :1;
    unsigned  BSIM3v32LmaxGiven   :1;

    unsigned  BSIM3v32WintGiven   :1;
    unsigned  BSIM3v32WlGiven   :1;
    unsigned  BSIM3v32WlcGiven   :1;
    unsigned  BSIM3v32WlnGiven   :1;
    unsigned  BSIM3v32WwGiven   :1;
    unsigned  BSIM3v32WwcGiven   :1;
    unsigned  BSIM3v32WwnGiven   :1;
    unsigned  BSIM3v32WwlGiven   :1;
    unsigned  BSIM3v32WwlcGiven   :1;
    unsigned  BSIM3v32WminGiven   :1;
    unsigned  BSIM3v32WmaxGiven   :1;

    unsigned  BSIM3v32vgsMaxGiven  :1;
    unsigned  BSIM3v32vgdMaxGiven  :1;
    unsigned  BSIM3v32vgbMaxGiven  :1;
    unsigned  BSIM3v32vdsMaxGiven  :1;
    unsigned  BSIM3v32vbsMaxGiven  :1;
    unsigned  BSIM3v32vbdMaxGiven  :1;
    unsigned  BSIM3v32vgsrMaxGiven  :1;
    unsigned  BSIM3v32vgdrMaxGiven  :1;
    unsigned  BSIM3v32vgbrMaxGiven  :1;
    unsigned  BSIM3v32vbsrMaxGiven  :1;
    unsigned  BSIM3v32vbdrMaxGiven  :1;

} BSIM3v32model;


#ifndef NMOS
#define NMOS 1
#define PMOS -1
#endif /*NMOS*/


/* device parameters */
#define BSIM3v32_W 1
#define BSIM3v32_L 2
#define BSIM3v32_AS 3
#define BSIM3v32_AD 4
#define BSIM3v32_PS 5
#define BSIM3v32_PD 6
#define BSIM3v32_NRS 7
#define BSIM3v32_NRD 8
#define BSIM3v32_OFF 9
#define BSIM3v32_IC_VBS 10
#define BSIM3v32_IC_VDS 11
#define BSIM3v32_IC_VGS 12
#define BSIM3v32_IC 13
#define BSIM3v32_NQSMOD 14
#define BSIM3v32_M 15
#define BSIM3v32_DELVTO 16
#define BSIM3v32_MULU0 17
#define BSIM3v32_GEO 18

/* model parameters */
#define BSIM3v32_MOD_CAPMOD          100
#define BSIM3v32_MOD_ACMMOD          101
#define BSIM3v32_MOD_CALCACM         102
#define BSIM3v32_MOD_MOBMOD          103
#define BSIM3v32_MOD_NOIMOD          104
#define BSIM3v32_MOD_NQSMOD          105

#define BSIM3v32_MOD_TOX             106

#define BSIM3v32_MOD_CDSC            107
#define BSIM3v32_MOD_CDSCB           108
#define BSIM3v32_MOD_CIT             109
#define BSIM3v32_MOD_NFACTOR         110
#define BSIM3v32_MOD_XJ              111
#define BSIM3v32_MOD_VSAT            112
#define BSIM3v32_MOD_AT              113
#define BSIM3v32_MOD_A0              114
#define BSIM3v32_MOD_A1              115
#define BSIM3v32_MOD_A2              116
#define BSIM3v32_MOD_KETA            117
#define BSIM3v32_MOD_NSUB            118
#define BSIM3v32_MOD_NPEAK           119
#define BSIM3v32_MOD_NGATE           120
#define BSIM3v32_MOD_GAMMA1          121
#define BSIM3v32_MOD_GAMMA2          122
#define BSIM3v32_MOD_VBX             123
#define BSIM3v32_MOD_BINUNIT         124

#define BSIM3v32_MOD_VBM             125

#define BSIM3v32_MOD_XT              128
#define BSIM3v32_MOD_K1              129
#define BSIM3v32_MOD_KT1             130
#define BSIM3v32_MOD_KT1L            131
#define BSIM3v32_MOD_K2              132
#define BSIM3v32_MOD_KT2             133
#define BSIM3v32_MOD_K3              134
#define BSIM3v32_MOD_K3B             135
#define BSIM3v32_MOD_W0              136
#define BSIM3v32_MOD_NLX             137

#define BSIM3v32_MOD_DVT0            138
#define BSIM3v32_MOD_DVT1            139
#define BSIM3v32_MOD_DVT2            140

#define BSIM3v32_MOD_DVT0W           141
#define BSIM3v32_MOD_DVT1W           142
#define BSIM3v32_MOD_DVT2W           143

#define BSIM3v32_MOD_DROUT           144
#define BSIM3v32_MOD_DSUB            145
#define BSIM3v32_MOD_VTH0            146
#define BSIM3v32_MOD_UA              147
#define BSIM3v32_MOD_UA1             148
#define BSIM3v32_MOD_UB              149
#define BSIM3v32_MOD_UB1             150
#define BSIM3v32_MOD_UC              151
#define BSIM3v32_MOD_UC1             152
#define BSIM3v32_MOD_U0              153
#define BSIM3v32_MOD_UTE             154
#define BSIM3v32_MOD_VOFF            155
#define BSIM3v32_MOD_DELTA           156
#define BSIM3v32_MOD_RDSW            157
#define BSIM3v32_MOD_PRT             158
#define BSIM3v32_MOD_LDD             159
#define BSIM3v32_MOD_ETA             160
#define BSIM3v32_MOD_ETA0            161
#define BSIM3v32_MOD_ETAB            162
#define BSIM3v32_MOD_PCLM            163
#define BSIM3v32_MOD_PDIBL1          164
#define BSIM3v32_MOD_PDIBL2          165
#define BSIM3v32_MOD_PSCBE1          166
#define BSIM3v32_MOD_PSCBE2          167
#define BSIM3v32_MOD_PVAG            168
#define BSIM3v32_MOD_WR              169
#define BSIM3v32_MOD_DWG             170
#define BSIM3v32_MOD_DWB             171
#define BSIM3v32_MOD_B0              172
#define BSIM3v32_MOD_B1              173
#define BSIM3v32_MOD_ALPHA0          174
#define BSIM3v32_MOD_BETA0           175
#define BSIM3v32_MOD_PDIBLB          178

#define BSIM3v32_MOD_PRWG            179
#define BSIM3v32_MOD_PRWB            180

#define BSIM3v32_MOD_CDSCD           181
#define BSIM3v32_MOD_AGS             182

#define BSIM3v32_MOD_FRINGE          184
#define BSIM3v32_MOD_ELM             185
#define BSIM3v32_MOD_CGSL            186
#define BSIM3v32_MOD_CGDL            187
#define BSIM3v32_MOD_CKAPPA          188
#define BSIM3v32_MOD_CF              189
#define BSIM3v32_MOD_CLC             190
#define BSIM3v32_MOD_CLE             191
#define BSIM3v32_MOD_PARAMCHK        192
#define BSIM3v32_MOD_VERSION         193
#define BSIM3v32_MOD_VFBCV           194
#define BSIM3v32_MOD_ACDE            195
#define BSIM3v32_MOD_MOIN            196
#define BSIM3v32_MOD_NOFF            197
#define BSIM3v32_MOD_IJTH            198
#define BSIM3v32_MOD_ALPHA1          199
#define BSIM3v32_MOD_VFB             200
#define BSIM3v32_MOD_TOXM            201
#define BSIM3v32_MOD_TCJ             202
#define BSIM3v32_MOD_TCJSW           203
#define BSIM3v32_MOD_TCJSWG          204
#define BSIM3v32_MOD_TPB             205
#define BSIM3v32_MOD_TPBSW           206
#define BSIM3v32_MOD_TPBSWG          207
#define BSIM3v32_MOD_VOFFCV          208

/* Length dependence */
#define BSIM3v32_MOD_LCDSC            251
#define BSIM3v32_MOD_LCDSCB           252
#define BSIM3v32_MOD_LCIT             253
#define BSIM3v32_MOD_LNFACTOR         254
#define BSIM3v32_MOD_LXJ              255
#define BSIM3v32_MOD_LVSAT            256
#define BSIM3v32_MOD_LAT              257
#define BSIM3v32_MOD_LA0              258
#define BSIM3v32_MOD_LA1              259
#define BSIM3v32_MOD_LA2              260
#define BSIM3v32_MOD_LKETA            261
#define BSIM3v32_MOD_LNSUB            262
#define BSIM3v32_MOD_LNPEAK           263
#define BSIM3v32_MOD_LNGATE           265
#define BSIM3v32_MOD_LGAMMA1          266
#define BSIM3v32_MOD_LGAMMA2          267
#define BSIM3v32_MOD_LVBX             268

#define BSIM3v32_MOD_LVBM             270

#define BSIM3v32_MOD_LXT              272
#define BSIM3v32_MOD_LK1              275
#define BSIM3v32_MOD_LKT1             276
#define BSIM3v32_MOD_LKT1L            277
#define BSIM3v32_MOD_LK2              278
#define BSIM3v32_MOD_LKT2             279
#define BSIM3v32_MOD_LK3              280
#define BSIM3v32_MOD_LK3B             281
#define BSIM3v32_MOD_LW0              282
#define BSIM3v32_MOD_LNLX             283

#define BSIM3v32_MOD_LDVT0            284
#define BSIM3v32_MOD_LDVT1            285
#define BSIM3v32_MOD_LDVT2            286

#define BSIM3v32_MOD_LDVT0W           287
#define BSIM3v32_MOD_LDVT1W           288
#define BSIM3v32_MOD_LDVT2W           289

#define BSIM3v32_MOD_LDROUT           290
#define BSIM3v32_MOD_LDSUB            291
#define BSIM3v32_MOD_LVTH0            292
#define BSIM3v32_MOD_LUA              293
#define BSIM3v32_MOD_LUA1             294
#define BSIM3v32_MOD_LUB              295
#define BSIM3v32_MOD_LUB1             296
#define BSIM3v32_MOD_LUC              297
#define BSIM3v32_MOD_LUC1             298
#define BSIM3v32_MOD_LU0              299
#define BSIM3v32_MOD_LUTE             300
#define BSIM3v32_MOD_LVOFF            301
#define BSIM3v32_MOD_LDELTA           302
#define BSIM3v32_MOD_LRDSW            303
#define BSIM3v32_MOD_LPRT             304
#define BSIM3v32_MOD_LLDD             305
#define BSIM3v32_MOD_LETA             306
#define BSIM3v32_MOD_LETA0            307
#define BSIM3v32_MOD_LETAB            308
#define BSIM3v32_MOD_LPCLM            309
#define BSIM3v32_MOD_LPDIBL1          310
#define BSIM3v32_MOD_LPDIBL2          311
#define BSIM3v32_MOD_LPSCBE1          312
#define BSIM3v32_MOD_LPSCBE2          313
#define BSIM3v32_MOD_LPVAG            314
#define BSIM3v32_MOD_LWR              315
#define BSIM3v32_MOD_LDWG             316
#define BSIM3v32_MOD_LDWB             317
#define BSIM3v32_MOD_LB0              318
#define BSIM3v32_MOD_LB1              319
#define BSIM3v32_MOD_LALPHA0          320
#define BSIM3v32_MOD_LBETA0           321
#define BSIM3v32_MOD_LPDIBLB          324

#define BSIM3v32_MOD_LPRWG            325
#define BSIM3v32_MOD_LPRWB            326

#define BSIM3v32_MOD_LCDSCD           327
#define BSIM3v32_MOD_LAGS             328


#define BSIM3v32_MOD_LFRINGE          331
#define BSIM3v32_MOD_LELM             332
#define BSIM3v32_MOD_LCGSL            333
#define BSIM3v32_MOD_LCGDL            334
#define BSIM3v32_MOD_LCKAPPA          335
#define BSIM3v32_MOD_LCF              336
#define BSIM3v32_MOD_LCLC             337
#define BSIM3v32_MOD_LCLE             338
#define BSIM3v32_MOD_LVFBCV           339
#define BSIM3v32_MOD_LACDE            340
#define BSIM3v32_MOD_LMOIN            341
#define BSIM3v32_MOD_LNOFF            342
#define BSIM3v32_MOD_LALPHA1          344
#define BSIM3v32_MOD_LVFB             345
#define BSIM3v32_MOD_LVOFFCV          346

/* Width dependence */
#define BSIM3v32_MOD_WCDSC            381
#define BSIM3v32_MOD_WCDSCB           382
#define BSIM3v32_MOD_WCIT             383
#define BSIM3v32_MOD_WNFACTOR         384
#define BSIM3v32_MOD_WXJ              385
#define BSIM3v32_MOD_WVSAT            386
#define BSIM3v32_MOD_WAT              387
#define BSIM3v32_MOD_WA0              388
#define BSIM3v32_MOD_WA1              389
#define BSIM3v32_MOD_WA2              390
#define BSIM3v32_MOD_WKETA            391
#define BSIM3v32_MOD_WNSUB            392
#define BSIM3v32_MOD_WNPEAK           393
#define BSIM3v32_MOD_WNGATE           395
#define BSIM3v32_MOD_WGAMMA1          396
#define BSIM3v32_MOD_WGAMMA2          397
#define BSIM3v32_MOD_WVBX             398

#define BSIM3v32_MOD_WVBM             400

#define BSIM3v32_MOD_WXT              402
#define BSIM3v32_MOD_WK1              405
#define BSIM3v32_MOD_WKT1             406
#define BSIM3v32_MOD_WKT1L            407
#define BSIM3v32_MOD_WK2              408
#define BSIM3v32_MOD_WKT2             409
#define BSIM3v32_MOD_WK3              410
#define BSIM3v32_MOD_WK3B             411
#define BSIM3v32_MOD_WW0              412
#define BSIM3v32_MOD_WNLX             413

#define BSIM3v32_MOD_WDVT0            414
#define BSIM3v32_MOD_WDVT1            415
#define BSIM3v32_MOD_WDVT2            416

#define BSIM3v32_MOD_WDVT0W           417
#define BSIM3v32_MOD_WDVT1W           418
#define BSIM3v32_MOD_WDVT2W           419

#define BSIM3v32_MOD_WDROUT           420
#define BSIM3v32_MOD_WDSUB            421
#define BSIM3v32_MOD_WVTH0            422
#define BSIM3v32_MOD_WUA              423
#define BSIM3v32_MOD_WUA1             424
#define BSIM3v32_MOD_WUB              425
#define BSIM3v32_MOD_WUB1             426
#define BSIM3v32_MOD_WUC              427
#define BSIM3v32_MOD_WUC1             428
#define BSIM3v32_MOD_WU0              429
#define BSIM3v32_MOD_WUTE             430
#define BSIM3v32_MOD_WVOFF            431
#define BSIM3v32_MOD_WDELTA           432
#define BSIM3v32_MOD_WRDSW            433
#define BSIM3v32_MOD_WPRT             434
#define BSIM3v32_MOD_WLDD             435
#define BSIM3v32_MOD_WETA             436
#define BSIM3v32_MOD_WETA0            437
#define BSIM3v32_MOD_WETAB            438
#define BSIM3v32_MOD_WPCLM            439
#define BSIM3v32_MOD_WPDIBL1          440
#define BSIM3v32_MOD_WPDIBL2          441
#define BSIM3v32_MOD_WPSCBE1          442
#define BSIM3v32_MOD_WPSCBE2          443
#define BSIM3v32_MOD_WPVAG            444
#define BSIM3v32_MOD_WWR              445
#define BSIM3v32_MOD_WDWG             446
#define BSIM3v32_MOD_WDWB             447
#define BSIM3v32_MOD_WB0              448
#define BSIM3v32_MOD_WB1              449
#define BSIM3v32_MOD_WALPHA0          450
#define BSIM3v32_MOD_WBETA0           451
#define BSIM3v32_MOD_WPDIBLB          454

#define BSIM3v32_MOD_WPRWG            455
#define BSIM3v32_MOD_WPRWB            456

#define BSIM3v32_MOD_WCDSCD           457
#define BSIM3v32_MOD_WAGS             458


#define BSIM3v32_MOD_WFRINGE          461
#define BSIM3v32_MOD_WELM             462
#define BSIM3v32_MOD_WCGSL            463
#define BSIM3v32_MOD_WCGDL            464
#define BSIM3v32_MOD_WCKAPPA          465
#define BSIM3v32_MOD_WCF              466
#define BSIM3v32_MOD_WCLC             467
#define BSIM3v32_MOD_WCLE             468
#define BSIM3v32_MOD_WVFBCV           469
#define BSIM3v32_MOD_WACDE            470
#define BSIM3v32_MOD_WMOIN            471
#define BSIM3v32_MOD_WNOFF            472
#define BSIM3v32_MOD_WALPHA1          474
#define BSIM3v32_MOD_WVFB             475
#define BSIM3v32_MOD_WVOFFCV          476

/* Cross-term dependence */
#define BSIM3v32_MOD_PCDSC            511
#define BSIM3v32_MOD_PCDSCB           512
#define BSIM3v32_MOD_PCIT             513
#define BSIM3v32_MOD_PNFACTOR         514
#define BSIM3v32_MOD_PXJ              515
#define BSIM3v32_MOD_PVSAT            516
#define BSIM3v32_MOD_PAT              517
#define BSIM3v32_MOD_PA0              518
#define BSIM3v32_MOD_PA1              519
#define BSIM3v32_MOD_PA2              520
#define BSIM3v32_MOD_PKETA            521
#define BSIM3v32_MOD_PNSUB            522
#define BSIM3v32_MOD_PNPEAK           523
#define BSIM3v32_MOD_PNGATE           525
#define BSIM3v32_MOD_PGAMMA1          526
#define BSIM3v32_MOD_PGAMMA2          527
#define BSIM3v32_MOD_PVBX             528

#define BSIM3v32_MOD_PVBM             530

#define BSIM3v32_MOD_PXT              532
#define BSIM3v32_MOD_PK1              535
#define BSIM3v32_MOD_PKT1             536
#define BSIM3v32_MOD_PKT1L            537
#define BSIM3v32_MOD_PK2              538
#define BSIM3v32_MOD_PKT2             539
#define BSIM3v32_MOD_PK3              540
#define BSIM3v32_MOD_PK3B             541
#define BSIM3v32_MOD_PW0              542
#define BSIM3v32_MOD_PNLX             543

#define BSIM3v32_MOD_PDVT0            544
#define BSIM3v32_MOD_PDVT1            545
#define BSIM3v32_MOD_PDVT2            546

#define BSIM3v32_MOD_PDVT0W           547
#define BSIM3v32_MOD_PDVT1W           548
#define BSIM3v32_MOD_PDVT2W           549

#define BSIM3v32_MOD_PDROUT           550
#define BSIM3v32_MOD_PDSUB            551
#define BSIM3v32_MOD_PVTH0            552
#define BSIM3v32_MOD_PUA              553
#define BSIM3v32_MOD_PUA1             554
#define BSIM3v32_MOD_PUB              555
#define BSIM3v32_MOD_PUB1             556
#define BSIM3v32_MOD_PUC              557
#define BSIM3v32_MOD_PUC1             558
#define BSIM3v32_MOD_PU0              559
#define BSIM3v32_MOD_PUTE             560
#define BSIM3v32_MOD_PVOFF            561
#define BSIM3v32_MOD_PDELTA           562
#define BSIM3v32_MOD_PRDSW            563
#define BSIM3v32_MOD_PPRT             564
#define BSIM3v32_MOD_PLDD             565
#define BSIM3v32_MOD_PETA             566
#define BSIM3v32_MOD_PETA0            567
#define BSIM3v32_MOD_PETAB            568
#define BSIM3v32_MOD_PPCLM            569
#define BSIM3v32_MOD_PPDIBL1          570
#define BSIM3v32_MOD_PPDIBL2          571
#define BSIM3v32_MOD_PPSCBE1          572
#define BSIM3v32_MOD_PPSCBE2          573
#define BSIM3v32_MOD_PPVAG            574
#define BSIM3v32_MOD_PWR              575
#define BSIM3v32_MOD_PDWG             576
#define BSIM3v32_MOD_PDWB             577
#define BSIM3v32_MOD_PB0              578
#define BSIM3v32_MOD_PB1              579
#define BSIM3v32_MOD_PALPHA0          580
#define BSIM3v32_MOD_PBETA0           581
#define BSIM3v32_MOD_PPDIBLB          584

#define BSIM3v32_MOD_PPRWG            585
#define BSIM3v32_MOD_PPRWB            586

#define BSIM3v32_MOD_PCDSCD           587
#define BSIM3v32_MOD_PAGS             588

#define BSIM3v32_MOD_PFRINGE          591
#define BSIM3v32_MOD_PELM             592
#define BSIM3v32_MOD_PCGSL            593
#define BSIM3v32_MOD_PCGDL            594
#define BSIM3v32_MOD_PCKAPPA          595
#define BSIM3v32_MOD_PCF              596
#define BSIM3v32_MOD_PCLC             597
#define BSIM3v32_MOD_PCLE             598
#define BSIM3v32_MOD_PVFBCV           599
#define BSIM3v32_MOD_PACDE            600
#define BSIM3v32_MOD_PMOIN            601
#define BSIM3v32_MOD_PNOFF            602
#define BSIM3v32_MOD_PALPHA1          604
#define BSIM3v32_MOD_PVFB             605
#define BSIM3v32_MOD_PVOFFCV          606

#define BSIM3v32_MOD_TNOM             651
#define BSIM3v32_MOD_CGSO             652
#define BSIM3v32_MOD_CGDO             653
#define BSIM3v32_MOD_CGBO             654
#define BSIM3v32_MOD_XPART            655

#define BSIM3v32_MOD_RSH              656
#define BSIM3v32_MOD_JS               657
#define BSIM3v32_MOD_PB               658
#define BSIM3v32_MOD_MJ               659
#define BSIM3v32_MOD_PBSW             660
#define BSIM3v32_MOD_MJSW             661
#define BSIM3v32_MOD_CJ               662
#define BSIM3v32_MOD_CJSW             663
#define BSIM3v32_MOD_NMOS             664
#define BSIM3v32_MOD_PMOS             665

#define BSIM3v32_MOD_NOIA             666
#define BSIM3v32_MOD_NOIB             667
#define BSIM3v32_MOD_NOIC             668

#define BSIM3v32_MOD_LINT             669
#define BSIM3v32_MOD_LL               670
#define BSIM3v32_MOD_LLN              671
#define BSIM3v32_MOD_LW               672
#define BSIM3v32_MOD_LWN              673
#define BSIM3v32_MOD_LWL              674
#define BSIM3v32_MOD_LMIN             675
#define BSIM3v32_MOD_LMAX             676

#define BSIM3v32_MOD_WINT             677
#define BSIM3v32_MOD_WL               678
#define BSIM3v32_MOD_WLN              679
#define BSIM3v32_MOD_WW               680
#define BSIM3v32_MOD_WWN              681
#define BSIM3v32_MOD_WWL              682
#define BSIM3v32_MOD_WMIN             683
#define BSIM3v32_MOD_WMAX             684

#define BSIM3v32_MOD_DWC              685
#define BSIM3v32_MOD_DLC              686

#define BSIM3v32_MOD_EM               687
#define BSIM3v32_MOD_EF               688
#define BSIM3v32_MOD_AF               689
#define BSIM3v32_MOD_KF               690

#define BSIM3v32_MOD_NJ               691
#define BSIM3v32_MOD_XTI              692

#define BSIM3v32_MOD_PBSWG            693
#define BSIM3v32_MOD_MJSWG            694
#define BSIM3v32_MOD_CJSWG            695
#define BSIM3v32_MOD_JSW              696

#define BSIM3v32_MOD_LLC              697
#define BSIM3v32_MOD_LWC              698
#define BSIM3v32_MOD_LWLC             699

#define BSIM3v32_MOD_WLC              700
#define BSIM3v32_MOD_WWC              701
#define BSIM3v32_MOD_WWLC             702

/* ACM parameters */
#define BSIM3v32_MOD_XL               703
#define BSIM3v32_MOD_XW               704
#define BSIM3v32_MOD_HDIF             711
#define BSIM3v32_MOD_LDIF             712
#define BSIM3v32_MOD_LD               713
#define BSIM3v32_MOD_RD               714
#define BSIM3v32_MOD_RS               715
#define BSIM3v32_MOD_RDC              716
#define BSIM3v32_MOD_RSC              717
#define BSIM3v32_MOD_WMLT             718

#define BSIM3v32_MOD_LMLT             719

/* device questions */
#define BSIM3v32_DNODE                751
#define BSIM3v32_GNODE                752
#define BSIM3v32_SNODE                753
#define BSIM3v32_BNODE                754
#define BSIM3v32_DNODEPRIME           755
#define BSIM3v32_SNODEPRIME           756
#define BSIM3v32_VBD                  757
#define BSIM3v32_VBS                  758
#define BSIM3v32_VGS                  759
#define BSIM3v32_VDS                  760
#define BSIM3v32_CD                   761
#define BSIM3v32_CBS                  762
#define BSIM3v32_CBD                  763
#define BSIM3v32_GM                   764
#define BSIM3v32_GDS                  765
#define BSIM3v32_GMBS                 766
#define BSIM3v32_GBD                  767
#define BSIM3v32_GBS                  768
#define BSIM3v32_QB                   769
#define BSIM3v32_CQB                  770
#define BSIM3v32_QG                   771
#define BSIM3v32_CQG                  772
#define BSIM3v32_QD                   773
#define BSIM3v32_CQD                  774
#define BSIM3v32_CGG                  775
#define BSIM3v32_CGD                  776
#define BSIM3v32_CGS                  777
#define BSIM3v32_CBG                  778
#define BSIM3v32_CAPBD                779
#define BSIM3v32_CQBD                 780
#define BSIM3v32_CAPBS                781
#define BSIM3v32_CQBS                 782
#define BSIM3v32_CDG                  783
#define BSIM3v32_CDD                  784
#define BSIM3v32_CDS                  785
#define BSIM3v32_VON                  786
#define BSIM3v32_VDSAT                787
#define BSIM3v32_QBS                  788
#define BSIM3v32_QBD                  789
#define BSIM3v32_SOURCECONDUCT        790
#define BSIM3v32_DRAINCONDUCT         791
#define BSIM3v32_CBDB                 792
#define BSIM3v32_CBSB                 793

#define BSIM3v32_MOD_VGS_MAX          801
#define BSIM3v32_MOD_VGD_MAX          802
#define BSIM3v32_MOD_VGB_MAX          803
#define BSIM3v32_MOD_VDS_MAX          804
#define BSIM3v32_MOD_VBS_MAX          805
#define BSIM3v32_MOD_VBD_MAX          806
#define BSIM3v32_MOD_VGSR_MAX         807
#define BSIM3v32_MOD_VGDR_MAX         808
#define BSIM3v32_MOD_VGBR_MAX         809
#define BSIM3v32_MOD_VBSR_MAX         810
#define BSIM3v32_MOD_VBDR_MAX         811

#include "bsim3v32ext.h"

extern void BSIM3v32evaluate(double,double,double,BSIM3v32instance*,BSIM3v32model*,
        double*,double*,double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, CKTcircuit*);
extern int BSIM3v32debug(BSIM3v32model*, BSIM3v32instance*, CKTcircuit*, int);
extern int BSIM3v32checkModel(BSIM3v32model*, BSIM3v32instance*, CKTcircuit*);

#endif /*BSIM3v32*/
