/**********
Copyright 2004 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Author: 1997-1999 Weidong Liu.
Author: 2001- Xuemei Xi
**** OpenMP support for ngspice by Holger Vogt 06/28/2010 ****
File: bsim3def.h
**********/

#ifndef BSIM3
#define BSIM3

#include "ngspice/ifsim.h"
#include "ngspice/gendefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/noisedef.h"

typedef struct sBSIM3instance
{

    struct GENinstance gen;

#define BSIM3modPtr(inst) ((struct sBSIM3model *)((inst)->gen.GENmodPtr))
#define BSIM3nextInstance(inst) ((struct sBSIM3instance *)((inst)->gen.GENnextInstance))
#define BSIM3name gen.GENname
#define BSIM3states gen.GENstate

    const int BSIM3dNode;
    const int BSIM3gNode;
    const int BSIM3sNode;
    const int BSIM3bNode;
    int BSIM3dNodePrime;
    int BSIM3sNodePrime;
    int BSIM3qNode; /* MCJ */

    /* MCJ */
    double BSIM3ueff;
    double BSIM3thetavth;
    double BSIM3von;
    double BSIM3vdsat;
    double BSIM3cgdo;
    double BSIM3cgso;
    double BSIM3vjsm;
    double BSIM3IsEvjsm;
    double BSIM3vjdm;
    double BSIM3IsEvjdm;

    double BSIM3l;
    double BSIM3w;
    double BSIM3m;
    double BSIM3drainArea;
    double BSIM3sourceArea;
    double BSIM3drainSquares;
    double BSIM3sourceSquares;
    double BSIM3drainPerimeter;
    double BSIM3sourcePerimeter;
    double BSIM3sourceConductance;
    double BSIM3drainConductance;
    double BSIM3delvto;
    double BSIM3mulu0;
    double BSIM3vth0;
    double BSIM3vfb;
    double BSIM3vfbzb;
    double BSIM3u0temp;
    double BSIM3tconst;

    double BSIM3icVBS;
    double BSIM3icVDS;
    double BSIM3icVGS;
    int BSIM3off;
    int BSIM3mode;
    int BSIM3nqsMod;
    int BSIM3acnqsMod;
    int BSIM3geo;

    /* OP point */
    double BSIM3qinv;
    double BSIM3cd;
    double BSIM3cbs;
    double BSIM3cbd;
    double BSIM3csub;
    double BSIM3gm;
    double BSIM3gds;
    double BSIM3gmbs;
    double BSIM3gbd;
    double BSIM3gbs;

    double BSIM3gbbs;
    double BSIM3gbgs;
    double BSIM3gbds;

    double BSIM3cggb;
    double BSIM3cgdb;
    double BSIM3cgsb;
    double BSIM3cbgb;
    double BSIM3cbdb;
    double BSIM3cbsb;
    double BSIM3cdgb;
    double BSIM3cddb;
    double BSIM3cdsb;
    double BSIM3capbd;
    double BSIM3capbs;

    double BSIM3cqgb;
    double BSIM3cqdb;
    double BSIM3cqsb;
    double BSIM3cqbb;

    double BSIM3qgate;
    double BSIM3qbulk;
    double BSIM3qdrn;

    double BSIM3gtau;
    double BSIM3gtg;
    double BSIM3gtd;
    double BSIM3gts;
    double BSIM3gtb;
    double BSIM3rds;  /* Noise bugfix */
    double BSIM3Vgsteff;
    double BSIM3Vdseff;
    double BSIM3Abulk;
    double BSIM3AbovVgst2Vtm;
    double BSIM3taunet;

    struct bsim3SizeDependParam  *pParam;

    unsigned BSIM3lGiven :1;
    unsigned BSIM3wGiven :1;
    unsigned BSIM3mGiven :1;
    unsigned BSIM3drainAreaGiven :1;
    unsigned BSIM3sourceAreaGiven    :1;
    unsigned BSIM3drainSquaresGiven  :1;
    unsigned BSIM3sourceSquaresGiven :1;
    unsigned BSIM3drainPerimeterGiven    :1;
    unsigned BSIM3sourcePerimeterGiven   :1;
    unsigned BSIM3delvtoGiven   :1;
    unsigned BSIM3mulu0Given   :1;
    unsigned BSIM3dNodePrimeSet  :1;
    unsigned BSIM3sNodePrimeSet  :1;
    unsigned BSIM3icVBSGiven :1;
    unsigned BSIM3icVDSGiven :1;
    unsigned BSIM3icVGSGiven :1;
    unsigned BSIM3nqsModGiven :1;
    unsigned BSIM3acnqsModGiven :1;
    unsigned BSIM3geoGiven :1;

    double *BSIM3DdPtr;
    double *BSIM3GgPtr;
    double *BSIM3SsPtr;
    double *BSIM3BbPtr;
    double *BSIM3DPdpPtr;
    double *BSIM3SPspPtr;
    double *BSIM3DdpPtr;
    double *BSIM3GbPtr;
    double *BSIM3GdpPtr;
    double *BSIM3GspPtr;
    double *BSIM3SspPtr;
    double *BSIM3BdpPtr;
    double *BSIM3BspPtr;
    double *BSIM3DPspPtr;
    double *BSIM3DPdPtr;
    double *BSIM3BgPtr;
    double *BSIM3DPgPtr;
    double *BSIM3SPgPtr;
    double *BSIM3SPsPtr;
    double *BSIM3DPbPtr;
    double *BSIM3SPbPtr;
    double *BSIM3SPdpPtr;

    double *BSIM3QqPtr;
    double *BSIM3QdpPtr;
    double *BSIM3QgPtr;
    double *BSIM3QspPtr;
    double *BSIM3QbPtr;
    double *BSIM3DPqPtr;
    double *BSIM3GqPtr;
    double *BSIM3SPqPtr;
    double *BSIM3BqPtr;

#ifdef USE_OMP
    /* per instance storage of results, to update matrix at a later stge */
    double BSIM3rhsG;
    double BSIM3rhsB;
    double BSIM3rhsD;
    double BSIM3rhsS;
    double BSIM3rhsQ;

    double BSIM3DdPt;
    double BSIM3GgPt;
    double BSIM3SsPt;
    double BSIM3BbPt;
    double BSIM3DPdpPt;
    double BSIM3SPspPt;
    double BSIM3DdpPt;
    double BSIM3GbPt;
    double BSIM3GdpPt;
    double BSIM3GspPt;
    double BSIM3SspPt;
    double BSIM3BdpPt;
    double BSIM3BspPt;
    double BSIM3DPspPt;
    double BSIM3DPdPt;
    double BSIM3BgPt;
    double BSIM3DPgPt;
    double BSIM3SPgPt;
    double BSIM3SPsPt;
    double BSIM3DPbPt;
    double BSIM3SPbPt;
    double BSIM3SPdpPt;

    double BSIM3QqPt;
    double BSIM3QdpPt;
    double BSIM3QgPt;
    double BSIM3QspPt;
    double BSIM3QbPt;
    double BSIM3DPqPt;
    double BSIM3GqPt;
    double BSIM3SPqPt;
    double BSIM3BqPt;
#endif

#define BSIM3vbd BSIM3states+ 0
#define BSIM3vbs BSIM3states+ 1
#define BSIM3vgs BSIM3states+ 2
#define BSIM3vds BSIM3states+ 3

#define BSIM3qb BSIM3states+ 4
#define BSIM3cqb BSIM3states+ 5
#define BSIM3qg BSIM3states+ 6
#define BSIM3cqg BSIM3states+ 7
#define BSIM3qd BSIM3states+ 8
#define BSIM3cqd BSIM3states+ 9

#define BSIM3qbs  BSIM3states+ 10
#define BSIM3qbd  BSIM3states+ 11

#define BSIM3qcheq BSIM3states+ 12
#define BSIM3cqcheq BSIM3states+ 13
#define BSIM3qcdump BSIM3states+ 14
#define BSIM3cqcdump BSIM3states+ 15

#define BSIM3qdef BSIM3states+ 16

#define BSIM3numStates 17


/* indices to the array of BSIM3 NOISE SOURCES */

#define BSIM3RDNOIZ       0
#define BSIM3RSNOIZ       1
#define BSIM3IDNOIZ       2
#define BSIM3FLNOIZ       3
#define BSIM3TOTNOIZ      4

#define BSIM3NSRCS        5  /* the number of BSIM3 MOSFET noise sources */

#ifndef NONOISE
    double BSIM3nVar[NSTATVARS][BSIM3NSRCS];
#else /* NONOISE */
        double **BSIM3nVar;
#endif /* NONOISE */

} BSIM3instance ;

struct bsim3SizeDependParam
{
    double Width;
    double Length;

    double BSIM3cdsc;
    double BSIM3cdscb;
    double BSIM3cdscd;
    double BSIM3cit;
    double BSIM3nfactor;
    double BSIM3xj;
    double BSIM3vsat;
    double BSIM3at;
    double BSIM3a0;
    double BSIM3ags;
    double BSIM3a1;
    double BSIM3a2;
    double BSIM3keta;
    double BSIM3nsub;
    double BSIM3npeak;
    double BSIM3ngate;
    double BSIM3gamma1;
    double BSIM3gamma2;
    double BSIM3vbx;
    double BSIM3vbi;
    double BSIM3vbm;
    double BSIM3vbsc;
    double BSIM3xt;
    double BSIM3phi;
    double BSIM3litl;
    double BSIM3k1;
    double BSIM3kt1;
    double BSIM3kt1l;
    double BSIM3kt2;
    double BSIM3k2;
    double BSIM3k3;
    double BSIM3k3b;
    double BSIM3w0;
    double BSIM3nlx;
    double BSIM3dvt0;
    double BSIM3dvt1;
    double BSIM3dvt2;
    double BSIM3dvt0w;
    double BSIM3dvt1w;
    double BSIM3dvt2w;
    double BSIM3drout;
    double BSIM3dsub;
    double BSIM3vth0;
    double BSIM3ua;
    double BSIM3ua1;
    double BSIM3ub;
    double BSIM3ub1;
    double BSIM3uc;
    double BSIM3uc1;
    double BSIM3u0;
    double BSIM3ute;
    double BSIM3voff;
    double BSIM3vfb;
    double BSIM3delta;
    double BSIM3rdsw;
    double BSIM3rds0;
    double BSIM3prwg;
    double BSIM3prwb;
    double BSIM3prt;
    double BSIM3eta0;
    double BSIM3etab;
    double BSIM3pclm;
    double BSIM3pdibl1;
    double BSIM3pdibl2;
    double BSIM3pdiblb;
    double BSIM3pscbe1;
    double BSIM3pscbe2;
    double BSIM3pvag;
    double BSIM3wr;
    double BSIM3dwg;
    double BSIM3dwb;
    double BSIM3b0;
    double BSIM3b1;
    double BSIM3alpha0;
    double BSIM3alpha1;
    double BSIM3beta0;


    /* CV model */
    double BSIM3elm;
    double BSIM3cgsl;
    double BSIM3cgdl;
    double BSIM3ckappa;
    double BSIM3cf;
    double BSIM3clc;
    double BSIM3cle;
    double BSIM3vfbcv;
    double BSIM3noff;
    double BSIM3voffcv;
    double BSIM3acde;
    double BSIM3moin;


/* Pre-calculated constants */

    double BSIM3dw;
    double BSIM3dl;
    double BSIM3leff;
    double BSIM3weff;

    double BSIM3dwc;
    double BSIM3dlc;
    double BSIM3leffCV;
    double BSIM3weffCV;
    double BSIM3abulkCVfactor;
    double BSIM3cgso;
    double BSIM3cgdo;
    double BSIM3cgbo;
    double BSIM3tconst;

    double BSIM3u0temp;
    double BSIM3vsattemp;
    double BSIM3sqrtPhi;
    double BSIM3phis3;
    double BSIM3Xdep0;
    double BSIM3sqrtXdep0;
    double BSIM3theta0vb0;
    double BSIM3thetaRout;

    double BSIM3cof1;
    double BSIM3cof2;
    double BSIM3cof3;
    double BSIM3cof4;
    double BSIM3cdep0;
    double BSIM3vfbzb;
    double BSIM3ldeb;
    double BSIM3k1ox;
    double BSIM3k2ox;

    struct bsim3SizeDependParam  *pNext;
};


typedef struct sBSIM3model
{

    struct GENmodel gen;

#define BSIM3modType gen.GENmodType
#define BSIM3nextModel(inst) ((struct sBSIM3model *)((inst)->gen.GENnextModel))
#define BSIM3instances(inst) ((BSIM3instance *)((inst)->gen.GENinstances))
#define BSIM3modName gen.GENmodName

    int BSIM3type;

    int    BSIM3mobMod;
    int    BSIM3capMod;
    int    BSIM3acmMod;
    int    BSIM3calcacm;
    int    BSIM3noiMod;
    int    BSIM3nqsMod;
    int    BSIM3acnqsMod;
    int    BSIM3binUnit;
    int    BSIM3paramChk;
    char   *BSIM3version;
    double BSIM3tox;
    double BSIM3toxm;
    double BSIM3cdsc;
    double BSIM3cdscb;
    double BSIM3cdscd;
    double BSIM3cit;
    double BSIM3nfactor;
    double BSIM3xj;
    double BSIM3vsat;
    double BSIM3at;
    double BSIM3a0;
    double BSIM3ags;
    double BSIM3a1;
    double BSIM3a2;
    double BSIM3keta;
    double BSIM3nsub;
    double BSIM3npeak;
    double BSIM3ngate;
    double BSIM3gamma1;
    double BSIM3gamma2;
    double BSIM3vbx;
    double BSIM3vbm;
    double BSIM3xt;
    double BSIM3k1;
    double BSIM3kt1;
    double BSIM3kt1l;
    double BSIM3kt2;
    double BSIM3k2;
    double BSIM3k3;
    double BSIM3k3b;
    double BSIM3w0;
    double BSIM3nlx;
    double BSIM3dvt0;
    double BSIM3dvt1;
    double BSIM3dvt2;
    double BSIM3dvt0w;
    double BSIM3dvt1w;
    double BSIM3dvt2w;
    double BSIM3drout;
    double BSIM3dsub;
    double BSIM3vth0;
    double BSIM3ua;
    double BSIM3ua1;
    double BSIM3ub;
    double BSIM3ub1;
    double BSIM3uc;
    double BSIM3uc1;
    double BSIM3u0;
    double BSIM3ute;
    double BSIM3voff;
    double BSIM3delta;
    double BSIM3rdsw;
    double BSIM3prwg;
    double BSIM3prwb;
    double BSIM3prt;
    double BSIM3eta0;
    double BSIM3etab;
    double BSIM3pclm;
    double BSIM3pdibl1;
    double BSIM3pdibl2;
    double BSIM3pdiblb;
    double BSIM3pscbe1;
    double BSIM3pscbe2;
    double BSIM3pvag;
    double BSIM3wr;
    double BSIM3dwg;
    double BSIM3dwb;
    double BSIM3b0;
    double BSIM3b1;
    double BSIM3alpha0;
    double BSIM3alpha1;
    double BSIM3beta0;
    double BSIM3ijth;
    double BSIM3vfb;

    /* CV model */
    double BSIM3elm;
    double BSIM3cgsl;
    double BSIM3cgdl;
    double BSIM3ckappa;
    double BSIM3cf;
    double BSIM3vfbcv;
    double BSIM3clc;
    double BSIM3cle;
    double BSIM3dwc;
    double BSIM3dlc;
    double BSIM3noff;
    double BSIM3voffcv;
    double BSIM3acde;
    double BSIM3moin;
    double BSIM3tcj;
    double BSIM3tcjsw;
    double BSIM3tcjswg;
    double BSIM3tpb;
    double BSIM3tpbsw;
    double BSIM3tpbswg;

    /* ACM model */
    double BSIM3xl;
    double BSIM3xw;
    double BSIM3hdif;
    double BSIM3ldif;
    double BSIM3ld;
    double BSIM3rd;
    double BSIM3rs;
    double BSIM3rdc;
    double BSIM3rsc;
    double BSIM3wmlt;

    /* Length Dependence */
    double BSIM3lcdsc;
    double BSIM3lcdscb;
    double BSIM3lcdscd;
    double BSIM3lcit;
    double BSIM3lnfactor;
    double BSIM3lxj;
    double BSIM3lvsat;
    double BSIM3lat;
    double BSIM3la0;
    double BSIM3lags;
    double BSIM3la1;
    double BSIM3la2;
    double BSIM3lketa;
    double BSIM3lnsub;
    double BSIM3lnpeak;
    double BSIM3lngate;
    double BSIM3lgamma1;
    double BSIM3lgamma2;
    double BSIM3lvbx;
    double BSIM3lvbm;
    double BSIM3lxt;
    double BSIM3lk1;
    double BSIM3lkt1;
    double BSIM3lkt1l;
    double BSIM3lkt2;
    double BSIM3lk2;
    double BSIM3lk3;
    double BSIM3lk3b;
    double BSIM3lw0;
    double BSIM3lnlx;
    double BSIM3ldvt0;
    double BSIM3ldvt1;
    double BSIM3ldvt2;
    double BSIM3ldvt0w;
    double BSIM3ldvt1w;
    double BSIM3ldvt2w;
    double BSIM3ldrout;
    double BSIM3ldsub;
    double BSIM3lvth0;
    double BSIM3lua;
    double BSIM3lua1;
    double BSIM3lub;
    double BSIM3lub1;
    double BSIM3luc;
    double BSIM3luc1;
    double BSIM3lu0;
    double BSIM3lute;
    double BSIM3lvoff;
    double BSIM3ldelta;
    double BSIM3lrdsw;
    double BSIM3lprwg;
    double BSIM3lprwb;
    double BSIM3lprt;
    double BSIM3leta0;
    double BSIM3letab;
    double BSIM3lpclm;
    double BSIM3lpdibl1;
    double BSIM3lpdibl2;
    double BSIM3lpdiblb;
    double BSIM3lpscbe1;
    double BSIM3lpscbe2;
    double BSIM3lpvag;
    double BSIM3lwr;
    double BSIM3ldwg;
    double BSIM3ldwb;
    double BSIM3lb0;
    double BSIM3lb1;
    double BSIM3lalpha0;
    double BSIM3lalpha1;
    double BSIM3lbeta0;
    double BSIM3lvfb;

    /* CV model */
    double BSIM3lelm;
    double BSIM3lcgsl;
    double BSIM3lcgdl;
    double BSIM3lckappa;
    double BSIM3lcf;
    double BSIM3lclc;
    double BSIM3lcle;
    double BSIM3lvfbcv;
    double BSIM3lnoff;
    double BSIM3lvoffcv;
    double BSIM3lacde;
    double BSIM3lmoin;

    /* Width Dependence */
    double BSIM3wcdsc;
    double BSIM3wcdscb;
    double BSIM3wcdscd;
    double BSIM3wcit;
    double BSIM3wnfactor;
    double BSIM3wxj;
    double BSIM3wvsat;
    double BSIM3wat;
    double BSIM3wa0;
    double BSIM3wags;
    double BSIM3wa1;
    double BSIM3wa2;
    double BSIM3wketa;
    double BSIM3wnsub;
    double BSIM3wnpeak;
    double BSIM3wngate;
    double BSIM3wgamma1;
    double BSIM3wgamma2;
    double BSIM3wvbx;
    double BSIM3wvbm;
    double BSIM3wxt;
    double BSIM3wk1;
    double BSIM3wkt1;
    double BSIM3wkt1l;
    double BSIM3wkt2;
    double BSIM3wk2;
    double BSIM3wk3;
    double BSIM3wk3b;
    double BSIM3ww0;
    double BSIM3wnlx;
    double BSIM3wdvt0;
    double BSIM3wdvt1;
    double BSIM3wdvt2;
    double BSIM3wdvt0w;
    double BSIM3wdvt1w;
    double BSIM3wdvt2w;
    double BSIM3wdrout;
    double BSIM3wdsub;
    double BSIM3wvth0;
    double BSIM3wua;
    double BSIM3wua1;
    double BSIM3wub;
    double BSIM3wub1;
    double BSIM3wuc;
    double BSIM3wuc1;
    double BSIM3wu0;
    double BSIM3wute;
    double BSIM3wvoff;
    double BSIM3wdelta;
    double BSIM3wrdsw;
    double BSIM3wprwg;
    double BSIM3wprwb;
    double BSIM3wprt;
    double BSIM3weta0;
    double BSIM3wetab;
    double BSIM3wpclm;
    double BSIM3wpdibl1;
    double BSIM3wpdibl2;
    double BSIM3wpdiblb;
    double BSIM3wpscbe1;
    double BSIM3wpscbe2;
    double BSIM3wpvag;
    double BSIM3wwr;
    double BSIM3wdwg;
    double BSIM3wdwb;
    double BSIM3wb0;
    double BSIM3wb1;
    double BSIM3walpha0;
    double BSIM3walpha1;
    double BSIM3wbeta0;
    double BSIM3wvfb;

    /* CV model */
    double BSIM3welm;
    double BSIM3wcgsl;
    double BSIM3wcgdl;
    double BSIM3wckappa;
    double BSIM3wcf;
    double BSIM3wclc;
    double BSIM3wcle;
    double BSIM3wvfbcv;
    double BSIM3wnoff;
    double BSIM3wvoffcv;
    double BSIM3wacde;
    double BSIM3wmoin;

    /* Cross-term Dependence */
    double BSIM3pcdsc;
    double BSIM3pcdscb;
    double BSIM3pcdscd;
    double BSIM3pcit;
    double BSIM3pnfactor;
    double BSIM3pxj;
    double BSIM3pvsat;
    double BSIM3pat;
    double BSIM3pa0;
    double BSIM3pags;
    double BSIM3pa1;
    double BSIM3pa2;
    double BSIM3pketa;
    double BSIM3pnsub;
    double BSIM3pnpeak;
    double BSIM3pngate;
    double BSIM3pgamma1;
    double BSIM3pgamma2;
    double BSIM3pvbx;
    double BSIM3pvbm;
    double BSIM3pxt;
    double BSIM3pk1;
    double BSIM3pkt1;
    double BSIM3pkt1l;
    double BSIM3pkt2;
    double BSIM3pk2;
    double BSIM3pk3;
    double BSIM3pk3b;
    double BSIM3pw0;
    double BSIM3pnlx;
    double BSIM3pdvt0;
    double BSIM3pdvt1;
    double BSIM3pdvt2;
    double BSIM3pdvt0w;
    double BSIM3pdvt1w;
    double BSIM3pdvt2w;
    double BSIM3pdrout;
    double BSIM3pdsub;
    double BSIM3pvth0;
    double BSIM3pua;
    double BSIM3pua1;
    double BSIM3pub;
    double BSIM3pub1;
    double BSIM3puc;
    double BSIM3puc1;
    double BSIM3pu0;
    double BSIM3pute;
    double BSIM3pvoff;
    double BSIM3pdelta;
    double BSIM3prdsw;
    double BSIM3pprwg;
    double BSIM3pprwb;
    double BSIM3pprt;
    double BSIM3peta0;
    double BSIM3petab;
    double BSIM3ppclm;
    double BSIM3ppdibl1;
    double BSIM3ppdibl2;
    double BSIM3ppdiblb;
    double BSIM3ppscbe1;
    double BSIM3ppscbe2;
    double BSIM3ppvag;
    double BSIM3pwr;
    double BSIM3pdwg;
    double BSIM3pdwb;
    double BSIM3pb0;
    double BSIM3pb1;
    double BSIM3palpha0;
    double BSIM3palpha1;
    double BSIM3pbeta0;
    double BSIM3pvfb;

    /* CV model */
    double BSIM3pelm;
    double BSIM3pcgsl;
    double BSIM3pcgdl;
    double BSIM3pckappa;
    double BSIM3pcf;
    double BSIM3pclc;
    double BSIM3pcle;
    double BSIM3pvfbcv;
    double BSIM3pnoff;
    double BSIM3pvoffcv;
    double BSIM3pacde;
    double BSIM3pmoin;

    double BSIM3tnom;
    double BSIM3cgso;
    double BSIM3cgdo;
    double BSIM3cgbo;
    double BSIM3xpart;
    double BSIM3cFringOut;
    double BSIM3cFringMax;

    double BSIM3sheetResistance;
    double BSIM3jctSatCurDensity;
    double BSIM3jctSidewallSatCurDensity;
    double BSIM3bulkJctPotential;
    double BSIM3bulkJctBotGradingCoeff;
    double BSIM3bulkJctSideGradingCoeff;
    double BSIM3bulkJctGateSideGradingCoeff;
    double BSIM3sidewallJctPotential;
    double BSIM3GatesidewallJctPotential;
    double BSIM3unitAreaJctCap;
    double BSIM3unitLengthSidewallJctCap;
    double BSIM3unitLengthGateSidewallJctCap;
    double BSIM3jctEmissionCoeff;
    double BSIM3jctTempExponent;

    double BSIM3Lint;
    double BSIM3Ll;
    double BSIM3Llc;
    double BSIM3Lln;
    double BSIM3Lw;
    double BSIM3Lwc;
    double BSIM3Lwn;
    double BSIM3Lwl;
    double BSIM3Lwlc;
    double BSIM3Lmin;
    double BSIM3Lmax;

    double BSIM3Wint;
    double BSIM3Wl;
    double BSIM3Wlc;
    double BSIM3Wln;
    double BSIM3Ww;
    double BSIM3Wwc;
    double BSIM3Wwn;
    double BSIM3Wwl;
    double BSIM3Wwlc;
    double BSIM3Wmin;
    double BSIM3Wmax;


/* Pre-calculated constants */
    /* MCJ: move to size-dependent param. */
    double BSIM3vtm;
    double BSIM3cox;
    double BSIM3cof1;
    double BSIM3cof2;
    double BSIM3cof3;
    double BSIM3cof4;
    double BSIM3vcrit;
    double BSIM3factor1;
    double BSIM3PhiB;
    double BSIM3PhiBSW;
    double BSIM3PhiBSWG;
    double BSIM3jctTempSatCurDensity;
    double BSIM3jctSidewallTempSatCurDensity;
    double BSIM3unitAreaTempJctCap;
    double BSIM3unitLengthSidewallTempJctCap;
    double BSIM3unitLengthGateSidewallTempJctCap;

    double BSIM3oxideTrapDensityA;
    double BSIM3oxideTrapDensityB;
    double BSIM3oxideTrapDensityC;
    double BSIM3em;
    double BSIM3ef;
    double BSIM3af;
    double BSIM3kf;
    double BSIM3lintnoi;  /* lint offset for noise calculation  */

    double BSIM3vgsMax;
    double BSIM3vgdMax;
    double BSIM3vgbMax;
    double BSIM3vdsMax;
    double BSIM3vbsMax;
    double BSIM3vbdMax;
    double BSIM3vgsrMax;
    double BSIM3vgdrMax;
    double BSIM3vgbrMax;
    double BSIM3vbsrMax;
    double BSIM3vbdrMax;

    struct bsim3SizeDependParam *pSizeDependParamKnot;


#ifdef USE_OMP
    int BSIM3InstCount;
    struct sBSIM3instance **BSIM3InstanceArray;
#endif

    /* Flags */
    unsigned  BSIM3mobModGiven :1;
    unsigned  BSIM3binUnitGiven :1;
    unsigned  BSIM3capModGiven :1;
    unsigned  BSIM3acmModGiven :1;
    unsigned  BSIM3calcacmGiven :1;
    unsigned  BSIM3paramChkGiven :1;
    unsigned  BSIM3noiModGiven :1;
    unsigned  BSIM3nqsModGiven :1;
    unsigned  BSIM3acnqsModGiven :1;
    unsigned  BSIM3typeGiven   :1;
    unsigned  BSIM3toxGiven   :1;
    unsigned  BSIM3versionGiven   :1;
    unsigned  BSIM3toxmGiven   :1;
    unsigned  BSIM3cdscGiven   :1;
    unsigned  BSIM3cdscbGiven   :1;
    unsigned  BSIM3cdscdGiven   :1;
    unsigned  BSIM3citGiven   :1;
    unsigned  BSIM3nfactorGiven   :1;
    unsigned  BSIM3xjGiven   :1;
    unsigned  BSIM3vsatGiven   :1;
    unsigned  BSIM3atGiven   :1;
    unsigned  BSIM3a0Given   :1;
    unsigned  BSIM3agsGiven   :1;
    unsigned  BSIM3a1Given   :1;
    unsigned  BSIM3a2Given   :1;
    unsigned  BSIM3ketaGiven   :1;
    unsigned  BSIM3nsubGiven   :1;
    unsigned  BSIM3npeakGiven   :1;
    unsigned  BSIM3ngateGiven   :1;
    unsigned  BSIM3gamma1Given   :1;
    unsigned  BSIM3gamma2Given   :1;
    unsigned  BSIM3vbxGiven   :1;
    unsigned  BSIM3vbmGiven   :1;
    unsigned  BSIM3xtGiven   :1;
    unsigned  BSIM3k1Given   :1;
    unsigned  BSIM3kt1Given   :1;
    unsigned  BSIM3kt1lGiven   :1;
    unsigned  BSIM3kt2Given   :1;
    unsigned  BSIM3k2Given   :1;
    unsigned  BSIM3k3Given   :1;
    unsigned  BSIM3k3bGiven   :1;
    unsigned  BSIM3w0Given   :1;
    unsigned  BSIM3nlxGiven   :1;
    unsigned  BSIM3dvt0Given   :1;
    unsigned  BSIM3dvt1Given   :1;
    unsigned  BSIM3dvt2Given   :1;
    unsigned  BSIM3dvt0wGiven   :1;
    unsigned  BSIM3dvt1wGiven   :1;
    unsigned  BSIM3dvt2wGiven   :1;
    unsigned  BSIM3droutGiven   :1;
    unsigned  BSIM3dsubGiven   :1;
    unsigned  BSIM3vth0Given   :1;
    unsigned  BSIM3uaGiven   :1;
    unsigned  BSIM3ua1Given   :1;
    unsigned  BSIM3ubGiven   :1;
    unsigned  BSIM3ub1Given   :1;
    unsigned  BSIM3ucGiven   :1;
    unsigned  BSIM3uc1Given   :1;
    unsigned  BSIM3u0Given   :1;
    unsigned  BSIM3uteGiven   :1;
    unsigned  BSIM3voffGiven   :1;
    unsigned  BSIM3rdswGiven   :1;
    unsigned  BSIM3prwgGiven   :1;
    unsigned  BSIM3prwbGiven   :1;
    unsigned  BSIM3prtGiven   :1;
    unsigned  BSIM3eta0Given   :1;
    unsigned  BSIM3etabGiven   :1;
    unsigned  BSIM3pclmGiven   :1;
    unsigned  BSIM3pdibl1Given   :1;
    unsigned  BSIM3pdibl2Given   :1;
    unsigned  BSIM3pdiblbGiven   :1;
    unsigned  BSIM3pscbe1Given   :1;
    unsigned  BSIM3pscbe2Given   :1;
    unsigned  BSIM3pvagGiven   :1;
    unsigned  BSIM3deltaGiven  :1;
    unsigned  BSIM3wrGiven   :1;
    unsigned  BSIM3dwgGiven   :1;
    unsigned  BSIM3dwbGiven   :1;
    unsigned  BSIM3b0Given   :1;
    unsigned  BSIM3b1Given   :1;
    unsigned  BSIM3alpha0Given   :1;
    unsigned  BSIM3alpha1Given   :1;
    unsigned  BSIM3beta0Given   :1;
    unsigned  BSIM3ijthGiven   :1;
    unsigned  BSIM3vfbGiven   :1;

    /* CV model */
    unsigned  BSIM3elmGiven  :1;
    unsigned  BSIM3cgslGiven   :1;
    unsigned  BSIM3cgdlGiven   :1;
    unsigned  BSIM3ckappaGiven   :1;
    unsigned  BSIM3cfGiven   :1;
    unsigned  BSIM3vfbcvGiven   :1;
    unsigned  BSIM3clcGiven   :1;
    unsigned  BSIM3cleGiven   :1;
    unsigned  BSIM3dwcGiven   :1;
    unsigned  BSIM3dlcGiven   :1;
    unsigned  BSIM3noffGiven  :1;
    unsigned  BSIM3voffcvGiven :1;
    unsigned  BSIM3acdeGiven  :1;
    unsigned  BSIM3moinGiven  :1;
    unsigned  BSIM3tcjGiven   :1;
    unsigned  BSIM3tcjswGiven :1;
    unsigned  BSIM3tcjswgGiven :1;
    unsigned  BSIM3tpbGiven    :1;
    unsigned  BSIM3tpbswGiven  :1;
    unsigned  BSIM3tpbswgGiven :1;

    /* ACM model */
    unsigned  BSIM3xlGiven   :1;
    unsigned  BSIM3xwGiven   :1;
    unsigned  BSIM3hdifGiven  :1;
    unsigned  BSIM3ldifGiven   :1;
    unsigned  BSIM3ldGiven   :1;
    unsigned  BSIM3rdGiven   :1;
    unsigned  BSIM3rsGiven   :1;
    unsigned  BSIM3rdcGiven   :1;
    unsigned  BSIM3rscGiven   :1;
    unsigned  BSIM3wmltGiven   :1;

    /* Length dependence */
    unsigned  BSIM3lcdscGiven   :1;
    unsigned  BSIM3lcdscbGiven   :1;
    unsigned  BSIM3lcdscdGiven   :1;
    unsigned  BSIM3lcitGiven   :1;
    unsigned  BSIM3lnfactorGiven   :1;
    unsigned  BSIM3lxjGiven   :1;
    unsigned  BSIM3lvsatGiven   :1;
    unsigned  BSIM3latGiven   :1;
    unsigned  BSIM3la0Given   :1;
    unsigned  BSIM3lagsGiven   :1;
    unsigned  BSIM3la1Given   :1;
    unsigned  BSIM3la2Given   :1;
    unsigned  BSIM3lketaGiven   :1;
    unsigned  BSIM3lnsubGiven   :1;
    unsigned  BSIM3lnpeakGiven   :1;
    unsigned  BSIM3lngateGiven   :1;
    unsigned  BSIM3lgamma1Given   :1;
    unsigned  BSIM3lgamma2Given   :1;
    unsigned  BSIM3lvbxGiven   :1;
    unsigned  BSIM3lvbmGiven   :1;
    unsigned  BSIM3lxtGiven   :1;
    unsigned  BSIM3lk1Given   :1;
    unsigned  BSIM3lkt1Given   :1;
    unsigned  BSIM3lkt1lGiven   :1;
    unsigned  BSIM3lkt2Given   :1;
    unsigned  BSIM3lk2Given   :1;
    unsigned  BSIM3lk3Given   :1;
    unsigned  BSIM3lk3bGiven   :1;
    unsigned  BSIM3lw0Given   :1;
    unsigned  BSIM3lnlxGiven   :1;
    unsigned  BSIM3ldvt0Given   :1;
    unsigned  BSIM3ldvt1Given   :1;
    unsigned  BSIM3ldvt2Given   :1;
    unsigned  BSIM3ldvt0wGiven   :1;
    unsigned  BSIM3ldvt1wGiven   :1;
    unsigned  BSIM3ldvt2wGiven   :1;
    unsigned  BSIM3ldroutGiven   :1;
    unsigned  BSIM3ldsubGiven   :1;
    unsigned  BSIM3lvth0Given   :1;
    unsigned  BSIM3luaGiven   :1;
    unsigned  BSIM3lua1Given   :1;
    unsigned  BSIM3lubGiven   :1;
    unsigned  BSIM3lub1Given   :1;
    unsigned  BSIM3lucGiven   :1;
    unsigned  BSIM3luc1Given   :1;
    unsigned  BSIM3lu0Given   :1;
    unsigned  BSIM3luteGiven   :1;
    unsigned  BSIM3lvoffGiven   :1;
    unsigned  BSIM3lrdswGiven   :1;
    unsigned  BSIM3lprwgGiven   :1;
    unsigned  BSIM3lprwbGiven   :1;
    unsigned  BSIM3lprtGiven   :1;
    unsigned  BSIM3leta0Given   :1;
    unsigned  BSIM3letabGiven   :1;
    unsigned  BSIM3lpclmGiven   :1;
    unsigned  BSIM3lpdibl1Given   :1;
    unsigned  BSIM3lpdibl2Given   :1;
    unsigned  BSIM3lpdiblbGiven   :1;
    unsigned  BSIM3lpscbe1Given   :1;
    unsigned  BSIM3lpscbe2Given   :1;
    unsigned  BSIM3lpvagGiven   :1;
    unsigned  BSIM3ldeltaGiven  :1;
    unsigned  BSIM3lwrGiven   :1;
    unsigned  BSIM3ldwgGiven   :1;
    unsigned  BSIM3ldwbGiven   :1;
    unsigned  BSIM3lb0Given   :1;
    unsigned  BSIM3lb1Given   :1;
    unsigned  BSIM3lalpha0Given   :1;
    unsigned  BSIM3lalpha1Given   :1;
    unsigned  BSIM3lbeta0Given   :1;
    unsigned  BSIM3lvfbGiven   :1;

    /* CV model */
    unsigned  BSIM3lelmGiven  :1;
    unsigned  BSIM3lcgslGiven   :1;
    unsigned  BSIM3lcgdlGiven   :1;
    unsigned  BSIM3lckappaGiven   :1;
    unsigned  BSIM3lcfGiven   :1;
    unsigned  BSIM3lclcGiven   :1;
    unsigned  BSIM3lcleGiven   :1;
    unsigned  BSIM3lvfbcvGiven   :1;
    unsigned  BSIM3lnoffGiven   :1;
    unsigned  BSIM3lvoffcvGiven :1;
    unsigned  BSIM3lacdeGiven   :1;
    unsigned  BSIM3lmoinGiven   :1;

    /* Width dependence */
    unsigned  BSIM3wcdscGiven   :1;
    unsigned  BSIM3wcdscbGiven   :1;
    unsigned  BSIM3wcdscdGiven   :1;
    unsigned  BSIM3wcitGiven   :1;
    unsigned  BSIM3wnfactorGiven   :1;
    unsigned  BSIM3wxjGiven   :1;
    unsigned  BSIM3wvsatGiven   :1;
    unsigned  BSIM3watGiven   :1;
    unsigned  BSIM3wa0Given   :1;
    unsigned  BSIM3wagsGiven   :1;
    unsigned  BSIM3wa1Given   :1;
    unsigned  BSIM3wa2Given   :1;
    unsigned  BSIM3wketaGiven   :1;
    unsigned  BSIM3wnsubGiven   :1;
    unsigned  BSIM3wnpeakGiven   :1;
    unsigned  BSIM3wngateGiven   :1;
    unsigned  BSIM3wgamma1Given   :1;
    unsigned  BSIM3wgamma2Given   :1;
    unsigned  BSIM3wvbxGiven   :1;
    unsigned  BSIM3wvbmGiven   :1;
    unsigned  BSIM3wxtGiven   :1;
    unsigned  BSIM3wk1Given   :1;
    unsigned  BSIM3wkt1Given   :1;
    unsigned  BSIM3wkt1lGiven   :1;
    unsigned  BSIM3wkt2Given   :1;
    unsigned  BSIM3wk2Given   :1;
    unsigned  BSIM3wk3Given   :1;
    unsigned  BSIM3wk3bGiven   :1;
    unsigned  BSIM3ww0Given   :1;
    unsigned  BSIM3wnlxGiven   :1;
    unsigned  BSIM3wdvt0Given   :1;
    unsigned  BSIM3wdvt1Given   :1;
    unsigned  BSIM3wdvt2Given   :1;
    unsigned  BSIM3wdvt0wGiven   :1;
    unsigned  BSIM3wdvt1wGiven   :1;
    unsigned  BSIM3wdvt2wGiven   :1;
    unsigned  BSIM3wdroutGiven   :1;
    unsigned  BSIM3wdsubGiven   :1;
    unsigned  BSIM3wvth0Given   :1;
    unsigned  BSIM3wuaGiven   :1;
    unsigned  BSIM3wua1Given   :1;
    unsigned  BSIM3wubGiven   :1;
    unsigned  BSIM3wub1Given   :1;
    unsigned  BSIM3wucGiven   :1;
    unsigned  BSIM3wuc1Given   :1;
    unsigned  BSIM3wu0Given   :1;
    unsigned  BSIM3wuteGiven   :1;
    unsigned  BSIM3wvoffGiven   :1;
    unsigned  BSIM3wrdswGiven   :1;
    unsigned  BSIM3wprwgGiven   :1;
    unsigned  BSIM3wprwbGiven   :1;
    unsigned  BSIM3wprtGiven   :1;
    unsigned  BSIM3weta0Given   :1;
    unsigned  BSIM3wetabGiven   :1;
    unsigned  BSIM3wpclmGiven   :1;
    unsigned  BSIM3wpdibl1Given   :1;
    unsigned  BSIM3wpdibl2Given   :1;
    unsigned  BSIM3wpdiblbGiven   :1;
    unsigned  BSIM3wpscbe1Given   :1;
    unsigned  BSIM3wpscbe2Given   :1;
    unsigned  BSIM3wpvagGiven   :1;
    unsigned  BSIM3wdeltaGiven  :1;
    unsigned  BSIM3wwrGiven   :1;
    unsigned  BSIM3wdwgGiven   :1;
    unsigned  BSIM3wdwbGiven   :1;
    unsigned  BSIM3wb0Given   :1;
    unsigned  BSIM3wb1Given   :1;
    unsigned  BSIM3walpha0Given   :1;
    unsigned  BSIM3walpha1Given   :1;
    unsigned  BSIM3wbeta0Given   :1;
    unsigned  BSIM3wvfbGiven   :1;

    /* CV model */
    unsigned  BSIM3welmGiven  :1;
    unsigned  BSIM3wcgslGiven   :1;
    unsigned  BSIM3wcgdlGiven   :1;
    unsigned  BSIM3wckappaGiven   :1;
    unsigned  BSIM3wcfGiven   :1;
    unsigned  BSIM3wclcGiven   :1;
    unsigned  BSIM3wcleGiven   :1;
    unsigned  BSIM3wvfbcvGiven   :1;
    unsigned  BSIM3wnoffGiven   :1;
    unsigned  BSIM3wvoffcvGiven :1;
    unsigned  BSIM3wacdeGiven   :1;
    unsigned  BSIM3wmoinGiven   :1;

    /* Cross-term dependence */
    unsigned  BSIM3pcdscGiven   :1;
    unsigned  BSIM3pcdscbGiven   :1;
    unsigned  BSIM3pcdscdGiven   :1;
    unsigned  BSIM3pcitGiven   :1;
    unsigned  BSIM3pnfactorGiven   :1;
    unsigned  BSIM3pxjGiven   :1;
    unsigned  BSIM3pvsatGiven   :1;
    unsigned  BSIM3patGiven   :1;
    unsigned  BSIM3pa0Given   :1;
    unsigned  BSIM3pagsGiven   :1;
    unsigned  BSIM3pa1Given   :1;
    unsigned  BSIM3pa2Given   :1;
    unsigned  BSIM3pketaGiven   :1;
    unsigned  BSIM3pnsubGiven   :1;
    unsigned  BSIM3pnpeakGiven   :1;
    unsigned  BSIM3pngateGiven   :1;
    unsigned  BSIM3pgamma1Given   :1;
    unsigned  BSIM3pgamma2Given   :1;
    unsigned  BSIM3pvbxGiven   :1;
    unsigned  BSIM3pvbmGiven   :1;
    unsigned  BSIM3pxtGiven   :1;
    unsigned  BSIM3pk1Given   :1;
    unsigned  BSIM3pkt1Given   :1;
    unsigned  BSIM3pkt1lGiven   :1;
    unsigned  BSIM3pkt2Given   :1;
    unsigned  BSIM3pk2Given   :1;
    unsigned  BSIM3pk3Given   :1;
    unsigned  BSIM3pk3bGiven   :1;
    unsigned  BSIM3pw0Given   :1;
    unsigned  BSIM3pnlxGiven   :1;
    unsigned  BSIM3pdvt0Given   :1;
    unsigned  BSIM3pdvt1Given   :1;
    unsigned  BSIM3pdvt2Given   :1;
    unsigned  BSIM3pdvt0wGiven   :1;
    unsigned  BSIM3pdvt1wGiven   :1;
    unsigned  BSIM3pdvt2wGiven   :1;
    unsigned  BSIM3pdroutGiven   :1;
    unsigned  BSIM3pdsubGiven   :1;
    unsigned  BSIM3pvth0Given   :1;
    unsigned  BSIM3puaGiven   :1;
    unsigned  BSIM3pua1Given   :1;
    unsigned  BSIM3pubGiven   :1;
    unsigned  BSIM3pub1Given   :1;
    unsigned  BSIM3pucGiven   :1;
    unsigned  BSIM3puc1Given   :1;
    unsigned  BSIM3pu0Given   :1;
    unsigned  BSIM3puteGiven   :1;
    unsigned  BSIM3pvoffGiven   :1;
    unsigned  BSIM3prdswGiven   :1;
    unsigned  BSIM3pprwgGiven   :1;
    unsigned  BSIM3pprwbGiven   :1;
    unsigned  BSIM3pprtGiven   :1;
    unsigned  BSIM3peta0Given   :1;
    unsigned  BSIM3petabGiven   :1;
    unsigned  BSIM3ppclmGiven   :1;
    unsigned  BSIM3ppdibl1Given   :1;
    unsigned  BSIM3ppdibl2Given   :1;
    unsigned  BSIM3ppdiblbGiven   :1;
    unsigned  BSIM3ppscbe1Given   :1;
    unsigned  BSIM3ppscbe2Given   :1;
    unsigned  BSIM3ppvagGiven   :1;
    unsigned  BSIM3pdeltaGiven  :1;
    unsigned  BSIM3pwrGiven   :1;
    unsigned  BSIM3pdwgGiven   :1;
    unsigned  BSIM3pdwbGiven   :1;
    unsigned  BSIM3pb0Given   :1;
    unsigned  BSIM3pb1Given   :1;
    unsigned  BSIM3palpha0Given   :1;
    unsigned  BSIM3palpha1Given   :1;
    unsigned  BSIM3pbeta0Given   :1;
    unsigned  BSIM3pvfbGiven   :1;

    /* CV model */
    unsigned  BSIM3pelmGiven  :1;
    unsigned  BSIM3pcgslGiven   :1;
    unsigned  BSIM3pcgdlGiven   :1;
    unsigned  BSIM3pckappaGiven   :1;
    unsigned  BSIM3pcfGiven   :1;
    unsigned  BSIM3pclcGiven   :1;
    unsigned  BSIM3pcleGiven   :1;
    unsigned  BSIM3pvfbcvGiven   :1;
    unsigned  BSIM3pnoffGiven   :1;
    unsigned  BSIM3pvoffcvGiven :1;
    unsigned  BSIM3pacdeGiven   :1;
    unsigned  BSIM3pmoinGiven   :1;

    unsigned  BSIM3useFringeGiven   :1;

    unsigned  BSIM3tnomGiven   :1;
    unsigned  BSIM3cgsoGiven   :1;
    unsigned  BSIM3cgdoGiven   :1;
    unsigned  BSIM3cgboGiven   :1;
    unsigned  BSIM3xpartGiven   :1;
    unsigned  BSIM3sheetResistanceGiven   :1;
    unsigned  BSIM3jctSatCurDensityGiven   :1;
    unsigned  BSIM3jctSidewallSatCurDensityGiven   :1;
    unsigned  BSIM3bulkJctPotentialGiven   :1;
    unsigned  BSIM3bulkJctBotGradingCoeffGiven   :1;
    unsigned  BSIM3sidewallJctPotentialGiven   :1;
    unsigned  BSIM3GatesidewallJctPotentialGiven   :1;
    unsigned  BSIM3bulkJctSideGradingCoeffGiven   :1;
    unsigned  BSIM3unitAreaJctCapGiven   :1;
    unsigned  BSIM3unitLengthSidewallJctCapGiven   :1;
    unsigned  BSIM3bulkJctGateSideGradingCoeffGiven   :1;
    unsigned  BSIM3unitLengthGateSidewallJctCapGiven   :1;
    unsigned  BSIM3jctEmissionCoeffGiven :1;
    unsigned  BSIM3jctTempExponentGiven        :1;

    unsigned  BSIM3oxideTrapDensityAGiven  :1;
    unsigned  BSIM3oxideTrapDensityBGiven  :1;
    unsigned  BSIM3oxideTrapDensityCGiven  :1;
    unsigned  BSIM3emGiven  :1;
    unsigned  BSIM3efGiven  :1;
    unsigned  BSIM3afGiven  :1;
    unsigned  BSIM3kfGiven  :1;
    unsigned  BSIM3lintnoiGiven  :1;

    unsigned  BSIM3vgsMaxGiven  :1;
    unsigned  BSIM3vgdMaxGiven  :1;
    unsigned  BSIM3vgbMaxGiven  :1;
    unsigned  BSIM3vdsMaxGiven  :1;
    unsigned  BSIM3vbsMaxGiven  :1;
    unsigned  BSIM3vbdMaxGiven  :1;
    unsigned  BSIM3vgsrMaxGiven  :1;
    unsigned  BSIM3vgdrMaxGiven  :1;
    unsigned  BSIM3vgbrMaxGiven  :1;
    unsigned  BSIM3vbsrMaxGiven  :1;
    unsigned  BSIM3vbdrMaxGiven  :1;

    unsigned  BSIM3LintGiven   :1;
    unsigned  BSIM3LlGiven   :1;
    unsigned  BSIM3LlcGiven   :1;
    unsigned  BSIM3LlnGiven   :1;
    unsigned  BSIM3LwGiven   :1;
    unsigned  BSIM3LwcGiven   :1;
    unsigned  BSIM3LwnGiven   :1;
    unsigned  BSIM3LwlGiven   :1;
    unsigned  BSIM3LwlcGiven   :1;
    unsigned  BSIM3LminGiven   :1;
    unsigned  BSIM3LmaxGiven   :1;

    unsigned  BSIM3WintGiven   :1;
    unsigned  BSIM3WlGiven   :1;
    unsigned  BSIM3WlcGiven   :1;
    unsigned  BSIM3WlnGiven   :1;
    unsigned  BSIM3WwGiven   :1;
    unsigned  BSIM3WwcGiven   :1;
    unsigned  BSIM3WwnGiven   :1;
    unsigned  BSIM3WwlGiven   :1;
    unsigned  BSIM3WwlcGiven   :1;
    unsigned  BSIM3WminGiven   :1;
    unsigned  BSIM3WmaxGiven   :1;

} BSIM3model;


#ifndef NMOS
#define NMOS 1
#define PMOS -1
#endif /*NMOS*/


/* device parameters */
#define BSIM3_W 1
#define BSIM3_L 2
#define BSIM3_AS 3
#define BSIM3_AD 4
#define BSIM3_PS 5
#define BSIM3_PD 6
#define BSIM3_NRS 7
#define BSIM3_NRD 8
#define BSIM3_OFF 9
#define BSIM3_IC_VBS 10
#define BSIM3_IC_VDS 11
#define BSIM3_IC_VGS 12
#define BSIM3_IC 13
#define BSIM3_NQSMOD 14
#define BSIM3_ACNQSMOD 15
#define BSIM3_M 16
#define BSIM3_DELVTO 17
#define BSIM3_MULU0 18
#define BSIM3_GEO 19

/* model parameters */
#define BSIM3_MOD_CAPMOD          100
#define BSIM3_MOD_ACMMOD          101
#define BSIM3_MOD_CALCACM         102
#define BSIM3_MOD_MOBMOD          103
#define BSIM3_MOD_NOIMOD          104

#define BSIM3_MOD_TOX             105

#define BSIM3_MOD_CDSC            106
#define BSIM3_MOD_CDSCB           107
#define BSIM3_MOD_CIT             108
#define BSIM3_MOD_NFACTOR         109
#define BSIM3_MOD_XJ              110
#define BSIM3_MOD_VSAT            111
#define BSIM3_MOD_AT              112
#define BSIM3_MOD_A0              113
#define BSIM3_MOD_A1              114
#define BSIM3_MOD_A2              115
#define BSIM3_MOD_KETA            116
#define BSIM3_MOD_NSUB            117
#define BSIM3_MOD_NPEAK           118
#define BSIM3_MOD_NGATE           120
#define BSIM3_MOD_GAMMA1          121
#define BSIM3_MOD_GAMMA2          122
#define BSIM3_MOD_VBX             123
#define BSIM3_MOD_BINUNIT         124

#define BSIM3_MOD_VBM             125

#define BSIM3_MOD_XT              126
#define BSIM3_MOD_K1              129
#define BSIM3_MOD_KT1             130
#define BSIM3_MOD_KT1L            131
#define BSIM3_MOD_K2              132
#define BSIM3_MOD_KT2             133
#define BSIM3_MOD_K3              134
#define BSIM3_MOD_K3B             135
#define BSIM3_MOD_W0              136
#define BSIM3_MOD_NLX             137

#define BSIM3_MOD_DVT0            138
#define BSIM3_MOD_DVT1            139
#define BSIM3_MOD_DVT2            140

#define BSIM3_MOD_DVT0W           141
#define BSIM3_MOD_DVT1W           142
#define BSIM3_MOD_DVT2W           143

#define BSIM3_MOD_DROUT           144
#define BSIM3_MOD_DSUB            145
#define BSIM3_MOD_VTH0            146
#define BSIM3_MOD_UA              147
#define BSIM3_MOD_UA1             148
#define BSIM3_MOD_UB              149
#define BSIM3_MOD_UB1             150
#define BSIM3_MOD_UC              151
#define BSIM3_MOD_UC1             152
#define BSIM3_MOD_U0              153
#define BSIM3_MOD_UTE             154
#define BSIM3_MOD_VOFF            155
#define BSIM3_MOD_DELTA           156
#define BSIM3_MOD_RDSW            157
#define BSIM3_MOD_PRT             158
#define BSIM3_MOD_LDD             159
#define BSIM3_MOD_ETA             160
#define BSIM3_MOD_ETA0            161
#define BSIM3_MOD_ETAB            162
#define BSIM3_MOD_PCLM            163
#define BSIM3_MOD_PDIBL1          164
#define BSIM3_MOD_PDIBL2          165
#define BSIM3_MOD_PSCBE1          166
#define BSIM3_MOD_PSCBE2          167
#define BSIM3_MOD_PVAG            168
#define BSIM3_MOD_WR              169
#define BSIM3_MOD_DWG             170
#define BSIM3_MOD_DWB             171
#define BSIM3_MOD_B0              172
#define BSIM3_MOD_B1              173
#define BSIM3_MOD_ALPHA0          174
#define BSIM3_MOD_BETA0           175
#define BSIM3_MOD_PDIBLB          178

#define BSIM3_MOD_PRWG            179
#define BSIM3_MOD_PRWB            180

#define BSIM3_MOD_CDSCD           181
#define BSIM3_MOD_AGS             182

#define BSIM3_MOD_FRINGE          184
#define BSIM3_MOD_ELM             185
#define BSIM3_MOD_CGSL            186
#define BSIM3_MOD_CGDL            187
#define BSIM3_MOD_CKAPPA          188
#define BSIM3_MOD_CF              189
#define BSIM3_MOD_CLC             190
#define BSIM3_MOD_CLE             191
#define BSIM3_MOD_PARAMCHK        192
#define BSIM3_MOD_VERSION         193
#define BSIM3_MOD_VFBCV           194
#define BSIM3_MOD_ACDE            195
#define BSIM3_MOD_MOIN            196
#define BSIM3_MOD_NOFF            197
#define BSIM3_MOD_IJTH            198
#define BSIM3_MOD_ALPHA1          199
#define BSIM3_MOD_VFB             200
#define BSIM3_MOD_TOXM            201
#define BSIM3_MOD_TCJ             202
#define BSIM3_MOD_TCJSW           203
#define BSIM3_MOD_TCJSWG          204
#define BSIM3_MOD_TPB             205
#define BSIM3_MOD_TPBSW           206
#define BSIM3_MOD_TPBSWG          207
#define BSIM3_MOD_VOFFCV          208
#define BSIM3_MOD_LINTNOI         209
#define BSIM3_MOD_NQSMOD          210
#define BSIM3_MOD_ACNQSMOD        211

/* Length dependence */
#define BSIM3_MOD_LCDSC            251
#define BSIM3_MOD_LCDSCB           252
#define BSIM3_MOD_LCIT             253
#define BSIM3_MOD_LNFACTOR         254
#define BSIM3_MOD_LXJ              255
#define BSIM3_MOD_LVSAT            256
#define BSIM3_MOD_LAT              257
#define BSIM3_MOD_LA0              258
#define BSIM3_MOD_LA1              259
#define BSIM3_MOD_LA2              260
#define BSIM3_MOD_LKETA            261
#define BSIM3_MOD_LNSUB            262
#define BSIM3_MOD_LNPEAK           263
#define BSIM3_MOD_LNGATE           265
#define BSIM3_MOD_LGAMMA1          266
#define BSIM3_MOD_LGAMMA2          267
#define BSIM3_MOD_LVBX             268

#define BSIM3_MOD_LVBM             270

#define BSIM3_MOD_LXT              272
#define BSIM3_MOD_LK1              275
#define BSIM3_MOD_LKT1             276
#define BSIM3_MOD_LKT1L            277
#define BSIM3_MOD_LK2              278
#define BSIM3_MOD_LKT2             279
#define BSIM3_MOD_LK3              280
#define BSIM3_MOD_LK3B             281
#define BSIM3_MOD_LW0              282
#define BSIM3_MOD_LNLX             283

#define BSIM3_MOD_LDVT0            284
#define BSIM3_MOD_LDVT1            285
#define BSIM3_MOD_LDVT2            286

#define BSIM3_MOD_LDVT0W           287
#define BSIM3_MOD_LDVT1W           288
#define BSIM3_MOD_LDVT2W           289

#define BSIM3_MOD_LDROUT           290
#define BSIM3_MOD_LDSUB            291
#define BSIM3_MOD_LVTH0            292
#define BSIM3_MOD_LUA              293
#define BSIM3_MOD_LUA1             294
#define BSIM3_MOD_LUB              295
#define BSIM3_MOD_LUB1             296
#define BSIM3_MOD_LUC              297
#define BSIM3_MOD_LUC1             298
#define BSIM3_MOD_LU0              299
#define BSIM3_MOD_LUTE             300
#define BSIM3_MOD_LVOFF            301
#define BSIM3_MOD_LDELTA           302
#define BSIM3_MOD_LRDSW            303
#define BSIM3_MOD_LPRT             304
#define BSIM3_MOD_LLDD             305
#define BSIM3_MOD_LETA             306
#define BSIM3_MOD_LETA0            307
#define BSIM3_MOD_LETAB            308
#define BSIM3_MOD_LPCLM            309
#define BSIM3_MOD_LPDIBL1          310
#define BSIM3_MOD_LPDIBL2          311
#define BSIM3_MOD_LPSCBE1          312
#define BSIM3_MOD_LPSCBE2          313
#define BSIM3_MOD_LPVAG            314
#define BSIM3_MOD_LWR              315
#define BSIM3_MOD_LDWG             316
#define BSIM3_MOD_LDWB             317
#define BSIM3_MOD_LB0              318
#define BSIM3_MOD_LB1              319
#define BSIM3_MOD_LALPHA0          320
#define BSIM3_MOD_LBETA0           321
#define BSIM3_MOD_LPDIBLB          324

#define BSIM3_MOD_LPRWG            325
#define BSIM3_MOD_LPRWB            326

#define BSIM3_MOD_LCDSCD           327
#define BSIM3_MOD_LAGS             328


#define BSIM3_MOD_LFRINGE          331
#define BSIM3_MOD_LELM             332
#define BSIM3_MOD_LCGSL            333
#define BSIM3_MOD_LCGDL            334
#define BSIM3_MOD_LCKAPPA          335
#define BSIM3_MOD_LCF              336
#define BSIM3_MOD_LCLC             337
#define BSIM3_MOD_LCLE             338
#define BSIM3_MOD_LVFBCV           339
#define BSIM3_MOD_LACDE            340
#define BSIM3_MOD_LMOIN            341
#define BSIM3_MOD_LNOFF            342
#define BSIM3_MOD_LALPHA1          344
#define BSIM3_MOD_LVFB             345
#define BSIM3_MOD_LVOFFCV          346

/* Width dependence */
#define BSIM3_MOD_WCDSC            381
#define BSIM3_MOD_WCDSCB           382
#define BSIM3_MOD_WCIT             383
#define BSIM3_MOD_WNFACTOR         384
#define BSIM3_MOD_WXJ              385
#define BSIM3_MOD_WVSAT            386
#define BSIM3_MOD_WAT              387
#define BSIM3_MOD_WA0              388
#define BSIM3_MOD_WA1              389
#define BSIM3_MOD_WA2              390
#define BSIM3_MOD_WKETA            391
#define BSIM3_MOD_WNSUB            392
#define BSIM3_MOD_WNPEAK           393
#define BSIM3_MOD_WNGATE           395
#define BSIM3_MOD_WGAMMA1          396
#define BSIM3_MOD_WGAMMA2          397
#define BSIM3_MOD_WVBX             398

#define BSIM3_MOD_WVBM             400

#define BSIM3_MOD_WXT              402
#define BSIM3_MOD_WK1              405
#define BSIM3_MOD_WKT1             406
#define BSIM3_MOD_WKT1L            407
#define BSIM3_MOD_WK2              408
#define BSIM3_MOD_WKT2             409
#define BSIM3_MOD_WK3              410
#define BSIM3_MOD_WK3B             411
#define BSIM3_MOD_WW0              412
#define BSIM3_MOD_WNLX             413

#define BSIM3_MOD_WDVT0            414
#define BSIM3_MOD_WDVT1            415
#define BSIM3_MOD_WDVT2            416

#define BSIM3_MOD_WDVT0W           417
#define BSIM3_MOD_WDVT1W           418
#define BSIM3_MOD_WDVT2W           419

#define BSIM3_MOD_WDROUT           420
#define BSIM3_MOD_WDSUB            421
#define BSIM3_MOD_WVTH0            422
#define BSIM3_MOD_WUA              423
#define BSIM3_MOD_WUA1             424
#define BSIM3_MOD_WUB              425
#define BSIM3_MOD_WUB1             426
#define BSIM3_MOD_WUC              427
#define BSIM3_MOD_WUC1             428
#define BSIM3_MOD_WU0              429
#define BSIM3_MOD_WUTE             430
#define BSIM3_MOD_WVOFF            431
#define BSIM3_MOD_WDELTA           432
#define BSIM3_MOD_WRDSW            433
#define BSIM3_MOD_WPRT             434
#define BSIM3_MOD_WLDD             435
#define BSIM3_MOD_WETA             436
#define BSIM3_MOD_WETA0            437
#define BSIM3_MOD_WETAB            438
#define BSIM3_MOD_WPCLM            439
#define BSIM3_MOD_WPDIBL1          440
#define BSIM3_MOD_WPDIBL2          441
#define BSIM3_MOD_WPSCBE1          442
#define BSIM3_MOD_WPSCBE2          443
#define BSIM3_MOD_WPVAG            444
#define BSIM3_MOD_WWR              445
#define BSIM3_MOD_WDWG             446
#define BSIM3_MOD_WDWB             447
#define BSIM3_MOD_WB0              448
#define BSIM3_MOD_WB1              449
#define BSIM3_MOD_WALPHA0          450
#define BSIM3_MOD_WBETA0           451
#define BSIM3_MOD_WPDIBLB          454

#define BSIM3_MOD_WPRWG            455
#define BSIM3_MOD_WPRWB            456

#define BSIM3_MOD_WCDSCD           457
#define BSIM3_MOD_WAGS             458


#define BSIM3_MOD_WFRINGE          461
#define BSIM3_MOD_WELM             462
#define BSIM3_MOD_WCGSL            463
#define BSIM3_MOD_WCGDL            464
#define BSIM3_MOD_WCKAPPA          465
#define BSIM3_MOD_WCF              466
#define BSIM3_MOD_WCLC             467
#define BSIM3_MOD_WCLE             468
#define BSIM3_MOD_WVFBCV           469
#define BSIM3_MOD_WACDE            470
#define BSIM3_MOD_WMOIN            471
#define BSIM3_MOD_WNOFF            472
#define BSIM3_MOD_WALPHA1          474
#define BSIM3_MOD_WVFB             475
#define BSIM3_MOD_WVOFFCV          476

/* Cross-term dependence */
#define BSIM3_MOD_PCDSC            511
#define BSIM3_MOD_PCDSCB           512
#define BSIM3_MOD_PCIT             513
#define BSIM3_MOD_PNFACTOR         514
#define BSIM3_MOD_PXJ              515
#define BSIM3_MOD_PVSAT            516
#define BSIM3_MOD_PAT              517
#define BSIM3_MOD_PA0              518
#define BSIM3_MOD_PA1              519
#define BSIM3_MOD_PA2              520
#define BSIM3_MOD_PKETA            521
#define BSIM3_MOD_PNSUB            522
#define BSIM3_MOD_PNPEAK           523
#define BSIM3_MOD_PNGATE           525
#define BSIM3_MOD_PGAMMA1          526
#define BSIM3_MOD_PGAMMA2          527
#define BSIM3_MOD_PVBX             528

#define BSIM3_MOD_PVBM             530

#define BSIM3_MOD_PXT              532
#define BSIM3_MOD_PK1              535
#define BSIM3_MOD_PKT1             536
#define BSIM3_MOD_PKT1L            537
#define BSIM3_MOD_PK2              538
#define BSIM3_MOD_PKT2             539
#define BSIM3_MOD_PK3              540
#define BSIM3_MOD_PK3B             541
#define BSIM3_MOD_PW0              542
#define BSIM3_MOD_PNLX             543

#define BSIM3_MOD_PDVT0            544
#define BSIM3_MOD_PDVT1            545
#define BSIM3_MOD_PDVT2            546

#define BSIM3_MOD_PDVT0W           547
#define BSIM3_MOD_PDVT1W           548
#define BSIM3_MOD_PDVT2W           549

#define BSIM3_MOD_PDROUT           550
#define BSIM3_MOD_PDSUB            551
#define BSIM3_MOD_PVTH0            552
#define BSIM3_MOD_PUA              553
#define BSIM3_MOD_PUA1             554
#define BSIM3_MOD_PUB              555
#define BSIM3_MOD_PUB1             556
#define BSIM3_MOD_PUC              557
#define BSIM3_MOD_PUC1             558
#define BSIM3_MOD_PU0              559
#define BSIM3_MOD_PUTE             560
#define BSIM3_MOD_PVOFF            561
#define BSIM3_MOD_PDELTA           562
#define BSIM3_MOD_PRDSW            563
#define BSIM3_MOD_PPRT             564
#define BSIM3_MOD_PLDD             565
#define BSIM3_MOD_PETA             566
#define BSIM3_MOD_PETA0            567
#define BSIM3_MOD_PETAB            568
#define BSIM3_MOD_PPCLM            569
#define BSIM3_MOD_PPDIBL1          570
#define BSIM3_MOD_PPDIBL2          571
#define BSIM3_MOD_PPSCBE1          572
#define BSIM3_MOD_PPSCBE2          573
#define BSIM3_MOD_PPVAG            574
#define BSIM3_MOD_PWR              575
#define BSIM3_MOD_PDWG             576
#define BSIM3_MOD_PDWB             577
#define BSIM3_MOD_PB0              578
#define BSIM3_MOD_PB1              579
#define BSIM3_MOD_PALPHA0          580
#define BSIM3_MOD_PBETA0           581
#define BSIM3_MOD_PPDIBLB          584

#define BSIM3_MOD_PPRWG            585
#define BSIM3_MOD_PPRWB            586

#define BSIM3_MOD_PCDSCD           587
#define BSIM3_MOD_PAGS             588

#define BSIM3_MOD_PFRINGE          591
#define BSIM3_MOD_PELM             592
#define BSIM3_MOD_PCGSL            593
#define BSIM3_MOD_PCGDL            594
#define BSIM3_MOD_PCKAPPA          595
#define BSIM3_MOD_PCF              596
#define BSIM3_MOD_PCLC             597
#define BSIM3_MOD_PCLE             598
#define BSIM3_MOD_PVFBCV           599
#define BSIM3_MOD_PACDE            600
#define BSIM3_MOD_PMOIN            601
#define BSIM3_MOD_PNOFF            602
#define BSIM3_MOD_PALPHA1          604
#define BSIM3_MOD_PVFB             605
#define BSIM3_MOD_PVOFFCV          606

#define BSIM3_MOD_TNOM             651
#define BSIM3_MOD_CGSO             652
#define BSIM3_MOD_CGDO             653
#define BSIM3_MOD_CGBO             654
#define BSIM3_MOD_XPART            655

#define BSIM3_MOD_RSH              656
#define BSIM3_MOD_JS               657
#define BSIM3_MOD_PB               658
#define BSIM3_MOD_MJ               659
#define BSIM3_MOD_PBSW             660
#define BSIM3_MOD_MJSW             661
#define BSIM3_MOD_CJ               662
#define BSIM3_MOD_CJSW             663
#define BSIM3_MOD_NMOS             664
#define BSIM3_MOD_PMOS             665

#define BSIM3_MOD_NOIA             666
#define BSIM3_MOD_NOIB             667
#define BSIM3_MOD_NOIC             668

#define BSIM3_MOD_LINT             669
#define BSIM3_MOD_LL               670
#define BSIM3_MOD_LLN              671
#define BSIM3_MOD_LW               672
#define BSIM3_MOD_LWN              673
#define BSIM3_MOD_LWL              674
#define BSIM3_MOD_LMIN             675
#define BSIM3_MOD_LMAX             676

#define BSIM3_MOD_WINT             677
#define BSIM3_MOD_WL               678
#define BSIM3_MOD_WLN              679
#define BSIM3_MOD_WW               680
#define BSIM3_MOD_WWN              681
#define BSIM3_MOD_WWL              682
#define BSIM3_MOD_WMIN             683
#define BSIM3_MOD_WMAX             684

#define BSIM3_MOD_DWC              685
#define BSIM3_MOD_DLC              686

#define BSIM3_MOD_EM               687
#define BSIM3_MOD_EF               688
#define BSIM3_MOD_AF               689
#define BSIM3_MOD_KF               690

#define BSIM3_MOD_NJ               691
#define BSIM3_MOD_XTI              692

#define BSIM3_MOD_PBSWG            693
#define BSIM3_MOD_MJSWG            694
#define BSIM3_MOD_CJSWG            695
#define BSIM3_MOD_JSW              696

#define BSIM3_MOD_LLC              697
#define BSIM3_MOD_LWC              698
#define BSIM3_MOD_LWLC             699

#define BSIM3_MOD_WLC              700
#define BSIM3_MOD_WWC              701
#define BSIM3_MOD_WWLC             702

/* ACM parameters */
#define BSIM3_MOD_XL               703
#define BSIM3_MOD_XW               704
#define BSIM3_MOD_HDIF             711
#define BSIM3_MOD_LDIF             712
#define BSIM3_MOD_LD               713
#define BSIM3_MOD_RD               714
#define BSIM3_MOD_RS               715
#define BSIM3_MOD_RDC              716
#define BSIM3_MOD_RSC              717
#define BSIM3_MOD_WMLT             718

/* device questions */
#define BSIM3_DNODE                751
#define BSIM3_GNODE                752
#define BSIM3_SNODE                753
#define BSIM3_BNODE                754
#define BSIM3_DNODEPRIME           755
#define BSIM3_SNODEPRIME           756
#define BSIM3_VBD                  757
#define BSIM3_VBS                  758
#define BSIM3_VGS                  759
#define BSIM3_VDS                  760
#define BSIM3_CD                   761
#define BSIM3_CBS                  762
#define BSIM3_CBD                  763
#define BSIM3_GM                   764
#define BSIM3_GDS                  765
#define BSIM3_GMBS                 766
#define BSIM3_GBD                  767
#define BSIM3_GBS                  768
#define BSIM3_QB                   769
#define BSIM3_CQB                  770
#define BSIM3_QG                   771
#define BSIM3_CQG                  772
#define BSIM3_QD                   773
#define BSIM3_CQD                  774
#define BSIM3_CGG                  775
#define BSIM3_CGD                  776
#define BSIM3_CGS                  777
#define BSIM3_CBG                  778
#define BSIM3_CAPBD                779
#define BSIM3_CQBD                 780
#define BSIM3_CAPBS                781
#define BSIM3_CQBS                 782
#define BSIM3_CDG                  783
#define BSIM3_CDD                  784
#define BSIM3_CDS                  785
#define BSIM3_VON                  786
#define BSIM3_VDSAT                787
#define BSIM3_QBS                  788
#define BSIM3_QBD                  789
#define BSIM3_SOURCECONDUCT        790
#define BSIM3_DRAINCONDUCT         791
#define BSIM3_CBDB                 792
#define BSIM3_CBSB                 793

#define BSIM3_MOD_VGS_MAX          801
#define BSIM3_MOD_VGD_MAX          802
#define BSIM3_MOD_VGB_MAX          803
#define BSIM3_MOD_VDS_MAX          804
#define BSIM3_MOD_VBS_MAX          805
#define BSIM3_MOD_VBD_MAX          806
#define BSIM3_MOD_VGSR_MAX         807
#define BSIM3_MOD_VGDR_MAX         808
#define BSIM3_MOD_VGBR_MAX         809
#define BSIM3_MOD_VBSR_MAX         810
#define BSIM3_MOD_VBDR_MAX         811

#include "bsim3ext.h"

extern void BSIM3evaluate(double,double,double,BSIM3instance*,BSIM3model*,
        double*,double*,double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, CKTcircuit*);
extern int BSIM3debug(BSIM3model*, BSIM3instance*, CKTcircuit*, int);
extern int BSIM3checkModel(BSIM3model*, BSIM3instance*, CKTcircuit*);

#endif /*BSIM3*/

