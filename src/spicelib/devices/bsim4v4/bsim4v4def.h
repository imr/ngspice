/* ngspice multirevision code extension covering 4.2.1 & 4.3.0 & 4.4.0 */
/**********
Copyright 2004 Regents of the University of California.  All rights reserved.
Author: 2000 Weidong Liu.
Modified by Xuemei Xi, 11/15/2002.
Modified by Xuemei Xi, 05/09/2003.
Modified by Xuemei Xi, 03/04/2004.
File: bsim4v4def.h
**********/

#ifndef BSIM4V4
#define BSIM4V4

#include "ngspice/ifsim.h"
#include "ngspice/gendefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/noisedef.h"

typedef struct sBSIM4v4instance
{
    struct sBSIM4v4model *BSIM4v4modPtr;
    struct sBSIM4v4instance *BSIM4v4nextInstance;
    IFuid BSIM4v4name;
    int BSIM4v4states;     /* index into state table for this device */
    int BSIM4v4dNode;
    int BSIM4v4gNodeExt;
    int BSIM4v4sNode;
    int BSIM4v4bNode;
    int BSIM4v4dNodePrime;
    int BSIM4v4gNodePrime;
    int BSIM4v4gNodeMid;
    int BSIM4v4sNodePrime;
    int BSIM4v4bNodePrime;
    int BSIM4v4dbNode;
    int BSIM4v4sbNode;
    int BSIM4v4qNode;

    double BSIM4v4ueff;
    double BSIM4v4thetavth;
    double BSIM4v4von;
    double BSIM4v4vdsat;
    double BSIM4v4cgdo;
    double BSIM4v4qgdo;
    double BSIM4v4cgso;
    double BSIM4v4qgso;
    double BSIM4v4grbsb;
    double BSIM4v4grbdb;
    double BSIM4v4grbpb;
    double BSIM4v4grbps;
    double BSIM4v4grbpd;

    double BSIM4v4vjsmFwd;
    double BSIM4v4vjsmRev;
    double BSIM4v4vjdmFwd;
    double BSIM4v4vjdmRev;
    double BSIM4v4XExpBVS;
    double BSIM4v4XExpBVD;
    double BSIM4v4SslpFwd;
    double BSIM4v4SslpRev;
    double BSIM4v4DslpFwd;
    double BSIM4v4DslpRev;
    double BSIM4v4IVjsmFwd;
    double BSIM4v4IVjsmRev;
    double BSIM4v4IVjdmFwd;
    double BSIM4v4IVjdmRev;

    double BSIM4v4grgeltd;
    double BSIM4v4Pseff;
    double BSIM4v4Pdeff;
    double BSIM4v4Aseff;
    double BSIM4v4Adeff;

    double BSIM4v4l;
    double BSIM4v4w;
    double BSIM4v4drainArea;
    double BSIM4v4sourceArea;
    double BSIM4v4drainSquares;
    double BSIM4v4sourceSquares;
    double BSIM4v4drainPerimeter;
    double BSIM4v4sourcePerimeter;
    double BSIM4v4sourceConductance;
    double BSIM4v4drainConductance;
     /* stress effect instance param */
    double BSIM4v4sa;
    double BSIM4v4sb;
    double BSIM4v4sd;

    double BSIM4v4rbdb;
    double BSIM4v4rbsb;
    double BSIM4v4rbpb;
    double BSIM4v4rbps;
    double BSIM4v4rbpd;

     /* added here to account stress effect instance dependence */
    double BSIM4v4u0temp;
    double BSIM4v4vsattemp;
    double BSIM4v4vth0;
    double BSIM4v4vfb;
    double BSIM4v4vfbzb;
    double BSIM4v4vtfbphi1;
    double BSIM4v4vtfbphi2;
    double BSIM4v4k2;
    double BSIM4v4vbsc;
    double BSIM4v4k2ox;
    double BSIM4v4eta0;

    double BSIM4v4icVDS;
    double BSIM4v4icVGS;
    double BSIM4v4icVBS;
    double BSIM4v4nf;
    double BSIM4v4m;
    int BSIM4v4off;
    int BSIM4v4mode;
    int BSIM4v4trnqsMod;
    int BSIM4v4acnqsMod;
    int BSIM4v4rbodyMod;
    int BSIM4v4rgateMod;
    int BSIM4v4geoMod;
    int BSIM4v4rgeoMod;
    int BSIM4v4min;


    /* OP point */
    double BSIM4v4Vgsteff;
    double BSIM4v4vgs_eff;
    double BSIM4v4vgd_eff;
    double BSIM4v4dvgs_eff_dvg;
    double BSIM4v4dvgd_eff_dvg;
    double BSIM4v4Vdseff;
    double BSIM4v4nstar;
    double BSIM4v4Abulk;
    double BSIM4v4EsatL;
    double BSIM4v4AbovVgst2Vtm;
    double BSIM4v4qinv;
    double BSIM4v4cd;
    double BSIM4v4cbs;
    double BSIM4v4cbd;
    double BSIM4v4csub;
    double BSIM4v4Igidl;
    double BSIM4v4Igisl;
    double BSIM4v4gm;
    double BSIM4v4gds;
    double BSIM4v4gmbs;
    double BSIM4v4gbd;
    double BSIM4v4gbs;

    double BSIM4v4gbbs;
    double BSIM4v4gbgs;
    double BSIM4v4gbds;
    double BSIM4v4ggidld;
    double BSIM4v4ggidlg;
    double BSIM4v4ggidls;
    double BSIM4v4ggidlb;
    double BSIM4v4ggisld;
    double BSIM4v4ggislg;
    double BSIM4v4ggisls;
    double BSIM4v4ggislb;

    double BSIM4v4Igcs;
    double BSIM4v4gIgcsg;
    double BSIM4v4gIgcsd;
    double BSIM4v4gIgcss;
    double BSIM4v4gIgcsb;
    double BSIM4v4Igcd;
    double BSIM4v4gIgcdg;
    double BSIM4v4gIgcdd;
    double BSIM4v4gIgcds;
    double BSIM4v4gIgcdb;

    double BSIM4v4Igs;
    double BSIM4v4gIgsg;
    double BSIM4v4gIgss;
    double BSIM4v4Igd;
    double BSIM4v4gIgdg;
    double BSIM4v4gIgdd;

    double BSIM4v4Igb;
    double BSIM4v4gIgbg;
    double BSIM4v4gIgbd;
    double BSIM4v4gIgbs;
    double BSIM4v4gIgbb;

    double BSIM4v4grdsw;
    double BSIM4v4IdovVds;
    double BSIM4v4gcrg;
    double BSIM4v4gcrgd;
    double BSIM4v4gcrgg;
    double BSIM4v4gcrgs;
    double BSIM4v4gcrgb;

    double BSIM4v4gstot;
    double BSIM4v4gstotd;
    double BSIM4v4gstotg;
    double BSIM4v4gstots;
    double BSIM4v4gstotb;

    double BSIM4v4gdtot;
    double BSIM4v4gdtotd;
    double BSIM4v4gdtotg;
    double BSIM4v4gdtots;
    double BSIM4v4gdtotb;

    double BSIM4v4cggb;
    double BSIM4v4cgdb;
    double BSIM4v4cgsb;
    double BSIM4v4cbgb;
    double BSIM4v4cbdb;
    double BSIM4v4cbsb;
    double BSIM4v4cdgb;
    double BSIM4v4cddb;
    double BSIM4v4cdsb;
    double BSIM4v4csgb;
    double BSIM4v4csdb;
    double BSIM4v4cssb;
    double BSIM4v4cgbb;
    double BSIM4v4cdbb;
    double BSIM4v4csbb;
    double BSIM4v4cbbb;
    double BSIM4v4capbd;
    double BSIM4v4capbs;

    double BSIM4v4cqgb;
    double BSIM4v4cqdb;
    double BSIM4v4cqsb;
    double BSIM4v4cqbb;

    double BSIM4v4qgate;
    double BSIM4v4qbulk;
    double BSIM4v4qdrn;
    double BSIM4v4qsrc;

    double BSIM4v4qchqs;
    double BSIM4v4taunet;
    double BSIM4v4gtau;
    double BSIM4v4gtg;
    double BSIM4v4gtd;
    double BSIM4v4gts;
    double BSIM4v4gtb;
    double BSIM4v4SjctTempRevSatCur;
    double BSIM4v4DjctTempRevSatCur;
    double BSIM4v4SswTempRevSatCur;
    double BSIM4v4DswTempRevSatCur;
    double BSIM4v4SswgTempRevSatCur;
    double BSIM4v4DswgTempRevSatCur;

    struct bsim4SizeDependParam  *pParam;

    unsigned BSIM4v4lGiven :1;
    unsigned BSIM4v4wGiven :1;
    unsigned BSIM4v4mGiven :1;
    unsigned BSIM4v4nfGiven :1;
    unsigned BSIM4v4minGiven :1;
    unsigned BSIM4v4drainAreaGiven :1;
    unsigned BSIM4v4sourceAreaGiven    :1;
    unsigned BSIM4v4drainSquaresGiven  :1;
    unsigned BSIM4v4sourceSquaresGiven :1;
    unsigned BSIM4v4drainPerimeterGiven    :1;
    unsigned BSIM4v4sourcePerimeterGiven   :1;
    unsigned BSIM4v4saGiven :1;
    unsigned BSIM4v4sbGiven :1;
    unsigned BSIM4v4sdGiven :1;
    unsigned BSIM4v4rbdbGiven   :1;
    unsigned BSIM4v4rbsbGiven   :1;
    unsigned BSIM4v4rbpbGiven   :1;
    unsigned BSIM4v4rbpdGiven   :1;
    unsigned BSIM4v4rbpsGiven   :1;
    unsigned BSIM4v4icVDSGiven :1;
    unsigned BSIM4v4icVGSGiven :1;
    unsigned BSIM4v4icVBSGiven :1;
    unsigned BSIM4v4trnqsModGiven :1;
    unsigned BSIM4v4acnqsModGiven :1;
    unsigned BSIM4v4rbodyModGiven :1;
    unsigned BSIM4v4rgateModGiven :1;
    unsigned BSIM4v4geoModGiven :1;
    unsigned BSIM4v4rgeoModGiven :1;


    double *BSIM4v4DPdPtr;
    double *BSIM4v4DPdpPtr;
    double *BSIM4v4DPgpPtr;
    double *BSIM4v4DPgmPtr;
    double *BSIM4v4DPspPtr;
    double *BSIM4v4DPbpPtr;
    double *BSIM4v4DPdbPtr;

    double *BSIM4v4DdPtr;
    double *BSIM4v4DdpPtr;

    double *BSIM4v4GPdpPtr;
    double *BSIM4v4GPgpPtr;
    double *BSIM4v4GPgmPtr;
    double *BSIM4v4GPgePtr;
    double *BSIM4v4GPspPtr;
    double *BSIM4v4GPbpPtr;

    double *BSIM4v4GMdpPtr;
    double *BSIM4v4GMgpPtr;
    double *BSIM4v4GMgmPtr;
    double *BSIM4v4GMgePtr;
    double *BSIM4v4GMspPtr;
    double *BSIM4v4GMbpPtr;

    double *BSIM4v4GEdpPtr;
    double *BSIM4v4GEgpPtr;
    double *BSIM4v4GEgmPtr;
    double *BSIM4v4GEgePtr;
    double *BSIM4v4GEspPtr;
    double *BSIM4v4GEbpPtr;

    double *BSIM4v4SPdpPtr;
    double *BSIM4v4SPgpPtr;
    double *BSIM4v4SPgmPtr;
    double *BSIM4v4SPsPtr;
    double *BSIM4v4SPspPtr;
    double *BSIM4v4SPbpPtr;
    double *BSIM4v4SPsbPtr;

    double *BSIM4v4SspPtr;
    double *BSIM4v4SsPtr;

    double *BSIM4v4BPdpPtr;
    double *BSIM4v4BPgpPtr;
    double *BSIM4v4BPgmPtr;
    double *BSIM4v4BPspPtr;
    double *BSIM4v4BPdbPtr;
    double *BSIM4v4BPbPtr;
    double *BSIM4v4BPsbPtr;
    double *BSIM4v4BPbpPtr;

    double *BSIM4v4DBdpPtr;
    double *BSIM4v4DBdbPtr;
    double *BSIM4v4DBbpPtr;
    double *BSIM4v4DBbPtr;

    double *BSIM4v4SBspPtr;
    double *BSIM4v4SBbpPtr;
    double *BSIM4v4SBbPtr;
    double *BSIM4v4SBsbPtr;

    double *BSIM4v4BdbPtr;
    double *BSIM4v4BbpPtr;
    double *BSIM4v4BsbPtr;
    double *BSIM4v4BbPtr;

    double *BSIM4v4DgpPtr;
    double *BSIM4v4DspPtr;
    double *BSIM4v4DbpPtr;
    double *BSIM4v4SdpPtr;
    double *BSIM4v4SgpPtr;
    double *BSIM4v4SbpPtr;

    double *BSIM4v4QdpPtr;
    double *BSIM4v4QgpPtr;
    double *BSIM4v4QspPtr;
    double *BSIM4v4QbpPtr;
    double *BSIM4v4QqPtr;
    double *BSIM4v4DPqPtr;
    double *BSIM4v4GPqPtr;
    double *BSIM4v4SPqPtr;


#define BSIM4v4vbd BSIM4v4states+ 0
#define BSIM4v4vbs BSIM4v4states+ 1
#define BSIM4v4vgs BSIM4v4states+ 2
#define BSIM4v4vds BSIM4v4states+ 3
#define BSIM4v4vdbs BSIM4v4states+ 4
#define BSIM4v4vdbd BSIM4v4states+ 5
#define BSIM4v4vsbs BSIM4v4states+ 6
#define BSIM4v4vges BSIM4v4states+ 7
#define BSIM4v4vgms BSIM4v4states+ 8
#define BSIM4v4vses BSIM4v4states+ 9
#define BSIM4v4vdes BSIM4v4states+ 10

#define BSIM4v4qb BSIM4v4states+ 11
#define BSIM4v4cqb BSIM4v4states+ 12
#define BSIM4v4qg BSIM4v4states+ 13
#define BSIM4v4cqg BSIM4v4states+ 14
#define BSIM4v4qd BSIM4v4states+ 15
#define BSIM4v4cqd BSIM4v4states+ 16
#define BSIM4v4qgmid BSIM4v4states+ 17
#define BSIM4v4cqgmid BSIM4v4states+ 18

#define BSIM4v4qbs  BSIM4v4states+ 19
#define BSIM4v4cqbs  BSIM4v4states+ 20
#define BSIM4v4qbd  BSIM4v4states+ 21
#define BSIM4v4cqbd  BSIM4v4states+ 22

#define BSIM4v4qcheq BSIM4v4states+ 23
#define BSIM4v4cqcheq BSIM4v4states+ 24
#define BSIM4v4qcdump BSIM4v4states+ 25
#define BSIM4v4cqcdump BSIM4v4states+ 26
#define BSIM4v4qdef BSIM4v4states+ 27
#define BSIM4v4qs BSIM4v4states+ 28

#define BSIM4v4numStates 29


/* indices to the array of BSIM4v4 NOISE SOURCES */

#define BSIM4v4RDNOIZ       0
#define BSIM4v4RSNOIZ       1
#define BSIM4v4RGNOIZ       2
#define BSIM4v4RBPSNOIZ     3
#define BSIM4v4RBPDNOIZ     4
#define BSIM4v4RBPBNOIZ     5
#define BSIM4v4RBSBNOIZ     6
#define BSIM4v4RBDBNOIZ     7
#define BSIM4v4IDNOIZ       8
#define BSIM4v4FLNOIZ       9
#define BSIM4v4IGSNOIZ      10
#define BSIM4v4IGDNOIZ      11
#define BSIM4v4IGBNOIZ      12
#define BSIM4v4TOTNOIZ      13

#define BSIM4v4NSRCS        14  /* Number of BSIM4v4 noise sources */

#ifndef NONOISE
    double BSIM4v4nVar[NSTATVARS][BSIM4v4NSRCS];
#else /* NONOISE */
        double **BSIM4v4nVar;
#endif /* NONOISE */

} BSIM4v4instance ;

struct bsim4SizeDependParam
{
    double Width;
    double Length;
    double NFinger;

    double BSIM4v4cdsc;
    double BSIM4v4cdscb;
    double BSIM4v4cdscd;
    double BSIM4v4cit;
    double BSIM4v4nfactor;
    double BSIM4v4xj;
    double BSIM4v4vsat;
    double BSIM4v4at;
    double BSIM4v4a0;
    double BSIM4v4ags;
    double BSIM4v4a1;
    double BSIM4v4a2;
    double BSIM4v4keta;
    double BSIM4v4nsub;
    double BSIM4v4ndep;
    double BSIM4v4nsd;
    double BSIM4v4phin;
    double BSIM4v4ngate;
    double BSIM4v4gamma1;
    double BSIM4v4gamma2;
    double BSIM4v4vbx;
    double BSIM4v4vbi;
    double BSIM4v4vbm;
    double BSIM4v4vbsc;
    double BSIM4v4xt;
    double BSIM4v4phi;
    double BSIM4v4litl;
    double BSIM4v4k1;
    double BSIM4v4kt1;
    double BSIM4v4kt1l;
    double BSIM4v4kt2;
    double BSIM4v4k2;
    double BSIM4v4k3;
    double BSIM4v4k3b;
    double BSIM4v4w0;
    double BSIM4v4dvtp0;
    double BSIM4v4dvtp1;
    double BSIM4v4lpe0;
    double BSIM4v4lpeb;
    double BSIM4v4dvt0;
    double BSIM4v4dvt1;
    double BSIM4v4dvt2;
    double BSIM4v4dvt0w;
    double BSIM4v4dvt1w;
    double BSIM4v4dvt2w;
    double BSIM4v4drout;
    double BSIM4v4dsub;
    double BSIM4v4vth0;
    double BSIM4v4ua;
    double BSIM4v4ua1;
    double BSIM4v4ub;
    double BSIM4v4ub1;
    double BSIM4v4uc;
    double BSIM4v4uc1;
    double BSIM4v4u0;
    double BSIM4v4eu;
    double BSIM4v4ute;
    double BSIM4v4voff;
    double BSIM4v4minv;
    double BSIM4v4vfb;
    double BSIM4v4delta;
    double BSIM4v4rdsw;
    double BSIM4v4rds0;
    double BSIM4v4rs0;
    double BSIM4v4rd0;
    double BSIM4v4rsw;
    double BSIM4v4rdw;
    double BSIM4v4prwg;
    double BSIM4v4prwb;
    double BSIM4v4prt;
    double BSIM4v4eta0;
    double BSIM4v4etab;
    double BSIM4v4pclm;
    double BSIM4v4pdibl1;
    double BSIM4v4pdibl2;
    double BSIM4v4pdiblb;
    double BSIM4v4fprout;
    double BSIM4v4pdits;
    double BSIM4v4pditsd;
    double BSIM4v4pscbe1;
    double BSIM4v4pscbe2;
    double BSIM4v4pvag;
    double BSIM4v4wr;
    double BSIM4v4dwg;
    double BSIM4v4dwb;
    double BSIM4v4b0;
    double BSIM4v4b1;
    double BSIM4v4alpha0;
    double BSIM4v4alpha1;
    double BSIM4v4beta0;
    double BSIM4v4agidl;
    double BSIM4v4bgidl;
    double BSIM4v4cgidl;
    double BSIM4v4egidl;
    double BSIM4v4aigc;
    double BSIM4v4bigc;
    double BSIM4v4cigc;
    double BSIM4v4aigsd;
    double BSIM4v4bigsd;
    double BSIM4v4cigsd;
    double BSIM4v4aigbacc;
    double BSIM4v4bigbacc;
    double BSIM4v4cigbacc;
    double BSIM4v4aigbinv;
    double BSIM4v4bigbinv;
    double BSIM4v4cigbinv;
    double BSIM4v4nigc;
    double BSIM4v4nigbacc;
    double BSIM4v4nigbinv;
    double BSIM4v4ntox;
    double BSIM4v4eigbinv;
    double BSIM4v4pigcd;
    double BSIM4v4poxedge;
    double BSIM4v4xrcrg1;
    double BSIM4v4xrcrg2;
    double BSIM4v4lambda; /* overshoot */
    double BSIM4v4vtl; /* thermal velocity limit */
    double BSIM4v4xn; /* back scattering parameter */
    double BSIM4v4lc; /* back scattering parameter */
    double BSIM4v4tfactor;  /* ballistic transportation factor  */
    double BSIM4v4vfbsdoff;  /* S/D flatband offset voltage  */

/* added for stress effect */
    double BSIM4v4ku0;
    double BSIM4v4kvth0;
    double BSIM4v4ku0temp;
    double BSIM4v4rho_ref;
    double BSIM4v4inv_od_ref;

    /* CV model */
    double BSIM4v4cgsl;
    double BSIM4v4cgdl;
    double BSIM4v4ckappas;
    double BSIM4v4ckappad;
    double BSIM4v4cf;
    double BSIM4v4clc;
    double BSIM4v4cle;
    double BSIM4v4vfbcv;
    double BSIM4v4noff;
    double BSIM4v4voffcv;
    double BSIM4v4acde;
    double BSIM4v4moin;

/* Pre-calculated constants */

    double BSIM4v4dw;
    double BSIM4v4dl;
    double BSIM4v4leff;
    double BSIM4v4weff;

    double BSIM4v4dwc;
    double BSIM4v4dlc;
    double BSIM4v4dlcig;
    double BSIM4v4dwj;
    double BSIM4v4leffCV;
    double BSIM4v4weffCV;
    double BSIM4v4weffCJ;
    double BSIM4v4abulkCVfactor;
    double BSIM4v4cgso;
    double BSIM4v4cgdo;
    double BSIM4v4cgbo;

    double BSIM4v4u0temp;
    double BSIM4v4vsattemp;
    double BSIM4v4sqrtPhi;
    double BSIM4v4phis3;
    double BSIM4v4Xdep0;
    double BSIM4v4sqrtXdep0;
    double BSIM4v4theta0vb0;
    double BSIM4v4thetaRout;
    double BSIM4v4mstar;
    double BSIM4v4voffcbn;
    double BSIM4v4rdswmin;
    double BSIM4v4rdwmin;
    double BSIM4v4rswmin;
    double BSIM4v4vfbsd;

    double BSIM4v4cof1;
    double BSIM4v4cof2;
    double BSIM4v4cof3;
    double BSIM4v4cof4;
    double BSIM4v4cdep0;
    double BSIM4v4vfbzb;
    double BSIM4v4vtfbphi1;
    double BSIM4v4vtfbphi2;
    double BSIM4v4ToxRatio;
    double BSIM4v4Aechvb;
    double BSIM4v4Bechvb;
    double BSIM4v4ToxRatioEdge;
    double BSIM4v4AechvbEdge;
    double BSIM4v4BechvbEdge;
    double BSIM4v4ldeb;
    double BSIM4v4k1ox;
    double BSIM4v4k2ox;

    struct bsim4SizeDependParam  *pNext;
};


typedef struct sBSIM4v4model
{
    int BSIM4v4modType;
    struct sBSIM4v4model *BSIM4v4nextModel;
    BSIM4v4instance *BSIM4v4instances;
    IFuid BSIM4v4modName;
    int BSIM4v4type;

    int    BSIM4v4mobMod;
    int    BSIM4v4capMod;
    int    BSIM4v4dioMod;
    int    BSIM4v4trnqsMod;
    int    BSIM4v4acnqsMod;
    int    BSIM4v4fnoiMod;
    int    BSIM4v4tnoiMod;
    int    BSIM4v4rdsMod;
    int    BSIM4v4rbodyMod;
    int    BSIM4v4rgateMod;
    int    BSIM4v4perMod;
    int    BSIM4v4geoMod;
    int    BSIM4v4igcMod;
    int    BSIM4v4igbMod;
    int    BSIM4v4tempMod;
    int    BSIM4v4binUnit;
    int    BSIM4v4paramChk;
    char   *BSIM4v4version;
    /* The following field is an integer coding
     * of BSIM4 Versions.
     */
    int    BSIM4v4intVersion;
#define BSIM4v40 440       /* BSIM4.4.0 */
#define BSIM4v30 430       /* BSIM4.3.0 */
#define BSIM4v21 421       /* BSIM4.2.1 */
#define BSIM4vOLD 420     /* < BSIM4.2.1 */
    double BSIM4v4toxe;
    double BSIM4v4toxp;
    double BSIM4v4toxm;
    double BSIM4v4dtox;
    double BSIM4v4epsrox;
    double BSIM4v4cdsc;
    double BSIM4v4cdscb;
    double BSIM4v4cdscd;
    double BSIM4v4cit;
    double BSIM4v4nfactor;
    double BSIM4v4xj;
    double BSIM4v4vsat;
    double BSIM4v4at;
    double BSIM4v4a0;
    double BSIM4v4ags;
    double BSIM4v4a1;
    double BSIM4v4a2;
    double BSIM4v4keta;
    double BSIM4v4nsub;
    double BSIM4v4ndep;
    double BSIM4v4nsd;
    double BSIM4v4phin;
    double BSIM4v4ngate;
    double BSIM4v4gamma1;
    double BSIM4v4gamma2;
    double BSIM4v4vbx;
    double BSIM4v4vbm;
    double BSIM4v4xt;
    double BSIM4v4k1;
    double BSIM4v4kt1;
    double BSIM4v4kt1l;
    double BSIM4v4kt2;
    double BSIM4v4k2;
    double BSIM4v4k3;
    double BSIM4v4k3b;
    double BSIM4v4w0;
    double BSIM4v4dvtp0;
    double BSIM4v4dvtp1;
    double BSIM4v4lpe0;
    double BSIM4v4lpeb;
    double BSIM4v4dvt0;
    double BSIM4v4dvt1;
    double BSIM4v4dvt2;
    double BSIM4v4dvt0w;
    double BSIM4v4dvt1w;
    double BSIM4v4dvt2w;
    double BSIM4v4drout;
    double BSIM4v4dsub;
    double BSIM4v4vth0;
    double BSIM4v4eu;
    double BSIM4v4ua;
    double BSIM4v4ua1;
    double BSIM4v4ub;
    double BSIM4v4ub1;
    double BSIM4v4uc;
    double BSIM4v4uc1;
    double BSIM4v4u0;
    double BSIM4v4ute;
    double BSIM4v4voff;
    double BSIM4v4minv;
    double BSIM4v4voffl;
    double BSIM4v4delta;
    double BSIM4v4rdsw;
    double BSIM4v4rdswmin;
    double BSIM4v4rdwmin;
    double BSIM4v4rswmin;
    double BSIM4v4rsw;
    double BSIM4v4rdw;
    double BSIM4v4prwg;
    double BSIM4v4prwb;
    double BSIM4v4prt;
    double BSIM4v4eta0;
    double BSIM4v4etab;
    double BSIM4v4pclm;
    double BSIM4v4pdibl1;
    double BSIM4v4pdibl2;
    double BSIM4v4pdiblb;
    double BSIM4v4fprout;
    double BSIM4v4pdits;
    double BSIM4v4pditsd;
    double BSIM4v4pditsl;
    double BSIM4v4pscbe1;
    double BSIM4v4pscbe2;
    double BSIM4v4pvag;
    double BSIM4v4wr;
    double BSIM4v4dwg;
    double BSIM4v4dwb;
    double BSIM4v4b0;
    double BSIM4v4b1;
    double BSIM4v4alpha0;
    double BSIM4v4alpha1;
    double BSIM4v4beta0;
    double BSIM4v4agidl;
    double BSIM4v4bgidl;
    double BSIM4v4cgidl;
    double BSIM4v4egidl;
    double BSIM4v4aigc;
    double BSIM4v4bigc;
    double BSIM4v4cigc;
    double BSIM4v4aigsd;
    double BSIM4v4bigsd;
    double BSIM4v4cigsd;
    double BSIM4v4aigbacc;
    double BSIM4v4bigbacc;
    double BSIM4v4cigbacc;
    double BSIM4v4aigbinv;
    double BSIM4v4bigbinv;
    double BSIM4v4cigbinv;
    double BSIM4v4nigc;
    double BSIM4v4nigbacc;
    double BSIM4v4nigbinv;
    double BSIM4v4ntox;
    double BSIM4v4eigbinv;
    double BSIM4v4pigcd;
    double BSIM4v4poxedge;
    double BSIM4v4toxref;
    double BSIM4v4ijthdfwd;
    double BSIM4v4ijthsfwd;
    double BSIM4v4ijthdrev;
    double BSIM4v4ijthsrev;
    double BSIM4v4xjbvd;
    double BSIM4v4xjbvs;
    double BSIM4v4bvd;
    double BSIM4v4bvs;

    double BSIM4v4jtss;
    double BSIM4v4jtsd;
    double BSIM4v4jtssws;
    double BSIM4v4jtsswd;
    double BSIM4v4jtsswgs;
    double BSIM4v4jtsswgd;
    double BSIM4v4njts;
    double BSIM4v4njtssw;
    double BSIM4v4njtsswg;
    double BSIM4v4xtss;
    double BSIM4v4xtsd;
    double BSIM4v4xtssws;
    double BSIM4v4xtsswd;
    double BSIM4v4xtsswgs;
    double BSIM4v4xtsswgd;
    double BSIM4v4tnjts;
    double BSIM4v4tnjtssw;
    double BSIM4v4tnjtsswg;
    double BSIM4v4vtss;
    double BSIM4v4vtsd;
    double BSIM4v4vtssws;
    double BSIM4v4vtsswd;
    double BSIM4v4vtsswgs;
    double BSIM4v4vtsswgd;

    double BSIM4v4xrcrg1;
    double BSIM4v4xrcrg2;
    double BSIM4v4lambda;
    double BSIM4v4vtl;
    double BSIM4v4lc;
    double BSIM4v4xn;
    double BSIM4v4vfbsdoff;  /* S/D flatband offset voltage  */
    double BSIM4v4lintnoi;  /* lint offset for noise calculation  */

    double BSIM4v4vfb;
    double BSIM4v4gbmin;
    double BSIM4v4rbdb;
    double BSIM4v4rbsb;
    double BSIM4v4rbpb;
    double BSIM4v4rbps;
    double BSIM4v4rbpd;
    double BSIM4v4tnoia;
    double BSIM4v4tnoib;
    double BSIM4v4rnoia;
    double BSIM4v4rnoib;
    double BSIM4v4ntnoi;

    /* CV model and Parasitics */
    double BSIM4v4cgsl;
    double BSIM4v4cgdl;
    double BSIM4v4ckappas;
    double BSIM4v4ckappad;
    double BSIM4v4cf;
    double BSIM4v4vfbcv;
    double BSIM4v4clc;
    double BSIM4v4cle;
    double BSIM4v4dwc;
    double BSIM4v4dlc;
    double BSIM4v4xw;
    double BSIM4v4xl;
    double BSIM4v4dlcig;
    double BSIM4v4dwj;
    double BSIM4v4noff;
    double BSIM4v4voffcv;
    double BSIM4v4acde;
    double BSIM4v4moin;
    double BSIM4v4tcj;
    double BSIM4v4tcjsw;
    double BSIM4v4tcjswg;
    double BSIM4v4tpb;
    double BSIM4v4tpbsw;
    double BSIM4v4tpbswg;
    double BSIM4v4dmcg;
    double BSIM4v4dmci;
    double BSIM4v4dmdg;
    double BSIM4v4dmcgt;
    double BSIM4v4xgw;
    double BSIM4v4xgl;
    double BSIM4v4rshg;
    double BSIM4v4ngcon;

    /* Length Dependence */
    double BSIM4v4lcdsc;
    double BSIM4v4lcdscb;
    double BSIM4v4lcdscd;
    double BSIM4v4lcit;
    double BSIM4v4lnfactor;
    double BSIM4v4lxj;
    double BSIM4v4lvsat;
    double BSIM4v4lat;
    double BSIM4v4la0;
    double BSIM4v4lags;
    double BSIM4v4la1;
    double BSIM4v4la2;
    double BSIM4v4lketa;
    double BSIM4v4lnsub;
    double BSIM4v4lndep;
    double BSIM4v4lnsd;
    double BSIM4v4lphin;
    double BSIM4v4lngate;
    double BSIM4v4lgamma1;
    double BSIM4v4lgamma2;
    double BSIM4v4lvbx;
    double BSIM4v4lvbm;
    double BSIM4v4lxt;
    double BSIM4v4lk1;
    double BSIM4v4lkt1;
    double BSIM4v4lkt1l;
    double BSIM4v4lkt2;
    double BSIM4v4lk2;
    double BSIM4v4lk3;
    double BSIM4v4lk3b;
    double BSIM4v4lw0;
    double BSIM4v4ldvtp0;
    double BSIM4v4ldvtp1;
    double BSIM4v4llpe0;
    double BSIM4v4llpeb;
    double BSIM4v4ldvt0;
    double BSIM4v4ldvt1;
    double BSIM4v4ldvt2;
    double BSIM4v4ldvt0w;
    double BSIM4v4ldvt1w;
    double BSIM4v4ldvt2w;
    double BSIM4v4ldrout;
    double BSIM4v4ldsub;
    double BSIM4v4lvth0;
    double BSIM4v4lua;
    double BSIM4v4lua1;
    double BSIM4v4lub;
    double BSIM4v4lub1;
    double BSIM4v4luc;
    double BSIM4v4luc1;
    double BSIM4v4lu0;
    double BSIM4v4leu;
    double BSIM4v4lute;
    double BSIM4v4lvoff;
    double BSIM4v4lminv;
    double BSIM4v4ldelta;
    double BSIM4v4lrdsw;
    double BSIM4v4lrsw;
    double BSIM4v4lrdw;
    double BSIM4v4lprwg;
    double BSIM4v4lprwb;
    double BSIM4v4lprt;
    double BSIM4v4leta0;
    double BSIM4v4letab;
    double BSIM4v4lpclm;
    double BSIM4v4lpdibl1;
    double BSIM4v4lpdibl2;
    double BSIM4v4lpdiblb;
    double BSIM4v4lfprout;
    double BSIM4v4lpdits;
    double BSIM4v4lpditsd;
    double BSIM4v4lpscbe1;
    double BSIM4v4lpscbe2;
    double BSIM4v4lpvag;
    double BSIM4v4lwr;
    double BSIM4v4ldwg;
    double BSIM4v4ldwb;
    double BSIM4v4lb0;
    double BSIM4v4lb1;
    double BSIM4v4lalpha0;
    double BSIM4v4lalpha1;
    double BSIM4v4lbeta0;
    double BSIM4v4lvfb;
    double BSIM4v4lagidl;
    double BSIM4v4lbgidl;
    double BSIM4v4lcgidl;
    double BSIM4v4legidl;
    double BSIM4v4laigc;
    double BSIM4v4lbigc;
    double BSIM4v4lcigc;
    double BSIM4v4laigsd;
    double BSIM4v4lbigsd;
    double BSIM4v4lcigsd;
    double BSIM4v4laigbacc;
    double BSIM4v4lbigbacc;
    double BSIM4v4lcigbacc;
    double BSIM4v4laigbinv;
    double BSIM4v4lbigbinv;
    double BSIM4v4lcigbinv;
    double BSIM4v4lnigc;
    double BSIM4v4lnigbacc;
    double BSIM4v4lnigbinv;
    double BSIM4v4lntox;
    double BSIM4v4leigbinv;
    double BSIM4v4lpigcd;
    double BSIM4v4lpoxedge;
    double BSIM4v4lxrcrg1;
    double BSIM4v4lxrcrg2;
    double BSIM4v4llambda;
    double BSIM4v4lvtl;
    double BSIM4v4lxn;
    double BSIM4v4lvfbsdoff;

    /* CV model */
    double BSIM4v4lcgsl;
    double BSIM4v4lcgdl;
    double BSIM4v4lckappas;
    double BSIM4v4lckappad;
    double BSIM4v4lcf;
    double BSIM4v4lclc;
    double BSIM4v4lcle;
    double BSIM4v4lvfbcv;
    double BSIM4v4lnoff;
    double BSIM4v4lvoffcv;
    double BSIM4v4lacde;
    double BSIM4v4lmoin;

    /* Width Dependence */
    double BSIM4v4wcdsc;
    double BSIM4v4wcdscb;
    double BSIM4v4wcdscd;
    double BSIM4v4wcit;
    double BSIM4v4wnfactor;
    double BSIM4v4wxj;
    double BSIM4v4wvsat;
    double BSIM4v4wat;
    double BSIM4v4wa0;
    double BSIM4v4wags;
    double BSIM4v4wa1;
    double BSIM4v4wa2;
    double BSIM4v4wketa;
    double BSIM4v4wnsub;
    double BSIM4v4wndep;
    double BSIM4v4wnsd;
    double BSIM4v4wphin;
    double BSIM4v4wngate;
    double BSIM4v4wgamma1;
    double BSIM4v4wgamma2;
    double BSIM4v4wvbx;
    double BSIM4v4wvbm;
    double BSIM4v4wxt;
    double BSIM4v4wk1;
    double BSIM4v4wkt1;
    double BSIM4v4wkt1l;
    double BSIM4v4wkt2;
    double BSIM4v4wk2;
    double BSIM4v4wk3;
    double BSIM4v4wk3b;
    double BSIM4v4ww0;
    double BSIM4v4wdvtp0;
    double BSIM4v4wdvtp1;
    double BSIM4v4wlpe0;
    double BSIM4v4wlpeb;
    double BSIM4v4wdvt0;
    double BSIM4v4wdvt1;
    double BSIM4v4wdvt2;
    double BSIM4v4wdvt0w;
    double BSIM4v4wdvt1w;
    double BSIM4v4wdvt2w;
    double BSIM4v4wdrout;
    double BSIM4v4wdsub;
    double BSIM4v4wvth0;
    double BSIM4v4wua;
    double BSIM4v4wua1;
    double BSIM4v4wub;
    double BSIM4v4wub1;
    double BSIM4v4wuc;
    double BSIM4v4wuc1;
    double BSIM4v4wu0;
    double BSIM4v4weu;
    double BSIM4v4wute;
    double BSIM4v4wvoff;
    double BSIM4v4wminv;
    double BSIM4v4wdelta;
    double BSIM4v4wrdsw;
    double BSIM4v4wrsw;
    double BSIM4v4wrdw;
    double BSIM4v4wprwg;
    double BSIM4v4wprwb;
    double BSIM4v4wprt;
    double BSIM4v4weta0;
    double BSIM4v4wetab;
    double BSIM4v4wpclm;
    double BSIM4v4wpdibl1;
    double BSIM4v4wpdibl2;
    double BSIM4v4wpdiblb;
    double BSIM4v4wfprout;
    double BSIM4v4wpdits;
    double BSIM4v4wpditsd;
    double BSIM4v4wpscbe1;
    double BSIM4v4wpscbe2;
    double BSIM4v4wpvag;
    double BSIM4v4wwr;
    double BSIM4v4wdwg;
    double BSIM4v4wdwb;
    double BSIM4v4wb0;
    double BSIM4v4wb1;
    double BSIM4v4walpha0;
    double BSIM4v4walpha1;
    double BSIM4v4wbeta0;
    double BSIM4v4wvfb;
    double BSIM4v4wagidl;
    double BSIM4v4wbgidl;
    double BSIM4v4wcgidl;
    double BSIM4v4wegidl;
    double BSIM4v4waigc;
    double BSIM4v4wbigc;
    double BSIM4v4wcigc;
    double BSIM4v4waigsd;
    double BSIM4v4wbigsd;
    double BSIM4v4wcigsd;
    double BSIM4v4waigbacc;
    double BSIM4v4wbigbacc;
    double BSIM4v4wcigbacc;
    double BSIM4v4waigbinv;
    double BSIM4v4wbigbinv;
    double BSIM4v4wcigbinv;
    double BSIM4v4wnigc;
    double BSIM4v4wnigbacc;
    double BSIM4v4wnigbinv;
    double BSIM4v4wntox;
    double BSIM4v4weigbinv;
    double BSIM4v4wpigcd;
    double BSIM4v4wpoxedge;
    double BSIM4v4wxrcrg1;
    double BSIM4v4wxrcrg2;
    double BSIM4v4wlambda;
    double BSIM4v4wvtl;
    double BSIM4v4wxn;
    double BSIM4v4wvfbsdoff;

    /* CV model */
    double BSIM4v4wcgsl;
    double BSIM4v4wcgdl;
    double BSIM4v4wckappas;
    double BSIM4v4wckappad;
    double BSIM4v4wcf;
    double BSIM4v4wclc;
    double BSIM4v4wcle;
    double BSIM4v4wvfbcv;
    double BSIM4v4wnoff;
    double BSIM4v4wvoffcv;
    double BSIM4v4wacde;
    double BSIM4v4wmoin;

    /* Cross-term Dependence */
    double BSIM4v4pcdsc;
    double BSIM4v4pcdscb;
    double BSIM4v4pcdscd;
    double BSIM4v4pcit;
    double BSIM4v4pnfactor;
    double BSIM4v4pxj;
    double BSIM4v4pvsat;
    double BSIM4v4pat;
    double BSIM4v4pa0;
    double BSIM4v4pags;
    double BSIM4v4pa1;
    double BSIM4v4pa2;
    double BSIM4v4pketa;
    double BSIM4v4pnsub;
    double BSIM4v4pndep;
    double BSIM4v4pnsd;
    double BSIM4v4pphin;
    double BSIM4v4pngate;
    double BSIM4v4pgamma1;
    double BSIM4v4pgamma2;
    double BSIM4v4pvbx;
    double BSIM4v4pvbm;
    double BSIM4v4pxt;
    double BSIM4v4pk1;
    double BSIM4v4pkt1;
    double BSIM4v4pkt1l;
    double BSIM4v4pkt2;
    double BSIM4v4pk2;
    double BSIM4v4pk3;
    double BSIM4v4pk3b;
    double BSIM4v4pw0;
    double BSIM4v4pdvtp0;
    double BSIM4v4pdvtp1;
    double BSIM4v4plpe0;
    double BSIM4v4plpeb;
    double BSIM4v4pdvt0;
    double BSIM4v4pdvt1;
    double BSIM4v4pdvt2;
    double BSIM4v4pdvt0w;
    double BSIM4v4pdvt1w;
    double BSIM4v4pdvt2w;
    double BSIM4v4pdrout;
    double BSIM4v4pdsub;
    double BSIM4v4pvth0;
    double BSIM4v4pua;
    double BSIM4v4pua1;
    double BSIM4v4pub;
    double BSIM4v4pub1;
    double BSIM4v4puc;
    double BSIM4v4puc1;
    double BSIM4v4pu0;
    double BSIM4v4peu;
    double BSIM4v4pute;
    double BSIM4v4pvoff;
    double BSIM4v4pminv;
    double BSIM4v4pdelta;
    double BSIM4v4prdsw;
    double BSIM4v4prsw;
    double BSIM4v4prdw;
    double BSIM4v4pprwg;
    double BSIM4v4pprwb;
    double BSIM4v4pprt;
    double BSIM4v4peta0;
    double BSIM4v4petab;
    double BSIM4v4ppclm;
    double BSIM4v4ppdibl1;
    double BSIM4v4ppdibl2;
    double BSIM4v4ppdiblb;
    double BSIM4v4pfprout;
    double BSIM4v4ppdits;
    double BSIM4v4ppditsd;
    double BSIM4v4ppscbe1;
    double BSIM4v4ppscbe2;
    double BSIM4v4ppvag;
    double BSIM4v4pwr;
    double BSIM4v4pdwg;
    double BSIM4v4pdwb;
    double BSIM4v4pb0;
    double BSIM4v4pb1;
    double BSIM4v4palpha0;
    double BSIM4v4palpha1;
    double BSIM4v4pbeta0;
    double BSIM4v4pvfb;
    double BSIM4v4pagidl;
    double BSIM4v4pbgidl;
    double BSIM4v4pcgidl;
    double BSIM4v4pegidl;
    double BSIM4v4paigc;
    double BSIM4v4pbigc;
    double BSIM4v4pcigc;
    double BSIM4v4paigsd;
    double BSIM4v4pbigsd;
    double BSIM4v4pcigsd;
    double BSIM4v4paigbacc;
    double BSIM4v4pbigbacc;
    double BSIM4v4pcigbacc;
    double BSIM4v4paigbinv;
    double BSIM4v4pbigbinv;
    double BSIM4v4pcigbinv;
    double BSIM4v4pnigc;
    double BSIM4v4pnigbacc;
    double BSIM4v4pnigbinv;
    double BSIM4v4pntox;
    double BSIM4v4peigbinv;
    double BSIM4v4ppigcd;
    double BSIM4v4ppoxedge;
    double BSIM4v4pxrcrg1;
    double BSIM4v4pxrcrg2;
    double BSIM4v4plambda;
    double BSIM4v4pvtl;
    double BSIM4v4pxn;
    double BSIM4v4pvfbsdoff;

    /* CV model */
    double BSIM4v4pcgsl;
    double BSIM4v4pcgdl;
    double BSIM4v4pckappas;
    double BSIM4v4pckappad;
    double BSIM4v4pcf;
    double BSIM4v4pclc;
    double BSIM4v4pcle;
    double BSIM4v4pvfbcv;
    double BSIM4v4pnoff;
    double BSIM4v4pvoffcv;
    double BSIM4v4pacde;
    double BSIM4v4pmoin;

    double BSIM4v4tnom;
    double BSIM4v4cgso;
    double BSIM4v4cgdo;
    double BSIM4v4cgbo;
    double BSIM4v4xpart;
    double BSIM4v4cFringOut;
    double BSIM4v4cFringMax;

    double BSIM4v4sheetResistance;
    double BSIM4v4SjctSatCurDensity;
    double BSIM4v4DjctSatCurDensity;
    double BSIM4v4SjctSidewallSatCurDensity;
    double BSIM4v4DjctSidewallSatCurDensity;
    double BSIM4v4SjctGateSidewallSatCurDensity;
    double BSIM4v4DjctGateSidewallSatCurDensity;
    double BSIM4v4SbulkJctPotential;
    double BSIM4v4DbulkJctPotential;
    double BSIM4v4SbulkJctBotGradingCoeff;
    double BSIM4v4DbulkJctBotGradingCoeff;
    double BSIM4v4SbulkJctSideGradingCoeff;
    double BSIM4v4DbulkJctSideGradingCoeff;
    double BSIM4v4SbulkJctGateSideGradingCoeff;
    double BSIM4v4DbulkJctGateSideGradingCoeff;
    double BSIM4v4SsidewallJctPotential;
    double BSIM4v4DsidewallJctPotential;
    double BSIM4v4SGatesidewallJctPotential;
    double BSIM4v4DGatesidewallJctPotential;
    double BSIM4v4SunitAreaJctCap;
    double BSIM4v4DunitAreaJctCap;
    double BSIM4v4SunitLengthSidewallJctCap;
    double BSIM4v4DunitLengthSidewallJctCap;
    double BSIM4v4SunitLengthGateSidewallJctCap;
    double BSIM4v4DunitLengthGateSidewallJctCap;
    double BSIM4v4SjctEmissionCoeff;
    double BSIM4v4DjctEmissionCoeff;
    double BSIM4v4SjctTempExponent;
    double BSIM4v4DjctTempExponent;
    double BSIM4v4njtstemp;
    double BSIM4v4njtsswtemp;
    double BSIM4v4njtsswgtemp;

    double BSIM4v4Lint;
    double BSIM4v4Ll;
    double BSIM4v4Llc;
    double BSIM4v4Lln;
    double BSIM4v4Lw;
    double BSIM4v4Lwc;
    double BSIM4v4Lwn;
    double BSIM4v4Lwl;
    double BSIM4v4Lwlc;
    double BSIM4v4Lmin;
    double BSIM4v4Lmax;

    double BSIM4v4Wint;
    double BSIM4v4Wl;
    double BSIM4v4Wlc;
    double BSIM4v4Wln;
    double BSIM4v4Ww;
    double BSIM4v4Wwc;
    double BSIM4v4Wwn;
    double BSIM4v4Wwl;
    double BSIM4v4Wwlc;
    double BSIM4v4Wmin;
    double BSIM4v4Wmax;

    /* added for stress effect */
    double BSIM4v4saref;
    double BSIM4v4sbref;
    double BSIM4v4wlod;
    double BSIM4v4ku0;
    double BSIM4v4kvsat;
    double BSIM4v4kvth0;
    double BSIM4v4tku0;
    double BSIM4v4llodku0;
    double BSIM4v4wlodku0;
    double BSIM4v4llodvth;
    double BSIM4v4wlodvth;
    double BSIM4v4lku0;
    double BSIM4v4wku0;
    double BSIM4v4pku0;
    double BSIM4v4lkvth0;
    double BSIM4v4wkvth0;
    double BSIM4v4pkvth0;
    double BSIM4v4stk2;
    double BSIM4v4lodk2;
    double BSIM4v4steta0;
    double BSIM4v4lodeta0;



/* Pre-calculated constants
 * move to size-dependent param */
    double BSIM4v4vtm;
    double BSIM4v4vtm0;
    double BSIM4v4coxe;
    double BSIM4v4coxp;
    double BSIM4v4cof1;
    double BSIM4v4cof2;
    double BSIM4v4cof3;
    double BSIM4v4cof4;
    double BSIM4v4vcrit;
    double BSIM4v4factor1;
    double BSIM4v4PhiBS;
    double BSIM4v4PhiBSWS;
    double BSIM4v4PhiBSWGS;
    double BSIM4v4SjctTempSatCurDensity;
    double BSIM4v4SjctSidewallTempSatCurDensity;
    double BSIM4v4SjctGateSidewallTempSatCurDensity;
    double BSIM4v4PhiBD;
    double BSIM4v4PhiBSWD;
    double BSIM4v4PhiBSWGD;
    double BSIM4v4DjctTempSatCurDensity;
    double BSIM4v4DjctSidewallTempSatCurDensity;
    double BSIM4v4DjctGateSidewallTempSatCurDensity;
    double BSIM4v4SunitAreaTempJctCap;
    double BSIM4v4DunitAreaTempJctCap;
    double BSIM4v4SunitLengthSidewallTempJctCap;
    double BSIM4v4DunitLengthSidewallTempJctCap;
    double BSIM4v4SunitLengthGateSidewallTempJctCap;
    double BSIM4v4DunitLengthGateSidewallTempJctCap;

    double BSIM4v4oxideTrapDensityA;
    double BSIM4v4oxideTrapDensityB;
    double BSIM4v4oxideTrapDensityC;
    double BSIM4v4em;
    double BSIM4v4ef;
    double BSIM4v4af;
    double BSIM4v4kf;

    struct bsim4SizeDependParam *pSizeDependParamKnot;

    /* Flags */
    unsigned  BSIM4v4mobModGiven :1;
    unsigned  BSIM4v4binUnitGiven :1;
    unsigned  BSIM4v4capModGiven :1;
    unsigned  BSIM4v4dioModGiven :1;
    unsigned  BSIM4v4rdsModGiven :1;
    unsigned  BSIM4v4rbodyModGiven :1;
    unsigned  BSIM4v4rgateModGiven :1;
    unsigned  BSIM4v4perModGiven :1;
    unsigned  BSIM4v4geoModGiven :1;
    unsigned  BSIM4v4paramChkGiven :1;
    unsigned  BSIM4v4trnqsModGiven :1;
    unsigned  BSIM4v4acnqsModGiven :1;
    unsigned  BSIM4v4fnoiModGiven :1;
    unsigned  BSIM4v4tnoiModGiven :1;
    unsigned  BSIM4v4igcModGiven :1;
    unsigned  BSIM4v4igbModGiven :1;
    unsigned  BSIM4v4tempModGiven :1;
    unsigned  BSIM4v4typeGiven   :1;
    unsigned  BSIM4v4toxrefGiven   :1;
    unsigned  BSIM4v4toxeGiven   :1;
    unsigned  BSIM4v4toxpGiven   :1;
    unsigned  BSIM4v4toxmGiven   :1;
    unsigned  BSIM4v4dtoxGiven   :1;
    unsigned  BSIM4v4epsroxGiven   :1;
    unsigned  BSIM4v4versionGiven   :1;
    unsigned  BSIM4v4cdscGiven   :1;
    unsigned  BSIM4v4cdscbGiven   :1;
    unsigned  BSIM4v4cdscdGiven   :1;
    unsigned  BSIM4v4citGiven   :1;
    unsigned  BSIM4v4nfactorGiven   :1;
    unsigned  BSIM4v4xjGiven   :1;
    unsigned  BSIM4v4vsatGiven   :1;
    unsigned  BSIM4v4atGiven   :1;
    unsigned  BSIM4v4a0Given   :1;
    unsigned  BSIM4v4agsGiven   :1;
    unsigned  BSIM4v4a1Given   :1;
    unsigned  BSIM4v4a2Given   :1;
    unsigned  BSIM4v4ketaGiven   :1;
    unsigned  BSIM4v4nsubGiven   :1;
    unsigned  BSIM4v4ndepGiven   :1;
    unsigned  BSIM4v4nsdGiven    :1;
    unsigned  BSIM4v4phinGiven   :1;
    unsigned  BSIM4v4ngateGiven   :1;
    unsigned  BSIM4v4gamma1Given   :1;
    unsigned  BSIM4v4gamma2Given   :1;
    unsigned  BSIM4v4vbxGiven   :1;
    unsigned  BSIM4v4vbmGiven   :1;
    unsigned  BSIM4v4xtGiven   :1;
    unsigned  BSIM4v4k1Given   :1;
    unsigned  BSIM4v4kt1Given   :1;
    unsigned  BSIM4v4kt1lGiven   :1;
    unsigned  BSIM4v4kt2Given   :1;
    unsigned  BSIM4v4k2Given   :1;
    unsigned  BSIM4v4k3Given   :1;
    unsigned  BSIM4v4k3bGiven   :1;
    unsigned  BSIM4v4w0Given   :1;
    unsigned  BSIM4v4dvtp0Given :1;
    unsigned  BSIM4v4dvtp1Given :1;
    unsigned  BSIM4v4lpe0Given   :1;
    unsigned  BSIM4v4lpebGiven   :1;
    unsigned  BSIM4v4dvt0Given   :1;
    unsigned  BSIM4v4dvt1Given   :1;
    unsigned  BSIM4v4dvt2Given   :1;
    unsigned  BSIM4v4dvt0wGiven   :1;
    unsigned  BSIM4v4dvt1wGiven   :1;
    unsigned  BSIM4v4dvt2wGiven   :1;
    unsigned  BSIM4v4droutGiven   :1;
    unsigned  BSIM4v4dsubGiven   :1;
    unsigned  BSIM4v4vth0Given   :1;
    unsigned  BSIM4v4euGiven   :1;
    unsigned  BSIM4v4uaGiven   :1;
    unsigned  BSIM4v4ua1Given   :1;
    unsigned  BSIM4v4ubGiven   :1;
    unsigned  BSIM4v4ub1Given   :1;
    unsigned  BSIM4v4ucGiven   :1;
    unsigned  BSIM4v4uc1Given   :1;
    unsigned  BSIM4v4u0Given   :1;
    unsigned  BSIM4v4uteGiven   :1;
    unsigned  BSIM4v4voffGiven   :1;
    unsigned  BSIM4v4vofflGiven  :1;
    unsigned  BSIM4v4minvGiven   :1;
    unsigned  BSIM4v4rdswGiven   :1;
    unsigned  BSIM4v4rdswminGiven :1;
    unsigned  BSIM4v4rdwminGiven :1;
    unsigned  BSIM4v4rswminGiven :1;
    unsigned  BSIM4v4rswGiven   :1;
    unsigned  BSIM4v4rdwGiven   :1;
    unsigned  BSIM4v4prwgGiven   :1;
    unsigned  BSIM4v4prwbGiven   :1;
    unsigned  BSIM4v4prtGiven   :1;
    unsigned  BSIM4v4eta0Given   :1;
    unsigned  BSIM4v4etabGiven   :1;
    unsigned  BSIM4v4pclmGiven   :1;
    unsigned  BSIM4v4pdibl1Given   :1;
    unsigned  BSIM4v4pdibl2Given   :1;
    unsigned  BSIM4v4pdiblbGiven   :1;
    unsigned  BSIM4v4fproutGiven   :1;
    unsigned  BSIM4v4pditsGiven    :1;
    unsigned  BSIM4v4pditsdGiven    :1;
    unsigned  BSIM4v4pditslGiven    :1;
    unsigned  BSIM4v4pscbe1Given   :1;
    unsigned  BSIM4v4pscbe2Given   :1;
    unsigned  BSIM4v4pvagGiven   :1;
    unsigned  BSIM4v4deltaGiven  :1;
    unsigned  BSIM4v4wrGiven   :1;
    unsigned  BSIM4v4dwgGiven   :1;
    unsigned  BSIM4v4dwbGiven   :1;
    unsigned  BSIM4v4b0Given   :1;
    unsigned  BSIM4v4b1Given   :1;
    unsigned  BSIM4v4alpha0Given   :1;
    unsigned  BSIM4v4alpha1Given   :1;
    unsigned  BSIM4v4beta0Given   :1;
    unsigned  BSIM4v4agidlGiven   :1;
    unsigned  BSIM4v4bgidlGiven   :1;
    unsigned  BSIM4v4cgidlGiven   :1;
    unsigned  BSIM4v4egidlGiven   :1;
    unsigned  BSIM4v4aigcGiven   :1;
    unsigned  BSIM4v4bigcGiven   :1;
    unsigned  BSIM4v4cigcGiven   :1;
    unsigned  BSIM4v4aigsdGiven   :1;
    unsigned  BSIM4v4bigsdGiven   :1;
    unsigned  BSIM4v4cigsdGiven   :1;
    unsigned  BSIM4v4aigbaccGiven   :1;
    unsigned  BSIM4v4bigbaccGiven   :1;
    unsigned  BSIM4v4cigbaccGiven   :1;
    unsigned  BSIM4v4aigbinvGiven   :1;
    unsigned  BSIM4v4bigbinvGiven   :1;
    unsigned  BSIM4v4cigbinvGiven   :1;
    unsigned  BSIM4v4nigcGiven   :1;
    unsigned  BSIM4v4nigbinvGiven   :1;
    unsigned  BSIM4v4nigbaccGiven   :1;
    unsigned  BSIM4v4ntoxGiven   :1;
    unsigned  BSIM4v4eigbinvGiven   :1;
    unsigned  BSIM4v4pigcdGiven   :1;
    unsigned  BSIM4v4poxedgeGiven   :1;
    unsigned  BSIM4v4ijthdfwdGiven  :1;
    unsigned  BSIM4v4ijthsfwdGiven  :1;
    unsigned  BSIM4v4ijthdrevGiven  :1;
    unsigned  BSIM4v4ijthsrevGiven  :1;
    unsigned  BSIM4v4xjbvdGiven   :1;
    unsigned  BSIM4v4xjbvsGiven   :1;
    unsigned  BSIM4v4bvdGiven   :1;
    unsigned  BSIM4v4bvsGiven   :1;

    unsigned  BSIM4v4jtssGiven   :1;
    unsigned  BSIM4v4jtsdGiven   :1;
    unsigned  BSIM4v4jtsswsGiven   :1;
    unsigned  BSIM4v4jtsswdGiven   :1;
    unsigned  BSIM4v4jtsswgsGiven   :1;
    unsigned  BSIM4v4jtsswgdGiven   :1;
    unsigned  BSIM4v4njtsGiven   :1;
    unsigned  BSIM4v4njtsswGiven   :1;
    unsigned  BSIM4v4njtsswgGiven   :1;
    unsigned  BSIM4v4xtssGiven   :1;
    unsigned  BSIM4v4xtsdGiven   :1;
    unsigned  BSIM4v4xtsswsGiven   :1;
    unsigned  BSIM4v4xtsswdGiven   :1;
    unsigned  BSIM4v4xtsswgsGiven   :1;
    unsigned  BSIM4v4xtsswgdGiven   :1;
    unsigned  BSIM4v4tnjtsGiven   :1;
    unsigned  BSIM4v4tnjtsswGiven   :1;
    unsigned  BSIM4v4tnjtsswgGiven   :1;
    unsigned  BSIM4v4vtssGiven   :1;
    unsigned  BSIM4v4vtsdGiven   :1;
    unsigned  BSIM4v4vtsswsGiven   :1;
    unsigned  BSIM4v4vtsswdGiven   :1;
    unsigned  BSIM4v4vtsswgsGiven   :1;
    unsigned  BSIM4v4vtsswgdGiven   :1;

    unsigned  BSIM4v4vfbGiven   :1;
    unsigned  BSIM4v4gbminGiven :1;
    unsigned  BSIM4v4rbdbGiven :1;
    unsigned  BSIM4v4rbsbGiven :1;
    unsigned  BSIM4v4rbpsGiven :1;
    unsigned  BSIM4v4rbpdGiven :1;
    unsigned  BSIM4v4rbpbGiven :1;
    unsigned  BSIM4v4xrcrg1Given   :1;
    unsigned  BSIM4v4xrcrg2Given   :1;
    unsigned  BSIM4v4tnoiaGiven    :1;
    unsigned  BSIM4v4tnoibGiven    :1;
    unsigned  BSIM4v4rnoiaGiven    :1;
    unsigned  BSIM4v4rnoibGiven    :1;
    unsigned  BSIM4v4ntnoiGiven    :1;

    unsigned  BSIM4v4lambdaGiven    :1;
    unsigned  BSIM4v4vtlGiven    :1;
    unsigned  BSIM4v4lcGiven    :1;
    unsigned  BSIM4v4xnGiven    :1;
    unsigned  BSIM4v4vfbsdoffGiven    :1;
    unsigned  BSIM4v4lintnoiGiven    :1;

    /* CV model and parasitics */
    unsigned  BSIM4v4cgslGiven   :1;
    unsigned  BSIM4v4cgdlGiven   :1;
    unsigned  BSIM4v4ckappasGiven   :1;
    unsigned  BSIM4v4ckappadGiven   :1;
    unsigned  BSIM4v4cfGiven   :1;
    unsigned  BSIM4v4vfbcvGiven   :1;
    unsigned  BSIM4v4clcGiven   :1;
    unsigned  BSIM4v4cleGiven   :1;
    unsigned  BSIM4v4dwcGiven   :1;
    unsigned  BSIM4v4dlcGiven   :1;
    unsigned  BSIM4v4xwGiven    :1;
    unsigned  BSIM4v4xlGiven    :1;
    unsigned  BSIM4v4dlcigGiven   :1;
    unsigned  BSIM4v4dwjGiven   :1;
    unsigned  BSIM4v4noffGiven  :1;
    unsigned  BSIM4v4voffcvGiven :1;
    unsigned  BSIM4v4acdeGiven  :1;
    unsigned  BSIM4v4moinGiven  :1;
    unsigned  BSIM4v4tcjGiven   :1;
    unsigned  BSIM4v4tcjswGiven :1;
    unsigned  BSIM4v4tcjswgGiven :1;
    unsigned  BSIM4v4tpbGiven    :1;
    unsigned  BSIM4v4tpbswGiven  :1;
    unsigned  BSIM4v4tpbswgGiven :1;
    unsigned  BSIM4v4dmcgGiven :1;
    unsigned  BSIM4v4dmciGiven :1;
    unsigned  BSIM4v4dmdgGiven :1;
    unsigned  BSIM4v4dmcgtGiven :1;
    unsigned  BSIM4v4xgwGiven :1;
    unsigned  BSIM4v4xglGiven :1;
    unsigned  BSIM4v4rshgGiven :1;
    unsigned  BSIM4v4ngconGiven :1;


    /* Length dependence */
    unsigned  BSIM4v4lcdscGiven   :1;
    unsigned  BSIM4v4lcdscbGiven   :1;
    unsigned  BSIM4v4lcdscdGiven   :1;
    unsigned  BSIM4v4lcitGiven   :1;
    unsigned  BSIM4v4lnfactorGiven   :1;
    unsigned  BSIM4v4lxjGiven   :1;
    unsigned  BSIM4v4lvsatGiven   :1;
    unsigned  BSIM4v4latGiven   :1;
    unsigned  BSIM4v4la0Given   :1;
    unsigned  BSIM4v4lagsGiven   :1;
    unsigned  BSIM4v4la1Given   :1;
    unsigned  BSIM4v4la2Given   :1;
    unsigned  BSIM4v4lketaGiven   :1;
    unsigned  BSIM4v4lnsubGiven   :1;
    unsigned  BSIM4v4lndepGiven   :1;
    unsigned  BSIM4v4lnsdGiven    :1;
    unsigned  BSIM4v4lphinGiven   :1;
    unsigned  BSIM4v4lngateGiven   :1;
    unsigned  BSIM4v4lgamma1Given   :1;
    unsigned  BSIM4v4lgamma2Given   :1;
    unsigned  BSIM4v4lvbxGiven   :1;
    unsigned  BSIM4v4lvbmGiven   :1;
    unsigned  BSIM4v4lxtGiven   :1;
    unsigned  BSIM4v4lk1Given   :1;
    unsigned  BSIM4v4lkt1Given   :1;
    unsigned  BSIM4v4lkt1lGiven   :1;
    unsigned  BSIM4v4lkt2Given   :1;
    unsigned  BSIM4v4lk2Given   :1;
    unsigned  BSIM4v4lk3Given   :1;
    unsigned  BSIM4v4lk3bGiven   :1;
    unsigned  BSIM4v4lw0Given   :1;
    unsigned  BSIM4v4ldvtp0Given :1;
    unsigned  BSIM4v4ldvtp1Given :1;
    unsigned  BSIM4v4llpe0Given   :1;
    unsigned  BSIM4v4llpebGiven   :1;
    unsigned  BSIM4v4ldvt0Given   :1;
    unsigned  BSIM4v4ldvt1Given   :1;
    unsigned  BSIM4v4ldvt2Given   :1;
    unsigned  BSIM4v4ldvt0wGiven   :1;
    unsigned  BSIM4v4ldvt1wGiven   :1;
    unsigned  BSIM4v4ldvt2wGiven   :1;
    unsigned  BSIM4v4ldroutGiven   :1;
    unsigned  BSIM4v4ldsubGiven   :1;
    unsigned  BSIM4v4lvth0Given   :1;
    unsigned  BSIM4v4luaGiven   :1;
    unsigned  BSIM4v4lua1Given   :1;
    unsigned  BSIM4v4lubGiven   :1;
    unsigned  BSIM4v4lub1Given   :1;
    unsigned  BSIM4v4lucGiven   :1;
    unsigned  BSIM4v4luc1Given   :1;
    unsigned  BSIM4v4lu0Given   :1;
    unsigned  BSIM4v4leuGiven   :1;
    unsigned  BSIM4v4luteGiven   :1;
    unsigned  BSIM4v4lvoffGiven   :1;
    unsigned  BSIM4v4lminvGiven   :1;
    unsigned  BSIM4v4lrdswGiven   :1;
    unsigned  BSIM4v4lrswGiven   :1;
    unsigned  BSIM4v4lrdwGiven   :1;
    unsigned  BSIM4v4lprwgGiven   :1;
    unsigned  BSIM4v4lprwbGiven   :1;
    unsigned  BSIM4v4lprtGiven   :1;
    unsigned  BSIM4v4leta0Given   :1;
    unsigned  BSIM4v4letabGiven   :1;
    unsigned  BSIM4v4lpclmGiven   :1;
    unsigned  BSIM4v4lpdibl1Given   :1;
    unsigned  BSIM4v4lpdibl2Given   :1;
    unsigned  BSIM4v4lpdiblbGiven   :1;
    unsigned  BSIM4v4lfproutGiven   :1;
    unsigned  BSIM4v4lpditsGiven    :1;
    unsigned  BSIM4v4lpditsdGiven    :1;
    unsigned  BSIM4v4lpscbe1Given   :1;
    unsigned  BSIM4v4lpscbe2Given   :1;
    unsigned  BSIM4v4lpvagGiven   :1;
    unsigned  BSIM4v4ldeltaGiven  :1;
    unsigned  BSIM4v4lwrGiven   :1;
    unsigned  BSIM4v4ldwgGiven   :1;
    unsigned  BSIM4v4ldwbGiven   :1;
    unsigned  BSIM4v4lb0Given   :1;
    unsigned  BSIM4v4lb1Given   :1;
    unsigned  BSIM4v4lalpha0Given   :1;
    unsigned  BSIM4v4lalpha1Given   :1;
    unsigned  BSIM4v4lbeta0Given   :1;
    unsigned  BSIM4v4lvfbGiven   :1;
    unsigned  BSIM4v4lagidlGiven   :1;
    unsigned  BSIM4v4lbgidlGiven   :1;
    unsigned  BSIM4v4lcgidlGiven   :1;
    unsigned  BSIM4v4legidlGiven   :1;
    unsigned  BSIM4v4laigcGiven   :1;
    unsigned  BSIM4v4lbigcGiven   :1;
    unsigned  BSIM4v4lcigcGiven   :1;
    unsigned  BSIM4v4laigsdGiven   :1;
    unsigned  BSIM4v4lbigsdGiven   :1;
    unsigned  BSIM4v4lcigsdGiven   :1;
    unsigned  BSIM4v4laigbaccGiven   :1;
    unsigned  BSIM4v4lbigbaccGiven   :1;
    unsigned  BSIM4v4lcigbaccGiven   :1;
    unsigned  BSIM4v4laigbinvGiven   :1;
    unsigned  BSIM4v4lbigbinvGiven   :1;
    unsigned  BSIM4v4lcigbinvGiven   :1;
    unsigned  BSIM4v4lnigcGiven   :1;
    unsigned  BSIM4v4lnigbinvGiven   :1;
    unsigned  BSIM4v4lnigbaccGiven   :1;
    unsigned  BSIM4v4lntoxGiven   :1;
    unsigned  BSIM4v4leigbinvGiven   :1;
    unsigned  BSIM4v4lpigcdGiven   :1;
    unsigned  BSIM4v4lpoxedgeGiven   :1;
    unsigned  BSIM4v4lxrcrg1Given   :1;
    unsigned  BSIM4v4lxrcrg2Given   :1;
    unsigned  BSIM4v4llambdaGiven    :1;
    unsigned  BSIM4v4lvtlGiven    :1;
    unsigned  BSIM4v4lxnGiven    :1;
    unsigned  BSIM4v4lvfbsdoffGiven    :1;

    /* CV model */
    unsigned  BSIM4v4lcgslGiven   :1;
    unsigned  BSIM4v4lcgdlGiven   :1;
    unsigned  BSIM4v4lckappasGiven   :1;
    unsigned  BSIM4v4lckappadGiven   :1;
    unsigned  BSIM4v4lcfGiven   :1;
    unsigned  BSIM4v4lclcGiven   :1;
    unsigned  BSIM4v4lcleGiven   :1;
    unsigned  BSIM4v4lvfbcvGiven   :1;
    unsigned  BSIM4v4lnoffGiven   :1;
    unsigned  BSIM4v4lvoffcvGiven :1;
    unsigned  BSIM4v4lacdeGiven   :1;
    unsigned  BSIM4v4lmoinGiven   :1;

    /* Width dependence */
    unsigned  BSIM4v4wcdscGiven   :1;
    unsigned  BSIM4v4wcdscbGiven   :1;
    unsigned  BSIM4v4wcdscdGiven   :1;
    unsigned  BSIM4v4wcitGiven   :1;
    unsigned  BSIM4v4wnfactorGiven   :1;
    unsigned  BSIM4v4wxjGiven   :1;
    unsigned  BSIM4v4wvsatGiven   :1;
    unsigned  BSIM4v4watGiven   :1;
    unsigned  BSIM4v4wa0Given   :1;
    unsigned  BSIM4v4wagsGiven   :1;
    unsigned  BSIM4v4wa1Given   :1;
    unsigned  BSIM4v4wa2Given   :1;
    unsigned  BSIM4v4wketaGiven   :1;
    unsigned  BSIM4v4wnsubGiven   :1;
    unsigned  BSIM4v4wndepGiven   :1;
    unsigned  BSIM4v4wnsdGiven    :1;
    unsigned  BSIM4v4wphinGiven   :1;
    unsigned  BSIM4v4wngateGiven   :1;
    unsigned  BSIM4v4wgamma1Given   :1;
    unsigned  BSIM4v4wgamma2Given   :1;
    unsigned  BSIM4v4wvbxGiven   :1;
    unsigned  BSIM4v4wvbmGiven   :1;
    unsigned  BSIM4v4wxtGiven   :1;
    unsigned  BSIM4v4wk1Given   :1;
    unsigned  BSIM4v4wkt1Given   :1;
    unsigned  BSIM4v4wkt1lGiven   :1;
    unsigned  BSIM4v4wkt2Given   :1;
    unsigned  BSIM4v4wk2Given   :1;
    unsigned  BSIM4v4wk3Given   :1;
    unsigned  BSIM4v4wk3bGiven   :1;
    unsigned  BSIM4v4ww0Given   :1;
    unsigned  BSIM4v4wdvtp0Given :1;
    unsigned  BSIM4v4wdvtp1Given :1;
    unsigned  BSIM4v4wlpe0Given   :1;
    unsigned  BSIM4v4wlpebGiven   :1;
    unsigned  BSIM4v4wdvt0Given   :1;
    unsigned  BSIM4v4wdvt1Given   :1;
    unsigned  BSIM4v4wdvt2Given   :1;
    unsigned  BSIM4v4wdvt0wGiven   :1;
    unsigned  BSIM4v4wdvt1wGiven   :1;
    unsigned  BSIM4v4wdvt2wGiven   :1;
    unsigned  BSIM4v4wdroutGiven   :1;
    unsigned  BSIM4v4wdsubGiven   :1;
    unsigned  BSIM4v4wvth0Given   :1;
    unsigned  BSIM4v4wuaGiven   :1;
    unsigned  BSIM4v4wua1Given   :1;
    unsigned  BSIM4v4wubGiven   :1;
    unsigned  BSIM4v4wub1Given   :1;
    unsigned  BSIM4v4wucGiven   :1;
    unsigned  BSIM4v4wuc1Given   :1;
    unsigned  BSIM4v4wu0Given   :1;
    unsigned  BSIM4v4weuGiven   :1;
    unsigned  BSIM4v4wuteGiven   :1;
    unsigned  BSIM4v4wvoffGiven   :1;
    unsigned  BSIM4v4wminvGiven   :1;
    unsigned  BSIM4v4wrdswGiven   :1;
    unsigned  BSIM4v4wrswGiven   :1;
    unsigned  BSIM4v4wrdwGiven   :1;
    unsigned  BSIM4v4wprwgGiven   :1;
    unsigned  BSIM4v4wprwbGiven   :1;
    unsigned  BSIM4v4wprtGiven   :1;
    unsigned  BSIM4v4weta0Given   :1;
    unsigned  BSIM4v4wetabGiven   :1;
    unsigned  BSIM4v4wpclmGiven   :1;
    unsigned  BSIM4v4wpdibl1Given   :1;
    unsigned  BSIM4v4wpdibl2Given   :1;
    unsigned  BSIM4v4wpdiblbGiven   :1;
    unsigned  BSIM4v4wfproutGiven   :1;
    unsigned  BSIM4v4wpditsGiven    :1;
    unsigned  BSIM4v4wpditsdGiven    :1;
    unsigned  BSIM4v4wpscbe1Given   :1;
    unsigned  BSIM4v4wpscbe2Given   :1;
    unsigned  BSIM4v4wpvagGiven   :1;
    unsigned  BSIM4v4wdeltaGiven  :1;
    unsigned  BSIM4v4wwrGiven   :1;
    unsigned  BSIM4v4wdwgGiven   :1;
    unsigned  BSIM4v4wdwbGiven   :1;
    unsigned  BSIM4v4wb0Given   :1;
    unsigned  BSIM4v4wb1Given   :1;
    unsigned  BSIM4v4walpha0Given   :1;
    unsigned  BSIM4v4walpha1Given   :1;
    unsigned  BSIM4v4wbeta0Given   :1;
    unsigned  BSIM4v4wvfbGiven   :1;
    unsigned  BSIM4v4wagidlGiven   :1;
    unsigned  BSIM4v4wbgidlGiven   :1;
    unsigned  BSIM4v4wcgidlGiven   :1;
    unsigned  BSIM4v4wegidlGiven   :1;
    unsigned  BSIM4v4waigcGiven   :1;
    unsigned  BSIM4v4wbigcGiven   :1;
    unsigned  BSIM4v4wcigcGiven   :1;
    unsigned  BSIM4v4waigsdGiven   :1;
    unsigned  BSIM4v4wbigsdGiven   :1;
    unsigned  BSIM4v4wcigsdGiven   :1;
    unsigned  BSIM4v4waigbaccGiven   :1;
    unsigned  BSIM4v4wbigbaccGiven   :1;
    unsigned  BSIM4v4wcigbaccGiven   :1;
    unsigned  BSIM4v4waigbinvGiven   :1;
    unsigned  BSIM4v4wbigbinvGiven   :1;
    unsigned  BSIM4v4wcigbinvGiven   :1;
    unsigned  BSIM4v4wnigcGiven   :1;
    unsigned  BSIM4v4wnigbinvGiven   :1;
    unsigned  BSIM4v4wnigbaccGiven   :1;
    unsigned  BSIM4v4wntoxGiven   :1;
    unsigned  BSIM4v4weigbinvGiven   :1;
    unsigned  BSIM4v4wpigcdGiven   :1;
    unsigned  BSIM4v4wpoxedgeGiven   :1;
    unsigned  BSIM4v4wxrcrg1Given   :1;
    unsigned  BSIM4v4wxrcrg2Given   :1;
    unsigned  BSIM4v4wlambdaGiven    :1;
    unsigned  BSIM4v4wvtlGiven    :1;
    unsigned  BSIM4v4wxnGiven    :1;
    unsigned  BSIM4v4wvfbsdoffGiven    :1;

    /* CV model */
    unsigned  BSIM4v4wcgslGiven   :1;
    unsigned  BSIM4v4wcgdlGiven   :1;
    unsigned  BSIM4v4wckappasGiven   :1;
    unsigned  BSIM4v4wckappadGiven   :1;
    unsigned  BSIM4v4wcfGiven   :1;
    unsigned  BSIM4v4wclcGiven   :1;
    unsigned  BSIM4v4wcleGiven   :1;
    unsigned  BSIM4v4wvfbcvGiven   :1;
    unsigned  BSIM4v4wnoffGiven   :1;
    unsigned  BSIM4v4wvoffcvGiven :1;
    unsigned  BSIM4v4wacdeGiven   :1;
    unsigned  BSIM4v4wmoinGiven   :1;

    /* Cross-term dependence */
    unsigned  BSIM4v4pcdscGiven   :1;
    unsigned  BSIM4v4pcdscbGiven   :1;
    unsigned  BSIM4v4pcdscdGiven   :1;
    unsigned  BSIM4v4pcitGiven   :1;
    unsigned  BSIM4v4pnfactorGiven   :1;
    unsigned  BSIM4v4pxjGiven   :1;
    unsigned  BSIM4v4pvsatGiven   :1;
    unsigned  BSIM4v4patGiven   :1;
    unsigned  BSIM4v4pa0Given   :1;
    unsigned  BSIM4v4pagsGiven   :1;
    unsigned  BSIM4v4pa1Given   :1;
    unsigned  BSIM4v4pa2Given   :1;
    unsigned  BSIM4v4pketaGiven   :1;
    unsigned  BSIM4v4pnsubGiven   :1;
    unsigned  BSIM4v4pndepGiven   :1;
    unsigned  BSIM4v4pnsdGiven    :1;
    unsigned  BSIM4v4pphinGiven   :1;
    unsigned  BSIM4v4pngateGiven   :1;
    unsigned  BSIM4v4pgamma1Given   :1;
    unsigned  BSIM4v4pgamma2Given   :1;
    unsigned  BSIM4v4pvbxGiven   :1;
    unsigned  BSIM4v4pvbmGiven   :1;
    unsigned  BSIM4v4pxtGiven   :1;
    unsigned  BSIM4v4pk1Given   :1;
    unsigned  BSIM4v4pkt1Given   :1;
    unsigned  BSIM4v4pkt1lGiven   :1;
    unsigned  BSIM4v4pkt2Given   :1;
    unsigned  BSIM4v4pk2Given   :1;
    unsigned  BSIM4v4pk3Given   :1;
    unsigned  BSIM4v4pk3bGiven   :1;
    unsigned  BSIM4v4pw0Given   :1;
    unsigned  BSIM4v4pdvtp0Given :1;
    unsigned  BSIM4v4pdvtp1Given :1;
    unsigned  BSIM4v4plpe0Given   :1;
    unsigned  BSIM4v4plpebGiven   :1;
    unsigned  BSIM4v4pdvt0Given   :1;
    unsigned  BSIM4v4pdvt1Given   :1;
    unsigned  BSIM4v4pdvt2Given   :1;
    unsigned  BSIM4v4pdvt0wGiven   :1;
    unsigned  BSIM4v4pdvt1wGiven   :1;
    unsigned  BSIM4v4pdvt2wGiven   :1;
    unsigned  BSIM4v4pdroutGiven   :1;
    unsigned  BSIM4v4pdsubGiven   :1;
    unsigned  BSIM4v4pvth0Given   :1;
    unsigned  BSIM4v4puaGiven   :1;
    unsigned  BSIM4v4pua1Given   :1;
    unsigned  BSIM4v4pubGiven   :1;
    unsigned  BSIM4v4pub1Given   :1;
    unsigned  BSIM4v4pucGiven   :1;
    unsigned  BSIM4v4puc1Given   :1;
    unsigned  BSIM4v4pu0Given   :1;
    unsigned  BSIM4v4peuGiven   :1;
    unsigned  BSIM4v4puteGiven   :1;
    unsigned  BSIM4v4pvoffGiven   :1;
    unsigned  BSIM4v4pminvGiven   :1;
    unsigned  BSIM4v4prdswGiven   :1;
    unsigned  BSIM4v4prswGiven   :1;
    unsigned  BSIM4v4prdwGiven   :1;
    unsigned  BSIM4v4pprwgGiven   :1;
    unsigned  BSIM4v4pprwbGiven   :1;
    unsigned  BSIM4v4pprtGiven   :1;
    unsigned  BSIM4v4peta0Given   :1;
    unsigned  BSIM4v4petabGiven   :1;
    unsigned  BSIM4v4ppclmGiven   :1;
    unsigned  BSIM4v4ppdibl1Given   :1;
    unsigned  BSIM4v4ppdibl2Given   :1;
    unsigned  BSIM4v4ppdiblbGiven   :1;
    unsigned  BSIM4v4pfproutGiven   :1;
    unsigned  BSIM4v4ppditsGiven    :1;
    unsigned  BSIM4v4ppditsdGiven    :1;
    unsigned  BSIM4v4ppscbe1Given   :1;
    unsigned  BSIM4v4ppscbe2Given   :1;
    unsigned  BSIM4v4ppvagGiven   :1;
    unsigned  BSIM4v4pdeltaGiven  :1;
    unsigned  BSIM4v4pwrGiven   :1;
    unsigned  BSIM4v4pdwgGiven   :1;
    unsigned  BSIM4v4pdwbGiven   :1;
    unsigned  BSIM4v4pb0Given   :1;
    unsigned  BSIM4v4pb1Given   :1;
    unsigned  BSIM4v4palpha0Given   :1;
    unsigned  BSIM4v4palpha1Given   :1;
    unsigned  BSIM4v4pbeta0Given   :1;
    unsigned  BSIM4v4pvfbGiven   :1;
    unsigned  BSIM4v4pagidlGiven   :1;
    unsigned  BSIM4v4pbgidlGiven   :1;
    unsigned  BSIM4v4pcgidlGiven   :1;
    unsigned  BSIM4v4pegidlGiven   :1;
    unsigned  BSIM4v4paigcGiven   :1;
    unsigned  BSIM4v4pbigcGiven   :1;
    unsigned  BSIM4v4pcigcGiven   :1;
    unsigned  BSIM4v4paigsdGiven   :1;
    unsigned  BSIM4v4pbigsdGiven   :1;
    unsigned  BSIM4v4pcigsdGiven   :1;
    unsigned  BSIM4v4paigbaccGiven   :1;
    unsigned  BSIM4v4pbigbaccGiven   :1;
    unsigned  BSIM4v4pcigbaccGiven   :1;
    unsigned  BSIM4v4paigbinvGiven   :1;
    unsigned  BSIM4v4pbigbinvGiven   :1;
    unsigned  BSIM4v4pcigbinvGiven   :1;
    unsigned  BSIM4v4pnigcGiven   :1;
    unsigned  BSIM4v4pnigbinvGiven   :1;
    unsigned  BSIM4v4pnigbaccGiven   :1;
    unsigned  BSIM4v4pntoxGiven   :1;
    unsigned  BSIM4v4peigbinvGiven   :1;
    unsigned  BSIM4v4ppigcdGiven   :1;
    unsigned  BSIM4v4ppoxedgeGiven   :1;
    unsigned  BSIM4v4pxrcrg1Given   :1;
    unsigned  BSIM4v4pxrcrg2Given   :1;
    unsigned  BSIM4v4plambdaGiven    :1;
    unsigned  BSIM4v4pvtlGiven    :1;
    unsigned  BSIM4v4pxnGiven    :1;
    unsigned  BSIM4v4pvfbsdoffGiven    :1;

    /* CV model */
    unsigned  BSIM4v4pcgslGiven   :1;
    unsigned  BSIM4v4pcgdlGiven   :1;
    unsigned  BSIM4v4pckappasGiven   :1;
    unsigned  BSIM4v4pckappadGiven   :1;
    unsigned  BSIM4v4pcfGiven   :1;
    unsigned  BSIM4v4pclcGiven   :1;
    unsigned  BSIM4v4pcleGiven   :1;
    unsigned  BSIM4v4pvfbcvGiven   :1;
    unsigned  BSIM4v4pnoffGiven   :1;
    unsigned  BSIM4v4pvoffcvGiven :1;
    unsigned  BSIM4v4pacdeGiven   :1;
    unsigned  BSIM4v4pmoinGiven   :1;

    unsigned  BSIM4v4useFringeGiven   :1;

    unsigned  BSIM4v4tnomGiven   :1;
    unsigned  BSIM4v4cgsoGiven   :1;
    unsigned  BSIM4v4cgdoGiven   :1;
    unsigned  BSIM4v4cgboGiven   :1;
    unsigned  BSIM4v4xpartGiven   :1;
    unsigned  BSIM4v4sheetResistanceGiven   :1;

    unsigned  BSIM4v4SjctSatCurDensityGiven   :1;
    unsigned  BSIM4v4SjctSidewallSatCurDensityGiven   :1;
    unsigned  BSIM4v4SjctGateSidewallSatCurDensityGiven   :1;
    unsigned  BSIM4v4SbulkJctPotentialGiven   :1;
    unsigned  BSIM4v4SbulkJctBotGradingCoeffGiven   :1;
    unsigned  BSIM4v4SsidewallJctPotentialGiven   :1;
    unsigned  BSIM4v4SGatesidewallJctPotentialGiven   :1;
    unsigned  BSIM4v4SbulkJctSideGradingCoeffGiven   :1;
    unsigned  BSIM4v4SunitAreaJctCapGiven   :1;
    unsigned  BSIM4v4SunitLengthSidewallJctCapGiven   :1;
    unsigned  BSIM4v4SbulkJctGateSideGradingCoeffGiven   :1;
    unsigned  BSIM4v4SunitLengthGateSidewallJctCapGiven   :1;
    unsigned  BSIM4v4SjctEmissionCoeffGiven :1;
    unsigned  BSIM4v4SjctTempExponentGiven        :1;

    unsigned  BSIM4v4DjctSatCurDensityGiven   :1;
    unsigned  BSIM4v4DjctSidewallSatCurDensityGiven   :1;
    unsigned  BSIM4v4DjctGateSidewallSatCurDensityGiven   :1;
    unsigned  BSIM4v4DbulkJctPotentialGiven   :1;
    unsigned  BSIM4v4DbulkJctBotGradingCoeffGiven   :1;
    unsigned  BSIM4v4DsidewallJctPotentialGiven   :1;
    unsigned  BSIM4v4DGatesidewallJctPotentialGiven   :1;
    unsigned  BSIM4v4DbulkJctSideGradingCoeffGiven   :1;
    unsigned  BSIM4v4DunitAreaJctCapGiven   :1;
    unsigned  BSIM4v4DunitLengthSidewallJctCapGiven   :1;
    unsigned  BSIM4v4DbulkJctGateSideGradingCoeffGiven   :1;
    unsigned  BSIM4v4DunitLengthGateSidewallJctCapGiven   :1;
    unsigned  BSIM4v4DjctEmissionCoeffGiven :1;
    unsigned  BSIM4v4DjctTempExponentGiven :1;

    unsigned  BSIM4v4oxideTrapDensityAGiven  :1;
    unsigned  BSIM4v4oxideTrapDensityBGiven  :1;
    unsigned  BSIM4v4oxideTrapDensityCGiven  :1;
    unsigned  BSIM4v4emGiven  :1;
    unsigned  BSIM4v4efGiven  :1;
    unsigned  BSIM4v4afGiven  :1;
    unsigned  BSIM4v4kfGiven  :1;

    unsigned  BSIM4v4LintGiven   :1;
    unsigned  BSIM4v4LlGiven   :1;
    unsigned  BSIM4v4LlcGiven   :1;
    unsigned  BSIM4v4LlnGiven   :1;
    unsigned  BSIM4v4LwGiven   :1;
    unsigned  BSIM4v4LwcGiven   :1;
    unsigned  BSIM4v4LwnGiven   :1;
    unsigned  BSIM4v4LwlGiven   :1;
    unsigned  BSIM4v4LwlcGiven   :1;
    unsigned  BSIM4v4LminGiven   :1;
    unsigned  BSIM4v4LmaxGiven   :1;

    unsigned  BSIM4v4WintGiven   :1;
    unsigned  BSIM4v4WlGiven   :1;
    unsigned  BSIM4v4WlcGiven   :1;
    unsigned  BSIM4v4WlnGiven   :1;
    unsigned  BSIM4v4WwGiven   :1;
    unsigned  BSIM4v4WwcGiven   :1;
    unsigned  BSIM4v4WwnGiven   :1;
    unsigned  BSIM4v4WwlGiven   :1;
    unsigned  BSIM4v4WwlcGiven   :1;
    unsigned  BSIM4v4WminGiven   :1;
    unsigned  BSIM4v4WmaxGiven   :1;

    /* added for stress effect */
    unsigned  BSIM4v4sarefGiven   :1;
    unsigned  BSIM4v4sbrefGiven   :1;
    unsigned  BSIM4v4wlodGiven  :1;
    unsigned  BSIM4v4ku0Given   :1;
    unsigned  BSIM4v4kvsatGiven  :1;
    unsigned  BSIM4v4kvth0Given  :1;
    unsigned  BSIM4v4tku0Given   :1;
    unsigned  BSIM4v4llodku0Given   :1;
    unsigned  BSIM4v4wlodku0Given   :1;
    unsigned  BSIM4v4llodvthGiven   :1;
    unsigned  BSIM4v4wlodvthGiven   :1;
    unsigned  BSIM4v4lku0Given   :1;
    unsigned  BSIM4v4wku0Given   :1;
    unsigned  BSIM4v4pku0Given   :1;
    unsigned  BSIM4v4lkvth0Given   :1;
    unsigned  BSIM4v4wkvth0Given   :1;
    unsigned  BSIM4v4pkvth0Given   :1;
    unsigned  BSIM4v4stk2Given   :1;
    unsigned  BSIM4v4lodk2Given  :1;
    unsigned  BSIM4v4steta0Given :1;
    unsigned  BSIM4v4lodeta0Given :1;


} BSIM4v4model;


#ifndef NMOS
#define NMOS 1
#define PMOS -1
#endif /*NMOS*/


/* Instance parameters */
#define BSIM4v4_W                   1
#define BSIM4v4_L                   2
#define BSIM4v4_AS                  3
#define BSIM4v4_AD                  4
#define BSIM4v4_PS                  5
#define BSIM4v4_PD                  6
#define BSIM4v4_NRS                 7
#define BSIM4v4_NRD                 8
#define BSIM4v4_OFF                 9
#define BSIM4v4_IC                  10
#define BSIM4v4_IC_VDS              11
#define BSIM4v4_IC_VGS              12
#define BSIM4v4_IC_VBS              13
#define BSIM4v4_TRNQSMOD            14
#define BSIM4v4_RBODYMOD            15
#define BSIM4v4_RGATEMOD            16
#define BSIM4v4_GEOMOD              17
#define BSIM4v4_RGEOMOD             18
#define BSIM4v4_NF                  19
#define BSIM4v4_MIN                 20
#define BSIM4v4_ACNQSMOD            22
#define BSIM4v4_RBDB                23
#define BSIM4v4_RBSB                24
#define BSIM4v4_RBPB                25
#define BSIM4v4_RBPS                26
#define BSIM4v4_RBPD                27
#define BSIM4v4_SA                  28
#define BSIM4v4_SB                  29
#define BSIM4v4_SD                  30
#define BSIM4v4_M                   31

/* Global parameters */
#define BSIM4v4_MOD_TEMPMOD         89
#define BSIM4v4_MOD_IGCMOD          90
#define BSIM4v4_MOD_IGBMOD          91
#define BSIM4v4_MOD_ACNQSMOD        92
#define BSIM4v4_MOD_FNOIMOD         93
#define BSIM4v4_MOD_RDSMOD          94
#define BSIM4v4_MOD_DIOMOD          96
#define BSIM4v4_MOD_PERMOD          97
#define BSIM4v4_MOD_GEOMOD          98
#define BSIM4v4_MOD_RGATEMOD        99
#define BSIM4v4_MOD_RBODYMOD        100
#define BSIM4v4_MOD_CAPMOD          101
#define BSIM4v4_MOD_TRNQSMOD        102
#define BSIM4v4_MOD_MOBMOD          103
#define BSIM4v4_MOD_TNOIMOD         104
#define BSIM4v4_MOD_TOXE            105
#define BSIM4v4_MOD_CDSC            106
#define BSIM4v4_MOD_CDSCB           107
#define BSIM4v4_MOD_CIT             108
#define BSIM4v4_MOD_NFACTOR         109
#define BSIM4v4_MOD_XJ              110
#define BSIM4v4_MOD_VSAT            111
#define BSIM4v4_MOD_AT              112
#define BSIM4v4_MOD_A0              113
#define BSIM4v4_MOD_A1              114
#define BSIM4v4_MOD_A2              115
#define BSIM4v4_MOD_KETA            116
#define BSIM4v4_MOD_NSUB            117
#define BSIM4v4_MOD_NDEP            118
#define BSIM4v4_MOD_NGATE           120
#define BSIM4v4_MOD_GAMMA1          121
#define BSIM4v4_MOD_GAMMA2          122
#define BSIM4v4_MOD_VBX             123
#define BSIM4v4_MOD_BINUNIT         124
#define BSIM4v4_MOD_VBM             125
#define BSIM4v4_MOD_XT              126
#define BSIM4v4_MOD_K1              129
#define BSIM4v4_MOD_KT1             130
#define BSIM4v4_MOD_KT1L            131
#define BSIM4v4_MOD_K2              132
#define BSIM4v4_MOD_KT2             133
#define BSIM4v4_MOD_K3              134
#define BSIM4v4_MOD_K3B             135
#define BSIM4v4_MOD_W0              136
#define BSIM4v4_MOD_LPE0            137
#define BSIM4v4_MOD_DVT0            138
#define BSIM4v4_MOD_DVT1            139
#define BSIM4v4_MOD_DVT2            140
#define BSIM4v4_MOD_DVT0W           141
#define BSIM4v4_MOD_DVT1W           142
#define BSIM4v4_MOD_DVT2W           143
#define BSIM4v4_MOD_DROUT           144
#define BSIM4v4_MOD_DSUB            145
#define BSIM4v4_MOD_VTH0            146
#define BSIM4v4_MOD_UA              147
#define BSIM4v4_MOD_UA1             148
#define BSIM4v4_MOD_UB              149
#define BSIM4v4_MOD_UB1             150
#define BSIM4v4_MOD_UC              151
#define BSIM4v4_MOD_UC1             152
#define BSIM4v4_MOD_U0              153
#define BSIM4v4_MOD_UTE             154
#define BSIM4v4_MOD_VOFF            155
#define BSIM4v4_MOD_DELTA           156
#define BSIM4v4_MOD_RDSW            157
#define BSIM4v4_MOD_PRT             158
#define BSIM4v4_MOD_LDD             159
#define BSIM4v4_MOD_ETA             160
#define BSIM4v4_MOD_ETA0            161
#define BSIM4v4_MOD_ETAB            162
#define BSIM4v4_MOD_PCLM            163
#define BSIM4v4_MOD_PDIBL1          164
#define BSIM4v4_MOD_PDIBL2          165
#define BSIM4v4_MOD_PSCBE1          166
#define BSIM4v4_MOD_PSCBE2          167
#define BSIM4v4_MOD_PVAG            168
#define BSIM4v4_MOD_WR              169
#define BSIM4v4_MOD_DWG             170
#define BSIM4v4_MOD_DWB             171
#define BSIM4v4_MOD_B0              172
#define BSIM4v4_MOD_B1              173
#define BSIM4v4_MOD_ALPHA0          174
#define BSIM4v4_MOD_BETA0           175
#define BSIM4v4_MOD_PDIBLB          178
#define BSIM4v4_MOD_PRWG            179
#define BSIM4v4_MOD_PRWB            180
#define BSIM4v4_MOD_CDSCD           181
#define BSIM4v4_MOD_AGS             182
#define BSIM4v4_MOD_FRINGE          184
#define BSIM4v4_MOD_CGSL            186
#define BSIM4v4_MOD_CGDL            187
#define BSIM4v4_MOD_CKAPPAS         188
#define BSIM4v4_MOD_CF              189
#define BSIM4v4_MOD_CLC             190
#define BSIM4v4_MOD_CLE             191
#define BSIM4v4_MOD_PARAMCHK        192
#define BSIM4v4_MOD_VERSION         193
#define BSIM4v4_MOD_VFBCV           194
#define BSIM4v4_MOD_ACDE            195
#define BSIM4v4_MOD_MOIN            196
#define BSIM4v4_MOD_NOFF            197
#define BSIM4v4_MOD_IJTHDFWD        198
#define BSIM4v4_MOD_ALPHA1          199
#define BSIM4v4_MOD_VFB             200
#define BSIM4v4_MOD_TOXM            201
#define BSIM4v4_MOD_TCJ             202
#define BSIM4v4_MOD_TCJSW           203
#define BSIM4v4_MOD_TCJSWG          204
#define BSIM4v4_MOD_TPB             205
#define BSIM4v4_MOD_TPBSW           206
#define BSIM4v4_MOD_TPBSWG          207
#define BSIM4v4_MOD_VOFFCV          208
#define BSIM4v4_MOD_GBMIN           209
#define BSIM4v4_MOD_RBDB            210
#define BSIM4v4_MOD_RBSB            211
#define BSIM4v4_MOD_RBPB            212
#define BSIM4v4_MOD_RBPS            213
#define BSIM4v4_MOD_RBPD            214
#define BSIM4v4_MOD_DMCG            215
#define BSIM4v4_MOD_DMCI            216
#define BSIM4v4_MOD_DMDG            217
#define BSIM4v4_MOD_XGW             218
#define BSIM4v4_MOD_XGL             219
#define BSIM4v4_MOD_RSHG            220
#define BSIM4v4_MOD_NGCON           221
#define BSIM4v4_MOD_AGIDL           222
#define BSIM4v4_MOD_BGIDL           223
#define BSIM4v4_MOD_EGIDL           224
#define BSIM4v4_MOD_IJTHSFWD        225
#define BSIM4v4_MOD_XJBVD           226
#define BSIM4v4_MOD_XJBVS           227
#define BSIM4v4_MOD_BVD             228
#define BSIM4v4_MOD_BVS             229
#define BSIM4v4_MOD_TOXP            230
#define BSIM4v4_MOD_DTOX            231
#define BSIM4v4_MOD_XRCRG1          232
#define BSIM4v4_MOD_XRCRG2          233
#define BSIM4v4_MOD_EU              234
#define BSIM4v4_MOD_IJTHSREV        235
#define BSIM4v4_MOD_IJTHDREV        236
#define BSIM4v4_MOD_MINV            237
#define BSIM4v4_MOD_VOFFL           238
#define BSIM4v4_MOD_PDITS           239
#define BSIM4v4_MOD_PDITSD          240
#define BSIM4v4_MOD_PDITSL          241
#define BSIM4v4_MOD_TNOIA           242
#define BSIM4v4_MOD_TNOIB           243
#define BSIM4v4_MOD_NTNOI           244
#define BSIM4v4_MOD_FPROUT          245
#define BSIM4v4_MOD_LPEB            246
#define BSIM4v4_MOD_DVTP0           247
#define BSIM4v4_MOD_DVTP1           248
#define BSIM4v4_MOD_CGIDL           249
#define BSIM4v4_MOD_PHIN            250
#define BSIM4v4_MOD_RDSWMIN         251
#define BSIM4v4_MOD_RSW             252
#define BSIM4v4_MOD_RDW             253
#define BSIM4v4_MOD_RDWMIN          254
#define BSIM4v4_MOD_RSWMIN          255
#define BSIM4v4_MOD_NSD             256
#define BSIM4v4_MOD_CKAPPAD         257
#define BSIM4v4_MOD_DMCGT           258
#define BSIM4v4_MOD_AIGC            259
#define BSIM4v4_MOD_BIGC            260
#define BSIM4v4_MOD_CIGC            261
#define BSIM4v4_MOD_AIGBACC         262
#define BSIM4v4_MOD_BIGBACC         263
#define BSIM4v4_MOD_CIGBACC         264
#define BSIM4v4_MOD_AIGBINV         265
#define BSIM4v4_MOD_BIGBINV         266
#define BSIM4v4_MOD_CIGBINV         267
#define BSIM4v4_MOD_NIGC            268
#define BSIM4v4_MOD_NIGBACC         269
#define BSIM4v4_MOD_NIGBINV         270
#define BSIM4v4_MOD_NTOX            271
#define BSIM4v4_MOD_TOXREF          272
#define BSIM4v4_MOD_EIGBINV         273
#define BSIM4v4_MOD_PIGCD           274
#define BSIM4v4_MOD_POXEDGE         275
#define BSIM4v4_MOD_EPSROX          276
#define BSIM4v4_MOD_AIGSD           277
#define BSIM4v4_MOD_BIGSD           278
#define BSIM4v4_MOD_CIGSD           279
#define BSIM4v4_MOD_JSWGS           280
#define BSIM4v4_MOD_JSWGD           281
#define BSIM4v4_MOD_LAMBDA          282
#define BSIM4v4_MOD_VTL             283
#define BSIM4v4_MOD_LC              284
#define BSIM4v4_MOD_XN              285
#define BSIM4v4_MOD_RNOIA           286
#define BSIM4v4_MOD_RNOIB           287
#define BSIM4v4_MOD_VFBSDOFF        288
#define BSIM4v4_MOD_LINTNOI         289

/* Length dependence */
#define BSIM4v4_MOD_LCDSC            301
#define BSIM4v4_MOD_LCDSCB           302
#define BSIM4v4_MOD_LCIT             303
#define BSIM4v4_MOD_LNFACTOR         304
#define BSIM4v4_MOD_LXJ              305
#define BSIM4v4_MOD_LVSAT            306
#define BSIM4v4_MOD_LAT              307
#define BSIM4v4_MOD_LA0              308
#define BSIM4v4_MOD_LA1              309
#define BSIM4v4_MOD_LA2              310
#define BSIM4v4_MOD_LKETA            311
#define BSIM4v4_MOD_LNSUB            312
#define BSIM4v4_MOD_LNDEP            313
#define BSIM4v4_MOD_LNGATE           315
#define BSIM4v4_MOD_LGAMMA1          316
#define BSIM4v4_MOD_LGAMMA2          317
#define BSIM4v4_MOD_LVBX             318
#define BSIM4v4_MOD_LVBM             320
#define BSIM4v4_MOD_LXT              322
#define BSIM4v4_MOD_LK1              325
#define BSIM4v4_MOD_LKT1             326
#define BSIM4v4_MOD_LKT1L            327
#define BSIM4v4_MOD_LK2              328
#define BSIM4v4_MOD_LKT2             329
#define BSIM4v4_MOD_LK3              330
#define BSIM4v4_MOD_LK3B             331
#define BSIM4v4_MOD_LW0              332
#define BSIM4v4_MOD_LLPE0            333
#define BSIM4v4_MOD_LDVT0            334
#define BSIM4v4_MOD_LDVT1            335
#define BSIM4v4_MOD_LDVT2            336
#define BSIM4v4_MOD_LDVT0W           337
#define BSIM4v4_MOD_LDVT1W           338
#define BSIM4v4_MOD_LDVT2W           339
#define BSIM4v4_MOD_LDROUT           340
#define BSIM4v4_MOD_LDSUB            341
#define BSIM4v4_MOD_LVTH0            342
#define BSIM4v4_MOD_LUA              343
#define BSIM4v4_MOD_LUA1             344
#define BSIM4v4_MOD_LUB              345
#define BSIM4v4_MOD_LUB1             346
#define BSIM4v4_MOD_LUC              347
#define BSIM4v4_MOD_LUC1             348
#define BSIM4v4_MOD_LU0              349
#define BSIM4v4_MOD_LUTE             350
#define BSIM4v4_MOD_LVOFF            351
#define BSIM4v4_MOD_LDELTA           352
#define BSIM4v4_MOD_LRDSW            353
#define BSIM4v4_MOD_LPRT             354
#define BSIM4v4_MOD_LLDD             355
#define BSIM4v4_MOD_LETA             356
#define BSIM4v4_MOD_LETA0            357
#define BSIM4v4_MOD_LETAB            358
#define BSIM4v4_MOD_LPCLM            359
#define BSIM4v4_MOD_LPDIBL1          360
#define BSIM4v4_MOD_LPDIBL2          361
#define BSIM4v4_MOD_LPSCBE1          362
#define BSIM4v4_MOD_LPSCBE2          363
#define BSIM4v4_MOD_LPVAG            364
#define BSIM4v4_MOD_LWR              365
#define BSIM4v4_MOD_LDWG             366
#define BSIM4v4_MOD_LDWB             367
#define BSIM4v4_MOD_LB0              368
#define BSIM4v4_MOD_LB1              369
#define BSIM4v4_MOD_LALPHA0          370
#define BSIM4v4_MOD_LBETA0           371
#define BSIM4v4_MOD_LPDIBLB          374
#define BSIM4v4_MOD_LPRWG            375
#define BSIM4v4_MOD_LPRWB            376
#define BSIM4v4_MOD_LCDSCD           377
#define BSIM4v4_MOD_LAGS             378

#define BSIM4v4_MOD_LFRINGE          381
#define BSIM4v4_MOD_LCGSL            383
#define BSIM4v4_MOD_LCGDL            384
#define BSIM4v4_MOD_LCKAPPAS         385
#define BSIM4v4_MOD_LCF              386
#define BSIM4v4_MOD_LCLC             387
#define BSIM4v4_MOD_LCLE             388
#define BSIM4v4_MOD_LVFBCV           389
#define BSIM4v4_MOD_LACDE            390
#define BSIM4v4_MOD_LMOIN            391
#define BSIM4v4_MOD_LNOFF            392
#define BSIM4v4_MOD_LALPHA1          394
#define BSIM4v4_MOD_LVFB             395
#define BSIM4v4_MOD_LVOFFCV          396
#define BSIM4v4_MOD_LAGIDL           397
#define BSIM4v4_MOD_LBGIDL           398
#define BSIM4v4_MOD_LEGIDL           399
#define BSIM4v4_MOD_LXRCRG1          400
#define BSIM4v4_MOD_LXRCRG2          401
#define BSIM4v4_MOD_LEU              402
#define BSIM4v4_MOD_LMINV            403
#define BSIM4v4_MOD_LPDITS           404
#define BSIM4v4_MOD_LPDITSD          405
#define BSIM4v4_MOD_LFPROUT          406
#define BSIM4v4_MOD_LLPEB            407
#define BSIM4v4_MOD_LDVTP0           408
#define BSIM4v4_MOD_LDVTP1           409
#define BSIM4v4_MOD_LCGIDL           410
#define BSIM4v4_MOD_LPHIN            411
#define BSIM4v4_MOD_LRSW             412
#define BSIM4v4_MOD_LRDW             413
#define BSIM4v4_MOD_LNSD             414
#define BSIM4v4_MOD_LCKAPPAD         415
#define BSIM4v4_MOD_LAIGC            416
#define BSIM4v4_MOD_LBIGC            417
#define BSIM4v4_MOD_LCIGC            418
#define BSIM4v4_MOD_LAIGBACC         419
#define BSIM4v4_MOD_LBIGBACC         420
#define BSIM4v4_MOD_LCIGBACC         421
#define BSIM4v4_MOD_LAIGBINV         422
#define BSIM4v4_MOD_LBIGBINV         423
#define BSIM4v4_MOD_LCIGBINV         424
#define BSIM4v4_MOD_LNIGC            425
#define BSIM4v4_MOD_LNIGBACC         426
#define BSIM4v4_MOD_LNIGBINV         427
#define BSIM4v4_MOD_LNTOX            428
#define BSIM4v4_MOD_LEIGBINV         429
#define BSIM4v4_MOD_LPIGCD           430
#define BSIM4v4_MOD_LPOXEDGE         431
#define BSIM4v4_MOD_LAIGSD           432
#define BSIM4v4_MOD_LBIGSD           433
#define BSIM4v4_MOD_LCIGSD           434

#define BSIM4v4_MOD_LLAMBDA          435
#define BSIM4v4_MOD_LVTL             436
#define BSIM4v4_MOD_LXN              437
#define BSIM4v4_MOD_LVFBSDOFF        438

/* Width dependence */
#define BSIM4v4_MOD_WCDSC            481
#define BSIM4v4_MOD_WCDSCB           482
#define BSIM4v4_MOD_WCIT             483
#define BSIM4v4_MOD_WNFACTOR         484
#define BSIM4v4_MOD_WXJ              485
#define BSIM4v4_MOD_WVSAT            486
#define BSIM4v4_MOD_WAT              487
#define BSIM4v4_MOD_WA0              488
#define BSIM4v4_MOD_WA1              489
#define BSIM4v4_MOD_WA2              490
#define BSIM4v4_MOD_WKETA            491
#define BSIM4v4_MOD_WNSUB            492
#define BSIM4v4_MOD_WNDEP            493
#define BSIM4v4_MOD_WNGATE           495
#define BSIM4v4_MOD_WGAMMA1          496
#define BSIM4v4_MOD_WGAMMA2          497
#define BSIM4v4_MOD_WVBX             498
#define BSIM4v4_MOD_WVBM             500
#define BSIM4v4_MOD_WXT              502
#define BSIM4v4_MOD_WK1              505
#define BSIM4v4_MOD_WKT1             506
#define BSIM4v4_MOD_WKT1L            507
#define BSIM4v4_MOD_WK2              508
#define BSIM4v4_MOD_WKT2             509
#define BSIM4v4_MOD_WK3              510
#define BSIM4v4_MOD_WK3B             511
#define BSIM4v4_MOD_WW0              512
#define BSIM4v4_MOD_WLPE0            513
#define BSIM4v4_MOD_WDVT0            514
#define BSIM4v4_MOD_WDVT1            515
#define BSIM4v4_MOD_WDVT2            516
#define BSIM4v4_MOD_WDVT0W           517
#define BSIM4v4_MOD_WDVT1W           518
#define BSIM4v4_MOD_WDVT2W           519
#define BSIM4v4_MOD_WDROUT           520
#define BSIM4v4_MOD_WDSUB            521
#define BSIM4v4_MOD_WVTH0            522
#define BSIM4v4_MOD_WUA              523
#define BSIM4v4_MOD_WUA1             524
#define BSIM4v4_MOD_WUB              525
#define BSIM4v4_MOD_WUB1             526
#define BSIM4v4_MOD_WUC              527
#define BSIM4v4_MOD_WUC1             528
#define BSIM4v4_MOD_WU0              529
#define BSIM4v4_MOD_WUTE             530
#define BSIM4v4_MOD_WVOFF            531
#define BSIM4v4_MOD_WDELTA           532
#define BSIM4v4_MOD_WRDSW            533
#define BSIM4v4_MOD_WPRT             534
#define BSIM4v4_MOD_WLDD             535
#define BSIM4v4_MOD_WETA             536
#define BSIM4v4_MOD_WETA0            537
#define BSIM4v4_MOD_WETAB            538
#define BSIM4v4_MOD_WPCLM            539
#define BSIM4v4_MOD_WPDIBL1          540
#define BSIM4v4_MOD_WPDIBL2          541
#define BSIM4v4_MOD_WPSCBE1          542
#define BSIM4v4_MOD_WPSCBE2          543
#define BSIM4v4_MOD_WPVAG            544
#define BSIM4v4_MOD_WWR              545
#define BSIM4v4_MOD_WDWG             546
#define BSIM4v4_MOD_WDWB             547
#define BSIM4v4_MOD_WB0              548
#define BSIM4v4_MOD_WB1              549
#define BSIM4v4_MOD_WALPHA0          550
#define BSIM4v4_MOD_WBETA0           551
#define BSIM4v4_MOD_WPDIBLB          554
#define BSIM4v4_MOD_WPRWG            555
#define BSIM4v4_MOD_WPRWB            556
#define BSIM4v4_MOD_WCDSCD           557
#define BSIM4v4_MOD_WAGS             558

#define BSIM4v4_MOD_WFRINGE          561
#define BSIM4v4_MOD_WCGSL            563
#define BSIM4v4_MOD_WCGDL            564
#define BSIM4v4_MOD_WCKAPPAS         565
#define BSIM4v4_MOD_WCF              566
#define BSIM4v4_MOD_WCLC             567
#define BSIM4v4_MOD_WCLE             568
#define BSIM4v4_MOD_WVFBCV           569
#define BSIM4v4_MOD_WACDE            570
#define BSIM4v4_MOD_WMOIN            571
#define BSIM4v4_MOD_WNOFF            572
#define BSIM4v4_MOD_WALPHA1          574
#define BSIM4v4_MOD_WVFB             575
#define BSIM4v4_MOD_WVOFFCV          576
#define BSIM4v4_MOD_WAGIDL           577
#define BSIM4v4_MOD_WBGIDL           578
#define BSIM4v4_MOD_WEGIDL           579
#define BSIM4v4_MOD_WXRCRG1          580
#define BSIM4v4_MOD_WXRCRG2          581
#define BSIM4v4_MOD_WEU              582
#define BSIM4v4_MOD_WMINV            583
#define BSIM4v4_MOD_WPDITS           584
#define BSIM4v4_MOD_WPDITSD          585
#define BSIM4v4_MOD_WFPROUT          586
#define BSIM4v4_MOD_WLPEB            587
#define BSIM4v4_MOD_WDVTP0           588
#define BSIM4v4_MOD_WDVTP1           589
#define BSIM4v4_MOD_WCGIDL           590
#define BSIM4v4_MOD_WPHIN            591
#define BSIM4v4_MOD_WRSW             592
#define BSIM4v4_MOD_WRDW             593
#define BSIM4v4_MOD_WNSD             594
#define BSIM4v4_MOD_WCKAPPAD         595
#define BSIM4v4_MOD_WAIGC            596
#define BSIM4v4_MOD_WBIGC            597
#define BSIM4v4_MOD_WCIGC            598
#define BSIM4v4_MOD_WAIGBACC         599
#define BSIM4v4_MOD_WBIGBACC         600
#define BSIM4v4_MOD_WCIGBACC         601
#define BSIM4v4_MOD_WAIGBINV         602
#define BSIM4v4_MOD_WBIGBINV         603
#define BSIM4v4_MOD_WCIGBINV         604
#define BSIM4v4_MOD_WNIGC            605
#define BSIM4v4_MOD_WNIGBACC         606
#define BSIM4v4_MOD_WNIGBINV         607
#define BSIM4v4_MOD_WNTOX            608
#define BSIM4v4_MOD_WEIGBINV         609
#define BSIM4v4_MOD_WPIGCD           610
#define BSIM4v4_MOD_WPOXEDGE         611
#define BSIM4v4_MOD_WAIGSD           612
#define BSIM4v4_MOD_WBIGSD           613
#define BSIM4v4_MOD_WCIGSD           614
#define BSIM4v4_MOD_WLAMBDA          615
#define BSIM4v4_MOD_WVTL             616
#define BSIM4v4_MOD_WXN              617
#define BSIM4v4_MOD_WVFBSDOFF        618

/* Cross-term dependence */
#define BSIM4v4_MOD_PCDSC            661
#define BSIM4v4_MOD_PCDSCB           662
#define BSIM4v4_MOD_PCIT             663
#define BSIM4v4_MOD_PNFACTOR         664
#define BSIM4v4_MOD_PXJ              665
#define BSIM4v4_MOD_PVSAT            666
#define BSIM4v4_MOD_PAT              667
#define BSIM4v4_MOD_PA0              668
#define BSIM4v4_MOD_PA1              669
#define BSIM4v4_MOD_PA2              670
#define BSIM4v4_MOD_PKETA            671
#define BSIM4v4_MOD_PNSUB            672
#define BSIM4v4_MOD_PNDEP            673
#define BSIM4v4_MOD_PNGATE           675
#define BSIM4v4_MOD_PGAMMA1          676
#define BSIM4v4_MOD_PGAMMA2          677
#define BSIM4v4_MOD_PVBX             678

#define BSIM4v4_MOD_PVBM             680

#define BSIM4v4_MOD_PXT              682
#define BSIM4v4_MOD_PK1              685
#define BSIM4v4_MOD_PKT1             686
#define BSIM4v4_MOD_PKT1L            687
#define BSIM4v4_MOD_PK2              688
#define BSIM4v4_MOD_PKT2             689
#define BSIM4v4_MOD_PK3              690
#define BSIM4v4_MOD_PK3B             691
#define BSIM4v4_MOD_PW0              692
#define BSIM4v4_MOD_PLPE0            693

#define BSIM4v4_MOD_PDVT0            694
#define BSIM4v4_MOD_PDVT1            695
#define BSIM4v4_MOD_PDVT2            696

#define BSIM4v4_MOD_PDVT0W           697
#define BSIM4v4_MOD_PDVT1W           698
#define BSIM4v4_MOD_PDVT2W           699

#define BSIM4v4_MOD_PDROUT           700
#define BSIM4v4_MOD_PDSUB            701
#define BSIM4v4_MOD_PVTH0            702
#define BSIM4v4_MOD_PUA              703
#define BSIM4v4_MOD_PUA1             704
#define BSIM4v4_MOD_PUB              705
#define BSIM4v4_MOD_PUB1             706
#define BSIM4v4_MOD_PUC              707
#define BSIM4v4_MOD_PUC1             708
#define BSIM4v4_MOD_PU0              709
#define BSIM4v4_MOD_PUTE             710
#define BSIM4v4_MOD_PVOFF            711
#define BSIM4v4_MOD_PDELTA           712
#define BSIM4v4_MOD_PRDSW            713
#define BSIM4v4_MOD_PPRT             714
#define BSIM4v4_MOD_PLDD             715
#define BSIM4v4_MOD_PETA             716
#define BSIM4v4_MOD_PETA0            717
#define BSIM4v4_MOD_PETAB            718
#define BSIM4v4_MOD_PPCLM            719
#define BSIM4v4_MOD_PPDIBL1          720
#define BSIM4v4_MOD_PPDIBL2          721
#define BSIM4v4_MOD_PPSCBE1          722
#define BSIM4v4_MOD_PPSCBE2          723
#define BSIM4v4_MOD_PPVAG            724
#define BSIM4v4_MOD_PWR              725
#define BSIM4v4_MOD_PDWG             726
#define BSIM4v4_MOD_PDWB             727
#define BSIM4v4_MOD_PB0              728
#define BSIM4v4_MOD_PB1              729
#define BSIM4v4_MOD_PALPHA0          730
#define BSIM4v4_MOD_PBETA0           731
#define BSIM4v4_MOD_PPDIBLB          734

#define BSIM4v4_MOD_PPRWG            735
#define BSIM4v4_MOD_PPRWB            736

#define BSIM4v4_MOD_PCDSCD           737
#define BSIM4v4_MOD_PAGS             738

#define BSIM4v4_MOD_PFRINGE          741
#define BSIM4v4_MOD_PCGSL            743
#define BSIM4v4_MOD_PCGDL            744
#define BSIM4v4_MOD_PCKAPPAS         745
#define BSIM4v4_MOD_PCF              746
#define BSIM4v4_MOD_PCLC             747
#define BSIM4v4_MOD_PCLE             748
#define BSIM4v4_MOD_PVFBCV           749
#define BSIM4v4_MOD_PACDE            750
#define BSIM4v4_MOD_PMOIN            751
#define BSIM4v4_MOD_PNOFF            752
#define BSIM4v4_MOD_PALPHA1          754
#define BSIM4v4_MOD_PVFB             755
#define BSIM4v4_MOD_PVOFFCV          756
#define BSIM4v4_MOD_PAGIDL           757
#define BSIM4v4_MOD_PBGIDL           758
#define BSIM4v4_MOD_PEGIDL           759
#define BSIM4v4_MOD_PXRCRG1          760
#define BSIM4v4_MOD_PXRCRG2          761
#define BSIM4v4_MOD_PEU              762
#define BSIM4v4_MOD_PMINV            763
#define BSIM4v4_MOD_PPDITS           764
#define BSIM4v4_MOD_PPDITSD          765
#define BSIM4v4_MOD_PFPROUT          766
#define BSIM4v4_MOD_PLPEB            767
#define BSIM4v4_MOD_PDVTP0           768
#define BSIM4v4_MOD_PDVTP1           769
#define BSIM4v4_MOD_PCGIDL           770
#define BSIM4v4_MOD_PPHIN            771
#define BSIM4v4_MOD_PRSW             772
#define BSIM4v4_MOD_PRDW             773
#define BSIM4v4_MOD_PNSD             774
#define BSIM4v4_MOD_PCKAPPAD         775
#define BSIM4v4_MOD_PAIGC            776
#define BSIM4v4_MOD_PBIGC            777
#define BSIM4v4_MOD_PCIGC            778
#define BSIM4v4_MOD_PAIGBACC         779
#define BSIM4v4_MOD_PBIGBACC         780
#define BSIM4v4_MOD_PCIGBACC         781
#define BSIM4v4_MOD_PAIGBINV         782
#define BSIM4v4_MOD_PBIGBINV         783
#define BSIM4v4_MOD_PCIGBINV         784
#define BSIM4v4_MOD_PNIGC            785
#define BSIM4v4_MOD_PNIGBACC         786
#define BSIM4v4_MOD_PNIGBINV         787
#define BSIM4v4_MOD_PNTOX            788
#define BSIM4v4_MOD_PEIGBINV         789
#define BSIM4v4_MOD_PPIGCD           790
#define BSIM4v4_MOD_PPOXEDGE         791
#define BSIM4v4_MOD_PAIGSD           792
#define BSIM4v4_MOD_PBIGSD           793
#define BSIM4v4_MOD_PCIGSD           794

#define BSIM4v4_MOD_SAREF            795
#define BSIM4v4_MOD_SBREF            796
#define BSIM4v4_MOD_KU0              797
#define BSIM4v4_MOD_KVSAT            798
#define BSIM4v4_MOD_TKU0             799
#define BSIM4v4_MOD_LLODKU0          800
#define BSIM4v4_MOD_WLODKU0          801
#define BSIM4v4_MOD_LLODVTH          802
#define BSIM4v4_MOD_WLODVTH          803
#define BSIM4v4_MOD_LKU0             804
#define BSIM4v4_MOD_WKU0             805
#define BSIM4v4_MOD_PKU0             806
#define BSIM4v4_MOD_KVTH0            807
#define BSIM4v4_MOD_LKVTH0           808
#define BSIM4v4_MOD_WKVTH0           809
#define BSIM4v4_MOD_PKVTH0           810
#define BSIM4v4_MOD_WLOD             811
#define BSIM4v4_MOD_STK2             812
#define BSIM4v4_MOD_LODK2            813
#define BSIM4v4_MOD_STETA0           814
#define BSIM4v4_MOD_LODETA0          815

#define BSIM4v4_MOD_PLAMBDA          825
#define BSIM4v4_MOD_PVTL             826
#define BSIM4v4_MOD_PXN              827
#define BSIM4v4_MOD_PVFBSDOFF        828

#define BSIM4v4_MOD_TNOM             831
#define BSIM4v4_MOD_CGSO             832
#define BSIM4v4_MOD_CGDO             833
#define BSIM4v4_MOD_CGBO             834
#define BSIM4v4_MOD_XPART            835
#define BSIM4v4_MOD_RSH              836
#define BSIM4v4_MOD_JSS              837
#define BSIM4v4_MOD_PBS              838
#define BSIM4v4_MOD_MJS              839
#define BSIM4v4_MOD_PBSWS            840
#define BSIM4v4_MOD_MJSWS            841
#define BSIM4v4_MOD_CJS              842
#define BSIM4v4_MOD_CJSWS            843
#define BSIM4v4_MOD_NMOS             844
#define BSIM4v4_MOD_PMOS             845
#define BSIM4v4_MOD_NOIA             846
#define BSIM4v4_MOD_NOIB             847
#define BSIM4v4_MOD_NOIC             848
#define BSIM4v4_MOD_LINT             849
#define BSIM4v4_MOD_LL               850
#define BSIM4v4_MOD_LLN              851
#define BSIM4v4_MOD_LW               852
#define BSIM4v4_MOD_LWN              853
#define BSIM4v4_MOD_LWL              854
#define BSIM4v4_MOD_LMIN             855
#define BSIM4v4_MOD_LMAX             856
#define BSIM4v4_MOD_WINT             857
#define BSIM4v4_MOD_WL               858
#define BSIM4v4_MOD_WLN              859
#define BSIM4v4_MOD_WW               860
#define BSIM4v4_MOD_WWN              861
#define BSIM4v4_MOD_WWL              862
#define BSIM4v4_MOD_WMIN             863
#define BSIM4v4_MOD_WMAX             864
#define BSIM4v4_MOD_DWC              865
#define BSIM4v4_MOD_DLC              866
#define BSIM4v4_MOD_XL               867
#define BSIM4v4_MOD_XW               868
#define BSIM4v4_MOD_EM               869
#define BSIM4v4_MOD_EF               870
#define BSIM4v4_MOD_AF               871
#define BSIM4v4_MOD_KF               872
#define BSIM4v4_MOD_NJS              873
#define BSIM4v4_MOD_XTIS             874
#define BSIM4v4_MOD_PBSWGS           875
#define BSIM4v4_MOD_MJSWGS           876
#define BSIM4v4_MOD_CJSWGS           877
#define BSIM4v4_MOD_JSWS             878
#define BSIM4v4_MOD_LLC              879
#define BSIM4v4_MOD_LWC              880
#define BSIM4v4_MOD_LWLC             881
#define BSIM4v4_MOD_WLC              882
#define BSIM4v4_MOD_WWC              883
#define BSIM4v4_MOD_WWLC             884
#define BSIM4v4_MOD_DWJ              885
#define BSIM4v4_MOD_JSD              886
#define BSIM4v4_MOD_PBD              887
#define BSIM4v4_MOD_MJD              888
#define BSIM4v4_MOD_PBSWD            889
#define BSIM4v4_MOD_MJSWD            890
#define BSIM4v4_MOD_CJD              891
#define BSIM4v4_MOD_CJSWD            892
#define BSIM4v4_MOD_NJD              893
#define BSIM4v4_MOD_XTID             894
#define BSIM4v4_MOD_PBSWGD           895
#define BSIM4v4_MOD_MJSWGD           896
#define BSIM4v4_MOD_CJSWGD           897
#define BSIM4v4_MOD_JSWD             898
#define BSIM4v4_MOD_DLCIG            899

/* trap-assisted tunneling */

#define BSIM4v4_MOD_JTSS             900
#define BSIM4v4_MOD_JTSD             901
#define BSIM4v4_MOD_JTSSWS           902
#define BSIM4v4_MOD_JTSSWD           903
#define BSIM4v4_MOD_JTSSWGS          904
#define BSIM4v4_MOD_JTSSWGD          905
#define BSIM4v4_MOD_NJTS             906
#define BSIM4v4_MOD_NJTSSW           907
#define BSIM4v4_MOD_NJTSSWG          908
#define BSIM4v4_MOD_XTSS             909
#define BSIM4v4_MOD_XTSD             910
#define BSIM4v4_MOD_XTSSWS           911
#define BSIM4v4_MOD_XTSSWD           912
#define BSIM4v4_MOD_XTSSWGS          913
#define BSIM4v4_MOD_XTSSWGD          914
#define BSIM4v4_MOD_TNJTS            915
#define BSIM4v4_MOD_TNJTSSW          916
#define BSIM4v4_MOD_TNJTSSWG         917
#define BSIM4v4_MOD_VTSS             918
#define BSIM4v4_MOD_VTSD             919
#define BSIM4v4_MOD_VTSSWS           920
#define BSIM4v4_MOD_VTSSWD           921
#define BSIM4v4_MOD_VTSSWGS          922
#define BSIM4v4_MOD_VTSSWGD          923

/* device questions */
#define BSIM4v4_DNODE                945
#define BSIM4v4_GNODEEXT             946
#define BSIM4v4_SNODE                947
#define BSIM4v4_BNODE                948
#define BSIM4v4_DNODEPRIME           949
#define BSIM4v4_GNODEPRIME           950
#define BSIM4v4_GNODEMIDE            951
#define BSIM4v4_GNODEMID             952
#define BSIM4v4_SNODEPRIME           953
#define BSIM4v4_BNODEPRIME           954
#define BSIM4v4_DBNODE               955
#define BSIM4v4_SBNODE               956
#define BSIM4v4_VBD                  957
#define BSIM4v4_VBS                  958
#define BSIM4v4_VGS                  959
#define BSIM4v4_VDS                  960
#define BSIM4v4_CD                   961
#define BSIM4v4_CBS                  962
#define BSIM4v4_CBD                  963
#define BSIM4v4_GM                   964
#define BSIM4v4_GDS                  965
#define BSIM4v4_GMBS                 966
#define BSIM4v4_GBD                  967
#define BSIM4v4_GBS                  968
#define BSIM4v4_QB                   969
#define BSIM4v4_CQB                  970
#define BSIM4v4_QG                   971
#define BSIM4v4_CQG                  972
#define BSIM4v4_QD                   973
#define BSIM4v4_CQD                  974
#define BSIM4v4_CGGB                 975
#define BSIM4v4_CGDB                 976
#define BSIM4v4_CGSB                 977
#define BSIM4v4_CBGB                 978
#define BSIM4v4_CAPBD                979
#define BSIM4v4_CQBD                 980
#define BSIM4v4_CAPBS                981
#define BSIM4v4_CQBS                 982
#define BSIM4v4_CDGB                 983
#define BSIM4v4_CDDB                 984
#define BSIM4v4_CDSB                 985
#define BSIM4v4_VON                  986
#define BSIM4v4_VDSAT                987
#define BSIM4v4_QBS                  988
#define BSIM4v4_QBD                  989
#define BSIM4v4_SOURCECONDUCT        990
#define BSIM4v4_DRAINCONDUCT         991
#define BSIM4v4_CBDB                 992
#define BSIM4v4_CBSB                 993
#define BSIM4v4_CSUB                 994
#define BSIM4v4_QINV                 995
#define BSIM4v4_IGIDL                996
#define BSIM4v4_CSGB                 997
#define BSIM4v4_CSDB                 998
#define BSIM4v4_CSSB                 999
#define BSIM4v4_CGBB                 1000
#define BSIM4v4_CDBB                 1001
#define BSIM4v4_CSBB                 1002
#define BSIM4v4_CBBB                 1003
#define BSIM4v4_QS                   1004
#define BSIM4v4_IGISL                1005
#define BSIM4v4_IGS                  1006
#define BSIM4v4_IGD                  1007
#define BSIM4v4_IGB                  1008
#define BSIM4v4_IGCS                 1009
#define BSIM4v4_IGCD                 1010

#include "bsim4v4ext.h"


extern void BSIM4v4evaluate(double,double,double,BSIM4v4instance*,BSIM4v4model*,
        double*,double*,double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, CKTcircuit*);
extern int BSIM4v4debug(BSIM4v4model*, BSIM4v4instance*, CKTcircuit*, int);
extern int BSIM4v4checkModel(BSIM4v4model*, BSIM4v4instance*, CKTcircuit*);
extern int BSIM4v4PAeffGeo(double, int, int, double, double, double, double, double *, double *, double *, double *);
extern int BSIM4v4RdseffGeo(double, int, int, int, double, double, double, double, double, int, double *);

#endif /*BSIM4v4*/
