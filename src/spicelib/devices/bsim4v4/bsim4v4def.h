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

#include "ifsim.h"
#include "gendefs.h"
#include "cktdefs.h"
#include "complex.h"
#include "noisedef.h"         

typedef struct sBSIM4V4instance
{
    struct sBSIM4V4model *BSIM4V4modPtr;
    struct sBSIM4V4instance *BSIM4V4nextInstance;
    IFuid BSIM4V4name;
    int BSIM4V4owner;      /* Number of owner process */
    int BSIM4V4states;     /* index into state table for this device */
    int BSIM4V4dNode;
    int BSIM4V4gNodeExt;
    int BSIM4V4sNode;
    int BSIM4V4bNode;
    int BSIM4V4dNodePrime;
    int BSIM4V4gNodePrime;
    int BSIM4V4gNodeMid;
    int BSIM4V4sNodePrime;
    int BSIM4V4bNodePrime;
    int BSIM4V4dbNode;
    int BSIM4V4sbNode;
    int BSIM4V4qNode;

    double BSIM4V4ueff;
    double BSIM4V4thetavth; 
    double BSIM4V4von;
    double BSIM4V4vdsat;
    double BSIM4V4cgdo;
    double BSIM4V4qgdo;
    double BSIM4V4cgso;
    double BSIM4V4qgso;
    double BSIM4V4grbsb;
    double BSIM4V4grbdb;
    double BSIM4V4grbpb;
    double BSIM4V4grbps;
    double BSIM4V4grbpd;

    double BSIM4V4vjsmFwd;
    double BSIM4V4vjsmRev;
    double BSIM4V4vjdmFwd;
    double BSIM4V4vjdmRev;
    double BSIM4V4XExpBVS;
    double BSIM4V4XExpBVD;
    double BSIM4V4SslpFwd;
    double BSIM4V4SslpRev;
    double BSIM4V4DslpFwd;
    double BSIM4V4DslpRev;
    double BSIM4V4IVjsmFwd;
    double BSIM4V4IVjsmRev;
    double BSIM4V4IVjdmFwd;
    double BSIM4V4IVjdmRev;

    double BSIM4V4grgeltd;
    double BSIM4V4Pseff;
    double BSIM4V4Pdeff;
    double BSIM4V4Aseff;
    double BSIM4V4Adeff;

    double BSIM4V4l;
    double BSIM4V4w;
    double BSIM4V4drainArea;
    double BSIM4V4sourceArea;
    double BSIM4V4drainSquares;
    double BSIM4V4sourceSquares;
    double BSIM4V4drainPerimeter;
    double BSIM4V4sourcePerimeter;
    double BSIM4V4sourceConductance;
    double BSIM4V4drainConductance;
     /* stress effect instance param */
    double BSIM4V4sa;
    double BSIM4V4sb;
    double BSIM4V4sd;

    double BSIM4V4rbdb;
    double BSIM4V4rbsb;
    double BSIM4V4rbpb;
    double BSIM4V4rbps;
    double BSIM4V4rbpd;

     /* added here to account stress effect instance dependence */
    double BSIM4V4u0temp;
    double BSIM4V4vsattemp;
    double BSIM4V4vth0;
    double BSIM4V4vfb;
    double BSIM4V4vfbzb;
    double BSIM4V4vtfbphi1;
    double BSIM4V4vtfbphi2;
    double BSIM4V4k2;
    double BSIM4V4vbsc;
    double BSIM4V4k2ox;
    double BSIM4V4eta0;

    double BSIM4V4icVDS;
    double BSIM4V4icVGS;
    double BSIM4V4icVBS;
    double BSIM4V4nf;
    double BSIM4V4m;
    int BSIM4V4off;
    int BSIM4V4mode;
    int BSIM4V4trnqsMod;
    int BSIM4V4acnqsMod;
    int BSIM4V4rbodyMod;
    int BSIM4V4rgateMod;
    int BSIM4V4geoMod;
    int BSIM4V4rgeoMod;
    int BSIM4V4min;


    /* OP point */
    double BSIM4V4Vgsteff;
    double BSIM4V4vgs_eff;
    double BSIM4V4vgd_eff;
    double BSIM4V4dvgs_eff_dvg;
    double BSIM4V4dvgd_eff_dvg;
    double BSIM4V4Vdseff;
    double BSIM4V4nstar;
    double BSIM4V4Abulk;
    double BSIM4V4EsatL;
    double BSIM4V4AbovVgst2Vtm;
    double BSIM4V4qinv;
    double BSIM4V4cd;
    double BSIM4V4cbs;
    double BSIM4V4cbd;
    double BSIM4V4csub;
    double BSIM4V4Igidl;
    double BSIM4V4Igisl;
    double BSIM4V4gm;
    double BSIM4V4gds;
    double BSIM4V4gmbs;
    double BSIM4V4gbd;
    double BSIM4V4gbs;

    double BSIM4V4gbbs;
    double BSIM4V4gbgs;
    double BSIM4V4gbds;
    double BSIM4V4ggidld;
    double BSIM4V4ggidlg;
    double BSIM4V4ggidls;
    double BSIM4V4ggidlb;
    double BSIM4V4ggisld;
    double BSIM4V4ggislg;
    double BSIM4V4ggisls;
    double BSIM4V4ggislb;

    double BSIM4V4Igcs;
    double BSIM4V4gIgcsg;
    double BSIM4V4gIgcsd;
    double BSIM4V4gIgcss;
    double BSIM4V4gIgcsb;
    double BSIM4V4Igcd;
    double BSIM4V4gIgcdg;
    double BSIM4V4gIgcdd;
    double BSIM4V4gIgcds;
    double BSIM4V4gIgcdb;

    double BSIM4V4Igs;
    double BSIM4V4gIgsg;
    double BSIM4V4gIgss;
    double BSIM4V4Igd;
    double BSIM4V4gIgdg;
    double BSIM4V4gIgdd;

    double BSIM4V4Igb;
    double BSIM4V4gIgbg;
    double BSIM4V4gIgbd;
    double BSIM4V4gIgbs;
    double BSIM4V4gIgbb;

    double BSIM4V4grdsw;
    double BSIM4V4IdovVds;
    double BSIM4V4gcrg;
    double BSIM4V4gcrgd;
    double BSIM4V4gcrgg;
    double BSIM4V4gcrgs;
    double BSIM4V4gcrgb;

    double BSIM4V4gstot;
    double BSIM4V4gstotd;
    double BSIM4V4gstotg;
    double BSIM4V4gstots;
    double BSIM4V4gstotb;

    double BSIM4V4gdtot;
    double BSIM4V4gdtotd;
    double BSIM4V4gdtotg;
    double BSIM4V4gdtots;
    double BSIM4V4gdtotb;

    double BSIM4V4cggb;
    double BSIM4V4cgdb;
    double BSIM4V4cgsb;
    double BSIM4V4cbgb;
    double BSIM4V4cbdb;
    double BSIM4V4cbsb;
    double BSIM4V4cdgb;
    double BSIM4V4cddb;
    double BSIM4V4cdsb;
    double BSIM4V4csgb;
    double BSIM4V4csdb;
    double BSIM4V4cssb;
    double BSIM4V4cgbb;
    double BSIM4V4cdbb;
    double BSIM4V4csbb;
    double BSIM4V4cbbb;
    double BSIM4V4capbd;
    double BSIM4V4capbs;

    double BSIM4V4cqgb;
    double BSIM4V4cqdb;
    double BSIM4V4cqsb;
    double BSIM4V4cqbb;

    double BSIM4V4qgate;
    double BSIM4V4qbulk;
    double BSIM4V4qdrn;
    double BSIM4V4qsrc;

    double BSIM4V4qchqs;
    double BSIM4V4taunet;
    double BSIM4V4gtau;
    double BSIM4V4gtg;
    double BSIM4V4gtd;
    double BSIM4V4gts;
    double BSIM4V4gtb;
    double BSIM4V4SjctTempRevSatCur;
    double BSIM4V4DjctTempRevSatCur;
    double BSIM4V4SswTempRevSatCur;
    double BSIM4V4DswTempRevSatCur;
    double BSIM4V4SswgTempRevSatCur;
    double BSIM4V4DswgTempRevSatCur;

    struct bsim4SizeDependParam  *pParam;

    unsigned BSIM4V4lGiven :1;
    unsigned BSIM4V4wGiven :1;
    unsigned BSIM4V4mGiven :1;
    unsigned BSIM4V4nfGiven :1;
    unsigned BSIM4V4minGiven :1;
    unsigned BSIM4V4drainAreaGiven :1;
    unsigned BSIM4V4sourceAreaGiven    :1;
    unsigned BSIM4V4drainSquaresGiven  :1;
    unsigned BSIM4V4sourceSquaresGiven :1;
    unsigned BSIM4V4drainPerimeterGiven    :1;
    unsigned BSIM4V4sourcePerimeterGiven   :1;
    unsigned BSIM4V4saGiven :1;
    unsigned BSIM4V4sbGiven :1;
    unsigned BSIM4V4sdGiven :1;
    unsigned BSIM4V4rbdbGiven   :1;
    unsigned BSIM4V4rbsbGiven   :1;
    unsigned BSIM4V4rbpbGiven   :1;
    unsigned BSIM4V4rbpdGiven   :1;
    unsigned BSIM4V4rbpsGiven   :1;
    unsigned BSIM4V4icVDSGiven :1;
    unsigned BSIM4V4icVGSGiven :1;
    unsigned BSIM4V4icVBSGiven :1;
    unsigned BSIM4V4trnqsModGiven :1;
    unsigned BSIM4V4acnqsModGiven :1;
    unsigned BSIM4V4rbodyModGiven :1;
    unsigned BSIM4V4rgateModGiven :1;
    unsigned BSIM4V4geoModGiven :1;
    unsigned BSIM4V4rgeoModGiven :1;


    double *BSIM4V4DPdPtr;
    double *BSIM4V4DPdpPtr;
    double *BSIM4V4DPgpPtr;
    double *BSIM4V4DPgmPtr;
    double *BSIM4V4DPspPtr;
    double *BSIM4V4DPbpPtr;
    double *BSIM4V4DPdbPtr;

    double *BSIM4V4DdPtr;
    double *BSIM4V4DdpPtr;

    double *BSIM4V4GPdpPtr;
    double *BSIM4V4GPgpPtr;
    double *BSIM4V4GPgmPtr;
    double *BSIM4V4GPgePtr;
    double *BSIM4V4GPspPtr;
    double *BSIM4V4GPbpPtr;

    double *BSIM4V4GMdpPtr;
    double *BSIM4V4GMgpPtr;
    double *BSIM4V4GMgmPtr;
    double *BSIM4V4GMgePtr;
    double *BSIM4V4GMspPtr;
    double *BSIM4V4GMbpPtr;

    double *BSIM4V4GEdpPtr;
    double *BSIM4V4GEgpPtr;
    double *BSIM4V4GEgmPtr;
    double *BSIM4V4GEgePtr;
    double *BSIM4V4GEspPtr;
    double *BSIM4V4GEbpPtr;

    double *BSIM4V4SPdpPtr;
    double *BSIM4V4SPgpPtr;
    double *BSIM4V4SPgmPtr;
    double *BSIM4V4SPsPtr;
    double *BSIM4V4SPspPtr;
    double *BSIM4V4SPbpPtr;
    double *BSIM4V4SPsbPtr;

    double *BSIM4V4SspPtr;
    double *BSIM4V4SsPtr;

    double *BSIM4V4BPdpPtr;
    double *BSIM4V4BPgpPtr;
    double *BSIM4V4BPgmPtr;
    double *BSIM4V4BPspPtr;
    double *BSIM4V4BPdbPtr;
    double *BSIM4V4BPbPtr;
    double *BSIM4V4BPsbPtr;
    double *BSIM4V4BPbpPtr;

    double *BSIM4V4DBdpPtr;
    double *BSIM4V4DBdbPtr;
    double *BSIM4V4DBbpPtr;
    double *BSIM4V4DBbPtr;

    double *BSIM4V4SBspPtr;
    double *BSIM4V4SBbpPtr;
    double *BSIM4V4SBbPtr;
    double *BSIM4V4SBsbPtr;

    double *BSIM4V4BdbPtr;
    double *BSIM4V4BbpPtr;
    double *BSIM4V4BsbPtr;
    double *BSIM4V4BbPtr;

    double *BSIM4V4DgpPtr;
    double *BSIM4V4DspPtr;
    double *BSIM4V4DbpPtr;
    double *BSIM4V4SdpPtr;
    double *BSIM4V4SgpPtr;
    double *BSIM4V4SbpPtr;

    double *BSIM4V4QdpPtr;
    double *BSIM4V4QgpPtr;
    double *BSIM4V4QspPtr;
    double *BSIM4V4QbpPtr;
    double *BSIM4V4QqPtr;
    double *BSIM4V4DPqPtr;
    double *BSIM4V4GPqPtr;
    double *BSIM4V4SPqPtr;


#define BSIM4V4vbd BSIM4V4states+ 0
#define BSIM4V4vbs BSIM4V4states+ 1
#define BSIM4V4vgs BSIM4V4states+ 2
#define BSIM4V4vds BSIM4V4states+ 3
#define BSIM4V4vdbs BSIM4V4states+ 4
#define BSIM4V4vdbd BSIM4V4states+ 5
#define BSIM4V4vsbs BSIM4V4states+ 6
#define BSIM4V4vges BSIM4V4states+ 7
#define BSIM4V4vgms BSIM4V4states+ 8
#define BSIM4V4vses BSIM4V4states+ 9
#define BSIM4V4vdes BSIM4V4states+ 10

#define BSIM4V4qb BSIM4V4states+ 11
#define BSIM4V4cqb BSIM4V4states+ 12
#define BSIM4V4qg BSIM4V4states+ 13
#define BSIM4V4cqg BSIM4V4states+ 14
#define BSIM4V4qd BSIM4V4states+ 15
#define BSIM4V4cqd BSIM4V4states+ 16
#define BSIM4V4qgmid BSIM4V4states+ 17
#define BSIM4V4cqgmid BSIM4V4states+ 18

#define BSIM4V4qbs  BSIM4V4states+ 19
#define BSIM4V4cqbs  BSIM4V4states+ 20
#define BSIM4V4qbd  BSIM4V4states+ 21
#define BSIM4V4cqbd  BSIM4V4states+ 22

#define BSIM4V4qcheq BSIM4V4states+ 23
#define BSIM4V4cqcheq BSIM4V4states+ 24
#define BSIM4V4qcdump BSIM4V4states+ 25
#define BSIM4V4cqcdump BSIM4V4states+ 26
#define BSIM4V4qdef BSIM4V4states+ 27
#define BSIM4V4qs BSIM4V4states+ 28

#define BSIM4V4numStates 29


/* indices to the array of BSIM4V4 NOISE SOURCES */

#define BSIM4V4RDNOIZ       0
#define BSIM4V4RSNOIZ       1
#define BSIM4V4RGNOIZ       2
#define BSIM4V4RBPSNOIZ     3
#define BSIM4V4RBPDNOIZ     4
#define BSIM4V4RBPBNOIZ     5
#define BSIM4V4RBSBNOIZ     6
#define BSIM4V4RBDBNOIZ     7
#define BSIM4V4IDNOIZ       8
#define BSIM4V4FLNOIZ       9
#define BSIM4V4IGSNOIZ      10
#define BSIM4V4IGDNOIZ      11
#define BSIM4V4IGBNOIZ      12
#define BSIM4V4TOTNOIZ      13

#define BSIM4V4NSRCS        14  /* Number of BSIM4V4 noise sources */

#ifndef NONOISE
    double BSIM4V4nVar[NSTATVARS][BSIM4V4NSRCS];
#else /* NONOISE */
        double **BSIM4V4nVar;
#endif /* NONOISE */

} BSIM4V4instance ;

struct bsim4SizeDependParam
{
    double Width;
    double Length;
    double NFinger;

    double BSIM4V4cdsc;           
    double BSIM4V4cdscb;    
    double BSIM4V4cdscd;       
    double BSIM4V4cit;           
    double BSIM4V4nfactor;      
    double BSIM4V4xj;
    double BSIM4V4vsat;         
    double BSIM4V4at;         
    double BSIM4V4a0;   
    double BSIM4V4ags;      
    double BSIM4V4a1;         
    double BSIM4V4a2;         
    double BSIM4V4keta;     
    double BSIM4V4nsub;
    double BSIM4V4ndep;        
    double BSIM4V4nsd;
    double BSIM4V4phin;
    double BSIM4V4ngate;        
    double BSIM4V4gamma1;      
    double BSIM4V4gamma2;     
    double BSIM4V4vbx;      
    double BSIM4V4vbi;       
    double BSIM4V4vbm;       
    double BSIM4V4vbsc;       
    double BSIM4V4xt;       
    double BSIM4V4phi;
    double BSIM4V4litl;
    double BSIM4V4k1;
    double BSIM4V4kt1;
    double BSIM4V4kt1l;
    double BSIM4V4kt2;
    double BSIM4V4k2;
    double BSIM4V4k3;
    double BSIM4V4k3b;
    double BSIM4V4w0;
    double BSIM4V4dvtp0;
    double BSIM4V4dvtp1;
    double BSIM4V4lpe0;
    double BSIM4V4lpeb;
    double BSIM4V4dvt0;      
    double BSIM4V4dvt1;      
    double BSIM4V4dvt2;      
    double BSIM4V4dvt0w;      
    double BSIM4V4dvt1w;      
    double BSIM4V4dvt2w;      
    double BSIM4V4drout;      
    double BSIM4V4dsub;      
    double BSIM4V4vth0;
    double BSIM4V4ua;
    double BSIM4V4ua1;
    double BSIM4V4ub;
    double BSIM4V4ub1;
    double BSIM4V4uc;
    double BSIM4V4uc1;
    double BSIM4V4u0;
    double BSIM4V4eu;
    double BSIM4V4ute;
    double BSIM4V4voff;
    double BSIM4V4minv;
    double BSIM4V4vfb;
    double BSIM4V4delta;
    double BSIM4V4rdsw;       
    double BSIM4V4rds0;       
    double BSIM4V4rs0;
    double BSIM4V4rd0;
    double BSIM4V4rsw;
    double BSIM4V4rdw;
    double BSIM4V4prwg;       
    double BSIM4V4prwb;       
    double BSIM4V4prt;       
    double BSIM4V4eta0;         
    double BSIM4V4etab;         
    double BSIM4V4pclm;      
    double BSIM4V4pdibl1;      
    double BSIM4V4pdibl2;      
    double BSIM4V4pdiblb;      
    double BSIM4V4fprout;
    double BSIM4V4pdits;
    double BSIM4V4pditsd;
    double BSIM4V4pscbe1;       
    double BSIM4V4pscbe2;       
    double BSIM4V4pvag;       
    double BSIM4V4wr;
    double BSIM4V4dwg;
    double BSIM4V4dwb;
    double BSIM4V4b0;
    double BSIM4V4b1;
    double BSIM4V4alpha0;
    double BSIM4V4alpha1;
    double BSIM4V4beta0;
    double BSIM4V4agidl;
    double BSIM4V4bgidl;
    double BSIM4V4cgidl;
    double BSIM4V4egidl;
    double BSIM4V4aigc;
    double BSIM4V4bigc;
    double BSIM4V4cigc;
    double BSIM4V4aigsd;
    double BSIM4V4bigsd;
    double BSIM4V4cigsd;
    double BSIM4V4aigbacc;
    double BSIM4V4bigbacc;
    double BSIM4V4cigbacc;
    double BSIM4V4aigbinv;
    double BSIM4V4bigbinv;
    double BSIM4V4cigbinv;
    double BSIM4V4nigc;
    double BSIM4V4nigbacc;
    double BSIM4V4nigbinv;
    double BSIM4V4ntox;
    double BSIM4V4eigbinv;
    double BSIM4V4pigcd;
    double BSIM4V4poxedge;
    double BSIM4V4xrcrg1;
    double BSIM4V4xrcrg2;
    double BSIM4V4lambda; /* overshoot */
    double BSIM4V4vtl; /* thermal velocity limit */
    double BSIM4V4xn; /* back scattering parameter */
    double BSIM4V4lc; /* back scattering parameter */
    double BSIM4V4tfactor;  /* ballistic transportation factor  */
    double BSIM4V4vfbsdoff;  /* S/D flatband offset voltage  */

/* added for stress effect */
    double BSIM4V4ku0;
    double BSIM4V4kvth0;
    double BSIM4V4ku0temp;
    double BSIM4V4rho_ref;
    double BSIM4V4inv_od_ref;

    /* CV model */
    double BSIM4V4cgsl;
    double BSIM4V4cgdl;
    double BSIM4V4ckappas;
    double BSIM4V4ckappad;
    double BSIM4V4cf;
    double BSIM4V4clc;
    double BSIM4V4cle;
    double BSIM4V4vfbcv;
    double BSIM4V4noff;
    double BSIM4V4voffcv;
    double BSIM4V4acde;
    double BSIM4V4moin;

/* Pre-calculated constants */

    double BSIM4V4dw;
    double BSIM4V4dl;
    double BSIM4V4leff;
    double BSIM4V4weff;

    double BSIM4V4dwc;
    double BSIM4V4dlc;
    double BSIM4V4dlcig;
    double BSIM4V4dwj;
    double BSIM4V4leffCV;
    double BSIM4V4weffCV;
    double BSIM4V4weffCJ;
    double BSIM4V4abulkCVfactor;
    double BSIM4V4cgso;
    double BSIM4V4cgdo;
    double BSIM4V4cgbo;

    double BSIM4V4u0temp;       
    double BSIM4V4vsattemp;   
    double BSIM4V4sqrtPhi;   
    double BSIM4V4phis3;   
    double BSIM4V4Xdep0;          
    double BSIM4V4sqrtXdep0;          
    double BSIM4V4theta0vb0;
    double BSIM4V4thetaRout; 
    double BSIM4V4mstar;
    double BSIM4V4voffcbn;
    double BSIM4V4rdswmin;
    double BSIM4V4rdwmin;
    double BSIM4V4rswmin;
    double BSIM4V4vfbsd;

    double BSIM4V4cof1;
    double BSIM4V4cof2;
    double BSIM4V4cof3;
    double BSIM4V4cof4;
    double BSIM4V4cdep0;
    double BSIM4V4vfbzb;
    double BSIM4V4vtfbphi1;
    double BSIM4V4vtfbphi2;
    double BSIM4V4ToxRatio;
    double BSIM4V4Aechvb;
    double BSIM4V4Bechvb;
    double BSIM4V4ToxRatioEdge;
    double BSIM4V4AechvbEdge;
    double BSIM4V4BechvbEdge;
    double BSIM4V4ldeb;
    double BSIM4V4k1ox;
    double BSIM4V4k2ox;

    struct bsim4SizeDependParam  *pNext;
};


typedef struct sBSIM4V4model 
{
    int BSIM4V4modType;
    struct sBSIM4V4model *BSIM4V4nextModel;
    BSIM4V4instance *BSIM4V4instances;
    IFuid BSIM4V4modName; 
    int BSIM4V4type;

    int    BSIM4V4mobMod;
    int    BSIM4V4capMod;
    int    BSIM4V4dioMod;
    int    BSIM4V4trnqsMod;
    int    BSIM4V4acnqsMod;
    int    BSIM4V4fnoiMod;
    int    BSIM4V4tnoiMod;
    int    BSIM4V4rdsMod;
    int    BSIM4V4rbodyMod;
    int    BSIM4V4rgateMod;
    int    BSIM4V4perMod;
    int    BSIM4V4geoMod;
    int    BSIM4V4igcMod;
    int    BSIM4V4igbMod;
    int    BSIM4V4tempMod;
    int    BSIM4V4binUnit;
    int    BSIM4V4paramChk;
    char   *BSIM4V4version;             
    double BSIM4V4toxe;             
    double BSIM4V4toxp;
    double BSIM4V4toxm;
    double BSIM4V4dtox;
    double BSIM4V4epsrox;
    double BSIM4V4cdsc;           
    double BSIM4V4cdscb; 
    double BSIM4V4cdscd;          
    double BSIM4V4cit;           
    double BSIM4V4nfactor;      
    double BSIM4V4xj;
    double BSIM4V4vsat;         
    double BSIM4V4at;         
    double BSIM4V4a0;   
    double BSIM4V4ags;      
    double BSIM4V4a1;         
    double BSIM4V4a2;         
    double BSIM4V4keta;     
    double BSIM4V4nsub;
    double BSIM4V4ndep;        
    double BSIM4V4nsd;
    double BSIM4V4phin;
    double BSIM4V4ngate;        
    double BSIM4V4gamma1;      
    double BSIM4V4gamma2;     
    double BSIM4V4vbx;      
    double BSIM4V4vbm;       
    double BSIM4V4xt;       
    double BSIM4V4k1;
    double BSIM4V4kt1;
    double BSIM4V4kt1l;
    double BSIM4V4kt2;
    double BSIM4V4k2;
    double BSIM4V4k3;
    double BSIM4V4k3b;
    double BSIM4V4w0;
    double BSIM4V4dvtp0;
    double BSIM4V4dvtp1;
    double BSIM4V4lpe0;
    double BSIM4V4lpeb;
    double BSIM4V4dvt0;      
    double BSIM4V4dvt1;      
    double BSIM4V4dvt2;      
    double BSIM4V4dvt0w;      
    double BSIM4V4dvt1w;      
    double BSIM4V4dvt2w;      
    double BSIM4V4drout;      
    double BSIM4V4dsub;      
    double BSIM4V4vth0;
    double BSIM4V4eu;
    double BSIM4V4ua;
    double BSIM4V4ua1;
    double BSIM4V4ub;
    double BSIM4V4ub1;
    double BSIM4V4uc;
    double BSIM4V4uc1;
    double BSIM4V4u0;
    double BSIM4V4ute;
    double BSIM4V4voff;
    double BSIM4V4minv;
    double BSIM4V4voffl;
    double BSIM4V4delta;
    double BSIM4V4rdsw;       
    double BSIM4V4rdswmin;
    double BSIM4V4rdwmin;
    double BSIM4V4rswmin;
    double BSIM4V4rsw;
    double BSIM4V4rdw;
    double BSIM4V4prwg;
    double BSIM4V4prwb;
    double BSIM4V4prt;       
    double BSIM4V4eta0;         
    double BSIM4V4etab;         
    double BSIM4V4pclm;      
    double BSIM4V4pdibl1;      
    double BSIM4V4pdibl2;      
    double BSIM4V4pdiblb;
    double BSIM4V4fprout;
    double BSIM4V4pdits;
    double BSIM4V4pditsd;
    double BSIM4V4pditsl;
    double BSIM4V4pscbe1;       
    double BSIM4V4pscbe2;       
    double BSIM4V4pvag;       
    double BSIM4V4wr;
    double BSIM4V4dwg;
    double BSIM4V4dwb;
    double BSIM4V4b0;
    double BSIM4V4b1;
    double BSIM4V4alpha0;
    double BSIM4V4alpha1;
    double BSIM4V4beta0;
    double BSIM4V4agidl;
    double BSIM4V4bgidl;
    double BSIM4V4cgidl;
    double BSIM4V4egidl;
    double BSIM4V4aigc;
    double BSIM4V4bigc;
    double BSIM4V4cigc;
    double BSIM4V4aigsd;
    double BSIM4V4bigsd;
    double BSIM4V4cigsd;
    double BSIM4V4aigbacc;
    double BSIM4V4bigbacc;
    double BSIM4V4cigbacc;
    double BSIM4V4aigbinv;
    double BSIM4V4bigbinv;
    double BSIM4V4cigbinv;
    double BSIM4V4nigc;
    double BSIM4V4nigbacc;
    double BSIM4V4nigbinv;
    double BSIM4V4ntox;
    double BSIM4V4eigbinv;
    double BSIM4V4pigcd;
    double BSIM4V4poxedge;
    double BSIM4V4toxref;
    double BSIM4V4ijthdfwd;
    double BSIM4V4ijthsfwd;
    double BSIM4V4ijthdrev;
    double BSIM4V4ijthsrev;
    double BSIM4V4xjbvd;
    double BSIM4V4xjbvs;
    double BSIM4V4bvd;
    double BSIM4V4bvs;

    double BSIM4V4jtss;
    double BSIM4V4jtsd;
    double BSIM4V4jtssws;
    double BSIM4V4jtsswd;
    double BSIM4V4jtsswgs;
    double BSIM4V4jtsswgd;
    double BSIM4V4njts;
    double BSIM4V4njtssw;
    double BSIM4V4njtsswg;
    double BSIM4V4xtss;
    double BSIM4V4xtsd;
    double BSIM4V4xtssws;
    double BSIM4V4xtsswd;
    double BSIM4V4xtsswgs;
    double BSIM4V4xtsswgd;
    double BSIM4V4tnjts;
    double BSIM4V4tnjtssw;
    double BSIM4V4tnjtsswg;
    double BSIM4V4vtss;
    double BSIM4V4vtsd;
    double BSIM4V4vtssws;
    double BSIM4V4vtsswd;
    double BSIM4V4vtsswgs;
    double BSIM4V4vtsswgd;

    double BSIM4V4xrcrg1;
    double BSIM4V4xrcrg2;
    double BSIM4V4lambda;
    double BSIM4V4vtl; 
    double BSIM4V4lc; 
    double BSIM4V4xn; 
    double BSIM4V4vfbsdoff;  /* S/D flatband offset voltage  */
    double BSIM4V4lintnoi;  /* lint offset for noise calculation  */

    double BSIM4V4vfb;
    double BSIM4V4gbmin;
    double BSIM4V4rbdb;
    double BSIM4V4rbsb;
    double BSIM4V4rbpb;
    double BSIM4V4rbps;
    double BSIM4V4rbpd;
    double BSIM4V4tnoia;
    double BSIM4V4tnoib;
    double BSIM4V4rnoia;
    double BSIM4V4rnoib;
    double BSIM4V4ntnoi;

    /* CV model and Parasitics */
    double BSIM4V4cgsl;
    double BSIM4V4cgdl;
    double BSIM4V4ckappas;
    double BSIM4V4ckappad;
    double BSIM4V4cf;
    double BSIM4V4vfbcv;
    double BSIM4V4clc;
    double BSIM4V4cle;
    double BSIM4V4dwc;
    double BSIM4V4dlc;
    double BSIM4V4xw;
    double BSIM4V4xl;
    double BSIM4V4dlcig;
    double BSIM4V4dwj;
    double BSIM4V4noff;
    double BSIM4V4voffcv;
    double BSIM4V4acde;
    double BSIM4V4moin;
    double BSIM4V4tcj;
    double BSIM4V4tcjsw;
    double BSIM4V4tcjswg;
    double BSIM4V4tpb;
    double BSIM4V4tpbsw;
    double BSIM4V4tpbswg;
    double BSIM4V4dmcg;
    double BSIM4V4dmci;
    double BSIM4V4dmdg;
    double BSIM4V4dmcgt;
    double BSIM4V4xgw;
    double BSIM4V4xgl;
    double BSIM4V4rshg;
    double BSIM4V4ngcon;

    /* Length Dependence */
    double BSIM4V4lcdsc;           
    double BSIM4V4lcdscb; 
    double BSIM4V4lcdscd;          
    double BSIM4V4lcit;           
    double BSIM4V4lnfactor;      
    double BSIM4V4lxj;
    double BSIM4V4lvsat;         
    double BSIM4V4lat;         
    double BSIM4V4la0;   
    double BSIM4V4lags;      
    double BSIM4V4la1;         
    double BSIM4V4la2;         
    double BSIM4V4lketa;     
    double BSIM4V4lnsub;
    double BSIM4V4lndep;        
    double BSIM4V4lnsd;
    double BSIM4V4lphin;
    double BSIM4V4lngate;        
    double BSIM4V4lgamma1;      
    double BSIM4V4lgamma2;     
    double BSIM4V4lvbx;      
    double BSIM4V4lvbm;       
    double BSIM4V4lxt;       
    double BSIM4V4lk1;
    double BSIM4V4lkt1;
    double BSIM4V4lkt1l;
    double BSIM4V4lkt2;
    double BSIM4V4lk2;
    double BSIM4V4lk3;
    double BSIM4V4lk3b;
    double BSIM4V4lw0;
    double BSIM4V4ldvtp0;
    double BSIM4V4ldvtp1;
    double BSIM4V4llpe0;
    double BSIM4V4llpeb;
    double BSIM4V4ldvt0;      
    double BSIM4V4ldvt1;      
    double BSIM4V4ldvt2;      
    double BSIM4V4ldvt0w;      
    double BSIM4V4ldvt1w;      
    double BSIM4V4ldvt2w;      
    double BSIM4V4ldrout;      
    double BSIM4V4ldsub;      
    double BSIM4V4lvth0;
    double BSIM4V4lua;
    double BSIM4V4lua1;
    double BSIM4V4lub;
    double BSIM4V4lub1;
    double BSIM4V4luc;
    double BSIM4V4luc1;
    double BSIM4V4lu0;
    double BSIM4V4leu;
    double BSIM4V4lute;
    double BSIM4V4lvoff;
    double BSIM4V4lminv;
    double BSIM4V4ldelta;
    double BSIM4V4lrdsw;       
    double BSIM4V4lrsw;
    double BSIM4V4lrdw;
    double BSIM4V4lprwg;
    double BSIM4V4lprwb;
    double BSIM4V4lprt;       
    double BSIM4V4leta0;         
    double BSIM4V4letab;         
    double BSIM4V4lpclm;      
    double BSIM4V4lpdibl1;      
    double BSIM4V4lpdibl2;      
    double BSIM4V4lpdiblb;
    double BSIM4V4lfprout;
    double BSIM4V4lpdits;
    double BSIM4V4lpditsd;
    double BSIM4V4lpscbe1;       
    double BSIM4V4lpscbe2;       
    double BSIM4V4lpvag;       
    double BSIM4V4lwr;
    double BSIM4V4ldwg;
    double BSIM4V4ldwb;
    double BSIM4V4lb0;
    double BSIM4V4lb1;
    double BSIM4V4lalpha0;
    double BSIM4V4lalpha1;
    double BSIM4V4lbeta0;
    double BSIM4V4lvfb;
    double BSIM4V4lagidl;
    double BSIM4V4lbgidl;
    double BSIM4V4lcgidl;
    double BSIM4V4legidl;
    double BSIM4V4laigc;
    double BSIM4V4lbigc;
    double BSIM4V4lcigc;
    double BSIM4V4laigsd;
    double BSIM4V4lbigsd;
    double BSIM4V4lcigsd;
    double BSIM4V4laigbacc;
    double BSIM4V4lbigbacc;
    double BSIM4V4lcigbacc;
    double BSIM4V4laigbinv;
    double BSIM4V4lbigbinv;
    double BSIM4V4lcigbinv;
    double BSIM4V4lnigc;
    double BSIM4V4lnigbacc;
    double BSIM4V4lnigbinv;
    double BSIM4V4lntox;
    double BSIM4V4leigbinv;
    double BSIM4V4lpigcd;
    double BSIM4V4lpoxedge;
    double BSIM4V4lxrcrg1;
    double BSIM4V4lxrcrg2;
    double BSIM4V4llambda;
    double BSIM4V4lvtl; 
    double BSIM4V4lxn; 
    double BSIM4V4lvfbsdoff; 

    /* CV model */
    double BSIM4V4lcgsl;
    double BSIM4V4lcgdl;
    double BSIM4V4lckappas;
    double BSIM4V4lckappad;
    double BSIM4V4lcf;
    double BSIM4V4lclc;
    double BSIM4V4lcle;
    double BSIM4V4lvfbcv;
    double BSIM4V4lnoff;
    double BSIM4V4lvoffcv;
    double BSIM4V4lacde;
    double BSIM4V4lmoin;

    /* Width Dependence */
    double BSIM4V4wcdsc;           
    double BSIM4V4wcdscb; 
    double BSIM4V4wcdscd;          
    double BSIM4V4wcit;           
    double BSIM4V4wnfactor;      
    double BSIM4V4wxj;
    double BSIM4V4wvsat;         
    double BSIM4V4wat;         
    double BSIM4V4wa0;   
    double BSIM4V4wags;      
    double BSIM4V4wa1;         
    double BSIM4V4wa2;         
    double BSIM4V4wketa;     
    double BSIM4V4wnsub;
    double BSIM4V4wndep;        
    double BSIM4V4wnsd;
    double BSIM4V4wphin;
    double BSIM4V4wngate;        
    double BSIM4V4wgamma1;      
    double BSIM4V4wgamma2;     
    double BSIM4V4wvbx;      
    double BSIM4V4wvbm;       
    double BSIM4V4wxt;       
    double BSIM4V4wk1;
    double BSIM4V4wkt1;
    double BSIM4V4wkt1l;
    double BSIM4V4wkt2;
    double BSIM4V4wk2;
    double BSIM4V4wk3;
    double BSIM4V4wk3b;
    double BSIM4V4ww0;
    double BSIM4V4wdvtp0;
    double BSIM4V4wdvtp1;
    double BSIM4V4wlpe0;
    double BSIM4V4wlpeb;
    double BSIM4V4wdvt0;      
    double BSIM4V4wdvt1;      
    double BSIM4V4wdvt2;      
    double BSIM4V4wdvt0w;      
    double BSIM4V4wdvt1w;      
    double BSIM4V4wdvt2w;      
    double BSIM4V4wdrout;      
    double BSIM4V4wdsub;      
    double BSIM4V4wvth0;
    double BSIM4V4wua;
    double BSIM4V4wua1;
    double BSIM4V4wub;
    double BSIM4V4wub1;
    double BSIM4V4wuc;
    double BSIM4V4wuc1;
    double BSIM4V4wu0;
    double BSIM4V4weu;
    double BSIM4V4wute;
    double BSIM4V4wvoff;
    double BSIM4V4wminv;
    double BSIM4V4wdelta;
    double BSIM4V4wrdsw;       
    double BSIM4V4wrsw;
    double BSIM4V4wrdw;
    double BSIM4V4wprwg;
    double BSIM4V4wprwb;
    double BSIM4V4wprt;       
    double BSIM4V4weta0;         
    double BSIM4V4wetab;         
    double BSIM4V4wpclm;      
    double BSIM4V4wpdibl1;      
    double BSIM4V4wpdibl2;      
    double BSIM4V4wpdiblb;
    double BSIM4V4wfprout;
    double BSIM4V4wpdits;
    double BSIM4V4wpditsd;
    double BSIM4V4wpscbe1;       
    double BSIM4V4wpscbe2;       
    double BSIM4V4wpvag;       
    double BSIM4V4wwr;
    double BSIM4V4wdwg;
    double BSIM4V4wdwb;
    double BSIM4V4wb0;
    double BSIM4V4wb1;
    double BSIM4V4walpha0;
    double BSIM4V4walpha1;
    double BSIM4V4wbeta0;
    double BSIM4V4wvfb;
    double BSIM4V4wagidl;
    double BSIM4V4wbgidl;
    double BSIM4V4wcgidl;
    double BSIM4V4wegidl;
    double BSIM4V4waigc;
    double BSIM4V4wbigc;
    double BSIM4V4wcigc;
    double BSIM4V4waigsd;
    double BSIM4V4wbigsd;
    double BSIM4V4wcigsd;
    double BSIM4V4waigbacc;
    double BSIM4V4wbigbacc;
    double BSIM4V4wcigbacc;
    double BSIM4V4waigbinv;
    double BSIM4V4wbigbinv;
    double BSIM4V4wcigbinv;
    double BSIM4V4wnigc;
    double BSIM4V4wnigbacc;
    double BSIM4V4wnigbinv;
    double BSIM4V4wntox;
    double BSIM4V4weigbinv;
    double BSIM4V4wpigcd;
    double BSIM4V4wpoxedge;
    double BSIM4V4wxrcrg1;
    double BSIM4V4wxrcrg2;
    double BSIM4V4wlambda;
    double BSIM4V4wvtl; 
    double BSIM4V4wxn; 
    double BSIM4V4wvfbsdoff;  

    /* CV model */
    double BSIM4V4wcgsl;
    double BSIM4V4wcgdl;
    double BSIM4V4wckappas;
    double BSIM4V4wckappad;
    double BSIM4V4wcf;
    double BSIM4V4wclc;
    double BSIM4V4wcle;
    double BSIM4V4wvfbcv;
    double BSIM4V4wnoff;
    double BSIM4V4wvoffcv;
    double BSIM4V4wacde;
    double BSIM4V4wmoin;

    /* Cross-term Dependence */
    double BSIM4V4pcdsc;           
    double BSIM4V4pcdscb; 
    double BSIM4V4pcdscd;          
    double BSIM4V4pcit;           
    double BSIM4V4pnfactor;      
    double BSIM4V4pxj;
    double BSIM4V4pvsat;         
    double BSIM4V4pat;         
    double BSIM4V4pa0;   
    double BSIM4V4pags;      
    double BSIM4V4pa1;         
    double BSIM4V4pa2;         
    double BSIM4V4pketa;     
    double BSIM4V4pnsub;
    double BSIM4V4pndep;        
    double BSIM4V4pnsd;
    double BSIM4V4pphin;
    double BSIM4V4pngate;        
    double BSIM4V4pgamma1;      
    double BSIM4V4pgamma2;     
    double BSIM4V4pvbx;      
    double BSIM4V4pvbm;       
    double BSIM4V4pxt;       
    double BSIM4V4pk1;
    double BSIM4V4pkt1;
    double BSIM4V4pkt1l;
    double BSIM4V4pkt2;
    double BSIM4V4pk2;
    double BSIM4V4pk3;
    double BSIM4V4pk3b;
    double BSIM4V4pw0;
    double BSIM4V4pdvtp0;
    double BSIM4V4pdvtp1;
    double BSIM4V4plpe0;
    double BSIM4V4plpeb;
    double BSIM4V4pdvt0;      
    double BSIM4V4pdvt1;      
    double BSIM4V4pdvt2;      
    double BSIM4V4pdvt0w;      
    double BSIM4V4pdvt1w;      
    double BSIM4V4pdvt2w;      
    double BSIM4V4pdrout;      
    double BSIM4V4pdsub;      
    double BSIM4V4pvth0;
    double BSIM4V4pua;
    double BSIM4V4pua1;
    double BSIM4V4pub;
    double BSIM4V4pub1;
    double BSIM4V4puc;
    double BSIM4V4puc1;
    double BSIM4V4pu0;
    double BSIM4V4peu;
    double BSIM4V4pute;
    double BSIM4V4pvoff;
    double BSIM4V4pminv;
    double BSIM4V4pdelta;
    double BSIM4V4prdsw;
    double BSIM4V4prsw;
    double BSIM4V4prdw;
    double BSIM4V4pprwg;
    double BSIM4V4pprwb;
    double BSIM4V4pprt;       
    double BSIM4V4peta0;         
    double BSIM4V4petab;         
    double BSIM4V4ppclm;      
    double BSIM4V4ppdibl1;      
    double BSIM4V4ppdibl2;      
    double BSIM4V4ppdiblb;
    double BSIM4V4pfprout;
    double BSIM4V4ppdits;
    double BSIM4V4ppditsd;
    double BSIM4V4ppscbe1;       
    double BSIM4V4ppscbe2;       
    double BSIM4V4ppvag;       
    double BSIM4V4pwr;
    double BSIM4V4pdwg;
    double BSIM4V4pdwb;
    double BSIM4V4pb0;
    double BSIM4V4pb1;
    double BSIM4V4palpha0;
    double BSIM4V4palpha1;
    double BSIM4V4pbeta0;
    double BSIM4V4pvfb;
    double BSIM4V4pagidl;
    double BSIM4V4pbgidl;
    double BSIM4V4pcgidl;
    double BSIM4V4pegidl;
    double BSIM4V4paigc;
    double BSIM4V4pbigc;
    double BSIM4V4pcigc;
    double BSIM4V4paigsd;
    double BSIM4V4pbigsd;
    double BSIM4V4pcigsd;
    double BSIM4V4paigbacc;
    double BSIM4V4pbigbacc;
    double BSIM4V4pcigbacc;
    double BSIM4V4paigbinv;
    double BSIM4V4pbigbinv;
    double BSIM4V4pcigbinv;
    double BSIM4V4pnigc;
    double BSIM4V4pnigbacc;
    double BSIM4V4pnigbinv;
    double BSIM4V4pntox;
    double BSIM4V4peigbinv;
    double BSIM4V4ppigcd;
    double BSIM4V4ppoxedge;
    double BSIM4V4pxrcrg1;
    double BSIM4V4pxrcrg2;
    double BSIM4V4plambda;
    double BSIM4V4pvtl;
    double BSIM4V4pxn; 
    double BSIM4V4pvfbsdoff;  

    /* CV model */
    double BSIM4V4pcgsl;
    double BSIM4V4pcgdl;
    double BSIM4V4pckappas;
    double BSIM4V4pckappad;
    double BSIM4V4pcf;
    double BSIM4V4pclc;
    double BSIM4V4pcle;
    double BSIM4V4pvfbcv;
    double BSIM4V4pnoff;
    double BSIM4V4pvoffcv;
    double BSIM4V4pacde;
    double BSIM4V4pmoin;

    double BSIM4V4tnom;
    double BSIM4V4cgso;
    double BSIM4V4cgdo;
    double BSIM4V4cgbo;
    double BSIM4V4xpart;
    double BSIM4V4cFringOut;
    double BSIM4V4cFringMax;

    double BSIM4V4sheetResistance;
    double BSIM4V4SjctSatCurDensity;
    double BSIM4V4DjctSatCurDensity;
    double BSIM4V4SjctSidewallSatCurDensity;
    double BSIM4V4DjctSidewallSatCurDensity;
    double BSIM4V4SjctGateSidewallSatCurDensity;
    double BSIM4V4DjctGateSidewallSatCurDensity;
    double BSIM4V4SbulkJctPotential;
    double BSIM4V4DbulkJctPotential;
    double BSIM4V4SbulkJctBotGradingCoeff;
    double BSIM4V4DbulkJctBotGradingCoeff;
    double BSIM4V4SbulkJctSideGradingCoeff;
    double BSIM4V4DbulkJctSideGradingCoeff;
    double BSIM4V4SbulkJctGateSideGradingCoeff;
    double BSIM4V4DbulkJctGateSideGradingCoeff;
    double BSIM4V4SsidewallJctPotential;
    double BSIM4V4DsidewallJctPotential;
    double BSIM4V4SGatesidewallJctPotential;
    double BSIM4V4DGatesidewallJctPotential;
    double BSIM4V4SunitAreaJctCap;
    double BSIM4V4DunitAreaJctCap;
    double BSIM4V4SunitLengthSidewallJctCap;
    double BSIM4V4DunitLengthSidewallJctCap;
    double BSIM4V4SunitLengthGateSidewallJctCap;
    double BSIM4V4DunitLengthGateSidewallJctCap;
    double BSIM4V4SjctEmissionCoeff;
    double BSIM4V4DjctEmissionCoeff;
    double BSIM4V4SjctTempExponent;
    double BSIM4V4DjctTempExponent;
    double BSIM4V4njtstemp;
    double BSIM4V4njtsswtemp;
    double BSIM4V4njtsswgtemp;

    double BSIM4V4Lint;
    double BSIM4V4Ll;
    double BSIM4V4Llc;
    double BSIM4V4Lln;
    double BSIM4V4Lw;
    double BSIM4V4Lwc;
    double BSIM4V4Lwn;
    double BSIM4V4Lwl;
    double BSIM4V4Lwlc;
    double BSIM4V4Lmin;
    double BSIM4V4Lmax;

    double BSIM4V4Wint;
    double BSIM4V4Wl;
    double BSIM4V4Wlc;
    double BSIM4V4Wln;
    double BSIM4V4Ww;
    double BSIM4V4Wwc;
    double BSIM4V4Wwn;
    double BSIM4V4Wwl;
    double BSIM4V4Wwlc;
    double BSIM4V4Wmin;
    double BSIM4V4Wmax;

    /* added for stress effect */
    double BSIM4V4saref;
    double BSIM4V4sbref;
    double BSIM4V4wlod;
    double BSIM4V4ku0;
    double BSIM4V4kvsat;
    double BSIM4V4kvth0;
    double BSIM4V4tku0;
    double BSIM4V4llodku0;
    double BSIM4V4wlodku0;
    double BSIM4V4llodvth;
    double BSIM4V4wlodvth;
    double BSIM4V4lku0;
    double BSIM4V4wku0;
    double BSIM4V4pku0;
    double BSIM4V4lkvth0;
    double BSIM4V4wkvth0;
    double BSIM4V4pkvth0;
    double BSIM4V4stk2;
    double BSIM4V4lodk2;
    double BSIM4V4steta0;
    double BSIM4V4lodeta0;



/* Pre-calculated constants
 * move to size-dependent param */
    double BSIM4V4vtm;   
    double BSIM4V4vtm0;   
    double BSIM4V4coxe;
    double BSIM4V4coxp;
    double BSIM4V4cof1;
    double BSIM4V4cof2;
    double BSIM4V4cof3;
    double BSIM4V4cof4;
    double BSIM4V4vcrit;
    double BSIM4V4factor1;
    double BSIM4V4PhiBS;
    double BSIM4V4PhiBSWS;
    double BSIM4V4PhiBSWGS;
    double BSIM4V4SjctTempSatCurDensity;
    double BSIM4V4SjctSidewallTempSatCurDensity;
    double BSIM4V4SjctGateSidewallTempSatCurDensity;
    double BSIM4V4PhiBD;
    double BSIM4V4PhiBSWD;
    double BSIM4V4PhiBSWGD;
    double BSIM4V4DjctTempSatCurDensity;
    double BSIM4V4DjctSidewallTempSatCurDensity;
    double BSIM4V4DjctGateSidewallTempSatCurDensity;
    double BSIM4V4SunitAreaTempJctCap;
    double BSIM4V4DunitAreaTempJctCap;
    double BSIM4V4SunitLengthSidewallTempJctCap;
    double BSIM4V4DunitLengthSidewallTempJctCap;
    double BSIM4V4SunitLengthGateSidewallTempJctCap;
    double BSIM4V4DunitLengthGateSidewallTempJctCap;

    double BSIM4V4oxideTrapDensityA;      
    double BSIM4V4oxideTrapDensityB;     
    double BSIM4V4oxideTrapDensityC;  
    double BSIM4V4em;  
    double BSIM4V4ef;  
    double BSIM4V4af;  
    double BSIM4V4kf;  

    int    BSIM4V4rgeomod;

    /* strain parameters */
    double BSIM4V4stimod;
    double BSIM4V4sa0;
    double BSIM4V4sb0;

    struct bsim4SizeDependParam *pSizeDependParamKnot;

    /* Flags */
    unsigned BSIM4V4rgeomodGiven :1;
    unsigned BSIM4V4stimodGiven :1;
    unsigned BSIM4V4sa0Given :1;
    unsigned BSIM4V4sb0Given :1;

    unsigned  BSIM4V4mobModGiven :1;
    unsigned  BSIM4V4binUnitGiven :1;
    unsigned  BSIM4V4capModGiven :1;
    unsigned  BSIM4V4dioModGiven :1;
    unsigned  BSIM4V4rdsModGiven :1;
    unsigned  BSIM4V4rbodyModGiven :1;
    unsigned  BSIM4V4rgateModGiven :1;
    unsigned  BSIM4V4perModGiven :1;
    unsigned  BSIM4V4geoModGiven :1;
    unsigned  BSIM4V4paramChkGiven :1;
    unsigned  BSIM4V4trnqsModGiven :1;
    unsigned  BSIM4V4acnqsModGiven :1;
    unsigned  BSIM4V4fnoiModGiven :1;
    unsigned  BSIM4V4tnoiModGiven :1;
    unsigned  BSIM4V4igcModGiven :1;
    unsigned  BSIM4V4igbModGiven :1;
    unsigned  BSIM4V4tempModGiven :1;
    unsigned  BSIM4V4typeGiven   :1;
    unsigned  BSIM4V4toxrefGiven   :1;
    unsigned  BSIM4V4toxeGiven   :1;
    unsigned  BSIM4V4toxpGiven   :1;
    unsigned  BSIM4V4toxmGiven   :1;
    unsigned  BSIM4V4dtoxGiven   :1;
    unsigned  BSIM4V4epsroxGiven   :1;
    unsigned  BSIM4V4versionGiven   :1;
    unsigned  BSIM4V4cdscGiven   :1;
    unsigned  BSIM4V4cdscbGiven   :1;
    unsigned  BSIM4V4cdscdGiven   :1;
    unsigned  BSIM4V4citGiven   :1;
    unsigned  BSIM4V4nfactorGiven   :1;
    unsigned  BSIM4V4xjGiven   :1;
    unsigned  BSIM4V4vsatGiven   :1;
    unsigned  BSIM4V4atGiven   :1;
    unsigned  BSIM4V4a0Given   :1;
    unsigned  BSIM4V4agsGiven   :1;
    unsigned  BSIM4V4a1Given   :1;
    unsigned  BSIM4V4a2Given   :1;
    unsigned  BSIM4V4ketaGiven   :1;    
    unsigned  BSIM4V4nsubGiven   :1;
    unsigned  BSIM4V4ndepGiven   :1;
    unsigned  BSIM4V4nsdGiven    :1;
    unsigned  BSIM4V4phinGiven   :1;
    unsigned  BSIM4V4ngateGiven   :1;
    unsigned  BSIM4V4gamma1Given   :1;
    unsigned  BSIM4V4gamma2Given   :1;
    unsigned  BSIM4V4vbxGiven   :1;
    unsigned  BSIM4V4vbmGiven   :1;
    unsigned  BSIM4V4xtGiven   :1;
    unsigned  BSIM4V4k1Given   :1;
    unsigned  BSIM4V4kt1Given   :1;
    unsigned  BSIM4V4kt1lGiven   :1;
    unsigned  BSIM4V4kt2Given   :1;
    unsigned  BSIM4V4k2Given   :1;
    unsigned  BSIM4V4k3Given   :1;
    unsigned  BSIM4V4k3bGiven   :1;
    unsigned  BSIM4V4w0Given   :1;
    unsigned  BSIM4V4dvtp0Given :1;
    unsigned  BSIM4V4dvtp1Given :1;
    unsigned  BSIM4V4lpe0Given   :1;
    unsigned  BSIM4V4lpebGiven   :1;
    unsigned  BSIM4V4dvt0Given   :1;   
    unsigned  BSIM4V4dvt1Given   :1;     
    unsigned  BSIM4V4dvt2Given   :1;     
    unsigned  BSIM4V4dvt0wGiven   :1;   
    unsigned  BSIM4V4dvt1wGiven   :1;     
    unsigned  BSIM4V4dvt2wGiven   :1;     
    unsigned  BSIM4V4droutGiven   :1;     
    unsigned  BSIM4V4dsubGiven   :1;     
    unsigned  BSIM4V4vth0Given   :1;
    unsigned  BSIM4V4euGiven   :1;
    unsigned  BSIM4V4uaGiven   :1;
    unsigned  BSIM4V4ua1Given   :1;
    unsigned  BSIM4V4ubGiven   :1;
    unsigned  BSIM4V4ub1Given   :1;
    unsigned  BSIM4V4ucGiven   :1;
    unsigned  BSIM4V4uc1Given   :1;
    unsigned  BSIM4V4u0Given   :1;
    unsigned  BSIM4V4uteGiven   :1;
    unsigned  BSIM4V4voffGiven   :1;
    unsigned  BSIM4V4vofflGiven  :1;
    unsigned  BSIM4V4minvGiven   :1;
    unsigned  BSIM4V4rdswGiven   :1;      
    unsigned  BSIM4V4rdswminGiven :1;
    unsigned  BSIM4V4rdwminGiven :1;
    unsigned  BSIM4V4rswminGiven :1;
    unsigned  BSIM4V4rswGiven   :1;
    unsigned  BSIM4V4rdwGiven   :1;
    unsigned  BSIM4V4prwgGiven   :1;      
    unsigned  BSIM4V4prwbGiven   :1;      
    unsigned  BSIM4V4prtGiven   :1;      
    unsigned  BSIM4V4eta0Given   :1;    
    unsigned  BSIM4V4etabGiven   :1;    
    unsigned  BSIM4V4pclmGiven   :1;   
    unsigned  BSIM4V4pdibl1Given   :1;   
    unsigned  BSIM4V4pdibl2Given   :1;  
    unsigned  BSIM4V4pdiblbGiven   :1;  
    unsigned  BSIM4V4fproutGiven   :1;
    unsigned  BSIM4V4pditsGiven    :1;
    unsigned  BSIM4V4pditsdGiven    :1;
    unsigned  BSIM4V4pditslGiven    :1;
    unsigned  BSIM4V4pscbe1Given   :1;    
    unsigned  BSIM4V4pscbe2Given   :1;    
    unsigned  BSIM4V4pvagGiven   :1;    
    unsigned  BSIM4V4deltaGiven  :1;     
    unsigned  BSIM4V4wrGiven   :1;
    unsigned  BSIM4V4dwgGiven   :1;
    unsigned  BSIM4V4dwbGiven   :1;
    unsigned  BSIM4V4b0Given   :1;
    unsigned  BSIM4V4b1Given   :1;
    unsigned  BSIM4V4alpha0Given   :1;
    unsigned  BSIM4V4alpha1Given   :1;
    unsigned  BSIM4V4beta0Given   :1;
    unsigned  BSIM4V4agidlGiven   :1;
    unsigned  BSIM4V4bgidlGiven   :1;
    unsigned  BSIM4V4cgidlGiven   :1;
    unsigned  BSIM4V4egidlGiven   :1;
    unsigned  BSIM4V4aigcGiven   :1;
    unsigned  BSIM4V4bigcGiven   :1;
    unsigned  BSIM4V4cigcGiven   :1;
    unsigned  BSIM4V4aigsdGiven   :1;
    unsigned  BSIM4V4bigsdGiven   :1;
    unsigned  BSIM4V4cigsdGiven   :1;
    unsigned  BSIM4V4aigbaccGiven   :1;
    unsigned  BSIM4V4bigbaccGiven   :1;
    unsigned  BSIM4V4cigbaccGiven   :1;
    unsigned  BSIM4V4aigbinvGiven   :1;
    unsigned  BSIM4V4bigbinvGiven   :1;
    unsigned  BSIM4V4cigbinvGiven   :1;
    unsigned  BSIM4V4nigcGiven   :1;
    unsigned  BSIM4V4nigbinvGiven   :1;
    unsigned  BSIM4V4nigbaccGiven   :1;
    unsigned  BSIM4V4ntoxGiven   :1;
    unsigned  BSIM4V4eigbinvGiven   :1;
    unsigned  BSIM4V4pigcdGiven   :1;
    unsigned  BSIM4V4poxedgeGiven   :1;
    unsigned  BSIM4V4ijthdfwdGiven  :1;
    unsigned  BSIM4V4ijthsfwdGiven  :1;
    unsigned  BSIM4V4ijthdrevGiven  :1;
    unsigned  BSIM4V4ijthsrevGiven  :1;
    unsigned  BSIM4V4xjbvdGiven   :1;
    unsigned  BSIM4V4xjbvsGiven   :1;
    unsigned  BSIM4V4bvdGiven   :1;
    unsigned  BSIM4V4bvsGiven   :1;

    unsigned  BSIM4V4jtssGiven   :1;
    unsigned  BSIM4V4jtsdGiven   :1;
    unsigned  BSIM4V4jtsswsGiven   :1;
    unsigned  BSIM4V4jtsswdGiven   :1;
    unsigned  BSIM4V4jtsswgsGiven   :1;
    unsigned  BSIM4V4jtsswgdGiven   :1;
    unsigned  BSIM4V4njtsGiven   :1;
    unsigned  BSIM4V4njtsswGiven   :1;
    unsigned  BSIM4V4njtsswgGiven   :1;
    unsigned  BSIM4V4xtssGiven   :1;
    unsigned  BSIM4V4xtsdGiven   :1;
    unsigned  BSIM4V4xtsswsGiven   :1;
    unsigned  BSIM4V4xtsswdGiven   :1;
    unsigned  BSIM4V4xtsswgsGiven   :1;
    unsigned  BSIM4V4xtsswgdGiven   :1;
    unsigned  BSIM4V4tnjtsGiven   :1;
    unsigned  BSIM4V4tnjtsswGiven   :1;
    unsigned  BSIM4V4tnjtsswgGiven   :1;
    unsigned  BSIM4V4vtssGiven   :1;
    unsigned  BSIM4V4vtsdGiven   :1;
    unsigned  BSIM4V4vtsswsGiven   :1;
    unsigned  BSIM4V4vtsswdGiven   :1;
    unsigned  BSIM4V4vtsswgsGiven   :1;
    unsigned  BSIM4V4vtsswgdGiven   :1;

    unsigned  BSIM4V4vfbGiven   :1;
    unsigned  BSIM4V4gbminGiven :1;
    unsigned  BSIM4V4rbdbGiven :1;
    unsigned  BSIM4V4rbsbGiven :1;
    unsigned  BSIM4V4rbpsGiven :1;
    unsigned  BSIM4V4rbpdGiven :1;
    unsigned  BSIM4V4rbpbGiven :1;
    unsigned  BSIM4V4xrcrg1Given   :1;
    unsigned  BSIM4V4xrcrg2Given   :1;
    unsigned  BSIM4V4tnoiaGiven    :1;
    unsigned  BSIM4V4tnoibGiven    :1;
    unsigned  BSIM4V4rnoiaGiven    :1;
    unsigned  BSIM4V4rnoibGiven    :1;
    unsigned  BSIM4V4ntnoiGiven    :1;

    unsigned  BSIM4V4lambdaGiven    :1;
    unsigned  BSIM4V4vtlGiven    :1;
    unsigned  BSIM4V4lcGiven    :1;
    unsigned  BSIM4V4xnGiven    :1;
    unsigned  BSIM4V4vfbsdoffGiven    :1;
    unsigned  BSIM4V4lintnoiGiven    :1;

    /* CV model and parasitics */
    unsigned  BSIM4V4cgslGiven   :1;
    unsigned  BSIM4V4cgdlGiven   :1;
    unsigned  BSIM4V4ckappasGiven   :1;
    unsigned  BSIM4V4ckappadGiven   :1;
    unsigned  BSIM4V4cfGiven   :1;
    unsigned  BSIM4V4vfbcvGiven   :1;
    unsigned  BSIM4V4clcGiven   :1;
    unsigned  BSIM4V4cleGiven   :1;
    unsigned  BSIM4V4dwcGiven   :1;
    unsigned  BSIM4V4dlcGiven   :1;
    unsigned  BSIM4V4xwGiven    :1;
    unsigned  BSIM4V4xlGiven    :1;
    unsigned  BSIM4V4dlcigGiven   :1;
    unsigned  BSIM4V4dwjGiven   :1;
    unsigned  BSIM4V4noffGiven  :1;
    unsigned  BSIM4V4voffcvGiven :1;
    unsigned  BSIM4V4acdeGiven  :1;
    unsigned  BSIM4V4moinGiven  :1;
    unsigned  BSIM4V4tcjGiven   :1;
    unsigned  BSIM4V4tcjswGiven :1;
    unsigned  BSIM4V4tcjswgGiven :1;
    unsigned  BSIM4V4tpbGiven    :1;
    unsigned  BSIM4V4tpbswGiven  :1;
    unsigned  BSIM4V4tpbswgGiven :1;
    unsigned  BSIM4V4dmcgGiven :1;
    unsigned  BSIM4V4dmciGiven :1;
    unsigned  BSIM4V4dmdgGiven :1;
    unsigned  BSIM4V4dmcgtGiven :1;
    unsigned  BSIM4V4xgwGiven :1;
    unsigned  BSIM4V4xglGiven :1;
    unsigned  BSIM4V4rshgGiven :1;
    unsigned  BSIM4V4ngconGiven :1;


    /* Length dependence */
    unsigned  BSIM4V4lcdscGiven   :1;
    unsigned  BSIM4V4lcdscbGiven   :1;
    unsigned  BSIM4V4lcdscdGiven   :1;
    unsigned  BSIM4V4lcitGiven   :1;
    unsigned  BSIM4V4lnfactorGiven   :1;
    unsigned  BSIM4V4lxjGiven   :1;
    unsigned  BSIM4V4lvsatGiven   :1;
    unsigned  BSIM4V4latGiven   :1;
    unsigned  BSIM4V4la0Given   :1;
    unsigned  BSIM4V4lagsGiven   :1;
    unsigned  BSIM4V4la1Given   :1;
    unsigned  BSIM4V4la2Given   :1;
    unsigned  BSIM4V4lketaGiven   :1;    
    unsigned  BSIM4V4lnsubGiven   :1;
    unsigned  BSIM4V4lndepGiven   :1;
    unsigned  BSIM4V4lnsdGiven    :1;
    unsigned  BSIM4V4lphinGiven   :1;
    unsigned  BSIM4V4lngateGiven   :1;
    unsigned  BSIM4V4lgamma1Given   :1;
    unsigned  BSIM4V4lgamma2Given   :1;
    unsigned  BSIM4V4lvbxGiven   :1;
    unsigned  BSIM4V4lvbmGiven   :1;
    unsigned  BSIM4V4lxtGiven   :1;
    unsigned  BSIM4V4lk1Given   :1;
    unsigned  BSIM4V4lkt1Given   :1;
    unsigned  BSIM4V4lkt1lGiven   :1;
    unsigned  BSIM4V4lkt2Given   :1;
    unsigned  BSIM4V4lk2Given   :1;
    unsigned  BSIM4V4lk3Given   :1;
    unsigned  BSIM4V4lk3bGiven   :1;
    unsigned  BSIM4V4lw0Given   :1;
    unsigned  BSIM4V4ldvtp0Given :1;
    unsigned  BSIM4V4ldvtp1Given :1;
    unsigned  BSIM4V4llpe0Given   :1;
    unsigned  BSIM4V4llpebGiven   :1;
    unsigned  BSIM4V4ldvt0Given   :1;   
    unsigned  BSIM4V4ldvt1Given   :1;     
    unsigned  BSIM4V4ldvt2Given   :1;     
    unsigned  BSIM4V4ldvt0wGiven   :1;   
    unsigned  BSIM4V4ldvt1wGiven   :1;     
    unsigned  BSIM4V4ldvt2wGiven   :1;     
    unsigned  BSIM4V4ldroutGiven   :1;     
    unsigned  BSIM4V4ldsubGiven   :1;     
    unsigned  BSIM4V4lvth0Given   :1;
    unsigned  BSIM4V4luaGiven   :1;
    unsigned  BSIM4V4lua1Given   :1;
    unsigned  BSIM4V4lubGiven   :1;
    unsigned  BSIM4V4lub1Given   :1;
    unsigned  BSIM4V4lucGiven   :1;
    unsigned  BSIM4V4luc1Given   :1;
    unsigned  BSIM4V4lu0Given   :1;
    unsigned  BSIM4V4leuGiven   :1;
    unsigned  BSIM4V4luteGiven   :1;
    unsigned  BSIM4V4lvoffGiven   :1;
    unsigned  BSIM4V4lminvGiven   :1;
    unsigned  BSIM4V4lrdswGiven   :1;      
    unsigned  BSIM4V4lrswGiven   :1;
    unsigned  BSIM4V4lrdwGiven   :1;
    unsigned  BSIM4V4lprwgGiven   :1;      
    unsigned  BSIM4V4lprwbGiven   :1;      
    unsigned  BSIM4V4lprtGiven   :1;      
    unsigned  BSIM4V4leta0Given   :1;    
    unsigned  BSIM4V4letabGiven   :1;    
    unsigned  BSIM4V4lpclmGiven   :1;   
    unsigned  BSIM4V4lpdibl1Given   :1;   
    unsigned  BSIM4V4lpdibl2Given   :1;  
    unsigned  BSIM4V4lpdiblbGiven   :1;  
    unsigned  BSIM4V4lfproutGiven   :1;
    unsigned  BSIM4V4lpditsGiven    :1;
    unsigned  BSIM4V4lpditsdGiven    :1;
    unsigned  BSIM4V4lpscbe1Given   :1;    
    unsigned  BSIM4V4lpscbe2Given   :1;    
    unsigned  BSIM4V4lpvagGiven   :1;    
    unsigned  BSIM4V4ldeltaGiven  :1;     
    unsigned  BSIM4V4lwrGiven   :1;
    unsigned  BSIM4V4ldwgGiven   :1;
    unsigned  BSIM4V4ldwbGiven   :1;
    unsigned  BSIM4V4lb0Given   :1;
    unsigned  BSIM4V4lb1Given   :1;
    unsigned  BSIM4V4lalpha0Given   :1;
    unsigned  BSIM4V4lalpha1Given   :1;
    unsigned  BSIM4V4lbeta0Given   :1;
    unsigned  BSIM4V4lvfbGiven   :1;
    unsigned  BSIM4V4lagidlGiven   :1;
    unsigned  BSIM4V4lbgidlGiven   :1;
    unsigned  BSIM4V4lcgidlGiven   :1;
    unsigned  BSIM4V4legidlGiven   :1;
    unsigned  BSIM4V4laigcGiven   :1;
    unsigned  BSIM4V4lbigcGiven   :1;
    unsigned  BSIM4V4lcigcGiven   :1;
    unsigned  BSIM4V4laigsdGiven   :1;
    unsigned  BSIM4V4lbigsdGiven   :1;
    unsigned  BSIM4V4lcigsdGiven   :1;
    unsigned  BSIM4V4laigbaccGiven   :1;
    unsigned  BSIM4V4lbigbaccGiven   :1;
    unsigned  BSIM4V4lcigbaccGiven   :1;
    unsigned  BSIM4V4laigbinvGiven   :1;
    unsigned  BSIM4V4lbigbinvGiven   :1;
    unsigned  BSIM4V4lcigbinvGiven   :1;
    unsigned  BSIM4V4lnigcGiven   :1;
    unsigned  BSIM4V4lnigbinvGiven   :1;
    unsigned  BSIM4V4lnigbaccGiven   :1;
    unsigned  BSIM4V4lntoxGiven   :1;
    unsigned  BSIM4V4leigbinvGiven   :1;
    unsigned  BSIM4V4lpigcdGiven   :1;
    unsigned  BSIM4V4lpoxedgeGiven   :1;
    unsigned  BSIM4V4lxrcrg1Given   :1;
    unsigned  BSIM4V4lxrcrg2Given   :1;
    unsigned  BSIM4V4llambdaGiven    :1;
    unsigned  BSIM4V4lvtlGiven    :1;
    unsigned  BSIM4V4lxnGiven    :1;
    unsigned  BSIM4V4lvfbsdoffGiven    :1;

    /* CV model */
    unsigned  BSIM4V4lcgslGiven   :1;
    unsigned  BSIM4V4lcgdlGiven   :1;
    unsigned  BSIM4V4lckappasGiven   :1;
    unsigned  BSIM4V4lckappadGiven   :1;
    unsigned  BSIM4V4lcfGiven   :1;
    unsigned  BSIM4V4lclcGiven   :1;
    unsigned  BSIM4V4lcleGiven   :1;
    unsigned  BSIM4V4lvfbcvGiven   :1;
    unsigned  BSIM4V4lnoffGiven   :1;
    unsigned  BSIM4V4lvoffcvGiven :1;
    unsigned  BSIM4V4lacdeGiven   :1;
    unsigned  BSIM4V4lmoinGiven   :1;

    /* Width dependence */
    unsigned  BSIM4V4wcdscGiven   :1;
    unsigned  BSIM4V4wcdscbGiven   :1;
    unsigned  BSIM4V4wcdscdGiven   :1;
    unsigned  BSIM4V4wcitGiven   :1;
    unsigned  BSIM4V4wnfactorGiven   :1;
    unsigned  BSIM4V4wxjGiven   :1;
    unsigned  BSIM4V4wvsatGiven   :1;
    unsigned  BSIM4V4watGiven   :1;
    unsigned  BSIM4V4wa0Given   :1;
    unsigned  BSIM4V4wagsGiven   :1;
    unsigned  BSIM4V4wa1Given   :1;
    unsigned  BSIM4V4wa2Given   :1;
    unsigned  BSIM4V4wketaGiven   :1;    
    unsigned  BSIM4V4wnsubGiven   :1;
    unsigned  BSIM4V4wndepGiven   :1;
    unsigned  BSIM4V4wnsdGiven    :1;
    unsigned  BSIM4V4wphinGiven   :1;
    unsigned  BSIM4V4wngateGiven   :1;
    unsigned  BSIM4V4wgamma1Given   :1;
    unsigned  BSIM4V4wgamma2Given   :1;
    unsigned  BSIM4V4wvbxGiven   :1;
    unsigned  BSIM4V4wvbmGiven   :1;
    unsigned  BSIM4V4wxtGiven   :1;
    unsigned  BSIM4V4wk1Given   :1;
    unsigned  BSIM4V4wkt1Given   :1;
    unsigned  BSIM4V4wkt1lGiven   :1;
    unsigned  BSIM4V4wkt2Given   :1;
    unsigned  BSIM4V4wk2Given   :1;
    unsigned  BSIM4V4wk3Given   :1;
    unsigned  BSIM4V4wk3bGiven   :1;
    unsigned  BSIM4V4ww0Given   :1;
    unsigned  BSIM4V4wdvtp0Given :1;
    unsigned  BSIM4V4wdvtp1Given :1;
    unsigned  BSIM4V4wlpe0Given   :1;
    unsigned  BSIM4V4wlpebGiven   :1;
    unsigned  BSIM4V4wdvt0Given   :1;   
    unsigned  BSIM4V4wdvt1Given   :1;     
    unsigned  BSIM4V4wdvt2Given   :1;     
    unsigned  BSIM4V4wdvt0wGiven   :1;   
    unsigned  BSIM4V4wdvt1wGiven   :1;     
    unsigned  BSIM4V4wdvt2wGiven   :1;     
    unsigned  BSIM4V4wdroutGiven   :1;     
    unsigned  BSIM4V4wdsubGiven   :1;     
    unsigned  BSIM4V4wvth0Given   :1;
    unsigned  BSIM4V4wuaGiven   :1;
    unsigned  BSIM4V4wua1Given   :1;
    unsigned  BSIM4V4wubGiven   :1;
    unsigned  BSIM4V4wub1Given   :1;
    unsigned  BSIM4V4wucGiven   :1;
    unsigned  BSIM4V4wuc1Given   :1;
    unsigned  BSIM4V4wu0Given   :1;
    unsigned  BSIM4V4weuGiven   :1;
    unsigned  BSIM4V4wuteGiven   :1;
    unsigned  BSIM4V4wvoffGiven   :1;
    unsigned  BSIM4V4wminvGiven   :1;
    unsigned  BSIM4V4wrdswGiven   :1;      
    unsigned  BSIM4V4wrswGiven   :1;
    unsigned  BSIM4V4wrdwGiven   :1;
    unsigned  BSIM4V4wprwgGiven   :1;      
    unsigned  BSIM4V4wprwbGiven   :1;      
    unsigned  BSIM4V4wprtGiven   :1;      
    unsigned  BSIM4V4weta0Given   :1;    
    unsigned  BSIM4V4wetabGiven   :1;    
    unsigned  BSIM4V4wpclmGiven   :1;   
    unsigned  BSIM4V4wpdibl1Given   :1;   
    unsigned  BSIM4V4wpdibl2Given   :1;  
    unsigned  BSIM4V4wpdiblbGiven   :1;  
    unsigned  BSIM4V4wfproutGiven   :1;
    unsigned  BSIM4V4wpditsGiven    :1;
    unsigned  BSIM4V4wpditsdGiven    :1;
    unsigned  BSIM4V4wpscbe1Given   :1;    
    unsigned  BSIM4V4wpscbe2Given   :1;    
    unsigned  BSIM4V4wpvagGiven   :1;    
    unsigned  BSIM4V4wdeltaGiven  :1;     
    unsigned  BSIM4V4wwrGiven   :1;
    unsigned  BSIM4V4wdwgGiven   :1;
    unsigned  BSIM4V4wdwbGiven   :1;
    unsigned  BSIM4V4wb0Given   :1;
    unsigned  BSIM4V4wb1Given   :1;
    unsigned  BSIM4V4walpha0Given   :1;
    unsigned  BSIM4V4walpha1Given   :1;
    unsigned  BSIM4V4wbeta0Given   :1;
    unsigned  BSIM4V4wvfbGiven   :1;
    unsigned  BSIM4V4wagidlGiven   :1;
    unsigned  BSIM4V4wbgidlGiven   :1;
    unsigned  BSIM4V4wcgidlGiven   :1;
    unsigned  BSIM4V4wegidlGiven   :1;
    unsigned  BSIM4V4waigcGiven   :1;
    unsigned  BSIM4V4wbigcGiven   :1;
    unsigned  BSIM4V4wcigcGiven   :1;
    unsigned  BSIM4V4waigsdGiven   :1;
    unsigned  BSIM4V4wbigsdGiven   :1;
    unsigned  BSIM4V4wcigsdGiven   :1;
    unsigned  BSIM4V4waigbaccGiven   :1;
    unsigned  BSIM4V4wbigbaccGiven   :1;
    unsigned  BSIM4V4wcigbaccGiven   :1;
    unsigned  BSIM4V4waigbinvGiven   :1;
    unsigned  BSIM4V4wbigbinvGiven   :1;
    unsigned  BSIM4V4wcigbinvGiven   :1;
    unsigned  BSIM4V4wnigcGiven   :1;
    unsigned  BSIM4V4wnigbinvGiven   :1;
    unsigned  BSIM4V4wnigbaccGiven   :1;
    unsigned  BSIM4V4wntoxGiven   :1;
    unsigned  BSIM4V4weigbinvGiven   :1;
    unsigned  BSIM4V4wpigcdGiven   :1;
    unsigned  BSIM4V4wpoxedgeGiven   :1;
    unsigned  BSIM4V4wxrcrg1Given   :1;
    unsigned  BSIM4V4wxrcrg2Given   :1;
    unsigned  BSIM4V4wlambdaGiven    :1;
    unsigned  BSIM4V4wvtlGiven    :1;
    unsigned  BSIM4V4wxnGiven    :1;
    unsigned  BSIM4V4wvfbsdoffGiven    :1;

    /* CV model */
    unsigned  BSIM4V4wcgslGiven   :1;
    unsigned  BSIM4V4wcgdlGiven   :1;
    unsigned  BSIM4V4wckappasGiven   :1;
    unsigned  BSIM4V4wckappadGiven   :1;
    unsigned  BSIM4V4wcfGiven   :1;
    unsigned  BSIM4V4wclcGiven   :1;
    unsigned  BSIM4V4wcleGiven   :1;
    unsigned  BSIM4V4wvfbcvGiven   :1;
    unsigned  BSIM4V4wnoffGiven   :1;
    unsigned  BSIM4V4wvoffcvGiven :1;
    unsigned  BSIM4V4wacdeGiven   :1;
    unsigned  BSIM4V4wmoinGiven   :1;

    /* Cross-term dependence */
    unsigned  BSIM4V4pcdscGiven   :1;
    unsigned  BSIM4V4pcdscbGiven   :1;
    unsigned  BSIM4V4pcdscdGiven   :1;
    unsigned  BSIM4V4pcitGiven   :1;
    unsigned  BSIM4V4pnfactorGiven   :1;
    unsigned  BSIM4V4pxjGiven   :1;
    unsigned  BSIM4V4pvsatGiven   :1;
    unsigned  BSIM4V4patGiven   :1;
    unsigned  BSIM4V4pa0Given   :1;
    unsigned  BSIM4V4pagsGiven   :1;
    unsigned  BSIM4V4pa1Given   :1;
    unsigned  BSIM4V4pa2Given   :1;
    unsigned  BSIM4V4pketaGiven   :1;    
    unsigned  BSIM4V4pnsubGiven   :1;
    unsigned  BSIM4V4pndepGiven   :1;
    unsigned  BSIM4V4pnsdGiven    :1;
    unsigned  BSIM4V4pphinGiven   :1;
    unsigned  BSIM4V4pngateGiven   :1;
    unsigned  BSIM4V4pgamma1Given   :1;
    unsigned  BSIM4V4pgamma2Given   :1;
    unsigned  BSIM4V4pvbxGiven   :1;
    unsigned  BSIM4V4pvbmGiven   :1;
    unsigned  BSIM4V4pxtGiven   :1;
    unsigned  BSIM4V4pk1Given   :1;
    unsigned  BSIM4V4pkt1Given   :1;
    unsigned  BSIM4V4pkt1lGiven   :1;
    unsigned  BSIM4V4pkt2Given   :1;
    unsigned  BSIM4V4pk2Given   :1;
    unsigned  BSIM4V4pk3Given   :1;
    unsigned  BSIM4V4pk3bGiven   :1;
    unsigned  BSIM4V4pw0Given   :1;
    unsigned  BSIM4V4pdvtp0Given :1;
    unsigned  BSIM4V4pdvtp1Given :1;
    unsigned  BSIM4V4plpe0Given   :1;
    unsigned  BSIM4V4plpebGiven   :1;
    unsigned  BSIM4V4pdvt0Given   :1;   
    unsigned  BSIM4V4pdvt1Given   :1;     
    unsigned  BSIM4V4pdvt2Given   :1;     
    unsigned  BSIM4V4pdvt0wGiven   :1;   
    unsigned  BSIM4V4pdvt1wGiven   :1;     
    unsigned  BSIM4V4pdvt2wGiven   :1;     
    unsigned  BSIM4V4pdroutGiven   :1;     
    unsigned  BSIM4V4pdsubGiven   :1;     
    unsigned  BSIM4V4pvth0Given   :1;
    unsigned  BSIM4V4puaGiven   :1;
    unsigned  BSIM4V4pua1Given   :1;
    unsigned  BSIM4V4pubGiven   :1;
    unsigned  BSIM4V4pub1Given   :1;
    unsigned  BSIM4V4pucGiven   :1;
    unsigned  BSIM4V4puc1Given   :1;
    unsigned  BSIM4V4pu0Given   :1;
    unsigned  BSIM4V4peuGiven   :1;
    unsigned  BSIM4V4puteGiven   :1;
    unsigned  BSIM4V4pvoffGiven   :1;
    unsigned  BSIM4V4pminvGiven   :1;
    unsigned  BSIM4V4prdswGiven   :1;      
    unsigned  BSIM4V4prswGiven   :1;
    unsigned  BSIM4V4prdwGiven   :1;
    unsigned  BSIM4V4pprwgGiven   :1;      
    unsigned  BSIM4V4pprwbGiven   :1;      
    unsigned  BSIM4V4pprtGiven   :1;      
    unsigned  BSIM4V4peta0Given   :1;    
    unsigned  BSIM4V4petabGiven   :1;    
    unsigned  BSIM4V4ppclmGiven   :1;   
    unsigned  BSIM4V4ppdibl1Given   :1;   
    unsigned  BSIM4V4ppdibl2Given   :1;  
    unsigned  BSIM4V4ppdiblbGiven   :1;  
    unsigned  BSIM4V4pfproutGiven   :1;
    unsigned  BSIM4V4ppditsGiven    :1;
    unsigned  BSIM4V4ppditsdGiven    :1;
    unsigned  BSIM4V4ppscbe1Given   :1;    
    unsigned  BSIM4V4ppscbe2Given   :1;    
    unsigned  BSIM4V4ppvagGiven   :1;    
    unsigned  BSIM4V4pdeltaGiven  :1;     
    unsigned  BSIM4V4pwrGiven   :1;
    unsigned  BSIM4V4pdwgGiven   :1;
    unsigned  BSIM4V4pdwbGiven   :1;
    unsigned  BSIM4V4pb0Given   :1;
    unsigned  BSIM4V4pb1Given   :1;
    unsigned  BSIM4V4palpha0Given   :1;
    unsigned  BSIM4V4palpha1Given   :1;
    unsigned  BSIM4V4pbeta0Given   :1;
    unsigned  BSIM4V4pvfbGiven   :1;
    unsigned  BSIM4V4pagidlGiven   :1;
    unsigned  BSIM4V4pbgidlGiven   :1;
    unsigned  BSIM4V4pcgidlGiven   :1;
    unsigned  BSIM4V4pegidlGiven   :1;
    unsigned  BSIM4V4paigcGiven   :1;
    unsigned  BSIM4V4pbigcGiven   :1;
    unsigned  BSIM4V4pcigcGiven   :1;
    unsigned  BSIM4V4paigsdGiven   :1;
    unsigned  BSIM4V4pbigsdGiven   :1;
    unsigned  BSIM4V4pcigsdGiven   :1;
    unsigned  BSIM4V4paigbaccGiven   :1;
    unsigned  BSIM4V4pbigbaccGiven   :1;
    unsigned  BSIM4V4pcigbaccGiven   :1;
    unsigned  BSIM4V4paigbinvGiven   :1;
    unsigned  BSIM4V4pbigbinvGiven   :1;
    unsigned  BSIM4V4pcigbinvGiven   :1;
    unsigned  BSIM4V4pnigcGiven   :1;
    unsigned  BSIM4V4pnigbinvGiven   :1;
    unsigned  BSIM4V4pnigbaccGiven   :1;
    unsigned  BSIM4V4pntoxGiven   :1;
    unsigned  BSIM4V4peigbinvGiven   :1;
    unsigned  BSIM4V4ppigcdGiven   :1;
    unsigned  BSIM4V4ppoxedgeGiven   :1;
    unsigned  BSIM4V4pxrcrg1Given   :1;
    unsigned  BSIM4V4pxrcrg2Given   :1;
    unsigned  BSIM4V4plambdaGiven    :1;
    unsigned  BSIM4V4pvtlGiven    :1;
    unsigned  BSIM4V4pxnGiven    :1;
    unsigned  BSIM4V4pvfbsdoffGiven    :1;

    /* CV model */
    unsigned  BSIM4V4pcgslGiven   :1;
    unsigned  BSIM4V4pcgdlGiven   :1;
    unsigned  BSIM4V4pckappasGiven   :1;
    unsigned  BSIM4V4pckappadGiven   :1;
    unsigned  BSIM4V4pcfGiven   :1;
    unsigned  BSIM4V4pclcGiven   :1;
    unsigned  BSIM4V4pcleGiven   :1;
    unsigned  BSIM4V4pvfbcvGiven   :1;
    unsigned  BSIM4V4pnoffGiven   :1;
    unsigned  BSIM4V4pvoffcvGiven :1;
    unsigned  BSIM4V4pacdeGiven   :1;
    unsigned  BSIM4V4pmoinGiven   :1;

    unsigned  BSIM4V4useFringeGiven   :1;

    unsigned  BSIM4V4tnomGiven   :1;
    unsigned  BSIM4V4cgsoGiven   :1;
    unsigned  BSIM4V4cgdoGiven   :1;
    unsigned  BSIM4V4cgboGiven   :1;
    unsigned  BSIM4V4xpartGiven   :1;
    unsigned  BSIM4V4sheetResistanceGiven   :1;

    unsigned  BSIM4V4SjctSatCurDensityGiven   :1;
    unsigned  BSIM4V4SjctSidewallSatCurDensityGiven   :1;
    unsigned  BSIM4V4SjctGateSidewallSatCurDensityGiven   :1;
    unsigned  BSIM4V4SbulkJctPotentialGiven   :1;
    unsigned  BSIM4V4SbulkJctBotGradingCoeffGiven   :1;
    unsigned  BSIM4V4SsidewallJctPotentialGiven   :1;
    unsigned  BSIM4V4SGatesidewallJctPotentialGiven   :1;
    unsigned  BSIM4V4SbulkJctSideGradingCoeffGiven   :1;
    unsigned  BSIM4V4SunitAreaJctCapGiven   :1;
    unsigned  BSIM4V4SunitLengthSidewallJctCapGiven   :1;
    unsigned  BSIM4V4SbulkJctGateSideGradingCoeffGiven   :1;
    unsigned  BSIM4V4SunitLengthGateSidewallJctCapGiven   :1;
    unsigned  BSIM4V4SjctEmissionCoeffGiven :1;
    unsigned  BSIM4V4SjctTempExponentGiven	:1;

    unsigned  BSIM4V4DjctSatCurDensityGiven   :1;
    unsigned  BSIM4V4DjctSidewallSatCurDensityGiven   :1;
    unsigned  BSIM4V4DjctGateSidewallSatCurDensityGiven   :1;
    unsigned  BSIM4V4DbulkJctPotentialGiven   :1;
    unsigned  BSIM4V4DbulkJctBotGradingCoeffGiven   :1;
    unsigned  BSIM4V4DsidewallJctPotentialGiven   :1;
    unsigned  BSIM4V4DGatesidewallJctPotentialGiven   :1;
    unsigned  BSIM4V4DbulkJctSideGradingCoeffGiven   :1;
    unsigned  BSIM4V4DunitAreaJctCapGiven   :1;
    unsigned  BSIM4V4DunitLengthSidewallJctCapGiven   :1;
    unsigned  BSIM4V4DbulkJctGateSideGradingCoeffGiven   :1;
    unsigned  BSIM4V4DunitLengthGateSidewallJctCapGiven   :1;
    unsigned  BSIM4V4DjctEmissionCoeffGiven :1;
    unsigned  BSIM4V4DjctTempExponentGiven :1;

    unsigned  BSIM4V4oxideTrapDensityAGiven  :1;         
    unsigned  BSIM4V4oxideTrapDensityBGiven  :1;        
    unsigned  BSIM4V4oxideTrapDensityCGiven  :1;     
    unsigned  BSIM4V4emGiven  :1;     
    unsigned  BSIM4V4efGiven  :1;     
    unsigned  BSIM4V4afGiven  :1;     
    unsigned  BSIM4V4kfGiven  :1;     

    unsigned  BSIM4V4LintGiven   :1;
    unsigned  BSIM4V4LlGiven   :1;
    unsigned  BSIM4V4LlcGiven   :1;
    unsigned  BSIM4V4LlnGiven   :1;
    unsigned  BSIM4V4LwGiven   :1;
    unsigned  BSIM4V4LwcGiven   :1;
    unsigned  BSIM4V4LwnGiven   :1;
    unsigned  BSIM4V4LwlGiven   :1;
    unsigned  BSIM4V4LwlcGiven   :1;
    unsigned  BSIM4V4LminGiven   :1;
    unsigned  BSIM4V4LmaxGiven   :1;

    unsigned  BSIM4V4WintGiven   :1;
    unsigned  BSIM4V4WlGiven   :1;
    unsigned  BSIM4V4WlcGiven   :1;
    unsigned  BSIM4V4WlnGiven   :1;
    unsigned  BSIM4V4WwGiven   :1;
    unsigned  BSIM4V4WwcGiven   :1;
    unsigned  BSIM4V4WwnGiven   :1;
    unsigned  BSIM4V4WwlGiven   :1;
    unsigned  BSIM4V4WwlcGiven   :1;
    unsigned  BSIM4V4WminGiven   :1;
    unsigned  BSIM4V4WmaxGiven   :1;

    /* added for stress effect */
    unsigned  BSIM4V4sarefGiven   :1;
    unsigned  BSIM4V4sbrefGiven   :1;
    unsigned  BSIM4V4wlodGiven  :1;
    unsigned  BSIM4V4ku0Given   :1;
    unsigned  BSIM4V4kvsatGiven  :1;
    unsigned  BSIM4V4kvth0Given  :1;
    unsigned  BSIM4V4tku0Given   :1;
    unsigned  BSIM4V4llodku0Given   :1;
    unsigned  BSIM4V4wlodku0Given   :1;
    unsigned  BSIM4V4llodvthGiven   :1;
    unsigned  BSIM4V4wlodvthGiven   :1;
    unsigned  BSIM4V4lku0Given   :1;
    unsigned  BSIM4V4wku0Given   :1;
    unsigned  BSIM4V4pku0Given   :1;
    unsigned  BSIM4V4lkvth0Given   :1;
    unsigned  BSIM4V4wkvth0Given   :1;
    unsigned  BSIM4V4pkvth0Given   :1;
    unsigned  BSIM4V4stk2Given   :1;
    unsigned  BSIM4V4lodk2Given  :1;
    unsigned  BSIM4V4steta0Given :1;
    unsigned  BSIM4V4lodeta0Given :1;


} BSIM4V4model;


#ifndef NMOS
#define NMOS 1
#define PMOS -1
#endif /*NMOS*/


/* Instance parameters */
#define BSIM4V4_W                   1
#define BSIM4V4_L                   2
#define BSIM4V4_AS                  3
#define BSIM4V4_AD                  4
#define BSIM4V4_PS                  5
#define BSIM4V4_PD                  6
#define BSIM4V4_NRS                 7
#define BSIM4V4_NRD                 8
#define BSIM4V4_OFF                 9
#define BSIM4V4_IC                  10
#define BSIM4V4_IC_VDS              11
#define BSIM4V4_IC_VGS              12
#define BSIM4V4_IC_VBS              13
#define BSIM4V4_TRNQSMOD            14
#define BSIM4V4_RBODYMOD            15
#define BSIM4V4_RGATEMOD            16
#define BSIM4V4_GEOMOD              17
#define BSIM4V4_RGEOMOD             18
#define BSIM4V4_NF                  19
#define BSIM4V4_MIN                 20 
#define BSIM4V4_ACNQSMOD            22
#define BSIM4V4_RBDB                23
#define BSIM4V4_RBSB                24
#define BSIM4V4_RBPB                25
#define BSIM4V4_RBPS                26
#define BSIM4V4_RBPD                27
#define BSIM4V4_SA                  28
#define BSIM4V4_SB                  29
#define BSIM4V4_SD                  30
#define BSIM4V4_M                   31

/* Global parameters */
#define BSIM4V4_MOD_TEMPMOD         89
#define BSIM4V4_MOD_IGCMOD          90
#define BSIM4V4_MOD_IGBMOD          91
#define BSIM4V4_MOD_ACNQSMOD        92
#define BSIM4V4_MOD_FNOIMOD         93
#define BSIM4V4_MOD_RDSMOD          94
#define BSIM4V4_MOD_DIOMOD          96
#define BSIM4V4_MOD_PERMOD          97
#define BSIM4V4_MOD_GEOMOD          98
#define BSIM4V4_MOD_RGATEMOD        99
#define BSIM4V4_MOD_RBODYMOD        100
#define BSIM4V4_MOD_CAPMOD          101
#define BSIM4V4_MOD_TRNQSMOD        102
#define BSIM4V4_MOD_MOBMOD          103    
#define BSIM4V4_MOD_TNOIMOD         104    
#define BSIM4V4_MOD_TOXE            105
#define BSIM4V4_MOD_CDSC            106
#define BSIM4V4_MOD_CDSCB           107
#define BSIM4V4_MOD_CIT             108
#define BSIM4V4_MOD_NFACTOR         109
#define BSIM4V4_MOD_XJ              110
#define BSIM4V4_MOD_VSAT            111
#define BSIM4V4_MOD_AT              112
#define BSIM4V4_MOD_A0              113
#define BSIM4V4_MOD_A1              114
#define BSIM4V4_MOD_A2              115
#define BSIM4V4_MOD_KETA            116   
#define BSIM4V4_MOD_NSUB            117
#define BSIM4V4_MOD_NDEP            118
#define BSIM4V4_MOD_NGATE           120
#define BSIM4V4_MOD_GAMMA1          121
#define BSIM4V4_MOD_GAMMA2          122
#define BSIM4V4_MOD_VBX             123
#define BSIM4V4_MOD_BINUNIT         124    
#define BSIM4V4_MOD_VBM             125
#define BSIM4V4_MOD_XT              126
#define BSIM4V4_MOD_K1              129
#define BSIM4V4_MOD_KT1             130
#define BSIM4V4_MOD_KT1L            131
#define BSIM4V4_MOD_K2              132
#define BSIM4V4_MOD_KT2             133
#define BSIM4V4_MOD_K3              134
#define BSIM4V4_MOD_K3B             135
#define BSIM4V4_MOD_W0              136
#define BSIM4V4_MOD_LPE0            137
#define BSIM4V4_MOD_DVT0            138
#define BSIM4V4_MOD_DVT1            139
#define BSIM4V4_MOD_DVT2            140
#define BSIM4V4_MOD_DVT0W           141
#define BSIM4V4_MOD_DVT1W           142
#define BSIM4V4_MOD_DVT2W           143
#define BSIM4V4_MOD_DROUT           144
#define BSIM4V4_MOD_DSUB            145
#define BSIM4V4_MOD_VTH0            146
#define BSIM4V4_MOD_UA              147
#define BSIM4V4_MOD_UA1             148
#define BSIM4V4_MOD_UB              149
#define BSIM4V4_MOD_UB1             150
#define BSIM4V4_MOD_UC              151
#define BSIM4V4_MOD_UC1             152
#define BSIM4V4_MOD_U0              153
#define BSIM4V4_MOD_UTE             154
#define BSIM4V4_MOD_VOFF            155
#define BSIM4V4_MOD_DELTA           156
#define BSIM4V4_MOD_RDSW            157
#define BSIM4V4_MOD_PRT             158
#define BSIM4V4_MOD_LDD             159
#define BSIM4V4_MOD_ETA             160
#define BSIM4V4_MOD_ETA0            161
#define BSIM4V4_MOD_ETAB            162
#define BSIM4V4_MOD_PCLM            163
#define BSIM4V4_MOD_PDIBL1          164
#define BSIM4V4_MOD_PDIBL2          165
#define BSIM4V4_MOD_PSCBE1          166
#define BSIM4V4_MOD_PSCBE2          167
#define BSIM4V4_MOD_PVAG            168
#define BSIM4V4_MOD_WR              169
#define BSIM4V4_MOD_DWG             170
#define BSIM4V4_MOD_DWB             171
#define BSIM4V4_MOD_B0              172
#define BSIM4V4_MOD_B1              173
#define BSIM4V4_MOD_ALPHA0          174
#define BSIM4V4_MOD_BETA0           175
#define BSIM4V4_MOD_PDIBLB          178
#define BSIM4V4_MOD_PRWG            179
#define BSIM4V4_MOD_PRWB            180
#define BSIM4V4_MOD_CDSCD           181
#define BSIM4V4_MOD_AGS             182
#define BSIM4V4_MOD_FRINGE          184
#define BSIM4V4_MOD_CGSL            186
#define BSIM4V4_MOD_CGDL            187
#define BSIM4V4_MOD_CKAPPAS         188
#define BSIM4V4_MOD_CF              189
#define BSIM4V4_MOD_CLC             190
#define BSIM4V4_MOD_CLE             191
#define BSIM4V4_MOD_PARAMCHK        192
#define BSIM4V4_MOD_VERSION         193
#define BSIM4V4_MOD_VFBCV           194
#define BSIM4V4_MOD_ACDE            195
#define BSIM4V4_MOD_MOIN            196
#define BSIM4V4_MOD_NOFF            197
#define BSIM4V4_MOD_IJTHDFWD        198
#define BSIM4V4_MOD_ALPHA1          199
#define BSIM4V4_MOD_VFB             200
#define BSIM4V4_MOD_TOXM            201
#define BSIM4V4_MOD_TCJ             202
#define BSIM4V4_MOD_TCJSW           203
#define BSIM4V4_MOD_TCJSWG          204
#define BSIM4V4_MOD_TPB             205
#define BSIM4V4_MOD_TPBSW           206
#define BSIM4V4_MOD_TPBSWG          207
#define BSIM4V4_MOD_VOFFCV          208
#define BSIM4V4_MOD_GBMIN           209
#define BSIM4V4_MOD_RBDB            210
#define BSIM4V4_MOD_RBSB            211
#define BSIM4V4_MOD_RBPB            212
#define BSIM4V4_MOD_RBPS            213
#define BSIM4V4_MOD_RBPD            214
#define BSIM4V4_MOD_DMCG            215
#define BSIM4V4_MOD_DMCI            216
#define BSIM4V4_MOD_DMDG            217
#define BSIM4V4_MOD_XGW             218
#define BSIM4V4_MOD_XGL             219
#define BSIM4V4_MOD_RSHG            220
#define BSIM4V4_MOD_NGCON           221
#define BSIM4V4_MOD_AGIDL           222
#define BSIM4V4_MOD_BGIDL           223
#define BSIM4V4_MOD_EGIDL           224
#define BSIM4V4_MOD_IJTHSFWD        225
#define BSIM4V4_MOD_XJBVD           226
#define BSIM4V4_MOD_XJBVS           227
#define BSIM4V4_MOD_BVD             228
#define BSIM4V4_MOD_BVS             229
#define BSIM4V4_MOD_TOXP            230
#define BSIM4V4_MOD_DTOX            231
#define BSIM4V4_MOD_XRCRG1          232
#define BSIM4V4_MOD_XRCRG2          233
#define BSIM4V4_MOD_EU              234
#define BSIM4V4_MOD_IJTHSREV        235
#define BSIM4V4_MOD_IJTHDREV        236
#define BSIM4V4_MOD_MINV            237
#define BSIM4V4_MOD_VOFFL           238
#define BSIM4V4_MOD_PDITS           239
#define BSIM4V4_MOD_PDITSD          240
#define BSIM4V4_MOD_PDITSL          241
#define BSIM4V4_MOD_TNOIA           242
#define BSIM4V4_MOD_TNOIB           243
#define BSIM4V4_MOD_NTNOI           244
#define BSIM4V4_MOD_FPROUT          245
#define BSIM4V4_MOD_LPEB            246
#define BSIM4V4_MOD_DVTP0           247
#define BSIM4V4_MOD_DVTP1           248
#define BSIM4V4_MOD_CGIDL           249
#define BSIM4V4_MOD_PHIN            250
#define BSIM4V4_MOD_RDSWMIN         251
#define BSIM4V4_MOD_RSW             252
#define BSIM4V4_MOD_RDW             253
#define BSIM4V4_MOD_RDWMIN          254
#define BSIM4V4_MOD_RSWMIN          255
#define BSIM4V4_MOD_NSD             256
#define BSIM4V4_MOD_CKAPPAD         257
#define BSIM4V4_MOD_DMCGT           258
#define BSIM4V4_MOD_AIGC            259
#define BSIM4V4_MOD_BIGC            260
#define BSIM4V4_MOD_CIGC            261
#define BSIM4V4_MOD_AIGBACC         262
#define BSIM4V4_MOD_BIGBACC         263
#define BSIM4V4_MOD_CIGBACC         264            
#define BSIM4V4_MOD_AIGBINV         265
#define BSIM4V4_MOD_BIGBINV         266
#define BSIM4V4_MOD_CIGBINV         267
#define BSIM4V4_MOD_NIGC            268
#define BSIM4V4_MOD_NIGBACC         269
#define BSIM4V4_MOD_NIGBINV         270
#define BSIM4V4_MOD_NTOX            271
#define BSIM4V4_MOD_TOXREF          272
#define BSIM4V4_MOD_EIGBINV         273
#define BSIM4V4_MOD_PIGCD           274
#define BSIM4V4_MOD_POXEDGE         275
#define BSIM4V4_MOD_EPSROX          276
#define BSIM4V4_MOD_AIGSD           277
#define BSIM4V4_MOD_BIGSD           278
#define BSIM4V4_MOD_CIGSD           279
#define BSIM4V4_MOD_JSWGS           280
#define BSIM4V4_MOD_JSWGD           281
#define BSIM4V4_MOD_LAMBDA          282
#define BSIM4V4_MOD_VTL             283
#define BSIM4V4_MOD_LC              284
#define BSIM4V4_MOD_XN              285
#define BSIM4V4_MOD_RNOIA           286
#define BSIM4V4_MOD_RNOIB           287
#define BSIM4V4_MOD_VFBSDOFF        288
#define BSIM4V4_MOD_LINTNOI         289

/* Length dependence */
#define BSIM4V4_MOD_LCDSC            301
#define BSIM4V4_MOD_LCDSCB           302
#define BSIM4V4_MOD_LCIT             303
#define BSIM4V4_MOD_LNFACTOR         304
#define BSIM4V4_MOD_LXJ              305
#define BSIM4V4_MOD_LVSAT            306
#define BSIM4V4_MOD_LAT              307
#define BSIM4V4_MOD_LA0              308
#define BSIM4V4_MOD_LA1              309
#define BSIM4V4_MOD_LA2              310
#define BSIM4V4_MOD_LKETA            311
#define BSIM4V4_MOD_LNSUB            312
#define BSIM4V4_MOD_LNDEP            313
#define BSIM4V4_MOD_LNGATE           315
#define BSIM4V4_MOD_LGAMMA1          316
#define BSIM4V4_MOD_LGAMMA2          317
#define BSIM4V4_MOD_LVBX             318
#define BSIM4V4_MOD_LVBM             320
#define BSIM4V4_MOD_LXT              322
#define BSIM4V4_MOD_LK1              325
#define BSIM4V4_MOD_LKT1             326
#define BSIM4V4_MOD_LKT1L            327
#define BSIM4V4_MOD_LK2              328
#define BSIM4V4_MOD_LKT2             329
#define BSIM4V4_MOD_LK3              330
#define BSIM4V4_MOD_LK3B             331
#define BSIM4V4_MOD_LW0              332
#define BSIM4V4_MOD_LLPE0            333
#define BSIM4V4_MOD_LDVT0            334
#define BSIM4V4_MOD_LDVT1            335
#define BSIM4V4_MOD_LDVT2            336
#define BSIM4V4_MOD_LDVT0W           337
#define BSIM4V4_MOD_LDVT1W           338
#define BSIM4V4_MOD_LDVT2W           339
#define BSIM4V4_MOD_LDROUT           340
#define BSIM4V4_MOD_LDSUB            341
#define BSIM4V4_MOD_LVTH0            342
#define BSIM4V4_MOD_LUA              343
#define BSIM4V4_MOD_LUA1             344
#define BSIM4V4_MOD_LUB              345
#define BSIM4V4_MOD_LUB1             346
#define BSIM4V4_MOD_LUC              347
#define BSIM4V4_MOD_LUC1             348
#define BSIM4V4_MOD_LU0              349
#define BSIM4V4_MOD_LUTE             350
#define BSIM4V4_MOD_LVOFF            351
#define BSIM4V4_MOD_LDELTA           352
#define BSIM4V4_MOD_LRDSW            353
#define BSIM4V4_MOD_LPRT             354
#define BSIM4V4_MOD_LLDD             355
#define BSIM4V4_MOD_LETA             356
#define BSIM4V4_MOD_LETA0            357
#define BSIM4V4_MOD_LETAB            358
#define BSIM4V4_MOD_LPCLM            359
#define BSIM4V4_MOD_LPDIBL1          360
#define BSIM4V4_MOD_LPDIBL2          361
#define BSIM4V4_MOD_LPSCBE1          362
#define BSIM4V4_MOD_LPSCBE2          363
#define BSIM4V4_MOD_LPVAG            364
#define BSIM4V4_MOD_LWR              365
#define BSIM4V4_MOD_LDWG             366
#define BSIM4V4_MOD_LDWB             367
#define BSIM4V4_MOD_LB0              368
#define BSIM4V4_MOD_LB1              369
#define BSIM4V4_MOD_LALPHA0          370
#define BSIM4V4_MOD_LBETA0           371
#define BSIM4V4_MOD_LPDIBLB          374
#define BSIM4V4_MOD_LPRWG            375
#define BSIM4V4_MOD_LPRWB            376
#define BSIM4V4_MOD_LCDSCD           377
#define BSIM4V4_MOD_LAGS             378

#define BSIM4V4_MOD_LFRINGE          381
#define BSIM4V4_MOD_LCGSL            383
#define BSIM4V4_MOD_LCGDL            384
#define BSIM4V4_MOD_LCKAPPAS         385
#define BSIM4V4_MOD_LCF              386
#define BSIM4V4_MOD_LCLC             387
#define BSIM4V4_MOD_LCLE             388
#define BSIM4V4_MOD_LVFBCV           389
#define BSIM4V4_MOD_LACDE            390
#define BSIM4V4_MOD_LMOIN            391
#define BSIM4V4_MOD_LNOFF            392
#define BSIM4V4_MOD_LALPHA1          394
#define BSIM4V4_MOD_LVFB             395
#define BSIM4V4_MOD_LVOFFCV          396
#define BSIM4V4_MOD_LAGIDL           397
#define BSIM4V4_MOD_LBGIDL           398
#define BSIM4V4_MOD_LEGIDL           399
#define BSIM4V4_MOD_LXRCRG1          400
#define BSIM4V4_MOD_LXRCRG2          401
#define BSIM4V4_MOD_LEU              402
#define BSIM4V4_MOD_LMINV            403
#define BSIM4V4_MOD_LPDITS           404
#define BSIM4V4_MOD_LPDITSD          405
#define BSIM4V4_MOD_LFPROUT          406
#define BSIM4V4_MOD_LLPEB            407
#define BSIM4V4_MOD_LDVTP0           408
#define BSIM4V4_MOD_LDVTP1           409
#define BSIM4V4_MOD_LCGIDL           410
#define BSIM4V4_MOD_LPHIN            411
#define BSIM4V4_MOD_LRSW             412
#define BSIM4V4_MOD_LRDW             413
#define BSIM4V4_MOD_LNSD             414
#define BSIM4V4_MOD_LCKAPPAD         415
#define BSIM4V4_MOD_LAIGC            416
#define BSIM4V4_MOD_LBIGC            417
#define BSIM4V4_MOD_LCIGC            418
#define BSIM4V4_MOD_LAIGBACC         419            
#define BSIM4V4_MOD_LBIGBACC         420
#define BSIM4V4_MOD_LCIGBACC         421
#define BSIM4V4_MOD_LAIGBINV         422
#define BSIM4V4_MOD_LBIGBINV         423
#define BSIM4V4_MOD_LCIGBINV         424
#define BSIM4V4_MOD_LNIGC            425
#define BSIM4V4_MOD_LNIGBACC         426
#define BSIM4V4_MOD_LNIGBINV         427
#define BSIM4V4_MOD_LNTOX            428
#define BSIM4V4_MOD_LEIGBINV         429
#define BSIM4V4_MOD_LPIGCD           430
#define BSIM4V4_MOD_LPOXEDGE         431
#define BSIM4V4_MOD_LAIGSD           432
#define BSIM4V4_MOD_LBIGSD           433
#define BSIM4V4_MOD_LCIGSD           434

#define BSIM4V4_MOD_LLAMBDA          435
#define BSIM4V4_MOD_LVTL             436
#define BSIM4V4_MOD_LXN              437
#define BSIM4V4_MOD_LVFBSDOFF        438

/* Width dependence */
#define BSIM4V4_MOD_WCDSC            481
#define BSIM4V4_MOD_WCDSCB           482
#define BSIM4V4_MOD_WCIT             483
#define BSIM4V4_MOD_WNFACTOR         484
#define BSIM4V4_MOD_WXJ              485
#define BSIM4V4_MOD_WVSAT            486
#define BSIM4V4_MOD_WAT              487
#define BSIM4V4_MOD_WA0              488
#define BSIM4V4_MOD_WA1              489
#define BSIM4V4_MOD_WA2              490
#define BSIM4V4_MOD_WKETA            491
#define BSIM4V4_MOD_WNSUB            492
#define BSIM4V4_MOD_WNDEP            493
#define BSIM4V4_MOD_WNGATE           495
#define BSIM4V4_MOD_WGAMMA1          496
#define BSIM4V4_MOD_WGAMMA2          497
#define BSIM4V4_MOD_WVBX             498
#define BSIM4V4_MOD_WVBM             500
#define BSIM4V4_MOD_WXT              502
#define BSIM4V4_MOD_WK1              505
#define BSIM4V4_MOD_WKT1             506
#define BSIM4V4_MOD_WKT1L            507
#define BSIM4V4_MOD_WK2              508
#define BSIM4V4_MOD_WKT2             509
#define BSIM4V4_MOD_WK3              510
#define BSIM4V4_MOD_WK3B             511
#define BSIM4V4_MOD_WW0              512
#define BSIM4V4_MOD_WLPE0            513
#define BSIM4V4_MOD_WDVT0            514
#define BSIM4V4_MOD_WDVT1            515
#define BSIM4V4_MOD_WDVT2            516
#define BSIM4V4_MOD_WDVT0W           517
#define BSIM4V4_MOD_WDVT1W           518
#define BSIM4V4_MOD_WDVT2W           519
#define BSIM4V4_MOD_WDROUT           520
#define BSIM4V4_MOD_WDSUB            521
#define BSIM4V4_MOD_WVTH0            522
#define BSIM4V4_MOD_WUA              523
#define BSIM4V4_MOD_WUA1             524
#define BSIM4V4_MOD_WUB              525
#define BSIM4V4_MOD_WUB1             526
#define BSIM4V4_MOD_WUC              527
#define BSIM4V4_MOD_WUC1             528
#define BSIM4V4_MOD_WU0              529
#define BSIM4V4_MOD_WUTE             530
#define BSIM4V4_MOD_WVOFF            531
#define BSIM4V4_MOD_WDELTA           532
#define BSIM4V4_MOD_WRDSW            533
#define BSIM4V4_MOD_WPRT             534
#define BSIM4V4_MOD_WLDD             535
#define BSIM4V4_MOD_WETA             536
#define BSIM4V4_MOD_WETA0            537
#define BSIM4V4_MOD_WETAB            538
#define BSIM4V4_MOD_WPCLM            539
#define BSIM4V4_MOD_WPDIBL1          540
#define BSIM4V4_MOD_WPDIBL2          541
#define BSIM4V4_MOD_WPSCBE1          542
#define BSIM4V4_MOD_WPSCBE2          543
#define BSIM4V4_MOD_WPVAG            544
#define BSIM4V4_MOD_WWR              545
#define BSIM4V4_MOD_WDWG             546
#define BSIM4V4_MOD_WDWB             547
#define BSIM4V4_MOD_WB0              548
#define BSIM4V4_MOD_WB1              549
#define BSIM4V4_MOD_WALPHA0          550
#define BSIM4V4_MOD_WBETA0           551
#define BSIM4V4_MOD_WPDIBLB          554
#define BSIM4V4_MOD_WPRWG            555
#define BSIM4V4_MOD_WPRWB            556
#define BSIM4V4_MOD_WCDSCD           557
#define BSIM4V4_MOD_WAGS             558

#define BSIM4V4_MOD_WFRINGE          561
#define BSIM4V4_MOD_WCGSL            563
#define BSIM4V4_MOD_WCGDL            564
#define BSIM4V4_MOD_WCKAPPAS         565
#define BSIM4V4_MOD_WCF              566
#define BSIM4V4_MOD_WCLC             567
#define BSIM4V4_MOD_WCLE             568
#define BSIM4V4_MOD_WVFBCV           569
#define BSIM4V4_MOD_WACDE            570
#define BSIM4V4_MOD_WMOIN            571
#define BSIM4V4_MOD_WNOFF            572
#define BSIM4V4_MOD_WALPHA1          574
#define BSIM4V4_MOD_WVFB             575
#define BSIM4V4_MOD_WVOFFCV          576
#define BSIM4V4_MOD_WAGIDL           577
#define BSIM4V4_MOD_WBGIDL           578
#define BSIM4V4_MOD_WEGIDL           579
#define BSIM4V4_MOD_WXRCRG1          580
#define BSIM4V4_MOD_WXRCRG2          581
#define BSIM4V4_MOD_WEU              582
#define BSIM4V4_MOD_WMINV            583
#define BSIM4V4_MOD_WPDITS           584
#define BSIM4V4_MOD_WPDITSD          585
#define BSIM4V4_MOD_WFPROUT          586
#define BSIM4V4_MOD_WLPEB            587
#define BSIM4V4_MOD_WDVTP0           588
#define BSIM4V4_MOD_WDVTP1           589
#define BSIM4V4_MOD_WCGIDL           590
#define BSIM4V4_MOD_WPHIN            591
#define BSIM4V4_MOD_WRSW             592
#define BSIM4V4_MOD_WRDW             593
#define BSIM4V4_MOD_WNSD             594
#define BSIM4V4_MOD_WCKAPPAD         595
#define BSIM4V4_MOD_WAIGC            596
#define BSIM4V4_MOD_WBIGC            597
#define BSIM4V4_MOD_WCIGC            598
#define BSIM4V4_MOD_WAIGBACC         599            
#define BSIM4V4_MOD_WBIGBACC         600
#define BSIM4V4_MOD_WCIGBACC         601
#define BSIM4V4_MOD_WAIGBINV         602
#define BSIM4V4_MOD_WBIGBINV         603
#define BSIM4V4_MOD_WCIGBINV         604
#define BSIM4V4_MOD_WNIGC            605
#define BSIM4V4_MOD_WNIGBACC         606
#define BSIM4V4_MOD_WNIGBINV         607
#define BSIM4V4_MOD_WNTOX            608
#define BSIM4V4_MOD_WEIGBINV         609
#define BSIM4V4_MOD_WPIGCD           610
#define BSIM4V4_MOD_WPOXEDGE         611
#define BSIM4V4_MOD_WAIGSD           612
#define BSIM4V4_MOD_WBIGSD           613
#define BSIM4V4_MOD_WCIGSD           614
#define BSIM4V4_MOD_WLAMBDA          615
#define BSIM4V4_MOD_WVTL             616
#define BSIM4V4_MOD_WXN              617
#define BSIM4V4_MOD_WVFBSDOFF        618

/* Cross-term dependence */
#define BSIM4V4_MOD_PCDSC            661
#define BSIM4V4_MOD_PCDSCB           662
#define BSIM4V4_MOD_PCIT             663
#define BSIM4V4_MOD_PNFACTOR         664
#define BSIM4V4_MOD_PXJ              665
#define BSIM4V4_MOD_PVSAT            666
#define BSIM4V4_MOD_PAT              667
#define BSIM4V4_MOD_PA0              668
#define BSIM4V4_MOD_PA1              669
#define BSIM4V4_MOD_PA2              670
#define BSIM4V4_MOD_PKETA            671
#define BSIM4V4_MOD_PNSUB            672
#define BSIM4V4_MOD_PNDEP            673
#define BSIM4V4_MOD_PNGATE           675
#define BSIM4V4_MOD_PGAMMA1          676
#define BSIM4V4_MOD_PGAMMA2          677
#define BSIM4V4_MOD_PVBX             678

#define BSIM4V4_MOD_PVBM             680

#define BSIM4V4_MOD_PXT              682
#define BSIM4V4_MOD_PK1              685
#define BSIM4V4_MOD_PKT1             686
#define BSIM4V4_MOD_PKT1L            687
#define BSIM4V4_MOD_PK2              688
#define BSIM4V4_MOD_PKT2             689
#define BSIM4V4_MOD_PK3              690
#define BSIM4V4_MOD_PK3B             691
#define BSIM4V4_MOD_PW0              692
#define BSIM4V4_MOD_PLPE0            693

#define BSIM4V4_MOD_PDVT0            694
#define BSIM4V4_MOD_PDVT1            695
#define BSIM4V4_MOD_PDVT2            696

#define BSIM4V4_MOD_PDVT0W           697
#define BSIM4V4_MOD_PDVT1W           698
#define BSIM4V4_MOD_PDVT2W           699

#define BSIM4V4_MOD_PDROUT           700
#define BSIM4V4_MOD_PDSUB            701
#define BSIM4V4_MOD_PVTH0            702
#define BSIM4V4_MOD_PUA              703
#define BSIM4V4_MOD_PUA1             704
#define BSIM4V4_MOD_PUB              705
#define BSIM4V4_MOD_PUB1             706
#define BSIM4V4_MOD_PUC              707
#define BSIM4V4_MOD_PUC1             708
#define BSIM4V4_MOD_PU0              709
#define BSIM4V4_MOD_PUTE             710
#define BSIM4V4_MOD_PVOFF            711
#define BSIM4V4_MOD_PDELTA           712
#define BSIM4V4_MOD_PRDSW            713
#define BSIM4V4_MOD_PPRT             714
#define BSIM4V4_MOD_PLDD             715
#define BSIM4V4_MOD_PETA             716
#define BSIM4V4_MOD_PETA0            717
#define BSIM4V4_MOD_PETAB            718
#define BSIM4V4_MOD_PPCLM            719
#define BSIM4V4_MOD_PPDIBL1          720
#define BSIM4V4_MOD_PPDIBL2          721
#define BSIM4V4_MOD_PPSCBE1          722
#define BSIM4V4_MOD_PPSCBE2          723
#define BSIM4V4_MOD_PPVAG            724
#define BSIM4V4_MOD_PWR              725
#define BSIM4V4_MOD_PDWG             726
#define BSIM4V4_MOD_PDWB             727
#define BSIM4V4_MOD_PB0              728
#define BSIM4V4_MOD_PB1              729
#define BSIM4V4_MOD_PALPHA0          730
#define BSIM4V4_MOD_PBETA0           731
#define BSIM4V4_MOD_PPDIBLB          734

#define BSIM4V4_MOD_PPRWG            735
#define BSIM4V4_MOD_PPRWB            736

#define BSIM4V4_MOD_PCDSCD           737
#define BSIM4V4_MOD_PAGS             738

#define BSIM4V4_MOD_PFRINGE          741
#define BSIM4V4_MOD_PCGSL            743
#define BSIM4V4_MOD_PCGDL            744
#define BSIM4V4_MOD_PCKAPPAS         745
#define BSIM4V4_MOD_PCF              746
#define BSIM4V4_MOD_PCLC             747
#define BSIM4V4_MOD_PCLE             748
#define BSIM4V4_MOD_PVFBCV           749
#define BSIM4V4_MOD_PACDE            750
#define BSIM4V4_MOD_PMOIN            751
#define BSIM4V4_MOD_PNOFF            752
#define BSIM4V4_MOD_PALPHA1          754
#define BSIM4V4_MOD_PVFB             755
#define BSIM4V4_MOD_PVOFFCV          756
#define BSIM4V4_MOD_PAGIDL           757
#define BSIM4V4_MOD_PBGIDL           758
#define BSIM4V4_MOD_PEGIDL           759
#define BSIM4V4_MOD_PXRCRG1          760
#define BSIM4V4_MOD_PXRCRG2          761
#define BSIM4V4_MOD_PEU              762
#define BSIM4V4_MOD_PMINV            763
#define BSIM4V4_MOD_PPDITS           764
#define BSIM4V4_MOD_PPDITSD          765
#define BSIM4V4_MOD_PFPROUT          766
#define BSIM4V4_MOD_PLPEB            767
#define BSIM4V4_MOD_PDVTP0           768
#define BSIM4V4_MOD_PDVTP1           769
#define BSIM4V4_MOD_PCGIDL           770
#define BSIM4V4_MOD_PPHIN            771
#define BSIM4V4_MOD_PRSW             772
#define BSIM4V4_MOD_PRDW             773
#define BSIM4V4_MOD_PNSD             774
#define BSIM4V4_MOD_PCKAPPAD         775
#define BSIM4V4_MOD_PAIGC            776
#define BSIM4V4_MOD_PBIGC            777
#define BSIM4V4_MOD_PCIGC            778
#define BSIM4V4_MOD_PAIGBACC         779            
#define BSIM4V4_MOD_PBIGBACC         780
#define BSIM4V4_MOD_PCIGBACC         781
#define BSIM4V4_MOD_PAIGBINV         782
#define BSIM4V4_MOD_PBIGBINV         783
#define BSIM4V4_MOD_PCIGBINV         784
#define BSIM4V4_MOD_PNIGC            785
#define BSIM4V4_MOD_PNIGBACC         786
#define BSIM4V4_MOD_PNIGBINV         787
#define BSIM4V4_MOD_PNTOX            788
#define BSIM4V4_MOD_PEIGBINV         789
#define BSIM4V4_MOD_PPIGCD           790
#define BSIM4V4_MOD_PPOXEDGE         791
#define BSIM4V4_MOD_PAIGSD           792
#define BSIM4V4_MOD_PBIGSD           793
#define BSIM4V4_MOD_PCIGSD           794

#define BSIM4V4_MOD_SAREF            795
#define BSIM4V4_MOD_SBREF            796
#define BSIM4V4_MOD_KU0              797
#define BSIM4V4_MOD_KVSAT            798
#define BSIM4V4_MOD_TKU0             799
#define BSIM4V4_MOD_LLODKU0          800
#define BSIM4V4_MOD_WLODKU0          801
#define BSIM4V4_MOD_LLODVTH          802
#define BSIM4V4_MOD_WLODVTH          803
#define BSIM4V4_MOD_LKU0             804
#define BSIM4V4_MOD_WKU0             805
#define BSIM4V4_MOD_PKU0             806
#define BSIM4V4_MOD_KVTH0            807
#define BSIM4V4_MOD_LKVTH0           808
#define BSIM4V4_MOD_WKVTH0           809
#define BSIM4V4_MOD_PKVTH0           810
#define BSIM4V4_MOD_WLOD		   811
#define BSIM4V4_MOD_STK2		   812
#define BSIM4V4_MOD_LODK2		   813
#define BSIM4V4_MOD_STETA0	   814
#define BSIM4V4_MOD_LODETA0	   815

#define BSIM4V4_MOD_PLAMBDA          825
#define BSIM4V4_MOD_PVTL             826
#define BSIM4V4_MOD_PXN              827
#define BSIM4V4_MOD_PVFBSDOFF        828

#define BSIM4V4_MOD_TNOM             831
#define BSIM4V4_MOD_CGSO             832
#define BSIM4V4_MOD_CGDO             833
#define BSIM4V4_MOD_CGBO             834
#define BSIM4V4_MOD_XPART            835
#define BSIM4V4_MOD_RSH              836
#define BSIM4V4_MOD_JSS              837
#define BSIM4V4_MOD_PBS              838
#define BSIM4V4_MOD_MJS              839
#define BSIM4V4_MOD_PBSWS            840
#define BSIM4V4_MOD_MJSWS            841
#define BSIM4V4_MOD_CJS              842
#define BSIM4V4_MOD_CJSWS            843
#define BSIM4V4_MOD_NMOS             844
#define BSIM4V4_MOD_PMOS             845
#define BSIM4V4_MOD_NOIA             846
#define BSIM4V4_MOD_NOIB             847
#define BSIM4V4_MOD_NOIC             848
#define BSIM4V4_MOD_LINT             849
#define BSIM4V4_MOD_LL               850
#define BSIM4V4_MOD_LLN              851
#define BSIM4V4_MOD_LW               852
#define BSIM4V4_MOD_LWN              853
#define BSIM4V4_MOD_LWL              854
#define BSIM4V4_MOD_LMIN             855
#define BSIM4V4_MOD_LMAX             856
#define BSIM4V4_MOD_WINT             857
#define BSIM4V4_MOD_WL               858
#define BSIM4V4_MOD_WLN              859
#define BSIM4V4_MOD_WW               860
#define BSIM4V4_MOD_WWN              861
#define BSIM4V4_MOD_WWL              862
#define BSIM4V4_MOD_WMIN             863
#define BSIM4V4_MOD_WMAX             864
#define BSIM4V4_MOD_DWC              865
#define BSIM4V4_MOD_DLC              866
#define BSIM4V4_MOD_XL               867
#define BSIM4V4_MOD_XW               868
#define BSIM4V4_MOD_EM               869
#define BSIM4V4_MOD_EF               870
#define BSIM4V4_MOD_AF               871
#define BSIM4V4_MOD_KF               872
#define BSIM4V4_MOD_NJS              873
#define BSIM4V4_MOD_XTIS             874
#define BSIM4V4_MOD_PBSWGS           875
#define BSIM4V4_MOD_MJSWGS           876
#define BSIM4V4_MOD_CJSWGS           877
#define BSIM4V4_MOD_JSWS             878
#define BSIM4V4_MOD_LLC              879
#define BSIM4V4_MOD_LWC              880
#define BSIM4V4_MOD_LWLC             881
#define BSIM4V4_MOD_WLC              882
#define BSIM4V4_MOD_WWC              883
#define BSIM4V4_MOD_WWLC             884
#define BSIM4V4_MOD_DWJ              885
#define BSIM4V4_MOD_JSD              886
#define BSIM4V4_MOD_PBD              887
#define BSIM4V4_MOD_MJD              888
#define BSIM4V4_MOD_PBSWD            889
#define BSIM4V4_MOD_MJSWD            890
#define BSIM4V4_MOD_CJD              891
#define BSIM4V4_MOD_CJSWD            892
#define BSIM4V4_MOD_NJD              893
#define BSIM4V4_MOD_XTID             894
#define BSIM4V4_MOD_PBSWGD           895
#define BSIM4V4_MOD_MJSWGD           896
#define BSIM4V4_MOD_CJSWGD           897
#define BSIM4V4_MOD_JSWD             898
#define BSIM4V4_MOD_DLCIG            899

/* trap-assisted tunneling */

#define BSIM4V4_MOD_JTSS             900
#define BSIM4V4_MOD_JTSD		   901
#define BSIM4V4_MOD_JTSSWS	   902
#define BSIM4V4_MOD_JTSSWD	   903
#define BSIM4V4_MOD_JTSSWGS	   904
#define BSIM4V4_MOD_JTSSWGD	   905
#define BSIM4V4_MOD_NJTS	 	   906
#define BSIM4V4_MOD_NJTSSW	   907
#define BSIM4V4_MOD_NJTSSWG	   908
#define BSIM4V4_MOD_XTSS		   909
#define BSIM4V4_MOD_XTSD		   910
#define BSIM4V4_MOD_XTSSWS	   911
#define BSIM4V4_MOD_XTSSWD	   912
#define BSIM4V4_MOD_XTSSWGS	   913
#define BSIM4V4_MOD_XTSSWGD	   914
#define BSIM4V4_MOD_TNJTS		   915
#define BSIM4V4_MOD_TNJTSSW	   916
#define BSIM4V4_MOD_TNJTSSWG	   917
#define BSIM4V4_MOD_VTSS             918
#define BSIM4V4_MOD_VTSD		   919
#define BSIM4V4_MOD_VTSSWS	   920
#define BSIM4V4_MOD_VTSSWD	   921
#define BSIM4V4_MOD_VTSSWGS	   922
#define BSIM4V4_MOD_VTSSWGD	   923

#define BSIM4V4_MOD_STIMOD	   930
#define BSIM4V4_MOD_SA0	           931
#define BSIM4V4_MOD_SB0	           932

#define BSIM4V4_MOD_RGEOMOD	   933

/* device questions */
#define BSIM4V4_DNODE                945
#define BSIM4V4_GNODEEXT             946
#define BSIM4V4_SNODE                947
#define BSIM4V4_BNODE                948
#define BSIM4V4_DNODEPRIME           949
#define BSIM4V4_GNODEPRIME           950
#define BSIM4V4_GNODEMIDE            951
#define BSIM4V4_GNODEMID             952
#define BSIM4V4_SNODEPRIME           953
#define BSIM4V4_BNODEPRIME           954
#define BSIM4V4_DBNODE               955
#define BSIM4V4_SBNODE               956
#define BSIM4V4_VBD                  957
#define BSIM4V4_VBS                  958
#define BSIM4V4_VGS                  959
#define BSIM4V4_VDS                  960
#define BSIM4V4_CD                   961
#define BSIM4V4_CBS                  962
#define BSIM4V4_CBD                  963
#define BSIM4V4_GM                   964
#define BSIM4V4_GDS                  965
#define BSIM4V4_GMBS                 966
#define BSIM4V4_GBD                  967
#define BSIM4V4_GBS                  968
#define BSIM4V4_QB                   969
#define BSIM4V4_CQB                  970
#define BSIM4V4_QG                   971
#define BSIM4V4_CQG                  972
#define BSIM4V4_QD                   973
#define BSIM4V4_CQD                  974
#define BSIM4V4_CGGB                 975
#define BSIM4V4_CGDB                 976
#define BSIM4V4_CGSB                 977
#define BSIM4V4_CBGB                 978
#define BSIM4V4_CAPBD                979
#define BSIM4V4_CQBD                 980
#define BSIM4V4_CAPBS                981
#define BSIM4V4_CQBS                 982
#define BSIM4V4_CDGB                 983
#define BSIM4V4_CDDB                 984
#define BSIM4V4_CDSB                 985
#define BSIM4V4_VON                  986
#define BSIM4V4_VDSAT                987
#define BSIM4V4_QBS                  988
#define BSIM4V4_QBD                  989
#define BSIM4V4_SOURCECONDUCT        990
#define BSIM4V4_DRAINCONDUCT         991
#define BSIM4V4_CBDB                 992
#define BSIM4V4_CBSB                 993
#define BSIM4V4_CSUB		   994
#define BSIM4V4_QINV		   995
#define BSIM4V4_IGIDL		   996
#define BSIM4V4_CSGB                 997
#define BSIM4V4_CSDB                 998
#define BSIM4V4_CSSB                 999
#define BSIM4V4_CGBB                 1000
#define BSIM4V4_CDBB                 1001
#define BSIM4V4_CSBB                 1002
#define BSIM4V4_CBBB                 1003
#define BSIM4V4_QS                   1004
#define BSIM4V4_IGISL		   1005
#define BSIM4V4_IGS		   1006
#define BSIM4V4_IGD		   1007
#define BSIM4V4_IGB		   1008
#define BSIM4V4_IGCS		   1009
#define BSIM4V4_IGCD		   1010

#
#include "bsim4v4ext.h"


extern void BSIM4V4evaluate(double,double,double,BSIM4V4instance*,BSIM4V4model*,
        double*,double*,double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, CKTcircuit*);
extern int BSIM4V4debug(BSIM4V4model*, BSIM4V4instance*, CKTcircuit*, int);
extern int BSIM4V4checkModel(BSIM4V4model*, BSIM4V4instance*, CKTcircuit*);

#endif /*BSIM4V4*/
