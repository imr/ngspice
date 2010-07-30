/**********
Copyright 2003 Regents of the University of California.  All rights reserved.
Author: 2000 Weidong Liu.
Modified by Xuemei Xi, 11/15/2002.
Modified by Xuemei Xi, 05/09/2003.
File: bsim4v3def.h
**********/

#ifndef BSIM4V3
#define BSIM4V3

#include "ifsim.h"
#include "gendefs.h"
#include "cktdefs.h"
#include "complex.h"
#include "noisedef.h"         

typedef struct sBSIM4v3instance
{
    struct sBSIM4v3model *BSIM4v3modPtr;
    struct sBSIM4v3instance *BSIM4v3nextInstance;
    IFuid BSIM4v3name;
    int BSIM4v3owner;      /* Number of owner process */
    int BSIM4v3states;     /* index into state table for this device */
    int BSIM4v3dNode;
    int BSIM4v3gNodeExt;
    int BSIM4v3sNode;
    int BSIM4v3bNode;
    int BSIM4v3dNodePrime;
    int BSIM4v3gNodePrime;
    int BSIM4v3gNodeMid;
    int BSIM4v3sNodePrime;
    int BSIM4v3bNodePrime;
    int BSIM4v3dbNode;
    int BSIM4v3sbNode;
    int BSIM4v3qNode;

    double BSIM4v3ueff;
    double BSIM4v3thetavth; 
    double BSIM4v3von;
    double BSIM4v3vdsat;
    double BSIM4v3cgdo;
    double BSIM4v3qgdo;
    double BSIM4v3cgso;
    double BSIM4v3qgso;
    double BSIM4v3grbsb;
    double BSIM4v3grbdb;
    double BSIM4v3grbpb;
    double BSIM4v3grbps;
    double BSIM4v3grbpd;

    double BSIM4v3vjsmFwd;
    double BSIM4v3vjsmRev;
    double BSIM4v3vjdmFwd;
    double BSIM4v3vjdmRev;
    double BSIM4v3XExpBVS;
    double BSIM4v3XExpBVD;
    double BSIM4v3SslpFwd;
    double BSIM4v3SslpRev;
    double BSIM4v3DslpFwd;
    double BSIM4v3DslpRev;
    double BSIM4v3IVjsmFwd;
    double BSIM4v3IVjsmRev;
    double BSIM4v3IVjdmFwd;
    double BSIM4v3IVjdmRev;

    double BSIM4v3grgeltd;
    double BSIM4v3Pseff;
    double BSIM4v3Pdeff;
    double BSIM4v3Aseff;
    double BSIM4v3Adeff;

    double BSIM4v3l;
    double BSIM4v3w;
    double BSIM4v3drainArea;
    double BSIM4v3sourceArea;
    double BSIM4v3drainSquares;
    double BSIM4v3sourceSquares;
    double BSIM4v3drainPerimeter;
    double BSIM4v3sourcePerimeter;
    double BSIM4v3sourceConductance;
    double BSIM4v3drainConductance;
     /* stress effect instance param */
    double BSIM4v3sa;
    double BSIM4v3sb;
    double BSIM4v3sd;

    double BSIM4v3rbdb;
    double BSIM4v3rbsb;
    double BSIM4v3rbpb;
    double BSIM4v3rbps;
    double BSIM4v3rbpd;

     /* added here to account stress effect instance dependence */
    double BSIM4v3u0temp;
    double BSIM4v3vsattemp;
    double BSIM4v3vth0;
    double BSIM4v3vfb;
    double BSIM4v3vfbzb;
    double BSIM4v3vtfbphi1;
    double BSIM4v3vtfbphi2;
    double BSIM4v3k2;
    double BSIM4v3vbsc;
    double BSIM4v3k2ox;
    double BSIM4v3eta0;

    double BSIM4v3icVDS;
    double BSIM4v3icVGS;
    double BSIM4v3icVBS;
    double BSIM4v3nf;
    double BSIM4v3m;  
    int BSIM4v3off;
    int BSIM4v3mode;
    int BSIM4v3trnqsMod;
    int BSIM4v3acnqsMod;
    int BSIM4v3rbodyMod;
    int BSIM4v3rgateMod;
    int BSIM4v3geoMod;
    int BSIM4v3rgeoMod;
    int BSIM4v3min;


    /* OP point */
    double BSIM4v3Vgsteff;
    double BSIM4v3vgs_eff;
    double BSIM4v3vgd_eff;
    double BSIM4v3dvgs_eff_dvg;
    double BSIM4v3dvgd_eff_dvg;
    double BSIM4v3Vdseff;
    double BSIM4v3nstar;
    double BSIM4v3Abulk;
    double BSIM4v3EsatL;
    double BSIM4v3AbovVgst2Vtm;
    double BSIM4v3qinv;
    double BSIM4v3cd;
    double BSIM4v3cbs;
    double BSIM4v3cbd;
    double BSIM4v3csub;
    double BSIM4v3Igidl;
    double BSIM4v3Igisl;
    double BSIM4v3gm;
    double BSIM4v3gds;
    double BSIM4v3gmbs;
    double BSIM4v3gbd;
    double BSIM4v3gbs;

    double BSIM4v3gbbs;
    double BSIM4v3gbgs;
    double BSIM4v3gbds;
    double BSIM4v3ggidld;
    double BSIM4v3ggidlg;
    double BSIM4v3ggidls;
    double BSIM4v3ggidlb;
    double BSIM4v3ggisld;
    double BSIM4v3ggislg;
    double BSIM4v3ggisls;
    double BSIM4v3ggislb;

    double BSIM4v3Igcs;
    double BSIM4v3gIgcsg;
    double BSIM4v3gIgcsd;
    double BSIM4v3gIgcss;
    double BSIM4v3gIgcsb;
    double BSIM4v3Igcd;
    double BSIM4v3gIgcdg;
    double BSIM4v3gIgcdd;
    double BSIM4v3gIgcds;
    double BSIM4v3gIgcdb;

    double BSIM4v3Igs;
    double BSIM4v3gIgsg;
    double BSIM4v3gIgss;
    double BSIM4v3Igd;
    double BSIM4v3gIgdg;
    double BSIM4v3gIgdd;

    double BSIM4v3Igb;
    double BSIM4v3gIgbg;
    double BSIM4v3gIgbd;
    double BSIM4v3gIgbs;
    double BSIM4v3gIgbb;

    double BSIM4v3grdsw;
    double BSIM4v3IdovVds;
    double BSIM4v3gcrg;
    double BSIM4v3gcrgd;
    double BSIM4v3gcrgg;
    double BSIM4v3gcrgs;
    double BSIM4v3gcrgb;

    double BSIM4v3gstot;
    double BSIM4v3gstotd;
    double BSIM4v3gstotg;
    double BSIM4v3gstots;
    double BSIM4v3gstotb;

    double BSIM4v3gdtot;
    double BSIM4v3gdtotd;
    double BSIM4v3gdtotg;
    double BSIM4v3gdtots;
    double BSIM4v3gdtotb;

    double BSIM4v3cggb;
    double BSIM4v3cgdb;
    double BSIM4v3cgsb;
    double BSIM4v3cbgb;
    double BSIM4v3cbdb;
    double BSIM4v3cbsb;
    double BSIM4v3cdgb;
    double BSIM4v3cddb;
    double BSIM4v3cdsb;
    double BSIM4v3csgb;
    double BSIM4v3csdb;
    double BSIM4v3cssb;
    double BSIM4v3cgbb;
    double BSIM4v3cdbb;
    double BSIM4v3csbb;
    double BSIM4v3cbbb;
    double BSIM4v3capbd;
    double BSIM4v3capbs;

    double BSIM4v3cqgb;
    double BSIM4v3cqdb;
    double BSIM4v3cqsb;
    double BSIM4v3cqbb;

    double BSIM4v3qgate;
    double BSIM4v3qbulk;
    double BSIM4v3qdrn;
    double BSIM4v3qsrc;

    double BSIM4v3qchqs;
    double BSIM4v3taunet;
    double BSIM4v3gtau;
    double BSIM4v3gtg;
    double BSIM4v3gtd;
    double BSIM4v3gts;
    double BSIM4v3gtb;

    struct bsim4v3SizeDependParam  *pParam;

    unsigned BSIM4v3lGiven :1;
    unsigned BSIM4v3wGiven :1;
    unsigned BSIM4v3nfGiven :1;
    unsigned BSIM4v3mGiven :1;
    unsigned BSIM4v3minGiven :1;
    unsigned BSIM4v3drainAreaGiven :1;
    unsigned BSIM4v3sourceAreaGiven    :1;
    unsigned BSIM4v3drainSquaresGiven  :1;
    unsigned BSIM4v3sourceSquaresGiven :1;
    unsigned BSIM4v3drainPerimeterGiven    :1;
    unsigned BSIM4v3sourcePerimeterGiven   :1;
    unsigned BSIM4v3saGiven :1;
    unsigned BSIM4v3sbGiven :1;
    unsigned BSIM4v3sdGiven :1;
    unsigned BSIM4v3rbdbGiven   :1;
    unsigned BSIM4v3rbsbGiven   :1;
    unsigned BSIM4v3rbpbGiven   :1;
    unsigned BSIM4v3rbpdGiven   :1;
    unsigned BSIM4v3rbpsGiven   :1;
    unsigned BSIM4v3icVDSGiven :1;
    unsigned BSIM4v3icVGSGiven :1;
    unsigned BSIM4v3icVBSGiven :1;
    unsigned BSIM4v3trnqsModGiven :1;
    unsigned BSIM4v3acnqsModGiven :1;
    unsigned BSIM4v3rbodyModGiven :1;
    unsigned BSIM4v3rgateModGiven :1;
    unsigned BSIM4v3geoModGiven :1;
    unsigned BSIM4v3rgeoModGiven :1;


    double *BSIM4v3DPdPtr;
    double *BSIM4v3DPdpPtr;
    double *BSIM4v3DPgpPtr;
    double *BSIM4v3DPgmPtr;
    double *BSIM4v3DPspPtr;
    double *BSIM4v3DPbpPtr;
    double *BSIM4v3DPdbPtr;

    double *BSIM4v3DdPtr;
    double *BSIM4v3DdpPtr;

    double *BSIM4v3GPdpPtr;
    double *BSIM4v3GPgpPtr;
    double *BSIM4v3GPgmPtr;
    double *BSIM4v3GPgePtr;
    double *BSIM4v3GPspPtr;
    double *BSIM4v3GPbpPtr;

    double *BSIM4v3GMdpPtr;
    double *BSIM4v3GMgpPtr;
    double *BSIM4v3GMgmPtr;
    double *BSIM4v3GMgePtr;
    double *BSIM4v3GMspPtr;
    double *BSIM4v3GMbpPtr;

    double *BSIM4v3GEdpPtr;
    double *BSIM4v3GEgpPtr;
    double *BSIM4v3GEgmPtr;
    double *BSIM4v3GEgePtr;
    double *BSIM4v3GEspPtr;
    double *BSIM4v3GEbpPtr;

    double *BSIM4v3SPdpPtr;
    double *BSIM4v3SPgpPtr;
    double *BSIM4v3SPgmPtr;
    double *BSIM4v3SPsPtr;
    double *BSIM4v3SPspPtr;
    double *BSIM4v3SPbpPtr;
    double *BSIM4v3SPsbPtr;

    double *BSIM4v3SspPtr;
    double *BSIM4v3SsPtr;

    double *BSIM4v3BPdpPtr;
    double *BSIM4v3BPgpPtr;
    double *BSIM4v3BPgmPtr;
    double *BSIM4v3BPspPtr;
    double *BSIM4v3BPdbPtr;
    double *BSIM4v3BPbPtr;
    double *BSIM4v3BPsbPtr;
    double *BSIM4v3BPbpPtr;

    double *BSIM4v3DBdpPtr;
    double *BSIM4v3DBdbPtr;
    double *BSIM4v3DBbpPtr;
    double *BSIM4v3DBbPtr;

    double *BSIM4v3SBspPtr;
    double *BSIM4v3SBbpPtr;
    double *BSIM4v3SBbPtr;
    double *BSIM4v3SBsbPtr;

    double *BSIM4v3BdbPtr;
    double *BSIM4v3BbpPtr;
    double *BSIM4v3BsbPtr;
    double *BSIM4v3BbPtr;

    double *BSIM4v3DgpPtr;
    double *BSIM4v3DspPtr;
    double *BSIM4v3DbpPtr;
    double *BSIM4v3SdpPtr;
    double *BSIM4v3SgpPtr;
    double *BSIM4v3SbpPtr;

    double *BSIM4v3QdpPtr;
    double *BSIM4v3QgpPtr;
    double *BSIM4v3QspPtr;
    double *BSIM4v3QbpPtr;
    double *BSIM4v3QqPtr;
    double *BSIM4v3DPqPtr;
    double *BSIM4v3GPqPtr;
    double *BSIM4v3SPqPtr;


#define BSIM4v3vbd BSIM4v3states+ 0
#define BSIM4v3vbs BSIM4v3states+ 1
#define BSIM4v3vgs BSIM4v3states+ 2
#define BSIM4v3vds BSIM4v3states+ 3
#define BSIM4v3vdbs BSIM4v3states+ 4
#define BSIM4v3vdbd BSIM4v3states+ 5
#define BSIM4v3vsbs BSIM4v3states+ 6
#define BSIM4v3vges BSIM4v3states+ 7
#define BSIM4v3vgms BSIM4v3states+ 8
#define BSIM4v3vses BSIM4v3states+ 9
#define BSIM4v3vdes BSIM4v3states+ 10

#define BSIM4v3qb BSIM4v3states+ 11
#define BSIM4v3cqb BSIM4v3states+ 12
#define BSIM4v3qg BSIM4v3states+ 13
#define BSIM4v3cqg BSIM4v3states+ 14
#define BSIM4v3qd BSIM4v3states+ 15
#define BSIM4v3cqd BSIM4v3states+ 16
#define BSIM4v3qgmid BSIM4v3states+ 17
#define BSIM4v3cqgmid BSIM4v3states+ 18

#define BSIM4v3qbs  BSIM4v3states+ 19
#define BSIM4v3cqbs  BSIM4v3states+ 20
#define BSIM4v3qbd  BSIM4v3states+ 21
#define BSIM4v3cqbd  BSIM4v3states+ 22

#define BSIM4v3qcheq BSIM4v3states+ 23
#define BSIM4v3cqcheq BSIM4v3states+ 24
#define BSIM4v3qcdump BSIM4v3states+ 25
#define BSIM4v3cqcdump BSIM4v3states+ 26
#define BSIM4v3qdef BSIM4v3states+ 27
#define BSIM4v3qs BSIM4v3states+ 28

#define BSIM4v3numStates 29


/* indices to the array of BSIM4v3 NOISE SOURCES */

#define BSIM4v3RDNOIZ       0
#define BSIM4v3RSNOIZ       1
#define BSIM4v3RGNOIZ       2
#define BSIM4v3RBPSNOIZ     3
#define BSIM4v3RBPDNOIZ     4
#define BSIM4v3RBPBNOIZ     5
#define BSIM4v3RBSBNOIZ     6
#define BSIM4v3RBDBNOIZ     7
#define BSIM4v3IDNOIZ       8
#define BSIM4v3FLNOIZ       9
#define BSIM4v3IGSNOIZ      10
#define BSIM4v3IGDNOIZ      11
#define BSIM4v3IGBNOIZ      12
#define BSIM4v3TOTNOIZ      13

#define BSIM4v3NSRCS        14  /* Number of BSIM4v3 noise sources */

#ifndef NONOISE
    double BSIM4v3nVar[NSTATVARS][BSIM4v3NSRCS];
#else /* NONOISE */
        double **BSIM4v3nVar;
#endif /* NONOISE */

} BSIM4v3instance ;

struct bsim4v3SizeDependParam
{
    double Width;
    double Length;
    double NFinger;

    double BSIM4v3cdsc;           
    double BSIM4v3cdscb;    
    double BSIM4v3cdscd;       
    double BSIM4v3cit;           
    double BSIM4v3nfactor;      
    double BSIM4v3xj;
    double BSIM4v3vsat;         
    double BSIM4v3at;         
    double BSIM4v3a0;   
    double BSIM4v3ags;      
    double BSIM4v3a1;         
    double BSIM4v3a2;         
    double BSIM4v3keta;     
    double BSIM4v3nsub;
    double BSIM4v3ndep;        
    double BSIM4v3nsd;
    double BSIM4v3phin;
    double BSIM4v3ngate;        
    double BSIM4v3gamma1;      
    double BSIM4v3gamma2;     
    double BSIM4v3vbx;      
    double BSIM4v3vbi;       
    double BSIM4v3vbm;       
    double BSIM4v3vbsc;       
    double BSIM4v3xt;       
    double BSIM4v3phi;
    double BSIM4v3litl;
    double BSIM4v3k1;
    double BSIM4v3kt1;
    double BSIM4v3kt1l;
    double BSIM4v3kt2;
    double BSIM4v3k2;
    double BSIM4v3k3;
    double BSIM4v3k3b;
    double BSIM4v3w0;
    double BSIM4v3dvtp0;
    double BSIM4v3dvtp1;
    double BSIM4v3lpe0;
    double BSIM4v3lpeb;
    double BSIM4v3dvt0;      
    double BSIM4v3dvt1;      
    double BSIM4v3dvt2;      
    double BSIM4v3dvt0w;      
    double BSIM4v3dvt1w;      
    double BSIM4v3dvt2w;      
    double BSIM4v3drout;      
    double BSIM4v3dsub;      
    double BSIM4v3vth0;
    double BSIM4v3ua;
    double BSIM4v3ua1;
    double BSIM4v3ub;
    double BSIM4v3ub1;
    double BSIM4v3uc;
    double BSIM4v3uc1;
    double BSIM4v3u0;
    double BSIM4v3eu;
    double BSIM4v3ute;
    double BSIM4v3voff;
    double BSIM4v3minv;
    double BSIM4v3vfb;
    double BSIM4v3delta;
    double BSIM4v3rdsw;       
    double BSIM4v3rds0;       
    double BSIM4v3rs0;
    double BSIM4v3rd0;
    double BSIM4v3rsw;
    double BSIM4v3rdw;
    double BSIM4v3prwg;       
    double BSIM4v3prwb;       
    double BSIM4v3prt;       
    double BSIM4v3eta0;         
    double BSIM4v3etab;         
    double BSIM4v3pclm;      
    double BSIM4v3pdibl1;      
    double BSIM4v3pdibl2;      
    double BSIM4v3pdiblb;      
    double BSIM4v3fprout;
    double BSIM4v3pdits;
    double BSIM4v3pditsd;
    double BSIM4v3pscbe1;       
    double BSIM4v3pscbe2;       
    double BSIM4v3pvag;       
    double BSIM4v3wr;
    double BSIM4v3dwg;
    double BSIM4v3dwb;
    double BSIM4v3b0;
    double BSIM4v3b1;
    double BSIM4v3alpha0;
    double BSIM4v3alpha1;
    double BSIM4v3beta0;
    double BSIM4v3agidl;
    double BSIM4v3bgidl;
    double BSIM4v3cgidl;
    double BSIM4v3egidl;
    double BSIM4v3aigc;
    double BSIM4v3bigc;
    double BSIM4v3cigc;
    double BSIM4v3aigsd;
    double BSIM4v3bigsd;
    double BSIM4v3cigsd;
    double BSIM4v3aigbacc;
    double BSIM4v3bigbacc;
    double BSIM4v3cigbacc;
    double BSIM4v3aigbinv;
    double BSIM4v3bigbinv;
    double BSIM4v3cigbinv;
    double BSIM4v3nigc;
    double BSIM4v3nigbacc;
    double BSIM4v3nigbinv;
    double BSIM4v3ntox;
    double BSIM4v3eigbinv;
    double BSIM4v3pigcd;
    double BSIM4v3poxedge;
    double BSIM4v3xrcrg1;
    double BSIM4v3xrcrg2;
    double BSIM4v3lambda; /* overshoot */
    double BSIM4v3vtl; /* thermal velocity limit */
    double BSIM4v3xn; /* back scattering parameter */
    double BSIM4v3lc; /* back scattering parameter */
    double BSIM4v3tfactor;  /* ballistic transportation factor  */

/* added for stress effect */
    double BSIM4v3ku0;
    double BSIM4v3kvth0;
    double BSIM4v3ku0temp;
    double BSIM4v3rho_ref;
    double BSIM4v3inv_od_ref;

    /* CV model */
    double BSIM4v3cgsl;
    double BSIM4v3cgdl;
    double BSIM4v3ckappas;
    double BSIM4v3ckappad;
    double BSIM4v3cf;
    double BSIM4v3clc;
    double BSIM4v3cle;
    double BSIM4v3vfbcv;
    double BSIM4v3noff;
    double BSIM4v3voffcv;
    double BSIM4v3acde;
    double BSIM4v3moin;

/* Pre-calculated constants */

    double BSIM4v3dw;
    double BSIM4v3dl;
    double BSIM4v3leff;
    double BSIM4v3weff;

    double BSIM4v3dwc;
    double BSIM4v3dlc;
    double BSIM4v3dlcig;
    double BSIM4v3dwj;
    double BSIM4v3leffCV;
    double BSIM4v3weffCV;
    double BSIM4v3weffCJ;
    double BSIM4v3abulkCVfactor;
    double BSIM4v3cgso;
    double BSIM4v3cgdo;
    double BSIM4v3cgbo;

    double BSIM4v3u0temp;       
    double BSIM4v3vsattemp;   
    double BSIM4v3sqrtPhi;   
    double BSIM4v3phis3;   
    double BSIM4v3Xdep0;          
    double BSIM4v3sqrtXdep0;          
    double BSIM4v3theta0vb0;
    double BSIM4v3thetaRout; 
    double BSIM4v3mstar;
    double BSIM4v3voffcbn;
    double BSIM4v3rdswmin;
    double BSIM4v3rdwmin;
    double BSIM4v3rswmin;
    double BSIM4v3vfbsd;

    double BSIM4v3cof1;
    double BSIM4v3cof2;
    double BSIM4v3cof3;
    double BSIM4v3cof4;
    double BSIM4v3cdep0;
    double BSIM4v3vfbzb;
    double BSIM4v3vtfbphi1;
    double BSIM4v3vtfbphi2;
    double BSIM4v3ToxRatio;
    double BSIM4v3Aechvb;
    double BSIM4v3Bechvb;
    double BSIM4v3ToxRatioEdge;
    double BSIM4v3AechvbEdge;
    double BSIM4v3BechvbEdge;
    double BSIM4v3ldeb;
    double BSIM4v3k1ox;
    double BSIM4v3k2ox;

    struct bsim4v3SizeDependParam  *pNext;
};


typedef struct sBSIM4v3model 
{
    int BSIM4v3modType;
    struct sBSIM4v3model *BSIM4v3nextModel;
    BSIM4v3instance *BSIM4v3instances;
    IFuid BSIM4v3modName; 
    int BSIM4v3type;

    int    BSIM4v3mobMod;
    int    BSIM4v3capMod;
    int    BSIM4v3dioMod;
    int    BSIM4v3trnqsMod;
    int    BSIM4v3acnqsMod;
    int    BSIM4v3fnoiMod;
    int    BSIM4v3tnoiMod;
    int    BSIM4v3rdsMod;
    int    BSIM4v3rbodyMod;
    int    BSIM4v3rgateMod;
    int    BSIM4v3perMod;
    int    BSIM4v3geoMod;
    int    BSIM4v3igcMod;
    int    BSIM4v3igbMod;
    int    BSIM4v3tempMod;
    int    BSIM4v3binUnit;
    int    BSIM4v3paramChk;
    char   *BSIM4v3version;             
    double BSIM4v3toxe;             
    double BSIM4v3toxp;
    double BSIM4v3toxm;
    double BSIM4v3dtox;
    double BSIM4v3epsrox;
    double BSIM4v3cdsc;           
    double BSIM4v3cdscb; 
    double BSIM4v3cdscd;          
    double BSIM4v3cit;           
    double BSIM4v3nfactor;      
    double BSIM4v3xj;
    double BSIM4v3vsat;         
    double BSIM4v3at;         
    double BSIM4v3a0;   
    double BSIM4v3ags;      
    double BSIM4v3a1;         
    double BSIM4v3a2;         
    double BSIM4v3keta;     
    double BSIM4v3nsub;
    double BSIM4v3ndep;        
    double BSIM4v3nsd;
    double BSIM4v3phin;
    double BSIM4v3ngate;        
    double BSIM4v3gamma1;      
    double BSIM4v3gamma2;     
    double BSIM4v3vbx;      
    double BSIM4v3vbm;       
    double BSIM4v3xt;       
    double BSIM4v3k1;
    double BSIM4v3kt1;
    double BSIM4v3kt1l;
    double BSIM4v3kt2;
    double BSIM4v3k2;
    double BSIM4v3k3;
    double BSIM4v3k3b;
    double BSIM4v3w0;
    double BSIM4v3dvtp0;
    double BSIM4v3dvtp1;
    double BSIM4v3lpe0;
    double BSIM4v3lpeb;
    double BSIM4v3dvt0;      
    double BSIM4v3dvt1;      
    double BSIM4v3dvt2;      
    double BSIM4v3dvt0w;      
    double BSIM4v3dvt1w;      
    double BSIM4v3dvt2w;      
    double BSIM4v3drout;      
    double BSIM4v3dsub;      
    double BSIM4v3vth0;
    double BSIM4v3eu;
    double BSIM4v3ua;
    double BSIM4v3ua1;
    double BSIM4v3ub;
    double BSIM4v3ub1;
    double BSIM4v3uc;
    double BSIM4v3uc1;
    double BSIM4v3u0;
    double BSIM4v3ute;
    double BSIM4v3voff;
    double BSIM4v3minv;
    double BSIM4v3voffl;
    double BSIM4v3delta;
    double BSIM4v3rdsw;       
    double BSIM4v3rdswmin;
    double BSIM4v3rdwmin;
    double BSIM4v3rswmin;
    double BSIM4v3rsw;
    double BSIM4v3rdw;
    double BSIM4v3prwg;
    double BSIM4v3prwb;
    double BSIM4v3prt;       
    double BSIM4v3eta0;         
    double BSIM4v3etab;         
    double BSIM4v3pclm;      
    double BSIM4v3pdibl1;      
    double BSIM4v3pdibl2;      
    double BSIM4v3pdiblb;
    double BSIM4v3fprout;
    double BSIM4v3pdits;
    double BSIM4v3pditsd;
    double BSIM4v3pditsl;
    double BSIM4v3pscbe1;       
    double BSIM4v3pscbe2;       
    double BSIM4v3pvag;       
    double BSIM4v3wr;
    double BSIM4v3dwg;
    double BSIM4v3dwb;
    double BSIM4v3b0;
    double BSIM4v3b1;
    double BSIM4v3alpha0;
    double BSIM4v3alpha1;
    double BSIM4v3beta0;
    double BSIM4v3agidl;
    double BSIM4v3bgidl;
    double BSIM4v3cgidl;
    double BSIM4v3egidl;
    double BSIM4v3aigc;
    double BSIM4v3bigc;
    double BSIM4v3cigc;
    double BSIM4v3aigsd;
    double BSIM4v3bigsd;
    double BSIM4v3cigsd;
    double BSIM4v3aigbacc;
    double BSIM4v3bigbacc;
    double BSIM4v3cigbacc;
    double BSIM4v3aigbinv;
    double BSIM4v3bigbinv;
    double BSIM4v3cigbinv;
    double BSIM4v3nigc;
    double BSIM4v3nigbacc;
    double BSIM4v3nigbinv;
    double BSIM4v3ntox;
    double BSIM4v3eigbinv;
    double BSIM4v3pigcd;
    double BSIM4v3poxedge;
    double BSIM4v3toxref;
    double BSIM4v3ijthdfwd;
    double BSIM4v3ijthsfwd;
    double BSIM4v3ijthdrev;
    double BSIM4v3ijthsrev;
    double BSIM4v3xjbvd;
    double BSIM4v3xjbvs;
    double BSIM4v3bvd;
    double BSIM4v3bvs;
    double BSIM4v3xrcrg1;
    double BSIM4v3xrcrg2;
    double BSIM4v3lambda;
    double BSIM4v3vtl; 
    double BSIM4v3lc; 
    double BSIM4v3xn; 

    double BSIM4v3vfb;
    double BSIM4v3gbmin;
    double BSIM4v3rbdb;
    double BSIM4v3rbsb;
    double BSIM4v3rbpb;
    double BSIM4v3rbps;
    double BSIM4v3rbpd;
    double BSIM4v3tnoia;
    double BSIM4v3tnoib;
    double BSIM4v3rnoia;
    double BSIM4v3rnoib;
    double BSIM4v3ntnoi;

    /* CV model and Parasitics */
    double BSIM4v3cgsl;
    double BSIM4v3cgdl;
    double BSIM4v3ckappas;
    double BSIM4v3ckappad;
    double BSIM4v3cf;
    double BSIM4v3vfbcv;
    double BSIM4v3clc;
    double BSIM4v3cle;
    double BSIM4v3dwc;
    double BSIM4v3dlc;
    double BSIM4v3xw;
    double BSIM4v3xl;
    double BSIM4v3dlcig;
    double BSIM4v3dwj;
    double BSIM4v3noff;
    double BSIM4v3voffcv;
    double BSIM4v3acde;
    double BSIM4v3moin;
    double BSIM4v3tcj;
    double BSIM4v3tcjsw;
    double BSIM4v3tcjswg;
    double BSIM4v3tpb;
    double BSIM4v3tpbsw;
    double BSIM4v3tpbswg;
    double BSIM4v3dmcg;
    double BSIM4v3dmci;
    double BSIM4v3dmdg;
    double BSIM4v3dmcgt;
    double BSIM4v3xgw;
    double BSIM4v3xgl;
    double BSIM4v3rshg;
    double BSIM4v3ngcon;

    /* Length Dependence */
    double BSIM4v3lcdsc;           
    double BSIM4v3lcdscb; 
    double BSIM4v3lcdscd;          
    double BSIM4v3lcit;           
    double BSIM4v3lnfactor;      
    double BSIM4v3lxj;
    double BSIM4v3lvsat;         
    double BSIM4v3lat;         
    double BSIM4v3la0;   
    double BSIM4v3lags;      
    double BSIM4v3la1;         
    double BSIM4v3la2;         
    double BSIM4v3lketa;     
    double BSIM4v3lnsub;
    double BSIM4v3lndep;        
    double BSIM4v3lnsd;
    double BSIM4v3lphin;
    double BSIM4v3lngate;        
    double BSIM4v3lgamma1;      
    double BSIM4v3lgamma2;     
    double BSIM4v3lvbx;      
    double BSIM4v3lvbm;       
    double BSIM4v3lxt;       
    double BSIM4v3lk1;
    double BSIM4v3lkt1;
    double BSIM4v3lkt1l;
    double BSIM4v3lkt2;
    double BSIM4v3lk2;
    double BSIM4v3lk3;
    double BSIM4v3lk3b;
    double BSIM4v3lw0;
    double BSIM4v3ldvtp0;
    double BSIM4v3ldvtp1;
    double BSIM4v3llpe0;
    double BSIM4v3llpeb;
    double BSIM4v3ldvt0;      
    double BSIM4v3ldvt1;      
    double BSIM4v3ldvt2;      
    double BSIM4v3ldvt0w;      
    double BSIM4v3ldvt1w;      
    double BSIM4v3ldvt2w;      
    double BSIM4v3ldrout;      
    double BSIM4v3ldsub;      
    double BSIM4v3lvth0;
    double BSIM4v3lua;
    double BSIM4v3lua1;
    double BSIM4v3lub;
    double BSIM4v3lub1;
    double BSIM4v3luc;
    double BSIM4v3luc1;
    double BSIM4v3lu0;
    double BSIM4v3leu;
    double BSIM4v3lute;
    double BSIM4v3lvoff;
    double BSIM4v3lminv;
    double BSIM4v3ldelta;
    double BSIM4v3lrdsw;       
    double BSIM4v3lrsw;
    double BSIM4v3lrdw;
    double BSIM4v3lprwg;
    double BSIM4v3lprwb;
    double BSIM4v3lprt;       
    double BSIM4v3leta0;         
    double BSIM4v3letab;         
    double BSIM4v3lpclm;      
    double BSIM4v3lpdibl1;      
    double BSIM4v3lpdibl2;      
    double BSIM4v3lpdiblb;
    double BSIM4v3lfprout;
    double BSIM4v3lpdits;
    double BSIM4v3lpditsd;
    double BSIM4v3lpscbe1;       
    double BSIM4v3lpscbe2;       
    double BSIM4v3lpvag;       
    double BSIM4v3lwr;
    double BSIM4v3ldwg;
    double BSIM4v3ldwb;
    double BSIM4v3lb0;
    double BSIM4v3lb1;
    double BSIM4v3lalpha0;
    double BSIM4v3lalpha1;
    double BSIM4v3lbeta0;
    double BSIM4v3lvfb;
    double BSIM4v3lagidl;
    double BSIM4v3lbgidl;
    double BSIM4v3lcgidl;
    double BSIM4v3legidl;
    double BSIM4v3laigc;
    double BSIM4v3lbigc;
    double BSIM4v3lcigc;
    double BSIM4v3laigsd;
    double BSIM4v3lbigsd;
    double BSIM4v3lcigsd;
    double BSIM4v3laigbacc;
    double BSIM4v3lbigbacc;
    double BSIM4v3lcigbacc;
    double BSIM4v3laigbinv;
    double BSIM4v3lbigbinv;
    double BSIM4v3lcigbinv;
    double BSIM4v3lnigc;
    double BSIM4v3lnigbacc;
    double BSIM4v3lnigbinv;
    double BSIM4v3lntox;
    double BSIM4v3leigbinv;
    double BSIM4v3lpigcd;
    double BSIM4v3lpoxedge;
    double BSIM4v3lxrcrg1;
    double BSIM4v3lxrcrg2;
    double BSIM4v3llambda;
    double BSIM4v3lvtl; 
    double BSIM4v3lxn; 

    /* CV model */
    double BSIM4v3lcgsl;
    double BSIM4v3lcgdl;
    double BSIM4v3lckappas;
    double BSIM4v3lckappad;
    double BSIM4v3lcf;
    double BSIM4v3lclc;
    double BSIM4v3lcle;
    double BSIM4v3lvfbcv;
    double BSIM4v3lnoff;
    double BSIM4v3lvoffcv;
    double BSIM4v3lacde;
    double BSIM4v3lmoin;

    /* Width Dependence */
    double BSIM4v3wcdsc;           
    double BSIM4v3wcdscb; 
    double BSIM4v3wcdscd;          
    double BSIM4v3wcit;           
    double BSIM4v3wnfactor;      
    double BSIM4v3wxj;
    double BSIM4v3wvsat;         
    double BSIM4v3wat;         
    double BSIM4v3wa0;   
    double BSIM4v3wags;      
    double BSIM4v3wa1;         
    double BSIM4v3wa2;         
    double BSIM4v3wketa;     
    double BSIM4v3wnsub;
    double BSIM4v3wndep;        
    double BSIM4v3wnsd;
    double BSIM4v3wphin;
    double BSIM4v3wngate;        
    double BSIM4v3wgamma1;      
    double BSIM4v3wgamma2;     
    double BSIM4v3wvbx;      
    double BSIM4v3wvbm;       
    double BSIM4v3wxt;       
    double BSIM4v3wk1;
    double BSIM4v3wkt1;
    double BSIM4v3wkt1l;
    double BSIM4v3wkt2;
    double BSIM4v3wk2;
    double BSIM4v3wk3;
    double BSIM4v3wk3b;
    double BSIM4v3ww0;
    double BSIM4v3wdvtp0;
    double BSIM4v3wdvtp1;
    double BSIM4v3wlpe0;
    double BSIM4v3wlpeb;
    double BSIM4v3wdvt0;      
    double BSIM4v3wdvt1;      
    double BSIM4v3wdvt2;      
    double BSIM4v3wdvt0w;      
    double BSIM4v3wdvt1w;      
    double BSIM4v3wdvt2w;      
    double BSIM4v3wdrout;      
    double BSIM4v3wdsub;      
    double BSIM4v3wvth0;
    double BSIM4v3wua;
    double BSIM4v3wua1;
    double BSIM4v3wub;
    double BSIM4v3wub1;
    double BSIM4v3wuc;
    double BSIM4v3wuc1;
    double BSIM4v3wu0;
    double BSIM4v3weu;
    double BSIM4v3wute;
    double BSIM4v3wvoff;
    double BSIM4v3wminv;
    double BSIM4v3wdelta;
    double BSIM4v3wrdsw;       
    double BSIM4v3wrsw;
    double BSIM4v3wrdw;
    double BSIM4v3wprwg;
    double BSIM4v3wprwb;
    double BSIM4v3wprt;       
    double BSIM4v3weta0;         
    double BSIM4v3wetab;         
    double BSIM4v3wpclm;      
    double BSIM4v3wpdibl1;      
    double BSIM4v3wpdibl2;      
    double BSIM4v3wpdiblb;
    double BSIM4v3wfprout;
    double BSIM4v3wpdits;
    double BSIM4v3wpditsd;
    double BSIM4v3wpscbe1;       
    double BSIM4v3wpscbe2;       
    double BSIM4v3wpvag;       
    double BSIM4v3wwr;
    double BSIM4v3wdwg;
    double BSIM4v3wdwb;
    double BSIM4v3wb0;
    double BSIM4v3wb1;
    double BSIM4v3walpha0;
    double BSIM4v3walpha1;
    double BSIM4v3wbeta0;
    double BSIM4v3wvfb;
    double BSIM4v3wagidl;
    double BSIM4v3wbgidl;
    double BSIM4v3wcgidl;
    double BSIM4v3wegidl;
    double BSIM4v3waigc;
    double BSIM4v3wbigc;
    double BSIM4v3wcigc;
    double BSIM4v3waigsd;
    double BSIM4v3wbigsd;
    double BSIM4v3wcigsd;
    double BSIM4v3waigbacc;
    double BSIM4v3wbigbacc;
    double BSIM4v3wcigbacc;
    double BSIM4v3waigbinv;
    double BSIM4v3wbigbinv;
    double BSIM4v3wcigbinv;
    double BSIM4v3wnigc;
    double BSIM4v3wnigbacc;
    double BSIM4v3wnigbinv;
    double BSIM4v3wntox;
    double BSIM4v3weigbinv;
    double BSIM4v3wpigcd;
    double BSIM4v3wpoxedge;
    double BSIM4v3wxrcrg1;
    double BSIM4v3wxrcrg2;
    double BSIM4v3wlambda;
    double BSIM4v3wvtl; 
    double BSIM4v3wxn; 

    /* CV model */
    double BSIM4v3wcgsl;
    double BSIM4v3wcgdl;
    double BSIM4v3wckappas;
    double BSIM4v3wckappad;
    double BSIM4v3wcf;
    double BSIM4v3wclc;
    double BSIM4v3wcle;
    double BSIM4v3wvfbcv;
    double BSIM4v3wnoff;
    double BSIM4v3wvoffcv;
    double BSIM4v3wacde;
    double BSIM4v3wmoin;

    /* Cross-term Dependence */
    double BSIM4v3pcdsc;           
    double BSIM4v3pcdscb; 
    double BSIM4v3pcdscd;          
    double BSIM4v3pcit;           
    double BSIM4v3pnfactor;      
    double BSIM4v3pxj;
    double BSIM4v3pvsat;         
    double BSIM4v3pat;         
    double BSIM4v3pa0;   
    double BSIM4v3pags;      
    double BSIM4v3pa1;         
    double BSIM4v3pa2;         
    double BSIM4v3pketa;     
    double BSIM4v3pnsub;
    double BSIM4v3pndep;        
    double BSIM4v3pnsd;
    double BSIM4v3pphin;
    double BSIM4v3pngate;        
    double BSIM4v3pgamma1;      
    double BSIM4v3pgamma2;     
    double BSIM4v3pvbx;      
    double BSIM4v3pvbm;       
    double BSIM4v3pxt;       
    double BSIM4v3pk1;
    double BSIM4v3pkt1;
    double BSIM4v3pkt1l;
    double BSIM4v3pkt2;
    double BSIM4v3pk2;
    double BSIM4v3pk3;
    double BSIM4v3pk3b;
    double BSIM4v3pw0;
    double BSIM4v3pdvtp0;
    double BSIM4v3pdvtp1;
    double BSIM4v3plpe0;
    double BSIM4v3plpeb;
    double BSIM4v3pdvt0;      
    double BSIM4v3pdvt1;      
    double BSIM4v3pdvt2;      
    double BSIM4v3pdvt0w;      
    double BSIM4v3pdvt1w;      
    double BSIM4v3pdvt2w;      
    double BSIM4v3pdrout;      
    double BSIM4v3pdsub;      
    double BSIM4v3pvth0;
    double BSIM4v3pua;
    double BSIM4v3pua1;
    double BSIM4v3pub;
    double BSIM4v3pub1;
    double BSIM4v3puc;
    double BSIM4v3puc1;
    double BSIM4v3pu0;
    double BSIM4v3peu;
    double BSIM4v3pute;
    double BSIM4v3pvoff;
    double BSIM4v3pminv;
    double BSIM4v3pdelta;
    double BSIM4v3prdsw;
    double BSIM4v3prsw;
    double BSIM4v3prdw;
    double BSIM4v3pprwg;
    double BSIM4v3pprwb;
    double BSIM4v3pprt;       
    double BSIM4v3peta0;         
    double BSIM4v3petab;         
    double BSIM4v3ppclm;      
    double BSIM4v3ppdibl1;      
    double BSIM4v3ppdibl2;      
    double BSIM4v3ppdiblb;
    double BSIM4v3pfprout;
    double BSIM4v3ppdits;
    double BSIM4v3ppditsd;
    double BSIM4v3ppscbe1;       
    double BSIM4v3ppscbe2;       
    double BSIM4v3ppvag;       
    double BSIM4v3pwr;
    double BSIM4v3pdwg;
    double BSIM4v3pdwb;
    double BSIM4v3pb0;
    double BSIM4v3pb1;
    double BSIM4v3palpha0;
    double BSIM4v3palpha1;
    double BSIM4v3pbeta0;
    double BSIM4v3pvfb;
    double BSIM4v3pagidl;
    double BSIM4v3pbgidl;
    double BSIM4v3pcgidl;
    double BSIM4v3pegidl;
    double BSIM4v3paigc;
    double BSIM4v3pbigc;
    double BSIM4v3pcigc;
    double BSIM4v3paigsd;
    double BSIM4v3pbigsd;
    double BSIM4v3pcigsd;
    double BSIM4v3paigbacc;
    double BSIM4v3pbigbacc;
    double BSIM4v3pcigbacc;
    double BSIM4v3paigbinv;
    double BSIM4v3pbigbinv;
    double BSIM4v3pcigbinv;
    double BSIM4v3pnigc;
    double BSIM4v3pnigbacc;
    double BSIM4v3pnigbinv;
    double BSIM4v3pntox;
    double BSIM4v3peigbinv;
    double BSIM4v3ppigcd;
    double BSIM4v3ppoxedge;
    double BSIM4v3pxrcrg1;
    double BSIM4v3pxrcrg2;
    double BSIM4v3plambda;
    double BSIM4v3pvtl;
    double BSIM4v3pxn; 

    /* CV model */
    double BSIM4v3pcgsl;
    double BSIM4v3pcgdl;
    double BSIM4v3pckappas;
    double BSIM4v3pckappad;
    double BSIM4v3pcf;
    double BSIM4v3pclc;
    double BSIM4v3pcle;
    double BSIM4v3pvfbcv;
    double BSIM4v3pnoff;
    double BSIM4v3pvoffcv;
    double BSIM4v3pacde;
    double BSIM4v3pmoin;

    double BSIM4v3tnom;
    double BSIM4v3cgso;
    double BSIM4v3cgdo;
    double BSIM4v3cgbo;
    double BSIM4v3xpart;
    double BSIM4v3cFringOut;
    double BSIM4v3cFringMax;

    double BSIM4v3sheetResistance;
    double BSIM4v3SjctSatCurDensity;
    double BSIM4v3DjctSatCurDensity;
    double BSIM4v3SjctSidewallSatCurDensity;
    double BSIM4v3DjctSidewallSatCurDensity;
    double BSIM4v3SjctGateSidewallSatCurDensity;
    double BSIM4v3DjctGateSidewallSatCurDensity;
    double BSIM4v3SbulkJctPotential;
    double BSIM4v3DbulkJctPotential;
    double BSIM4v3SbulkJctBotGradingCoeff;
    double BSIM4v3DbulkJctBotGradingCoeff;
    double BSIM4v3SbulkJctSideGradingCoeff;
    double BSIM4v3DbulkJctSideGradingCoeff;
    double BSIM4v3SbulkJctGateSideGradingCoeff;
    double BSIM4v3DbulkJctGateSideGradingCoeff;
    double BSIM4v3SsidewallJctPotential;
    double BSIM4v3DsidewallJctPotential;
    double BSIM4v3SGatesidewallJctPotential;
    double BSIM4v3DGatesidewallJctPotential;
    double BSIM4v3SunitAreaJctCap;
    double BSIM4v3DunitAreaJctCap;
    double BSIM4v3SunitLengthSidewallJctCap;
    double BSIM4v3DunitLengthSidewallJctCap;
    double BSIM4v3SunitLengthGateSidewallJctCap;
    double BSIM4v3DunitLengthGateSidewallJctCap;
    double BSIM4v3SjctEmissionCoeff;
    double BSIM4v3DjctEmissionCoeff;
    double BSIM4v3SjctTempExponent;
    double BSIM4v3DjctTempExponent;

    double BSIM4v3Lint;
    double BSIM4v3Ll;
    double BSIM4v3Llc;
    double BSIM4v3Lln;
    double BSIM4v3Lw;
    double BSIM4v3Lwc;
    double BSIM4v3Lwn;
    double BSIM4v3Lwl;
    double BSIM4v3Lwlc;
    double BSIM4v3Lmin;
    double BSIM4v3Lmax;

    double BSIM4v3Wint;
    double BSIM4v3Wl;
    double BSIM4v3Wlc;
    double BSIM4v3Wln;
    double BSIM4v3Ww;
    double BSIM4v3Wwc;
    double BSIM4v3Wwn;
    double BSIM4v3Wwl;
    double BSIM4v3Wwlc;
    double BSIM4v3Wmin;
    double BSIM4v3Wmax;

    /* added for stress effect */
    double BSIM4v3saref;
    double BSIM4v3sbref;
    double BSIM4v3wlod;
    double BSIM4v3ku0;
    double BSIM4v3kvsat;
    double BSIM4v3kvth0;
    double BSIM4v3tku0;
    double BSIM4v3llodku0;
    double BSIM4v3wlodku0;
    double BSIM4v3llodvth;
    double BSIM4v3wlodvth;
    double BSIM4v3lku0;
    double BSIM4v3wku0;
    double BSIM4v3pku0;
    double BSIM4v3lkvth0;
    double BSIM4v3wkvth0;
    double BSIM4v3pkvth0;
    double BSIM4v3stk2;
    double BSIM4v3lodk2;
    double BSIM4v3steta0;
    double BSIM4v3lodeta0;



/* Pre-calculated constants
 * move to size-dependent param */
    double BSIM4v3vtm;   
    double BSIM4v3coxe;
    double BSIM4v3coxp;
    double BSIM4v3cof1;
    double BSIM4v3cof2;
    double BSIM4v3cof3;
    double BSIM4v3cof4;
    double BSIM4v3vcrit;
    double BSIM4v3factor1;
    double BSIM4v3PhiBS;
    double BSIM4v3PhiBSWS;
    double BSIM4v3PhiBSWGS;
    double BSIM4v3SjctTempSatCurDensity;
    double BSIM4v3SjctSidewallTempSatCurDensity;
    double BSIM4v3SjctGateSidewallTempSatCurDensity;
    double BSIM4v3PhiBD;
    double BSIM4v3PhiBSWD;
    double BSIM4v3PhiBSWGD;
    double BSIM4v3DjctTempSatCurDensity;
    double BSIM4v3DjctSidewallTempSatCurDensity;
    double BSIM4v3DjctGateSidewallTempSatCurDensity;
    double BSIM4v3SunitAreaTempJctCap;
    double BSIM4v3DunitAreaTempJctCap;
    double BSIM4v3SunitLengthSidewallTempJctCap;
    double BSIM4v3DunitLengthSidewallTempJctCap;
    double BSIM4v3SunitLengthGateSidewallTempJctCap;
    double BSIM4v3DunitLengthGateSidewallTempJctCap;

    double BSIM4v3oxideTrapDensityA;      
    double BSIM4v3oxideTrapDensityB;     
    double BSIM4v3oxideTrapDensityC;  
    double BSIM4v3em;  
    double BSIM4v3ef;  
    double BSIM4v3af;  
    double BSIM4v3kf;  

    struct bsim4v3SizeDependParam *pSizeDependParamKnot;

    /* Flags */
    unsigned  BSIM4v3mobModGiven :1;
    unsigned  BSIM4v3binUnitGiven :1;
    unsigned  BSIM4v3capModGiven :1;
    unsigned  BSIM4v3dioModGiven :1;
    unsigned  BSIM4v3rdsModGiven :1;
    unsigned  BSIM4v3rbodyModGiven :1;
    unsigned  BSIM4v3rgateModGiven :1;
    unsigned  BSIM4v3perModGiven :1;
    unsigned  BSIM4v3geoModGiven :1;
    unsigned  BSIM4v3paramChkGiven :1;
    unsigned  BSIM4v3trnqsModGiven :1;
    unsigned  BSIM4v3acnqsModGiven :1;
    unsigned  BSIM4v3fnoiModGiven :1;
    unsigned  BSIM4v3tnoiModGiven :1;
    unsigned  BSIM4v3igcModGiven :1;
    unsigned  BSIM4v3igbModGiven :1;
    unsigned  BSIM4v3tempModGiven :1;
    unsigned  BSIM4v3typeGiven   :1;
    unsigned  BSIM4v3toxrefGiven   :1;
    unsigned  BSIM4v3toxeGiven   :1;
    unsigned  BSIM4v3toxpGiven   :1;
    unsigned  BSIM4v3toxmGiven   :1;
    unsigned  BSIM4v3dtoxGiven   :1;
    unsigned  BSIM4v3epsroxGiven   :1;
    unsigned  BSIM4v3versionGiven   :1;
    unsigned  BSIM4v3cdscGiven   :1;
    unsigned  BSIM4v3cdscbGiven   :1;
    unsigned  BSIM4v3cdscdGiven   :1;
    unsigned  BSIM4v3citGiven   :1;
    unsigned  BSIM4v3nfactorGiven   :1;
    unsigned  BSIM4v3xjGiven   :1;
    unsigned  BSIM4v3vsatGiven   :1;
    unsigned  BSIM4v3atGiven   :1;
    unsigned  BSIM4v3a0Given   :1;
    unsigned  BSIM4v3agsGiven   :1;
    unsigned  BSIM4v3a1Given   :1;
    unsigned  BSIM4v3a2Given   :1;
    unsigned  BSIM4v3ketaGiven   :1;    
    unsigned  BSIM4v3nsubGiven   :1;
    unsigned  BSIM4v3ndepGiven   :1;
    unsigned  BSIM4v3nsdGiven    :1;
    unsigned  BSIM4v3phinGiven   :1;
    unsigned  BSIM4v3ngateGiven   :1;
    unsigned  BSIM4v3gamma1Given   :1;
    unsigned  BSIM4v3gamma2Given   :1;
    unsigned  BSIM4v3vbxGiven   :1;
    unsigned  BSIM4v3vbmGiven   :1;
    unsigned  BSIM4v3xtGiven   :1;
    unsigned  BSIM4v3k1Given   :1;
    unsigned  BSIM4v3kt1Given   :1;
    unsigned  BSIM4v3kt1lGiven   :1;
    unsigned  BSIM4v3kt2Given   :1;
    unsigned  BSIM4v3k2Given   :1;
    unsigned  BSIM4v3k3Given   :1;
    unsigned  BSIM4v3k3bGiven   :1;
    unsigned  BSIM4v3w0Given   :1;
    unsigned  BSIM4v3dvtp0Given :1;
    unsigned  BSIM4v3dvtp1Given :1;
    unsigned  BSIM4v3lpe0Given   :1;
    unsigned  BSIM4v3lpebGiven   :1;
    unsigned  BSIM4v3dvt0Given   :1;   
    unsigned  BSIM4v3dvt1Given   :1;     
    unsigned  BSIM4v3dvt2Given   :1;     
    unsigned  BSIM4v3dvt0wGiven   :1;   
    unsigned  BSIM4v3dvt1wGiven   :1;     
    unsigned  BSIM4v3dvt2wGiven   :1;     
    unsigned  BSIM4v3droutGiven   :1;     
    unsigned  BSIM4v3dsubGiven   :1;     
    unsigned  BSIM4v3vth0Given   :1;
    unsigned  BSIM4v3euGiven   :1;
    unsigned  BSIM4v3uaGiven   :1;
    unsigned  BSIM4v3ua1Given   :1;
    unsigned  BSIM4v3ubGiven   :1;
    unsigned  BSIM4v3ub1Given   :1;
    unsigned  BSIM4v3ucGiven   :1;
    unsigned  BSIM4v3uc1Given   :1;
    unsigned  BSIM4v3u0Given   :1;
    unsigned  BSIM4v3uteGiven   :1;
    unsigned  BSIM4v3voffGiven   :1;
    unsigned  BSIM4v3vofflGiven  :1;
    unsigned  BSIM4v3minvGiven   :1;
    unsigned  BSIM4v3rdswGiven   :1;      
    unsigned  BSIM4v3rdswminGiven :1;
    unsigned  BSIM4v3rdwminGiven :1;
    unsigned  BSIM4v3rswminGiven :1;
    unsigned  BSIM4v3rswGiven   :1;
    unsigned  BSIM4v3rdwGiven   :1;
    unsigned  BSIM4v3prwgGiven   :1;      
    unsigned  BSIM4v3prwbGiven   :1;      
    unsigned  BSIM4v3prtGiven   :1;      
    unsigned  BSIM4v3eta0Given   :1;    
    unsigned  BSIM4v3etabGiven   :1;    
    unsigned  BSIM4v3pclmGiven   :1;   
    unsigned  BSIM4v3pdibl1Given   :1;   
    unsigned  BSIM4v3pdibl2Given   :1;  
    unsigned  BSIM4v3pdiblbGiven   :1;  
    unsigned  BSIM4v3fproutGiven   :1;
    unsigned  BSIM4v3pditsGiven    :1;
    unsigned  BSIM4v3pditsdGiven    :1;
    unsigned  BSIM4v3pditslGiven    :1;
    unsigned  BSIM4v3pscbe1Given   :1;    
    unsigned  BSIM4v3pscbe2Given   :1;    
    unsigned  BSIM4v3pvagGiven   :1;    
    unsigned  BSIM4v3deltaGiven  :1;     
    unsigned  BSIM4v3wrGiven   :1;
    unsigned  BSIM4v3dwgGiven   :1;
    unsigned  BSIM4v3dwbGiven   :1;
    unsigned  BSIM4v3b0Given   :1;
    unsigned  BSIM4v3b1Given   :1;
    unsigned  BSIM4v3alpha0Given   :1;
    unsigned  BSIM4v3alpha1Given   :1;
    unsigned  BSIM4v3beta0Given   :1;
    unsigned  BSIM4v3agidlGiven   :1;
    unsigned  BSIM4v3bgidlGiven   :1;
    unsigned  BSIM4v3cgidlGiven   :1;
    unsigned  BSIM4v3egidlGiven   :1;
    unsigned  BSIM4v3aigcGiven   :1;
    unsigned  BSIM4v3bigcGiven   :1;
    unsigned  BSIM4v3cigcGiven   :1;
    unsigned  BSIM4v3aigsdGiven   :1;
    unsigned  BSIM4v3bigsdGiven   :1;
    unsigned  BSIM4v3cigsdGiven   :1;
    unsigned  BSIM4v3aigbaccGiven   :1;
    unsigned  BSIM4v3bigbaccGiven   :1;
    unsigned  BSIM4v3cigbaccGiven   :1;
    unsigned  BSIM4v3aigbinvGiven   :1;
    unsigned  BSIM4v3bigbinvGiven   :1;
    unsigned  BSIM4v3cigbinvGiven   :1;
    unsigned  BSIM4v3nigcGiven   :1;
    unsigned  BSIM4v3nigbinvGiven   :1;
    unsigned  BSIM4v3nigbaccGiven   :1;
    unsigned  BSIM4v3ntoxGiven   :1;
    unsigned  BSIM4v3eigbinvGiven   :1;
    unsigned  BSIM4v3pigcdGiven   :1;
    unsigned  BSIM4v3poxedgeGiven   :1;
    unsigned  BSIM4v3ijthdfwdGiven  :1;
    unsigned  BSIM4v3ijthsfwdGiven  :1;
    unsigned  BSIM4v3ijthdrevGiven  :1;
    unsigned  BSIM4v3ijthsrevGiven  :1;
    unsigned  BSIM4v3xjbvdGiven   :1;
    unsigned  BSIM4v3xjbvsGiven   :1;
    unsigned  BSIM4v3bvdGiven   :1;
    unsigned  BSIM4v3bvsGiven   :1;
    unsigned  BSIM4v3vfbGiven   :1;
    unsigned  BSIM4v3gbminGiven :1;
    unsigned  BSIM4v3rbdbGiven :1;
    unsigned  BSIM4v3rbsbGiven :1;
    unsigned  BSIM4v3rbpsGiven :1;
    unsigned  BSIM4v3rbpdGiven :1;
    unsigned  BSIM4v3rbpbGiven :1;
    unsigned  BSIM4v3xrcrg1Given   :1;
    unsigned  BSIM4v3xrcrg2Given   :1;
    unsigned  BSIM4v3tnoiaGiven    :1;
    unsigned  BSIM4v3tnoibGiven    :1;
    unsigned  BSIM4v3rnoiaGiven    :1;
    unsigned  BSIM4v3rnoibGiven    :1;
    unsigned  BSIM4v3ntnoiGiven    :1;

    unsigned  BSIM4v3lambdaGiven    :1;
    unsigned  BSIM4v3vtlGiven    :1;
    unsigned  BSIM4v3lcGiven    :1;
    unsigned  BSIM4v3xnGiven    :1;

    /* CV model and parasitics */
    unsigned  BSIM4v3cgslGiven   :1;
    unsigned  BSIM4v3cgdlGiven   :1;
    unsigned  BSIM4v3ckappasGiven   :1;
    unsigned  BSIM4v3ckappadGiven   :1;
    unsigned  BSIM4v3cfGiven   :1;
    unsigned  BSIM4v3vfbcvGiven   :1;
    unsigned  BSIM4v3clcGiven   :1;
    unsigned  BSIM4v3cleGiven   :1;
    unsigned  BSIM4v3dwcGiven   :1;
    unsigned  BSIM4v3dlcGiven   :1;
    unsigned  BSIM4v3xwGiven    :1;
    unsigned  BSIM4v3xlGiven    :1;
    unsigned  BSIM4v3dlcigGiven   :1;
    unsigned  BSIM4v3dwjGiven   :1;
    unsigned  BSIM4v3noffGiven  :1;
    unsigned  BSIM4v3voffcvGiven :1;
    unsigned  BSIM4v3acdeGiven  :1;
    unsigned  BSIM4v3moinGiven  :1;
    unsigned  BSIM4v3tcjGiven   :1;
    unsigned  BSIM4v3tcjswGiven :1;
    unsigned  BSIM4v3tcjswgGiven :1;
    unsigned  BSIM4v3tpbGiven    :1;
    unsigned  BSIM4v3tpbswGiven  :1;
    unsigned  BSIM4v3tpbswgGiven :1;
    unsigned  BSIM4v3dmcgGiven :1;
    unsigned  BSIM4v3dmciGiven :1;
    unsigned  BSIM4v3dmdgGiven :1;
    unsigned  BSIM4v3dmcgtGiven :1;
    unsigned  BSIM4v3xgwGiven :1;
    unsigned  BSIM4v3xglGiven :1;
    unsigned  BSIM4v3rshgGiven :1;
    unsigned  BSIM4v3ngconGiven :1;


    /* Length dependence */
    unsigned  BSIM4v3lcdscGiven   :1;
    unsigned  BSIM4v3lcdscbGiven   :1;
    unsigned  BSIM4v3lcdscdGiven   :1;
    unsigned  BSIM4v3lcitGiven   :1;
    unsigned  BSIM4v3lnfactorGiven   :1;
    unsigned  BSIM4v3lxjGiven   :1;
    unsigned  BSIM4v3lvsatGiven   :1;
    unsigned  BSIM4v3latGiven   :1;
    unsigned  BSIM4v3la0Given   :1;
    unsigned  BSIM4v3lagsGiven   :1;
    unsigned  BSIM4v3la1Given   :1;
    unsigned  BSIM4v3la2Given   :1;
    unsigned  BSIM4v3lketaGiven   :1;    
    unsigned  BSIM4v3lnsubGiven   :1;
    unsigned  BSIM4v3lndepGiven   :1;
    unsigned  BSIM4v3lnsdGiven    :1;
    unsigned  BSIM4v3lphinGiven   :1;
    unsigned  BSIM4v3lngateGiven   :1;
    unsigned  BSIM4v3lgamma1Given   :1;
    unsigned  BSIM4v3lgamma2Given   :1;
    unsigned  BSIM4v3lvbxGiven   :1;
    unsigned  BSIM4v3lvbmGiven   :1;
    unsigned  BSIM4v3lxtGiven   :1;
    unsigned  BSIM4v3lk1Given   :1;
    unsigned  BSIM4v3lkt1Given   :1;
    unsigned  BSIM4v3lkt1lGiven   :1;
    unsigned  BSIM4v3lkt2Given   :1;
    unsigned  BSIM4v3lk2Given   :1;
    unsigned  BSIM4v3lk3Given   :1;
    unsigned  BSIM4v3lk3bGiven   :1;
    unsigned  BSIM4v3lw0Given   :1;
    unsigned  BSIM4v3ldvtp0Given :1;
    unsigned  BSIM4v3ldvtp1Given :1;
    unsigned  BSIM4v3llpe0Given   :1;
    unsigned  BSIM4v3llpebGiven   :1;
    unsigned  BSIM4v3ldvt0Given   :1;   
    unsigned  BSIM4v3ldvt1Given   :1;     
    unsigned  BSIM4v3ldvt2Given   :1;     
    unsigned  BSIM4v3ldvt0wGiven   :1;   
    unsigned  BSIM4v3ldvt1wGiven   :1;     
    unsigned  BSIM4v3ldvt2wGiven   :1;     
    unsigned  BSIM4v3ldroutGiven   :1;     
    unsigned  BSIM4v3ldsubGiven   :1;     
    unsigned  BSIM4v3lvth0Given   :1;
    unsigned  BSIM4v3luaGiven   :1;
    unsigned  BSIM4v3lua1Given   :1;
    unsigned  BSIM4v3lubGiven   :1;
    unsigned  BSIM4v3lub1Given   :1;
    unsigned  BSIM4v3lucGiven   :1;
    unsigned  BSIM4v3luc1Given   :1;
    unsigned  BSIM4v3lu0Given   :1;
    unsigned  BSIM4v3leuGiven   :1;
    unsigned  BSIM4v3luteGiven   :1;
    unsigned  BSIM4v3lvoffGiven   :1;
    unsigned  BSIM4v3lminvGiven   :1;
    unsigned  BSIM4v3lrdswGiven   :1;      
    unsigned  BSIM4v3lrswGiven   :1;
    unsigned  BSIM4v3lrdwGiven   :1;
    unsigned  BSIM4v3lprwgGiven   :1;      
    unsigned  BSIM4v3lprwbGiven   :1;      
    unsigned  BSIM4v3lprtGiven   :1;      
    unsigned  BSIM4v3leta0Given   :1;    
    unsigned  BSIM4v3letabGiven   :1;    
    unsigned  BSIM4v3lpclmGiven   :1;   
    unsigned  BSIM4v3lpdibl1Given   :1;   
    unsigned  BSIM4v3lpdibl2Given   :1;  
    unsigned  BSIM4v3lpdiblbGiven   :1;  
    unsigned  BSIM4v3lfproutGiven   :1;
    unsigned  BSIM4v3lpditsGiven    :1;
    unsigned  BSIM4v3lpditsdGiven    :1;
    unsigned  BSIM4v3lpscbe1Given   :1;    
    unsigned  BSIM4v3lpscbe2Given   :1;    
    unsigned  BSIM4v3lpvagGiven   :1;    
    unsigned  BSIM4v3ldeltaGiven  :1;     
    unsigned  BSIM4v3lwrGiven   :1;
    unsigned  BSIM4v3ldwgGiven   :1;
    unsigned  BSIM4v3ldwbGiven   :1;
    unsigned  BSIM4v3lb0Given   :1;
    unsigned  BSIM4v3lb1Given   :1;
    unsigned  BSIM4v3lalpha0Given   :1;
    unsigned  BSIM4v3lalpha1Given   :1;
    unsigned  BSIM4v3lbeta0Given   :1;
    unsigned  BSIM4v3lvfbGiven   :1;
    unsigned  BSIM4v3lagidlGiven   :1;
    unsigned  BSIM4v3lbgidlGiven   :1;
    unsigned  BSIM4v3lcgidlGiven   :1;
    unsigned  BSIM4v3legidlGiven   :1;
    unsigned  BSIM4v3laigcGiven   :1;
    unsigned  BSIM4v3lbigcGiven   :1;
    unsigned  BSIM4v3lcigcGiven   :1;
    unsigned  BSIM4v3laigsdGiven   :1;
    unsigned  BSIM4v3lbigsdGiven   :1;
    unsigned  BSIM4v3lcigsdGiven   :1;
    unsigned  BSIM4v3laigbaccGiven   :1;
    unsigned  BSIM4v3lbigbaccGiven   :1;
    unsigned  BSIM4v3lcigbaccGiven   :1;
    unsigned  BSIM4v3laigbinvGiven   :1;
    unsigned  BSIM4v3lbigbinvGiven   :1;
    unsigned  BSIM4v3lcigbinvGiven   :1;
    unsigned  BSIM4v3lnigcGiven   :1;
    unsigned  BSIM4v3lnigbinvGiven   :1;
    unsigned  BSIM4v3lnigbaccGiven   :1;
    unsigned  BSIM4v3lntoxGiven   :1;
    unsigned  BSIM4v3leigbinvGiven   :1;
    unsigned  BSIM4v3lpigcdGiven   :1;
    unsigned  BSIM4v3lpoxedgeGiven   :1;
    unsigned  BSIM4v3lxrcrg1Given   :1;
    unsigned  BSIM4v3lxrcrg2Given   :1;
    unsigned  BSIM4v3llambdaGiven    :1;
    unsigned  BSIM4v3lvtlGiven    :1;
    unsigned  BSIM4v3lxnGiven    :1;

    /* CV model */
    unsigned  BSIM4v3lcgslGiven   :1;
    unsigned  BSIM4v3lcgdlGiven   :1;
    unsigned  BSIM4v3lckappasGiven   :1;
    unsigned  BSIM4v3lckappadGiven   :1;
    unsigned  BSIM4v3lcfGiven   :1;
    unsigned  BSIM4v3lclcGiven   :1;
    unsigned  BSIM4v3lcleGiven   :1;
    unsigned  BSIM4v3lvfbcvGiven   :1;
    unsigned  BSIM4v3lnoffGiven   :1;
    unsigned  BSIM4v3lvoffcvGiven :1;
    unsigned  BSIM4v3lacdeGiven   :1;
    unsigned  BSIM4v3lmoinGiven   :1;

    /* Width dependence */
    unsigned  BSIM4v3wcdscGiven   :1;
    unsigned  BSIM4v3wcdscbGiven   :1;
    unsigned  BSIM4v3wcdscdGiven   :1;
    unsigned  BSIM4v3wcitGiven   :1;
    unsigned  BSIM4v3wnfactorGiven   :1;
    unsigned  BSIM4v3wxjGiven   :1;
    unsigned  BSIM4v3wvsatGiven   :1;
    unsigned  BSIM4v3watGiven   :1;
    unsigned  BSIM4v3wa0Given   :1;
    unsigned  BSIM4v3wagsGiven   :1;
    unsigned  BSIM4v3wa1Given   :1;
    unsigned  BSIM4v3wa2Given   :1;
    unsigned  BSIM4v3wketaGiven   :1;    
    unsigned  BSIM4v3wnsubGiven   :1;
    unsigned  BSIM4v3wndepGiven   :1;
    unsigned  BSIM4v3wnsdGiven    :1;
    unsigned  BSIM4v3wphinGiven   :1;
    unsigned  BSIM4v3wngateGiven   :1;
    unsigned  BSIM4v3wgamma1Given   :1;
    unsigned  BSIM4v3wgamma2Given   :1;
    unsigned  BSIM4v3wvbxGiven   :1;
    unsigned  BSIM4v3wvbmGiven   :1;
    unsigned  BSIM4v3wxtGiven   :1;
    unsigned  BSIM4v3wk1Given   :1;
    unsigned  BSIM4v3wkt1Given   :1;
    unsigned  BSIM4v3wkt1lGiven   :1;
    unsigned  BSIM4v3wkt2Given   :1;
    unsigned  BSIM4v3wk2Given   :1;
    unsigned  BSIM4v3wk3Given   :1;
    unsigned  BSIM4v3wk3bGiven   :1;
    unsigned  BSIM4v3ww0Given   :1;
    unsigned  BSIM4v3wdvtp0Given :1;
    unsigned  BSIM4v3wdvtp1Given :1;
    unsigned  BSIM4v3wlpe0Given   :1;
    unsigned  BSIM4v3wlpebGiven   :1;
    unsigned  BSIM4v3wdvt0Given   :1;   
    unsigned  BSIM4v3wdvt1Given   :1;     
    unsigned  BSIM4v3wdvt2Given   :1;     
    unsigned  BSIM4v3wdvt0wGiven   :1;   
    unsigned  BSIM4v3wdvt1wGiven   :1;     
    unsigned  BSIM4v3wdvt2wGiven   :1;     
    unsigned  BSIM4v3wdroutGiven   :1;     
    unsigned  BSIM4v3wdsubGiven   :1;     
    unsigned  BSIM4v3wvth0Given   :1;
    unsigned  BSIM4v3wuaGiven   :1;
    unsigned  BSIM4v3wua1Given   :1;
    unsigned  BSIM4v3wubGiven   :1;
    unsigned  BSIM4v3wub1Given   :1;
    unsigned  BSIM4v3wucGiven   :1;
    unsigned  BSIM4v3wuc1Given   :1;
    unsigned  BSIM4v3wu0Given   :1;
    unsigned  BSIM4v3weuGiven   :1;
    unsigned  BSIM4v3wuteGiven   :1;
    unsigned  BSIM4v3wvoffGiven   :1;
    unsigned  BSIM4v3wminvGiven   :1;
    unsigned  BSIM4v3wrdswGiven   :1;      
    unsigned  BSIM4v3wrswGiven   :1;
    unsigned  BSIM4v3wrdwGiven   :1;
    unsigned  BSIM4v3wprwgGiven   :1;      
    unsigned  BSIM4v3wprwbGiven   :1;      
    unsigned  BSIM4v3wprtGiven   :1;      
    unsigned  BSIM4v3weta0Given   :1;    
    unsigned  BSIM4v3wetabGiven   :1;    
    unsigned  BSIM4v3wpclmGiven   :1;   
    unsigned  BSIM4v3wpdibl1Given   :1;   
    unsigned  BSIM4v3wpdibl2Given   :1;  
    unsigned  BSIM4v3wpdiblbGiven   :1;  
    unsigned  BSIM4v3wfproutGiven   :1;
    unsigned  BSIM4v3wpditsGiven    :1;
    unsigned  BSIM4v3wpditsdGiven    :1;
    unsigned  BSIM4v3wpscbe1Given   :1;    
    unsigned  BSIM4v3wpscbe2Given   :1;    
    unsigned  BSIM4v3wpvagGiven   :1;    
    unsigned  BSIM4v3wdeltaGiven  :1;     
    unsigned  BSIM4v3wwrGiven   :1;
    unsigned  BSIM4v3wdwgGiven   :1;
    unsigned  BSIM4v3wdwbGiven   :1;
    unsigned  BSIM4v3wb0Given   :1;
    unsigned  BSIM4v3wb1Given   :1;
    unsigned  BSIM4v3walpha0Given   :1;
    unsigned  BSIM4v3walpha1Given   :1;
    unsigned  BSIM4v3wbeta0Given   :1;
    unsigned  BSIM4v3wvfbGiven   :1;
    unsigned  BSIM4v3wagidlGiven   :1;
    unsigned  BSIM4v3wbgidlGiven   :1;
    unsigned  BSIM4v3wcgidlGiven   :1;
    unsigned  BSIM4v3wegidlGiven   :1;
    unsigned  BSIM4v3waigcGiven   :1;
    unsigned  BSIM4v3wbigcGiven   :1;
    unsigned  BSIM4v3wcigcGiven   :1;
    unsigned  BSIM4v3waigsdGiven   :1;
    unsigned  BSIM4v3wbigsdGiven   :1;
    unsigned  BSIM4v3wcigsdGiven   :1;
    unsigned  BSIM4v3waigbaccGiven   :1;
    unsigned  BSIM4v3wbigbaccGiven   :1;
    unsigned  BSIM4v3wcigbaccGiven   :1;
    unsigned  BSIM4v3waigbinvGiven   :1;
    unsigned  BSIM4v3wbigbinvGiven   :1;
    unsigned  BSIM4v3wcigbinvGiven   :1;
    unsigned  BSIM4v3wnigcGiven   :1;
    unsigned  BSIM4v3wnigbinvGiven   :1;
    unsigned  BSIM4v3wnigbaccGiven   :1;
    unsigned  BSIM4v3wntoxGiven   :1;
    unsigned  BSIM4v3weigbinvGiven   :1;
    unsigned  BSIM4v3wpigcdGiven   :1;
    unsigned  BSIM4v3wpoxedgeGiven   :1;
    unsigned  BSIM4v3wxrcrg1Given   :1;
    unsigned  BSIM4v3wxrcrg2Given   :1;
    unsigned  BSIM4v3wlambdaGiven    :1;
    unsigned  BSIM4v3wvtlGiven    :1;
    unsigned  BSIM4v3wxnGiven    :1;

    /* CV model */
    unsigned  BSIM4v3wcgslGiven   :1;
    unsigned  BSIM4v3wcgdlGiven   :1;
    unsigned  BSIM4v3wckappasGiven   :1;
    unsigned  BSIM4v3wckappadGiven   :1;
    unsigned  BSIM4v3wcfGiven   :1;
    unsigned  BSIM4v3wclcGiven   :1;
    unsigned  BSIM4v3wcleGiven   :1;
    unsigned  BSIM4v3wvfbcvGiven   :1;
    unsigned  BSIM4v3wnoffGiven   :1;
    unsigned  BSIM4v3wvoffcvGiven :1;
    unsigned  BSIM4v3wacdeGiven   :1;
    unsigned  BSIM4v3wmoinGiven   :1;

    /* Cross-term dependence */
    unsigned  BSIM4v3pcdscGiven   :1;
    unsigned  BSIM4v3pcdscbGiven   :1;
    unsigned  BSIM4v3pcdscdGiven   :1;
    unsigned  BSIM4v3pcitGiven   :1;
    unsigned  BSIM4v3pnfactorGiven   :1;
    unsigned  BSIM4v3pxjGiven   :1;
    unsigned  BSIM4v3pvsatGiven   :1;
    unsigned  BSIM4v3patGiven   :1;
    unsigned  BSIM4v3pa0Given   :1;
    unsigned  BSIM4v3pagsGiven   :1;
    unsigned  BSIM4v3pa1Given   :1;
    unsigned  BSIM4v3pa2Given   :1;
    unsigned  BSIM4v3pketaGiven   :1;    
    unsigned  BSIM4v3pnsubGiven   :1;
    unsigned  BSIM4v3pndepGiven   :1;
    unsigned  BSIM4v3pnsdGiven    :1;
    unsigned  BSIM4v3pphinGiven   :1;
    unsigned  BSIM4v3pngateGiven   :1;
    unsigned  BSIM4v3pgamma1Given   :1;
    unsigned  BSIM4v3pgamma2Given   :1;
    unsigned  BSIM4v3pvbxGiven   :1;
    unsigned  BSIM4v3pvbmGiven   :1;
    unsigned  BSIM4v3pxtGiven   :1;
    unsigned  BSIM4v3pk1Given   :1;
    unsigned  BSIM4v3pkt1Given   :1;
    unsigned  BSIM4v3pkt1lGiven   :1;
    unsigned  BSIM4v3pkt2Given   :1;
    unsigned  BSIM4v3pk2Given   :1;
    unsigned  BSIM4v3pk3Given   :1;
    unsigned  BSIM4v3pk3bGiven   :1;
    unsigned  BSIM4v3pw0Given   :1;
    unsigned  BSIM4v3pdvtp0Given :1;
    unsigned  BSIM4v3pdvtp1Given :1;
    unsigned  BSIM4v3plpe0Given   :1;
    unsigned  BSIM4v3plpebGiven   :1;
    unsigned  BSIM4v3pdvt0Given   :1;   
    unsigned  BSIM4v3pdvt1Given   :1;     
    unsigned  BSIM4v3pdvt2Given   :1;     
    unsigned  BSIM4v3pdvt0wGiven   :1;   
    unsigned  BSIM4v3pdvt1wGiven   :1;     
    unsigned  BSIM4v3pdvt2wGiven   :1;     
    unsigned  BSIM4v3pdroutGiven   :1;     
    unsigned  BSIM4v3pdsubGiven   :1;     
    unsigned  BSIM4v3pvth0Given   :1;
    unsigned  BSIM4v3puaGiven   :1;
    unsigned  BSIM4v3pua1Given   :1;
    unsigned  BSIM4v3pubGiven   :1;
    unsigned  BSIM4v3pub1Given   :1;
    unsigned  BSIM4v3pucGiven   :1;
    unsigned  BSIM4v3puc1Given   :1;
    unsigned  BSIM4v3pu0Given   :1;
    unsigned  BSIM4v3peuGiven   :1;
    unsigned  BSIM4v3puteGiven   :1;
    unsigned  BSIM4v3pvoffGiven   :1;
    unsigned  BSIM4v3pminvGiven   :1;
    unsigned  BSIM4v3prdswGiven   :1;      
    unsigned  BSIM4v3prswGiven   :1;
    unsigned  BSIM4v3prdwGiven   :1;
    unsigned  BSIM4v3pprwgGiven   :1;      
    unsigned  BSIM4v3pprwbGiven   :1;      
    unsigned  BSIM4v3pprtGiven   :1;      
    unsigned  BSIM4v3peta0Given   :1;    
    unsigned  BSIM4v3petabGiven   :1;    
    unsigned  BSIM4v3ppclmGiven   :1;   
    unsigned  BSIM4v3ppdibl1Given   :1;   
    unsigned  BSIM4v3ppdibl2Given   :1;  
    unsigned  BSIM4v3ppdiblbGiven   :1;  
    unsigned  BSIM4v3pfproutGiven   :1;
    unsigned  BSIM4v3ppditsGiven    :1;
    unsigned  BSIM4v3ppditsdGiven    :1;
    unsigned  BSIM4v3ppscbe1Given   :1;    
    unsigned  BSIM4v3ppscbe2Given   :1;    
    unsigned  BSIM4v3ppvagGiven   :1;    
    unsigned  BSIM4v3pdeltaGiven  :1;     
    unsigned  BSIM4v3pwrGiven   :1;
    unsigned  BSIM4v3pdwgGiven   :1;
    unsigned  BSIM4v3pdwbGiven   :1;
    unsigned  BSIM4v3pb0Given   :1;
    unsigned  BSIM4v3pb1Given   :1;
    unsigned  BSIM4v3palpha0Given   :1;
    unsigned  BSIM4v3palpha1Given   :1;
    unsigned  BSIM4v3pbeta0Given   :1;
    unsigned  BSIM4v3pvfbGiven   :1;
    unsigned  BSIM4v3pagidlGiven   :1;
    unsigned  BSIM4v3pbgidlGiven   :1;
    unsigned  BSIM4v3pcgidlGiven   :1;
    unsigned  BSIM4v3pegidlGiven   :1;
    unsigned  BSIM4v3paigcGiven   :1;
    unsigned  BSIM4v3pbigcGiven   :1;
    unsigned  BSIM4v3pcigcGiven   :1;
    unsigned  BSIM4v3paigsdGiven   :1;
    unsigned  BSIM4v3pbigsdGiven   :1;
    unsigned  BSIM4v3pcigsdGiven   :1;
    unsigned  BSIM4v3paigbaccGiven   :1;
    unsigned  BSIM4v3pbigbaccGiven   :1;
    unsigned  BSIM4v3pcigbaccGiven   :1;
    unsigned  BSIM4v3paigbinvGiven   :1;
    unsigned  BSIM4v3pbigbinvGiven   :1;
    unsigned  BSIM4v3pcigbinvGiven   :1;
    unsigned  BSIM4v3pnigcGiven   :1;
    unsigned  BSIM4v3pnigbinvGiven   :1;
    unsigned  BSIM4v3pnigbaccGiven   :1;
    unsigned  BSIM4v3pntoxGiven   :1;
    unsigned  BSIM4v3peigbinvGiven   :1;
    unsigned  BSIM4v3ppigcdGiven   :1;
    unsigned  BSIM4v3ppoxedgeGiven   :1;
    unsigned  BSIM4v3pxrcrg1Given   :1;
    unsigned  BSIM4v3pxrcrg2Given   :1;
    unsigned  BSIM4v3plambdaGiven    :1;
    unsigned  BSIM4v3pvtlGiven    :1;
    unsigned  BSIM4v3pxnGiven    :1;

    /* CV model */
    unsigned  BSIM4v3pcgslGiven   :1;
    unsigned  BSIM4v3pcgdlGiven   :1;
    unsigned  BSIM4v3pckappasGiven   :1;
    unsigned  BSIM4v3pckappadGiven   :1;
    unsigned  BSIM4v3pcfGiven   :1;
    unsigned  BSIM4v3pclcGiven   :1;
    unsigned  BSIM4v3pcleGiven   :1;
    unsigned  BSIM4v3pvfbcvGiven   :1;
    unsigned  BSIM4v3pnoffGiven   :1;
    unsigned  BSIM4v3pvoffcvGiven :1;
    unsigned  BSIM4v3pacdeGiven   :1;
    unsigned  BSIM4v3pmoinGiven   :1;

    unsigned  BSIM4v3useFringeGiven   :1;

    unsigned  BSIM4v3tnomGiven   :1;
    unsigned  BSIM4v3cgsoGiven   :1;
    unsigned  BSIM4v3cgdoGiven   :1;
    unsigned  BSIM4v3cgboGiven   :1;
    unsigned  BSIM4v3xpartGiven   :1;
    unsigned  BSIM4v3sheetResistanceGiven   :1;

    unsigned  BSIM4v3SjctSatCurDensityGiven   :1;
    unsigned  BSIM4v3SjctSidewallSatCurDensityGiven   :1;
    unsigned  BSIM4v3SjctGateSidewallSatCurDensityGiven   :1;
    unsigned  BSIM4v3SbulkJctPotentialGiven   :1;
    unsigned  BSIM4v3SbulkJctBotGradingCoeffGiven   :1;
    unsigned  BSIM4v3SsidewallJctPotentialGiven   :1;
    unsigned  BSIM4v3SGatesidewallJctPotentialGiven   :1;
    unsigned  BSIM4v3SbulkJctSideGradingCoeffGiven   :1;
    unsigned  BSIM4v3SunitAreaJctCapGiven   :1;
    unsigned  BSIM4v3SunitLengthSidewallJctCapGiven   :1;
    unsigned  BSIM4v3SbulkJctGateSideGradingCoeffGiven   :1;
    unsigned  BSIM4v3SunitLengthGateSidewallJctCapGiven   :1;
    unsigned  BSIM4v3SjctEmissionCoeffGiven :1;
    unsigned  BSIM4v3SjctTempExponentGiven	:1;

    unsigned  BSIM4v3DjctSatCurDensityGiven   :1;
    unsigned  BSIM4v3DjctSidewallSatCurDensityGiven   :1;
    unsigned  BSIM4v3DjctGateSidewallSatCurDensityGiven   :1;
    unsigned  BSIM4v3DbulkJctPotentialGiven   :1;
    unsigned  BSIM4v3DbulkJctBotGradingCoeffGiven   :1;
    unsigned  BSIM4v3DsidewallJctPotentialGiven   :1;
    unsigned  BSIM4v3DGatesidewallJctPotentialGiven   :1;
    unsigned  BSIM4v3DbulkJctSideGradingCoeffGiven   :1;
    unsigned  BSIM4v3DunitAreaJctCapGiven   :1;
    unsigned  BSIM4v3DunitLengthSidewallJctCapGiven   :1;
    unsigned  BSIM4v3DbulkJctGateSideGradingCoeffGiven   :1;
    unsigned  BSIM4v3DunitLengthGateSidewallJctCapGiven   :1;
    unsigned  BSIM4v3DjctEmissionCoeffGiven :1;
    unsigned  BSIM4v3DjctTempExponentGiven :1;

    unsigned  BSIM4v3oxideTrapDensityAGiven  :1;         
    unsigned  BSIM4v3oxideTrapDensityBGiven  :1;        
    unsigned  BSIM4v3oxideTrapDensityCGiven  :1;     
    unsigned  BSIM4v3emGiven  :1;     
    unsigned  BSIM4v3efGiven  :1;     
    unsigned  BSIM4v3afGiven  :1;     
    unsigned  BSIM4v3kfGiven  :1;     

    unsigned  BSIM4v3LintGiven   :1;
    unsigned  BSIM4v3LlGiven   :1;
    unsigned  BSIM4v3LlcGiven   :1;
    unsigned  BSIM4v3LlnGiven   :1;
    unsigned  BSIM4v3LwGiven   :1;
    unsigned  BSIM4v3LwcGiven   :1;
    unsigned  BSIM4v3LwnGiven   :1;
    unsigned  BSIM4v3LwlGiven   :1;
    unsigned  BSIM4v3LwlcGiven   :1;
    unsigned  BSIM4v3LminGiven   :1;
    unsigned  BSIM4v3LmaxGiven   :1;

    unsigned  BSIM4v3WintGiven   :1;
    unsigned  BSIM4v3WlGiven   :1;
    unsigned  BSIM4v3WlcGiven   :1;
    unsigned  BSIM4v3WlnGiven   :1;
    unsigned  BSIM4v3WwGiven   :1;
    unsigned  BSIM4v3WwcGiven   :1;
    unsigned  BSIM4v3WwnGiven   :1;
    unsigned  BSIM4v3WwlGiven   :1;
    unsigned  BSIM4v3WwlcGiven   :1;
    unsigned  BSIM4v3WminGiven   :1;
    unsigned  BSIM4v3WmaxGiven   :1;

    /* added for stress effect */
    unsigned  BSIM4v3sarefGiven   :1;
    unsigned  BSIM4v3sbrefGiven   :1;
    unsigned  BSIM4v3wlodGiven  :1;
    unsigned  BSIM4v3ku0Given   :1;
    unsigned  BSIM4v3kvsatGiven  :1;
    unsigned  BSIM4v3kvth0Given  :1;
    unsigned  BSIM4v3tku0Given   :1;
    unsigned  BSIM4v3llodku0Given   :1;
    unsigned  BSIM4v3wlodku0Given   :1;
    unsigned  BSIM4v3llodvthGiven   :1;
    unsigned  BSIM4v3wlodvthGiven   :1;
    unsigned  BSIM4v3lku0Given   :1;
    unsigned  BSIM4v3wku0Given   :1;
    unsigned  BSIM4v3pku0Given   :1;
    unsigned  BSIM4v3lkvth0Given   :1;
    unsigned  BSIM4v3wkvth0Given   :1;
    unsigned  BSIM4v3pkvth0Given   :1;
    unsigned  BSIM4v3stk2Given   :1;
    unsigned  BSIM4v3lodk2Given  :1;
    unsigned  BSIM4v3steta0Given :1;
    unsigned  BSIM4v3lodeta0Given :1;


} BSIM4v3model;


#ifndef NMOS
#define NMOS 1
#define PMOS -1
#endif /*NMOS*/


/* Instance parameters */
#define BSIM4v3_W                   1
#define BSIM4v3_L                   2
#define BSIM4v3_AS                  3
#define BSIM4v3_AD                  4
#define BSIM4v3_PS                  5
#define BSIM4v3_PD                  6
#define BSIM4v3_NRS                 7
#define BSIM4v3_NRD                 8
#define BSIM4v3_OFF                 9
#define BSIM4v3_IC                  10
#define BSIM4v3_IC_VDS              11
#define BSIM4v3_IC_VGS              12
#define BSIM4v3_IC_VBS              13
#define BSIM4v3_TRNQSMOD            14
#define BSIM4v3_RBODYMOD            15
#define BSIM4v3_RGATEMOD            16
#define BSIM4v3_GEOMOD              17
#define BSIM4v3_RGEOMOD             18
#define BSIM4v3_NF                  19
#define BSIM4v3_MIN                 20 
#define BSIM4v3_ACNQSMOD            22
#define BSIM4v3_RBDB                23
#define BSIM4v3_RBSB                24
#define BSIM4v3_RBPB                25
#define BSIM4v3_RBPS                26
#define BSIM4v3_RBPD                27
#define BSIM4v3_SA                  28
#define BSIM4v3_SB                  29
#define BSIM4v3_SD                  30
#define BSIM4v3_M                   31

/* Global parameters */
#define BSIM4v3_MOD_TEMPMOD         89
#define BSIM4v3_MOD_IGCMOD          90
#define BSIM4v3_MOD_IGBMOD          91
#define BSIM4v3_MOD_ACNQSMOD        92
#define BSIM4v3_MOD_FNOIMOD         93
#define BSIM4v3_MOD_RDSMOD          94
#define BSIM4v3_MOD_DIOMOD          96
#define BSIM4v3_MOD_PERMOD          97
#define BSIM4v3_MOD_GEOMOD          98
#define BSIM4v3_MOD_RGATEMOD        99
#define BSIM4v3_MOD_RBODYMOD        100
#define BSIM4v3_MOD_CAPMOD          101
#define BSIM4v3_MOD_TRNQSMOD        102
#define BSIM4v3_MOD_MOBMOD          103    
#define BSIM4v3_MOD_TNOIMOD         104    
#define BSIM4v3_MOD_TOXE            105
#define BSIM4v3_MOD_CDSC            106
#define BSIM4v3_MOD_CDSCB           107
#define BSIM4v3_MOD_CIT             108
#define BSIM4v3_MOD_NFACTOR         109
#define BSIM4v3_MOD_XJ              110
#define BSIM4v3_MOD_VSAT            111
#define BSIM4v3_MOD_AT              112
#define BSIM4v3_MOD_A0              113
#define BSIM4v3_MOD_A1              114
#define BSIM4v3_MOD_A2              115
#define BSIM4v3_MOD_KETA            116   
#define BSIM4v3_MOD_NSUB            117
#define BSIM4v3_MOD_NDEP            118
#define BSIM4v3_MOD_NGATE           120
#define BSIM4v3_MOD_GAMMA1          121
#define BSIM4v3_MOD_GAMMA2          122
#define BSIM4v3_MOD_VBX             123
#define BSIM4v3_MOD_BINUNIT         124    
#define BSIM4v3_MOD_VBM             125
#define BSIM4v3_MOD_XT              126
#define BSIM4v3_MOD_K1              129
#define BSIM4v3_MOD_KT1             130
#define BSIM4v3_MOD_KT1L            131
#define BSIM4v3_MOD_K2              132
#define BSIM4v3_MOD_KT2             133
#define BSIM4v3_MOD_K3              134
#define BSIM4v3_MOD_K3B             135
#define BSIM4v3_MOD_W0              136
#define BSIM4v3_MOD_LPE0            137
#define BSIM4v3_MOD_DVT0            138
#define BSIM4v3_MOD_DVT1            139
#define BSIM4v3_MOD_DVT2            140
#define BSIM4v3_MOD_DVT0W           141
#define BSIM4v3_MOD_DVT1W           142
#define BSIM4v3_MOD_DVT2W           143
#define BSIM4v3_MOD_DROUT           144
#define BSIM4v3_MOD_DSUB            145
#define BSIM4v3_MOD_VTH0            146
#define BSIM4v3_MOD_UA              147
#define BSIM4v3_MOD_UA1             148
#define BSIM4v3_MOD_UB              149
#define BSIM4v3_MOD_UB1             150
#define BSIM4v3_MOD_UC              151
#define BSIM4v3_MOD_UC1             152
#define BSIM4v3_MOD_U0              153
#define BSIM4v3_MOD_UTE             154
#define BSIM4v3_MOD_VOFF            155
#define BSIM4v3_MOD_DELTA           156
#define BSIM4v3_MOD_RDSW            157
#define BSIM4v3_MOD_PRT             158
#define BSIM4v3_MOD_LDD             159
#define BSIM4v3_MOD_ETA             160
#define BSIM4v3_MOD_ETA0            161
#define BSIM4v3_MOD_ETAB            162
#define BSIM4v3_MOD_PCLM            163
#define BSIM4v3_MOD_PDIBL1          164
#define BSIM4v3_MOD_PDIBL2          165
#define BSIM4v3_MOD_PSCBE1          166
#define BSIM4v3_MOD_PSCBE2          167
#define BSIM4v3_MOD_PVAG            168
#define BSIM4v3_MOD_WR              169
#define BSIM4v3_MOD_DWG             170
#define BSIM4v3_MOD_DWB             171
#define BSIM4v3_MOD_B0              172
#define BSIM4v3_MOD_B1              173
#define BSIM4v3_MOD_ALPHA0          174
#define BSIM4v3_MOD_BETA0           175
#define BSIM4v3_MOD_PDIBLB          178
#define BSIM4v3_MOD_PRWG            179
#define BSIM4v3_MOD_PRWB            180
#define BSIM4v3_MOD_CDSCD           181
#define BSIM4v3_MOD_AGS             182
#define BSIM4v3_MOD_FRINGE          184
#define BSIM4v3_MOD_CGSL            186
#define BSIM4v3_MOD_CGDL            187
#define BSIM4v3_MOD_CKAPPAS         188
#define BSIM4v3_MOD_CF              189
#define BSIM4v3_MOD_CLC             190
#define BSIM4v3_MOD_CLE             191
#define BSIM4v3_MOD_PARAMCHK        192
#define BSIM4v3_MOD_VERSION         193
#define BSIM4v3_MOD_VFBCV           194
#define BSIM4v3_MOD_ACDE            195
#define BSIM4v3_MOD_MOIN            196
#define BSIM4v3_MOD_NOFF            197
#define BSIM4v3_MOD_IJTHDFWD        198
#define BSIM4v3_MOD_ALPHA1          199
#define BSIM4v3_MOD_VFB             200
#define BSIM4v3_MOD_TOXM            201
#define BSIM4v3_MOD_TCJ             202
#define BSIM4v3_MOD_TCJSW           203
#define BSIM4v3_MOD_TCJSWG          204
#define BSIM4v3_MOD_TPB             205
#define BSIM4v3_MOD_TPBSW           206
#define BSIM4v3_MOD_TPBSWG          207
#define BSIM4v3_MOD_VOFFCV          208
#define BSIM4v3_MOD_GBMIN           209
#define BSIM4v3_MOD_RBDB            210
#define BSIM4v3_MOD_RBSB            211
#define BSIM4v3_MOD_RBPB            212
#define BSIM4v3_MOD_RBPS            213
#define BSIM4v3_MOD_RBPD            214
#define BSIM4v3_MOD_DMCG            215
#define BSIM4v3_MOD_DMCI            216
#define BSIM4v3_MOD_DMDG            217
#define BSIM4v3_MOD_XGW             218
#define BSIM4v3_MOD_XGL             219
#define BSIM4v3_MOD_RSHG            220
#define BSIM4v3_MOD_NGCON           221
#define BSIM4v3_MOD_AGIDL           222
#define BSIM4v3_MOD_BGIDL           223
#define BSIM4v3_MOD_EGIDL           224
#define BSIM4v3_MOD_IJTHSFWD        225
#define BSIM4v3_MOD_XJBVD           226
#define BSIM4v3_MOD_XJBVS           227
#define BSIM4v3_MOD_BVD             228
#define BSIM4v3_MOD_BVS             229
#define BSIM4v3_MOD_TOXP            230
#define BSIM4v3_MOD_DTOX            231
#define BSIM4v3_MOD_XRCRG1          232
#define BSIM4v3_MOD_XRCRG2          233
#define BSIM4v3_MOD_EU              234
#define BSIM4v3_MOD_IJTHSREV        235
#define BSIM4v3_MOD_IJTHDREV        236
#define BSIM4v3_MOD_MINV            237
#define BSIM4v3_MOD_VOFFL           238
#define BSIM4v3_MOD_PDITS           239
#define BSIM4v3_MOD_PDITSD          240
#define BSIM4v3_MOD_PDITSL          241
#define BSIM4v3_MOD_TNOIA           242
#define BSIM4v3_MOD_TNOIB           243
#define BSIM4v3_MOD_NTNOI           244
#define BSIM4v3_MOD_FPROUT          245
#define BSIM4v3_MOD_LPEB            246
#define BSIM4v3_MOD_DVTP0           247
#define BSIM4v3_MOD_DVTP1           248
#define BSIM4v3_MOD_CGIDL           249
#define BSIM4v3_MOD_PHIN            250
#define BSIM4v3_MOD_RDSWMIN         251
#define BSIM4v3_MOD_RSW             252
#define BSIM4v3_MOD_RDW             253
#define BSIM4v3_MOD_RDWMIN          254
#define BSIM4v3_MOD_RSWMIN          255
#define BSIM4v3_MOD_NSD             256
#define BSIM4v3_MOD_CKAPPAD         257
#define BSIM4v3_MOD_DMCGT           258
#define BSIM4v3_MOD_AIGC            259
#define BSIM4v3_MOD_BIGC            260
#define BSIM4v3_MOD_CIGC            261
#define BSIM4v3_MOD_AIGBACC         262
#define BSIM4v3_MOD_BIGBACC         263
#define BSIM4v3_MOD_CIGBACC         264            
#define BSIM4v3_MOD_AIGBINV         265
#define BSIM4v3_MOD_BIGBINV         266
#define BSIM4v3_MOD_CIGBINV         267
#define BSIM4v3_MOD_NIGC            268
#define BSIM4v3_MOD_NIGBACC         269
#define BSIM4v3_MOD_NIGBINV         270
#define BSIM4v3_MOD_NTOX            271
#define BSIM4v3_MOD_TOXREF          272
#define BSIM4v3_MOD_EIGBINV         273
#define BSIM4v3_MOD_PIGCD           274
#define BSIM4v3_MOD_POXEDGE         275
#define BSIM4v3_MOD_EPSROX          276
#define BSIM4v3_MOD_AIGSD           277
#define BSIM4v3_MOD_BIGSD           278
#define BSIM4v3_MOD_CIGSD           279
#define BSIM4v3_MOD_JSWGS           280
#define BSIM4v3_MOD_JSWGD           281
#define BSIM4v3_MOD_LAMBDA          282
#define BSIM4v3_MOD_VTL             283
#define BSIM4v3_MOD_LC              284
#define BSIM4v3_MOD_XN              285
#define BSIM4v3_MOD_RNOIA           286
#define BSIM4v3_MOD_RNOIB           287


/* Length dependence */
#define BSIM4v3_MOD_LCDSC            301
#define BSIM4v3_MOD_LCDSCB           302
#define BSIM4v3_MOD_LCIT             303
#define BSIM4v3_MOD_LNFACTOR         304
#define BSIM4v3_MOD_LXJ              305
#define BSIM4v3_MOD_LVSAT            306
#define BSIM4v3_MOD_LAT              307
#define BSIM4v3_MOD_LA0              308
#define BSIM4v3_MOD_LA1              309
#define BSIM4v3_MOD_LA2              310
#define BSIM4v3_MOD_LKETA            311
#define BSIM4v3_MOD_LNSUB            312
#define BSIM4v3_MOD_LNDEP            313
#define BSIM4v3_MOD_LNGATE           315
#define BSIM4v3_MOD_LGAMMA1          316
#define BSIM4v3_MOD_LGAMMA2          317
#define BSIM4v3_MOD_LVBX             318
#define BSIM4v3_MOD_LVBM             320
#define BSIM4v3_MOD_LXT              322
#define BSIM4v3_MOD_LK1              325
#define BSIM4v3_MOD_LKT1             326
#define BSIM4v3_MOD_LKT1L            327
#define BSIM4v3_MOD_LK2              328
#define BSIM4v3_MOD_LKT2             329
#define BSIM4v3_MOD_LK3              330
#define BSIM4v3_MOD_LK3B             331
#define BSIM4v3_MOD_LW0              332
#define BSIM4v3_MOD_LLPE0            333
#define BSIM4v3_MOD_LDVT0            334
#define BSIM4v3_MOD_LDVT1            335
#define BSIM4v3_MOD_LDVT2            336
#define BSIM4v3_MOD_LDVT0W           337
#define BSIM4v3_MOD_LDVT1W           338
#define BSIM4v3_MOD_LDVT2W           339
#define BSIM4v3_MOD_LDROUT           340
#define BSIM4v3_MOD_LDSUB            341
#define BSIM4v3_MOD_LVTH0            342
#define BSIM4v3_MOD_LUA              343
#define BSIM4v3_MOD_LUA1             344
#define BSIM4v3_MOD_LUB              345
#define BSIM4v3_MOD_LUB1             346
#define BSIM4v3_MOD_LUC              347
#define BSIM4v3_MOD_LUC1             348
#define BSIM4v3_MOD_LU0              349
#define BSIM4v3_MOD_LUTE             350
#define BSIM4v3_MOD_LVOFF            351
#define BSIM4v3_MOD_LDELTA           352
#define BSIM4v3_MOD_LRDSW            353
#define BSIM4v3_MOD_LPRT             354
#define BSIM4v3_MOD_LLDD             355
#define BSIM4v3_MOD_LETA             356
#define BSIM4v3_MOD_LETA0            357
#define BSIM4v3_MOD_LETAB            358
#define BSIM4v3_MOD_LPCLM            359
#define BSIM4v3_MOD_LPDIBL1          360
#define BSIM4v3_MOD_LPDIBL2          361
#define BSIM4v3_MOD_LPSCBE1          362
#define BSIM4v3_MOD_LPSCBE2          363
#define BSIM4v3_MOD_LPVAG            364
#define BSIM4v3_MOD_LWR              365
#define BSIM4v3_MOD_LDWG             366
#define BSIM4v3_MOD_LDWB             367
#define BSIM4v3_MOD_LB0              368
#define BSIM4v3_MOD_LB1              369
#define BSIM4v3_MOD_LALPHA0          370
#define BSIM4v3_MOD_LBETA0           371
#define BSIM4v3_MOD_LPDIBLB          374
#define BSIM4v3_MOD_LPRWG            375
#define BSIM4v3_MOD_LPRWB            376
#define BSIM4v3_MOD_LCDSCD           377
#define BSIM4v3_MOD_LAGS             378

#define BSIM4v3_MOD_LFRINGE          381
#define BSIM4v3_MOD_LCGSL            383
#define BSIM4v3_MOD_LCGDL            384
#define BSIM4v3_MOD_LCKAPPAS         385
#define BSIM4v3_MOD_LCF              386
#define BSIM4v3_MOD_LCLC             387
#define BSIM4v3_MOD_LCLE             388
#define BSIM4v3_MOD_LVFBCV           389
#define BSIM4v3_MOD_LACDE            390
#define BSIM4v3_MOD_LMOIN            391
#define BSIM4v3_MOD_LNOFF            392
#define BSIM4v3_MOD_LALPHA1          394
#define BSIM4v3_MOD_LVFB             395
#define BSIM4v3_MOD_LVOFFCV          396
#define BSIM4v3_MOD_LAGIDL           397
#define BSIM4v3_MOD_LBGIDL           398
#define BSIM4v3_MOD_LEGIDL           399
#define BSIM4v3_MOD_LXRCRG1          400
#define BSIM4v3_MOD_LXRCRG2          401
#define BSIM4v3_MOD_LEU              402
#define BSIM4v3_MOD_LMINV            403
#define BSIM4v3_MOD_LPDITS           404
#define BSIM4v3_MOD_LPDITSD          405
#define BSIM4v3_MOD_LFPROUT          406
#define BSIM4v3_MOD_LLPEB            407
#define BSIM4v3_MOD_LDVTP0           408
#define BSIM4v3_MOD_LDVTP1           409
#define BSIM4v3_MOD_LCGIDL           410
#define BSIM4v3_MOD_LPHIN            411
#define BSIM4v3_MOD_LRSW             412
#define BSIM4v3_MOD_LRDW             413
#define BSIM4v3_MOD_LNSD             414
#define BSIM4v3_MOD_LCKAPPAD         415
#define BSIM4v3_MOD_LAIGC            416
#define BSIM4v3_MOD_LBIGC            417
#define BSIM4v3_MOD_LCIGC            418
#define BSIM4v3_MOD_LAIGBACC         419            
#define BSIM4v3_MOD_LBIGBACC         420
#define BSIM4v3_MOD_LCIGBACC         421
#define BSIM4v3_MOD_LAIGBINV         422
#define BSIM4v3_MOD_LBIGBINV         423
#define BSIM4v3_MOD_LCIGBINV         424
#define BSIM4v3_MOD_LNIGC            425
#define BSIM4v3_MOD_LNIGBACC         426
#define BSIM4v3_MOD_LNIGBINV         427
#define BSIM4v3_MOD_LNTOX            428
#define BSIM4v3_MOD_LEIGBINV         429
#define BSIM4v3_MOD_LPIGCD           430
#define BSIM4v3_MOD_LPOXEDGE         431
#define BSIM4v3_MOD_LAIGSD           432
#define BSIM4v3_MOD_LBIGSD           433
#define BSIM4v3_MOD_LCIGSD           434

#define BSIM4v3_MOD_LLAMBDA          435
#define BSIM4v3_MOD_LVTL             436
#define BSIM4v3_MOD_LXN              437

/* Width dependence */
#define BSIM4v3_MOD_WCDSC            481
#define BSIM4v3_MOD_WCDSCB           482
#define BSIM4v3_MOD_WCIT             483
#define BSIM4v3_MOD_WNFACTOR         484
#define BSIM4v3_MOD_WXJ              485
#define BSIM4v3_MOD_WVSAT            486
#define BSIM4v3_MOD_WAT              487
#define BSIM4v3_MOD_WA0              488
#define BSIM4v3_MOD_WA1              489
#define BSIM4v3_MOD_WA2              490
#define BSIM4v3_MOD_WKETA            491
#define BSIM4v3_MOD_WNSUB            492
#define BSIM4v3_MOD_WNDEP            493
#define BSIM4v3_MOD_WNGATE           495
#define BSIM4v3_MOD_WGAMMA1          496
#define BSIM4v3_MOD_WGAMMA2          497
#define BSIM4v3_MOD_WVBX             498
#define BSIM4v3_MOD_WVBM             500
#define BSIM4v3_MOD_WXT              502
#define BSIM4v3_MOD_WK1              505
#define BSIM4v3_MOD_WKT1             506
#define BSIM4v3_MOD_WKT1L            507
#define BSIM4v3_MOD_WK2              508
#define BSIM4v3_MOD_WKT2             509
#define BSIM4v3_MOD_WK3              510
#define BSIM4v3_MOD_WK3B             511
#define BSIM4v3_MOD_WW0              512
#define BSIM4v3_MOD_WLPE0            513
#define BSIM4v3_MOD_WDVT0            514
#define BSIM4v3_MOD_WDVT1            515
#define BSIM4v3_MOD_WDVT2            516
#define BSIM4v3_MOD_WDVT0W           517
#define BSIM4v3_MOD_WDVT1W           518
#define BSIM4v3_MOD_WDVT2W           519
#define BSIM4v3_MOD_WDROUT           520
#define BSIM4v3_MOD_WDSUB            521
#define BSIM4v3_MOD_WVTH0            522
#define BSIM4v3_MOD_WUA              523
#define BSIM4v3_MOD_WUA1             524
#define BSIM4v3_MOD_WUB              525
#define BSIM4v3_MOD_WUB1             526
#define BSIM4v3_MOD_WUC              527
#define BSIM4v3_MOD_WUC1             528
#define BSIM4v3_MOD_WU0              529
#define BSIM4v3_MOD_WUTE             530
#define BSIM4v3_MOD_WVOFF            531
#define BSIM4v3_MOD_WDELTA           532
#define BSIM4v3_MOD_WRDSW            533
#define BSIM4v3_MOD_WPRT             534
#define BSIM4v3_MOD_WLDD             535
#define BSIM4v3_MOD_WETA             536
#define BSIM4v3_MOD_WETA0            537
#define BSIM4v3_MOD_WETAB            538
#define BSIM4v3_MOD_WPCLM            539
#define BSIM4v3_MOD_WPDIBL1          540
#define BSIM4v3_MOD_WPDIBL2          541
#define BSIM4v3_MOD_WPSCBE1          542
#define BSIM4v3_MOD_WPSCBE2          543
#define BSIM4v3_MOD_WPVAG            544
#define BSIM4v3_MOD_WWR              545
#define BSIM4v3_MOD_WDWG             546
#define BSIM4v3_MOD_WDWB             547
#define BSIM4v3_MOD_WB0              548
#define BSIM4v3_MOD_WB1              549
#define BSIM4v3_MOD_WALPHA0          550
#define BSIM4v3_MOD_WBETA0           551
#define BSIM4v3_MOD_WPDIBLB          554
#define BSIM4v3_MOD_WPRWG            555
#define BSIM4v3_MOD_WPRWB            556
#define BSIM4v3_MOD_WCDSCD           557
#define BSIM4v3_MOD_WAGS             558

#define BSIM4v3_MOD_WFRINGE          561
#define BSIM4v3_MOD_WCGSL            563
#define BSIM4v3_MOD_WCGDL            564
#define BSIM4v3_MOD_WCKAPPAS         565
#define BSIM4v3_MOD_WCF              566
#define BSIM4v3_MOD_WCLC             567
#define BSIM4v3_MOD_WCLE             568
#define BSIM4v3_MOD_WVFBCV           569
#define BSIM4v3_MOD_WACDE            570
#define BSIM4v3_MOD_WMOIN            571
#define BSIM4v3_MOD_WNOFF            572
#define BSIM4v3_MOD_WALPHA1          574
#define BSIM4v3_MOD_WVFB             575
#define BSIM4v3_MOD_WVOFFCV          576
#define BSIM4v3_MOD_WAGIDL           577
#define BSIM4v3_MOD_WBGIDL           578
#define BSIM4v3_MOD_WEGIDL           579
#define BSIM4v3_MOD_WXRCRG1          580
#define BSIM4v3_MOD_WXRCRG2          581
#define BSIM4v3_MOD_WEU              582
#define BSIM4v3_MOD_WMINV            583
#define BSIM4v3_MOD_WPDITS           584
#define BSIM4v3_MOD_WPDITSD          585
#define BSIM4v3_MOD_WFPROUT          586
#define BSIM4v3_MOD_WLPEB            587
#define BSIM4v3_MOD_WDVTP0           588
#define BSIM4v3_MOD_WDVTP1           589
#define BSIM4v3_MOD_WCGIDL           590
#define BSIM4v3_MOD_WPHIN            591
#define BSIM4v3_MOD_WRSW             592
#define BSIM4v3_MOD_WRDW             593
#define BSIM4v3_MOD_WNSD             594
#define BSIM4v3_MOD_WCKAPPAD         595
#define BSIM4v3_MOD_WAIGC            596
#define BSIM4v3_MOD_WBIGC            597
#define BSIM4v3_MOD_WCIGC            598
#define BSIM4v3_MOD_WAIGBACC         599            
#define BSIM4v3_MOD_WBIGBACC         600
#define BSIM4v3_MOD_WCIGBACC         601
#define BSIM4v3_MOD_WAIGBINV         602
#define BSIM4v3_MOD_WBIGBINV         603
#define BSIM4v3_MOD_WCIGBINV         604
#define BSIM4v3_MOD_WNIGC            605
#define BSIM4v3_MOD_WNIGBACC         606
#define BSIM4v3_MOD_WNIGBINV         607
#define BSIM4v3_MOD_WNTOX            608
#define BSIM4v3_MOD_WEIGBINV         609
#define BSIM4v3_MOD_WPIGCD           610
#define BSIM4v3_MOD_WPOXEDGE         611
#define BSIM4v3_MOD_WAIGSD           612
#define BSIM4v3_MOD_WBIGSD           613
#define BSIM4v3_MOD_WCIGSD           614
#define BSIM4v3_MOD_WLAMBDA          615
#define BSIM4v3_MOD_WVTL             616
#define BSIM4v3_MOD_WXN              617


/* Cross-term dependence */
#define BSIM4v3_MOD_PCDSC            661
#define BSIM4v3_MOD_PCDSCB           662
#define BSIM4v3_MOD_PCIT             663
#define BSIM4v3_MOD_PNFACTOR         664
#define BSIM4v3_MOD_PXJ              665
#define BSIM4v3_MOD_PVSAT            666
#define BSIM4v3_MOD_PAT              667
#define BSIM4v3_MOD_PA0              668
#define BSIM4v3_MOD_PA1              669
#define BSIM4v3_MOD_PA2              670
#define BSIM4v3_MOD_PKETA            671
#define BSIM4v3_MOD_PNSUB            672
#define BSIM4v3_MOD_PNDEP            673
#define BSIM4v3_MOD_PNGATE           675
#define BSIM4v3_MOD_PGAMMA1          676
#define BSIM4v3_MOD_PGAMMA2          677
#define BSIM4v3_MOD_PVBX             678

#define BSIM4v3_MOD_PVBM             680

#define BSIM4v3_MOD_PXT              682
#define BSIM4v3_MOD_PK1              685
#define BSIM4v3_MOD_PKT1             686
#define BSIM4v3_MOD_PKT1L            687
#define BSIM4v3_MOD_PK2              688
#define BSIM4v3_MOD_PKT2             689
#define BSIM4v3_MOD_PK3              690
#define BSIM4v3_MOD_PK3B             691
#define BSIM4v3_MOD_PW0              692
#define BSIM4v3_MOD_PLPE0            693

#define BSIM4v3_MOD_PDVT0            694
#define BSIM4v3_MOD_PDVT1            695
#define BSIM4v3_MOD_PDVT2            696

#define BSIM4v3_MOD_PDVT0W           697
#define BSIM4v3_MOD_PDVT1W           698
#define BSIM4v3_MOD_PDVT2W           699

#define BSIM4v3_MOD_PDROUT           700
#define BSIM4v3_MOD_PDSUB            701
#define BSIM4v3_MOD_PVTH0            702
#define BSIM4v3_MOD_PUA              703
#define BSIM4v3_MOD_PUA1             704
#define BSIM4v3_MOD_PUB              705
#define BSIM4v3_MOD_PUB1             706
#define BSIM4v3_MOD_PUC              707
#define BSIM4v3_MOD_PUC1             708
#define BSIM4v3_MOD_PU0              709
#define BSIM4v3_MOD_PUTE             710
#define BSIM4v3_MOD_PVOFF            711
#define BSIM4v3_MOD_PDELTA           712
#define BSIM4v3_MOD_PRDSW            713
#define BSIM4v3_MOD_PPRT             714
#define BSIM4v3_MOD_PLDD             715
#define BSIM4v3_MOD_PETA             716
#define BSIM4v3_MOD_PETA0            717
#define BSIM4v3_MOD_PETAB            718
#define BSIM4v3_MOD_PPCLM            719
#define BSIM4v3_MOD_PPDIBL1          720
#define BSIM4v3_MOD_PPDIBL2          721
#define BSIM4v3_MOD_PPSCBE1          722
#define BSIM4v3_MOD_PPSCBE2          723
#define BSIM4v3_MOD_PPVAG            724
#define BSIM4v3_MOD_PWR              725
#define BSIM4v3_MOD_PDWG             726
#define BSIM4v3_MOD_PDWB             727
#define BSIM4v3_MOD_PB0              728
#define BSIM4v3_MOD_PB1              729
#define BSIM4v3_MOD_PALPHA0          730
#define BSIM4v3_MOD_PBETA0           731
#define BSIM4v3_MOD_PPDIBLB          734

#define BSIM4v3_MOD_PPRWG            735
#define BSIM4v3_MOD_PPRWB            736

#define BSIM4v3_MOD_PCDSCD           737
#define BSIM4v3_MOD_PAGS             738

#define BSIM4v3_MOD_PFRINGE          741
#define BSIM4v3_MOD_PCGSL            743
#define BSIM4v3_MOD_PCGDL            744
#define BSIM4v3_MOD_PCKAPPAS         745
#define BSIM4v3_MOD_PCF              746
#define BSIM4v3_MOD_PCLC             747
#define BSIM4v3_MOD_PCLE             748
#define BSIM4v3_MOD_PVFBCV           749
#define BSIM4v3_MOD_PACDE            750
#define BSIM4v3_MOD_PMOIN            751
#define BSIM4v3_MOD_PNOFF            752
#define BSIM4v3_MOD_PALPHA1          754
#define BSIM4v3_MOD_PVFB             755
#define BSIM4v3_MOD_PVOFFCV          756
#define BSIM4v3_MOD_PAGIDL           757
#define BSIM4v3_MOD_PBGIDL           758
#define BSIM4v3_MOD_PEGIDL           759
#define BSIM4v3_MOD_PXRCRG1          760
#define BSIM4v3_MOD_PXRCRG2          761
#define BSIM4v3_MOD_PEU              762
#define BSIM4v3_MOD_PMINV            763
#define BSIM4v3_MOD_PPDITS           764
#define BSIM4v3_MOD_PPDITSD          765
#define BSIM4v3_MOD_PFPROUT          766
#define BSIM4v3_MOD_PLPEB            767
#define BSIM4v3_MOD_PDVTP0           768
#define BSIM4v3_MOD_PDVTP1           769
#define BSIM4v3_MOD_PCGIDL           770
#define BSIM4v3_MOD_PPHIN            771
#define BSIM4v3_MOD_PRSW             772
#define BSIM4v3_MOD_PRDW             773
#define BSIM4v3_MOD_PNSD             774
#define BSIM4v3_MOD_PCKAPPAD         775
#define BSIM4v3_MOD_PAIGC            776
#define BSIM4v3_MOD_PBIGC            777
#define BSIM4v3_MOD_PCIGC            778
#define BSIM4v3_MOD_PAIGBACC         779            
#define BSIM4v3_MOD_PBIGBACC         780
#define BSIM4v3_MOD_PCIGBACC         781
#define BSIM4v3_MOD_PAIGBINV         782
#define BSIM4v3_MOD_PBIGBINV         783
#define BSIM4v3_MOD_PCIGBINV         784
#define BSIM4v3_MOD_PNIGC            785
#define BSIM4v3_MOD_PNIGBACC         786
#define BSIM4v3_MOD_PNIGBINV         787
#define BSIM4v3_MOD_PNTOX            788
#define BSIM4v3_MOD_PEIGBINV         789
#define BSIM4v3_MOD_PPIGCD           790
#define BSIM4v3_MOD_PPOXEDGE         791
#define BSIM4v3_MOD_PAIGSD           792
#define BSIM4v3_MOD_PBIGSD           793
#define BSIM4v3_MOD_PCIGSD           794

#define BSIM4v3_MOD_SAREF            795
#define BSIM4v3_MOD_SBREF            796
#define BSIM4v3_MOD_KU0              797
#define BSIM4v3_MOD_KVSAT            798
#define BSIM4v3_MOD_TKU0             799
#define BSIM4v3_MOD_LLODKU0          800
#define BSIM4v3_MOD_WLODKU0          801
#define BSIM4v3_MOD_LLODVTH          802
#define BSIM4v3_MOD_WLODVTH          803
#define BSIM4v3_MOD_LKU0             804
#define BSIM4v3_MOD_WKU0             805
#define BSIM4v3_MOD_PKU0             806
#define BSIM4v3_MOD_KVTH0            807
#define BSIM4v3_MOD_LKVTH0           808
#define BSIM4v3_MOD_WKVTH0           809
#define BSIM4v3_MOD_PKVTH0           810
#define BSIM4v3_MOD_WLOD		   811
#define BSIM4v3_MOD_STK2		   812
#define BSIM4v3_MOD_LODK2		   813
#define BSIM4v3_MOD_STETA0	   814
#define BSIM4v3_MOD_LODETA0	   815

#define BSIM4v3_MOD_PLAMBDA          825
#define BSIM4v3_MOD_PVTL             826
#define BSIM4v3_MOD_PXN              827

#define BSIM4v3_MOD_TNOM             831
#define BSIM4v3_MOD_CGSO             832
#define BSIM4v3_MOD_CGDO             833
#define BSIM4v3_MOD_CGBO             834
#define BSIM4v3_MOD_XPART            835
#define BSIM4v3_MOD_RSH              836
#define BSIM4v3_MOD_JSS              837
#define BSIM4v3_MOD_PBS              838
#define BSIM4v3_MOD_MJS              839
#define BSIM4v3_MOD_PBSWS            840
#define BSIM4v3_MOD_MJSWS            841
#define BSIM4v3_MOD_CJS              842
#define BSIM4v3_MOD_CJSWS            843
#define BSIM4v3_MOD_NMOS             844
#define BSIM4v3_MOD_PMOS             845
#define BSIM4v3_MOD_NOIA             846
#define BSIM4v3_MOD_NOIB             847
#define BSIM4v3_MOD_NOIC             848
#define BSIM4v3_MOD_LINT             849
#define BSIM4v3_MOD_LL               850
#define BSIM4v3_MOD_LLN              851
#define BSIM4v3_MOD_LW               852
#define BSIM4v3_MOD_LWN              853
#define BSIM4v3_MOD_LWL              854
#define BSIM4v3_MOD_LMIN             855
#define BSIM4v3_MOD_LMAX             856
#define BSIM4v3_MOD_WINT             857
#define BSIM4v3_MOD_WL               858
#define BSIM4v3_MOD_WLN              859
#define BSIM4v3_MOD_WW               860
#define BSIM4v3_MOD_WWN              861
#define BSIM4v3_MOD_WWL              862
#define BSIM4v3_MOD_WMIN             863
#define BSIM4v3_MOD_WMAX             864
#define BSIM4v3_MOD_DWC              865
#define BSIM4v3_MOD_DLC              866
#define BSIM4v3_MOD_XL               867
#define BSIM4v3_MOD_XW               868
#define BSIM4v3_MOD_EM               869
#define BSIM4v3_MOD_EF               870
#define BSIM4v3_MOD_AF               871
#define BSIM4v3_MOD_KF               872
#define BSIM4v3_MOD_NJS              873
#define BSIM4v3_MOD_XTIS             874
#define BSIM4v3_MOD_PBSWGS           875
#define BSIM4v3_MOD_MJSWGS           876
#define BSIM4v3_MOD_CJSWGS           877
#define BSIM4v3_MOD_JSWS             878
#define BSIM4v3_MOD_LLC              879
#define BSIM4v3_MOD_LWC              880
#define BSIM4v3_MOD_LWLC             881
#define BSIM4v3_MOD_WLC              882
#define BSIM4v3_MOD_WWC              883
#define BSIM4v3_MOD_WWLC             884
#define BSIM4v3_MOD_DWJ              885
#define BSIM4v3_MOD_JSD              886
#define BSIM4v3_MOD_PBD              887
#define BSIM4v3_MOD_MJD              888
#define BSIM4v3_MOD_PBSWD            889
#define BSIM4v3_MOD_MJSWD            890
#define BSIM4v3_MOD_CJD              891
#define BSIM4v3_MOD_CJSWD            892
#define BSIM4v3_MOD_NJD              893
#define BSIM4v3_MOD_XTID             894
#define BSIM4v3_MOD_PBSWGD           895
#define BSIM4v3_MOD_MJSWGD           896
#define BSIM4v3_MOD_CJSWGD           897
#define BSIM4v3_MOD_JSWD             898
#define BSIM4v3_MOD_DLCIG            899

/* device questions */
#define BSIM4v3_DNODE                945
#define BSIM4v3_GNODEEXT             946
#define BSIM4v3_SNODE                947
#define BSIM4v3_BNODE                948
#define BSIM4v3_DNODEPRIME           949
#define BSIM4v3_GNODEPRIME           950
#define BSIM4v3_GNODEMIDE            951
#define BSIM4v3_GNODEMID             952
#define BSIM4v3_SNODEPRIME           953
#define BSIM4v3_BNODEPRIME           954
#define BSIM4v3_DBNODE               955
#define BSIM4v3_SBNODE               956
#define BSIM4v3_VBD                  957
#define BSIM4v3_VBS                  958
#define BSIM4v3_VGS                  959
#define BSIM4v3_VDS                  960
#define BSIM4v3_CD                   961
#define BSIM4v3_CBS                  962
#define BSIM4v3_CBD                  963
#define BSIM4v3_GM                   964
#define BSIM4v3_GDS                  965
#define BSIM4v3_GMBS                 966
#define BSIM4v3_GBD                  967
#define BSIM4v3_GBS                  968
#define BSIM4v3_QB                   969
#define BSIM4v3_CQB                  970
#define BSIM4v3_QG                   971
#define BSIM4v3_CQG                  972
#define BSIM4v3_QD                   973
#define BSIM4v3_CQD                  974
#define BSIM4v3_CGGB                 975
#define BSIM4v3_CGDB                 976
#define BSIM4v3_CGSB                 977
#define BSIM4v3_CBGB                 978
#define BSIM4v3_CAPBD                979
#define BSIM4v3_CQBD                 980
#define BSIM4v3_CAPBS                981
#define BSIM4v3_CQBS                 982
#define BSIM4v3_CDGB                 983
#define BSIM4v3_CDDB                 984
#define BSIM4v3_CDSB                 985
#define BSIM4v3_VON                  986
#define BSIM4v3_VDSAT                987
#define BSIM4v3_QBS                  988
#define BSIM4v3_QBD                  989
#define BSIM4v3_SOURCECONDUCT        990
#define BSIM4v3_DRAINCONDUCT         991
#define BSIM4v3_CBDB                 992
#define BSIM4v3_CBSB                 993
#define BSIM4v3_CSUB		   994
#define BSIM4v3_QINV		   995
#define BSIM4v3_IGIDL		   996
#define BSIM4v3_CSGB                 997
#define BSIM4v3_CSDB                 998
#define BSIM4v3_CSSB                 999
#define BSIM4v3_CGBB                 1000
#define BSIM4v3_CDBB                 1001
#define BSIM4v3_CSBB                 1002
#define BSIM4v3_CBBB                 1003
#define BSIM4v3_QS                   1004
#define BSIM4v3_IGISL		   1005
#define BSIM4v3_IGS		   1006
#define BSIM4v3_IGD		   1007
#define BSIM4v3_IGB		   1008
#define BSIM4v3_IGCS		   1009
#define BSIM4v3_IGCD		   1010


#include "bsim4v3ext.h"

extern void BSIM4v3evaluate(double,double,double,BSIM4v3instance*,BSIM4v3model*,
        double*,double*,double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, CKTcircuit*);
extern int BSIM4v3debug(BSIM4v3model*, BSIM4v3instance*, CKTcircuit*, int);
extern int BSIM4v3checkModel(BSIM4v3model*, BSIM4v3instance*, CKTcircuit*);

#endif /*BSIM4v3*/
