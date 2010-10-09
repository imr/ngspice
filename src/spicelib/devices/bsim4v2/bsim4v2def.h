/**********
Copyright 2001 Regents of the University of California.  All rights reserved.
Author: 2000 Weidong Liu.
Modified by Xuemei Xi October 2001
File: bsim4def.h
**********/

#ifndef BSIM4V2
#define BSIM4V2

#include "ifsim.h"
#include "gendefs.h"
#include "cktdefs.h"
#include "complex.h"
#include "noisedef.h"         

typedef struct sBSIM4v2instance
{
    struct sBSIM4v2model *BSIM4v2modPtr;
    struct sBSIM4v2instance *BSIM4v2nextInstance;
    IFuid BSIM4v2name;
    int BSIM4v2owner;      /* Number of owner process */
    int BSIM4v2states;     /* index into state table for this device */
    int BSIM4v2dNode;
    int BSIM4v2gNodeExt;
    int BSIM4v2sNode;
    int BSIM4v2bNode;
    int BSIM4v2dNodePrime;
    int BSIM4v2gNodePrime;
    int BSIM4v2gNodeMid;
    int BSIM4v2sNodePrime;
    int BSIM4v2bNodePrime;
    int BSIM4v2dbNode;
    int BSIM4v2sbNode;
    int BSIM4v2qNode;

    double BSIM4v2ueff;
    double BSIM4v2thetavth; 
    double BSIM4v2von;
    double BSIM4v2vdsat;
    double BSIM4v2cgdo;
    double BSIM4v2qgdo;
    double BSIM4v2cgso;
    double BSIM4v2qgso;
    double BSIM4v2grbsb;
    double BSIM4v2grbdb;
    double BSIM4v2grbpb;
    double BSIM4v2grbps;
    double BSIM4v2grbpd;

    double BSIM4v2vjsmFwd;
    double BSIM4v2vjsmRev;
    double BSIM4v2vjdmFwd;
    double BSIM4v2vjdmRev;
    double BSIM4v2XExpBVS;
    double BSIM4v2XExpBVD;
    double BSIM4v2SslpFwd;
    double BSIM4v2SslpRev;
    double BSIM4v2DslpFwd;
    double BSIM4v2DslpRev;
    double BSIM4v2IVjsmFwd;
    double BSIM4v2IVjsmRev;
    double BSIM4v2IVjdmFwd;
    double BSIM4v2IVjdmRev;

    double BSIM4v2grgeltd;
    double BSIM4v2Pseff;
    double BSIM4v2Pdeff;
    double BSIM4v2Aseff;
    double BSIM4v2Adeff;

    double BSIM4v2l;
    double BSIM4v2w;
    double BSIM4v2drainArea;
    double BSIM4v2sourceArea;
    double BSIM4v2drainSquares;
    double BSIM4v2sourceSquares;
    double BSIM4v2drainPerimeter;
    double BSIM4v2sourcePerimeter;
    double BSIM4v2sourceConductance;
    double BSIM4v2drainConductance;
    double BSIM4v2rbdb;
    double BSIM4v2rbsb;
    double BSIM4v2rbpb;
    double BSIM4v2rbps;
    double BSIM4v2rbpd;

    double BSIM4v2icVDS;
    double BSIM4v2icVGS;
    double BSIM4v2icVBS;
    double BSIM4v2nf;
    double BSIM4v2m;  
    int BSIM4v2off;
    int BSIM4v2mode;
    int BSIM4v2trnqsMod;
    int BSIM4v2acnqsMod;
    int BSIM4v2rbodyMod;
    int BSIM4v2rgateMod;
    int BSIM4v2geoMod;
    int BSIM4v2rgeoMod;
    int BSIM4v2min;

    /* OP point */
    double BSIM4v2Vgsteff;
    double BSIM4v2vgs_eff;
    double BSIM4v2vgd_eff;
    double BSIM4v2dvgs_eff_dvg;
    double BSIM4v2dvgd_eff_dvg;
    double BSIM4v2Vdseff;
    double BSIM4v2nstar;
    double BSIM4v2Abulk;
    double BSIM4v2EsatL;
    double BSIM4v2AbovVgst2Vtm;
    double BSIM4v2qinv;
    double BSIM4v2cd;
    double BSIM4v2cbs;
    double BSIM4v2cbd;
    double BSIM4v2csub;
    double BSIM4v2Igidl;
    double BSIM4v2Igisl;
    double BSIM4v2gm;
    double BSIM4v2gds;
    double BSIM4v2gmbs;
    double BSIM4v2gbd;
    double BSIM4v2gbs;

    double BSIM4v2gbbs;
    double BSIM4v2gbgs;
    double BSIM4v2gbds;
    double BSIM4v2ggidld;
    double BSIM4v2ggidlg;
    double BSIM4v2ggidls;
    double BSIM4v2ggidlb;
    double BSIM4v2ggisld;
    double BSIM4v2ggislg;
    double BSIM4v2ggisls;
    double BSIM4v2ggislb;

    double BSIM4v2Igcs;
    double BSIM4v2gIgcsg;
    double BSIM4v2gIgcsd;
    double BSIM4v2gIgcss;
    double BSIM4v2gIgcsb;
    double BSIM4v2Igcd;
    double BSIM4v2gIgcdg;
    double BSIM4v2gIgcdd;
    double BSIM4v2gIgcds;
    double BSIM4v2gIgcdb;

    double BSIM4v2Igs;
    double BSIM4v2gIgsg;
    double BSIM4v2gIgss;
    double BSIM4v2Igd;
    double BSIM4v2gIgdg;
    double BSIM4v2gIgdd;

    double BSIM4v2Igb;
    double BSIM4v2gIgbg;
    double BSIM4v2gIgbd;
    double BSIM4v2gIgbs;
    double BSIM4v2gIgbb;

    double BSIM4v2grdsw;
    double BSIM4v2IdovVds;
    double BSIM4v2gcrg;
    double BSIM4v2gcrgd;
    double BSIM4v2gcrgg;
    double BSIM4v2gcrgs;
    double BSIM4v2gcrgb;

    double BSIM4v2gstot;
    double BSIM4v2gstotd;
    double BSIM4v2gstotg;
    double BSIM4v2gstots;
    double BSIM4v2gstotb;

    double BSIM4v2gdtot;
    double BSIM4v2gdtotd;
    double BSIM4v2gdtotg;
    double BSIM4v2gdtots;
    double BSIM4v2gdtotb;

    double BSIM4v2cggb;
    double BSIM4v2cgdb;
    double BSIM4v2cgsb;
    double BSIM4v2cbgb;
    double BSIM4v2cbdb;
    double BSIM4v2cbsb;
    double BSIM4v2cdgb;
    double BSIM4v2cddb;
    double BSIM4v2cdsb;
    double BSIM4v2csgb;
    double BSIM4v2csdb;
    double BSIM4v2cssb;
    double BSIM4v2cgbb;
    double BSIM4v2cdbb;
    double BSIM4v2csbb;
    double BSIM4v2cbbb;
    double BSIM4v2capbd;
    double BSIM4v2capbs;

    double BSIM4v2cqgb;
    double BSIM4v2cqdb;
    double BSIM4v2cqsb;
    double BSIM4v2cqbb;

    double BSIM4v2qgate;
    double BSIM4v2qbulk;
    double BSIM4v2qdrn;
    double BSIM4v2qsrc;

    double BSIM4v2qchqs;
    double BSIM4v2taunet;
    double BSIM4v2gtau;
    double BSIM4v2gtg;
    double BSIM4v2gtd;
    double BSIM4v2gts;
    double BSIM4v2gtb;

    struct bsim4SizeDependParam  *pParam;

    unsigned BSIM4v2lGiven :1;
    unsigned BSIM4v2wGiven :1;
    unsigned BSIM4v2nfGiven :1;
    unsigned BSIM4v2mGiven :1;
    unsigned BSIM4v2minGiven :1;
    unsigned BSIM4v2drainAreaGiven :1;
    unsigned BSIM4v2sourceAreaGiven    :1;
    unsigned BSIM4v2drainSquaresGiven  :1;
    unsigned BSIM4v2sourceSquaresGiven :1;
    unsigned BSIM4v2drainPerimeterGiven    :1;
    unsigned BSIM4v2sourcePerimeterGiven   :1;
    unsigned BSIM4v2rbdbGiven   :1;
    unsigned BSIM4v2rbsbGiven   :1;
    unsigned BSIM4v2rbpbGiven   :1;
    unsigned BSIM4v2rbpdGiven   :1;
    unsigned BSIM4v2rbpsGiven   :1;
    unsigned BSIM4v2icVDSGiven :1;
    unsigned BSIM4v2icVGSGiven :1;
    unsigned BSIM4v2icVBSGiven :1;
    unsigned BSIM4v2trnqsModGiven :1;
    unsigned BSIM4v2acnqsModGiven :1;
    unsigned BSIM4v2rbodyModGiven :1;
    unsigned BSIM4v2rgateModGiven :1;
    unsigned BSIM4v2geoModGiven :1;
    unsigned BSIM4v2rgeoModGiven :1;

    double *BSIM4v2DPdPtr;
    double *BSIM4v2DPdpPtr;
    double *BSIM4v2DPgpPtr;
    double *BSIM4v2DPgmPtr;
    double *BSIM4v2DPspPtr;
    double *BSIM4v2DPbpPtr;
    double *BSIM4v2DPdbPtr;

    double *BSIM4v2DdPtr;
    double *BSIM4v2DdpPtr;

    double *BSIM4v2GPdpPtr;
    double *BSIM4v2GPgpPtr;
    double *BSIM4v2GPgmPtr;
    double *BSIM4v2GPgePtr;
    double *BSIM4v2GPspPtr;
    double *BSIM4v2GPbpPtr;

    double *BSIM4v2GMdpPtr;
    double *BSIM4v2GMgpPtr;
    double *BSIM4v2GMgmPtr;
    double *BSIM4v2GMgePtr;
    double *BSIM4v2GMspPtr;
    double *BSIM4v2GMbpPtr;

    double *BSIM4v2GEdpPtr;
    double *BSIM4v2GEgpPtr;
    double *BSIM4v2GEgmPtr;
    double *BSIM4v2GEgePtr;
    double *BSIM4v2GEspPtr;
    double *BSIM4v2GEbpPtr;

    double *BSIM4v2SPdpPtr;
    double *BSIM4v2SPgpPtr;
    double *BSIM4v2SPgmPtr;
    double *BSIM4v2SPsPtr;
    double *BSIM4v2SPspPtr;
    double *BSIM4v2SPbpPtr;
    double *BSIM4v2SPsbPtr;

    double *BSIM4v2SspPtr;
    double *BSIM4v2SsPtr;

    double *BSIM4v2BPdpPtr;
    double *BSIM4v2BPgpPtr;
    double *BSIM4v2BPgmPtr;
    double *BSIM4v2BPspPtr;
    double *BSIM4v2BPdbPtr;
    double *BSIM4v2BPbPtr;
    double *BSIM4v2BPsbPtr;
    double *BSIM4v2BPbpPtr;

    double *BSIM4v2DBdpPtr;
    double *BSIM4v2DBdbPtr;
    double *BSIM4v2DBbpPtr;
    double *BSIM4v2DBbPtr;

    double *BSIM4v2SBspPtr;
    double *BSIM4v2SBbpPtr;
    double *BSIM4v2SBbPtr;
    double *BSIM4v2SBsbPtr;

    double *BSIM4v2BdbPtr;
    double *BSIM4v2BbpPtr;
    double *BSIM4v2BsbPtr;
    double *BSIM4v2BbPtr;

    double *BSIM4v2DgpPtr;
    double *BSIM4v2DspPtr;
    double *BSIM4v2DbpPtr;
    double *BSIM4v2SdpPtr;
    double *BSIM4v2SgpPtr;
    double *BSIM4v2SbpPtr;

    double *BSIM4v2QdpPtr;
    double *BSIM4v2QgpPtr;
    double *BSIM4v2QspPtr;
    double *BSIM4v2QbpPtr;
    double *BSIM4v2QqPtr;
    double *BSIM4v2DPqPtr;
    double *BSIM4v2GPqPtr;
    double *BSIM4v2SPqPtr;


#define BSIM4v2vbd BSIM4v2states+ 0
#define BSIM4v2vbs BSIM4v2states+ 1
#define BSIM4v2vgs BSIM4v2states+ 2
#define BSIM4v2vds BSIM4v2states+ 3
#define BSIM4v2vdbs BSIM4v2states+ 4
#define BSIM4v2vdbd BSIM4v2states+ 5
#define BSIM4v2vsbs BSIM4v2states+ 6
#define BSIM4v2vges BSIM4v2states+ 7
#define BSIM4v2vgms BSIM4v2states+ 8
#define BSIM4v2vses BSIM4v2states+ 9
#define BSIM4v2vdes BSIM4v2states+ 10

#define BSIM4v2qb BSIM4v2states+ 11
#define BSIM4v2cqb BSIM4v2states+ 12
#define BSIM4v2qg BSIM4v2states+ 13
#define BSIM4v2cqg BSIM4v2states+ 14
#define BSIM4v2qd BSIM4v2states+ 15
#define BSIM4v2cqd BSIM4v2states+ 16
#define BSIM4v2qgmid BSIM4v2states+ 17
#define BSIM4v2cqgmid BSIM4v2states+ 18

#define BSIM4v2qbs  BSIM4v2states+ 19
#define BSIM4v2cqbs  BSIM4v2states+ 20
#define BSIM4v2qbd  BSIM4v2states+ 21
#define BSIM4v2cqbd  BSIM4v2states+ 22

#define BSIM4v2qcheq BSIM4v2states+ 23
#define BSIM4v2cqcheq BSIM4v2states+ 24
#define BSIM4v2qcdump BSIM4v2states+ 25
#define BSIM4v2cqcdump BSIM4v2states+ 26
#define BSIM4v2qdef BSIM4v2states+ 27
#define BSIM4v2qs BSIM4v2states+ 28

#define BSIM4v2numStates 29


/* indices to the array of BSIM4v2 NOISE SOURCES */

#define BSIM4v2RDNOIZ       0
#define BSIM4v2RSNOIZ       1
#define BSIM4v2RGNOIZ       2
#define BSIM4v2RBPSNOIZ     3
#define BSIM4v2RBPDNOIZ     4
#define BSIM4v2RBPBNOIZ     5
#define BSIM4v2RBSBNOIZ     6
#define BSIM4v2RBDBNOIZ     7
#define BSIM4v2IDNOIZ       8
#define BSIM4v2FLNOIZ       9
#define BSIM4v2IGSNOIZ      10
#define BSIM4v2IGDNOIZ      11
#define BSIM4v2IGBNOIZ      12
#define BSIM4v2TOTNOIZ      13

#define BSIM4v2NSRCS        14  /* Number of BSIM4v2 noise sources */

#ifndef NONOISE
    double BSIM4v2nVar[NSTATVARS][BSIM4v2NSRCS];
#else /* NONOISE */
        double **BSIM4v2nVar;
#endif /* NONOISE */

} BSIM4v2instance ;

struct bsim4SizeDependParam
{
    double Width;
    double Length;
    double NFinger;

    double BSIM4v2cdsc;           
    double BSIM4v2cdscb;    
    double BSIM4v2cdscd;       
    double BSIM4v2cit;           
    double BSIM4v2nfactor;      
    double BSIM4v2xj;
    double BSIM4v2vsat;         
    double BSIM4v2at;         
    double BSIM4v2a0;   
    double BSIM4v2ags;      
    double BSIM4v2a1;         
    double BSIM4v2a2;         
    double BSIM4v2keta;     
    double BSIM4v2nsub;
    double BSIM4v2ndep;        
    double BSIM4v2nsd;
    double BSIM4v2phin;
    double BSIM4v2ngate;        
    double BSIM4v2gamma1;      
    double BSIM4v2gamma2;     
    double BSIM4v2vbx;      
    double BSIM4v2vbi;       
    double BSIM4v2vbm;       
    double BSIM4v2vbsc;       
    double BSIM4v2xt;       
    double BSIM4v2phi;
    double BSIM4v2litl;
    double BSIM4v2k1;
    double BSIM4v2kt1;
    double BSIM4v2kt1l;
    double BSIM4v2kt2;
    double BSIM4v2k2;
    double BSIM4v2k3;
    double BSIM4v2k3b;
    double BSIM4v2w0;
    double BSIM4v2dvtp0;
    double BSIM4v2dvtp1;
    double BSIM4v2lpe0;
    double BSIM4v2lpeb;
    double BSIM4v2dvt0;      
    double BSIM4v2dvt1;      
    double BSIM4v2dvt2;      
    double BSIM4v2dvt0w;      
    double BSIM4v2dvt1w;      
    double BSIM4v2dvt2w;      
    double BSIM4v2drout;      
    double BSIM4v2dsub;      
    double BSIM4v2vth0;
    double BSIM4v2ua;
    double BSIM4v2ua1;
    double BSIM4v2ub;
    double BSIM4v2ub1;
    double BSIM4v2uc;
    double BSIM4v2uc1;
    double BSIM4v2u0;
    double BSIM4v2eu;
    double BSIM4v2ute;
    double BSIM4v2voff;
    double BSIM4v2minv;
    double BSIM4v2vfb;
    double BSIM4v2delta;
    double BSIM4v2rdsw;       
    double BSIM4v2rds0;       
    double BSIM4v2rs0;
    double BSIM4v2rd0;
    double BSIM4v2rsw;
    double BSIM4v2rdw;
    double BSIM4v2prwg;       
    double BSIM4v2prwb;       
    double BSIM4v2prt;       
    double BSIM4v2eta0;         
    double BSIM4v2etab;         
    double BSIM4v2pclm;      
    double BSIM4v2pdibl1;      
    double BSIM4v2pdibl2;      
    double BSIM4v2pdiblb;      
    double BSIM4v2fprout;
    double BSIM4v2pdits;
    double BSIM4v2pditsd;
    double BSIM4v2pscbe1;       
    double BSIM4v2pscbe2;       
    double BSIM4v2pvag;       
    double BSIM4v2wr;
    double BSIM4v2dwg;
    double BSIM4v2dwb;
    double BSIM4v2b0;
    double BSIM4v2b1;
    double BSIM4v2alpha0;
    double BSIM4v2alpha1;
    double BSIM4v2beta0;
    double BSIM4v2agidl;
    double BSIM4v2bgidl;
    double BSIM4v2cgidl;
    double BSIM4v2egidl;
    double BSIM4v2aigc;
    double BSIM4v2bigc;
    double BSIM4v2cigc;
    double BSIM4v2aigsd;
    double BSIM4v2bigsd;
    double BSIM4v2cigsd;
    double BSIM4v2aigbacc;
    double BSIM4v2bigbacc;
    double BSIM4v2cigbacc;
    double BSIM4v2aigbinv;
    double BSIM4v2bigbinv;
    double BSIM4v2cigbinv;
    double BSIM4v2nigc;
    double BSIM4v2nigbacc;
    double BSIM4v2nigbinv;
    double BSIM4v2ntox;
    double BSIM4v2eigbinv;
    double BSIM4v2pigcd;
    double BSIM4v2poxedge;
    double BSIM4v2xrcrg1;
    double BSIM4v2xrcrg2;
    double BSIM4v2fpkt;
    double BSIM4v2plcr;
    double BSIM4v2plcrl;
    double BSIM4v2plcrd;


    /* CV model */
    double BSIM4v2cgsl;
    double BSIM4v2cgdl;
    double BSIM4v2ckappas;
    double BSIM4v2ckappad;
    double BSIM4v2cf;
    double BSIM4v2clc;
    double BSIM4v2cle;
    double BSIM4v2vfbcv;
    double BSIM4v2noff;
    double BSIM4v2voffcv;
    double BSIM4v2acde;
    double BSIM4v2moin;

/* Pre-calculated constants */

    double BSIM4v2dw;
    double BSIM4v2dl;
    double BSIM4v2leff;
    double BSIM4v2weff;

    double BSIM4v2dwc;
    double BSIM4v2dlc;
    double BSIM4v2dlcig;
    double BSIM4v2dwj;
    double BSIM4v2leffCV;
    double BSIM4v2weffCV;
    double BSIM4v2weffCJ;
    double BSIM4v2abulkCVfactor;
    double BSIM4v2cgso;
    double BSIM4v2cgdo;
    double BSIM4v2cgbo;

    double BSIM4v2u0temp;       
    double BSIM4v2vsattemp;   
    double BSIM4v2sqrtPhi;   
    double BSIM4v2phis3;   
    double BSIM4v2Xdep0;          
    double BSIM4v2sqrtXdep0;          
    double BSIM4v2theta0vb0;
    double BSIM4v2thetaRout; 
    double BSIM4v2mstar;
    double BSIM4v2voffcbn;
    double BSIM4v2rdswmin;
    double BSIM4v2rdwmin;
    double BSIM4v2rswmin;
    double BSIM4v2vfbsd;

    double BSIM4v2cof1;
    double BSIM4v2cof2;
    double BSIM4v2cof3;
    double BSIM4v2cof4;
    double BSIM4v2cdep0;
    double BSIM4v2vfbzb;
    double BSIM4v2vtfbphi1;
    double BSIM4v2vtfbphi2;
    double BSIM4v2ToxRatio;
    double BSIM4v2Aechvb;
    double BSIM4v2Bechvb;
    double BSIM4v2ToxRatioEdge;
    double BSIM4v2AechvbEdge;
    double BSIM4v2BechvbEdge;
    double BSIM4v2ldeb;
    double BSIM4v2k1ox;
    double BSIM4v2k2ox;

    struct bsim4SizeDependParam  *pNext;
};


typedef struct sBSIM4v2model 
{
    int BSIM4v2modType;
    struct sBSIM4v2model *BSIM4v2nextModel;
    BSIM4v2instance *BSIM4v2instances;
    IFuid BSIM4v2modName; 
    int BSIM4v2type;

    int    BSIM4v2mobMod;
    int    BSIM4v2capMod;
    int    BSIM4v2dioMod;
    int    BSIM4v2trnqsMod;
    int    BSIM4v2acnqsMod;
    int    BSIM4v2fnoiMod;
    int    BSIM4v2tnoiMod;
    int    BSIM4v2rdsMod;
    int    BSIM4v2rbodyMod;
    int    BSIM4v2rgateMod;
    int    BSIM4v2perMod;
    int    BSIM4v2geoMod;
    int    BSIM4v2igcMod;
    int    BSIM4v2igbMod;
    int    BSIM4v2binUnit;
    int    BSIM4v2paramChk;
    char   *BSIM4v2version;             
    double BSIM4v2toxe;             
    double BSIM4v2toxp;
    double BSIM4v2toxm;
    double BSIM4v2dtox;
    double BSIM4v2epsrox;
    double BSIM4v2cdsc;           
    double BSIM4v2cdscb; 
    double BSIM4v2cdscd;          
    double BSIM4v2cit;           
    double BSIM4v2nfactor;      
    double BSIM4v2xj;
    double BSIM4v2vsat;         
    double BSIM4v2at;         
    double BSIM4v2a0;   
    double BSIM4v2ags;      
    double BSIM4v2a1;         
    double BSIM4v2a2;         
    double BSIM4v2keta;     
    double BSIM4v2nsub;
    double BSIM4v2ndep;        
    double BSIM4v2nsd;
    double BSIM4v2phin;
    double BSIM4v2ngate;        
    double BSIM4v2gamma1;      
    double BSIM4v2gamma2;     
    double BSIM4v2vbx;      
    double BSIM4v2vbm;       
    double BSIM4v2xt;       
    double BSIM4v2k1;
    double BSIM4v2kt1;
    double BSIM4v2kt1l;
    double BSIM4v2kt2;
    double BSIM4v2k2;
    double BSIM4v2k3;
    double BSIM4v2k3b;
    double BSIM4v2w0;
    double BSIM4v2dvtp0;
    double BSIM4v2dvtp1;
    double BSIM4v2lpe0;
    double BSIM4v2lpeb;
    double BSIM4v2dvt0;      
    double BSIM4v2dvt1;      
    double BSIM4v2dvt2;      
    double BSIM4v2dvt0w;      
    double BSIM4v2dvt1w;      
    double BSIM4v2dvt2w;      
    double BSIM4v2drout;      
    double BSIM4v2dsub;      
    double BSIM4v2vth0;
    double BSIM4v2eu;
    double BSIM4v2ua;
    double BSIM4v2ua1;
    double BSIM4v2ub;
    double BSIM4v2ub1;
    double BSIM4v2uc;
    double BSIM4v2uc1;
    double BSIM4v2u0;
    double BSIM4v2ute;
    double BSIM4v2voff;
    double BSIM4v2minv;
    double BSIM4v2voffl;
    double BSIM4v2delta;
    double BSIM4v2rdsw;       
    double BSIM4v2rdswmin;
    double BSIM4v2rdwmin;
    double BSIM4v2rswmin;
    double BSIM4v2rsw;
    double BSIM4v2rdw;
    double BSIM4v2prwg;
    double BSIM4v2prwb;
    double BSIM4v2prt;       
    double BSIM4v2eta0;         
    double BSIM4v2etab;         
    double BSIM4v2pclm;      
    double BSIM4v2pdibl1;      
    double BSIM4v2pdibl2;      
    double BSIM4v2pdiblb;
    double BSIM4v2fprout;
    double BSIM4v2pdits;
    double BSIM4v2pditsd;
    double BSIM4v2pditsl;
    double BSIM4v2pscbe1;       
    double BSIM4v2pscbe2;       
    double BSIM4v2pvag;       
    double BSIM4v2wr;
    double BSIM4v2dwg;
    double BSIM4v2dwb;
    double BSIM4v2b0;
    double BSIM4v2b1;
    double BSIM4v2alpha0;
    double BSIM4v2alpha1;
    double BSIM4v2beta0;
    double BSIM4v2agidl;
    double BSIM4v2bgidl;
    double BSIM4v2cgidl;
    double BSIM4v2egidl;
    double BSIM4v2aigc;
    double BSIM4v2bigc;
    double BSIM4v2cigc;
    double BSIM4v2aigsd;
    double BSIM4v2bigsd;
    double BSIM4v2cigsd;
    double BSIM4v2aigbacc;
    double BSIM4v2bigbacc;
    double BSIM4v2cigbacc;
    double BSIM4v2aigbinv;
    double BSIM4v2bigbinv;
    double BSIM4v2cigbinv;
    double BSIM4v2nigc;
    double BSIM4v2nigbacc;
    double BSIM4v2nigbinv;
    double BSIM4v2ntox;
    double BSIM4v2eigbinv;
    double BSIM4v2pigcd;
    double BSIM4v2poxedge;
    double BSIM4v2toxref;
    double BSIM4v2ijthdfwd;
    double BSIM4v2ijthsfwd;
    double BSIM4v2ijthdrev;
    double BSIM4v2ijthsrev;
    double BSIM4v2xjbvd;
    double BSIM4v2xjbvs;
    double BSIM4v2bvd;
    double BSIM4v2bvs;
    double BSIM4v2xrcrg1;
    double BSIM4v2xrcrg2;

    double BSIM4v2vfb;
    double BSIM4v2gbmin;
    double BSIM4v2rbdb;
    double BSIM4v2rbsb;
    double BSIM4v2rbpb;
    double BSIM4v2rbps;
    double BSIM4v2rbpd;
    double BSIM4v2tnoia;
    double BSIM4v2tnoib;
    double BSIM4v2ntnoi;

    /* CV model and Parasitics */
    double BSIM4v2cgsl;
    double BSIM4v2cgdl;
    double BSIM4v2ckappas;
    double BSIM4v2ckappad;
    double BSIM4v2cf;
    double BSIM4v2vfbcv;
    double BSIM4v2clc;
    double BSIM4v2cle;
    double BSIM4v2dwc;
    double BSIM4v2dlc;
    double BSIM4v2xw;
    double BSIM4v2xl;
    double BSIM4v2dlcig;
    double BSIM4v2dwj;
    double BSIM4v2noff;
    double BSIM4v2voffcv;
    double BSIM4v2acde;
    double BSIM4v2moin;
    double BSIM4v2tcj;
    double BSIM4v2tcjsw;
    double BSIM4v2tcjswg;
    double BSIM4v2tpb;
    double BSIM4v2tpbsw;
    double BSIM4v2tpbswg;
    double BSIM4v2dmcg;
    double BSIM4v2dmci;
    double BSIM4v2dmdg;
    double BSIM4v2dmcgt;
    double BSIM4v2xgw;
    double BSIM4v2xgl;
    double BSIM4v2rshg;
    double BSIM4v2ngcon;

    /* Length Dependence */
    double BSIM4v2lcdsc;           
    double BSIM4v2lcdscb; 
    double BSIM4v2lcdscd;          
    double BSIM4v2lcit;           
    double BSIM4v2lnfactor;      
    double BSIM4v2lxj;
    double BSIM4v2lvsat;         
    double BSIM4v2lat;         
    double BSIM4v2la0;   
    double BSIM4v2lags;      
    double BSIM4v2la1;         
    double BSIM4v2la2;         
    double BSIM4v2lketa;     
    double BSIM4v2lnsub;
    double BSIM4v2lndep;        
    double BSIM4v2lnsd;
    double BSIM4v2lphin;
    double BSIM4v2lngate;        
    double BSIM4v2lgamma1;      
    double BSIM4v2lgamma2;     
    double BSIM4v2lvbx;      
    double BSIM4v2lvbm;       
    double BSIM4v2lxt;       
    double BSIM4v2lk1;
    double BSIM4v2lkt1;
    double BSIM4v2lkt1l;
    double BSIM4v2lkt2;
    double BSIM4v2lk2;
    double BSIM4v2lk3;
    double BSIM4v2lk3b;
    double BSIM4v2lw0;
    double BSIM4v2ldvtp0;
    double BSIM4v2ldvtp1;
    double BSIM4v2llpe0;
    double BSIM4v2llpeb;
    double BSIM4v2ldvt0;      
    double BSIM4v2ldvt1;      
    double BSIM4v2ldvt2;      
    double BSIM4v2ldvt0w;      
    double BSIM4v2ldvt1w;      
    double BSIM4v2ldvt2w;      
    double BSIM4v2ldrout;      
    double BSIM4v2ldsub;      
    double BSIM4v2lvth0;
    double BSIM4v2lua;
    double BSIM4v2lua1;
    double BSIM4v2lub;
    double BSIM4v2lub1;
    double BSIM4v2luc;
    double BSIM4v2luc1;
    double BSIM4v2lu0;
    double BSIM4v2leu;
    double BSIM4v2lute;
    double BSIM4v2lvoff;
    double BSIM4v2lminv;
    double BSIM4v2ldelta;
    double BSIM4v2lrdsw;       
    double BSIM4v2lrsw;
    double BSIM4v2lrdw;
    double BSIM4v2lprwg;
    double BSIM4v2lprwb;
    double BSIM4v2lprt;       
    double BSIM4v2leta0;         
    double BSIM4v2letab;         
    double BSIM4v2lpclm;      
    double BSIM4v2lpdibl1;      
    double BSIM4v2lpdibl2;      
    double BSIM4v2lpdiblb;
    double BSIM4v2lfprout;
    double BSIM4v2lpdits;
    double BSIM4v2lpditsd;
    double BSIM4v2lpscbe1;       
    double BSIM4v2lpscbe2;       
    double BSIM4v2lpvag;       
    double BSIM4v2lwr;
    double BSIM4v2ldwg;
    double BSIM4v2ldwb;
    double BSIM4v2lb0;
    double BSIM4v2lb1;
    double BSIM4v2lalpha0;
    double BSIM4v2lalpha1;
    double BSIM4v2lbeta0;
    double BSIM4v2lvfb;
    double BSIM4v2lagidl;
    double BSIM4v2lbgidl;
    double BSIM4v2lcgidl;
    double BSIM4v2legidl;
    double BSIM4v2laigc;
    double BSIM4v2lbigc;
    double BSIM4v2lcigc;
    double BSIM4v2laigsd;
    double BSIM4v2lbigsd;
    double BSIM4v2lcigsd;
    double BSIM4v2laigbacc;
    double BSIM4v2lbigbacc;
    double BSIM4v2lcigbacc;
    double BSIM4v2laigbinv;
    double BSIM4v2lbigbinv;
    double BSIM4v2lcigbinv;
    double BSIM4v2lnigc;
    double BSIM4v2lnigbacc;
    double BSIM4v2lnigbinv;
    double BSIM4v2lntox;
    double BSIM4v2leigbinv;
    double BSIM4v2lpigcd;
    double BSIM4v2lpoxedge;
    double BSIM4v2lxrcrg1;
    double BSIM4v2lxrcrg2;

    /* CV model */
    double BSIM4v2lcgsl;
    double BSIM4v2lcgdl;
    double BSIM4v2lckappas;
    double BSIM4v2lckappad;
    double BSIM4v2lcf;
    double BSIM4v2lclc;
    double BSIM4v2lcle;
    double BSIM4v2lvfbcv;
    double BSIM4v2lnoff;
    double BSIM4v2lvoffcv;
    double BSIM4v2lacde;
    double BSIM4v2lmoin;

    /* Width Dependence */
    double BSIM4v2wcdsc;           
    double BSIM4v2wcdscb; 
    double BSIM4v2wcdscd;          
    double BSIM4v2wcit;           
    double BSIM4v2wnfactor;      
    double BSIM4v2wxj;
    double BSIM4v2wvsat;         
    double BSIM4v2wat;         
    double BSIM4v2wa0;   
    double BSIM4v2wags;      
    double BSIM4v2wa1;         
    double BSIM4v2wa2;         
    double BSIM4v2wketa;     
    double BSIM4v2wnsub;
    double BSIM4v2wndep;        
    double BSIM4v2wnsd;
    double BSIM4v2wphin;
    double BSIM4v2wngate;        
    double BSIM4v2wgamma1;      
    double BSIM4v2wgamma2;     
    double BSIM4v2wvbx;      
    double BSIM4v2wvbm;       
    double BSIM4v2wxt;       
    double BSIM4v2wk1;
    double BSIM4v2wkt1;
    double BSIM4v2wkt1l;
    double BSIM4v2wkt2;
    double BSIM4v2wk2;
    double BSIM4v2wk3;
    double BSIM4v2wk3b;
    double BSIM4v2ww0;
    double BSIM4v2wdvtp0;
    double BSIM4v2wdvtp1;
    double BSIM4v2wlpe0;
    double BSIM4v2wlpeb;
    double BSIM4v2wdvt0;      
    double BSIM4v2wdvt1;      
    double BSIM4v2wdvt2;      
    double BSIM4v2wdvt0w;      
    double BSIM4v2wdvt1w;      
    double BSIM4v2wdvt2w;      
    double BSIM4v2wdrout;      
    double BSIM4v2wdsub;      
    double BSIM4v2wvth0;
    double BSIM4v2wua;
    double BSIM4v2wua1;
    double BSIM4v2wub;
    double BSIM4v2wub1;
    double BSIM4v2wuc;
    double BSIM4v2wuc1;
    double BSIM4v2wu0;
    double BSIM4v2weu;
    double BSIM4v2wute;
    double BSIM4v2wvoff;
    double BSIM4v2wminv;
    double BSIM4v2wdelta;
    double BSIM4v2wrdsw;       
    double BSIM4v2wrsw;
    double BSIM4v2wrdw;
    double BSIM4v2wprwg;
    double BSIM4v2wprwb;
    double BSIM4v2wprt;       
    double BSIM4v2weta0;         
    double BSIM4v2wetab;         
    double BSIM4v2wpclm;      
    double BSIM4v2wpdibl1;      
    double BSIM4v2wpdibl2;      
    double BSIM4v2wpdiblb;
    double BSIM4v2wfprout;
    double BSIM4v2wpdits;
    double BSIM4v2wpditsd;
    double BSIM4v2wpscbe1;       
    double BSIM4v2wpscbe2;       
    double BSIM4v2wpvag;       
    double BSIM4v2wwr;
    double BSIM4v2wdwg;
    double BSIM4v2wdwb;
    double BSIM4v2wb0;
    double BSIM4v2wb1;
    double BSIM4v2walpha0;
    double BSIM4v2walpha1;
    double BSIM4v2wbeta0;
    double BSIM4v2wvfb;
    double BSIM4v2wagidl;
    double BSIM4v2wbgidl;
    double BSIM4v2wcgidl;
    double BSIM4v2wegidl;
    double BSIM4v2waigc;
    double BSIM4v2wbigc;
    double BSIM4v2wcigc;
    double BSIM4v2waigsd;
    double BSIM4v2wbigsd;
    double BSIM4v2wcigsd;
    double BSIM4v2waigbacc;
    double BSIM4v2wbigbacc;
    double BSIM4v2wcigbacc;
    double BSIM4v2waigbinv;
    double BSIM4v2wbigbinv;
    double BSIM4v2wcigbinv;
    double BSIM4v2wnigc;
    double BSIM4v2wnigbacc;
    double BSIM4v2wnigbinv;
    double BSIM4v2wntox;
    double BSIM4v2weigbinv;
    double BSIM4v2wpigcd;
    double BSIM4v2wpoxedge;
    double BSIM4v2wxrcrg1;
    double BSIM4v2wxrcrg2;

    /* CV model */
    double BSIM4v2wcgsl;
    double BSIM4v2wcgdl;
    double BSIM4v2wckappas;
    double BSIM4v2wckappad;
    double BSIM4v2wcf;
    double BSIM4v2wclc;
    double BSIM4v2wcle;
    double BSIM4v2wvfbcv;
    double BSIM4v2wnoff;
    double BSIM4v2wvoffcv;
    double BSIM4v2wacde;
    double BSIM4v2wmoin;

    /* Cross-term Dependence */
    double BSIM4v2pcdsc;           
    double BSIM4v2pcdscb; 
    double BSIM4v2pcdscd;          
    double BSIM4v2pcit;           
    double BSIM4v2pnfactor;      
    double BSIM4v2pxj;
    double BSIM4v2pvsat;         
    double BSIM4v2pat;         
    double BSIM4v2pa0;   
    double BSIM4v2pags;      
    double BSIM4v2pa1;         
    double BSIM4v2pa2;         
    double BSIM4v2pketa;     
    double BSIM4v2pnsub;
    double BSIM4v2pndep;        
    double BSIM4v2pnsd;
    double BSIM4v2pphin;
    double BSIM4v2pngate;        
    double BSIM4v2pgamma1;      
    double BSIM4v2pgamma2;     
    double BSIM4v2pvbx;      
    double BSIM4v2pvbm;       
    double BSIM4v2pxt;       
    double BSIM4v2pk1;
    double BSIM4v2pkt1;
    double BSIM4v2pkt1l;
    double BSIM4v2pkt2;
    double BSIM4v2pk2;
    double BSIM4v2pk3;
    double BSIM4v2pk3b;
    double BSIM4v2pw0;
    double BSIM4v2pdvtp0;
    double BSIM4v2pdvtp1;
    double BSIM4v2plpe0;
    double BSIM4v2plpeb;
    double BSIM4v2pdvt0;      
    double BSIM4v2pdvt1;      
    double BSIM4v2pdvt2;      
    double BSIM4v2pdvt0w;      
    double BSIM4v2pdvt1w;      
    double BSIM4v2pdvt2w;      
    double BSIM4v2pdrout;      
    double BSIM4v2pdsub;      
    double BSIM4v2pvth0;
    double BSIM4v2pua;
    double BSIM4v2pua1;
    double BSIM4v2pub;
    double BSIM4v2pub1;
    double BSIM4v2puc;
    double BSIM4v2puc1;
    double BSIM4v2pu0;
    double BSIM4v2peu;
    double BSIM4v2pute;
    double BSIM4v2pvoff;
    double BSIM4v2pminv;
    double BSIM4v2pdelta;
    double BSIM4v2prdsw;
    double BSIM4v2prsw;
    double BSIM4v2prdw;
    double BSIM4v2pprwg;
    double BSIM4v2pprwb;
    double BSIM4v2pprt;       
    double BSIM4v2peta0;         
    double BSIM4v2petab;         
    double BSIM4v2ppclm;      
    double BSIM4v2ppdibl1;      
    double BSIM4v2ppdibl2;      
    double BSIM4v2ppdiblb;
    double BSIM4v2pfprout;
    double BSIM4v2ppdits;
    double BSIM4v2ppditsd;
    double BSIM4v2ppscbe1;       
    double BSIM4v2ppscbe2;       
    double BSIM4v2ppvag;       
    double BSIM4v2pwr;
    double BSIM4v2pdwg;
    double BSIM4v2pdwb;
    double BSIM4v2pb0;
    double BSIM4v2pb1;
    double BSIM4v2palpha0;
    double BSIM4v2palpha1;
    double BSIM4v2pbeta0;
    double BSIM4v2pvfb;
    double BSIM4v2pagidl;
    double BSIM4v2pbgidl;
    double BSIM4v2pcgidl;
    double BSIM4v2pegidl;
    double BSIM4v2paigc;
    double BSIM4v2pbigc;
    double BSIM4v2pcigc;
    double BSIM4v2paigsd;
    double BSIM4v2pbigsd;
    double BSIM4v2pcigsd;
    double BSIM4v2paigbacc;
    double BSIM4v2pbigbacc;
    double BSIM4v2pcigbacc;
    double BSIM4v2paigbinv;
    double BSIM4v2pbigbinv;
    double BSIM4v2pcigbinv;
    double BSIM4v2pnigc;
    double BSIM4v2pnigbacc;
    double BSIM4v2pnigbinv;
    double BSIM4v2pntox;
    double BSIM4v2peigbinv;
    double BSIM4v2ppigcd;
    double BSIM4v2ppoxedge;
    double BSIM4v2pxrcrg1;
    double BSIM4v2pxrcrg2;

    /* CV model */
    double BSIM4v2pcgsl;
    double BSIM4v2pcgdl;
    double BSIM4v2pckappas;
    double BSIM4v2pckappad;
    double BSIM4v2pcf;
    double BSIM4v2pclc;
    double BSIM4v2pcle;
    double BSIM4v2pvfbcv;
    double BSIM4v2pnoff;
    double BSIM4v2pvoffcv;
    double BSIM4v2pacde;
    double BSIM4v2pmoin;

    double BSIM4v2tnom;
    double BSIM4v2cgso;
    double BSIM4v2cgdo;
    double BSIM4v2cgbo;
    double BSIM4v2xpart;
    double BSIM4v2cFringOut;
    double BSIM4v2cFringMax;

    double BSIM4v2sheetResistance;
    double BSIM4v2SjctSatCurDensity;
    double BSIM4v2DjctSatCurDensity;
    double BSIM4v2SjctSidewallSatCurDensity;
    double BSIM4v2DjctSidewallSatCurDensity;
    double BSIM4v2SjctGateSidewallSatCurDensity;
    double BSIM4v2DjctGateSidewallSatCurDensity;
    double BSIM4v2SbulkJctPotential;
    double BSIM4v2DbulkJctPotential;
    double BSIM4v2SbulkJctBotGradingCoeff;
    double BSIM4v2DbulkJctBotGradingCoeff;
    double BSIM4v2SbulkJctSideGradingCoeff;
    double BSIM4v2DbulkJctSideGradingCoeff;
    double BSIM4v2SbulkJctGateSideGradingCoeff;
    double BSIM4v2DbulkJctGateSideGradingCoeff;
    double BSIM4v2SsidewallJctPotential;
    double BSIM4v2DsidewallJctPotential;
    double BSIM4v2SGatesidewallJctPotential;
    double BSIM4v2DGatesidewallJctPotential;
    double BSIM4v2SunitAreaJctCap;
    double BSIM4v2DunitAreaJctCap;
    double BSIM4v2SunitLengthSidewallJctCap;
    double BSIM4v2DunitLengthSidewallJctCap;
    double BSIM4v2SunitLengthGateSidewallJctCap;
    double BSIM4v2DunitLengthGateSidewallJctCap;
    double BSIM4v2SjctEmissionCoeff;
    double BSIM4v2DjctEmissionCoeff;
    double BSIM4v2SjctTempExponent;
    double BSIM4v2DjctTempExponent;

    double BSIM4v2Lint;
    double BSIM4v2Ll;
    double BSIM4v2Llc;
    double BSIM4v2Lln;
    double BSIM4v2Lw;
    double BSIM4v2Lwc;
    double BSIM4v2Lwn;
    double BSIM4v2Lwl;
    double BSIM4v2Lwlc;
    double BSIM4v2Lmin;
    double BSIM4v2Lmax;

    double BSIM4v2Wint;
    double BSIM4v2Wl;
    double BSIM4v2Wlc;
    double BSIM4v2Wln;
    double BSIM4v2Ww;
    double BSIM4v2Wwc;
    double BSIM4v2Wwn;
    double BSIM4v2Wwl;
    double BSIM4v2Wwlc;
    double BSIM4v2Wmin;
    double BSIM4v2Wmax;


/* Pre-calculated constants
 * move to size-dependent param */
    double BSIM4v2vtm;   
    double BSIM4v2coxe;
    double BSIM4v2coxp;
    double BSIM4v2cof1;
    double BSIM4v2cof2;
    double BSIM4v2cof3;
    double BSIM4v2cof4;
    double BSIM4v2vcrit;
    double BSIM4v2factor1;
    double BSIM4v2PhiBS;
    double BSIM4v2PhiBSWS;
    double BSIM4v2PhiBSWGS;
    double BSIM4v2SjctTempSatCurDensity;
    double BSIM4v2SjctSidewallTempSatCurDensity;
    double BSIM4v2SjctGateSidewallTempSatCurDensity;
    double BSIM4v2PhiBD;
    double BSIM4v2PhiBSWD;
    double BSIM4v2PhiBSWGD;
    double BSIM4v2DjctTempSatCurDensity;
    double BSIM4v2DjctSidewallTempSatCurDensity;
    double BSIM4v2DjctGateSidewallTempSatCurDensity;
    double BSIM4v2SunitAreaTempJctCap;
    double BSIM4v2DunitAreaTempJctCap;
    double BSIM4v2SunitLengthSidewallTempJctCap;
    double BSIM4v2DunitLengthSidewallTempJctCap;
    double BSIM4v2SunitLengthGateSidewallTempJctCap;
    double BSIM4v2DunitLengthGateSidewallTempJctCap;

    double BSIM4v2oxideTrapDensityA;      
    double BSIM4v2oxideTrapDensityB;     
    double BSIM4v2oxideTrapDensityC;  
    double BSIM4v2em;  
    double BSIM4v2ef;  
    double BSIM4v2af;  
    double BSIM4v2kf;  

    struct bsim4SizeDependParam *pSizeDependParamKnot;

    /* Flags */
    unsigned  BSIM4v2mobModGiven :1;
    unsigned  BSIM4v2binUnitGiven :1;
    unsigned  BSIM4v2capModGiven :1;
    unsigned  BSIM4v2dioModGiven :1;
    unsigned  BSIM4v2rdsModGiven :1;
    unsigned  BSIM4v2rbodyModGiven :1;
    unsigned  BSIM4v2rgateModGiven :1;
    unsigned  BSIM4v2perModGiven :1;
    unsigned  BSIM4v2geoModGiven :1;
    unsigned  BSIM4v2paramChkGiven :1;
    unsigned  BSIM4v2trnqsModGiven :1;
    unsigned  BSIM4v2acnqsModGiven :1;
    unsigned  BSIM4v2fnoiModGiven :1;
    unsigned  BSIM4v2tnoiModGiven :1;
    unsigned  BSIM4v2igcModGiven :1;
    unsigned  BSIM4v2igbModGiven :1;
    unsigned  BSIM4v2typeGiven   :1;
    unsigned  BSIM4v2toxrefGiven   :1;
    unsigned  BSIM4v2toxeGiven   :1;
    unsigned  BSIM4v2toxpGiven   :1;
    unsigned  BSIM4v2toxmGiven   :1;
    unsigned  BSIM4v2dtoxGiven   :1;
    unsigned  BSIM4v2epsroxGiven   :1;
    unsigned  BSIM4v2versionGiven   :1;
    unsigned  BSIM4v2cdscGiven   :1;
    unsigned  BSIM4v2cdscbGiven   :1;
    unsigned  BSIM4v2cdscdGiven   :1;
    unsigned  BSIM4v2citGiven   :1;
    unsigned  BSIM4v2nfactorGiven   :1;
    unsigned  BSIM4v2xjGiven   :1;
    unsigned  BSIM4v2vsatGiven   :1;
    unsigned  BSIM4v2atGiven   :1;
    unsigned  BSIM4v2a0Given   :1;
    unsigned  BSIM4v2agsGiven   :1;
    unsigned  BSIM4v2a1Given   :1;
    unsigned  BSIM4v2a2Given   :1;
    unsigned  BSIM4v2ketaGiven   :1;    
    unsigned  BSIM4v2nsubGiven   :1;
    unsigned  BSIM4v2ndepGiven   :1;
    unsigned  BSIM4v2nsdGiven    :1;
    unsigned  BSIM4v2phinGiven   :1;
    unsigned  BSIM4v2ngateGiven   :1;
    unsigned  BSIM4v2gamma1Given   :1;
    unsigned  BSIM4v2gamma2Given   :1;
    unsigned  BSIM4v2vbxGiven   :1;
    unsigned  BSIM4v2vbmGiven   :1;
    unsigned  BSIM4v2xtGiven   :1;
    unsigned  BSIM4v2k1Given   :1;
    unsigned  BSIM4v2kt1Given   :1;
    unsigned  BSIM4v2kt1lGiven   :1;
    unsigned  BSIM4v2kt2Given   :1;
    unsigned  BSIM4v2k2Given   :1;
    unsigned  BSIM4v2k3Given   :1;
    unsigned  BSIM4v2k3bGiven   :1;
    unsigned  BSIM4v2w0Given   :1;
    unsigned  BSIM4v2dvtp0Given :1;
    unsigned  BSIM4v2dvtp1Given :1;
    unsigned  BSIM4v2lpe0Given   :1;
    unsigned  BSIM4v2lpebGiven   :1;
    unsigned  BSIM4v2dvt0Given   :1;   
    unsigned  BSIM4v2dvt1Given   :1;     
    unsigned  BSIM4v2dvt2Given   :1;     
    unsigned  BSIM4v2dvt0wGiven   :1;   
    unsigned  BSIM4v2dvt1wGiven   :1;     
    unsigned  BSIM4v2dvt2wGiven   :1;     
    unsigned  BSIM4v2droutGiven   :1;     
    unsigned  BSIM4v2dsubGiven   :1;     
    unsigned  BSIM4v2vth0Given   :1;
    unsigned  BSIM4v2euGiven   :1;
    unsigned  BSIM4v2uaGiven   :1;
    unsigned  BSIM4v2ua1Given   :1;
    unsigned  BSIM4v2ubGiven   :1;
    unsigned  BSIM4v2ub1Given   :1;
    unsigned  BSIM4v2ucGiven   :1;
    unsigned  BSIM4v2uc1Given   :1;
    unsigned  BSIM4v2u0Given   :1;
    unsigned  BSIM4v2uteGiven   :1;
    unsigned  BSIM4v2voffGiven   :1;
    unsigned  BSIM4v2vofflGiven  :1;
    unsigned  BSIM4v2minvGiven   :1;
    unsigned  BSIM4v2rdswGiven   :1;      
    unsigned  BSIM4v2rdswminGiven :1;
    unsigned  BSIM4v2rdwminGiven :1;
    unsigned  BSIM4v2rswminGiven :1;
    unsigned  BSIM4v2rswGiven   :1;
    unsigned  BSIM4v2rdwGiven   :1;
    unsigned  BSIM4v2prwgGiven   :1;      
    unsigned  BSIM4v2prwbGiven   :1;      
    unsigned  BSIM4v2prtGiven   :1;      
    unsigned  BSIM4v2eta0Given   :1;    
    unsigned  BSIM4v2etabGiven   :1;    
    unsigned  BSIM4v2pclmGiven   :1;   
    unsigned  BSIM4v2pdibl1Given   :1;   
    unsigned  BSIM4v2pdibl2Given   :1;  
    unsigned  BSIM4v2pdiblbGiven   :1;  
    unsigned  BSIM4v2fproutGiven   :1;
    unsigned  BSIM4v2pditsGiven    :1;
    unsigned  BSIM4v2pditsdGiven    :1;
    unsigned  BSIM4v2pditslGiven    :1;
    unsigned  BSIM4v2pscbe1Given   :1;    
    unsigned  BSIM4v2pscbe2Given   :1;    
    unsigned  BSIM4v2pvagGiven   :1;    
    unsigned  BSIM4v2deltaGiven  :1;     
    unsigned  BSIM4v2wrGiven   :1;
    unsigned  BSIM4v2dwgGiven   :1;
    unsigned  BSIM4v2dwbGiven   :1;
    unsigned  BSIM4v2b0Given   :1;
    unsigned  BSIM4v2b1Given   :1;
    unsigned  BSIM4v2alpha0Given   :1;
    unsigned  BSIM4v2alpha1Given   :1;
    unsigned  BSIM4v2beta0Given   :1;
    unsigned  BSIM4v2agidlGiven   :1;
    unsigned  BSIM4v2bgidlGiven   :1;
    unsigned  BSIM4v2cgidlGiven   :1;
    unsigned  BSIM4v2egidlGiven   :1;
    unsigned  BSIM4v2aigcGiven   :1;
    unsigned  BSIM4v2bigcGiven   :1;
    unsigned  BSIM4v2cigcGiven   :1;
    unsigned  BSIM4v2aigsdGiven   :1;
    unsigned  BSIM4v2bigsdGiven   :1;
    unsigned  BSIM4v2cigsdGiven   :1;
    unsigned  BSIM4v2aigbaccGiven   :1;
    unsigned  BSIM4v2bigbaccGiven   :1;
    unsigned  BSIM4v2cigbaccGiven   :1;
    unsigned  BSIM4v2aigbinvGiven   :1;
    unsigned  BSIM4v2bigbinvGiven   :1;
    unsigned  BSIM4v2cigbinvGiven   :1;
    unsigned  BSIM4v2nigcGiven   :1;
    unsigned  BSIM4v2nigbinvGiven   :1;
    unsigned  BSIM4v2nigbaccGiven   :1;
    unsigned  BSIM4v2ntoxGiven   :1;
    unsigned  BSIM4v2eigbinvGiven   :1;
    unsigned  BSIM4v2pigcdGiven   :1;
    unsigned  BSIM4v2poxedgeGiven   :1;
    unsigned  BSIM4v2ijthdfwdGiven  :1;
    unsigned  BSIM4v2ijthsfwdGiven  :1;
    unsigned  BSIM4v2ijthdrevGiven  :1;
    unsigned  BSIM4v2ijthsrevGiven  :1;
    unsigned  BSIM4v2xjbvdGiven   :1;
    unsigned  BSIM4v2xjbvsGiven   :1;
    unsigned  BSIM4v2bvdGiven   :1;
    unsigned  BSIM4v2bvsGiven   :1;
    unsigned  BSIM4v2vfbGiven   :1;
    unsigned  BSIM4v2gbminGiven :1;
    unsigned  BSIM4v2rbdbGiven :1;
    unsigned  BSIM4v2rbsbGiven :1;
    unsigned  BSIM4v2rbpsGiven :1;
    unsigned  BSIM4v2rbpdGiven :1;
    unsigned  BSIM4v2rbpbGiven :1;
    unsigned  BSIM4v2xrcrg1Given   :1;
    unsigned  BSIM4v2xrcrg2Given   :1;
    unsigned  BSIM4v2tnoiaGiven    :1;
    unsigned  BSIM4v2tnoibGiven    :1;
    unsigned  BSIM4v2ntnoiGiven    :1;

    /* CV model and parasitics */
    unsigned  BSIM4v2cgslGiven   :1;
    unsigned  BSIM4v2cgdlGiven   :1;
    unsigned  BSIM4v2ckappasGiven   :1;
    unsigned  BSIM4v2ckappadGiven   :1;
    unsigned  BSIM4v2cfGiven   :1;
    unsigned  BSIM4v2vfbcvGiven   :1;
    unsigned  BSIM4v2clcGiven   :1;
    unsigned  BSIM4v2cleGiven   :1;
    unsigned  BSIM4v2dwcGiven   :1;
    unsigned  BSIM4v2dlcGiven   :1;
    unsigned  BSIM4v2xwGiven    :1;
    unsigned  BSIM4v2xlGiven    :1;
    unsigned  BSIM4v2dlcigGiven   :1;
    unsigned  BSIM4v2dwjGiven   :1;
    unsigned  BSIM4v2noffGiven  :1;
    unsigned  BSIM4v2voffcvGiven :1;
    unsigned  BSIM4v2acdeGiven  :1;
    unsigned  BSIM4v2moinGiven  :1;
    unsigned  BSIM4v2tcjGiven   :1;
    unsigned  BSIM4v2tcjswGiven :1;
    unsigned  BSIM4v2tcjswgGiven :1;
    unsigned  BSIM4v2tpbGiven    :1;
    unsigned  BSIM4v2tpbswGiven  :1;
    unsigned  BSIM4v2tpbswgGiven :1;
    unsigned  BSIM4v2dmcgGiven :1;
    unsigned  BSIM4v2dmciGiven :1;
    unsigned  BSIM4v2dmdgGiven :1;
    unsigned  BSIM4v2dmcgtGiven :1;
    unsigned  BSIM4v2xgwGiven :1;
    unsigned  BSIM4v2xglGiven :1;
    unsigned  BSIM4v2rshgGiven :1;
    unsigned  BSIM4v2ngconGiven :1;


    /* Length dependence */
    unsigned  BSIM4v2lcdscGiven   :1;
    unsigned  BSIM4v2lcdscbGiven   :1;
    unsigned  BSIM4v2lcdscdGiven   :1;
    unsigned  BSIM4v2lcitGiven   :1;
    unsigned  BSIM4v2lnfactorGiven   :1;
    unsigned  BSIM4v2lxjGiven   :1;
    unsigned  BSIM4v2lvsatGiven   :1;
    unsigned  BSIM4v2latGiven   :1;
    unsigned  BSIM4v2la0Given   :1;
    unsigned  BSIM4v2lagsGiven   :1;
    unsigned  BSIM4v2la1Given   :1;
    unsigned  BSIM4v2la2Given   :1;
    unsigned  BSIM4v2lketaGiven   :1;    
    unsigned  BSIM4v2lnsubGiven   :1;
    unsigned  BSIM4v2lndepGiven   :1;
    unsigned  BSIM4v2lnsdGiven    :1;
    unsigned  BSIM4v2lphinGiven   :1;
    unsigned  BSIM4v2lngateGiven   :1;
    unsigned  BSIM4v2lgamma1Given   :1;
    unsigned  BSIM4v2lgamma2Given   :1;
    unsigned  BSIM4v2lvbxGiven   :1;
    unsigned  BSIM4v2lvbmGiven   :1;
    unsigned  BSIM4v2lxtGiven   :1;
    unsigned  BSIM4v2lk1Given   :1;
    unsigned  BSIM4v2lkt1Given   :1;
    unsigned  BSIM4v2lkt1lGiven   :1;
    unsigned  BSIM4v2lkt2Given   :1;
    unsigned  BSIM4v2lk2Given   :1;
    unsigned  BSIM4v2lk3Given   :1;
    unsigned  BSIM4v2lk3bGiven   :1;
    unsigned  BSIM4v2lw0Given   :1;
    unsigned  BSIM4v2ldvtp0Given :1;
    unsigned  BSIM4v2ldvtp1Given :1;
    unsigned  BSIM4v2llpe0Given   :1;
    unsigned  BSIM4v2llpebGiven   :1;
    unsigned  BSIM4v2ldvt0Given   :1;   
    unsigned  BSIM4v2ldvt1Given   :1;     
    unsigned  BSIM4v2ldvt2Given   :1;     
    unsigned  BSIM4v2ldvt0wGiven   :1;   
    unsigned  BSIM4v2ldvt1wGiven   :1;     
    unsigned  BSIM4v2ldvt2wGiven   :1;     
    unsigned  BSIM4v2ldroutGiven   :1;     
    unsigned  BSIM4v2ldsubGiven   :1;     
    unsigned  BSIM4v2lvth0Given   :1;
    unsigned  BSIM4v2luaGiven   :1;
    unsigned  BSIM4v2lua1Given   :1;
    unsigned  BSIM4v2lubGiven   :1;
    unsigned  BSIM4v2lub1Given   :1;
    unsigned  BSIM4v2lucGiven   :1;
    unsigned  BSIM4v2luc1Given   :1;
    unsigned  BSIM4v2lu0Given   :1;
    unsigned  BSIM4v2leuGiven   :1;
    unsigned  BSIM4v2luteGiven   :1;
    unsigned  BSIM4v2lvoffGiven   :1;
    unsigned  BSIM4v2lminvGiven   :1;
    unsigned  BSIM4v2lrdswGiven   :1;      
    unsigned  BSIM4v2lrswGiven   :1;
    unsigned  BSIM4v2lrdwGiven   :1;
    unsigned  BSIM4v2lprwgGiven   :1;      
    unsigned  BSIM4v2lprwbGiven   :1;      
    unsigned  BSIM4v2lprtGiven   :1;      
    unsigned  BSIM4v2leta0Given   :1;    
    unsigned  BSIM4v2letabGiven   :1;    
    unsigned  BSIM4v2lpclmGiven   :1;   
    unsigned  BSIM4v2lpdibl1Given   :1;   
    unsigned  BSIM4v2lpdibl2Given   :1;  
    unsigned  BSIM4v2lpdiblbGiven   :1;  
    unsigned  BSIM4v2lfproutGiven   :1;
    unsigned  BSIM4v2lpditsGiven    :1;
    unsigned  BSIM4v2lpditsdGiven    :1;
    unsigned  BSIM4v2lpscbe1Given   :1;    
    unsigned  BSIM4v2lpscbe2Given   :1;    
    unsigned  BSIM4v2lpvagGiven   :1;    
    unsigned  BSIM4v2ldeltaGiven  :1;     
    unsigned  BSIM4v2lwrGiven   :1;
    unsigned  BSIM4v2ldwgGiven   :1;
    unsigned  BSIM4v2ldwbGiven   :1;
    unsigned  BSIM4v2lb0Given   :1;
    unsigned  BSIM4v2lb1Given   :1;
    unsigned  BSIM4v2lalpha0Given   :1;
    unsigned  BSIM4v2lalpha1Given   :1;
    unsigned  BSIM4v2lbeta0Given   :1;
    unsigned  BSIM4v2lvfbGiven   :1;
    unsigned  BSIM4v2lagidlGiven   :1;
    unsigned  BSIM4v2lbgidlGiven   :1;
    unsigned  BSIM4v2lcgidlGiven   :1;
    unsigned  BSIM4v2legidlGiven   :1;
    unsigned  BSIM4v2laigcGiven   :1;
    unsigned  BSIM4v2lbigcGiven   :1;
    unsigned  BSIM4v2lcigcGiven   :1;
    unsigned  BSIM4v2laigsdGiven   :1;
    unsigned  BSIM4v2lbigsdGiven   :1;
    unsigned  BSIM4v2lcigsdGiven   :1;
    unsigned  BSIM4v2laigbaccGiven   :1;
    unsigned  BSIM4v2lbigbaccGiven   :1;
    unsigned  BSIM4v2lcigbaccGiven   :1;
    unsigned  BSIM4v2laigbinvGiven   :1;
    unsigned  BSIM4v2lbigbinvGiven   :1;
    unsigned  BSIM4v2lcigbinvGiven   :1;
    unsigned  BSIM4v2lnigcGiven   :1;
    unsigned  BSIM4v2lnigbinvGiven   :1;
    unsigned  BSIM4v2lnigbaccGiven   :1;
    unsigned  BSIM4v2lntoxGiven   :1;
    unsigned  BSIM4v2leigbinvGiven   :1;
    unsigned  BSIM4v2lpigcdGiven   :1;
    unsigned  BSIM4v2lpoxedgeGiven   :1;
    unsigned  BSIM4v2lxrcrg1Given   :1;
    unsigned  BSIM4v2lxrcrg2Given   :1;

    /* CV model */
    unsigned  BSIM4v2lcgslGiven   :1;
    unsigned  BSIM4v2lcgdlGiven   :1;
    unsigned  BSIM4v2lckappasGiven   :1;
    unsigned  BSIM4v2lckappadGiven   :1;
    unsigned  BSIM4v2lcfGiven   :1;
    unsigned  BSIM4v2lclcGiven   :1;
    unsigned  BSIM4v2lcleGiven   :1;
    unsigned  BSIM4v2lvfbcvGiven   :1;
    unsigned  BSIM4v2lnoffGiven   :1;
    unsigned  BSIM4v2lvoffcvGiven :1;
    unsigned  BSIM4v2lacdeGiven   :1;
    unsigned  BSIM4v2lmoinGiven   :1;

    /* Width dependence */
    unsigned  BSIM4v2wcdscGiven   :1;
    unsigned  BSIM4v2wcdscbGiven   :1;
    unsigned  BSIM4v2wcdscdGiven   :1;
    unsigned  BSIM4v2wcitGiven   :1;
    unsigned  BSIM4v2wnfactorGiven   :1;
    unsigned  BSIM4v2wxjGiven   :1;
    unsigned  BSIM4v2wvsatGiven   :1;
    unsigned  BSIM4v2watGiven   :1;
    unsigned  BSIM4v2wa0Given   :1;
    unsigned  BSIM4v2wagsGiven   :1;
    unsigned  BSIM4v2wa1Given   :1;
    unsigned  BSIM4v2wa2Given   :1;
    unsigned  BSIM4v2wketaGiven   :1;    
    unsigned  BSIM4v2wnsubGiven   :1;
    unsigned  BSIM4v2wndepGiven   :1;
    unsigned  BSIM4v2wnsdGiven    :1;
    unsigned  BSIM4v2wphinGiven   :1;
    unsigned  BSIM4v2wngateGiven   :1;
    unsigned  BSIM4v2wgamma1Given   :1;
    unsigned  BSIM4v2wgamma2Given   :1;
    unsigned  BSIM4v2wvbxGiven   :1;
    unsigned  BSIM4v2wvbmGiven   :1;
    unsigned  BSIM4v2wxtGiven   :1;
    unsigned  BSIM4v2wk1Given   :1;
    unsigned  BSIM4v2wkt1Given   :1;
    unsigned  BSIM4v2wkt1lGiven   :1;
    unsigned  BSIM4v2wkt2Given   :1;
    unsigned  BSIM4v2wk2Given   :1;
    unsigned  BSIM4v2wk3Given   :1;
    unsigned  BSIM4v2wk3bGiven   :1;
    unsigned  BSIM4v2ww0Given   :1;
    unsigned  BSIM4v2wdvtp0Given :1;
    unsigned  BSIM4v2wdvtp1Given :1;
    unsigned  BSIM4v2wlpe0Given   :1;
    unsigned  BSIM4v2wlpebGiven   :1;
    unsigned  BSIM4v2wdvt0Given   :1;   
    unsigned  BSIM4v2wdvt1Given   :1;     
    unsigned  BSIM4v2wdvt2Given   :1;     
    unsigned  BSIM4v2wdvt0wGiven   :1;   
    unsigned  BSIM4v2wdvt1wGiven   :1;     
    unsigned  BSIM4v2wdvt2wGiven   :1;     
    unsigned  BSIM4v2wdroutGiven   :1;     
    unsigned  BSIM4v2wdsubGiven   :1;     
    unsigned  BSIM4v2wvth0Given   :1;
    unsigned  BSIM4v2wuaGiven   :1;
    unsigned  BSIM4v2wua1Given   :1;
    unsigned  BSIM4v2wubGiven   :1;
    unsigned  BSIM4v2wub1Given   :1;
    unsigned  BSIM4v2wucGiven   :1;
    unsigned  BSIM4v2wuc1Given   :1;
    unsigned  BSIM4v2wu0Given   :1;
    unsigned  BSIM4v2weuGiven   :1;
    unsigned  BSIM4v2wuteGiven   :1;
    unsigned  BSIM4v2wvoffGiven   :1;
    unsigned  BSIM4v2wminvGiven   :1;
    unsigned  BSIM4v2wrdswGiven   :1;      
    unsigned  BSIM4v2wrswGiven   :1;
    unsigned  BSIM4v2wrdwGiven   :1;
    unsigned  BSIM4v2wprwgGiven   :1;      
    unsigned  BSIM4v2wprwbGiven   :1;      
    unsigned  BSIM4v2wprtGiven   :1;      
    unsigned  BSIM4v2weta0Given   :1;    
    unsigned  BSIM4v2wetabGiven   :1;    
    unsigned  BSIM4v2wpclmGiven   :1;   
    unsigned  BSIM4v2wpdibl1Given   :1;   
    unsigned  BSIM4v2wpdibl2Given   :1;  
    unsigned  BSIM4v2wpdiblbGiven   :1;  
    unsigned  BSIM4v2wfproutGiven   :1;
    unsigned  BSIM4v2wpditsGiven    :1;
    unsigned  BSIM4v2wpditsdGiven    :1;
    unsigned  BSIM4v2wpscbe1Given   :1;    
    unsigned  BSIM4v2wpscbe2Given   :1;    
    unsigned  BSIM4v2wpvagGiven   :1;    
    unsigned  BSIM4v2wdeltaGiven  :1;     
    unsigned  BSIM4v2wwrGiven   :1;
    unsigned  BSIM4v2wdwgGiven   :1;
    unsigned  BSIM4v2wdwbGiven   :1;
    unsigned  BSIM4v2wb0Given   :1;
    unsigned  BSIM4v2wb1Given   :1;
    unsigned  BSIM4v2walpha0Given   :1;
    unsigned  BSIM4v2walpha1Given   :1;
    unsigned  BSIM4v2wbeta0Given   :1;
    unsigned  BSIM4v2wvfbGiven   :1;
    unsigned  BSIM4v2wagidlGiven   :1;
    unsigned  BSIM4v2wbgidlGiven   :1;
    unsigned  BSIM4v2wcgidlGiven   :1;
    unsigned  BSIM4v2wegidlGiven   :1;
    unsigned  BSIM4v2waigcGiven   :1;
    unsigned  BSIM4v2wbigcGiven   :1;
    unsigned  BSIM4v2wcigcGiven   :1;
    unsigned  BSIM4v2waigsdGiven   :1;
    unsigned  BSIM4v2wbigsdGiven   :1;
    unsigned  BSIM4v2wcigsdGiven   :1;
    unsigned  BSIM4v2waigbaccGiven   :1;
    unsigned  BSIM4v2wbigbaccGiven   :1;
    unsigned  BSIM4v2wcigbaccGiven   :1;
    unsigned  BSIM4v2waigbinvGiven   :1;
    unsigned  BSIM4v2wbigbinvGiven   :1;
    unsigned  BSIM4v2wcigbinvGiven   :1;
    unsigned  BSIM4v2wnigcGiven   :1;
    unsigned  BSIM4v2wnigbinvGiven   :1;
    unsigned  BSIM4v2wnigbaccGiven   :1;
    unsigned  BSIM4v2wntoxGiven   :1;
    unsigned  BSIM4v2weigbinvGiven   :1;
    unsigned  BSIM4v2wpigcdGiven   :1;
    unsigned  BSIM4v2wpoxedgeGiven   :1;
    unsigned  BSIM4v2wxrcrg1Given   :1;
    unsigned  BSIM4v2wxrcrg2Given   :1;

    /* CV model */
    unsigned  BSIM4v2wcgslGiven   :1;
    unsigned  BSIM4v2wcgdlGiven   :1;
    unsigned  BSIM4v2wckappasGiven   :1;
    unsigned  BSIM4v2wckappadGiven   :1;
    unsigned  BSIM4v2wcfGiven   :1;
    unsigned  BSIM4v2wclcGiven   :1;
    unsigned  BSIM4v2wcleGiven   :1;
    unsigned  BSIM4v2wvfbcvGiven   :1;
    unsigned  BSIM4v2wnoffGiven   :1;
    unsigned  BSIM4v2wvoffcvGiven :1;
    unsigned  BSIM4v2wacdeGiven   :1;
    unsigned  BSIM4v2wmoinGiven   :1;

    /* Cross-term dependence */
    unsigned  BSIM4v2pcdscGiven   :1;
    unsigned  BSIM4v2pcdscbGiven   :1;
    unsigned  BSIM4v2pcdscdGiven   :1;
    unsigned  BSIM4v2pcitGiven   :1;
    unsigned  BSIM4v2pnfactorGiven   :1;
    unsigned  BSIM4v2pxjGiven   :1;
    unsigned  BSIM4v2pvsatGiven   :1;
    unsigned  BSIM4v2patGiven   :1;
    unsigned  BSIM4v2pa0Given   :1;
    unsigned  BSIM4v2pagsGiven   :1;
    unsigned  BSIM4v2pa1Given   :1;
    unsigned  BSIM4v2pa2Given   :1;
    unsigned  BSIM4v2pketaGiven   :1;    
    unsigned  BSIM4v2pnsubGiven   :1;
    unsigned  BSIM4v2pndepGiven   :1;
    unsigned  BSIM4v2pnsdGiven    :1;
    unsigned  BSIM4v2pphinGiven   :1;
    unsigned  BSIM4v2pngateGiven   :1;
    unsigned  BSIM4v2pgamma1Given   :1;
    unsigned  BSIM4v2pgamma2Given   :1;
    unsigned  BSIM4v2pvbxGiven   :1;
    unsigned  BSIM4v2pvbmGiven   :1;
    unsigned  BSIM4v2pxtGiven   :1;
    unsigned  BSIM4v2pk1Given   :1;
    unsigned  BSIM4v2pkt1Given   :1;
    unsigned  BSIM4v2pkt1lGiven   :1;
    unsigned  BSIM4v2pkt2Given   :1;
    unsigned  BSIM4v2pk2Given   :1;
    unsigned  BSIM4v2pk3Given   :1;
    unsigned  BSIM4v2pk3bGiven   :1;
    unsigned  BSIM4v2pw0Given   :1;
    unsigned  BSIM4v2pdvtp0Given :1;
    unsigned  BSIM4v2pdvtp1Given :1;
    unsigned  BSIM4v2plpe0Given   :1;
    unsigned  BSIM4v2plpebGiven   :1;
    unsigned  BSIM4v2pdvt0Given   :1;   
    unsigned  BSIM4v2pdvt1Given   :1;     
    unsigned  BSIM4v2pdvt2Given   :1;     
    unsigned  BSIM4v2pdvt0wGiven   :1;   
    unsigned  BSIM4v2pdvt1wGiven   :1;     
    unsigned  BSIM4v2pdvt2wGiven   :1;     
    unsigned  BSIM4v2pdroutGiven   :1;     
    unsigned  BSIM4v2pdsubGiven   :1;     
    unsigned  BSIM4v2pvth0Given   :1;
    unsigned  BSIM4v2puaGiven   :1;
    unsigned  BSIM4v2pua1Given   :1;
    unsigned  BSIM4v2pubGiven   :1;
    unsigned  BSIM4v2pub1Given   :1;
    unsigned  BSIM4v2pucGiven   :1;
    unsigned  BSIM4v2puc1Given   :1;
    unsigned  BSIM4v2pu0Given   :1;
    unsigned  BSIM4v2peuGiven   :1;
    unsigned  BSIM4v2puteGiven   :1;
    unsigned  BSIM4v2pvoffGiven   :1;
    unsigned  BSIM4v2pminvGiven   :1;
    unsigned  BSIM4v2prdswGiven   :1;      
    unsigned  BSIM4v2prswGiven   :1;
    unsigned  BSIM4v2prdwGiven   :1;
    unsigned  BSIM4v2pprwgGiven   :1;      
    unsigned  BSIM4v2pprwbGiven   :1;      
    unsigned  BSIM4v2pprtGiven   :1;      
    unsigned  BSIM4v2peta0Given   :1;    
    unsigned  BSIM4v2petabGiven   :1;    
    unsigned  BSIM4v2ppclmGiven   :1;   
    unsigned  BSIM4v2ppdibl1Given   :1;   
    unsigned  BSIM4v2ppdibl2Given   :1;  
    unsigned  BSIM4v2ppdiblbGiven   :1;  
    unsigned  BSIM4v2pfproutGiven   :1;
    unsigned  BSIM4v2ppditsGiven    :1;
    unsigned  BSIM4v2ppditsdGiven    :1;
    unsigned  BSIM4v2ppscbe1Given   :1;    
    unsigned  BSIM4v2ppscbe2Given   :1;    
    unsigned  BSIM4v2ppvagGiven   :1;    
    unsigned  BSIM4v2pdeltaGiven  :1;     
    unsigned  BSIM4v2pwrGiven   :1;
    unsigned  BSIM4v2pdwgGiven   :1;
    unsigned  BSIM4v2pdwbGiven   :1;
    unsigned  BSIM4v2pb0Given   :1;
    unsigned  BSIM4v2pb1Given   :1;
    unsigned  BSIM4v2palpha0Given   :1;
    unsigned  BSIM4v2palpha1Given   :1;
    unsigned  BSIM4v2pbeta0Given   :1;
    unsigned  BSIM4v2pvfbGiven   :1;
    unsigned  BSIM4v2pagidlGiven   :1;
    unsigned  BSIM4v2pbgidlGiven   :1;
    unsigned  BSIM4v2pcgidlGiven   :1;
    unsigned  BSIM4v2pegidlGiven   :1;
    unsigned  BSIM4v2paigcGiven   :1;
    unsigned  BSIM4v2pbigcGiven   :1;
    unsigned  BSIM4v2pcigcGiven   :1;
    unsigned  BSIM4v2paigsdGiven   :1;
    unsigned  BSIM4v2pbigsdGiven   :1;
    unsigned  BSIM4v2pcigsdGiven   :1;
    unsigned  BSIM4v2paigbaccGiven   :1;
    unsigned  BSIM4v2pbigbaccGiven   :1;
    unsigned  BSIM4v2pcigbaccGiven   :1;
    unsigned  BSIM4v2paigbinvGiven   :1;
    unsigned  BSIM4v2pbigbinvGiven   :1;
    unsigned  BSIM4v2pcigbinvGiven   :1;
    unsigned  BSIM4v2pnigcGiven   :1;
    unsigned  BSIM4v2pnigbinvGiven   :1;
    unsigned  BSIM4v2pnigbaccGiven   :1;
    unsigned  BSIM4v2pntoxGiven   :1;
    unsigned  BSIM4v2peigbinvGiven   :1;
    unsigned  BSIM4v2ppigcdGiven   :1;
    unsigned  BSIM4v2ppoxedgeGiven   :1;
    unsigned  BSIM4v2pxrcrg1Given   :1;
    unsigned  BSIM4v2pxrcrg2Given   :1;

    /* CV model */
    unsigned  BSIM4v2pcgslGiven   :1;
    unsigned  BSIM4v2pcgdlGiven   :1;
    unsigned  BSIM4v2pckappasGiven   :1;
    unsigned  BSIM4v2pckappadGiven   :1;
    unsigned  BSIM4v2pcfGiven   :1;
    unsigned  BSIM4v2pclcGiven   :1;
    unsigned  BSIM4v2pcleGiven   :1;
    unsigned  BSIM4v2pvfbcvGiven   :1;
    unsigned  BSIM4v2pnoffGiven   :1;
    unsigned  BSIM4v2pvoffcvGiven :1;
    unsigned  BSIM4v2pacdeGiven   :1;
    unsigned  BSIM4v2pmoinGiven   :1;

    unsigned  BSIM4v2useFringeGiven   :1;

    unsigned  BSIM4v2tnomGiven   :1;
    unsigned  BSIM4v2cgsoGiven   :1;
    unsigned  BSIM4v2cgdoGiven   :1;
    unsigned  BSIM4v2cgboGiven   :1;
    unsigned  BSIM4v2xpartGiven   :1;
    unsigned  BSIM4v2sheetResistanceGiven   :1;

    unsigned  BSIM4v2SjctSatCurDensityGiven   :1;
    unsigned  BSIM4v2SjctSidewallSatCurDensityGiven   :1;
    unsigned  BSIM4v2SjctGateSidewallSatCurDensityGiven   :1;
    unsigned  BSIM4v2SbulkJctPotentialGiven   :1;
    unsigned  BSIM4v2SbulkJctBotGradingCoeffGiven   :1;
    unsigned  BSIM4v2SsidewallJctPotentialGiven   :1;
    unsigned  BSIM4v2SGatesidewallJctPotentialGiven   :1;
    unsigned  BSIM4v2SbulkJctSideGradingCoeffGiven   :1;
    unsigned  BSIM4v2SunitAreaJctCapGiven   :1;
    unsigned  BSIM4v2SunitLengthSidewallJctCapGiven   :1;
    unsigned  BSIM4v2SbulkJctGateSideGradingCoeffGiven   :1;
    unsigned  BSIM4v2SunitLengthGateSidewallJctCapGiven   :1;
    unsigned  BSIM4v2SjctEmissionCoeffGiven :1;
    unsigned  BSIM4v2SjctTempExponentGiven	:1;

    unsigned  BSIM4v2DjctSatCurDensityGiven   :1;
    unsigned  BSIM4v2DjctSidewallSatCurDensityGiven   :1;
    unsigned  BSIM4v2DjctGateSidewallSatCurDensityGiven   :1;
    unsigned  BSIM4v2DbulkJctPotentialGiven   :1;
    unsigned  BSIM4v2DbulkJctBotGradingCoeffGiven   :1;
    unsigned  BSIM4v2DsidewallJctPotentialGiven   :1;
    unsigned  BSIM4v2DGatesidewallJctPotentialGiven   :1;
    unsigned  BSIM4v2DbulkJctSideGradingCoeffGiven   :1;
    unsigned  BSIM4v2DunitAreaJctCapGiven   :1;
    unsigned  BSIM4v2DunitLengthSidewallJctCapGiven   :1;
    unsigned  BSIM4v2DbulkJctGateSideGradingCoeffGiven   :1;
    unsigned  BSIM4v2DunitLengthGateSidewallJctCapGiven   :1;
    unsigned  BSIM4v2DjctEmissionCoeffGiven :1;
    unsigned  BSIM4v2DjctTempExponentGiven :1;

    unsigned  BSIM4v2oxideTrapDensityAGiven  :1;         
    unsigned  BSIM4v2oxideTrapDensityBGiven  :1;        
    unsigned  BSIM4v2oxideTrapDensityCGiven  :1;     
    unsigned  BSIM4v2emGiven  :1;     
    unsigned  BSIM4v2efGiven  :1;     
    unsigned  BSIM4v2afGiven  :1;     
    unsigned  BSIM4v2kfGiven  :1;     

    unsigned  BSIM4v2LintGiven   :1;
    unsigned  BSIM4v2LlGiven   :1;
    unsigned  BSIM4v2LlcGiven   :1;
    unsigned  BSIM4v2LlnGiven   :1;
    unsigned  BSIM4v2LwGiven   :1;
    unsigned  BSIM4v2LwcGiven   :1;
    unsigned  BSIM4v2LwnGiven   :1;
    unsigned  BSIM4v2LwlGiven   :1;
    unsigned  BSIM4v2LwlcGiven   :1;
    unsigned  BSIM4v2LminGiven   :1;
    unsigned  BSIM4v2LmaxGiven   :1;

    unsigned  BSIM4v2WintGiven   :1;
    unsigned  BSIM4v2WlGiven   :1;
    unsigned  BSIM4v2WlcGiven   :1;
    unsigned  BSIM4v2WlnGiven   :1;
    unsigned  BSIM4v2WwGiven   :1;
    unsigned  BSIM4v2WwcGiven   :1;
    unsigned  BSIM4v2WwnGiven   :1;
    unsigned  BSIM4v2WwlGiven   :1;
    unsigned  BSIM4v2WwlcGiven   :1;
    unsigned  BSIM4v2WminGiven   :1;
    unsigned  BSIM4v2WmaxGiven   :1;

} BSIM4v2model;


#ifndef NMOS
#define NMOS 1
#define PMOS -1
#endif /*NMOS*/


/* Instance parameters */
#define BSIM4v2_W                   1
#define BSIM4v2_L                   2
#define BSIM4v2_AS                  3
#define BSIM4v2_AD                  4
#define BSIM4v2_PS                  5
#define BSIM4v2_PD                  6
#define BSIM4v2_NRS                 7
#define BSIM4v2_NRD                 8
#define BSIM4v2_OFF                 9
#define BSIM4v2_IC                  10
#define BSIM4v2_IC_VDS              11
#define BSIM4v2_IC_VGS              12
#define BSIM4v2_IC_VBS              13
#define BSIM4v2_TRNQSMOD            14
#define BSIM4v2_RBODYMOD            15
#define BSIM4v2_RGATEMOD            16
#define BSIM4v2_GEOMOD              17
#define BSIM4v2_RGEOMOD             18
#define BSIM4v2_NF                  19
#define BSIM4v2_MIN                 20 
#define BSIM4v2_ACNQSMOD            22
#define BSIM4v2_RBDB                23
#define BSIM4v2_RBSB                24
#define BSIM4v2_RBPB                25
#define BSIM4v2_RBPS                26
#define BSIM4v2_RBPD                27
#define BSIM4v2_M                   28


/* Global parameters */
#define BSIM4v2_MOD_IGCMOD          90
#define BSIM4v2_MOD_IGBMOD          91
#define BSIM4v2_MOD_ACNQSMOD        92
#define BSIM4v2_MOD_FNOIMOD         93
#define BSIM4v2_MOD_RDSMOD          94
#define BSIM4v2_MOD_DIOMOD          96
#define BSIM4v2_MOD_PERMOD          97
#define BSIM4v2_MOD_GEOMOD          98
#define BSIM4v2_MOD_RGATEMOD        99
#define BSIM4v2_MOD_RBODYMOD        100
#define BSIM4v2_MOD_CAPMOD          101
#define BSIM4v2_MOD_TRNQSMOD        102
#define BSIM4v2_MOD_MOBMOD          103    
#define BSIM4v2_MOD_TNOIMOD         104    
#define BSIM4v2_MOD_TOXE            105
#define BSIM4v2_MOD_CDSC            106
#define BSIM4v2_MOD_CDSCB           107
#define BSIM4v2_MOD_CIT             108
#define BSIM4v2_MOD_NFACTOR         109
#define BSIM4v2_MOD_XJ              110
#define BSIM4v2_MOD_VSAT            111
#define BSIM4v2_MOD_AT              112
#define BSIM4v2_MOD_A0              113
#define BSIM4v2_MOD_A1              114
#define BSIM4v2_MOD_A2              115
#define BSIM4v2_MOD_KETA            116   
#define BSIM4v2_MOD_NSUB            117
#define BSIM4v2_MOD_NDEP            118
#define BSIM4v2_MOD_NGATE           120
#define BSIM4v2_MOD_GAMMA1          121
#define BSIM4v2_MOD_GAMMA2          122
#define BSIM4v2_MOD_VBX             123
#define BSIM4v2_MOD_BINUNIT         124    
#define BSIM4v2_MOD_VBM             125
#define BSIM4v2_MOD_XT              126
#define BSIM4v2_MOD_K1              129
#define BSIM4v2_MOD_KT1             130
#define BSIM4v2_MOD_KT1L            131
#define BSIM4v2_MOD_K2              132
#define BSIM4v2_MOD_KT2             133
#define BSIM4v2_MOD_K3              134
#define BSIM4v2_MOD_K3B             135
#define BSIM4v2_MOD_W0              136
#define BSIM4v2_MOD_LPE0            137
#define BSIM4v2_MOD_DVT0            138
#define BSIM4v2_MOD_DVT1            139
#define BSIM4v2_MOD_DVT2            140
#define BSIM4v2_MOD_DVT0W           141
#define BSIM4v2_MOD_DVT1W           142
#define BSIM4v2_MOD_DVT2W           143
#define BSIM4v2_MOD_DROUT           144
#define BSIM4v2_MOD_DSUB            145
#define BSIM4v2_MOD_VTH0            146
#define BSIM4v2_MOD_UA              147
#define BSIM4v2_MOD_UA1             148
#define BSIM4v2_MOD_UB              149
#define BSIM4v2_MOD_UB1             150
#define BSIM4v2_MOD_UC              151
#define BSIM4v2_MOD_UC1             152
#define BSIM4v2_MOD_U0              153
#define BSIM4v2_MOD_UTE             154
#define BSIM4v2_MOD_VOFF            155
#define BSIM4v2_MOD_DELTA           156
#define BSIM4v2_MOD_RDSW            157
#define BSIM4v2_MOD_PRT             158
#define BSIM4v2_MOD_LDD             159
#define BSIM4v2_MOD_ETA             160
#define BSIM4v2_MOD_ETA0            161
#define BSIM4v2_MOD_ETAB            162
#define BSIM4v2_MOD_PCLM            163
#define BSIM4v2_MOD_PDIBL1          164
#define BSIM4v2_MOD_PDIBL2          165
#define BSIM4v2_MOD_PSCBE1          166
#define BSIM4v2_MOD_PSCBE2          167
#define BSIM4v2_MOD_PVAG            168
#define BSIM4v2_MOD_WR              169
#define BSIM4v2_MOD_DWG             170
#define BSIM4v2_MOD_DWB             171
#define BSIM4v2_MOD_B0              172
#define BSIM4v2_MOD_B1              173
#define BSIM4v2_MOD_ALPHA0          174
#define BSIM4v2_MOD_BETA0           175
#define BSIM4v2_MOD_PDIBLB          178
#define BSIM4v2_MOD_PRWG            179
#define BSIM4v2_MOD_PRWB            180
#define BSIM4v2_MOD_CDSCD           181
#define BSIM4v2_MOD_AGS             182
#define BSIM4v2_MOD_FRINGE          184
#define BSIM4v2_MOD_CGSL            186
#define BSIM4v2_MOD_CGDL            187
#define BSIM4v2_MOD_CKAPPAS         188
#define BSIM4v2_MOD_CF              189
#define BSIM4v2_MOD_CLC             190
#define BSIM4v2_MOD_CLE             191
#define BSIM4v2_MOD_PARAMCHK        192
#define BSIM4v2_MOD_VERSION         193
#define BSIM4v2_MOD_VFBCV           194
#define BSIM4v2_MOD_ACDE            195
#define BSIM4v2_MOD_MOIN            196
#define BSIM4v2_MOD_NOFF            197
#define BSIM4v2_MOD_IJTHDFWD        198
#define BSIM4v2_MOD_ALPHA1          199
#define BSIM4v2_MOD_VFB             200
#define BSIM4v2_MOD_TOXM            201
#define BSIM4v2_MOD_TCJ             202
#define BSIM4v2_MOD_TCJSW           203
#define BSIM4v2_MOD_TCJSWG          204
#define BSIM4v2_MOD_TPB             205
#define BSIM4v2_MOD_TPBSW           206
#define BSIM4v2_MOD_TPBSWG          207
#define BSIM4v2_MOD_VOFFCV          208
#define BSIM4v2_MOD_GBMIN           209
#define BSIM4v2_MOD_RBDB            210
#define BSIM4v2_MOD_RBSB            211
#define BSIM4v2_MOD_RBPB            212
#define BSIM4v2_MOD_RBPS            213
#define BSIM4v2_MOD_RBPD            214
#define BSIM4v2_MOD_DMCG            215
#define BSIM4v2_MOD_DMCI            216
#define BSIM4v2_MOD_DMDG            217
#define BSIM4v2_MOD_XGW             218
#define BSIM4v2_MOD_XGL             219
#define BSIM4v2_MOD_RSHG            220
#define BSIM4v2_MOD_NGCON           221
#define BSIM4v2_MOD_AGIDL           222
#define BSIM4v2_MOD_BGIDL           223
#define BSIM4v2_MOD_EGIDL           224
#define BSIM4v2_MOD_IJTHSFWD        225
#define BSIM4v2_MOD_XJBVD           226
#define BSIM4v2_MOD_XJBVS           227
#define BSIM4v2_MOD_BVD             228
#define BSIM4v2_MOD_BVS             229
#define BSIM4v2_MOD_TOXP            230
#define BSIM4v2_MOD_DTOX            231
#define BSIM4v2_MOD_XRCRG1          232
#define BSIM4v2_MOD_XRCRG2          233
#define BSIM4v2_MOD_EU              234
#define BSIM4v2_MOD_IJTHSREV        235
#define BSIM4v2_MOD_IJTHDREV        236
#define BSIM4v2_MOD_MINV            237
#define BSIM4v2_MOD_VOFFL           238
#define BSIM4v2_MOD_PDITS           239
#define BSIM4v2_MOD_PDITSD          240
#define BSIM4v2_MOD_PDITSL          241
#define BSIM4v2_MOD_TNOIA           242
#define BSIM4v2_MOD_TNOIB           243
#define BSIM4v2_MOD_NTNOI           244
#define BSIM4v2_MOD_FPROUT          245
#define BSIM4v2_MOD_LPEB            246
#define BSIM4v2_MOD_DVTP0           247
#define BSIM4v2_MOD_DVTP1           248
#define BSIM4v2_MOD_CGIDL           249
#define BSIM4v2_MOD_PHIN            250
#define BSIM4v2_MOD_RDSWMIN         251
#define BSIM4v2_MOD_RSW             252
#define BSIM4v2_MOD_RDW             253
#define BSIM4v2_MOD_RDWMIN          254
#define BSIM4v2_MOD_RSWMIN          255
#define BSIM4v2_MOD_NSD             256
#define BSIM4v2_MOD_CKAPPAD         257
#define BSIM4v2_MOD_DMCGT           258
#define BSIM4v2_MOD_AIGC            259
#define BSIM4v2_MOD_BIGC            260
#define BSIM4v2_MOD_CIGC            261
#define BSIM4v2_MOD_AIGBACC         262
#define BSIM4v2_MOD_BIGBACC         263
#define BSIM4v2_MOD_CIGBACC         264            
#define BSIM4v2_MOD_AIGBINV         265
#define BSIM4v2_MOD_BIGBINV         266
#define BSIM4v2_MOD_CIGBINV         267
#define BSIM4v2_MOD_NIGC            268
#define BSIM4v2_MOD_NIGBACC         269
#define BSIM4v2_MOD_NIGBINV         270
#define BSIM4v2_MOD_NTOX            271
#define BSIM4v2_MOD_TOXREF          272
#define BSIM4v2_MOD_EIGBINV         273
#define BSIM4v2_MOD_PIGCD           274
#define BSIM4v2_MOD_POXEDGE         275
#define BSIM4v2_MOD_EPSROX          276
#define BSIM4v2_MOD_AIGSD           277
#define BSIM4v2_MOD_BIGSD           278
#define BSIM4v2_MOD_CIGSD           279
#define BSIM4v2_MOD_JSWGS           280
#define BSIM4v2_MOD_JSWGD           281


/* Length dependence */
#define BSIM4v2_MOD_LCDSC            301
#define BSIM4v2_MOD_LCDSCB           302
#define BSIM4v2_MOD_LCIT             303
#define BSIM4v2_MOD_LNFACTOR         304
#define BSIM4v2_MOD_LXJ              305
#define BSIM4v2_MOD_LVSAT            306
#define BSIM4v2_MOD_LAT              307
#define BSIM4v2_MOD_LA0              308
#define BSIM4v2_MOD_LA1              309
#define BSIM4v2_MOD_LA2              310
#define BSIM4v2_MOD_LKETA            311
#define BSIM4v2_MOD_LNSUB            312
#define BSIM4v2_MOD_LNDEP            313
#define BSIM4v2_MOD_LNGATE           315
#define BSIM4v2_MOD_LGAMMA1          316
#define BSIM4v2_MOD_LGAMMA2          317
#define BSIM4v2_MOD_LVBX             318
#define BSIM4v2_MOD_LVBM             320
#define BSIM4v2_MOD_LXT              322
#define BSIM4v2_MOD_LK1              325
#define BSIM4v2_MOD_LKT1             326
#define BSIM4v2_MOD_LKT1L            327
#define BSIM4v2_MOD_LK2              328
#define BSIM4v2_MOD_LKT2             329
#define BSIM4v2_MOD_LK3              330
#define BSIM4v2_MOD_LK3B             331
#define BSIM4v2_MOD_LW0              332
#define BSIM4v2_MOD_LLPE0            333
#define BSIM4v2_MOD_LDVT0            334
#define BSIM4v2_MOD_LDVT1            335
#define BSIM4v2_MOD_LDVT2            336
#define BSIM4v2_MOD_LDVT0W           337
#define BSIM4v2_MOD_LDVT1W           338
#define BSIM4v2_MOD_LDVT2W           339
#define BSIM4v2_MOD_LDROUT           340
#define BSIM4v2_MOD_LDSUB            341
#define BSIM4v2_MOD_LVTH0            342
#define BSIM4v2_MOD_LUA              343
#define BSIM4v2_MOD_LUA1             344
#define BSIM4v2_MOD_LUB              345
#define BSIM4v2_MOD_LUB1             346
#define BSIM4v2_MOD_LUC              347
#define BSIM4v2_MOD_LUC1             348
#define BSIM4v2_MOD_LU0              349
#define BSIM4v2_MOD_LUTE             350
#define BSIM4v2_MOD_LVOFF            351
#define BSIM4v2_MOD_LDELTA           352
#define BSIM4v2_MOD_LRDSW            353
#define BSIM4v2_MOD_LPRT             354
#define BSIM4v2_MOD_LLDD             355
#define BSIM4v2_MOD_LETA             356
#define BSIM4v2_MOD_LETA0            357
#define BSIM4v2_MOD_LETAB            358
#define BSIM4v2_MOD_LPCLM            359
#define BSIM4v2_MOD_LPDIBL1          360
#define BSIM4v2_MOD_LPDIBL2          361
#define BSIM4v2_MOD_LPSCBE1          362
#define BSIM4v2_MOD_LPSCBE2          363
#define BSIM4v2_MOD_LPVAG            364
#define BSIM4v2_MOD_LWR              365
#define BSIM4v2_MOD_LDWG             366
#define BSIM4v2_MOD_LDWB             367
#define BSIM4v2_MOD_LB0              368
#define BSIM4v2_MOD_LB1              369
#define BSIM4v2_MOD_LALPHA0          370
#define BSIM4v2_MOD_LBETA0           371
#define BSIM4v2_MOD_LPDIBLB          374
#define BSIM4v2_MOD_LPRWG            375
#define BSIM4v2_MOD_LPRWB            376
#define BSIM4v2_MOD_LCDSCD           377
#define BSIM4v2_MOD_LAGS             378

#define BSIM4v2_MOD_LFRINGE          381
#define BSIM4v2_MOD_LCGSL            383
#define BSIM4v2_MOD_LCGDL            384
#define BSIM4v2_MOD_LCKAPPAS         385
#define BSIM4v2_MOD_LCF              386
#define BSIM4v2_MOD_LCLC             387
#define BSIM4v2_MOD_LCLE             388
#define BSIM4v2_MOD_LVFBCV           389
#define BSIM4v2_MOD_LACDE            390
#define BSIM4v2_MOD_LMOIN            391
#define BSIM4v2_MOD_LNOFF            392
#define BSIM4v2_MOD_LALPHA1          394
#define BSIM4v2_MOD_LVFB             395
#define BSIM4v2_MOD_LVOFFCV          396
#define BSIM4v2_MOD_LAGIDL           397
#define BSIM4v2_MOD_LBGIDL           398
#define BSIM4v2_MOD_LEGIDL           399
#define BSIM4v2_MOD_LXRCRG1          400
#define BSIM4v2_MOD_LXRCRG2          401
#define BSIM4v2_MOD_LEU              402
#define BSIM4v2_MOD_LMINV            403
#define BSIM4v2_MOD_LPDITS           404
#define BSIM4v2_MOD_LPDITSD          405
#define BSIM4v2_MOD_LFPROUT          406
#define BSIM4v2_MOD_LLPEB            407
#define BSIM4v2_MOD_LDVTP0           408
#define BSIM4v2_MOD_LDVTP1           409
#define BSIM4v2_MOD_LCGIDL           410
#define BSIM4v2_MOD_LPHIN            411
#define BSIM4v2_MOD_LRSW             412
#define BSIM4v2_MOD_LRDW             413
#define BSIM4v2_MOD_LNSD             414
#define BSIM4v2_MOD_LCKAPPAD         415
#define BSIM4v2_MOD_LAIGC            416
#define BSIM4v2_MOD_LBIGC            417
#define BSIM4v2_MOD_LCIGC            418
#define BSIM4v2_MOD_LAIGBACC         419            
#define BSIM4v2_MOD_LBIGBACC         420
#define BSIM4v2_MOD_LCIGBACC         421
#define BSIM4v2_MOD_LAIGBINV         422
#define BSIM4v2_MOD_LBIGBINV         423
#define BSIM4v2_MOD_LCIGBINV         424
#define BSIM4v2_MOD_LNIGC            425
#define BSIM4v2_MOD_LNIGBACC         426
#define BSIM4v2_MOD_LNIGBINV         427
#define BSIM4v2_MOD_LNTOX            428
#define BSIM4v2_MOD_LEIGBINV         429
#define BSIM4v2_MOD_LPIGCD           430
#define BSIM4v2_MOD_LPOXEDGE         431
#define BSIM4v2_MOD_LAIGSD           432
#define BSIM4v2_MOD_LBIGSD           433
#define BSIM4v2_MOD_LCIGSD           434


/* Width dependence */
#define BSIM4v2_MOD_WCDSC            481
#define BSIM4v2_MOD_WCDSCB           482
#define BSIM4v2_MOD_WCIT             483
#define BSIM4v2_MOD_WNFACTOR         484
#define BSIM4v2_MOD_WXJ              485
#define BSIM4v2_MOD_WVSAT            486
#define BSIM4v2_MOD_WAT              487
#define BSIM4v2_MOD_WA0              488
#define BSIM4v2_MOD_WA1              489
#define BSIM4v2_MOD_WA2              490
#define BSIM4v2_MOD_WKETA            491
#define BSIM4v2_MOD_WNSUB            492
#define BSIM4v2_MOD_WNDEP            493
#define BSIM4v2_MOD_WNGATE           495
#define BSIM4v2_MOD_WGAMMA1          496
#define BSIM4v2_MOD_WGAMMA2          497
#define BSIM4v2_MOD_WVBX             498
#define BSIM4v2_MOD_WVBM             500
#define BSIM4v2_MOD_WXT              502
#define BSIM4v2_MOD_WK1              505
#define BSIM4v2_MOD_WKT1             506
#define BSIM4v2_MOD_WKT1L            507
#define BSIM4v2_MOD_WK2              508
#define BSIM4v2_MOD_WKT2             509
#define BSIM4v2_MOD_WK3              510
#define BSIM4v2_MOD_WK3B             511
#define BSIM4v2_MOD_WW0              512
#define BSIM4v2_MOD_WLPE0            513
#define BSIM4v2_MOD_WDVT0            514
#define BSIM4v2_MOD_WDVT1            515
#define BSIM4v2_MOD_WDVT2            516
#define BSIM4v2_MOD_WDVT0W           517
#define BSIM4v2_MOD_WDVT1W           518
#define BSIM4v2_MOD_WDVT2W           519
#define BSIM4v2_MOD_WDROUT           520
#define BSIM4v2_MOD_WDSUB            521
#define BSIM4v2_MOD_WVTH0            522
#define BSIM4v2_MOD_WUA              523
#define BSIM4v2_MOD_WUA1             524
#define BSIM4v2_MOD_WUB              525
#define BSIM4v2_MOD_WUB1             526
#define BSIM4v2_MOD_WUC              527
#define BSIM4v2_MOD_WUC1             528
#define BSIM4v2_MOD_WU0              529
#define BSIM4v2_MOD_WUTE             530
#define BSIM4v2_MOD_WVOFF            531
#define BSIM4v2_MOD_WDELTA           532
#define BSIM4v2_MOD_WRDSW            533
#define BSIM4v2_MOD_WPRT             534
#define BSIM4v2_MOD_WLDD             535
#define BSIM4v2_MOD_WETA             536
#define BSIM4v2_MOD_WETA0            537
#define BSIM4v2_MOD_WETAB            538
#define BSIM4v2_MOD_WPCLM            539
#define BSIM4v2_MOD_WPDIBL1          540
#define BSIM4v2_MOD_WPDIBL2          541
#define BSIM4v2_MOD_WPSCBE1          542
#define BSIM4v2_MOD_WPSCBE2          543
#define BSIM4v2_MOD_WPVAG            544
#define BSIM4v2_MOD_WWR              545
#define BSIM4v2_MOD_WDWG             546
#define BSIM4v2_MOD_WDWB             547
#define BSIM4v2_MOD_WB0              548
#define BSIM4v2_MOD_WB1              549
#define BSIM4v2_MOD_WALPHA0          550
#define BSIM4v2_MOD_WBETA0           551
#define BSIM4v2_MOD_WPDIBLB          554
#define BSIM4v2_MOD_WPRWG            555
#define BSIM4v2_MOD_WPRWB            556
#define BSIM4v2_MOD_WCDSCD           557
#define BSIM4v2_MOD_WAGS             558

#define BSIM4v2_MOD_WFRINGE          561
#define BSIM4v2_MOD_WCGSL            563
#define BSIM4v2_MOD_WCGDL            564
#define BSIM4v2_MOD_WCKAPPAS         565
#define BSIM4v2_MOD_WCF              566
#define BSIM4v2_MOD_WCLC             567
#define BSIM4v2_MOD_WCLE             568
#define BSIM4v2_MOD_WVFBCV           569
#define BSIM4v2_MOD_WACDE            570
#define BSIM4v2_MOD_WMOIN            571
#define BSIM4v2_MOD_WNOFF            572
#define BSIM4v2_MOD_WALPHA1          574
#define BSIM4v2_MOD_WVFB             575
#define BSIM4v2_MOD_WVOFFCV          576
#define BSIM4v2_MOD_WAGIDL           577
#define BSIM4v2_MOD_WBGIDL           578
#define BSIM4v2_MOD_WEGIDL           579
#define BSIM4v2_MOD_WXRCRG1          580
#define BSIM4v2_MOD_WXRCRG2          581
#define BSIM4v2_MOD_WEU              582
#define BSIM4v2_MOD_WMINV            583
#define BSIM4v2_MOD_WPDITS           584
#define BSIM4v2_MOD_WPDITSD          585
#define BSIM4v2_MOD_WFPROUT          586
#define BSIM4v2_MOD_WLPEB            587
#define BSIM4v2_MOD_WDVTP0           588
#define BSIM4v2_MOD_WDVTP1           589
#define BSIM4v2_MOD_WCGIDL           590
#define BSIM4v2_MOD_WPHIN            591
#define BSIM4v2_MOD_WRSW             592
#define BSIM4v2_MOD_WRDW             593
#define BSIM4v2_MOD_WNSD             594
#define BSIM4v2_MOD_WCKAPPAD         595
#define BSIM4v2_MOD_WAIGC            596
#define BSIM4v2_MOD_WBIGC            597
#define BSIM4v2_MOD_WCIGC            598
#define BSIM4v2_MOD_WAIGBACC         599            
#define BSIM4v2_MOD_WBIGBACC         600
#define BSIM4v2_MOD_WCIGBACC         601
#define BSIM4v2_MOD_WAIGBINV         602
#define BSIM4v2_MOD_WBIGBINV         603
#define BSIM4v2_MOD_WCIGBINV         604
#define BSIM4v2_MOD_WNIGC            605
#define BSIM4v2_MOD_WNIGBACC         606
#define BSIM4v2_MOD_WNIGBINV         607
#define BSIM4v2_MOD_WNTOX            608
#define BSIM4v2_MOD_WEIGBINV         609
#define BSIM4v2_MOD_WPIGCD           610
#define BSIM4v2_MOD_WPOXEDGE         611
#define BSIM4v2_MOD_WAIGSD           612
#define BSIM4v2_MOD_WBIGSD           613
#define BSIM4v2_MOD_WCIGSD           614


/* Cross-term dependence */
#define BSIM4v2_MOD_PCDSC            661
#define BSIM4v2_MOD_PCDSCB           662
#define BSIM4v2_MOD_PCIT             663
#define BSIM4v2_MOD_PNFACTOR         664
#define BSIM4v2_MOD_PXJ              665
#define BSIM4v2_MOD_PVSAT            666
#define BSIM4v2_MOD_PAT              667
#define BSIM4v2_MOD_PA0              668
#define BSIM4v2_MOD_PA1              669
#define BSIM4v2_MOD_PA2              670
#define BSIM4v2_MOD_PKETA            671
#define BSIM4v2_MOD_PNSUB            672
#define BSIM4v2_MOD_PNDEP            673
#define BSIM4v2_MOD_PNGATE           675
#define BSIM4v2_MOD_PGAMMA1          676
#define BSIM4v2_MOD_PGAMMA2          677
#define BSIM4v2_MOD_PVBX             678

#define BSIM4v2_MOD_PVBM             680

#define BSIM4v2_MOD_PXT              682
#define BSIM4v2_MOD_PK1              685
#define BSIM4v2_MOD_PKT1             686
#define BSIM4v2_MOD_PKT1L            687
#define BSIM4v2_MOD_PK2              688
#define BSIM4v2_MOD_PKT2             689
#define BSIM4v2_MOD_PK3              690
#define BSIM4v2_MOD_PK3B             691
#define BSIM4v2_MOD_PW0              692
#define BSIM4v2_MOD_PLPE0            693

#define BSIM4v2_MOD_PDVT0            694
#define BSIM4v2_MOD_PDVT1            695
#define BSIM4v2_MOD_PDVT2            696

#define BSIM4v2_MOD_PDVT0W           697
#define BSIM4v2_MOD_PDVT1W           698
#define BSIM4v2_MOD_PDVT2W           699

#define BSIM4v2_MOD_PDROUT           700
#define BSIM4v2_MOD_PDSUB            701
#define BSIM4v2_MOD_PVTH0            702
#define BSIM4v2_MOD_PUA              703
#define BSIM4v2_MOD_PUA1             704
#define BSIM4v2_MOD_PUB              705
#define BSIM4v2_MOD_PUB1             706
#define BSIM4v2_MOD_PUC              707
#define BSIM4v2_MOD_PUC1             708
#define BSIM4v2_MOD_PU0              709
#define BSIM4v2_MOD_PUTE             710
#define BSIM4v2_MOD_PVOFF            711
#define BSIM4v2_MOD_PDELTA           712
#define BSIM4v2_MOD_PRDSW            713
#define BSIM4v2_MOD_PPRT             714
#define BSIM4v2_MOD_PLDD             715
#define BSIM4v2_MOD_PETA             716
#define BSIM4v2_MOD_PETA0            717
#define BSIM4v2_MOD_PETAB            718
#define BSIM4v2_MOD_PPCLM            719
#define BSIM4v2_MOD_PPDIBL1          720
#define BSIM4v2_MOD_PPDIBL2          721
#define BSIM4v2_MOD_PPSCBE1          722
#define BSIM4v2_MOD_PPSCBE2          723
#define BSIM4v2_MOD_PPVAG            724
#define BSIM4v2_MOD_PWR              725
#define BSIM4v2_MOD_PDWG             726
#define BSIM4v2_MOD_PDWB             727
#define BSIM4v2_MOD_PB0              728
#define BSIM4v2_MOD_PB1              729
#define BSIM4v2_MOD_PALPHA0          730
#define BSIM4v2_MOD_PBETA0           731
#define BSIM4v2_MOD_PPDIBLB          734

#define BSIM4v2_MOD_PPRWG            735
#define BSIM4v2_MOD_PPRWB            736

#define BSIM4v2_MOD_PCDSCD           737
#define BSIM4v2_MOD_PAGS             738

#define BSIM4v2_MOD_PFRINGE          741
#define BSIM4v2_MOD_PCGSL            743
#define BSIM4v2_MOD_PCGDL            744
#define BSIM4v2_MOD_PCKAPPAS         745
#define BSIM4v2_MOD_PCF              746
#define BSIM4v2_MOD_PCLC             747
#define BSIM4v2_MOD_PCLE             748
#define BSIM4v2_MOD_PVFBCV           749
#define BSIM4v2_MOD_PACDE            750
#define BSIM4v2_MOD_PMOIN            751
#define BSIM4v2_MOD_PNOFF            752
#define BSIM4v2_MOD_PALPHA1          754
#define BSIM4v2_MOD_PVFB             755
#define BSIM4v2_MOD_PVOFFCV          756
#define BSIM4v2_MOD_PAGIDL           757
#define BSIM4v2_MOD_PBGIDL           758
#define BSIM4v2_MOD_PEGIDL           759
#define BSIM4v2_MOD_PXRCRG1          760
#define BSIM4v2_MOD_PXRCRG2          761
#define BSIM4v2_MOD_PEU              762
#define BSIM4v2_MOD_PMINV            763
#define BSIM4v2_MOD_PPDITS           764
#define BSIM4v2_MOD_PPDITSD          765
#define BSIM4v2_MOD_PFPROUT          766
#define BSIM4v2_MOD_PLPEB            767
#define BSIM4v2_MOD_PDVTP0           768
#define BSIM4v2_MOD_PDVTP1           769
#define BSIM4v2_MOD_PCGIDL           770
#define BSIM4v2_MOD_PPHIN            771
#define BSIM4v2_MOD_PRSW             772
#define BSIM4v2_MOD_PRDW             773
#define BSIM4v2_MOD_PNSD             774
#define BSIM4v2_MOD_PCKAPPAD         775
#define BSIM4v2_MOD_PAIGC            776
#define BSIM4v2_MOD_PBIGC            777
#define BSIM4v2_MOD_PCIGC            778
#define BSIM4v2_MOD_PAIGBACC         779            
#define BSIM4v2_MOD_PBIGBACC         780
#define BSIM4v2_MOD_PCIGBACC         781
#define BSIM4v2_MOD_PAIGBINV         782
#define BSIM4v2_MOD_PBIGBINV         783
#define BSIM4v2_MOD_PCIGBINV         784
#define BSIM4v2_MOD_PNIGC            785
#define BSIM4v2_MOD_PNIGBACC         786
#define BSIM4v2_MOD_PNIGBINV         787
#define BSIM4v2_MOD_PNTOX            788
#define BSIM4v2_MOD_PEIGBINV         789
#define BSIM4v2_MOD_PPIGCD           790
#define BSIM4v2_MOD_PPOXEDGE         791
#define BSIM4v2_MOD_PAIGSD           792
#define BSIM4v2_MOD_PBIGSD           793
#define BSIM4v2_MOD_PCIGSD           794

#define BSIM4v2_MOD_TNOM             831
#define BSIM4v2_MOD_CGSO             832
#define BSIM4v2_MOD_CGDO             833
#define BSIM4v2_MOD_CGBO             834
#define BSIM4v2_MOD_XPART            835
#define BSIM4v2_MOD_RSH              836
#define BSIM4v2_MOD_JSS              837
#define BSIM4v2_MOD_PBS              838
#define BSIM4v2_MOD_MJS              839
#define BSIM4v2_MOD_PBSWS            840
#define BSIM4v2_MOD_MJSWS            841
#define BSIM4v2_MOD_CJS              842
#define BSIM4v2_MOD_CJSWS            843
#define BSIM4v2_MOD_NMOS             844
#define BSIM4v2_MOD_PMOS             845
#define BSIM4v2_MOD_NOIA             846
#define BSIM4v2_MOD_NOIB             847
#define BSIM4v2_MOD_NOIC             848
#define BSIM4v2_MOD_LINT             849
#define BSIM4v2_MOD_LL               850
#define BSIM4v2_MOD_LLN              851
#define BSIM4v2_MOD_LW               852
#define BSIM4v2_MOD_LWN              853
#define BSIM4v2_MOD_LWL              854
#define BSIM4v2_MOD_LMIN             855
#define BSIM4v2_MOD_LMAX             856
#define BSIM4v2_MOD_WINT             857
#define BSIM4v2_MOD_WL               858
#define BSIM4v2_MOD_WLN              859
#define BSIM4v2_MOD_WW               860
#define BSIM4v2_MOD_WWN              861
#define BSIM4v2_MOD_WWL              862
#define BSIM4v2_MOD_WMIN             863
#define BSIM4v2_MOD_WMAX             864
#define BSIM4v2_MOD_DWC              865
#define BSIM4v2_MOD_DLC              866
#define BSIM4v2_MOD_XL               867
#define BSIM4v2_MOD_XW               868
#define BSIM4v2_MOD_EM               869
#define BSIM4v2_MOD_EF               870
#define BSIM4v2_MOD_AF               871
#define BSIM4v2_MOD_KF               872
#define BSIM4v2_MOD_NJS              873
#define BSIM4v2_MOD_XTIS             874
#define BSIM4v2_MOD_PBSWGS           875
#define BSIM4v2_MOD_MJSWGS           876
#define BSIM4v2_MOD_CJSWGS           877
#define BSIM4v2_MOD_JSWS             878
#define BSIM4v2_MOD_LLC              879
#define BSIM4v2_MOD_LWC              880
#define BSIM4v2_MOD_LWLC             881
#define BSIM4v2_MOD_WLC              882
#define BSIM4v2_MOD_WWC              883
#define BSIM4v2_MOD_WWLC             884
#define BSIM4v2_MOD_DWJ              885
#define BSIM4v2_MOD_JSD              886
#define BSIM4v2_MOD_PBD              887
#define BSIM4v2_MOD_MJD              888
#define BSIM4v2_MOD_PBSWD            889
#define BSIM4v2_MOD_MJSWD            890
#define BSIM4v2_MOD_CJD              891
#define BSIM4v2_MOD_CJSWD            892
#define BSIM4v2_MOD_NJD              893
#define BSIM4v2_MOD_XTID             894
#define BSIM4v2_MOD_PBSWGD           895
#define BSIM4v2_MOD_MJSWGD           896
#define BSIM4v2_MOD_CJSWGD           897
#define BSIM4v2_MOD_JSWD             898
#define BSIM4v2_MOD_DLCIG            899

/* device questions */
#define BSIM4v2_DNODE                945
#define BSIM4v2_GNODEEXT             946
#define BSIM4v2_SNODE                947
#define BSIM4v2_BNODE                948
#define BSIM4v2_DNODEPRIME           949
#define BSIM4v2_GNODEPRIME           950
#define BSIM4v2_GNODEMIDE            951
#define BSIM4v2_GNODEMID             952
#define BSIM4v2_SNODEPRIME           953
#define BSIM4v2_BNODEPRIME           954
#define BSIM4v2_DBNODE               955
#define BSIM4v2_SBNODE               956
#define BSIM4v2_VBD                  957
#define BSIM4v2_VBS                  958
#define BSIM4v2_VGS                  959
#define BSIM4v2_VDS                  960
#define BSIM4v2_CD                   961
#define BSIM4v2_CBS                  962
#define BSIM4v2_CBD                  963
#define BSIM4v2_GM                   964
#define BSIM4v2_GDS                  965
#define BSIM4v2_GMBS                 966
#define BSIM4v2_GBD                  967
#define BSIM4v2_GBS                  968
#define BSIM4v2_QB                   969
#define BSIM4v2_CQB                  970
#define BSIM4v2_QG                   971
#define BSIM4v2_CQG                  972
#define BSIM4v2_QD                   973
#define BSIM4v2_CQD                  974
#define BSIM4v2_CGGB                 975
#define BSIM4v2_CGDB                 976
#define BSIM4v2_CGSB                 977
#define BSIM4v2_CBGB                 978
#define BSIM4v2_CAPBD                979
#define BSIM4v2_CQBD                 980
#define BSIM4v2_CAPBS                981
#define BSIM4v2_CQBS                 982
#define BSIM4v2_CDGB                 983
#define BSIM4v2_CDDB                 984
#define BSIM4v2_CDSB                 985
#define BSIM4v2_VON                  986
#define BSIM4v2_VDSAT                987
#define BSIM4v2_QBS                  988
#define BSIM4v2_QBD                  989
#define BSIM4v2_SOURCECONDUCT        990
#define BSIM4v2_DRAINCONDUCT         991
#define BSIM4v2_CBDB                 992
#define BSIM4v2_CBSB                 993
#define BSIM4v2_CSUB		   994
#define BSIM4v2_QINV		   995
#define BSIM4v2_IGIDL		   996
#define BSIM4v2_CSGB                 997
#define BSIM4v2_CSDB                 998
#define BSIM4v2_CSSB                 999
#define BSIM4v2_CGBB                 1000
#define BSIM4v2_CDBB                 1001
#define BSIM4v2_CSBB                 1002
#define BSIM4v2_CBBB                 1003
#define BSIM4v2_QS                   1004
#define BSIM4v2_IGISL		   1005
#define BSIM4v2_IGS		   1006
#define BSIM4v2_IGD		   1007
#define BSIM4v2_IGB		   1008
#define BSIM4v2_IGCS		   1009
#define BSIM4v2_IGCD		   1010


#include "bsim4v2ext.h"

extern void BSIM4v2evaluate(double,double,double,BSIM4v2instance*,BSIM4v2model*,
        double*,double*,double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, CKTcircuit*);
extern int BSIM4v2debug(BSIM4v2model*, BSIM4v2instance*, CKTcircuit*, int);
extern int BSIM4v2checkModel(BSIM4v2model*, BSIM4v2instance*, CKTcircuit*);
extern int BSIM4v2RdseffGeo(double nf, int geo, int rgeo, int minSD, double Weffcj, double Rsh, double DMCG, double DMCI, double DMDG, int Type,double *Rtot);
extern int BSIM4v2PAeffGeo(double nf, int geo, int minSD, double Weffcj, double DMCG, double DMCI, double DMDG, double *Ps, double *Pd, double *As, double *Ad);

#endif /*BSIM4v2*/
