/**********
Copyright 2000 Regents of the University of California.  All rights reserved.
Author: 2000 Weidong Liu.
File: bsim4v0def.h
**********/

#ifndef BSIM4v0
#define BSIM4v0

#include "ngspice/ifsim.h"
#include "ngspice/gendefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/noisedef.h"         

typedef struct sBSIM4v0instance
{
    struct sBSIM4v0model *BSIM4v0modPtr;
    struct sBSIM4v0instance *BSIM4v0nextInstance;
    IFuid BSIM4v0name;
    int BSIM4v0states;     /* index into state table for this device */
    int BSIM4v0dNode;
    int BSIM4v0gNodeExt;
    int BSIM4v0sNode;
    int BSIM4v0bNode;
    int BSIM4v0dNodePrime;
    int BSIM4v0gNodePrime;
    int BSIM4v0gNodeMid;
    int BSIM4v0sNodePrime;
    int BSIM4v0bNodePrime;
    int BSIM4v0dbNode;
    int BSIM4v0sbNode;
    int BSIM4v0qNode;

    double BSIM4v0ueff;
    double BSIM4v0thetavth; 
    double BSIM4v0von;
    double BSIM4v0vdsat;
    double BSIM4v0cgdo;
    double BSIM4v0qgdo;
    double BSIM4v0cgso;
    double BSIM4v0qgso;
    double BSIM4v0grbsb;
    double BSIM4v0grbdb;
    double BSIM4v0grbpb;
    double BSIM4v0grbps;
    double BSIM4v0grbpd;

    double BSIM4v0vjsmFwd;
    double BSIM4v0vjsmRev;
    double BSIM4v0vjdmFwd;
    double BSIM4v0vjdmRev;
    double BSIM4v0XExpBVS;
    double BSIM4v0XExpBVD;
    double BSIM4v0SslpFwd;
    double BSIM4v0SslpRev;
    double BSIM4v0DslpFwd;
    double BSIM4v0DslpRev;
    double BSIM4v0IVjsmFwd;
    double BSIM4v0IVjsmRev;
    double BSIM4v0IVjdmFwd;
    double BSIM4v0IVjdmRev;

    double BSIM4v0grgeltd;
    double BSIM4v0Pseff;
    double BSIM4v0Pdeff;
    double BSIM4v0Aseff;
    double BSIM4v0Adeff;

    double BSIM4v0l;
    double BSIM4v0w;
    double BSIM4v0drainArea;
    double BSIM4v0sourceArea;
    double BSIM4v0drainSquares;
    double BSIM4v0sourceSquares;
    double BSIM4v0drainPerimeter;
    double BSIM4v0sourcePerimeter;
    double BSIM4v0sourceConductance;
    double BSIM4v0drainConductance;
    double BSIM4v0rbdb;
    double BSIM4v0rbsb;
    double BSIM4v0rbpb;
    double BSIM4v0rbps;
    double BSIM4v0rbpd;

    double BSIM4v0icVDS;
    double BSIM4v0icVGS;
    double BSIM4v0icVBS;
    double BSIM4v0nf;
    int BSIM4v0off;
    int BSIM4v0mode;
    int BSIM4v0trnqsMod;
    int BSIM4v0acnqsMod;
    int BSIM4v0rbodyMod;
    int BSIM4v0rgateMod;
    int BSIM4v0geoMod;
    int BSIM4v0rgeoMod;
    int BSIM4v0min;

    /* OP point */
    double BSIM4v0Vgsteff;
    double BSIM4v0Vdseff;
    double BSIM4v0nstar;
    double BSIM4v0Abulk;
    double BSIM4v0EsatL;
    double BSIM4v0AbovVgst2Vtm;
    double BSIM4v0qinv;
    double BSIM4v0cd;
    double BSIM4v0cbs;
    double BSIM4v0cbd;
    double BSIM4v0csub;
    double BSIM4v0Igidl;
    double BSIM4v0gm;
    double BSIM4v0gds;
    double BSIM4v0gmbs;
    double BSIM4v0gbd;
    double BSIM4v0gbs;

    double BSIM4v0gbbs;
    double BSIM4v0gbgs;
    double BSIM4v0gbds;
    double BSIM4v0ggidld;
    double BSIM4v0ggidlg;
    double BSIM4v0ggidls;
    double BSIM4v0ggidlb;

    double BSIM4v0Igcs;
    double BSIM4v0gIgcsg;
    double BSIM4v0gIgcsd;
    double BSIM4v0gIgcss;
    double BSIM4v0gIgcsb;
    double BSIM4v0Igcd;
    double BSIM4v0gIgcdg;
    double BSIM4v0gIgcdd;
    double BSIM4v0gIgcds;
    double BSIM4v0gIgcdb;

    double BSIM4v0Igs;
    double BSIM4v0gIgsg;
    double BSIM4v0gIgss;
    double BSIM4v0Igd;
    double BSIM4v0gIgdg;
    double BSIM4v0gIgdd;

    double BSIM4v0Igb;
    double BSIM4v0gIgbg;
    double BSIM4v0gIgbd;
    double BSIM4v0gIgbs;
    double BSIM4v0gIgbb;

    double BSIM4v0grdsw;
    double BSIM4v0IdovVds;
    double BSIM4v0gcrg;
    double BSIM4v0gcrgd;
    double BSIM4v0gcrgg;
    double BSIM4v0gcrgs;
    double BSIM4v0gcrgb;

    double BSIM4v0gstot;
    double BSIM4v0gstotd;
    double BSIM4v0gstotg;
    double BSIM4v0gstots;
    double BSIM4v0gstotb;

    double BSIM4v0gdtot;
    double BSIM4v0gdtotd;
    double BSIM4v0gdtotg;
    double BSIM4v0gdtots;
    double BSIM4v0gdtotb;

    double BSIM4v0cggb;
    double BSIM4v0cgdb;
    double BSIM4v0cgsb;
    double BSIM4v0cbgb;
    double BSIM4v0cbdb;
    double BSIM4v0cbsb;
    double BSIM4v0cdgb;
    double BSIM4v0cddb;
    double BSIM4v0cdsb;
    double BSIM4v0capbd;
    double BSIM4v0capbs;

    double BSIM4v0cqgb;
    double BSIM4v0cqdb;
    double BSIM4v0cqsb;
    double BSIM4v0cqbb;

    double BSIM4v0qgate;
    double BSIM4v0qbulk;
    double BSIM4v0qdrn;

    double BSIM4v0qchqs;
    double BSIM4v0taunet;
    double BSIM4v0gtau;
    double BSIM4v0gtg;
    double BSIM4v0gtd;
    double BSIM4v0gts;
    double BSIM4v0gtb;

    struct bsim4v0SizeDependParam  *pParam;

    unsigned BSIM4v0lGiven :1;
    unsigned BSIM4v0wGiven :1;
    unsigned BSIM4v0nfGiven :1;
    unsigned BSIM4v0minGiven :1;
    unsigned BSIM4v0drainAreaGiven :1;
    unsigned BSIM4v0sourceAreaGiven    :1;
    unsigned BSIM4v0drainSquaresGiven  :1;
    unsigned BSIM4v0sourceSquaresGiven :1;
    unsigned BSIM4v0drainPerimeterGiven    :1;
    unsigned BSIM4v0sourcePerimeterGiven   :1;
    unsigned BSIM4v0rbdbGiven   :1;
    unsigned BSIM4v0rbsbGiven   :1;
    unsigned BSIM4v0rbpbGiven   :1;
    unsigned BSIM4v0rbpdGiven   :1;
    unsigned BSIM4v0rbpsGiven   :1;
    unsigned BSIM4v0icVDSGiven :1;
    unsigned BSIM4v0icVGSGiven :1;
    unsigned BSIM4v0icVBSGiven :1;
    unsigned BSIM4v0trnqsModGiven :1;
    unsigned BSIM4v0acnqsModGiven :1;
    unsigned BSIM4v0rbodyModGiven :1;
    unsigned BSIM4v0rgateModGiven :1;
    unsigned BSIM4v0geoModGiven :1;
    unsigned BSIM4v0rgeoModGiven :1;

    double *BSIM4v0DPdPtr;
    double *BSIM4v0DPdpPtr;
    double *BSIM4v0DPgpPtr;
    double *BSIM4v0DPgmPtr;
    double *BSIM4v0DPspPtr;
    double *BSIM4v0DPbpPtr;
    double *BSIM4v0DPdbPtr;

    double *BSIM4v0DdPtr;
    double *BSIM4v0DdpPtr;

    double *BSIM4v0GPdpPtr;
    double *BSIM4v0GPgpPtr;
    double *BSIM4v0GPgmPtr;
    double *BSIM4v0GPgePtr;
    double *BSIM4v0GPspPtr;
    double *BSIM4v0GPbpPtr;

    double *BSIM4v0GMdpPtr;
    double *BSIM4v0GMgpPtr;
    double *BSIM4v0GMgmPtr;
    double *BSIM4v0GMgePtr;
    double *BSIM4v0GMspPtr;
    double *BSIM4v0GMbpPtr;

    double *BSIM4v0GEdpPtr;
    double *BSIM4v0GEgpPtr;
    double *BSIM4v0GEgmPtr;
    double *BSIM4v0GEgePtr;
    double *BSIM4v0GEspPtr;
    double *BSIM4v0GEbpPtr;

    double *BSIM4v0SPdpPtr;
    double *BSIM4v0SPgpPtr;
    double *BSIM4v0SPgmPtr;
    double *BSIM4v0SPsPtr;
    double *BSIM4v0SPspPtr;
    double *BSIM4v0SPbpPtr;
    double *BSIM4v0SPsbPtr;

    double *BSIM4v0SspPtr;
    double *BSIM4v0SsPtr;

    double *BSIM4v0BPdpPtr;
    double *BSIM4v0BPgpPtr;
    double *BSIM4v0BPgmPtr;
    double *BSIM4v0BPspPtr;
    double *BSIM4v0BPdbPtr;
    double *BSIM4v0BPbPtr;
    double *BSIM4v0BPsbPtr;
    double *BSIM4v0BPbpPtr;

    double *BSIM4v0DBdpPtr;
    double *BSIM4v0DBdbPtr;
    double *BSIM4v0DBbpPtr;
    double *BSIM4v0DBbPtr;

    double *BSIM4v0SBspPtr;
    double *BSIM4v0SBbpPtr;
    double *BSIM4v0SBbPtr;
    double *BSIM4v0SBsbPtr;

    double *BSIM4v0BdbPtr;
    double *BSIM4v0BbpPtr;
    double *BSIM4v0BsbPtr;
    double *BSIM4v0BbPtr;

    double *BSIM4v0DgpPtr;
    double *BSIM4v0DspPtr;
    double *BSIM4v0DbpPtr;
    double *BSIM4v0SdpPtr;
    double *BSIM4v0SgpPtr;
    double *BSIM4v0SbpPtr;

    double *BSIM4v0QdpPtr;
    double *BSIM4v0QgpPtr;
    double *BSIM4v0QspPtr;
    double *BSIM4v0QbpPtr;
    double *BSIM4v0QqPtr;
    double *BSIM4v0DPqPtr;
    double *BSIM4v0GPqPtr;
    double *BSIM4v0SPqPtr;

#define BSIM4v0vbd BSIM4v0states+ 0
#define BSIM4v0vbs BSIM4v0states+ 1
#define BSIM4v0vgs BSIM4v0states+ 2
#define BSIM4v0vds BSIM4v0states+ 3
#define BSIM4v0vdbs BSIM4v0states+ 4
#define BSIM4v0vdbd BSIM4v0states+ 5
#define BSIM4v0vsbs BSIM4v0states+ 6
#define BSIM4v0vges BSIM4v0states+ 7
#define BSIM4v0vgms BSIM4v0states+ 8
#define BSIM4v0vses BSIM4v0states+ 9
#define BSIM4v0vdes BSIM4v0states+ 10

#define BSIM4v0qb BSIM4v0states+ 11
#define BSIM4v0cqb BSIM4v0states+ 12
#define BSIM4v0qg BSIM4v0states+ 13
#define BSIM4v0cqg BSIM4v0states+ 14
#define BSIM4v0qd BSIM4v0states+ 15
#define BSIM4v0cqd BSIM4v0states+ 16
#define BSIM4v0qgmid BSIM4v0states+ 17
#define BSIM4v0cqgmid BSIM4v0states+ 18

#define BSIM4v0qbs  BSIM4v0states+ 19
#define BSIM4v0cqbs  BSIM4v0states+ 20
#define BSIM4v0qbd  BSIM4v0states+ 21
#define BSIM4v0cqbd  BSIM4v0states+ 22

#define BSIM4v0qcheq BSIM4v0states+ 23
#define BSIM4v0cqcheq BSIM4v0states+ 24
#define BSIM4v0qcdump BSIM4v0states+ 25
#define BSIM4v0cqcdump BSIM4v0states+ 26
#define BSIM4v0qdef BSIM4v0states+ 27

#define BSIM4v0numStates 28


/* indices to the array of BSIM4v0 NOISE SOURCES */

#define BSIM4v0RDNOIZ       0
#define BSIM4v0RSNOIZ       1
#define BSIM4v0RGNOIZ       2
#define BSIM4v0RBPSNOIZ     3
#define BSIM4v0RBPDNOIZ     4
#define BSIM4v0RBPBNOIZ     5
#define BSIM4v0RBSBNOIZ     6
#define BSIM4v0RBDBNOIZ     7
#define BSIM4v0IDNOIZ       8
#define BSIM4v0FLNOIZ       9
#define BSIM4v0IGSNOIZ      10
#define BSIM4v0IGDNOIZ      11
#define BSIM4v0IGBNOIZ      12
#define BSIM4v0TOTNOIZ      13

#define BSIM4v0NSRCS        14  /* Number of BSIM4v0 noise sources */

#ifndef NONOISE
    double BSIM4v0nVar[NSTATVARS][BSIM4v0NSRCS];
#else /* NONOISE */
        double **BSIM4v0nVar;
#endif /* NONOISE */

} BSIM4v0instance ;

struct bsim4v0SizeDependParam
{
    double Width;
    double Length;
    double NFinger;

    double BSIM4v0cdsc;           
    double BSIM4v0cdscb;    
    double BSIM4v0cdscd;       
    double BSIM4v0cit;           
    double BSIM4v0nfactor;      
    double BSIM4v0xj;
    double BSIM4v0vsat;         
    double BSIM4v0at;         
    double BSIM4v0a0;   
    double BSIM4v0ags;      
    double BSIM4v0a1;         
    double BSIM4v0a2;         
    double BSIM4v0keta;     
    double BSIM4v0nsub;
    double BSIM4v0ndep;        
    double BSIM4v0nsd;
    double BSIM4v0phin;
    double BSIM4v0ngate;        
    double BSIM4v0gamma1;      
    double BSIM4v0gamma2;     
    double BSIM4v0vbx;      
    double BSIM4v0vbi;       
    double BSIM4v0vbm;       
    double BSIM4v0vbsc;       
    double BSIM4v0xt;       
    double BSIM4v0phi;
    double BSIM4v0litl;
    double BSIM4v0k1;
    double BSIM4v0kt1;
    double BSIM4v0kt1l;
    double BSIM4v0kt2;
    double BSIM4v0k2;
    double BSIM4v0k3;
    double BSIM4v0k3b;
    double BSIM4v0w0;
    double BSIM4v0dvtp0;
    double BSIM4v0dvtp1;
    double BSIM4v0lpe0;
    double BSIM4v0lpeb;
    double BSIM4v0dvt0;      
    double BSIM4v0dvt1;      
    double BSIM4v0dvt2;      
    double BSIM4v0dvt0w;      
    double BSIM4v0dvt1w;      
    double BSIM4v0dvt2w;      
    double BSIM4v0drout;      
    double BSIM4v0dsub;      
    double BSIM4v0vth0;
    double BSIM4v0ua;
    double BSIM4v0ua1;
    double BSIM4v0ub;
    double BSIM4v0ub1;
    double BSIM4v0uc;
    double BSIM4v0uc1;
    double BSIM4v0u0;
    double BSIM4v0eu;
    double BSIM4v0ute;
    double BSIM4v0voff;
    double BSIM4v0minv;
    double BSIM4v0vfb;
    double BSIM4v0delta;
    double BSIM4v0rdsw;       
    double BSIM4v0rds0;       
    double BSIM4v0rs0;
    double BSIM4v0rd0;
    double BSIM4v0rsw;
    double BSIM4v0rdw;
    double BSIM4v0prwg;       
    double BSIM4v0prwb;       
    double BSIM4v0prt;       
    double BSIM4v0eta0;         
    double BSIM4v0etab;         
    double BSIM4v0pclm;      
    double BSIM4v0pdibl1;      
    double BSIM4v0pdibl2;      
    double BSIM4v0pdiblb;      
    double BSIM4v0fprout;
    double BSIM4v0pdits;
    double BSIM4v0pditsd;
    double BSIM4v0pscbe1;       
    double BSIM4v0pscbe2;       
    double BSIM4v0pvag;       
    double BSIM4v0wr;
    double BSIM4v0dwg;
    double BSIM4v0dwb;
    double BSIM4v0b0;
    double BSIM4v0b1;
    double BSIM4v0alpha0;
    double BSIM4v0alpha1;
    double BSIM4v0beta0;
    double BSIM4v0agidl;
    double BSIM4v0bgidl;
    double BSIM4v0cgidl;
    double BSIM4v0egidl;
    double BSIM4v0aigc;
    double BSIM4v0bigc;
    double BSIM4v0cigc;
    double BSIM4v0aigsd;
    double BSIM4v0bigsd;
    double BSIM4v0cigsd;
    double BSIM4v0aigbacc;
    double BSIM4v0bigbacc;
    double BSIM4v0cigbacc;
    double BSIM4v0aigbinv;
    double BSIM4v0bigbinv;
    double BSIM4v0cigbinv;
    double BSIM4v0nigc;
    double BSIM4v0nigbacc;
    double BSIM4v0nigbinv;
    double BSIM4v0ntox;
    double BSIM4v0eigbinv;
    double BSIM4v0pigcd;
    double BSIM4v0poxedge;
    double BSIM4v0xrcrg1;
    double BSIM4v0xrcrg2;
    double BSIM4v0fpkt;
    double BSIM4v0plcr;
    double BSIM4v0plcrl;
    double BSIM4v0plcrd;


    /* CV model */
    double BSIM4v0cgsl;
    double BSIM4v0cgdl;
    double BSIM4v0ckappas;
    double BSIM4v0ckappad;
    double BSIM4v0cf;
    double BSIM4v0clc;
    double BSIM4v0cle;
    double BSIM4v0vfbcv;
    double BSIM4v0noff;
    double BSIM4v0voffcv;
    double BSIM4v0acde;
    double BSIM4v0moin;

/* Pre-calculated constants */

    double BSIM4v0dw;
    double BSIM4v0dl;
    double BSIM4v0leff;
    double BSIM4v0weff;

    double BSIM4v0dwc;
    double BSIM4v0dlc;
    double BSIM4v0dlcig;
    double BSIM4v0dwj;
    double BSIM4v0leffCV;
    double BSIM4v0weffCV;
    double BSIM4v0weffCJ;
    double BSIM4v0abulkCVfactor;
    double BSIM4v0cgso;
    double BSIM4v0cgdo;
    double BSIM4v0cgbo;

    double BSIM4v0u0temp;       
    double BSIM4v0vsattemp;   
    double BSIM4v0sqrtPhi;   
    double BSIM4v0phis3;   
    double BSIM4v0Xdep0;          
    double BSIM4v0sqrtXdep0;          
    double BSIM4v0theta0vb0;
    double BSIM4v0thetaRout; 
    double BSIM4v0mstar;
    double BSIM4v0voffcbn;
    double BSIM4v0rdswmin;
    double BSIM4v0rdwmin;
    double BSIM4v0rswmin;
    double BSIM4v0vfbsd;

    double BSIM4v0cof1;
    double BSIM4v0cof2;
    double BSIM4v0cof3;
    double BSIM4v0cof4;
    double BSIM4v0cdep0;
    double BSIM4v0vfbzb;
    double BSIM4v0vtfbphi1;
    double BSIM4v0vtfbphi2;
    double BSIM4v0ToxRatio;
    double BSIM4v0Aechvb;
    double BSIM4v0Bechvb;
    double BSIM4v0ToxRatioEdge;
    double BSIM4v0AechvbEdge;
    double BSIM4v0BechvbEdge;
    double BSIM4v0ldeb;
    double BSIM4v0k1ox;
    double BSIM4v0k2ox;

    struct bsim4v0SizeDependParam  *pNext;
};


typedef struct sBSIM4v0model 
{
    int BSIM4v0modType;
    struct sBSIM4v0model *BSIM4v0nextModel;
    BSIM4v0instance *BSIM4v0instances;
    IFuid BSIM4v0modName; 

    /* --- end of generic struct GENmodel --- */

    int BSIM4v0type;

    int    BSIM4v0mobMod;
    int    BSIM4v0capMod;
    int    BSIM4v0dioMod;
    int    BSIM4v0trnqsMod;
    int    BSIM4v0acnqsMod;
    int    BSIM4v0fnoiMod;
    int    BSIM4v0tnoiMod;
    int    BSIM4v0rdsMod;
    int    BSIM4v0rbodyMod;
    int    BSIM4v0rgateMod;
    int    BSIM4v0perMod;
    int    BSIM4v0geoMod;
    int    BSIM4v0igcMod;
    int    BSIM4v0igbMod;
    int    BSIM4v0binUnit;
    int    BSIM4v0paramChk;
    char   *BSIM4v0version;             
    double BSIM4v0toxe;             
    double BSIM4v0toxp;
    double BSIM4v0toxm;
    double BSIM4v0dtox;
    double BSIM4v0epsrox;
    double BSIM4v0cdsc;           
    double BSIM4v0cdscb; 
    double BSIM4v0cdscd;          
    double BSIM4v0cit;           
    double BSIM4v0nfactor;      
    double BSIM4v0xj;
    double BSIM4v0vsat;         
    double BSIM4v0at;         
    double BSIM4v0a0;   
    double BSIM4v0ags;      
    double BSIM4v0a1;         
    double BSIM4v0a2;         
    double BSIM4v0keta;     
    double BSIM4v0nsub;
    double BSIM4v0ndep;        
    double BSIM4v0nsd;
    double BSIM4v0phin;
    double BSIM4v0ngate;        
    double BSIM4v0gamma1;      
    double BSIM4v0gamma2;     
    double BSIM4v0vbx;      
    double BSIM4v0vbm;       
    double BSIM4v0xt;       
    double BSIM4v0k1;
    double BSIM4v0kt1;
    double BSIM4v0kt1l;
    double BSIM4v0kt2;
    double BSIM4v0k2;
    double BSIM4v0k3;
    double BSIM4v0k3b;
    double BSIM4v0w0;
    double BSIM4v0dvtp0;
    double BSIM4v0dvtp1;
    double BSIM4v0lpe0;
    double BSIM4v0lpeb;
    double BSIM4v0dvt0;      
    double BSIM4v0dvt1;      
    double BSIM4v0dvt2;      
    double BSIM4v0dvt0w;      
    double BSIM4v0dvt1w;      
    double BSIM4v0dvt2w;      
    double BSIM4v0drout;      
    double BSIM4v0dsub;      
    double BSIM4v0vth0;
    double BSIM4v0eu;
    double BSIM4v0ua;
    double BSIM4v0ua1;
    double BSIM4v0ub;
    double BSIM4v0ub1;
    double BSIM4v0uc;
    double BSIM4v0uc1;
    double BSIM4v0u0;
    double BSIM4v0ute;
    double BSIM4v0voff;
    double BSIM4v0minv;
    double BSIM4v0voffl;
    double BSIM4v0delta;
    double BSIM4v0rdsw;       
    double BSIM4v0rdswmin;
    double BSIM4v0rdwmin;
    double BSIM4v0rswmin;
    double BSIM4v0rsw;
    double BSIM4v0rdw;
    double BSIM4v0prwg;
    double BSIM4v0prwb;
    double BSIM4v0prt;       
    double BSIM4v0eta0;         
    double BSIM4v0etab;         
    double BSIM4v0pclm;      
    double BSIM4v0pdibl1;      
    double BSIM4v0pdibl2;      
    double BSIM4v0pdiblb;
    double BSIM4v0fprout;
    double BSIM4v0pdits;
    double BSIM4v0pditsd;
    double BSIM4v0pditsl;
    double BSIM4v0pscbe1;       
    double BSIM4v0pscbe2;       
    double BSIM4v0pvag;       
    double BSIM4v0wr;
    double BSIM4v0dwg;
    double BSIM4v0dwb;
    double BSIM4v0b0;
    double BSIM4v0b1;
    double BSIM4v0alpha0;
    double BSIM4v0alpha1;
    double BSIM4v0beta0;
    double BSIM4v0agidl;
    double BSIM4v0bgidl;
    double BSIM4v0cgidl;
    double BSIM4v0egidl;
    double BSIM4v0aigc;
    double BSIM4v0bigc;
    double BSIM4v0cigc;
    double BSIM4v0aigsd;
    double BSIM4v0bigsd;
    double BSIM4v0cigsd;
    double BSIM4v0aigbacc;
    double BSIM4v0bigbacc;
    double BSIM4v0cigbacc;
    double BSIM4v0aigbinv;
    double BSIM4v0bigbinv;
    double BSIM4v0cigbinv;
    double BSIM4v0nigc;
    double BSIM4v0nigbacc;
    double BSIM4v0nigbinv;
    double BSIM4v0ntox;
    double BSIM4v0eigbinv;
    double BSIM4v0pigcd;
    double BSIM4v0poxedge;
    double BSIM4v0toxref;
    double BSIM4v0ijthdfwd;
    double BSIM4v0ijthsfwd;
    double BSIM4v0ijthdrev;
    double BSIM4v0ijthsrev;
    double BSIM4v0xjbvd;
    double BSIM4v0xjbvs;
    double BSIM4v0bvd;
    double BSIM4v0bvs;
    double BSIM4v0xrcrg1;
    double BSIM4v0xrcrg2;

    double BSIM4v0vfb;
    double BSIM4v0gbmin;
    double BSIM4v0rbdb;
    double BSIM4v0rbsb;
    double BSIM4v0rbpb;
    double BSIM4v0rbps;
    double BSIM4v0rbpd;
    double BSIM4v0tnoia;
    double BSIM4v0tnoib;
    double BSIM4v0ntnoi;

    /* CV model and Parasitics */
    double BSIM4v0cgsl;
    double BSIM4v0cgdl;
    double BSIM4v0ckappas;
    double BSIM4v0ckappad;
    double BSIM4v0cf;
    double BSIM4v0vfbcv;
    double BSIM4v0clc;
    double BSIM4v0cle;
    double BSIM4v0dwc;
    double BSIM4v0dlc;
    double BSIM4v0dlcig;
    double BSIM4v0dwj;
    double BSIM4v0noff;
    double BSIM4v0voffcv;
    double BSIM4v0acde;
    double BSIM4v0moin;
    double BSIM4v0tcj;
    double BSIM4v0tcjsw;
    double BSIM4v0tcjswg;
    double BSIM4v0tpb;
    double BSIM4v0tpbsw;
    double BSIM4v0tpbswg;
    double BSIM4v0dmcg;
    double BSIM4v0dmci;
    double BSIM4v0dmdg;
    double BSIM4v0dmcgt;
    double BSIM4v0xgw;
    double BSIM4v0xgl;
    double BSIM4v0rshg;
    double BSIM4v0ngcon;

    /* Length Dependence */
    double BSIM4v0lcdsc;           
    double BSIM4v0lcdscb; 
    double BSIM4v0lcdscd;          
    double BSIM4v0lcit;           
    double BSIM4v0lnfactor;      
    double BSIM4v0lxj;
    double BSIM4v0lvsat;         
    double BSIM4v0lat;         
    double BSIM4v0la0;   
    double BSIM4v0lags;      
    double BSIM4v0la1;         
    double BSIM4v0la2;         
    double BSIM4v0lketa;     
    double BSIM4v0lnsub;
    double BSIM4v0lndep;        
    double BSIM4v0lnsd;
    double BSIM4v0lphin;
    double BSIM4v0lngate;        
    double BSIM4v0lgamma1;      
    double BSIM4v0lgamma2;     
    double BSIM4v0lvbx;      
    double BSIM4v0lvbm;       
    double BSIM4v0lxt;       
    double BSIM4v0lk1;
    double BSIM4v0lkt1;
    double BSIM4v0lkt1l;
    double BSIM4v0lkt2;
    double BSIM4v0lk2;
    double BSIM4v0lk3;
    double BSIM4v0lk3b;
    double BSIM4v0lw0;
    double BSIM4v0ldvtp0;
    double BSIM4v0ldvtp1;
    double BSIM4v0llpe0;
    double BSIM4v0llpeb;
    double BSIM4v0ldvt0;      
    double BSIM4v0ldvt1;      
    double BSIM4v0ldvt2;      
    double BSIM4v0ldvt0w;      
    double BSIM4v0ldvt1w;      
    double BSIM4v0ldvt2w;      
    double BSIM4v0ldrout;      
    double BSIM4v0ldsub;      
    double BSIM4v0lvth0;
    double BSIM4v0lua;
    double BSIM4v0lua1;
    double BSIM4v0lub;
    double BSIM4v0lub1;
    double BSIM4v0luc;
    double BSIM4v0luc1;
    double BSIM4v0lu0;
    double BSIM4v0leu;
    double BSIM4v0lute;
    double BSIM4v0lvoff;
    double BSIM4v0lminv;
    double BSIM4v0ldelta;
    double BSIM4v0lrdsw;       
    double BSIM4v0lrsw;
    double BSIM4v0lrdw;
    double BSIM4v0lprwg;
    double BSIM4v0lprwb;
    double BSIM4v0lprt;       
    double BSIM4v0leta0;         
    double BSIM4v0letab;         
    double BSIM4v0lpclm;      
    double BSIM4v0lpdibl1;      
    double BSIM4v0lpdibl2;      
    double BSIM4v0lpdiblb;
    double BSIM4v0lfprout;
    double BSIM4v0lpdits;
    double BSIM4v0lpditsd;
    double BSIM4v0lpscbe1;       
    double BSIM4v0lpscbe2;       
    double BSIM4v0lpvag;       
    double BSIM4v0lwr;
    double BSIM4v0ldwg;
    double BSIM4v0ldwb;
    double BSIM4v0lb0;
    double BSIM4v0lb1;
    double BSIM4v0lalpha0;
    double BSIM4v0lalpha1;
    double BSIM4v0lbeta0;
    double BSIM4v0lvfb;
    double BSIM4v0lagidl;
    double BSIM4v0lbgidl;
    double BSIM4v0lcgidl;
    double BSIM4v0legidl;
    double BSIM4v0laigc;
    double BSIM4v0lbigc;
    double BSIM4v0lcigc;
    double BSIM4v0laigsd;
    double BSIM4v0lbigsd;
    double BSIM4v0lcigsd;
    double BSIM4v0laigbacc;
    double BSIM4v0lbigbacc;
    double BSIM4v0lcigbacc;
    double BSIM4v0laigbinv;
    double BSIM4v0lbigbinv;
    double BSIM4v0lcigbinv;
    double BSIM4v0lnigc;
    double BSIM4v0lnigbacc;
    double BSIM4v0lnigbinv;
    double BSIM4v0lntox;
    double BSIM4v0leigbinv;
    double BSIM4v0lpigcd;
    double BSIM4v0lpoxedge;
    double BSIM4v0lxrcrg1;
    double BSIM4v0lxrcrg2;

    /* CV model */
    double BSIM4v0lcgsl;
    double BSIM4v0lcgdl;
    double BSIM4v0lckappas;
    double BSIM4v0lckappad;
    double BSIM4v0lcf;
    double BSIM4v0lclc;
    double BSIM4v0lcle;
    double BSIM4v0lvfbcv;
    double BSIM4v0lnoff;
    double BSIM4v0lvoffcv;
    double BSIM4v0lacde;
    double BSIM4v0lmoin;

    /* Width Dependence */
    double BSIM4v0wcdsc;           
    double BSIM4v0wcdscb; 
    double BSIM4v0wcdscd;          
    double BSIM4v0wcit;           
    double BSIM4v0wnfactor;      
    double BSIM4v0wxj;
    double BSIM4v0wvsat;         
    double BSIM4v0wat;         
    double BSIM4v0wa0;   
    double BSIM4v0wags;      
    double BSIM4v0wa1;         
    double BSIM4v0wa2;         
    double BSIM4v0wketa;     
    double BSIM4v0wnsub;
    double BSIM4v0wndep;        
    double BSIM4v0wnsd;
    double BSIM4v0wphin;
    double BSIM4v0wngate;        
    double BSIM4v0wgamma1;      
    double BSIM4v0wgamma2;     
    double BSIM4v0wvbx;      
    double BSIM4v0wvbm;       
    double BSIM4v0wxt;       
    double BSIM4v0wk1;
    double BSIM4v0wkt1;
    double BSIM4v0wkt1l;
    double BSIM4v0wkt2;
    double BSIM4v0wk2;
    double BSIM4v0wk3;
    double BSIM4v0wk3b;
    double BSIM4v0ww0;
    double BSIM4v0wdvtp0;
    double BSIM4v0wdvtp1;
    double BSIM4v0wlpe0;
    double BSIM4v0wlpeb;
    double BSIM4v0wdvt0;      
    double BSIM4v0wdvt1;      
    double BSIM4v0wdvt2;      
    double BSIM4v0wdvt0w;      
    double BSIM4v0wdvt1w;      
    double BSIM4v0wdvt2w;      
    double BSIM4v0wdrout;      
    double BSIM4v0wdsub;      
    double BSIM4v0wvth0;
    double BSIM4v0wua;
    double BSIM4v0wua1;
    double BSIM4v0wub;
    double BSIM4v0wub1;
    double BSIM4v0wuc;
    double BSIM4v0wuc1;
    double BSIM4v0wu0;
    double BSIM4v0weu;
    double BSIM4v0wute;
    double BSIM4v0wvoff;
    double BSIM4v0wminv;
    double BSIM4v0wdelta;
    double BSIM4v0wrdsw;       
    double BSIM4v0wrsw;
    double BSIM4v0wrdw;
    double BSIM4v0wprwg;
    double BSIM4v0wprwb;
    double BSIM4v0wprt;       
    double BSIM4v0weta0;         
    double BSIM4v0wetab;         
    double BSIM4v0wpclm;      
    double BSIM4v0wpdibl1;      
    double BSIM4v0wpdibl2;      
    double BSIM4v0wpdiblb;
    double BSIM4v0wfprout;
    double BSIM4v0wpdits;
    double BSIM4v0wpditsd;
    double BSIM4v0wpscbe1;       
    double BSIM4v0wpscbe2;       
    double BSIM4v0wpvag;       
    double BSIM4v0wwr;
    double BSIM4v0wdwg;
    double BSIM4v0wdwb;
    double BSIM4v0wb0;
    double BSIM4v0wb1;
    double BSIM4v0walpha0;
    double BSIM4v0walpha1;
    double BSIM4v0wbeta0;
    double BSIM4v0wvfb;
    double BSIM4v0wagidl;
    double BSIM4v0wbgidl;
    double BSIM4v0wcgidl;
    double BSIM4v0wegidl;
    double BSIM4v0waigc;
    double BSIM4v0wbigc;
    double BSIM4v0wcigc;
    double BSIM4v0waigsd;
    double BSIM4v0wbigsd;
    double BSIM4v0wcigsd;
    double BSIM4v0waigbacc;
    double BSIM4v0wbigbacc;
    double BSIM4v0wcigbacc;
    double BSIM4v0waigbinv;
    double BSIM4v0wbigbinv;
    double BSIM4v0wcigbinv;
    double BSIM4v0wnigc;
    double BSIM4v0wnigbacc;
    double BSIM4v0wnigbinv;
    double BSIM4v0wntox;
    double BSIM4v0weigbinv;
    double BSIM4v0wpigcd;
    double BSIM4v0wpoxedge;
    double BSIM4v0wxrcrg1;
    double BSIM4v0wxrcrg2;

    /* CV model */
    double BSIM4v0wcgsl;
    double BSIM4v0wcgdl;
    double BSIM4v0wckappas;
    double BSIM4v0wckappad;
    double BSIM4v0wcf;
    double BSIM4v0wclc;
    double BSIM4v0wcle;
    double BSIM4v0wvfbcv;
    double BSIM4v0wnoff;
    double BSIM4v0wvoffcv;
    double BSIM4v0wacde;
    double BSIM4v0wmoin;

    /* Cross-term Dependence */
    double BSIM4v0pcdsc;           
    double BSIM4v0pcdscb; 
    double BSIM4v0pcdscd;          
    double BSIM4v0pcit;           
    double BSIM4v0pnfactor;      
    double BSIM4v0pxj;
    double BSIM4v0pvsat;         
    double BSIM4v0pat;         
    double BSIM4v0pa0;   
    double BSIM4v0pags;      
    double BSIM4v0pa1;         
    double BSIM4v0pa2;         
    double BSIM4v0pketa;     
    double BSIM4v0pnsub;
    double BSIM4v0pndep;        
    double BSIM4v0pnsd;
    double BSIM4v0pphin;
    double BSIM4v0pngate;        
    double BSIM4v0pgamma1;      
    double BSIM4v0pgamma2;     
    double BSIM4v0pvbx;      
    double BSIM4v0pvbm;       
    double BSIM4v0pxt;       
    double BSIM4v0pk1;
    double BSIM4v0pkt1;
    double BSIM4v0pkt1l;
    double BSIM4v0pkt2;
    double BSIM4v0pk2;
    double BSIM4v0pk3;
    double BSIM4v0pk3b;
    double BSIM4v0pw0;
    double BSIM4v0pdvtp0;
    double BSIM4v0pdvtp1;
    double BSIM4v0plpe0;
    double BSIM4v0plpeb;
    double BSIM4v0pdvt0;      
    double BSIM4v0pdvt1;      
    double BSIM4v0pdvt2;      
    double BSIM4v0pdvt0w;      
    double BSIM4v0pdvt1w;      
    double BSIM4v0pdvt2w;      
    double BSIM4v0pdrout;      
    double BSIM4v0pdsub;      
    double BSIM4v0pvth0;
    double BSIM4v0pua;
    double BSIM4v0pua1;
    double BSIM4v0pub;
    double BSIM4v0pub1;
    double BSIM4v0puc;
    double BSIM4v0puc1;
    double BSIM4v0pu0;
    double BSIM4v0peu;
    double BSIM4v0pute;
    double BSIM4v0pvoff;
    double BSIM4v0pminv;
    double BSIM4v0pdelta;
    double BSIM4v0prdsw;
    double BSIM4v0prsw;
    double BSIM4v0prdw;
    double BSIM4v0pprwg;
    double BSIM4v0pprwb;
    double BSIM4v0pprt;       
    double BSIM4v0peta0;         
    double BSIM4v0petab;         
    double BSIM4v0ppclm;      
    double BSIM4v0ppdibl1;      
    double BSIM4v0ppdibl2;      
    double BSIM4v0ppdiblb;
    double BSIM4v0pfprout;
    double BSIM4v0ppdits;
    double BSIM4v0ppditsd;
    double BSIM4v0ppscbe1;       
    double BSIM4v0ppscbe2;       
    double BSIM4v0ppvag;       
    double BSIM4v0pwr;
    double BSIM4v0pdwg;
    double BSIM4v0pdwb;
    double BSIM4v0pb0;
    double BSIM4v0pb1;
    double BSIM4v0palpha0;
    double BSIM4v0palpha1;
    double BSIM4v0pbeta0;
    double BSIM4v0pvfb;
    double BSIM4v0pagidl;
    double BSIM4v0pbgidl;
    double BSIM4v0pcgidl;
    double BSIM4v0pegidl;
    double BSIM4v0paigc;
    double BSIM4v0pbigc;
    double BSIM4v0pcigc;
    double BSIM4v0paigsd;
    double BSIM4v0pbigsd;
    double BSIM4v0pcigsd;
    double BSIM4v0paigbacc;
    double BSIM4v0pbigbacc;
    double BSIM4v0pcigbacc;
    double BSIM4v0paigbinv;
    double BSIM4v0pbigbinv;
    double BSIM4v0pcigbinv;
    double BSIM4v0pnigc;
    double BSIM4v0pnigbacc;
    double BSIM4v0pnigbinv;
    double BSIM4v0pntox;
    double BSIM4v0peigbinv;
    double BSIM4v0ppigcd;
    double BSIM4v0ppoxedge;
    double BSIM4v0pxrcrg1;
    double BSIM4v0pxrcrg2;

    /* CV model */
    double BSIM4v0pcgsl;
    double BSIM4v0pcgdl;
    double BSIM4v0pckappas;
    double BSIM4v0pckappad;
    double BSIM4v0pcf;
    double BSIM4v0pclc;
    double BSIM4v0pcle;
    double BSIM4v0pvfbcv;
    double BSIM4v0pnoff;
    double BSIM4v0pvoffcv;
    double BSIM4v0pacde;
    double BSIM4v0pmoin;

    double BSIM4v0tnom;
    double BSIM4v0cgso;
    double BSIM4v0cgdo;
    double BSIM4v0cgbo;
    double BSIM4v0xpart;
    double BSIM4v0cFringOut;
    double BSIM4v0cFringMax;

    double BSIM4v0sheetResistance;
    double BSIM4v0SjctSatCurDensity;
    double BSIM4v0DjctSatCurDensity;
    double BSIM4v0SjctSidewallSatCurDensity;
    double BSIM4v0DjctSidewallSatCurDensity;
    double BSIM4v0SjctGateSidewallSatCurDensity;
    double BSIM4v0DjctGateSidewallSatCurDensity;
    double BSIM4v0SbulkJctPotential;
    double BSIM4v0DbulkJctPotential;
    double BSIM4v0SbulkJctBotGradingCoeff;
    double BSIM4v0DbulkJctBotGradingCoeff;
    double BSIM4v0SbulkJctSideGradingCoeff;
    double BSIM4v0DbulkJctSideGradingCoeff;
    double BSIM4v0SbulkJctGateSideGradingCoeff;
    double BSIM4v0DbulkJctGateSideGradingCoeff;
    double BSIM4v0SsidewallJctPotential;
    double BSIM4v0DsidewallJctPotential;
    double BSIM4v0SGatesidewallJctPotential;
    double BSIM4v0DGatesidewallJctPotential;
    double BSIM4v0SunitAreaJctCap;
    double BSIM4v0DunitAreaJctCap;
    double BSIM4v0SunitLengthSidewallJctCap;
    double BSIM4v0DunitLengthSidewallJctCap;
    double BSIM4v0SunitLengthGateSidewallJctCap;
    double BSIM4v0DunitLengthGateSidewallJctCap;
    double BSIM4v0SjctEmissionCoeff;
    double BSIM4v0DjctEmissionCoeff;
    double BSIM4v0SjctTempExponent;
    double BSIM4v0DjctTempExponent;

    double BSIM4v0Lint;
    double BSIM4v0Ll;
    double BSIM4v0Llc;
    double BSIM4v0Lln;
    double BSIM4v0Lw;
    double BSIM4v0Lwc;
    double BSIM4v0Lwn;
    double BSIM4v0Lwl;
    double BSIM4v0Lwlc;
    double BSIM4v0Lmin;
    double BSIM4v0Lmax;

    double BSIM4v0Wint;
    double BSIM4v0Wl;
    double BSIM4v0Wlc;
    double BSIM4v0Wln;
    double BSIM4v0Ww;
    double BSIM4v0Wwc;
    double BSIM4v0Wwn;
    double BSIM4v0Wwl;
    double BSIM4v0Wwlc;
    double BSIM4v0Wmin;
    double BSIM4v0Wmax;


/* Pre-calculated constants
 * move to size-dependent param */
    double BSIM4v0vtm;   
    double BSIM4v0coxe;
    double BSIM4v0coxp;
    double BSIM4v0cof1;
    double BSIM4v0cof2;
    double BSIM4v0cof3;
    double BSIM4v0cof4;
    double BSIM4v0vcrit;
    double BSIM4v0factor1;
    double BSIM4v0PhiBS;
    double BSIM4v0PhiBSWS;
    double BSIM4v0PhiBSWGS;
    double BSIM4v0SjctTempSatCurDensity;
    double BSIM4v0SjctSidewallTempSatCurDensity;
    double BSIM4v0SjctGateSidewallTempSatCurDensity;
    double BSIM4v0PhiBD;
    double BSIM4v0PhiBSWD;
    double BSIM4v0PhiBSWGD;
    double BSIM4v0DjctTempSatCurDensity;
    double BSIM4v0DjctSidewallTempSatCurDensity;
    double BSIM4v0DjctGateSidewallTempSatCurDensity;

    double BSIM4v0oxideTrapDensityA;      
    double BSIM4v0oxideTrapDensityB;     
    double BSIM4v0oxideTrapDensityC;  
    double BSIM4v0em;  
    double BSIM4v0ef;  
    double BSIM4v0af;  
    double BSIM4v0kf;  

    struct bsim4v0SizeDependParam *pSizeDependParamKnot;

    /* Flags */
    unsigned  BSIM4v0mobModGiven :1;
    unsigned  BSIM4v0binUnitGiven :1;
    unsigned  BSIM4v0capModGiven :1;
    unsigned  BSIM4v0dioModGiven :1;
    unsigned  BSIM4v0rdsModGiven :1;
    unsigned  BSIM4v0rbodyModGiven :1;
    unsigned  BSIM4v0rgateModGiven :1;
    unsigned  BSIM4v0perModGiven :1;
    unsigned  BSIM4v0geoModGiven :1;
    unsigned  BSIM4v0paramChkGiven :1;
    unsigned  BSIM4v0trnqsModGiven :1;
    unsigned  BSIM4v0acnqsModGiven :1;
    unsigned  BSIM4v0fnoiModGiven :1;
    unsigned  BSIM4v0tnoiModGiven :1;
    unsigned  BSIM4v0igcModGiven :1;
    unsigned  BSIM4v0igbModGiven :1;
    unsigned  BSIM4v0typeGiven   :1;
    unsigned  BSIM4v0toxrefGiven   :1;
    unsigned  BSIM4v0toxeGiven   :1;
    unsigned  BSIM4v0toxpGiven   :1;
    unsigned  BSIM4v0toxmGiven   :1;
    unsigned  BSIM4v0dtoxGiven   :1;
    unsigned  BSIM4v0epsroxGiven   :1;
    unsigned  BSIM4v0versionGiven   :1;
    unsigned  BSIM4v0cdscGiven   :1;
    unsigned  BSIM4v0cdscbGiven   :1;
    unsigned  BSIM4v0cdscdGiven   :1;
    unsigned  BSIM4v0citGiven   :1;
    unsigned  BSIM4v0nfactorGiven   :1;
    unsigned  BSIM4v0xjGiven   :1;
    unsigned  BSIM4v0vsatGiven   :1;
    unsigned  BSIM4v0atGiven   :1;
    unsigned  BSIM4v0a0Given   :1;
    unsigned  BSIM4v0agsGiven   :1;
    unsigned  BSIM4v0a1Given   :1;
    unsigned  BSIM4v0a2Given   :1;
    unsigned  BSIM4v0ketaGiven   :1;    
    unsigned  BSIM4v0nsubGiven   :1;
    unsigned  BSIM4v0ndepGiven   :1;
    unsigned  BSIM4v0nsdGiven    :1;
    unsigned  BSIM4v0phinGiven   :1;
    unsigned  BSIM4v0ngateGiven   :1;
    unsigned  BSIM4v0gamma1Given   :1;
    unsigned  BSIM4v0gamma2Given   :1;
    unsigned  BSIM4v0vbxGiven   :1;
    unsigned  BSIM4v0vbmGiven   :1;
    unsigned  BSIM4v0xtGiven   :1;
    unsigned  BSIM4v0k1Given   :1;
    unsigned  BSIM4v0kt1Given   :1;
    unsigned  BSIM4v0kt1lGiven   :1;
    unsigned  BSIM4v0kt2Given   :1;
    unsigned  BSIM4v0k2Given   :1;
    unsigned  BSIM4v0k3Given   :1;
    unsigned  BSIM4v0k3bGiven   :1;
    unsigned  BSIM4v0w0Given   :1;
    unsigned  BSIM4v0dvtp0Given :1;
    unsigned  BSIM4v0dvtp1Given :1;
    unsigned  BSIM4v0lpe0Given   :1;
    unsigned  BSIM4v0lpebGiven   :1;
    unsigned  BSIM4v0dvt0Given   :1;   
    unsigned  BSIM4v0dvt1Given   :1;     
    unsigned  BSIM4v0dvt2Given   :1;     
    unsigned  BSIM4v0dvt0wGiven   :1;   
    unsigned  BSIM4v0dvt1wGiven   :1;     
    unsigned  BSIM4v0dvt2wGiven   :1;     
    unsigned  BSIM4v0droutGiven   :1;     
    unsigned  BSIM4v0dsubGiven   :1;     
    unsigned  BSIM4v0vth0Given   :1;
    unsigned  BSIM4v0euGiven   :1;
    unsigned  BSIM4v0uaGiven   :1;
    unsigned  BSIM4v0ua1Given   :1;
    unsigned  BSIM4v0ubGiven   :1;
    unsigned  BSIM4v0ub1Given   :1;
    unsigned  BSIM4v0ucGiven   :1;
    unsigned  BSIM4v0uc1Given   :1;
    unsigned  BSIM4v0u0Given   :1;
    unsigned  BSIM4v0uteGiven   :1;
    unsigned  BSIM4v0voffGiven   :1;
    unsigned  BSIM4v0vofflGiven  :1;
    unsigned  BSIM4v0minvGiven   :1;
    unsigned  BSIM4v0rdswGiven   :1;      
    unsigned  BSIM4v0rdswminGiven :1;
    unsigned  BSIM4v0rdwminGiven :1;
    unsigned  BSIM4v0rswminGiven :1;
    unsigned  BSIM4v0rswGiven   :1;
    unsigned  BSIM4v0rdwGiven   :1;
    unsigned  BSIM4v0prwgGiven   :1;      
    unsigned  BSIM4v0prwbGiven   :1;      
    unsigned  BSIM4v0prtGiven   :1;      
    unsigned  BSIM4v0eta0Given   :1;    
    unsigned  BSIM4v0etabGiven   :1;    
    unsigned  BSIM4v0pclmGiven   :1;   
    unsigned  BSIM4v0pdibl1Given   :1;   
    unsigned  BSIM4v0pdibl2Given   :1;  
    unsigned  BSIM4v0pdiblbGiven   :1;  
    unsigned  BSIM4v0fproutGiven   :1;
    unsigned  BSIM4v0pditsGiven    :1;
    unsigned  BSIM4v0pditsdGiven    :1;
    unsigned  BSIM4v0pditslGiven    :1;
    unsigned  BSIM4v0pscbe1Given   :1;    
    unsigned  BSIM4v0pscbe2Given   :1;    
    unsigned  BSIM4v0pvagGiven   :1;    
    unsigned  BSIM4v0deltaGiven  :1;     
    unsigned  BSIM4v0wrGiven   :1;
    unsigned  BSIM4v0dwgGiven   :1;
    unsigned  BSIM4v0dwbGiven   :1;
    unsigned  BSIM4v0b0Given   :1;
    unsigned  BSIM4v0b1Given   :1;
    unsigned  BSIM4v0alpha0Given   :1;
    unsigned  BSIM4v0alpha1Given   :1;
    unsigned  BSIM4v0beta0Given   :1;
    unsigned  BSIM4v0agidlGiven   :1;
    unsigned  BSIM4v0bgidlGiven   :1;
    unsigned  BSIM4v0cgidlGiven   :1;
    unsigned  BSIM4v0egidlGiven   :1;
    unsigned  BSIM4v0aigcGiven   :1;
    unsigned  BSIM4v0bigcGiven   :1;
    unsigned  BSIM4v0cigcGiven   :1;
    unsigned  BSIM4v0aigsdGiven   :1;
    unsigned  BSIM4v0bigsdGiven   :1;
    unsigned  BSIM4v0cigsdGiven   :1;
    unsigned  BSIM4v0aigbaccGiven   :1;
    unsigned  BSIM4v0bigbaccGiven   :1;
    unsigned  BSIM4v0cigbaccGiven   :1;
    unsigned  BSIM4v0aigbinvGiven   :1;
    unsigned  BSIM4v0bigbinvGiven   :1;
    unsigned  BSIM4v0cigbinvGiven   :1;
    unsigned  BSIM4v0nigcGiven   :1;
    unsigned  BSIM4v0nigbinvGiven   :1;
    unsigned  BSIM4v0nigbaccGiven   :1;
    unsigned  BSIM4v0ntoxGiven   :1;
    unsigned  BSIM4v0eigbinvGiven   :1;
    unsigned  BSIM4v0pigcdGiven   :1;
    unsigned  BSIM4v0poxedgeGiven   :1;
    unsigned  BSIM4v0ijthdfwdGiven  :1;
    unsigned  BSIM4v0ijthsfwdGiven  :1;
    unsigned  BSIM4v0ijthdrevGiven  :1;
    unsigned  BSIM4v0ijthsrevGiven  :1;
    unsigned  BSIM4v0xjbvdGiven   :1;
    unsigned  BSIM4v0xjbvsGiven   :1;
    unsigned  BSIM4v0bvdGiven   :1;
    unsigned  BSIM4v0bvsGiven   :1;
    unsigned  BSIM4v0vfbGiven   :1;
    unsigned  BSIM4v0gbminGiven :1;
    unsigned  BSIM4v0rbdbGiven :1;
    unsigned  BSIM4v0rbsbGiven :1;
    unsigned  BSIM4v0rbpsGiven :1;
    unsigned  BSIM4v0rbpdGiven :1;
    unsigned  BSIM4v0rbpbGiven :1;
    unsigned  BSIM4v0xrcrg1Given   :1;
    unsigned  BSIM4v0xrcrg2Given   :1;
    unsigned  BSIM4v0tnoiaGiven    :1;
    unsigned  BSIM4v0tnoibGiven    :1;
    unsigned  BSIM4v0ntnoiGiven    :1;

    /* CV model and parasitics */
    unsigned  BSIM4v0cgslGiven   :1;
    unsigned  BSIM4v0cgdlGiven   :1;
    unsigned  BSIM4v0ckappasGiven   :1;
    unsigned  BSIM4v0ckappadGiven   :1;
    unsigned  BSIM4v0cfGiven   :1;
    unsigned  BSIM4v0vfbcvGiven   :1;
    unsigned  BSIM4v0clcGiven   :1;
    unsigned  BSIM4v0cleGiven   :1;
    unsigned  BSIM4v0dwcGiven   :1;
    unsigned  BSIM4v0dlcGiven   :1;
    unsigned  BSIM4v0dlcigGiven   :1;
    unsigned  BSIM4v0dwjGiven   :1;
    unsigned  BSIM4v0noffGiven  :1;
    unsigned  BSIM4v0voffcvGiven :1;
    unsigned  BSIM4v0acdeGiven  :1;
    unsigned  BSIM4v0moinGiven  :1;
    unsigned  BSIM4v0tcjGiven   :1;
    unsigned  BSIM4v0tcjswGiven :1;
    unsigned  BSIM4v0tcjswgGiven :1;
    unsigned  BSIM4v0tpbGiven    :1;
    unsigned  BSIM4v0tpbswGiven  :1;
    unsigned  BSIM4v0tpbswgGiven :1;
    unsigned  BSIM4v0dmcgGiven :1;
    unsigned  BSIM4v0dmciGiven :1;
    unsigned  BSIM4v0dmdgGiven :1;
    unsigned  BSIM4v0dmcgtGiven :1;
    unsigned  BSIM4v0xgwGiven :1;
    unsigned  BSIM4v0xglGiven :1;
    unsigned  BSIM4v0rshgGiven :1;
    unsigned  BSIM4v0ngconGiven :1;


    /* Length dependence */
    unsigned  BSIM4v0lcdscGiven   :1;
    unsigned  BSIM4v0lcdscbGiven   :1;
    unsigned  BSIM4v0lcdscdGiven   :1;
    unsigned  BSIM4v0lcitGiven   :1;
    unsigned  BSIM4v0lnfactorGiven   :1;
    unsigned  BSIM4v0lxjGiven   :1;
    unsigned  BSIM4v0lvsatGiven   :1;
    unsigned  BSIM4v0latGiven   :1;
    unsigned  BSIM4v0la0Given   :1;
    unsigned  BSIM4v0lagsGiven   :1;
    unsigned  BSIM4v0la1Given   :1;
    unsigned  BSIM4v0la2Given   :1;
    unsigned  BSIM4v0lketaGiven   :1;    
    unsigned  BSIM4v0lnsubGiven   :1;
    unsigned  BSIM4v0lndepGiven   :1;
    unsigned  BSIM4v0lnsdGiven    :1;
    unsigned  BSIM4v0lphinGiven   :1;
    unsigned  BSIM4v0lngateGiven   :1;
    unsigned  BSIM4v0lgamma1Given   :1;
    unsigned  BSIM4v0lgamma2Given   :1;
    unsigned  BSIM4v0lvbxGiven   :1;
    unsigned  BSIM4v0lvbmGiven   :1;
    unsigned  BSIM4v0lxtGiven   :1;
    unsigned  BSIM4v0lk1Given   :1;
    unsigned  BSIM4v0lkt1Given   :1;
    unsigned  BSIM4v0lkt1lGiven   :1;
    unsigned  BSIM4v0lkt2Given   :1;
    unsigned  BSIM4v0lk2Given   :1;
    unsigned  BSIM4v0lk3Given   :1;
    unsigned  BSIM4v0lk3bGiven   :1;
    unsigned  BSIM4v0lw0Given   :1;
    unsigned  BSIM4v0ldvtp0Given :1;
    unsigned  BSIM4v0ldvtp1Given :1;
    unsigned  BSIM4v0llpe0Given   :1;
    unsigned  BSIM4v0llpebGiven   :1;
    unsigned  BSIM4v0ldvt0Given   :1;   
    unsigned  BSIM4v0ldvt1Given   :1;     
    unsigned  BSIM4v0ldvt2Given   :1;     
    unsigned  BSIM4v0ldvt0wGiven   :1;   
    unsigned  BSIM4v0ldvt1wGiven   :1;     
    unsigned  BSIM4v0ldvt2wGiven   :1;     
    unsigned  BSIM4v0ldroutGiven   :1;     
    unsigned  BSIM4v0ldsubGiven   :1;     
    unsigned  BSIM4v0lvth0Given   :1;
    unsigned  BSIM4v0luaGiven   :1;
    unsigned  BSIM4v0lua1Given   :1;
    unsigned  BSIM4v0lubGiven   :1;
    unsigned  BSIM4v0lub1Given   :1;
    unsigned  BSIM4v0lucGiven   :1;
    unsigned  BSIM4v0luc1Given   :1;
    unsigned  BSIM4v0lu0Given   :1;
    unsigned  BSIM4v0leuGiven   :1;
    unsigned  BSIM4v0luteGiven   :1;
    unsigned  BSIM4v0lvoffGiven   :1;
    unsigned  BSIM4v0lminvGiven   :1;
    unsigned  BSIM4v0lrdswGiven   :1;      
    unsigned  BSIM4v0lrswGiven   :1;
    unsigned  BSIM4v0lrdwGiven   :1;
    unsigned  BSIM4v0lprwgGiven   :1;      
    unsigned  BSIM4v0lprwbGiven   :1;      
    unsigned  BSIM4v0lprtGiven   :1;      
    unsigned  BSIM4v0leta0Given   :1;    
    unsigned  BSIM4v0letabGiven   :1;    
    unsigned  BSIM4v0lpclmGiven   :1;   
    unsigned  BSIM4v0lpdibl1Given   :1;   
    unsigned  BSIM4v0lpdibl2Given   :1;  
    unsigned  BSIM4v0lpdiblbGiven   :1;  
    unsigned  BSIM4v0lfproutGiven   :1;
    unsigned  BSIM4v0lpditsGiven    :1;
    unsigned  BSIM4v0lpditsdGiven    :1;
    unsigned  BSIM4v0lpscbe1Given   :1;    
    unsigned  BSIM4v0lpscbe2Given   :1;    
    unsigned  BSIM4v0lpvagGiven   :1;    
    unsigned  BSIM4v0ldeltaGiven  :1;     
    unsigned  BSIM4v0lwrGiven   :1;
    unsigned  BSIM4v0ldwgGiven   :1;
    unsigned  BSIM4v0ldwbGiven   :1;
    unsigned  BSIM4v0lb0Given   :1;
    unsigned  BSIM4v0lb1Given   :1;
    unsigned  BSIM4v0lalpha0Given   :1;
    unsigned  BSIM4v0lalpha1Given   :1;
    unsigned  BSIM4v0lbeta0Given   :1;
    unsigned  BSIM4v0lvfbGiven   :1;
    unsigned  BSIM4v0lagidlGiven   :1;
    unsigned  BSIM4v0lbgidlGiven   :1;
    unsigned  BSIM4v0lcgidlGiven   :1;
    unsigned  BSIM4v0legidlGiven   :1;
    unsigned  BSIM4v0laigcGiven   :1;
    unsigned  BSIM4v0lbigcGiven   :1;
    unsigned  BSIM4v0lcigcGiven   :1;
    unsigned  BSIM4v0laigsdGiven   :1;
    unsigned  BSIM4v0lbigsdGiven   :1;
    unsigned  BSIM4v0lcigsdGiven   :1;
    unsigned  BSIM4v0laigbaccGiven   :1;
    unsigned  BSIM4v0lbigbaccGiven   :1;
    unsigned  BSIM4v0lcigbaccGiven   :1;
    unsigned  BSIM4v0laigbinvGiven   :1;
    unsigned  BSIM4v0lbigbinvGiven   :1;
    unsigned  BSIM4v0lcigbinvGiven   :1;
    unsigned  BSIM4v0lnigcGiven   :1;
    unsigned  BSIM4v0lnigbinvGiven   :1;
    unsigned  BSIM4v0lnigbaccGiven   :1;
    unsigned  BSIM4v0lntoxGiven   :1;
    unsigned  BSIM4v0leigbinvGiven   :1;
    unsigned  BSIM4v0lpigcdGiven   :1;
    unsigned  BSIM4v0lpoxedgeGiven   :1;
    unsigned  BSIM4v0lxrcrg1Given   :1;
    unsigned  BSIM4v0lxrcrg2Given   :1;

    /* CV model */
    unsigned  BSIM4v0lcgslGiven   :1;
    unsigned  BSIM4v0lcgdlGiven   :1;
    unsigned  BSIM4v0lckappasGiven   :1;
    unsigned  BSIM4v0lckappadGiven   :1;
    unsigned  BSIM4v0lcfGiven   :1;
    unsigned  BSIM4v0lclcGiven   :1;
    unsigned  BSIM4v0lcleGiven   :1;
    unsigned  BSIM4v0lvfbcvGiven   :1;
    unsigned  BSIM4v0lnoffGiven   :1;
    unsigned  BSIM4v0lvoffcvGiven :1;
    unsigned  BSIM4v0lacdeGiven   :1;
    unsigned  BSIM4v0lmoinGiven   :1;

    /* Width dependence */
    unsigned  BSIM4v0wcdscGiven   :1;
    unsigned  BSIM4v0wcdscbGiven   :1;
    unsigned  BSIM4v0wcdscdGiven   :1;
    unsigned  BSIM4v0wcitGiven   :1;
    unsigned  BSIM4v0wnfactorGiven   :1;
    unsigned  BSIM4v0wxjGiven   :1;
    unsigned  BSIM4v0wvsatGiven   :1;
    unsigned  BSIM4v0watGiven   :1;
    unsigned  BSIM4v0wa0Given   :1;
    unsigned  BSIM4v0wagsGiven   :1;
    unsigned  BSIM4v0wa1Given   :1;
    unsigned  BSIM4v0wa2Given   :1;
    unsigned  BSIM4v0wketaGiven   :1;    
    unsigned  BSIM4v0wnsubGiven   :1;
    unsigned  BSIM4v0wndepGiven   :1;
    unsigned  BSIM4v0wnsdGiven    :1;
    unsigned  BSIM4v0wphinGiven   :1;
    unsigned  BSIM4v0wngateGiven   :1;
    unsigned  BSIM4v0wgamma1Given   :1;
    unsigned  BSIM4v0wgamma2Given   :1;
    unsigned  BSIM4v0wvbxGiven   :1;
    unsigned  BSIM4v0wvbmGiven   :1;
    unsigned  BSIM4v0wxtGiven   :1;
    unsigned  BSIM4v0wk1Given   :1;
    unsigned  BSIM4v0wkt1Given   :1;
    unsigned  BSIM4v0wkt1lGiven   :1;
    unsigned  BSIM4v0wkt2Given   :1;
    unsigned  BSIM4v0wk2Given   :1;
    unsigned  BSIM4v0wk3Given   :1;
    unsigned  BSIM4v0wk3bGiven   :1;
    unsigned  BSIM4v0ww0Given   :1;
    unsigned  BSIM4v0wdvtp0Given :1;
    unsigned  BSIM4v0wdvtp1Given :1;
    unsigned  BSIM4v0wlpe0Given   :1;
    unsigned  BSIM4v0wlpebGiven   :1;
    unsigned  BSIM4v0wdvt0Given   :1;   
    unsigned  BSIM4v0wdvt1Given   :1;     
    unsigned  BSIM4v0wdvt2Given   :1;     
    unsigned  BSIM4v0wdvt0wGiven   :1;   
    unsigned  BSIM4v0wdvt1wGiven   :1;     
    unsigned  BSIM4v0wdvt2wGiven   :1;     
    unsigned  BSIM4v0wdroutGiven   :1;     
    unsigned  BSIM4v0wdsubGiven   :1;     
    unsigned  BSIM4v0wvth0Given   :1;
    unsigned  BSIM4v0wuaGiven   :1;
    unsigned  BSIM4v0wua1Given   :1;
    unsigned  BSIM4v0wubGiven   :1;
    unsigned  BSIM4v0wub1Given   :1;
    unsigned  BSIM4v0wucGiven   :1;
    unsigned  BSIM4v0wuc1Given   :1;
    unsigned  BSIM4v0wu0Given   :1;
    unsigned  BSIM4v0weuGiven   :1;
    unsigned  BSIM4v0wuteGiven   :1;
    unsigned  BSIM4v0wvoffGiven   :1;
    unsigned  BSIM4v0wminvGiven   :1;
    unsigned  BSIM4v0wrdswGiven   :1;      
    unsigned  BSIM4v0wrswGiven   :1;
    unsigned  BSIM4v0wrdwGiven   :1;
    unsigned  BSIM4v0wprwgGiven   :1;      
    unsigned  BSIM4v0wprwbGiven   :1;      
    unsigned  BSIM4v0wprtGiven   :1;      
    unsigned  BSIM4v0weta0Given   :1;    
    unsigned  BSIM4v0wetabGiven   :1;    
    unsigned  BSIM4v0wpclmGiven   :1;   
    unsigned  BSIM4v0wpdibl1Given   :1;   
    unsigned  BSIM4v0wpdibl2Given   :1;  
    unsigned  BSIM4v0wpdiblbGiven   :1;  
    unsigned  BSIM4v0wfproutGiven   :1;
    unsigned  BSIM4v0wpditsGiven    :1;
    unsigned  BSIM4v0wpditsdGiven    :1;
    unsigned  BSIM4v0wpscbe1Given   :1;    
    unsigned  BSIM4v0wpscbe2Given   :1;    
    unsigned  BSIM4v0wpvagGiven   :1;    
    unsigned  BSIM4v0wdeltaGiven  :1;     
    unsigned  BSIM4v0wwrGiven   :1;
    unsigned  BSIM4v0wdwgGiven   :1;
    unsigned  BSIM4v0wdwbGiven   :1;
    unsigned  BSIM4v0wb0Given   :1;
    unsigned  BSIM4v0wb1Given   :1;
    unsigned  BSIM4v0walpha0Given   :1;
    unsigned  BSIM4v0walpha1Given   :1;
    unsigned  BSIM4v0wbeta0Given   :1;
    unsigned  BSIM4v0wvfbGiven   :1;
    unsigned  BSIM4v0wagidlGiven   :1;
    unsigned  BSIM4v0wbgidlGiven   :1;
    unsigned  BSIM4v0wcgidlGiven   :1;
    unsigned  BSIM4v0wegidlGiven   :1;
    unsigned  BSIM4v0waigcGiven   :1;
    unsigned  BSIM4v0wbigcGiven   :1;
    unsigned  BSIM4v0wcigcGiven   :1;
    unsigned  BSIM4v0waigsdGiven   :1;
    unsigned  BSIM4v0wbigsdGiven   :1;
    unsigned  BSIM4v0wcigsdGiven   :1;
    unsigned  BSIM4v0waigbaccGiven   :1;
    unsigned  BSIM4v0wbigbaccGiven   :1;
    unsigned  BSIM4v0wcigbaccGiven   :1;
    unsigned  BSIM4v0waigbinvGiven   :1;
    unsigned  BSIM4v0wbigbinvGiven   :1;
    unsigned  BSIM4v0wcigbinvGiven   :1;
    unsigned  BSIM4v0wnigcGiven   :1;
    unsigned  BSIM4v0wnigbinvGiven   :1;
    unsigned  BSIM4v0wnigbaccGiven   :1;
    unsigned  BSIM4v0wntoxGiven   :1;
    unsigned  BSIM4v0weigbinvGiven   :1;
    unsigned  BSIM4v0wpigcdGiven   :1;
    unsigned  BSIM4v0wpoxedgeGiven   :1;
    unsigned  BSIM4v0wxrcrg1Given   :1;
    unsigned  BSIM4v0wxrcrg2Given   :1;

    /* CV model */
    unsigned  BSIM4v0wcgslGiven   :1;
    unsigned  BSIM4v0wcgdlGiven   :1;
    unsigned  BSIM4v0wckappasGiven   :1;
    unsigned  BSIM4v0wckappadGiven   :1;
    unsigned  BSIM4v0wcfGiven   :1;
    unsigned  BSIM4v0wclcGiven   :1;
    unsigned  BSIM4v0wcleGiven   :1;
    unsigned  BSIM4v0wvfbcvGiven   :1;
    unsigned  BSIM4v0wnoffGiven   :1;
    unsigned  BSIM4v0wvoffcvGiven :1;
    unsigned  BSIM4v0wacdeGiven   :1;
    unsigned  BSIM4v0wmoinGiven   :1;

    /* Cross-term dependence */
    unsigned  BSIM4v0pcdscGiven   :1;
    unsigned  BSIM4v0pcdscbGiven   :1;
    unsigned  BSIM4v0pcdscdGiven   :1;
    unsigned  BSIM4v0pcitGiven   :1;
    unsigned  BSIM4v0pnfactorGiven   :1;
    unsigned  BSIM4v0pxjGiven   :1;
    unsigned  BSIM4v0pvsatGiven   :1;
    unsigned  BSIM4v0patGiven   :1;
    unsigned  BSIM4v0pa0Given   :1;
    unsigned  BSIM4v0pagsGiven   :1;
    unsigned  BSIM4v0pa1Given   :1;
    unsigned  BSIM4v0pa2Given   :1;
    unsigned  BSIM4v0pketaGiven   :1;    
    unsigned  BSIM4v0pnsubGiven   :1;
    unsigned  BSIM4v0pndepGiven   :1;
    unsigned  BSIM4v0pnsdGiven    :1;
    unsigned  BSIM4v0pphinGiven   :1;
    unsigned  BSIM4v0pngateGiven   :1;
    unsigned  BSIM4v0pgamma1Given   :1;
    unsigned  BSIM4v0pgamma2Given   :1;
    unsigned  BSIM4v0pvbxGiven   :1;
    unsigned  BSIM4v0pvbmGiven   :1;
    unsigned  BSIM4v0pxtGiven   :1;
    unsigned  BSIM4v0pk1Given   :1;
    unsigned  BSIM4v0pkt1Given   :1;
    unsigned  BSIM4v0pkt1lGiven   :1;
    unsigned  BSIM4v0pkt2Given   :1;
    unsigned  BSIM4v0pk2Given   :1;
    unsigned  BSIM4v0pk3Given   :1;
    unsigned  BSIM4v0pk3bGiven   :1;
    unsigned  BSIM4v0pw0Given   :1;
    unsigned  BSIM4v0pdvtp0Given :1;
    unsigned  BSIM4v0pdvtp1Given :1;
    unsigned  BSIM4v0plpe0Given   :1;
    unsigned  BSIM4v0plpebGiven   :1;
    unsigned  BSIM4v0pdvt0Given   :1;   
    unsigned  BSIM4v0pdvt1Given   :1;     
    unsigned  BSIM4v0pdvt2Given   :1;     
    unsigned  BSIM4v0pdvt0wGiven   :1;   
    unsigned  BSIM4v0pdvt1wGiven   :1;     
    unsigned  BSIM4v0pdvt2wGiven   :1;     
    unsigned  BSIM4v0pdroutGiven   :1;     
    unsigned  BSIM4v0pdsubGiven   :1;     
    unsigned  BSIM4v0pvth0Given   :1;
    unsigned  BSIM4v0puaGiven   :1;
    unsigned  BSIM4v0pua1Given   :1;
    unsigned  BSIM4v0pubGiven   :1;
    unsigned  BSIM4v0pub1Given   :1;
    unsigned  BSIM4v0pucGiven   :1;
    unsigned  BSIM4v0puc1Given   :1;
    unsigned  BSIM4v0pu0Given   :1;
    unsigned  BSIM4v0peuGiven   :1;
    unsigned  BSIM4v0puteGiven   :1;
    unsigned  BSIM4v0pvoffGiven   :1;
    unsigned  BSIM4v0pminvGiven   :1;
    unsigned  BSIM4v0prdswGiven   :1;      
    unsigned  BSIM4v0prswGiven   :1;
    unsigned  BSIM4v0prdwGiven   :1;
    unsigned  BSIM4v0pprwgGiven   :1;      
    unsigned  BSIM4v0pprwbGiven   :1;      
    unsigned  BSIM4v0pprtGiven   :1;      
    unsigned  BSIM4v0peta0Given   :1;    
    unsigned  BSIM4v0petabGiven   :1;    
    unsigned  BSIM4v0ppclmGiven   :1;   
    unsigned  BSIM4v0ppdibl1Given   :1;   
    unsigned  BSIM4v0ppdibl2Given   :1;  
    unsigned  BSIM4v0ppdiblbGiven   :1;  
    unsigned  BSIM4v0pfproutGiven   :1;
    unsigned  BSIM4v0ppditsGiven    :1;
    unsigned  BSIM4v0ppditsdGiven    :1;
    unsigned  BSIM4v0ppscbe1Given   :1;    
    unsigned  BSIM4v0ppscbe2Given   :1;    
    unsigned  BSIM4v0ppvagGiven   :1;    
    unsigned  BSIM4v0pdeltaGiven  :1;     
    unsigned  BSIM4v0pwrGiven   :1;
    unsigned  BSIM4v0pdwgGiven   :1;
    unsigned  BSIM4v0pdwbGiven   :1;
    unsigned  BSIM4v0pb0Given   :1;
    unsigned  BSIM4v0pb1Given   :1;
    unsigned  BSIM4v0palpha0Given   :1;
    unsigned  BSIM4v0palpha1Given   :1;
    unsigned  BSIM4v0pbeta0Given   :1;
    unsigned  BSIM4v0pvfbGiven   :1;
    unsigned  BSIM4v0pagidlGiven   :1;
    unsigned  BSIM4v0pbgidlGiven   :1;
    unsigned  BSIM4v0pcgidlGiven   :1;
    unsigned  BSIM4v0pegidlGiven   :1;
    unsigned  BSIM4v0paigcGiven   :1;
    unsigned  BSIM4v0pbigcGiven   :1;
    unsigned  BSIM4v0pcigcGiven   :1;
    unsigned  BSIM4v0paigsdGiven   :1;
    unsigned  BSIM4v0pbigsdGiven   :1;
    unsigned  BSIM4v0pcigsdGiven   :1;
    unsigned  BSIM4v0paigbaccGiven   :1;
    unsigned  BSIM4v0pbigbaccGiven   :1;
    unsigned  BSIM4v0pcigbaccGiven   :1;
    unsigned  BSIM4v0paigbinvGiven   :1;
    unsigned  BSIM4v0pbigbinvGiven   :1;
    unsigned  BSIM4v0pcigbinvGiven   :1;
    unsigned  BSIM4v0pnigcGiven   :1;
    unsigned  BSIM4v0pnigbinvGiven   :1;
    unsigned  BSIM4v0pnigbaccGiven   :1;
    unsigned  BSIM4v0pntoxGiven   :1;
    unsigned  BSIM4v0peigbinvGiven   :1;
    unsigned  BSIM4v0ppigcdGiven   :1;
    unsigned  BSIM4v0ppoxedgeGiven   :1;
    unsigned  BSIM4v0pxrcrg1Given   :1;
    unsigned  BSIM4v0pxrcrg2Given   :1;

    /* CV model */
    unsigned  BSIM4v0pcgslGiven   :1;
    unsigned  BSIM4v0pcgdlGiven   :1;
    unsigned  BSIM4v0pckappasGiven   :1;
    unsigned  BSIM4v0pckappadGiven   :1;
    unsigned  BSIM4v0pcfGiven   :1;
    unsigned  BSIM4v0pclcGiven   :1;
    unsigned  BSIM4v0pcleGiven   :1;
    unsigned  BSIM4v0pvfbcvGiven   :1;
    unsigned  BSIM4v0pnoffGiven   :1;
    unsigned  BSIM4v0pvoffcvGiven :1;
    unsigned  BSIM4v0pacdeGiven   :1;
    unsigned  BSIM4v0pmoinGiven   :1;

    unsigned  BSIM4v0useFringeGiven   :1;

    unsigned  BSIM4v0tnomGiven   :1;
    unsigned  BSIM4v0cgsoGiven   :1;
    unsigned  BSIM4v0cgdoGiven   :1;
    unsigned  BSIM4v0cgboGiven   :1;
    unsigned  BSIM4v0xpartGiven   :1;
    unsigned  BSIM4v0sheetResistanceGiven   :1;

    unsigned  BSIM4v0SjctSatCurDensityGiven   :1;
    unsigned  BSIM4v0SjctSidewallSatCurDensityGiven   :1;
    unsigned  BSIM4v0SjctGateSidewallSatCurDensityGiven   :1;
    unsigned  BSIM4v0SbulkJctPotentialGiven   :1;
    unsigned  BSIM4v0SbulkJctBotGradingCoeffGiven   :1;
    unsigned  BSIM4v0SsidewallJctPotentialGiven   :1;
    unsigned  BSIM4v0SGatesidewallJctPotentialGiven   :1;
    unsigned  BSIM4v0SbulkJctSideGradingCoeffGiven   :1;
    unsigned  BSIM4v0SunitAreaJctCapGiven   :1;
    unsigned  BSIM4v0SunitLengthSidewallJctCapGiven   :1;
    unsigned  BSIM4v0SbulkJctGateSideGradingCoeffGiven   :1;
    unsigned  BSIM4v0SunitLengthGateSidewallJctCapGiven   :1;
    unsigned  BSIM4v0SjctEmissionCoeffGiven :1;
    unsigned  BSIM4v0SjctTempExponentGiven	:1;

    unsigned  BSIM4v0DjctSatCurDensityGiven   :1;
    unsigned  BSIM4v0DjctSidewallSatCurDensityGiven   :1;
    unsigned  BSIM4v0DjctGateSidewallSatCurDensityGiven   :1;
    unsigned  BSIM4v0DbulkJctPotentialGiven   :1;
    unsigned  BSIM4v0DbulkJctBotGradingCoeffGiven   :1;
    unsigned  BSIM4v0DsidewallJctPotentialGiven   :1;
    unsigned  BSIM4v0DGatesidewallJctPotentialGiven   :1;
    unsigned  BSIM4v0DbulkJctSideGradingCoeffGiven   :1;
    unsigned  BSIM4v0DunitAreaJctCapGiven   :1;
    unsigned  BSIM4v0DunitLengthSidewallJctCapGiven   :1;
    unsigned  BSIM4v0DbulkJctGateSideGradingCoeffGiven   :1;
    unsigned  BSIM4v0DunitLengthGateSidewallJctCapGiven   :1;
    unsigned  BSIM4v0DjctEmissionCoeffGiven :1;
    unsigned  BSIM4v0DjctTempExponentGiven :1;

    unsigned  BSIM4v0oxideTrapDensityAGiven  :1;         
    unsigned  BSIM4v0oxideTrapDensityBGiven  :1;        
    unsigned  BSIM4v0oxideTrapDensityCGiven  :1;     
    unsigned  BSIM4v0emGiven  :1;     
    unsigned  BSIM4v0efGiven  :1;     
    unsigned  BSIM4v0afGiven  :1;     
    unsigned  BSIM4v0kfGiven  :1;     

    unsigned  BSIM4v0LintGiven   :1;
    unsigned  BSIM4v0LlGiven   :1;
    unsigned  BSIM4v0LlcGiven   :1;
    unsigned  BSIM4v0LlnGiven   :1;
    unsigned  BSIM4v0LwGiven   :1;
    unsigned  BSIM4v0LwcGiven   :1;
    unsigned  BSIM4v0LwnGiven   :1;
    unsigned  BSIM4v0LwlGiven   :1;
    unsigned  BSIM4v0LwlcGiven   :1;
    unsigned  BSIM4v0LminGiven   :1;
    unsigned  BSIM4v0LmaxGiven   :1;

    unsigned  BSIM4v0WintGiven   :1;
    unsigned  BSIM4v0WlGiven   :1;
    unsigned  BSIM4v0WlcGiven   :1;
    unsigned  BSIM4v0WlnGiven   :1;
    unsigned  BSIM4v0WwGiven   :1;
    unsigned  BSIM4v0WwcGiven   :1;
    unsigned  BSIM4v0WwnGiven   :1;
    unsigned  BSIM4v0WwlGiven   :1;
    unsigned  BSIM4v0WwlcGiven   :1;
    unsigned  BSIM4v0WminGiven   :1;
    unsigned  BSIM4v0WmaxGiven   :1;

} BSIM4v0model;


#ifndef NMOS
#define NMOS 1
#define PMOS -1
#endif /*NMOS*/


/* Instance parameters */
#define BSIM4v0_W                   1
#define BSIM4v0_L                   2
#define BSIM4v0_AS                  3
#define BSIM4v0_AD                  4
#define BSIM4v0_PS                  5
#define BSIM4v0_PD                  6
#define BSIM4v0_NRS                 7
#define BSIM4v0_NRD                 8
#define BSIM4v0_OFF                 9
#define BSIM4v0_IC                  10
#define BSIM4v0_IC_VDS              11
#define BSIM4v0_IC_VGS              12
#define BSIM4v0_IC_VBS              13
#define BSIM4v0_TRNQSMOD            14
#define BSIM4v0_RBODYMOD            15
#define BSIM4v0_RGATEMOD            16
#define BSIM4v0_GEOMOD              17
#define BSIM4v0_RGEOMOD             18
#define BSIM4v0_NF                  19
#define BSIM4v0_MIN                 20 
#define BSIM4v0_ACNQSMOD            22
#define BSIM4v0_RBDB                23
#define BSIM4v0_RBSB                24
#define BSIM4v0_RBPB                25
#define BSIM4v0_RBPS                26
#define BSIM4v0_RBPD                27


/* Global parameters */
#define BSIM4v0_MOD_IGCMOD          90
#define BSIM4v0_MOD_IGBMOD          91
#define BSIM4v0_MOD_ACNQSMOD        92
#define BSIM4v0_MOD_FNOIMOD         93
#define BSIM4v0_MOD_RDSMOD          94
#define BSIM4v0_MOD_DIOMOD          96
#define BSIM4v0_MOD_PERMOD          97
#define BSIM4v0_MOD_GEOMOD          98
#define BSIM4v0_MOD_RGATEMOD        99
#define BSIM4v0_MOD_RBODYMOD        100
#define BSIM4v0_MOD_CAPMOD          101
#define BSIM4v0_MOD_TRNQSMOD        102
#define BSIM4v0_MOD_MOBMOD          103    
#define BSIM4v0_MOD_TNOIMOD         104    
#define BSIM4v0_MOD_TOXE            105
#define BSIM4v0_MOD_CDSC            106
#define BSIM4v0_MOD_CDSCB           107
#define BSIM4v0_MOD_CIT             108
#define BSIM4v0_MOD_NFACTOR         109
#define BSIM4v0_MOD_XJ              110
#define BSIM4v0_MOD_VSAT            111
#define BSIM4v0_MOD_AT              112
#define BSIM4v0_MOD_A0              113
#define BSIM4v0_MOD_A1              114
#define BSIM4v0_MOD_A2              115
#define BSIM4v0_MOD_KETA            116   
#define BSIM4v0_MOD_NSUB            117
#define BSIM4v0_MOD_NDEP            118
#define BSIM4v0_MOD_NGATE           120
#define BSIM4v0_MOD_GAMMA1          121
#define BSIM4v0_MOD_GAMMA2          122
#define BSIM4v0_MOD_VBX             123
#define BSIM4v0_MOD_BINUNIT         124    
#define BSIM4v0_MOD_VBM             125
#define BSIM4v0_MOD_XT              126
#define BSIM4v0_MOD_K1              129
#define BSIM4v0_MOD_KT1             130
#define BSIM4v0_MOD_KT1L            131
#define BSIM4v0_MOD_K2              132
#define BSIM4v0_MOD_KT2             133
#define BSIM4v0_MOD_K3              134
#define BSIM4v0_MOD_K3B             135
#define BSIM4v0_MOD_W0              136
#define BSIM4v0_MOD_LPE0            137
#define BSIM4v0_MOD_DVT0            138
#define BSIM4v0_MOD_DVT1            139
#define BSIM4v0_MOD_DVT2            140
#define BSIM4v0_MOD_DVT0W           141
#define BSIM4v0_MOD_DVT1W           142
#define BSIM4v0_MOD_DVT2W           143
#define BSIM4v0_MOD_DROUT           144
#define BSIM4v0_MOD_DSUB            145
#define BSIM4v0_MOD_VTH0            146
#define BSIM4v0_MOD_UA              147
#define BSIM4v0_MOD_UA1             148
#define BSIM4v0_MOD_UB              149
#define BSIM4v0_MOD_UB1             150
#define BSIM4v0_MOD_UC              151
#define BSIM4v0_MOD_UC1             152
#define BSIM4v0_MOD_U0              153
#define BSIM4v0_MOD_UTE             154
#define BSIM4v0_MOD_VOFF            155
#define BSIM4v0_MOD_DELTA           156
#define BSIM4v0_MOD_RDSW            157
#define BSIM4v0_MOD_PRT             158
#define BSIM4v0_MOD_LDD             159
#define BSIM4v0_MOD_ETA             160
#define BSIM4v0_MOD_ETA0            161
#define BSIM4v0_MOD_ETAB            162
#define BSIM4v0_MOD_PCLM            163
#define BSIM4v0_MOD_PDIBL1          164
#define BSIM4v0_MOD_PDIBL2          165
#define BSIM4v0_MOD_PSCBE1          166
#define BSIM4v0_MOD_PSCBE2          167
#define BSIM4v0_MOD_PVAG            168
#define BSIM4v0_MOD_WR              169
#define BSIM4v0_MOD_DWG             170
#define BSIM4v0_MOD_DWB             171
#define BSIM4v0_MOD_B0              172
#define BSIM4v0_MOD_B1              173
#define BSIM4v0_MOD_ALPHA0          174
#define BSIM4v0_MOD_BETA0           175
#define BSIM4v0_MOD_PDIBLB          178
#define BSIM4v0_MOD_PRWG            179
#define BSIM4v0_MOD_PRWB            180
#define BSIM4v0_MOD_CDSCD           181
#define BSIM4v0_MOD_AGS             182
#define BSIM4v0_MOD_FRINGE          184
#define BSIM4v0_MOD_CGSL            186
#define BSIM4v0_MOD_CGDL            187
#define BSIM4v0_MOD_CKAPPAS         188
#define BSIM4v0_MOD_CF              189
#define BSIM4v0_MOD_CLC             190
#define BSIM4v0_MOD_CLE             191
#define BSIM4v0_MOD_PARAMCHK        192
#define BSIM4v0_MOD_VERSION         193
#define BSIM4v0_MOD_VFBCV           194
#define BSIM4v0_MOD_ACDE            195
#define BSIM4v0_MOD_MOIN            196
#define BSIM4v0_MOD_NOFF            197
#define BSIM4v0_MOD_IJTHDFWD        198
#define BSIM4v0_MOD_ALPHA1          199
#define BSIM4v0_MOD_VFB             200
#define BSIM4v0_MOD_TOXM            201
#define BSIM4v0_MOD_TCJ             202
#define BSIM4v0_MOD_TCJSW           203
#define BSIM4v0_MOD_TCJSWG          204
#define BSIM4v0_MOD_TPB             205
#define BSIM4v0_MOD_TPBSW           206
#define BSIM4v0_MOD_TPBSWG          207
#define BSIM4v0_MOD_VOFFCV          208
#define BSIM4v0_MOD_GBMIN           209
#define BSIM4v0_MOD_RBDB            210
#define BSIM4v0_MOD_RBSB            211
#define BSIM4v0_MOD_RBPB            212
#define BSIM4v0_MOD_RBPS            213
#define BSIM4v0_MOD_RBPD            214
#define BSIM4v0_MOD_DMCG            215
#define BSIM4v0_MOD_DMCI            216
#define BSIM4v0_MOD_DMDG            217
#define BSIM4v0_MOD_XGW             218
#define BSIM4v0_MOD_XGL             219
#define BSIM4v0_MOD_RSHG            220
#define BSIM4v0_MOD_NGCON           221
#define BSIM4v0_MOD_AGIDL           222
#define BSIM4v0_MOD_BGIDL           223
#define BSIM4v0_MOD_EGIDL           224
#define BSIM4v0_MOD_IJTHSFWD        225
#define BSIM4v0_MOD_XJBVD           226
#define BSIM4v0_MOD_XJBVS           227
#define BSIM4v0_MOD_BVD             228
#define BSIM4v0_MOD_BVS             229
#define BSIM4v0_MOD_TOXP            230
#define BSIM4v0_MOD_DTOX            231
#define BSIM4v0_MOD_XRCRG1          232
#define BSIM4v0_MOD_XRCRG2          233
#define BSIM4v0_MOD_EU              234
#define BSIM4v0_MOD_IJTHSREV        235
#define BSIM4v0_MOD_IJTHDREV        236
#define BSIM4v0_MOD_MINV            237
#define BSIM4v0_MOD_VOFFL           238
#define BSIM4v0_MOD_PDITS           239
#define BSIM4v0_MOD_PDITSD          240
#define BSIM4v0_MOD_PDITSL          241
#define BSIM4v0_MOD_TNOIA           242
#define BSIM4v0_MOD_TNOIB           243
#define BSIM4v0_MOD_NTNOI           244
#define BSIM4v0_MOD_FPROUT          245
#define BSIM4v0_MOD_LPEB            246
#define BSIM4v0_MOD_DVTP0           247
#define BSIM4v0_MOD_DVTP1           248
#define BSIM4v0_MOD_CGIDL           249
#define BSIM4v0_MOD_PHIN            250
#define BSIM4v0_MOD_RDSWMIN         251
#define BSIM4v0_MOD_RSW             252
#define BSIM4v0_MOD_RDW             253
#define BSIM4v0_MOD_RDWMIN          254
#define BSIM4v0_MOD_RSWMIN          255
#define BSIM4v0_MOD_NSD             256
#define BSIM4v0_MOD_CKAPPAD         257
#define BSIM4v0_MOD_DMCGT           258
#define BSIM4v0_MOD_AIGC            259
#define BSIM4v0_MOD_BIGC            260
#define BSIM4v0_MOD_CIGC            261
#define BSIM4v0_MOD_AIGBACC         262
#define BSIM4v0_MOD_BIGBACC         263
#define BSIM4v0_MOD_CIGBACC         264            
#define BSIM4v0_MOD_AIGBINV         265
#define BSIM4v0_MOD_BIGBINV         266
#define BSIM4v0_MOD_CIGBINV         267
#define BSIM4v0_MOD_NIGC            268
#define BSIM4v0_MOD_NIGBACC         269
#define BSIM4v0_MOD_NIGBINV         270
#define BSIM4v0_MOD_NTOX            271
#define BSIM4v0_MOD_TOXREF          272
#define BSIM4v0_MOD_EIGBINV         273
#define BSIM4v0_MOD_PIGCD           274
#define BSIM4v0_MOD_POXEDGE         275
#define BSIM4v0_MOD_EPSROX          276
#define BSIM4v0_MOD_AIGSD           277
#define BSIM4v0_MOD_BIGSD           278
#define BSIM4v0_MOD_CIGSD           279
#define BSIM4v0_MOD_JSWGS           280
#define BSIM4v0_MOD_JSWGD           281


/* Length dependence */
#define BSIM4v0_MOD_LCDSC            301
#define BSIM4v0_MOD_LCDSCB           302
#define BSIM4v0_MOD_LCIT             303
#define BSIM4v0_MOD_LNFACTOR         304
#define BSIM4v0_MOD_LXJ              305
#define BSIM4v0_MOD_LVSAT            306
#define BSIM4v0_MOD_LAT              307
#define BSIM4v0_MOD_LA0              308
#define BSIM4v0_MOD_LA1              309
#define BSIM4v0_MOD_LA2              310
#define BSIM4v0_MOD_LKETA            311
#define BSIM4v0_MOD_LNSUB            312
#define BSIM4v0_MOD_LNDEP            313
#define BSIM4v0_MOD_LNGATE           315
#define BSIM4v0_MOD_LGAMMA1          316
#define BSIM4v0_MOD_LGAMMA2          317
#define BSIM4v0_MOD_LVBX             318
#define BSIM4v0_MOD_LVBM             320
#define BSIM4v0_MOD_LXT              322
#define BSIM4v0_MOD_LK1              325
#define BSIM4v0_MOD_LKT1             326
#define BSIM4v0_MOD_LKT1L            327
#define BSIM4v0_MOD_LK2              328
#define BSIM4v0_MOD_LKT2             329
#define BSIM4v0_MOD_LK3              330
#define BSIM4v0_MOD_LK3B             331
#define BSIM4v0_MOD_LW0              332
#define BSIM4v0_MOD_LLPE0            333
#define BSIM4v0_MOD_LDVT0            334
#define BSIM4v0_MOD_LDVT1            335
#define BSIM4v0_MOD_LDVT2            336
#define BSIM4v0_MOD_LDVT0W           337
#define BSIM4v0_MOD_LDVT1W           338
#define BSIM4v0_MOD_LDVT2W           339
#define BSIM4v0_MOD_LDROUT           340
#define BSIM4v0_MOD_LDSUB            341
#define BSIM4v0_MOD_LVTH0            342
#define BSIM4v0_MOD_LUA              343
#define BSIM4v0_MOD_LUA1             344
#define BSIM4v0_MOD_LUB              345
#define BSIM4v0_MOD_LUB1             346
#define BSIM4v0_MOD_LUC              347
#define BSIM4v0_MOD_LUC1             348
#define BSIM4v0_MOD_LU0              349
#define BSIM4v0_MOD_LUTE             350
#define BSIM4v0_MOD_LVOFF            351
#define BSIM4v0_MOD_LDELTA           352
#define BSIM4v0_MOD_LRDSW            353
#define BSIM4v0_MOD_LPRT             354
#define BSIM4v0_MOD_LLDD             355
#define BSIM4v0_MOD_LETA             356
#define BSIM4v0_MOD_LETA0            357
#define BSIM4v0_MOD_LETAB            358
#define BSIM4v0_MOD_LPCLM            359
#define BSIM4v0_MOD_LPDIBL1          360
#define BSIM4v0_MOD_LPDIBL2          361
#define BSIM4v0_MOD_LPSCBE1          362
#define BSIM4v0_MOD_LPSCBE2          363
#define BSIM4v0_MOD_LPVAG            364
#define BSIM4v0_MOD_LWR              365
#define BSIM4v0_MOD_LDWG             366
#define BSIM4v0_MOD_LDWB             367
#define BSIM4v0_MOD_LB0              368
#define BSIM4v0_MOD_LB1              369
#define BSIM4v0_MOD_LALPHA0          370
#define BSIM4v0_MOD_LBETA0           371
#define BSIM4v0_MOD_LPDIBLB          374
#define BSIM4v0_MOD_LPRWG            375
#define BSIM4v0_MOD_LPRWB            376
#define BSIM4v0_MOD_LCDSCD           377
#define BSIM4v0_MOD_LAGS             378

#define BSIM4v0_MOD_LFRINGE          381
#define BSIM4v0_MOD_LCGSL            383
#define BSIM4v0_MOD_LCGDL            384
#define BSIM4v0_MOD_LCKAPPAS         385
#define BSIM4v0_MOD_LCF              386
#define BSIM4v0_MOD_LCLC             387
#define BSIM4v0_MOD_LCLE             388
#define BSIM4v0_MOD_LVFBCV           389
#define BSIM4v0_MOD_LACDE            390
#define BSIM4v0_MOD_LMOIN            391
#define BSIM4v0_MOD_LNOFF            392
#define BSIM4v0_MOD_LALPHA1          394
#define BSIM4v0_MOD_LVFB             395
#define BSIM4v0_MOD_LVOFFCV          396
#define BSIM4v0_MOD_LAGIDL           397
#define BSIM4v0_MOD_LBGIDL           398
#define BSIM4v0_MOD_LEGIDL           399
#define BSIM4v0_MOD_LXRCRG1          400
#define BSIM4v0_MOD_LXRCRG2          401
#define BSIM4v0_MOD_LEU              402
#define BSIM4v0_MOD_LMINV            403
#define BSIM4v0_MOD_LPDITS           404
#define BSIM4v0_MOD_LPDITSD          405
#define BSIM4v0_MOD_LFPROUT          406
#define BSIM4v0_MOD_LLPEB            407
#define BSIM4v0_MOD_LDVTP0           408
#define BSIM4v0_MOD_LDVTP1           409
#define BSIM4v0_MOD_LCGIDL           410
#define BSIM4v0_MOD_LPHIN            411
#define BSIM4v0_MOD_LRSW             412
#define BSIM4v0_MOD_LRDW             413
#define BSIM4v0_MOD_LNSD             414
#define BSIM4v0_MOD_LCKAPPAD         415
#define BSIM4v0_MOD_LAIGC            416
#define BSIM4v0_MOD_LBIGC            417
#define BSIM4v0_MOD_LCIGC            418
#define BSIM4v0_MOD_LAIGBACC         419            
#define BSIM4v0_MOD_LBIGBACC         420
#define BSIM4v0_MOD_LCIGBACC         421
#define BSIM4v0_MOD_LAIGBINV         422
#define BSIM4v0_MOD_LBIGBINV         423
#define BSIM4v0_MOD_LCIGBINV         424
#define BSIM4v0_MOD_LNIGC            425
#define BSIM4v0_MOD_LNIGBACC         426
#define BSIM4v0_MOD_LNIGBINV         427
#define BSIM4v0_MOD_LNTOX            428
#define BSIM4v0_MOD_LEIGBINV         429
#define BSIM4v0_MOD_LPIGCD           430
#define BSIM4v0_MOD_LPOXEDGE         431
#define BSIM4v0_MOD_LAIGSD           432
#define BSIM4v0_MOD_LBIGSD           433
#define BSIM4v0_MOD_LCIGSD           434


/* Width dependence */
#define BSIM4v0_MOD_WCDSC            481
#define BSIM4v0_MOD_WCDSCB           482
#define BSIM4v0_MOD_WCIT             483
#define BSIM4v0_MOD_WNFACTOR         484
#define BSIM4v0_MOD_WXJ              485
#define BSIM4v0_MOD_WVSAT            486
#define BSIM4v0_MOD_WAT              487
#define BSIM4v0_MOD_WA0              488
#define BSIM4v0_MOD_WA1              489
#define BSIM4v0_MOD_WA2              490
#define BSIM4v0_MOD_WKETA            491
#define BSIM4v0_MOD_WNSUB            492
#define BSIM4v0_MOD_WNDEP            493
#define BSIM4v0_MOD_WNGATE           495
#define BSIM4v0_MOD_WGAMMA1          496
#define BSIM4v0_MOD_WGAMMA2          497
#define BSIM4v0_MOD_WVBX             498
#define BSIM4v0_MOD_WVBM             500
#define BSIM4v0_MOD_WXT              502
#define BSIM4v0_MOD_WK1              505
#define BSIM4v0_MOD_WKT1             506
#define BSIM4v0_MOD_WKT1L            507
#define BSIM4v0_MOD_WK2              508
#define BSIM4v0_MOD_WKT2             509
#define BSIM4v0_MOD_WK3              510
#define BSIM4v0_MOD_WK3B             511
#define BSIM4v0_MOD_WW0              512
#define BSIM4v0_MOD_WLPE0            513
#define BSIM4v0_MOD_WDVT0            514
#define BSIM4v0_MOD_WDVT1            515
#define BSIM4v0_MOD_WDVT2            516
#define BSIM4v0_MOD_WDVT0W           517
#define BSIM4v0_MOD_WDVT1W           518
#define BSIM4v0_MOD_WDVT2W           519
#define BSIM4v0_MOD_WDROUT           520
#define BSIM4v0_MOD_WDSUB            521
#define BSIM4v0_MOD_WVTH0            522
#define BSIM4v0_MOD_WUA              523
#define BSIM4v0_MOD_WUA1             524
#define BSIM4v0_MOD_WUB              525
#define BSIM4v0_MOD_WUB1             526
#define BSIM4v0_MOD_WUC              527
#define BSIM4v0_MOD_WUC1             528
#define BSIM4v0_MOD_WU0              529
#define BSIM4v0_MOD_WUTE             530
#define BSIM4v0_MOD_WVOFF            531
#define BSIM4v0_MOD_WDELTA           532
#define BSIM4v0_MOD_WRDSW            533
#define BSIM4v0_MOD_WPRT             534
#define BSIM4v0_MOD_WLDD             535
#define BSIM4v0_MOD_WETA             536
#define BSIM4v0_MOD_WETA0            537
#define BSIM4v0_MOD_WETAB            538
#define BSIM4v0_MOD_WPCLM            539
#define BSIM4v0_MOD_WPDIBL1          540
#define BSIM4v0_MOD_WPDIBL2          541
#define BSIM4v0_MOD_WPSCBE1          542
#define BSIM4v0_MOD_WPSCBE2          543
#define BSIM4v0_MOD_WPVAG            544
#define BSIM4v0_MOD_WWR              545
#define BSIM4v0_MOD_WDWG             546
#define BSIM4v0_MOD_WDWB             547
#define BSIM4v0_MOD_WB0              548
#define BSIM4v0_MOD_WB1              549
#define BSIM4v0_MOD_WALPHA0          550
#define BSIM4v0_MOD_WBETA0           551
#define BSIM4v0_MOD_WPDIBLB          554
#define BSIM4v0_MOD_WPRWG            555
#define BSIM4v0_MOD_WPRWB            556
#define BSIM4v0_MOD_WCDSCD           557
#define BSIM4v0_MOD_WAGS             558

#define BSIM4v0_MOD_WFRINGE          561
#define BSIM4v0_MOD_WCGSL            563
#define BSIM4v0_MOD_WCGDL            564
#define BSIM4v0_MOD_WCKAPPAS         565
#define BSIM4v0_MOD_WCF              566
#define BSIM4v0_MOD_WCLC             567
#define BSIM4v0_MOD_WCLE             568
#define BSIM4v0_MOD_WVFBCV           569
#define BSIM4v0_MOD_WACDE            570
#define BSIM4v0_MOD_WMOIN            571
#define BSIM4v0_MOD_WNOFF            572
#define BSIM4v0_MOD_WALPHA1          574
#define BSIM4v0_MOD_WVFB             575
#define BSIM4v0_MOD_WVOFFCV          576
#define BSIM4v0_MOD_WAGIDL           577
#define BSIM4v0_MOD_WBGIDL           578
#define BSIM4v0_MOD_WEGIDL           579
#define BSIM4v0_MOD_WXRCRG1          580
#define BSIM4v0_MOD_WXRCRG2          581
#define BSIM4v0_MOD_WEU              582
#define BSIM4v0_MOD_WMINV            583
#define BSIM4v0_MOD_WPDITS           584
#define BSIM4v0_MOD_WPDITSD          585
#define BSIM4v0_MOD_WFPROUT          586
#define BSIM4v0_MOD_WLPEB            587
#define BSIM4v0_MOD_WDVTP0           588
#define BSIM4v0_MOD_WDVTP1           589
#define BSIM4v0_MOD_WCGIDL           590
#define BSIM4v0_MOD_WPHIN            591
#define BSIM4v0_MOD_WRSW             592
#define BSIM4v0_MOD_WRDW             593
#define BSIM4v0_MOD_WNSD             594
#define BSIM4v0_MOD_WCKAPPAD         595
#define BSIM4v0_MOD_WAIGC            596
#define BSIM4v0_MOD_WBIGC            597
#define BSIM4v0_MOD_WCIGC            598
#define BSIM4v0_MOD_WAIGBACC         599            
#define BSIM4v0_MOD_WBIGBACC         600
#define BSIM4v0_MOD_WCIGBACC         601
#define BSIM4v0_MOD_WAIGBINV         602
#define BSIM4v0_MOD_WBIGBINV         603
#define BSIM4v0_MOD_WCIGBINV         604
#define BSIM4v0_MOD_WNIGC            605
#define BSIM4v0_MOD_WNIGBACC         606
#define BSIM4v0_MOD_WNIGBINV         607
#define BSIM4v0_MOD_WNTOX            608
#define BSIM4v0_MOD_WEIGBINV         609
#define BSIM4v0_MOD_WPIGCD           610
#define BSIM4v0_MOD_WPOXEDGE         611
#define BSIM4v0_MOD_WAIGSD           612
#define BSIM4v0_MOD_WBIGSD           613
#define BSIM4v0_MOD_WCIGSD           614


/* Cross-term dependence */
#define BSIM4v0_MOD_PCDSC            661
#define BSIM4v0_MOD_PCDSCB           662
#define BSIM4v0_MOD_PCIT             663
#define BSIM4v0_MOD_PNFACTOR         664
#define BSIM4v0_MOD_PXJ              665
#define BSIM4v0_MOD_PVSAT            666
#define BSIM4v0_MOD_PAT              667
#define BSIM4v0_MOD_PA0              668
#define BSIM4v0_MOD_PA1              669
#define BSIM4v0_MOD_PA2              670
#define BSIM4v0_MOD_PKETA            671
#define BSIM4v0_MOD_PNSUB            672
#define BSIM4v0_MOD_PNDEP            673
#define BSIM4v0_MOD_PNGATE           675
#define BSIM4v0_MOD_PGAMMA1          676
#define BSIM4v0_MOD_PGAMMA2          677
#define BSIM4v0_MOD_PVBX             678

#define BSIM4v0_MOD_PVBM             680

#define BSIM4v0_MOD_PXT              682
#define BSIM4v0_MOD_PK1              685
#define BSIM4v0_MOD_PKT1             686
#define BSIM4v0_MOD_PKT1L            687
#define BSIM4v0_MOD_PK2              688
#define BSIM4v0_MOD_PKT2             689
#define BSIM4v0_MOD_PK3              690
#define BSIM4v0_MOD_PK3B             691
#define BSIM4v0_MOD_PW0              692
#define BSIM4v0_MOD_PLPE0            693

#define BSIM4v0_MOD_PDVT0            694
#define BSIM4v0_MOD_PDVT1            695
#define BSIM4v0_MOD_PDVT2            696

#define BSIM4v0_MOD_PDVT0W           697
#define BSIM4v0_MOD_PDVT1W           698
#define BSIM4v0_MOD_PDVT2W           699

#define BSIM4v0_MOD_PDROUT           700
#define BSIM4v0_MOD_PDSUB            701
#define BSIM4v0_MOD_PVTH0            702
#define BSIM4v0_MOD_PUA              703
#define BSIM4v0_MOD_PUA1             704
#define BSIM4v0_MOD_PUB              705
#define BSIM4v0_MOD_PUB1             706
#define BSIM4v0_MOD_PUC              707
#define BSIM4v0_MOD_PUC1             708
#define BSIM4v0_MOD_PU0              709
#define BSIM4v0_MOD_PUTE             710
#define BSIM4v0_MOD_PVOFF            711
#define BSIM4v0_MOD_PDELTA           712
#define BSIM4v0_MOD_PRDSW            713
#define BSIM4v0_MOD_PPRT             714
#define BSIM4v0_MOD_PLDD             715
#define BSIM4v0_MOD_PETA             716
#define BSIM4v0_MOD_PETA0            717
#define BSIM4v0_MOD_PETAB            718
#define BSIM4v0_MOD_PPCLM            719
#define BSIM4v0_MOD_PPDIBL1          720
#define BSIM4v0_MOD_PPDIBL2          721
#define BSIM4v0_MOD_PPSCBE1          722
#define BSIM4v0_MOD_PPSCBE2          723
#define BSIM4v0_MOD_PPVAG            724
#define BSIM4v0_MOD_PWR              725
#define BSIM4v0_MOD_PDWG             726
#define BSIM4v0_MOD_PDWB             727
#define BSIM4v0_MOD_PB0              728
#define BSIM4v0_MOD_PB1              729
#define BSIM4v0_MOD_PALPHA0          730
#define BSIM4v0_MOD_PBETA0           731
#define BSIM4v0_MOD_PPDIBLB          734

#define BSIM4v0_MOD_PPRWG            735
#define BSIM4v0_MOD_PPRWB            736

#define BSIM4v0_MOD_PCDSCD           737
#define BSIM4v0_MOD_PAGS             738

#define BSIM4v0_MOD_PFRINGE          741
#define BSIM4v0_MOD_PCGSL            743
#define BSIM4v0_MOD_PCGDL            744
#define BSIM4v0_MOD_PCKAPPAS         745
#define BSIM4v0_MOD_PCF              746
#define BSIM4v0_MOD_PCLC             747
#define BSIM4v0_MOD_PCLE             748
#define BSIM4v0_MOD_PVFBCV           749
#define BSIM4v0_MOD_PACDE            750
#define BSIM4v0_MOD_PMOIN            751
#define BSIM4v0_MOD_PNOFF            752
#define BSIM4v0_MOD_PALPHA1          754
#define BSIM4v0_MOD_PVFB             755
#define BSIM4v0_MOD_PVOFFCV          756
#define BSIM4v0_MOD_PAGIDL           757
#define BSIM4v0_MOD_PBGIDL           758
#define BSIM4v0_MOD_PEGIDL           759
#define BSIM4v0_MOD_PXRCRG1          760
#define BSIM4v0_MOD_PXRCRG2          761
#define BSIM4v0_MOD_PEU              762
#define BSIM4v0_MOD_PMINV            763
#define BSIM4v0_MOD_PPDITS           764
#define BSIM4v0_MOD_PPDITSD          765
#define BSIM4v0_MOD_PFPROUT          766
#define BSIM4v0_MOD_PLPEB            767
#define BSIM4v0_MOD_PDVTP0           768
#define BSIM4v0_MOD_PDVTP1           769
#define BSIM4v0_MOD_PCGIDL           770
#define BSIM4v0_MOD_PPHIN            771
#define BSIM4v0_MOD_PRSW             772
#define BSIM4v0_MOD_PRDW             773
#define BSIM4v0_MOD_PNSD             774
#define BSIM4v0_MOD_PCKAPPAD         775
#define BSIM4v0_MOD_PAIGC            776
#define BSIM4v0_MOD_PBIGC            777
#define BSIM4v0_MOD_PCIGC            778
#define BSIM4v0_MOD_PAIGBACC         779            
#define BSIM4v0_MOD_PBIGBACC         780
#define BSIM4v0_MOD_PCIGBACC         781
#define BSIM4v0_MOD_PAIGBINV         782
#define BSIM4v0_MOD_PBIGBINV         783
#define BSIM4v0_MOD_PCIGBINV         784
#define BSIM4v0_MOD_PNIGC            785
#define BSIM4v0_MOD_PNIGBACC         786
#define BSIM4v0_MOD_PNIGBINV         787
#define BSIM4v0_MOD_PNTOX            788
#define BSIM4v0_MOD_PEIGBINV         789
#define BSIM4v0_MOD_PPIGCD           790
#define BSIM4v0_MOD_PPOXEDGE         791
#define BSIM4v0_MOD_PAIGSD           792
#define BSIM4v0_MOD_PBIGSD           793
#define BSIM4v0_MOD_PCIGSD           794

#define BSIM4v0_MOD_TNOM             831
#define BSIM4v0_MOD_CGSO             832
#define BSIM4v0_MOD_CGDO             833
#define BSIM4v0_MOD_CGBO             834
#define BSIM4v0_MOD_XPART            835
#define BSIM4v0_MOD_RSH              836
#define BSIM4v0_MOD_JSS              837
#define BSIM4v0_MOD_PBS              838
#define BSIM4v0_MOD_MJS              839
#define BSIM4v0_MOD_PBSWS            840
#define BSIM4v0_MOD_MJSWS            841
#define BSIM4v0_MOD_CJS              842
#define BSIM4v0_MOD_CJSWS            843
#define BSIM4v0_MOD_NMOS             844
#define BSIM4v0_MOD_PMOS             845
#define BSIM4v0_MOD_NOIA             846
#define BSIM4v0_MOD_NOIB             847
#define BSIM4v0_MOD_NOIC             848
#define BSIM4v0_MOD_LINT             849
#define BSIM4v0_MOD_LL               850
#define BSIM4v0_MOD_LLN              851
#define BSIM4v0_MOD_LW               852
#define BSIM4v0_MOD_LWN              853
#define BSIM4v0_MOD_LWL              854
#define BSIM4v0_MOD_LMIN             855
#define BSIM4v0_MOD_LMAX             856
#define BSIM4v0_MOD_WINT             857
#define BSIM4v0_MOD_WL               858
#define BSIM4v0_MOD_WLN              859
#define BSIM4v0_MOD_WW               860
#define BSIM4v0_MOD_WWN              861
#define BSIM4v0_MOD_WWL              862
#define BSIM4v0_MOD_WMIN             863
#define BSIM4v0_MOD_WMAX             864
#define BSIM4v0_MOD_DWC              865
#define BSIM4v0_MOD_DLC              866
#define BSIM4v0_MOD_EM               867
#define BSIM4v0_MOD_EF               868
#define BSIM4v0_MOD_AF               869
#define BSIM4v0_MOD_KF               870
#define BSIM4v0_MOD_NJS              871
#define BSIM4v0_MOD_XTIS             872
#define BSIM4v0_MOD_PBSWGS           873
#define BSIM4v0_MOD_MJSWGS           874
#define BSIM4v0_MOD_CJSWGS           875
#define BSIM4v0_MOD_JSWS             876
#define BSIM4v0_MOD_LLC              877
#define BSIM4v0_MOD_LWC              878
#define BSIM4v0_MOD_LWLC             879
#define BSIM4v0_MOD_WLC              880
#define BSIM4v0_MOD_WWC              881
#define BSIM4v0_MOD_WWLC             882
#define BSIM4v0_MOD_DWJ              883
#define BSIM4v0_MOD_JSD              884
#define BSIM4v0_MOD_PBD              885
#define BSIM4v0_MOD_MJD              886
#define BSIM4v0_MOD_PBSWD            887
#define BSIM4v0_MOD_MJSWD            888
#define BSIM4v0_MOD_CJD              889
#define BSIM4v0_MOD_CJSWD            890
#define BSIM4v0_MOD_NJD              891
#define BSIM4v0_MOD_XTID             892
#define BSIM4v0_MOD_PBSWGD           893
#define BSIM4v0_MOD_MJSWGD           894
#define BSIM4v0_MOD_CJSWGD           895
#define BSIM4v0_MOD_JSWD             896
#define BSIM4v0_MOD_DLCIG            897

/* device questions */
#define BSIM4v0_DNODE                945
#define BSIM4v0_GNODEEXT             946
#define BSIM4v0_SNODE                947
#define BSIM4v0_BNODE                948
#define BSIM4v0_DNODEPRIME           949
#define BSIM4v0_GNODEPRIME           950
#define BSIM4v0_GNODEMIDE            951
#define BSIM4v0_GNODEMID             952
#define BSIM4v0_SNODEPRIME           953
#define BSIM4v0_BNODEPRIME           954
#define BSIM4v0_DBNODE               955
#define BSIM4v0_SBNODE               956
#define BSIM4v0_VBD                  957
#define BSIM4v0_VBS                  958
#define BSIM4v0_VGS                  959
#define BSIM4v0_VDS                  960
#define BSIM4v0_CD                   961
#define BSIM4v0_CBS                  962
#define BSIM4v0_CBD                  963
#define BSIM4v0_GM                   964
#define BSIM4v0_GDS                  965
#define BSIM4v0_GMBS                 966
#define BSIM4v0_GBD                  967
#define BSIM4v0_GBS                  968
#define BSIM4v0_QB                   969
#define BSIM4v0_CQB                  970
#define BSIM4v0_QG                   971
#define BSIM4v0_CQG                  972
#define BSIM4v0_QD                   973
#define BSIM4v0_CQD                  974
#define BSIM4v0_CGG                  975
#define BSIM4v0_CGD                  976
#define BSIM4v0_CGS                  977
#define BSIM4v0_CBG                  978
#define BSIM4v0_CAPBD                979
#define BSIM4v0_CQBD                 980
#define BSIM4v0_CAPBS                981
#define BSIM4v0_CQBS                 982
#define BSIM4v0_CDG                  983
#define BSIM4v0_CDD                  984
#define BSIM4v0_CDS                  985
#define BSIM4v0_VON                  986
#define BSIM4v0_VDSAT                987
#define BSIM4v0_QBS                  988
#define BSIM4v0_QBD                  989
#define BSIM4v0_SOURCECONDUCT        990
#define BSIM4v0_DRAINCONDUCT         991
#define BSIM4v0_CBDB                 992
#define BSIM4v0_CBSB                 993

#include "bsim4v0ext.h"

extern void BSIM4v0evaluate(double,double,double,BSIM4v0instance*,BSIM4v0model*,
        double*,double*,double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, CKTcircuit*);
extern int BSIM4v0debug(BSIM4v0model*, BSIM4v0instance*, CKTcircuit*, int);
extern int BSIM4v0checkModel(BSIM4v0model*, BSIM4v0instance*, CKTcircuit*);
extern int BSIM4v0PAeffGeo(double, int, int, double, double, double, double, double *, double *, double *, double *);
extern int BSIM4v0RdseffGeo(double, int, int, int, double, double, double, double, double, int, double *);

#endif /*BSIM4v0*/
