/**********
Copyright 2001 Regents of the University of California.  All rights reserved.
Author: 2000 Weidong Liu.
Modified by Xuemei Xi October 2001
File: bsim4def.h
**********/

#ifndef BSIM4
#define BSIM4

#include "ifsim.h"
#include "gendefs.h"
#include "cktdefs.h"
#include "complex.h"
#include "noisedef.h"         

typedef struct sBSIM4instance
{
    struct sBSIM4model *BSIM4modPtr;
    struct sBSIM4instance *BSIM4nextInstance;
    IFuid BSIM4name;
    int BSIM4states;     /* index into state table for this device */
    int BSIM4dNode;
    int BSIM4gNodeExt;
    int BSIM4sNode;
    int BSIM4bNode;
    int BSIM4dNodePrime;
    int BSIM4gNodePrime;
    int BSIM4gNodeMid;
    int BSIM4sNodePrime;
    int BSIM4bNodePrime;
    int BSIM4dbNode;
    int BSIM4sbNode;
    int BSIM4qNode;

    double BSIM4ueff;
    double BSIM4thetavth; 
    double BSIM4von;
    double BSIM4vdsat;
    double BSIM4cgdo;
    double BSIM4qgdo;
    double BSIM4cgso;
    double BSIM4qgso;
    double BSIM4grbsb;
    double BSIM4grbdb;
    double BSIM4grbpb;
    double BSIM4grbps;
    double BSIM4grbpd;

    double BSIM4vjsmFwd;
    double BSIM4vjsmRev;
    double BSIM4vjdmFwd;
    double BSIM4vjdmRev;
    double BSIM4XExpBVS;
    double BSIM4XExpBVD;
    double BSIM4SslpFwd;
    double BSIM4SslpRev;
    double BSIM4DslpFwd;
    double BSIM4DslpRev;
    double BSIM4IVjsmFwd;
    double BSIM4IVjsmRev;
    double BSIM4IVjdmFwd;
    double BSIM4IVjdmRev;

    double BSIM4grgeltd;
    double BSIM4Pseff;
    double BSIM4Pdeff;
    double BSIM4Aseff;
    double BSIM4Adeff;

    double BSIM4l;
    double BSIM4w;
    double BSIM4drainArea;
    double BSIM4sourceArea;
    double BSIM4drainSquares;
    double BSIM4sourceSquares;
    double BSIM4drainPerimeter;
    double BSIM4sourcePerimeter;
    double BSIM4sourceConductance;
    double BSIM4drainConductance;
    double BSIM4rbdb;
    double BSIM4rbsb;
    double BSIM4rbpb;
    double BSIM4rbps;
    double BSIM4rbpd;

    double BSIM4icVDS;
    double BSIM4icVGS;
    double BSIM4icVBS;
    double BSIM4nf;
    int BSIM4off;
    int BSIM4mode;
    int BSIM4trnqsMod;
    int BSIM4acnqsMod;
    int BSIM4rbodyMod;
    int BSIM4rgateMod;
    int BSIM4geoMod;
    int BSIM4rgeoMod;
    int BSIM4min;

    /* OP point */
    double BSIM4Vgsteff;
    double BSIM4vgs_eff;
    double BSIM4vgd_eff;
    double BSIM4dvgs_eff_dvg;
    double BSIM4dvgd_eff_dvg;
    double BSIM4Vdseff;
    double BSIM4nstar;
    double BSIM4Abulk;
    double BSIM4EsatL;
    double BSIM4AbovVgst2Vtm;
    double BSIM4qinv;
    double BSIM4cd;
    double BSIM4cbs;
    double BSIM4cbd;
    double BSIM4csub;
    double BSIM4Igidl;
    double BSIM4Igisl;
    double BSIM4gm;
    double BSIM4gds;
    double BSIM4gmbs;
    double BSIM4gbd;
    double BSIM4gbs;

    double BSIM4gbbs;
    double BSIM4gbgs;
    double BSIM4gbds;
    double BSIM4ggidld;
    double BSIM4ggidlg;
    double BSIM4ggidls;
    double BSIM4ggidlb;
    double BSIM4ggisld;
    double BSIM4ggislg;
    double BSIM4ggisls;
    double BSIM4ggislb;

    double BSIM4Igcs;
    double BSIM4gIgcsg;
    double BSIM4gIgcsd;
    double BSIM4gIgcss;
    double BSIM4gIgcsb;
    double BSIM4Igcd;
    double BSIM4gIgcdg;
    double BSIM4gIgcdd;
    double BSIM4gIgcds;
    double BSIM4gIgcdb;

    double BSIM4Igs;
    double BSIM4gIgsg;
    double BSIM4gIgss;
    double BSIM4Igd;
    double BSIM4gIgdg;
    double BSIM4gIgdd;

    double BSIM4Igb;
    double BSIM4gIgbg;
    double BSIM4gIgbd;
    double BSIM4gIgbs;
    double BSIM4gIgbb;

    double BSIM4grdsw;
    double BSIM4IdovVds;
    double BSIM4gcrg;
    double BSIM4gcrgd;
    double BSIM4gcrgg;
    double BSIM4gcrgs;
    double BSIM4gcrgb;

    double BSIM4gstot;
    double BSIM4gstotd;
    double BSIM4gstotg;
    double BSIM4gstots;
    double BSIM4gstotb;

    double BSIM4gdtot;
    double BSIM4gdtotd;
    double BSIM4gdtotg;
    double BSIM4gdtots;
    double BSIM4gdtotb;

    double BSIM4cggb;
    double BSIM4cgdb;
    double BSIM4cgsb;
    double BSIM4cbgb;
    double BSIM4cbdb;
    double BSIM4cbsb;
    double BSIM4cdgb;
    double BSIM4cddb;
    double BSIM4cdsb;
    double BSIM4csgb;
    double BSIM4csdb;
    double BSIM4cssb;
    double BSIM4cgbb;
    double BSIM4cdbb;
    double BSIM4csbb;
    double BSIM4cbbb;
    double BSIM4capbd;
    double BSIM4capbs;

    double BSIM4cqgb;
    double BSIM4cqdb;
    double BSIM4cqsb;
    double BSIM4cqbb;

    double BSIM4qgate;
    double BSIM4qbulk;
    double BSIM4qdrn;
    double BSIM4qsrc;

    double BSIM4qchqs;
    double BSIM4taunet;
    double BSIM4gtau;
    double BSIM4gtg;
    double BSIM4gtd;
    double BSIM4gts;
    double BSIM4gtb;

    struct bsim4SizeDependParam  *pParam;

    unsigned BSIM4lGiven :1;
    unsigned BSIM4wGiven :1;
    unsigned BSIM4nfGiven :1;
    unsigned BSIM4minGiven :1;
    unsigned BSIM4drainAreaGiven :1;
    unsigned BSIM4sourceAreaGiven    :1;
    unsigned BSIM4drainSquaresGiven  :1;
    unsigned BSIM4sourceSquaresGiven :1;
    unsigned BSIM4drainPerimeterGiven    :1;
    unsigned BSIM4sourcePerimeterGiven   :1;
    unsigned BSIM4rbdbGiven   :1;
    unsigned BSIM4rbsbGiven   :1;
    unsigned BSIM4rbpbGiven   :1;
    unsigned BSIM4rbpdGiven   :1;
    unsigned BSIM4rbpsGiven   :1;
    unsigned BSIM4icVDSGiven :1;
    unsigned BSIM4icVGSGiven :1;
    unsigned BSIM4icVBSGiven :1;
    unsigned BSIM4trnqsModGiven :1;
    unsigned BSIM4acnqsModGiven :1;
    unsigned BSIM4rbodyModGiven :1;
    unsigned BSIM4rgateModGiven :1;
    unsigned BSIM4geoModGiven :1;
    unsigned BSIM4rgeoModGiven :1;

    double *BSIM4DPdPtr;
    double *BSIM4DPdpPtr;
    double *BSIM4DPgpPtr;
    double *BSIM4DPgmPtr;
    double *BSIM4DPspPtr;
    double *BSIM4DPbpPtr;
    double *BSIM4DPdbPtr;

    double *BSIM4DdPtr;
    double *BSIM4DdpPtr;

    double *BSIM4GPdpPtr;
    double *BSIM4GPgpPtr;
    double *BSIM4GPgmPtr;
    double *BSIM4GPgePtr;
    double *BSIM4GPspPtr;
    double *BSIM4GPbpPtr;

    double *BSIM4GMdpPtr;
    double *BSIM4GMgpPtr;
    double *BSIM4GMgmPtr;
    double *BSIM4GMgePtr;
    double *BSIM4GMspPtr;
    double *BSIM4GMbpPtr;

    double *BSIM4GEdpPtr;
    double *BSIM4GEgpPtr;
    double *BSIM4GEgmPtr;
    double *BSIM4GEgePtr;
    double *BSIM4GEspPtr;
    double *BSIM4GEbpPtr;

    double *BSIM4SPdpPtr;
    double *BSIM4SPgpPtr;
    double *BSIM4SPgmPtr;
    double *BSIM4SPsPtr;
    double *BSIM4SPspPtr;
    double *BSIM4SPbpPtr;
    double *BSIM4SPsbPtr;

    double *BSIM4SspPtr;
    double *BSIM4SsPtr;

    double *BSIM4BPdpPtr;
    double *BSIM4BPgpPtr;
    double *BSIM4BPgmPtr;
    double *BSIM4BPspPtr;
    double *BSIM4BPdbPtr;
    double *BSIM4BPbPtr;
    double *BSIM4BPsbPtr;
    double *BSIM4BPbpPtr;

    double *BSIM4DBdpPtr;
    double *BSIM4DBdbPtr;
    double *BSIM4DBbpPtr;
    double *BSIM4DBbPtr;

    double *BSIM4SBspPtr;
    double *BSIM4SBbpPtr;
    double *BSIM4SBbPtr;
    double *BSIM4SBsbPtr;

    double *BSIM4BdbPtr;
    double *BSIM4BbpPtr;
    double *BSIM4BsbPtr;
    double *BSIM4BbPtr;

    double *BSIM4DgpPtr;
    double *BSIM4DspPtr;
    double *BSIM4DbpPtr;
    double *BSIM4SdpPtr;
    double *BSIM4SgpPtr;
    double *BSIM4SbpPtr;

    double *BSIM4QdpPtr;
    double *BSIM4QgpPtr;
    double *BSIM4QspPtr;
    double *BSIM4QbpPtr;
    double *BSIM4QqPtr;
    double *BSIM4DPqPtr;
    double *BSIM4GPqPtr;
    double *BSIM4SPqPtr;


#define BSIM4vbd BSIM4states+ 0
#define BSIM4vbs BSIM4states+ 1
#define BSIM4vgs BSIM4states+ 2
#define BSIM4vds BSIM4states+ 3
#define BSIM4vdbs BSIM4states+ 4
#define BSIM4vdbd BSIM4states+ 5
#define BSIM4vsbs BSIM4states+ 6
#define BSIM4vges BSIM4states+ 7
#define BSIM4vgms BSIM4states+ 8
#define BSIM4vses BSIM4states+ 9
#define BSIM4vdes BSIM4states+ 10

#define BSIM4qb BSIM4states+ 11
#define BSIM4cqb BSIM4states+ 12
#define BSIM4qg BSIM4states+ 13
#define BSIM4cqg BSIM4states+ 14
#define BSIM4qd BSIM4states+ 15
#define BSIM4cqd BSIM4states+ 16
#define BSIM4qgmid BSIM4states+ 17
#define BSIM4cqgmid BSIM4states+ 18

#define BSIM4qbs  BSIM4states+ 19
#define BSIM4cqbs  BSIM4states+ 20
#define BSIM4qbd  BSIM4states+ 21
#define BSIM4cqbd  BSIM4states+ 22

#define BSIM4qcheq BSIM4states+ 23
#define BSIM4cqcheq BSIM4states+ 24
#define BSIM4qcdump BSIM4states+ 25
#define BSIM4cqcdump BSIM4states+ 26
#define BSIM4qdef BSIM4states+ 27
#define BSIM4qs BSIM4states+ 28

#define BSIM4numStates 29


/* indices to the array of BSIM4 NOISE SOURCES */

#define BSIM4RDNOIZ       0
#define BSIM4RSNOIZ       1
#define BSIM4RGNOIZ       2
#define BSIM4RBPSNOIZ     3
#define BSIM4RBPDNOIZ     4
#define BSIM4RBPBNOIZ     5
#define BSIM4RBSBNOIZ     6
#define BSIM4RBDBNOIZ     7
#define BSIM4IDNOIZ       8
#define BSIM4FLNOIZ       9
#define BSIM4IGSNOIZ      10
#define BSIM4IGDNOIZ      11
#define BSIM4IGBNOIZ      12
#define BSIM4TOTNOIZ      13

#define BSIM4NSRCS        14  /* Number of BSIM4 noise sources */

#ifndef NONOISE
    double BSIM4nVar[NSTATVARS][BSIM4NSRCS];
#else /* NONOISE */
        double **BSIM4nVar;
#endif /* NONOISE */

} BSIM4instance ;

struct bsim4SizeDependParam
{
    double Width;
    double Length;
    double NFinger;

    double BSIM4cdsc;           
    double BSIM4cdscb;    
    double BSIM4cdscd;       
    double BSIM4cit;           
    double BSIM4nfactor;      
    double BSIM4xj;
    double BSIM4vsat;         
    double BSIM4at;         
    double BSIM4a0;   
    double BSIM4ags;      
    double BSIM4a1;         
    double BSIM4a2;         
    double BSIM4keta;     
    double BSIM4nsub;
    double BSIM4ndep;        
    double BSIM4nsd;
    double BSIM4phin;
    double BSIM4ngate;        
    double BSIM4gamma1;      
    double BSIM4gamma2;     
    double BSIM4vbx;      
    double BSIM4vbi;       
    double BSIM4vbm;       
    double BSIM4vbsc;       
    double BSIM4xt;       
    double BSIM4phi;
    double BSIM4litl;
    double BSIM4k1;
    double BSIM4kt1;
    double BSIM4kt1l;
    double BSIM4kt2;
    double BSIM4k2;
    double BSIM4k3;
    double BSIM4k3b;
    double BSIM4w0;
    double BSIM4dvtp0;
    double BSIM4dvtp1;
    double BSIM4lpe0;
    double BSIM4lpeb;
    double BSIM4dvt0;      
    double BSIM4dvt1;      
    double BSIM4dvt2;      
    double BSIM4dvt0w;      
    double BSIM4dvt1w;      
    double BSIM4dvt2w;      
    double BSIM4drout;      
    double BSIM4dsub;      
    double BSIM4vth0;
    double BSIM4ua;
    double BSIM4ua1;
    double BSIM4ub;
    double BSIM4ub1;
    double BSIM4uc;
    double BSIM4uc1;
    double BSIM4u0;
    double BSIM4eu;
    double BSIM4ute;
    double BSIM4voff;
    double BSIM4minv;
    double BSIM4vfb;
    double BSIM4delta;
    double BSIM4rdsw;       
    double BSIM4rds0;       
    double BSIM4rs0;
    double BSIM4rd0;
    double BSIM4rsw;
    double BSIM4rdw;
    double BSIM4prwg;       
    double BSIM4prwb;       
    double BSIM4prt;       
    double BSIM4eta0;         
    double BSIM4etab;         
    double BSIM4pclm;      
    double BSIM4pdibl1;      
    double BSIM4pdibl2;      
    double BSIM4pdiblb;      
    double BSIM4fprout;
    double BSIM4pdits;
    double BSIM4pditsd;
    double BSIM4pscbe1;       
    double BSIM4pscbe2;       
    double BSIM4pvag;       
    double BSIM4wr;
    double BSIM4dwg;
    double BSIM4dwb;
    double BSIM4b0;
    double BSIM4b1;
    double BSIM4alpha0;
    double BSIM4alpha1;
    double BSIM4beta0;
    double BSIM4agidl;
    double BSIM4bgidl;
    double BSIM4cgidl;
    double BSIM4egidl;
    double BSIM4aigc;
    double BSIM4bigc;
    double BSIM4cigc;
    double BSIM4aigsd;
    double BSIM4bigsd;
    double BSIM4cigsd;
    double BSIM4aigbacc;
    double BSIM4bigbacc;
    double BSIM4cigbacc;
    double BSIM4aigbinv;
    double BSIM4bigbinv;
    double BSIM4cigbinv;
    double BSIM4nigc;
    double BSIM4nigbacc;
    double BSIM4nigbinv;
    double BSIM4ntox;
    double BSIM4eigbinv;
    double BSIM4pigcd;
    double BSIM4poxedge;
    double BSIM4xrcrg1;
    double BSIM4xrcrg2;
    double BSIM4fpkt;
    double BSIM4plcr;
    double BSIM4plcrl;
    double BSIM4plcrd;


    /* CV model */
    double BSIM4cgsl;
    double BSIM4cgdl;
    double BSIM4ckappas;
    double BSIM4ckappad;
    double BSIM4cf;
    double BSIM4clc;
    double BSIM4cle;
    double BSIM4vfbcv;
    double BSIM4noff;
    double BSIM4voffcv;
    double BSIM4acde;
    double BSIM4moin;

/* Pre-calculated constants */

    double BSIM4dw;
    double BSIM4dl;
    double BSIM4leff;
    double BSIM4weff;

    double BSIM4dwc;
    double BSIM4dlc;
    double BSIM4dlcig;
    double BSIM4dwj;
    double BSIM4leffCV;
    double BSIM4weffCV;
    double BSIM4weffCJ;
    double BSIM4abulkCVfactor;
    double BSIM4cgso;
    double BSIM4cgdo;
    double BSIM4cgbo;

    double BSIM4u0temp;       
    double BSIM4vsattemp;   
    double BSIM4sqrtPhi;   
    double BSIM4phis3;   
    double BSIM4Xdep0;          
    double BSIM4sqrtXdep0;          
    double BSIM4theta0vb0;
    double BSIM4thetaRout; 
    double BSIM4mstar;
    double BSIM4voffcbn;
    double BSIM4rdswmin;
    double BSIM4rdwmin;
    double BSIM4rswmin;
    double BSIM4vfbsd;

    double BSIM4cof1;
    double BSIM4cof2;
    double BSIM4cof3;
    double BSIM4cof4;
    double BSIM4cdep0;
    double BSIM4vfbzb;
    double BSIM4vtfbphi1;
    double BSIM4vtfbphi2;
    double BSIM4ToxRatio;
    double BSIM4Aechvb;
    double BSIM4Bechvb;
    double BSIM4ToxRatioEdge;
    double BSIM4AechvbEdge;
    double BSIM4BechvbEdge;
    double BSIM4ldeb;
    double BSIM4k1ox;
    double BSIM4k2ox;

    struct bsim4SizeDependParam  *pNext;
};


typedef struct sBSIM4model 
{
    int BSIM4modType;
    struct sBSIM4model *BSIM4nextModel;
    BSIM4instance *BSIM4instances;
    IFuid BSIM4modName; 
    int BSIM4type;

    int    BSIM4mobMod;
    int    BSIM4capMod;
    int    BSIM4dioMod;
    int    BSIM4trnqsMod;
    int    BSIM4acnqsMod;
    int    BSIM4fnoiMod;
    int    BSIM4tnoiMod;
    int    BSIM4rdsMod;
    int    BSIM4rbodyMod;
    int    BSIM4rgateMod;
    int    BSIM4perMod;
    int    BSIM4geoMod;
    int    BSIM4igcMod;
    int    BSIM4igbMod;
    int    BSIM4binUnit;
    int    BSIM4paramChk;
    char   *BSIM4version;             
    double BSIM4toxe;             
    double BSIM4toxp;
    double BSIM4toxm;
    double BSIM4dtox;
    double BSIM4epsrox;
    double BSIM4cdsc;           
    double BSIM4cdscb; 
    double BSIM4cdscd;          
    double BSIM4cit;           
    double BSIM4nfactor;      
    double BSIM4xj;
    double BSIM4vsat;         
    double BSIM4at;         
    double BSIM4a0;   
    double BSIM4ags;      
    double BSIM4a1;         
    double BSIM4a2;         
    double BSIM4keta;     
    double BSIM4nsub;
    double BSIM4ndep;        
    double BSIM4nsd;
    double BSIM4phin;
    double BSIM4ngate;        
    double BSIM4gamma1;      
    double BSIM4gamma2;     
    double BSIM4vbx;      
    double BSIM4vbm;       
    double BSIM4xt;       
    double BSIM4k1;
    double BSIM4kt1;
    double BSIM4kt1l;
    double BSIM4kt2;
    double BSIM4k2;
    double BSIM4k3;
    double BSIM4k3b;
    double BSIM4w0;
    double BSIM4dvtp0;
    double BSIM4dvtp1;
    double BSIM4lpe0;
    double BSIM4lpeb;
    double BSIM4dvt0;      
    double BSIM4dvt1;      
    double BSIM4dvt2;      
    double BSIM4dvt0w;      
    double BSIM4dvt1w;      
    double BSIM4dvt2w;      
    double BSIM4drout;      
    double BSIM4dsub;      
    double BSIM4vth0;
    double BSIM4eu;
    double BSIM4ua;
    double BSIM4ua1;
    double BSIM4ub;
    double BSIM4ub1;
    double BSIM4uc;
    double BSIM4uc1;
    double BSIM4u0;
    double BSIM4ute;
    double BSIM4voff;
    double BSIM4minv;
    double BSIM4voffl;
    double BSIM4delta;
    double BSIM4rdsw;       
    double BSIM4rdswmin;
    double BSIM4rdwmin;
    double BSIM4rswmin;
    double BSIM4rsw;
    double BSIM4rdw;
    double BSIM4prwg;
    double BSIM4prwb;
    double BSIM4prt;       
    double BSIM4eta0;         
    double BSIM4etab;         
    double BSIM4pclm;      
    double BSIM4pdibl1;      
    double BSIM4pdibl2;      
    double BSIM4pdiblb;
    double BSIM4fprout;
    double BSIM4pdits;
    double BSIM4pditsd;
    double BSIM4pditsl;
    double BSIM4pscbe1;       
    double BSIM4pscbe2;       
    double BSIM4pvag;       
    double BSIM4wr;
    double BSIM4dwg;
    double BSIM4dwb;
    double BSIM4b0;
    double BSIM4b1;
    double BSIM4alpha0;
    double BSIM4alpha1;
    double BSIM4beta0;
    double BSIM4agidl;
    double BSIM4bgidl;
    double BSIM4cgidl;
    double BSIM4egidl;
    double BSIM4aigc;
    double BSIM4bigc;
    double BSIM4cigc;
    double BSIM4aigsd;
    double BSIM4bigsd;
    double BSIM4cigsd;
    double BSIM4aigbacc;
    double BSIM4bigbacc;
    double BSIM4cigbacc;
    double BSIM4aigbinv;
    double BSIM4bigbinv;
    double BSIM4cigbinv;
    double BSIM4nigc;
    double BSIM4nigbacc;
    double BSIM4nigbinv;
    double BSIM4ntox;
    double BSIM4eigbinv;
    double BSIM4pigcd;
    double BSIM4poxedge;
    double BSIM4toxref;
    double BSIM4ijthdfwd;
    double BSIM4ijthsfwd;
    double BSIM4ijthdrev;
    double BSIM4ijthsrev;
    double BSIM4xjbvd;
    double BSIM4xjbvs;
    double BSIM4bvd;
    double BSIM4bvs;
    double BSIM4xrcrg1;
    double BSIM4xrcrg2;

    double BSIM4vfb;
    double BSIM4gbmin;
    double BSIM4rbdb;
    double BSIM4rbsb;
    double BSIM4rbpb;
    double BSIM4rbps;
    double BSIM4rbpd;
    double BSIM4tnoia;
    double BSIM4tnoib;
    double BSIM4ntnoi;

    /* CV model and Parasitics */
    double BSIM4cgsl;
    double BSIM4cgdl;
    double BSIM4ckappas;
    double BSIM4ckappad;
    double BSIM4cf;
    double BSIM4vfbcv;
    double BSIM4clc;
    double BSIM4cle;
    double BSIM4dwc;
    double BSIM4dlc;
    double BSIM4xw;
    double BSIM4xl;
    double BSIM4dlcig;
    double BSIM4dwj;
    double BSIM4noff;
    double BSIM4voffcv;
    double BSIM4acde;
    double BSIM4moin;
    double BSIM4tcj;
    double BSIM4tcjsw;
    double BSIM4tcjswg;
    double BSIM4tpb;
    double BSIM4tpbsw;
    double BSIM4tpbswg;
    double BSIM4dmcg;
    double BSIM4dmci;
    double BSIM4dmdg;
    double BSIM4dmcgt;
    double BSIM4xgw;
    double BSIM4xgl;
    double BSIM4rshg;
    double BSIM4ngcon;

    /* Length Dependence */
    double BSIM4lcdsc;           
    double BSIM4lcdscb; 
    double BSIM4lcdscd;          
    double BSIM4lcit;           
    double BSIM4lnfactor;      
    double BSIM4lxj;
    double BSIM4lvsat;         
    double BSIM4lat;         
    double BSIM4la0;   
    double BSIM4lags;      
    double BSIM4la1;         
    double BSIM4la2;         
    double BSIM4lketa;     
    double BSIM4lnsub;
    double BSIM4lndep;        
    double BSIM4lnsd;
    double BSIM4lphin;
    double BSIM4lngate;        
    double BSIM4lgamma1;      
    double BSIM4lgamma2;     
    double BSIM4lvbx;      
    double BSIM4lvbm;       
    double BSIM4lxt;       
    double BSIM4lk1;
    double BSIM4lkt1;
    double BSIM4lkt1l;
    double BSIM4lkt2;
    double BSIM4lk2;
    double BSIM4lk3;
    double BSIM4lk3b;
    double BSIM4lw0;
    double BSIM4ldvtp0;
    double BSIM4ldvtp1;
    double BSIM4llpe0;
    double BSIM4llpeb;
    double BSIM4ldvt0;      
    double BSIM4ldvt1;      
    double BSIM4ldvt2;      
    double BSIM4ldvt0w;      
    double BSIM4ldvt1w;      
    double BSIM4ldvt2w;      
    double BSIM4ldrout;      
    double BSIM4ldsub;      
    double BSIM4lvth0;
    double BSIM4lua;
    double BSIM4lua1;
    double BSIM4lub;
    double BSIM4lub1;
    double BSIM4luc;
    double BSIM4luc1;
    double BSIM4lu0;
    double BSIM4leu;
    double BSIM4lute;
    double BSIM4lvoff;
    double BSIM4lminv;
    double BSIM4ldelta;
    double BSIM4lrdsw;       
    double BSIM4lrsw;
    double BSIM4lrdw;
    double BSIM4lprwg;
    double BSIM4lprwb;
    double BSIM4lprt;       
    double BSIM4leta0;         
    double BSIM4letab;         
    double BSIM4lpclm;      
    double BSIM4lpdibl1;      
    double BSIM4lpdibl2;      
    double BSIM4lpdiblb;
    double BSIM4lfprout;
    double BSIM4lpdits;
    double BSIM4lpditsd;
    double BSIM4lpscbe1;       
    double BSIM4lpscbe2;       
    double BSIM4lpvag;       
    double BSIM4lwr;
    double BSIM4ldwg;
    double BSIM4ldwb;
    double BSIM4lb0;
    double BSIM4lb1;
    double BSIM4lalpha0;
    double BSIM4lalpha1;
    double BSIM4lbeta0;
    double BSIM4lvfb;
    double BSIM4lagidl;
    double BSIM4lbgidl;
    double BSIM4lcgidl;
    double BSIM4legidl;
    double BSIM4laigc;
    double BSIM4lbigc;
    double BSIM4lcigc;
    double BSIM4laigsd;
    double BSIM4lbigsd;
    double BSIM4lcigsd;
    double BSIM4laigbacc;
    double BSIM4lbigbacc;
    double BSIM4lcigbacc;
    double BSIM4laigbinv;
    double BSIM4lbigbinv;
    double BSIM4lcigbinv;
    double BSIM4lnigc;
    double BSIM4lnigbacc;
    double BSIM4lnigbinv;
    double BSIM4lntox;
    double BSIM4leigbinv;
    double BSIM4lpigcd;
    double BSIM4lpoxedge;
    double BSIM4lxrcrg1;
    double BSIM4lxrcrg2;

    /* CV model */
    double BSIM4lcgsl;
    double BSIM4lcgdl;
    double BSIM4lckappas;
    double BSIM4lckappad;
    double BSIM4lcf;
    double BSIM4lclc;
    double BSIM4lcle;
    double BSIM4lvfbcv;
    double BSIM4lnoff;
    double BSIM4lvoffcv;
    double BSIM4lacde;
    double BSIM4lmoin;

    /* Width Dependence */
    double BSIM4wcdsc;           
    double BSIM4wcdscb; 
    double BSIM4wcdscd;          
    double BSIM4wcit;           
    double BSIM4wnfactor;      
    double BSIM4wxj;
    double BSIM4wvsat;         
    double BSIM4wat;         
    double BSIM4wa0;   
    double BSIM4wags;      
    double BSIM4wa1;         
    double BSIM4wa2;         
    double BSIM4wketa;     
    double BSIM4wnsub;
    double BSIM4wndep;        
    double BSIM4wnsd;
    double BSIM4wphin;
    double BSIM4wngate;        
    double BSIM4wgamma1;      
    double BSIM4wgamma2;     
    double BSIM4wvbx;      
    double BSIM4wvbm;       
    double BSIM4wxt;       
    double BSIM4wk1;
    double BSIM4wkt1;
    double BSIM4wkt1l;
    double BSIM4wkt2;
    double BSIM4wk2;
    double BSIM4wk3;
    double BSIM4wk3b;
    double BSIM4ww0;
    double BSIM4wdvtp0;
    double BSIM4wdvtp1;
    double BSIM4wlpe0;
    double BSIM4wlpeb;
    double BSIM4wdvt0;      
    double BSIM4wdvt1;      
    double BSIM4wdvt2;      
    double BSIM4wdvt0w;      
    double BSIM4wdvt1w;      
    double BSIM4wdvt2w;      
    double BSIM4wdrout;      
    double BSIM4wdsub;      
    double BSIM4wvth0;
    double BSIM4wua;
    double BSIM4wua1;
    double BSIM4wub;
    double BSIM4wub1;
    double BSIM4wuc;
    double BSIM4wuc1;
    double BSIM4wu0;
    double BSIM4weu;
    double BSIM4wute;
    double BSIM4wvoff;
    double BSIM4wminv;
    double BSIM4wdelta;
    double BSIM4wrdsw;       
    double BSIM4wrsw;
    double BSIM4wrdw;
    double BSIM4wprwg;
    double BSIM4wprwb;
    double BSIM4wprt;       
    double BSIM4weta0;         
    double BSIM4wetab;         
    double BSIM4wpclm;      
    double BSIM4wpdibl1;      
    double BSIM4wpdibl2;      
    double BSIM4wpdiblb;
    double BSIM4wfprout;
    double BSIM4wpdits;
    double BSIM4wpditsd;
    double BSIM4wpscbe1;       
    double BSIM4wpscbe2;       
    double BSIM4wpvag;       
    double BSIM4wwr;
    double BSIM4wdwg;
    double BSIM4wdwb;
    double BSIM4wb0;
    double BSIM4wb1;
    double BSIM4walpha0;
    double BSIM4walpha1;
    double BSIM4wbeta0;
    double BSIM4wvfb;
    double BSIM4wagidl;
    double BSIM4wbgidl;
    double BSIM4wcgidl;
    double BSIM4wegidl;
    double BSIM4waigc;
    double BSIM4wbigc;
    double BSIM4wcigc;
    double BSIM4waigsd;
    double BSIM4wbigsd;
    double BSIM4wcigsd;
    double BSIM4waigbacc;
    double BSIM4wbigbacc;
    double BSIM4wcigbacc;
    double BSIM4waigbinv;
    double BSIM4wbigbinv;
    double BSIM4wcigbinv;
    double BSIM4wnigc;
    double BSIM4wnigbacc;
    double BSIM4wnigbinv;
    double BSIM4wntox;
    double BSIM4weigbinv;
    double BSIM4wpigcd;
    double BSIM4wpoxedge;
    double BSIM4wxrcrg1;
    double BSIM4wxrcrg2;

    /* CV model */
    double BSIM4wcgsl;
    double BSIM4wcgdl;
    double BSIM4wckappas;
    double BSIM4wckappad;
    double BSIM4wcf;
    double BSIM4wclc;
    double BSIM4wcle;
    double BSIM4wvfbcv;
    double BSIM4wnoff;
    double BSIM4wvoffcv;
    double BSIM4wacde;
    double BSIM4wmoin;

    /* Cross-term Dependence */
    double BSIM4pcdsc;           
    double BSIM4pcdscb; 
    double BSIM4pcdscd;          
    double BSIM4pcit;           
    double BSIM4pnfactor;      
    double BSIM4pxj;
    double BSIM4pvsat;         
    double BSIM4pat;         
    double BSIM4pa0;   
    double BSIM4pags;      
    double BSIM4pa1;         
    double BSIM4pa2;         
    double BSIM4pketa;     
    double BSIM4pnsub;
    double BSIM4pndep;        
    double BSIM4pnsd;
    double BSIM4pphin;
    double BSIM4pngate;        
    double BSIM4pgamma1;      
    double BSIM4pgamma2;     
    double BSIM4pvbx;      
    double BSIM4pvbm;       
    double BSIM4pxt;       
    double BSIM4pk1;
    double BSIM4pkt1;
    double BSIM4pkt1l;
    double BSIM4pkt2;
    double BSIM4pk2;
    double BSIM4pk3;
    double BSIM4pk3b;
    double BSIM4pw0;
    double BSIM4pdvtp0;
    double BSIM4pdvtp1;
    double BSIM4plpe0;
    double BSIM4plpeb;
    double BSIM4pdvt0;      
    double BSIM4pdvt1;      
    double BSIM4pdvt2;      
    double BSIM4pdvt0w;      
    double BSIM4pdvt1w;      
    double BSIM4pdvt2w;      
    double BSIM4pdrout;      
    double BSIM4pdsub;      
    double BSIM4pvth0;
    double BSIM4pua;
    double BSIM4pua1;
    double BSIM4pub;
    double BSIM4pub1;
    double BSIM4puc;
    double BSIM4puc1;
    double BSIM4pu0;
    double BSIM4peu;
    double BSIM4pute;
    double BSIM4pvoff;
    double BSIM4pminv;
    double BSIM4pdelta;
    double BSIM4prdsw;
    double BSIM4prsw;
    double BSIM4prdw;
    double BSIM4pprwg;
    double BSIM4pprwb;
    double BSIM4pprt;       
    double BSIM4peta0;         
    double BSIM4petab;         
    double BSIM4ppclm;      
    double BSIM4ppdibl1;      
    double BSIM4ppdibl2;      
    double BSIM4ppdiblb;
    double BSIM4pfprout;
    double BSIM4ppdits;
    double BSIM4ppditsd;
    double BSIM4ppscbe1;       
    double BSIM4ppscbe2;       
    double BSIM4ppvag;       
    double BSIM4pwr;
    double BSIM4pdwg;
    double BSIM4pdwb;
    double BSIM4pb0;
    double BSIM4pb1;
    double BSIM4palpha0;
    double BSIM4palpha1;
    double BSIM4pbeta0;
    double BSIM4pvfb;
    double BSIM4pagidl;
    double BSIM4pbgidl;
    double BSIM4pcgidl;
    double BSIM4pegidl;
    double BSIM4paigc;
    double BSIM4pbigc;
    double BSIM4pcigc;
    double BSIM4paigsd;
    double BSIM4pbigsd;
    double BSIM4pcigsd;
    double BSIM4paigbacc;
    double BSIM4pbigbacc;
    double BSIM4pcigbacc;
    double BSIM4paigbinv;
    double BSIM4pbigbinv;
    double BSIM4pcigbinv;
    double BSIM4pnigc;
    double BSIM4pnigbacc;
    double BSIM4pnigbinv;
    double BSIM4pntox;
    double BSIM4peigbinv;
    double BSIM4ppigcd;
    double BSIM4ppoxedge;
    double BSIM4pxrcrg1;
    double BSIM4pxrcrg2;

    /* CV model */
    double BSIM4pcgsl;
    double BSIM4pcgdl;
    double BSIM4pckappas;
    double BSIM4pckappad;
    double BSIM4pcf;
    double BSIM4pclc;
    double BSIM4pcle;
    double BSIM4pvfbcv;
    double BSIM4pnoff;
    double BSIM4pvoffcv;
    double BSIM4pacde;
    double BSIM4pmoin;

    double BSIM4tnom;
    double BSIM4cgso;
    double BSIM4cgdo;
    double BSIM4cgbo;
    double BSIM4xpart;
    double BSIM4cFringOut;
    double BSIM4cFringMax;

    double BSIM4sheetResistance;
    double BSIM4SjctSatCurDensity;
    double BSIM4DjctSatCurDensity;
    double BSIM4SjctSidewallSatCurDensity;
    double BSIM4DjctSidewallSatCurDensity;
    double BSIM4SjctGateSidewallSatCurDensity;
    double BSIM4DjctGateSidewallSatCurDensity;
    double BSIM4SbulkJctPotential;
    double BSIM4DbulkJctPotential;
    double BSIM4SbulkJctBotGradingCoeff;
    double BSIM4DbulkJctBotGradingCoeff;
    double BSIM4SbulkJctSideGradingCoeff;
    double BSIM4DbulkJctSideGradingCoeff;
    double BSIM4SbulkJctGateSideGradingCoeff;
    double BSIM4DbulkJctGateSideGradingCoeff;
    double BSIM4SsidewallJctPotential;
    double BSIM4DsidewallJctPotential;
    double BSIM4SGatesidewallJctPotential;
    double BSIM4DGatesidewallJctPotential;
    double BSIM4SunitAreaJctCap;
    double BSIM4DunitAreaJctCap;
    double BSIM4SunitLengthSidewallJctCap;
    double BSIM4DunitLengthSidewallJctCap;
    double BSIM4SunitLengthGateSidewallJctCap;
    double BSIM4DunitLengthGateSidewallJctCap;
    double BSIM4SjctEmissionCoeff;
    double BSIM4DjctEmissionCoeff;
    double BSIM4SjctTempExponent;
    double BSIM4DjctTempExponent;

    double BSIM4Lint;
    double BSIM4Ll;
    double BSIM4Llc;
    double BSIM4Lln;
    double BSIM4Lw;
    double BSIM4Lwc;
    double BSIM4Lwn;
    double BSIM4Lwl;
    double BSIM4Lwlc;
    double BSIM4Lmin;
    double BSIM4Lmax;

    double BSIM4Wint;
    double BSIM4Wl;
    double BSIM4Wlc;
    double BSIM4Wln;
    double BSIM4Ww;
    double BSIM4Wwc;
    double BSIM4Wwn;
    double BSIM4Wwl;
    double BSIM4Wwlc;
    double BSIM4Wmin;
    double BSIM4Wmax;


/* Pre-calculated constants
 * move to size-dependent param */
    double BSIM4vtm;   
    double BSIM4coxe;
    double BSIM4coxp;
    double BSIM4cof1;
    double BSIM4cof2;
    double BSIM4cof3;
    double BSIM4cof4;
    double BSIM4vcrit;
    double BSIM4factor1;
    double BSIM4PhiBS;
    double BSIM4PhiBSWS;
    double BSIM4PhiBSWGS;
    double BSIM4SjctTempSatCurDensity;
    double BSIM4SjctSidewallTempSatCurDensity;
    double BSIM4SjctGateSidewallTempSatCurDensity;
    double BSIM4PhiBD;
    double BSIM4PhiBSWD;
    double BSIM4PhiBSWGD;
    double BSIM4DjctTempSatCurDensity;
    double BSIM4DjctSidewallTempSatCurDensity;
    double BSIM4DjctGateSidewallTempSatCurDensity;
    double BSIM4SunitAreaTempJctCap;
    double BSIM4DunitAreaTempJctCap;
    double BSIM4SunitLengthSidewallTempJctCap;
    double BSIM4DunitLengthSidewallTempJctCap;
    double BSIM4SunitLengthGateSidewallTempJctCap;
    double BSIM4DunitLengthGateSidewallTempJctCap;

    double BSIM4oxideTrapDensityA;      
    double BSIM4oxideTrapDensityB;     
    double BSIM4oxideTrapDensityC;  
    double BSIM4em;  
    double BSIM4ef;  
    double BSIM4af;  
    double BSIM4kf;  

    struct bsim4SizeDependParam *pSizeDependParamKnot;

    /* Flags */
    unsigned  BSIM4mobModGiven :1;
    unsigned  BSIM4binUnitGiven :1;
    unsigned  BSIM4capModGiven :1;
    unsigned  BSIM4dioModGiven :1;
    unsigned  BSIM4rdsModGiven :1;
    unsigned  BSIM4rbodyModGiven :1;
    unsigned  BSIM4rgateModGiven :1;
    unsigned  BSIM4perModGiven :1;
    unsigned  BSIM4geoModGiven :1;
    unsigned  BSIM4paramChkGiven :1;
    unsigned  BSIM4trnqsModGiven :1;
    unsigned  BSIM4acnqsModGiven :1;
    unsigned  BSIM4fnoiModGiven :1;
    unsigned  BSIM4tnoiModGiven :1;
    unsigned  BSIM4igcModGiven :1;
    unsigned  BSIM4igbModGiven :1;
    unsigned  BSIM4typeGiven   :1;
    unsigned  BSIM4toxrefGiven   :1;
    unsigned  BSIM4toxeGiven   :1;
    unsigned  BSIM4toxpGiven   :1;
    unsigned  BSIM4toxmGiven   :1;
    unsigned  BSIM4dtoxGiven   :1;
    unsigned  BSIM4epsroxGiven   :1;
    unsigned  BSIM4versionGiven   :1;
    unsigned  BSIM4cdscGiven   :1;
    unsigned  BSIM4cdscbGiven   :1;
    unsigned  BSIM4cdscdGiven   :1;
    unsigned  BSIM4citGiven   :1;
    unsigned  BSIM4nfactorGiven   :1;
    unsigned  BSIM4xjGiven   :1;
    unsigned  BSIM4vsatGiven   :1;
    unsigned  BSIM4atGiven   :1;
    unsigned  BSIM4a0Given   :1;
    unsigned  BSIM4agsGiven   :1;
    unsigned  BSIM4a1Given   :1;
    unsigned  BSIM4a2Given   :1;
    unsigned  BSIM4ketaGiven   :1;    
    unsigned  BSIM4nsubGiven   :1;
    unsigned  BSIM4ndepGiven   :1;
    unsigned  BSIM4nsdGiven    :1;
    unsigned  BSIM4phinGiven   :1;
    unsigned  BSIM4ngateGiven   :1;
    unsigned  BSIM4gamma1Given   :1;
    unsigned  BSIM4gamma2Given   :1;
    unsigned  BSIM4vbxGiven   :1;
    unsigned  BSIM4vbmGiven   :1;
    unsigned  BSIM4xtGiven   :1;
    unsigned  BSIM4k1Given   :1;
    unsigned  BSIM4kt1Given   :1;
    unsigned  BSIM4kt1lGiven   :1;
    unsigned  BSIM4kt2Given   :1;
    unsigned  BSIM4k2Given   :1;
    unsigned  BSIM4k3Given   :1;
    unsigned  BSIM4k3bGiven   :1;
    unsigned  BSIM4w0Given   :1;
    unsigned  BSIM4dvtp0Given :1;
    unsigned  BSIM4dvtp1Given :1;
    unsigned  BSIM4lpe0Given   :1;
    unsigned  BSIM4lpebGiven   :1;
    unsigned  BSIM4dvt0Given   :1;   
    unsigned  BSIM4dvt1Given   :1;     
    unsigned  BSIM4dvt2Given   :1;     
    unsigned  BSIM4dvt0wGiven   :1;   
    unsigned  BSIM4dvt1wGiven   :1;     
    unsigned  BSIM4dvt2wGiven   :1;     
    unsigned  BSIM4droutGiven   :1;     
    unsigned  BSIM4dsubGiven   :1;     
    unsigned  BSIM4vth0Given   :1;
    unsigned  BSIM4euGiven   :1;
    unsigned  BSIM4uaGiven   :1;
    unsigned  BSIM4ua1Given   :1;
    unsigned  BSIM4ubGiven   :1;
    unsigned  BSIM4ub1Given   :1;
    unsigned  BSIM4ucGiven   :1;
    unsigned  BSIM4uc1Given   :1;
    unsigned  BSIM4u0Given   :1;
    unsigned  BSIM4uteGiven   :1;
    unsigned  BSIM4voffGiven   :1;
    unsigned  BSIM4vofflGiven  :1;
    unsigned  BSIM4minvGiven   :1;
    unsigned  BSIM4rdswGiven   :1;      
    unsigned  BSIM4rdswminGiven :1;
    unsigned  BSIM4rdwminGiven :1;
    unsigned  BSIM4rswminGiven :1;
    unsigned  BSIM4rswGiven   :1;
    unsigned  BSIM4rdwGiven   :1;
    unsigned  BSIM4prwgGiven   :1;      
    unsigned  BSIM4prwbGiven   :1;      
    unsigned  BSIM4prtGiven   :1;      
    unsigned  BSIM4eta0Given   :1;    
    unsigned  BSIM4etabGiven   :1;    
    unsigned  BSIM4pclmGiven   :1;   
    unsigned  BSIM4pdibl1Given   :1;   
    unsigned  BSIM4pdibl2Given   :1;  
    unsigned  BSIM4pdiblbGiven   :1;  
    unsigned  BSIM4fproutGiven   :1;
    unsigned  BSIM4pditsGiven    :1;
    unsigned  BSIM4pditsdGiven    :1;
    unsigned  BSIM4pditslGiven    :1;
    unsigned  BSIM4pscbe1Given   :1;    
    unsigned  BSIM4pscbe2Given   :1;    
    unsigned  BSIM4pvagGiven   :1;    
    unsigned  BSIM4deltaGiven  :1;     
    unsigned  BSIM4wrGiven   :1;
    unsigned  BSIM4dwgGiven   :1;
    unsigned  BSIM4dwbGiven   :1;
    unsigned  BSIM4b0Given   :1;
    unsigned  BSIM4b1Given   :1;
    unsigned  BSIM4alpha0Given   :1;
    unsigned  BSIM4alpha1Given   :1;
    unsigned  BSIM4beta0Given   :1;
    unsigned  BSIM4agidlGiven   :1;
    unsigned  BSIM4bgidlGiven   :1;
    unsigned  BSIM4cgidlGiven   :1;
    unsigned  BSIM4egidlGiven   :1;
    unsigned  BSIM4aigcGiven   :1;
    unsigned  BSIM4bigcGiven   :1;
    unsigned  BSIM4cigcGiven   :1;
    unsigned  BSIM4aigsdGiven   :1;
    unsigned  BSIM4bigsdGiven   :1;
    unsigned  BSIM4cigsdGiven   :1;
    unsigned  BSIM4aigbaccGiven   :1;
    unsigned  BSIM4bigbaccGiven   :1;
    unsigned  BSIM4cigbaccGiven   :1;
    unsigned  BSIM4aigbinvGiven   :1;
    unsigned  BSIM4bigbinvGiven   :1;
    unsigned  BSIM4cigbinvGiven   :1;
    unsigned  BSIM4nigcGiven   :1;
    unsigned  BSIM4nigbinvGiven   :1;
    unsigned  BSIM4nigbaccGiven   :1;
    unsigned  BSIM4ntoxGiven   :1;
    unsigned  BSIM4eigbinvGiven   :1;
    unsigned  BSIM4pigcdGiven   :1;
    unsigned  BSIM4poxedgeGiven   :1;
    unsigned  BSIM4ijthdfwdGiven  :1;
    unsigned  BSIM4ijthsfwdGiven  :1;
    unsigned  BSIM4ijthdrevGiven  :1;
    unsigned  BSIM4ijthsrevGiven  :1;
    unsigned  BSIM4xjbvdGiven   :1;
    unsigned  BSIM4xjbvsGiven   :1;
    unsigned  BSIM4bvdGiven   :1;
    unsigned  BSIM4bvsGiven   :1;
    unsigned  BSIM4vfbGiven   :1;
    unsigned  BSIM4gbminGiven :1;
    unsigned  BSIM4rbdbGiven :1;
    unsigned  BSIM4rbsbGiven :1;
    unsigned  BSIM4rbpsGiven :1;
    unsigned  BSIM4rbpdGiven :1;
    unsigned  BSIM4rbpbGiven :1;
    unsigned  BSIM4xrcrg1Given   :1;
    unsigned  BSIM4xrcrg2Given   :1;
    unsigned  BSIM4tnoiaGiven    :1;
    unsigned  BSIM4tnoibGiven    :1;
    unsigned  BSIM4ntnoiGiven    :1;

    /* CV model and parasitics */
    unsigned  BSIM4cgslGiven   :1;
    unsigned  BSIM4cgdlGiven   :1;
    unsigned  BSIM4ckappasGiven   :1;
    unsigned  BSIM4ckappadGiven   :1;
    unsigned  BSIM4cfGiven   :1;
    unsigned  BSIM4vfbcvGiven   :1;
    unsigned  BSIM4clcGiven   :1;
    unsigned  BSIM4cleGiven   :1;
    unsigned  BSIM4dwcGiven   :1;
    unsigned  BSIM4dlcGiven   :1;
    unsigned  BSIM4xwGiven    :1;
    unsigned  BSIM4xlGiven    :1;
    unsigned  BSIM4dlcigGiven   :1;
    unsigned  BSIM4dwjGiven   :1;
    unsigned  BSIM4noffGiven  :1;
    unsigned  BSIM4voffcvGiven :1;
    unsigned  BSIM4acdeGiven  :1;
    unsigned  BSIM4moinGiven  :1;
    unsigned  BSIM4tcjGiven   :1;
    unsigned  BSIM4tcjswGiven :1;
    unsigned  BSIM4tcjswgGiven :1;
    unsigned  BSIM4tpbGiven    :1;
    unsigned  BSIM4tpbswGiven  :1;
    unsigned  BSIM4tpbswgGiven :1;
    unsigned  BSIM4dmcgGiven :1;
    unsigned  BSIM4dmciGiven :1;
    unsigned  BSIM4dmdgGiven :1;
    unsigned  BSIM4dmcgtGiven :1;
    unsigned  BSIM4xgwGiven :1;
    unsigned  BSIM4xglGiven :1;
    unsigned  BSIM4rshgGiven :1;
    unsigned  BSIM4ngconGiven :1;


    /* Length dependence */
    unsigned  BSIM4lcdscGiven   :1;
    unsigned  BSIM4lcdscbGiven   :1;
    unsigned  BSIM4lcdscdGiven   :1;
    unsigned  BSIM4lcitGiven   :1;
    unsigned  BSIM4lnfactorGiven   :1;
    unsigned  BSIM4lxjGiven   :1;
    unsigned  BSIM4lvsatGiven   :1;
    unsigned  BSIM4latGiven   :1;
    unsigned  BSIM4la0Given   :1;
    unsigned  BSIM4lagsGiven   :1;
    unsigned  BSIM4la1Given   :1;
    unsigned  BSIM4la2Given   :1;
    unsigned  BSIM4lketaGiven   :1;    
    unsigned  BSIM4lnsubGiven   :1;
    unsigned  BSIM4lndepGiven   :1;
    unsigned  BSIM4lnsdGiven    :1;
    unsigned  BSIM4lphinGiven   :1;
    unsigned  BSIM4lngateGiven   :1;
    unsigned  BSIM4lgamma1Given   :1;
    unsigned  BSIM4lgamma2Given   :1;
    unsigned  BSIM4lvbxGiven   :1;
    unsigned  BSIM4lvbmGiven   :1;
    unsigned  BSIM4lxtGiven   :1;
    unsigned  BSIM4lk1Given   :1;
    unsigned  BSIM4lkt1Given   :1;
    unsigned  BSIM4lkt1lGiven   :1;
    unsigned  BSIM4lkt2Given   :1;
    unsigned  BSIM4lk2Given   :1;
    unsigned  BSIM4lk3Given   :1;
    unsigned  BSIM4lk3bGiven   :1;
    unsigned  BSIM4lw0Given   :1;
    unsigned  BSIM4ldvtp0Given :1;
    unsigned  BSIM4ldvtp1Given :1;
    unsigned  BSIM4llpe0Given   :1;
    unsigned  BSIM4llpebGiven   :1;
    unsigned  BSIM4ldvt0Given   :1;   
    unsigned  BSIM4ldvt1Given   :1;     
    unsigned  BSIM4ldvt2Given   :1;     
    unsigned  BSIM4ldvt0wGiven   :1;   
    unsigned  BSIM4ldvt1wGiven   :1;     
    unsigned  BSIM4ldvt2wGiven   :1;     
    unsigned  BSIM4ldroutGiven   :1;     
    unsigned  BSIM4ldsubGiven   :1;     
    unsigned  BSIM4lvth0Given   :1;
    unsigned  BSIM4luaGiven   :1;
    unsigned  BSIM4lua1Given   :1;
    unsigned  BSIM4lubGiven   :1;
    unsigned  BSIM4lub1Given   :1;
    unsigned  BSIM4lucGiven   :1;
    unsigned  BSIM4luc1Given   :1;
    unsigned  BSIM4lu0Given   :1;
    unsigned  BSIM4leuGiven   :1;
    unsigned  BSIM4luteGiven   :1;
    unsigned  BSIM4lvoffGiven   :1;
    unsigned  BSIM4lminvGiven   :1;
    unsigned  BSIM4lrdswGiven   :1;      
    unsigned  BSIM4lrswGiven   :1;
    unsigned  BSIM4lrdwGiven   :1;
    unsigned  BSIM4lprwgGiven   :1;      
    unsigned  BSIM4lprwbGiven   :1;      
    unsigned  BSIM4lprtGiven   :1;      
    unsigned  BSIM4leta0Given   :1;    
    unsigned  BSIM4letabGiven   :1;    
    unsigned  BSIM4lpclmGiven   :1;   
    unsigned  BSIM4lpdibl1Given   :1;   
    unsigned  BSIM4lpdibl2Given   :1;  
    unsigned  BSIM4lpdiblbGiven   :1;  
    unsigned  BSIM4lfproutGiven   :1;
    unsigned  BSIM4lpditsGiven    :1;
    unsigned  BSIM4lpditsdGiven    :1;
    unsigned  BSIM4lpscbe1Given   :1;    
    unsigned  BSIM4lpscbe2Given   :1;    
    unsigned  BSIM4lpvagGiven   :1;    
    unsigned  BSIM4ldeltaGiven  :1;     
    unsigned  BSIM4lwrGiven   :1;
    unsigned  BSIM4ldwgGiven   :1;
    unsigned  BSIM4ldwbGiven   :1;
    unsigned  BSIM4lb0Given   :1;
    unsigned  BSIM4lb1Given   :1;
    unsigned  BSIM4lalpha0Given   :1;
    unsigned  BSIM4lalpha1Given   :1;
    unsigned  BSIM4lbeta0Given   :1;
    unsigned  BSIM4lvfbGiven   :1;
    unsigned  BSIM4lagidlGiven   :1;
    unsigned  BSIM4lbgidlGiven   :1;
    unsigned  BSIM4lcgidlGiven   :1;
    unsigned  BSIM4legidlGiven   :1;
    unsigned  BSIM4laigcGiven   :1;
    unsigned  BSIM4lbigcGiven   :1;
    unsigned  BSIM4lcigcGiven   :1;
    unsigned  BSIM4laigsdGiven   :1;
    unsigned  BSIM4lbigsdGiven   :1;
    unsigned  BSIM4lcigsdGiven   :1;
    unsigned  BSIM4laigbaccGiven   :1;
    unsigned  BSIM4lbigbaccGiven   :1;
    unsigned  BSIM4lcigbaccGiven   :1;
    unsigned  BSIM4laigbinvGiven   :1;
    unsigned  BSIM4lbigbinvGiven   :1;
    unsigned  BSIM4lcigbinvGiven   :1;
    unsigned  BSIM4lnigcGiven   :1;
    unsigned  BSIM4lnigbinvGiven   :1;
    unsigned  BSIM4lnigbaccGiven   :1;
    unsigned  BSIM4lntoxGiven   :1;
    unsigned  BSIM4leigbinvGiven   :1;
    unsigned  BSIM4lpigcdGiven   :1;
    unsigned  BSIM4lpoxedgeGiven   :1;
    unsigned  BSIM4lxrcrg1Given   :1;
    unsigned  BSIM4lxrcrg2Given   :1;

    /* CV model */
    unsigned  BSIM4lcgslGiven   :1;
    unsigned  BSIM4lcgdlGiven   :1;
    unsigned  BSIM4lckappasGiven   :1;
    unsigned  BSIM4lckappadGiven   :1;
    unsigned  BSIM4lcfGiven   :1;
    unsigned  BSIM4lclcGiven   :1;
    unsigned  BSIM4lcleGiven   :1;
    unsigned  BSIM4lvfbcvGiven   :1;
    unsigned  BSIM4lnoffGiven   :1;
    unsigned  BSIM4lvoffcvGiven :1;
    unsigned  BSIM4lacdeGiven   :1;
    unsigned  BSIM4lmoinGiven   :1;

    /* Width dependence */
    unsigned  BSIM4wcdscGiven   :1;
    unsigned  BSIM4wcdscbGiven   :1;
    unsigned  BSIM4wcdscdGiven   :1;
    unsigned  BSIM4wcitGiven   :1;
    unsigned  BSIM4wnfactorGiven   :1;
    unsigned  BSIM4wxjGiven   :1;
    unsigned  BSIM4wvsatGiven   :1;
    unsigned  BSIM4watGiven   :1;
    unsigned  BSIM4wa0Given   :1;
    unsigned  BSIM4wagsGiven   :1;
    unsigned  BSIM4wa1Given   :1;
    unsigned  BSIM4wa2Given   :1;
    unsigned  BSIM4wketaGiven   :1;    
    unsigned  BSIM4wnsubGiven   :1;
    unsigned  BSIM4wndepGiven   :1;
    unsigned  BSIM4wnsdGiven    :1;
    unsigned  BSIM4wphinGiven   :1;
    unsigned  BSIM4wngateGiven   :1;
    unsigned  BSIM4wgamma1Given   :1;
    unsigned  BSIM4wgamma2Given   :1;
    unsigned  BSIM4wvbxGiven   :1;
    unsigned  BSIM4wvbmGiven   :1;
    unsigned  BSIM4wxtGiven   :1;
    unsigned  BSIM4wk1Given   :1;
    unsigned  BSIM4wkt1Given   :1;
    unsigned  BSIM4wkt1lGiven   :1;
    unsigned  BSIM4wkt2Given   :1;
    unsigned  BSIM4wk2Given   :1;
    unsigned  BSIM4wk3Given   :1;
    unsigned  BSIM4wk3bGiven   :1;
    unsigned  BSIM4ww0Given   :1;
    unsigned  BSIM4wdvtp0Given :1;
    unsigned  BSIM4wdvtp1Given :1;
    unsigned  BSIM4wlpe0Given   :1;
    unsigned  BSIM4wlpebGiven   :1;
    unsigned  BSIM4wdvt0Given   :1;   
    unsigned  BSIM4wdvt1Given   :1;     
    unsigned  BSIM4wdvt2Given   :1;     
    unsigned  BSIM4wdvt0wGiven   :1;   
    unsigned  BSIM4wdvt1wGiven   :1;     
    unsigned  BSIM4wdvt2wGiven   :1;     
    unsigned  BSIM4wdroutGiven   :1;     
    unsigned  BSIM4wdsubGiven   :1;     
    unsigned  BSIM4wvth0Given   :1;
    unsigned  BSIM4wuaGiven   :1;
    unsigned  BSIM4wua1Given   :1;
    unsigned  BSIM4wubGiven   :1;
    unsigned  BSIM4wub1Given   :1;
    unsigned  BSIM4wucGiven   :1;
    unsigned  BSIM4wuc1Given   :1;
    unsigned  BSIM4wu0Given   :1;
    unsigned  BSIM4weuGiven   :1;
    unsigned  BSIM4wuteGiven   :1;
    unsigned  BSIM4wvoffGiven   :1;
    unsigned  BSIM4wminvGiven   :1;
    unsigned  BSIM4wrdswGiven   :1;      
    unsigned  BSIM4wrswGiven   :1;
    unsigned  BSIM4wrdwGiven   :1;
    unsigned  BSIM4wprwgGiven   :1;      
    unsigned  BSIM4wprwbGiven   :1;      
    unsigned  BSIM4wprtGiven   :1;      
    unsigned  BSIM4weta0Given   :1;    
    unsigned  BSIM4wetabGiven   :1;    
    unsigned  BSIM4wpclmGiven   :1;   
    unsigned  BSIM4wpdibl1Given   :1;   
    unsigned  BSIM4wpdibl2Given   :1;  
    unsigned  BSIM4wpdiblbGiven   :1;  
    unsigned  BSIM4wfproutGiven   :1;
    unsigned  BSIM4wpditsGiven    :1;
    unsigned  BSIM4wpditsdGiven    :1;
    unsigned  BSIM4wpscbe1Given   :1;    
    unsigned  BSIM4wpscbe2Given   :1;    
    unsigned  BSIM4wpvagGiven   :1;    
    unsigned  BSIM4wdeltaGiven  :1;     
    unsigned  BSIM4wwrGiven   :1;
    unsigned  BSIM4wdwgGiven   :1;
    unsigned  BSIM4wdwbGiven   :1;
    unsigned  BSIM4wb0Given   :1;
    unsigned  BSIM4wb1Given   :1;
    unsigned  BSIM4walpha0Given   :1;
    unsigned  BSIM4walpha1Given   :1;
    unsigned  BSIM4wbeta0Given   :1;
    unsigned  BSIM4wvfbGiven   :1;
    unsigned  BSIM4wagidlGiven   :1;
    unsigned  BSIM4wbgidlGiven   :1;
    unsigned  BSIM4wcgidlGiven   :1;
    unsigned  BSIM4wegidlGiven   :1;
    unsigned  BSIM4waigcGiven   :1;
    unsigned  BSIM4wbigcGiven   :1;
    unsigned  BSIM4wcigcGiven   :1;
    unsigned  BSIM4waigsdGiven   :1;
    unsigned  BSIM4wbigsdGiven   :1;
    unsigned  BSIM4wcigsdGiven   :1;
    unsigned  BSIM4waigbaccGiven   :1;
    unsigned  BSIM4wbigbaccGiven   :1;
    unsigned  BSIM4wcigbaccGiven   :1;
    unsigned  BSIM4waigbinvGiven   :1;
    unsigned  BSIM4wbigbinvGiven   :1;
    unsigned  BSIM4wcigbinvGiven   :1;
    unsigned  BSIM4wnigcGiven   :1;
    unsigned  BSIM4wnigbinvGiven   :1;
    unsigned  BSIM4wnigbaccGiven   :1;
    unsigned  BSIM4wntoxGiven   :1;
    unsigned  BSIM4weigbinvGiven   :1;
    unsigned  BSIM4wpigcdGiven   :1;
    unsigned  BSIM4wpoxedgeGiven   :1;
    unsigned  BSIM4wxrcrg1Given   :1;
    unsigned  BSIM4wxrcrg2Given   :1;

    /* CV model */
    unsigned  BSIM4wcgslGiven   :1;
    unsigned  BSIM4wcgdlGiven   :1;
    unsigned  BSIM4wckappasGiven   :1;
    unsigned  BSIM4wckappadGiven   :1;
    unsigned  BSIM4wcfGiven   :1;
    unsigned  BSIM4wclcGiven   :1;
    unsigned  BSIM4wcleGiven   :1;
    unsigned  BSIM4wvfbcvGiven   :1;
    unsigned  BSIM4wnoffGiven   :1;
    unsigned  BSIM4wvoffcvGiven :1;
    unsigned  BSIM4wacdeGiven   :1;
    unsigned  BSIM4wmoinGiven   :1;

    /* Cross-term dependence */
    unsigned  BSIM4pcdscGiven   :1;
    unsigned  BSIM4pcdscbGiven   :1;
    unsigned  BSIM4pcdscdGiven   :1;
    unsigned  BSIM4pcitGiven   :1;
    unsigned  BSIM4pnfactorGiven   :1;
    unsigned  BSIM4pxjGiven   :1;
    unsigned  BSIM4pvsatGiven   :1;
    unsigned  BSIM4patGiven   :1;
    unsigned  BSIM4pa0Given   :1;
    unsigned  BSIM4pagsGiven   :1;
    unsigned  BSIM4pa1Given   :1;
    unsigned  BSIM4pa2Given   :1;
    unsigned  BSIM4pketaGiven   :1;    
    unsigned  BSIM4pnsubGiven   :1;
    unsigned  BSIM4pndepGiven   :1;
    unsigned  BSIM4pnsdGiven    :1;
    unsigned  BSIM4pphinGiven   :1;
    unsigned  BSIM4pngateGiven   :1;
    unsigned  BSIM4pgamma1Given   :1;
    unsigned  BSIM4pgamma2Given   :1;
    unsigned  BSIM4pvbxGiven   :1;
    unsigned  BSIM4pvbmGiven   :1;
    unsigned  BSIM4pxtGiven   :1;
    unsigned  BSIM4pk1Given   :1;
    unsigned  BSIM4pkt1Given   :1;
    unsigned  BSIM4pkt1lGiven   :1;
    unsigned  BSIM4pkt2Given   :1;
    unsigned  BSIM4pk2Given   :1;
    unsigned  BSIM4pk3Given   :1;
    unsigned  BSIM4pk3bGiven   :1;
    unsigned  BSIM4pw0Given   :1;
    unsigned  BSIM4pdvtp0Given :1;
    unsigned  BSIM4pdvtp1Given :1;
    unsigned  BSIM4plpe0Given   :1;
    unsigned  BSIM4plpebGiven   :1;
    unsigned  BSIM4pdvt0Given   :1;   
    unsigned  BSIM4pdvt1Given   :1;     
    unsigned  BSIM4pdvt2Given   :1;     
    unsigned  BSIM4pdvt0wGiven   :1;   
    unsigned  BSIM4pdvt1wGiven   :1;     
    unsigned  BSIM4pdvt2wGiven   :1;     
    unsigned  BSIM4pdroutGiven   :1;     
    unsigned  BSIM4pdsubGiven   :1;     
    unsigned  BSIM4pvth0Given   :1;
    unsigned  BSIM4puaGiven   :1;
    unsigned  BSIM4pua1Given   :1;
    unsigned  BSIM4pubGiven   :1;
    unsigned  BSIM4pub1Given   :1;
    unsigned  BSIM4pucGiven   :1;
    unsigned  BSIM4puc1Given   :1;
    unsigned  BSIM4pu0Given   :1;
    unsigned  BSIM4peuGiven   :1;
    unsigned  BSIM4puteGiven   :1;
    unsigned  BSIM4pvoffGiven   :1;
    unsigned  BSIM4pminvGiven   :1;
    unsigned  BSIM4prdswGiven   :1;      
    unsigned  BSIM4prswGiven   :1;
    unsigned  BSIM4prdwGiven   :1;
    unsigned  BSIM4pprwgGiven   :1;      
    unsigned  BSIM4pprwbGiven   :1;      
    unsigned  BSIM4pprtGiven   :1;      
    unsigned  BSIM4peta0Given   :1;    
    unsigned  BSIM4petabGiven   :1;    
    unsigned  BSIM4ppclmGiven   :1;   
    unsigned  BSIM4ppdibl1Given   :1;   
    unsigned  BSIM4ppdibl2Given   :1;  
    unsigned  BSIM4ppdiblbGiven   :1;  
    unsigned  BSIM4pfproutGiven   :1;
    unsigned  BSIM4ppditsGiven    :1;
    unsigned  BSIM4ppditsdGiven    :1;
    unsigned  BSIM4ppscbe1Given   :1;    
    unsigned  BSIM4ppscbe2Given   :1;    
    unsigned  BSIM4ppvagGiven   :1;    
    unsigned  BSIM4pdeltaGiven  :1;     
    unsigned  BSIM4pwrGiven   :1;
    unsigned  BSIM4pdwgGiven   :1;
    unsigned  BSIM4pdwbGiven   :1;
    unsigned  BSIM4pb0Given   :1;
    unsigned  BSIM4pb1Given   :1;
    unsigned  BSIM4palpha0Given   :1;
    unsigned  BSIM4palpha1Given   :1;
    unsigned  BSIM4pbeta0Given   :1;
    unsigned  BSIM4pvfbGiven   :1;
    unsigned  BSIM4pagidlGiven   :1;
    unsigned  BSIM4pbgidlGiven   :1;
    unsigned  BSIM4pcgidlGiven   :1;
    unsigned  BSIM4pegidlGiven   :1;
    unsigned  BSIM4paigcGiven   :1;
    unsigned  BSIM4pbigcGiven   :1;
    unsigned  BSIM4pcigcGiven   :1;
    unsigned  BSIM4paigsdGiven   :1;
    unsigned  BSIM4pbigsdGiven   :1;
    unsigned  BSIM4pcigsdGiven   :1;
    unsigned  BSIM4paigbaccGiven   :1;
    unsigned  BSIM4pbigbaccGiven   :1;
    unsigned  BSIM4pcigbaccGiven   :1;
    unsigned  BSIM4paigbinvGiven   :1;
    unsigned  BSIM4pbigbinvGiven   :1;
    unsigned  BSIM4pcigbinvGiven   :1;
    unsigned  BSIM4pnigcGiven   :1;
    unsigned  BSIM4pnigbinvGiven   :1;
    unsigned  BSIM4pnigbaccGiven   :1;
    unsigned  BSIM4pntoxGiven   :1;
    unsigned  BSIM4peigbinvGiven   :1;
    unsigned  BSIM4ppigcdGiven   :1;
    unsigned  BSIM4ppoxedgeGiven   :1;
    unsigned  BSIM4pxrcrg1Given   :1;
    unsigned  BSIM4pxrcrg2Given   :1;

    /* CV model */
    unsigned  BSIM4pcgslGiven   :1;
    unsigned  BSIM4pcgdlGiven   :1;
    unsigned  BSIM4pckappasGiven   :1;
    unsigned  BSIM4pckappadGiven   :1;
    unsigned  BSIM4pcfGiven   :1;
    unsigned  BSIM4pclcGiven   :1;
    unsigned  BSIM4pcleGiven   :1;
    unsigned  BSIM4pvfbcvGiven   :1;
    unsigned  BSIM4pnoffGiven   :1;
    unsigned  BSIM4pvoffcvGiven :1;
    unsigned  BSIM4pacdeGiven   :1;
    unsigned  BSIM4pmoinGiven   :1;

    unsigned  BSIM4useFringeGiven   :1;

    unsigned  BSIM4tnomGiven   :1;
    unsigned  BSIM4cgsoGiven   :1;
    unsigned  BSIM4cgdoGiven   :1;
    unsigned  BSIM4cgboGiven   :1;
    unsigned  BSIM4xpartGiven   :1;
    unsigned  BSIM4sheetResistanceGiven   :1;

    unsigned  BSIM4SjctSatCurDensityGiven   :1;
    unsigned  BSIM4SjctSidewallSatCurDensityGiven   :1;
    unsigned  BSIM4SjctGateSidewallSatCurDensityGiven   :1;
    unsigned  BSIM4SbulkJctPotentialGiven   :1;
    unsigned  BSIM4SbulkJctBotGradingCoeffGiven   :1;
    unsigned  BSIM4SsidewallJctPotentialGiven   :1;
    unsigned  BSIM4SGatesidewallJctPotentialGiven   :1;
    unsigned  BSIM4SbulkJctSideGradingCoeffGiven   :1;
    unsigned  BSIM4SunitAreaJctCapGiven   :1;
    unsigned  BSIM4SunitLengthSidewallJctCapGiven   :1;
    unsigned  BSIM4SbulkJctGateSideGradingCoeffGiven   :1;
    unsigned  BSIM4SunitLengthGateSidewallJctCapGiven   :1;
    unsigned  BSIM4SjctEmissionCoeffGiven :1;
    unsigned  BSIM4SjctTempExponentGiven	:1;

    unsigned  BSIM4DjctSatCurDensityGiven   :1;
    unsigned  BSIM4DjctSidewallSatCurDensityGiven   :1;
    unsigned  BSIM4DjctGateSidewallSatCurDensityGiven   :1;
    unsigned  BSIM4DbulkJctPotentialGiven   :1;
    unsigned  BSIM4DbulkJctBotGradingCoeffGiven   :1;
    unsigned  BSIM4DsidewallJctPotentialGiven   :1;
    unsigned  BSIM4DGatesidewallJctPotentialGiven   :1;
    unsigned  BSIM4DbulkJctSideGradingCoeffGiven   :1;
    unsigned  BSIM4DunitAreaJctCapGiven   :1;
    unsigned  BSIM4DunitLengthSidewallJctCapGiven   :1;
    unsigned  BSIM4DbulkJctGateSideGradingCoeffGiven   :1;
    unsigned  BSIM4DunitLengthGateSidewallJctCapGiven   :1;
    unsigned  BSIM4DjctEmissionCoeffGiven :1;
    unsigned  BSIM4DjctTempExponentGiven :1;

    unsigned  BSIM4oxideTrapDensityAGiven  :1;         
    unsigned  BSIM4oxideTrapDensityBGiven  :1;        
    unsigned  BSIM4oxideTrapDensityCGiven  :1;     
    unsigned  BSIM4emGiven  :1;     
    unsigned  BSIM4efGiven  :1;     
    unsigned  BSIM4afGiven  :1;     
    unsigned  BSIM4kfGiven  :1;     

    unsigned  BSIM4LintGiven   :1;
    unsigned  BSIM4LlGiven   :1;
    unsigned  BSIM4LlcGiven   :1;
    unsigned  BSIM4LlnGiven   :1;
    unsigned  BSIM4LwGiven   :1;
    unsigned  BSIM4LwcGiven   :1;
    unsigned  BSIM4LwnGiven   :1;
    unsigned  BSIM4LwlGiven   :1;
    unsigned  BSIM4LwlcGiven   :1;
    unsigned  BSIM4LminGiven   :1;
    unsigned  BSIM4LmaxGiven   :1;

    unsigned  BSIM4WintGiven   :1;
    unsigned  BSIM4WlGiven   :1;
    unsigned  BSIM4WlcGiven   :1;
    unsigned  BSIM4WlnGiven   :1;
    unsigned  BSIM4WwGiven   :1;
    unsigned  BSIM4WwcGiven   :1;
    unsigned  BSIM4WwnGiven   :1;
    unsigned  BSIM4WwlGiven   :1;
    unsigned  BSIM4WwlcGiven   :1;
    unsigned  BSIM4WminGiven   :1;
    unsigned  BSIM4WmaxGiven   :1;

} BSIM4model;


#ifndef NMOS
#define NMOS 1
#define PMOS -1
#endif /*NMOS*/


/* Instance parameters */
#define BSIM4_W                   1
#define BSIM4_L                   2
#define BSIM4_AS                  3
#define BSIM4_AD                  4
#define BSIM4_PS                  5
#define BSIM4_PD                  6
#define BSIM4_NRS                 7
#define BSIM4_NRD                 8
#define BSIM4_OFF                 9
#define BSIM4_IC                  10
#define BSIM4_IC_VDS              11
#define BSIM4_IC_VGS              12
#define BSIM4_IC_VBS              13
#define BSIM4_TRNQSMOD            14
#define BSIM4_RBODYMOD            15
#define BSIM4_RGATEMOD            16
#define BSIM4_GEOMOD              17
#define BSIM4_RGEOMOD             18
#define BSIM4_NF                  19
#define BSIM4_MIN                 20 
#define BSIM4_ACNQSMOD            22
#define BSIM4_RBDB                23
#define BSIM4_RBSB                24
#define BSIM4_RBPB                25
#define BSIM4_RBPS                26
#define BSIM4_RBPD                27


/* Global parameters */
#define BSIM4_MOD_IGCMOD          90
#define BSIM4_MOD_IGBMOD          91
#define BSIM4_MOD_ACNQSMOD        92
#define BSIM4_MOD_FNOIMOD         93
#define BSIM4_MOD_RDSMOD          94
#define BSIM4_MOD_DIOMOD          96
#define BSIM4_MOD_PERMOD          97
#define BSIM4_MOD_GEOMOD          98
#define BSIM4_MOD_RGATEMOD        99
#define BSIM4_MOD_RBODYMOD        100
#define BSIM4_MOD_CAPMOD          101
#define BSIM4_MOD_TRNQSMOD        102
#define BSIM4_MOD_MOBMOD          103    
#define BSIM4_MOD_TNOIMOD         104    
#define BSIM4_MOD_TOXE            105
#define BSIM4_MOD_CDSC            106
#define BSIM4_MOD_CDSCB           107
#define BSIM4_MOD_CIT             108
#define BSIM4_MOD_NFACTOR         109
#define BSIM4_MOD_XJ              110
#define BSIM4_MOD_VSAT            111
#define BSIM4_MOD_AT              112
#define BSIM4_MOD_A0              113
#define BSIM4_MOD_A1              114
#define BSIM4_MOD_A2              115
#define BSIM4_MOD_KETA            116   
#define BSIM4_MOD_NSUB            117
#define BSIM4_MOD_NDEP            118
#define BSIM4_MOD_NGATE           120
#define BSIM4_MOD_GAMMA1          121
#define BSIM4_MOD_GAMMA2          122
#define BSIM4_MOD_VBX             123
#define BSIM4_MOD_BINUNIT         124    
#define BSIM4_MOD_VBM             125
#define BSIM4_MOD_XT              126
#define BSIM4_MOD_K1              129
#define BSIM4_MOD_KT1             130
#define BSIM4_MOD_KT1L            131
#define BSIM4_MOD_K2              132
#define BSIM4_MOD_KT2             133
#define BSIM4_MOD_K3              134
#define BSIM4_MOD_K3B             135
#define BSIM4_MOD_W0              136
#define BSIM4_MOD_LPE0            137
#define BSIM4_MOD_DVT0            138
#define BSIM4_MOD_DVT1            139
#define BSIM4_MOD_DVT2            140
#define BSIM4_MOD_DVT0W           141
#define BSIM4_MOD_DVT1W           142
#define BSIM4_MOD_DVT2W           143
#define BSIM4_MOD_DROUT           144
#define BSIM4_MOD_DSUB            145
#define BSIM4_MOD_VTH0            146
#define BSIM4_MOD_UA              147
#define BSIM4_MOD_UA1             148
#define BSIM4_MOD_UB              149
#define BSIM4_MOD_UB1             150
#define BSIM4_MOD_UC              151
#define BSIM4_MOD_UC1             152
#define BSIM4_MOD_U0              153
#define BSIM4_MOD_UTE             154
#define BSIM4_MOD_VOFF            155
#define BSIM4_MOD_DELTA           156
#define BSIM4_MOD_RDSW            157
#define BSIM4_MOD_PRT             158
#define BSIM4_MOD_LDD             159
#define BSIM4_MOD_ETA             160
#define BSIM4_MOD_ETA0            161
#define BSIM4_MOD_ETAB            162
#define BSIM4_MOD_PCLM            163
#define BSIM4_MOD_PDIBL1          164
#define BSIM4_MOD_PDIBL2          165
#define BSIM4_MOD_PSCBE1          166
#define BSIM4_MOD_PSCBE2          167
#define BSIM4_MOD_PVAG            168
#define BSIM4_MOD_WR              169
#define BSIM4_MOD_DWG             170
#define BSIM4_MOD_DWB             171
#define BSIM4_MOD_B0              172
#define BSIM4_MOD_B1              173
#define BSIM4_MOD_ALPHA0          174
#define BSIM4_MOD_BETA0           175
#define BSIM4_MOD_PDIBLB          178
#define BSIM4_MOD_PRWG            179
#define BSIM4_MOD_PRWB            180
#define BSIM4_MOD_CDSCD           181
#define BSIM4_MOD_AGS             182
#define BSIM4_MOD_FRINGE          184
#define BSIM4_MOD_CGSL            186
#define BSIM4_MOD_CGDL            187
#define BSIM4_MOD_CKAPPAS         188
#define BSIM4_MOD_CF              189
#define BSIM4_MOD_CLC             190
#define BSIM4_MOD_CLE             191
#define BSIM4_MOD_PARAMCHK        192
#define BSIM4_MOD_VERSION         193
#define BSIM4_MOD_VFBCV           194
#define BSIM4_MOD_ACDE            195
#define BSIM4_MOD_MOIN            196
#define BSIM4_MOD_NOFF            197
#define BSIM4_MOD_IJTHDFWD        198
#define BSIM4_MOD_ALPHA1          199
#define BSIM4_MOD_VFB             200
#define BSIM4_MOD_TOXM            201
#define BSIM4_MOD_TCJ             202
#define BSIM4_MOD_TCJSW           203
#define BSIM4_MOD_TCJSWG          204
#define BSIM4_MOD_TPB             205
#define BSIM4_MOD_TPBSW           206
#define BSIM4_MOD_TPBSWG          207
#define BSIM4_MOD_VOFFCV          208
#define BSIM4_MOD_GBMIN           209
#define BSIM4_MOD_RBDB            210
#define BSIM4_MOD_RBSB            211
#define BSIM4_MOD_RBPB            212
#define BSIM4_MOD_RBPS            213
#define BSIM4_MOD_RBPD            214
#define BSIM4_MOD_DMCG            215
#define BSIM4_MOD_DMCI            216
#define BSIM4_MOD_DMDG            217
#define BSIM4_MOD_XGW             218
#define BSIM4_MOD_XGL             219
#define BSIM4_MOD_RSHG            220
#define BSIM4_MOD_NGCON           221
#define BSIM4_MOD_AGIDL           222
#define BSIM4_MOD_BGIDL           223
#define BSIM4_MOD_EGIDL           224
#define BSIM4_MOD_IJTHSFWD        225
#define BSIM4_MOD_XJBVD           226
#define BSIM4_MOD_XJBVS           227
#define BSIM4_MOD_BVD             228
#define BSIM4_MOD_BVS             229
#define BSIM4_MOD_TOXP            230
#define BSIM4_MOD_DTOX            231
#define BSIM4_MOD_XRCRG1          232
#define BSIM4_MOD_XRCRG2          233
#define BSIM4_MOD_EU              234
#define BSIM4_MOD_IJTHSREV        235
#define BSIM4_MOD_IJTHDREV        236
#define BSIM4_MOD_MINV            237
#define BSIM4_MOD_VOFFL           238
#define BSIM4_MOD_PDITS           239
#define BSIM4_MOD_PDITSD          240
#define BSIM4_MOD_PDITSL          241
#define BSIM4_MOD_TNOIA           242
#define BSIM4_MOD_TNOIB           243
#define BSIM4_MOD_NTNOI           244
#define BSIM4_MOD_FPROUT          245
#define BSIM4_MOD_LPEB            246
#define BSIM4_MOD_DVTP0           247
#define BSIM4_MOD_DVTP1           248
#define BSIM4_MOD_CGIDL           249
#define BSIM4_MOD_PHIN            250
#define BSIM4_MOD_RDSWMIN         251
#define BSIM4_MOD_RSW             252
#define BSIM4_MOD_RDW             253
#define BSIM4_MOD_RDWMIN          254
#define BSIM4_MOD_RSWMIN          255
#define BSIM4_MOD_NSD             256
#define BSIM4_MOD_CKAPPAD         257
#define BSIM4_MOD_DMCGT           258
#define BSIM4_MOD_AIGC            259
#define BSIM4_MOD_BIGC            260
#define BSIM4_MOD_CIGC            261
#define BSIM4_MOD_AIGBACC         262
#define BSIM4_MOD_BIGBACC         263
#define BSIM4_MOD_CIGBACC         264            
#define BSIM4_MOD_AIGBINV         265
#define BSIM4_MOD_BIGBINV         266
#define BSIM4_MOD_CIGBINV         267
#define BSIM4_MOD_NIGC            268
#define BSIM4_MOD_NIGBACC         269
#define BSIM4_MOD_NIGBINV         270
#define BSIM4_MOD_NTOX            271
#define BSIM4_MOD_TOXREF          272
#define BSIM4_MOD_EIGBINV         273
#define BSIM4_MOD_PIGCD           274
#define BSIM4_MOD_POXEDGE         275
#define BSIM4_MOD_EPSROX          276
#define BSIM4_MOD_AIGSD           277
#define BSIM4_MOD_BIGSD           278
#define BSIM4_MOD_CIGSD           279
#define BSIM4_MOD_JSWGS           280
#define BSIM4_MOD_JSWGD           281


/* Length dependence */
#define BSIM4_MOD_LCDSC            301
#define BSIM4_MOD_LCDSCB           302
#define BSIM4_MOD_LCIT             303
#define BSIM4_MOD_LNFACTOR         304
#define BSIM4_MOD_LXJ              305
#define BSIM4_MOD_LVSAT            306
#define BSIM4_MOD_LAT              307
#define BSIM4_MOD_LA0              308
#define BSIM4_MOD_LA1              309
#define BSIM4_MOD_LA2              310
#define BSIM4_MOD_LKETA            311
#define BSIM4_MOD_LNSUB            312
#define BSIM4_MOD_LNDEP            313
#define BSIM4_MOD_LNGATE           315
#define BSIM4_MOD_LGAMMA1          316
#define BSIM4_MOD_LGAMMA2          317
#define BSIM4_MOD_LVBX             318
#define BSIM4_MOD_LVBM             320
#define BSIM4_MOD_LXT              322
#define BSIM4_MOD_LK1              325
#define BSIM4_MOD_LKT1             326
#define BSIM4_MOD_LKT1L            327
#define BSIM4_MOD_LK2              328
#define BSIM4_MOD_LKT2             329
#define BSIM4_MOD_LK3              330
#define BSIM4_MOD_LK3B             331
#define BSIM4_MOD_LW0              332
#define BSIM4_MOD_LLPE0            333
#define BSIM4_MOD_LDVT0            334
#define BSIM4_MOD_LDVT1            335
#define BSIM4_MOD_LDVT2            336
#define BSIM4_MOD_LDVT0W           337
#define BSIM4_MOD_LDVT1W           338
#define BSIM4_MOD_LDVT2W           339
#define BSIM4_MOD_LDROUT           340
#define BSIM4_MOD_LDSUB            341
#define BSIM4_MOD_LVTH0            342
#define BSIM4_MOD_LUA              343
#define BSIM4_MOD_LUA1             344
#define BSIM4_MOD_LUB              345
#define BSIM4_MOD_LUB1             346
#define BSIM4_MOD_LUC              347
#define BSIM4_MOD_LUC1             348
#define BSIM4_MOD_LU0              349
#define BSIM4_MOD_LUTE             350
#define BSIM4_MOD_LVOFF            351
#define BSIM4_MOD_LDELTA           352
#define BSIM4_MOD_LRDSW            353
#define BSIM4_MOD_LPRT             354
#define BSIM4_MOD_LLDD             355
#define BSIM4_MOD_LETA             356
#define BSIM4_MOD_LETA0            357
#define BSIM4_MOD_LETAB            358
#define BSIM4_MOD_LPCLM            359
#define BSIM4_MOD_LPDIBL1          360
#define BSIM4_MOD_LPDIBL2          361
#define BSIM4_MOD_LPSCBE1          362
#define BSIM4_MOD_LPSCBE2          363
#define BSIM4_MOD_LPVAG            364
#define BSIM4_MOD_LWR              365
#define BSIM4_MOD_LDWG             366
#define BSIM4_MOD_LDWB             367
#define BSIM4_MOD_LB0              368
#define BSIM4_MOD_LB1              369
#define BSIM4_MOD_LALPHA0          370
#define BSIM4_MOD_LBETA0           371
#define BSIM4_MOD_LPDIBLB          374
#define BSIM4_MOD_LPRWG            375
#define BSIM4_MOD_LPRWB            376
#define BSIM4_MOD_LCDSCD           377
#define BSIM4_MOD_LAGS             378

#define BSIM4_MOD_LFRINGE          381
#define BSIM4_MOD_LCGSL            383
#define BSIM4_MOD_LCGDL            384
#define BSIM4_MOD_LCKAPPAS         385
#define BSIM4_MOD_LCF              386
#define BSIM4_MOD_LCLC             387
#define BSIM4_MOD_LCLE             388
#define BSIM4_MOD_LVFBCV           389
#define BSIM4_MOD_LACDE            390
#define BSIM4_MOD_LMOIN            391
#define BSIM4_MOD_LNOFF            392
#define BSIM4_MOD_LALPHA1          394
#define BSIM4_MOD_LVFB             395
#define BSIM4_MOD_LVOFFCV          396
#define BSIM4_MOD_LAGIDL           397
#define BSIM4_MOD_LBGIDL           398
#define BSIM4_MOD_LEGIDL           399
#define BSIM4_MOD_LXRCRG1          400
#define BSIM4_MOD_LXRCRG2          401
#define BSIM4_MOD_LEU              402
#define BSIM4_MOD_LMINV            403
#define BSIM4_MOD_LPDITS           404
#define BSIM4_MOD_LPDITSD          405
#define BSIM4_MOD_LFPROUT          406
#define BSIM4_MOD_LLPEB            407
#define BSIM4_MOD_LDVTP0           408
#define BSIM4_MOD_LDVTP1           409
#define BSIM4_MOD_LCGIDL           410
#define BSIM4_MOD_LPHIN            411
#define BSIM4_MOD_LRSW             412
#define BSIM4_MOD_LRDW             413
#define BSIM4_MOD_LNSD             414
#define BSIM4_MOD_LCKAPPAD         415
#define BSIM4_MOD_LAIGC            416
#define BSIM4_MOD_LBIGC            417
#define BSIM4_MOD_LCIGC            418
#define BSIM4_MOD_LAIGBACC         419            
#define BSIM4_MOD_LBIGBACC         420
#define BSIM4_MOD_LCIGBACC         421
#define BSIM4_MOD_LAIGBINV         422
#define BSIM4_MOD_LBIGBINV         423
#define BSIM4_MOD_LCIGBINV         424
#define BSIM4_MOD_LNIGC            425
#define BSIM4_MOD_LNIGBACC         426
#define BSIM4_MOD_LNIGBINV         427
#define BSIM4_MOD_LNTOX            428
#define BSIM4_MOD_LEIGBINV         429
#define BSIM4_MOD_LPIGCD           430
#define BSIM4_MOD_LPOXEDGE         431
#define BSIM4_MOD_LAIGSD           432
#define BSIM4_MOD_LBIGSD           433
#define BSIM4_MOD_LCIGSD           434


/* Width dependence */
#define BSIM4_MOD_WCDSC            481
#define BSIM4_MOD_WCDSCB           482
#define BSIM4_MOD_WCIT             483
#define BSIM4_MOD_WNFACTOR         484
#define BSIM4_MOD_WXJ              485
#define BSIM4_MOD_WVSAT            486
#define BSIM4_MOD_WAT              487
#define BSIM4_MOD_WA0              488
#define BSIM4_MOD_WA1              489
#define BSIM4_MOD_WA2              490
#define BSIM4_MOD_WKETA            491
#define BSIM4_MOD_WNSUB            492
#define BSIM4_MOD_WNDEP            493
#define BSIM4_MOD_WNGATE           495
#define BSIM4_MOD_WGAMMA1          496
#define BSIM4_MOD_WGAMMA2          497
#define BSIM4_MOD_WVBX             498
#define BSIM4_MOD_WVBM             500
#define BSIM4_MOD_WXT              502
#define BSIM4_MOD_WK1              505
#define BSIM4_MOD_WKT1             506
#define BSIM4_MOD_WKT1L            507
#define BSIM4_MOD_WK2              508
#define BSIM4_MOD_WKT2             509
#define BSIM4_MOD_WK3              510
#define BSIM4_MOD_WK3B             511
#define BSIM4_MOD_WW0              512
#define BSIM4_MOD_WLPE0            513
#define BSIM4_MOD_WDVT0            514
#define BSIM4_MOD_WDVT1            515
#define BSIM4_MOD_WDVT2            516
#define BSIM4_MOD_WDVT0W           517
#define BSIM4_MOD_WDVT1W           518
#define BSIM4_MOD_WDVT2W           519
#define BSIM4_MOD_WDROUT           520
#define BSIM4_MOD_WDSUB            521
#define BSIM4_MOD_WVTH0            522
#define BSIM4_MOD_WUA              523
#define BSIM4_MOD_WUA1             524
#define BSIM4_MOD_WUB              525
#define BSIM4_MOD_WUB1             526
#define BSIM4_MOD_WUC              527
#define BSIM4_MOD_WUC1             528
#define BSIM4_MOD_WU0              529
#define BSIM4_MOD_WUTE             530
#define BSIM4_MOD_WVOFF            531
#define BSIM4_MOD_WDELTA           532
#define BSIM4_MOD_WRDSW            533
#define BSIM4_MOD_WPRT             534
#define BSIM4_MOD_WLDD             535
#define BSIM4_MOD_WETA             536
#define BSIM4_MOD_WETA0            537
#define BSIM4_MOD_WETAB            538
#define BSIM4_MOD_WPCLM            539
#define BSIM4_MOD_WPDIBL1          540
#define BSIM4_MOD_WPDIBL2          541
#define BSIM4_MOD_WPSCBE1          542
#define BSIM4_MOD_WPSCBE2          543
#define BSIM4_MOD_WPVAG            544
#define BSIM4_MOD_WWR              545
#define BSIM4_MOD_WDWG             546
#define BSIM4_MOD_WDWB             547
#define BSIM4_MOD_WB0              548
#define BSIM4_MOD_WB1              549
#define BSIM4_MOD_WALPHA0          550
#define BSIM4_MOD_WBETA0           551
#define BSIM4_MOD_WPDIBLB          554
#define BSIM4_MOD_WPRWG            555
#define BSIM4_MOD_WPRWB            556
#define BSIM4_MOD_WCDSCD           557
#define BSIM4_MOD_WAGS             558

#define BSIM4_MOD_WFRINGE          561
#define BSIM4_MOD_WCGSL            563
#define BSIM4_MOD_WCGDL            564
#define BSIM4_MOD_WCKAPPAS         565
#define BSIM4_MOD_WCF              566
#define BSIM4_MOD_WCLC             567
#define BSIM4_MOD_WCLE             568
#define BSIM4_MOD_WVFBCV           569
#define BSIM4_MOD_WACDE            570
#define BSIM4_MOD_WMOIN            571
#define BSIM4_MOD_WNOFF            572
#define BSIM4_MOD_WALPHA1          574
#define BSIM4_MOD_WVFB             575
#define BSIM4_MOD_WVOFFCV          576
#define BSIM4_MOD_WAGIDL           577
#define BSIM4_MOD_WBGIDL           578
#define BSIM4_MOD_WEGIDL           579
#define BSIM4_MOD_WXRCRG1          580
#define BSIM4_MOD_WXRCRG2          581
#define BSIM4_MOD_WEU              582
#define BSIM4_MOD_WMINV            583
#define BSIM4_MOD_WPDITS           584
#define BSIM4_MOD_WPDITSD          585
#define BSIM4_MOD_WFPROUT          586
#define BSIM4_MOD_WLPEB            587
#define BSIM4_MOD_WDVTP0           588
#define BSIM4_MOD_WDVTP1           589
#define BSIM4_MOD_WCGIDL           590
#define BSIM4_MOD_WPHIN            591
#define BSIM4_MOD_WRSW             592
#define BSIM4_MOD_WRDW             593
#define BSIM4_MOD_WNSD             594
#define BSIM4_MOD_WCKAPPAD         595
#define BSIM4_MOD_WAIGC            596
#define BSIM4_MOD_WBIGC            597
#define BSIM4_MOD_WCIGC            598
#define BSIM4_MOD_WAIGBACC         599            
#define BSIM4_MOD_WBIGBACC         600
#define BSIM4_MOD_WCIGBACC         601
#define BSIM4_MOD_WAIGBINV         602
#define BSIM4_MOD_WBIGBINV         603
#define BSIM4_MOD_WCIGBINV         604
#define BSIM4_MOD_WNIGC            605
#define BSIM4_MOD_WNIGBACC         606
#define BSIM4_MOD_WNIGBINV         607
#define BSIM4_MOD_WNTOX            608
#define BSIM4_MOD_WEIGBINV         609
#define BSIM4_MOD_WPIGCD           610
#define BSIM4_MOD_WPOXEDGE         611
#define BSIM4_MOD_WAIGSD           612
#define BSIM4_MOD_WBIGSD           613
#define BSIM4_MOD_WCIGSD           614


/* Cross-term dependence */
#define BSIM4_MOD_PCDSC            661
#define BSIM4_MOD_PCDSCB           662
#define BSIM4_MOD_PCIT             663
#define BSIM4_MOD_PNFACTOR         664
#define BSIM4_MOD_PXJ              665
#define BSIM4_MOD_PVSAT            666
#define BSIM4_MOD_PAT              667
#define BSIM4_MOD_PA0              668
#define BSIM4_MOD_PA1              669
#define BSIM4_MOD_PA2              670
#define BSIM4_MOD_PKETA            671
#define BSIM4_MOD_PNSUB            672
#define BSIM4_MOD_PNDEP            673
#define BSIM4_MOD_PNGATE           675
#define BSIM4_MOD_PGAMMA1          676
#define BSIM4_MOD_PGAMMA2          677
#define BSIM4_MOD_PVBX             678

#define BSIM4_MOD_PVBM             680

#define BSIM4_MOD_PXT              682
#define BSIM4_MOD_PK1              685
#define BSIM4_MOD_PKT1             686
#define BSIM4_MOD_PKT1L            687
#define BSIM4_MOD_PK2              688
#define BSIM4_MOD_PKT2             689
#define BSIM4_MOD_PK3              690
#define BSIM4_MOD_PK3B             691
#define BSIM4_MOD_PW0              692
#define BSIM4_MOD_PLPE0            693

#define BSIM4_MOD_PDVT0            694
#define BSIM4_MOD_PDVT1            695
#define BSIM4_MOD_PDVT2            696

#define BSIM4_MOD_PDVT0W           697
#define BSIM4_MOD_PDVT1W           698
#define BSIM4_MOD_PDVT2W           699

#define BSIM4_MOD_PDROUT           700
#define BSIM4_MOD_PDSUB            701
#define BSIM4_MOD_PVTH0            702
#define BSIM4_MOD_PUA              703
#define BSIM4_MOD_PUA1             704
#define BSIM4_MOD_PUB              705
#define BSIM4_MOD_PUB1             706
#define BSIM4_MOD_PUC              707
#define BSIM4_MOD_PUC1             708
#define BSIM4_MOD_PU0              709
#define BSIM4_MOD_PUTE             710
#define BSIM4_MOD_PVOFF            711
#define BSIM4_MOD_PDELTA           712
#define BSIM4_MOD_PRDSW            713
#define BSIM4_MOD_PPRT             714
#define BSIM4_MOD_PLDD             715
#define BSIM4_MOD_PETA             716
#define BSIM4_MOD_PETA0            717
#define BSIM4_MOD_PETAB            718
#define BSIM4_MOD_PPCLM            719
#define BSIM4_MOD_PPDIBL1          720
#define BSIM4_MOD_PPDIBL2          721
#define BSIM4_MOD_PPSCBE1          722
#define BSIM4_MOD_PPSCBE2          723
#define BSIM4_MOD_PPVAG            724
#define BSIM4_MOD_PWR              725
#define BSIM4_MOD_PDWG             726
#define BSIM4_MOD_PDWB             727
#define BSIM4_MOD_PB0              728
#define BSIM4_MOD_PB1              729
#define BSIM4_MOD_PALPHA0          730
#define BSIM4_MOD_PBETA0           731
#define BSIM4_MOD_PPDIBLB          734

#define BSIM4_MOD_PPRWG            735
#define BSIM4_MOD_PPRWB            736

#define BSIM4_MOD_PCDSCD           737
#define BSIM4_MOD_PAGS             738

#define BSIM4_MOD_PFRINGE          741
#define BSIM4_MOD_PCGSL            743
#define BSIM4_MOD_PCGDL            744
#define BSIM4_MOD_PCKAPPAS         745
#define BSIM4_MOD_PCF              746
#define BSIM4_MOD_PCLC             747
#define BSIM4_MOD_PCLE             748
#define BSIM4_MOD_PVFBCV           749
#define BSIM4_MOD_PACDE            750
#define BSIM4_MOD_PMOIN            751
#define BSIM4_MOD_PNOFF            752
#define BSIM4_MOD_PALPHA1          754
#define BSIM4_MOD_PVFB             755
#define BSIM4_MOD_PVOFFCV          756
#define BSIM4_MOD_PAGIDL           757
#define BSIM4_MOD_PBGIDL           758
#define BSIM4_MOD_PEGIDL           759
#define BSIM4_MOD_PXRCRG1          760
#define BSIM4_MOD_PXRCRG2          761
#define BSIM4_MOD_PEU              762
#define BSIM4_MOD_PMINV            763
#define BSIM4_MOD_PPDITS           764
#define BSIM4_MOD_PPDITSD          765
#define BSIM4_MOD_PFPROUT          766
#define BSIM4_MOD_PLPEB            767
#define BSIM4_MOD_PDVTP0           768
#define BSIM4_MOD_PDVTP1           769
#define BSIM4_MOD_PCGIDL           770
#define BSIM4_MOD_PPHIN            771
#define BSIM4_MOD_PRSW             772
#define BSIM4_MOD_PRDW             773
#define BSIM4_MOD_PNSD             774
#define BSIM4_MOD_PCKAPPAD         775
#define BSIM4_MOD_PAIGC            776
#define BSIM4_MOD_PBIGC            777
#define BSIM4_MOD_PCIGC            778
#define BSIM4_MOD_PAIGBACC         779            
#define BSIM4_MOD_PBIGBACC         780
#define BSIM4_MOD_PCIGBACC         781
#define BSIM4_MOD_PAIGBINV         782
#define BSIM4_MOD_PBIGBINV         783
#define BSIM4_MOD_PCIGBINV         784
#define BSIM4_MOD_PNIGC            785
#define BSIM4_MOD_PNIGBACC         786
#define BSIM4_MOD_PNIGBINV         787
#define BSIM4_MOD_PNTOX            788
#define BSIM4_MOD_PEIGBINV         789
#define BSIM4_MOD_PPIGCD           790
#define BSIM4_MOD_PPOXEDGE         791
#define BSIM4_MOD_PAIGSD           792
#define BSIM4_MOD_PBIGSD           793
#define BSIM4_MOD_PCIGSD           794

#define BSIM4_MOD_TNOM             831
#define BSIM4_MOD_CGSO             832
#define BSIM4_MOD_CGDO             833
#define BSIM4_MOD_CGBO             834
#define BSIM4_MOD_XPART            835
#define BSIM4_MOD_RSH              836
#define BSIM4_MOD_JSS              837
#define BSIM4_MOD_PBS              838
#define BSIM4_MOD_MJS              839
#define BSIM4_MOD_PBSWS            840
#define BSIM4_MOD_MJSWS            841
#define BSIM4_MOD_CJS              842
#define BSIM4_MOD_CJSWS            843
#define BSIM4_MOD_NMOS             844
#define BSIM4_MOD_PMOS             845
#define BSIM4_MOD_NOIA             846
#define BSIM4_MOD_NOIB             847
#define BSIM4_MOD_NOIC             848
#define BSIM4_MOD_LINT             849
#define BSIM4_MOD_LL               850
#define BSIM4_MOD_LLN              851
#define BSIM4_MOD_LW               852
#define BSIM4_MOD_LWN              853
#define BSIM4_MOD_LWL              854
#define BSIM4_MOD_LMIN             855
#define BSIM4_MOD_LMAX             856
#define BSIM4_MOD_WINT             857
#define BSIM4_MOD_WL               858
#define BSIM4_MOD_WLN              859
#define BSIM4_MOD_WW               860
#define BSIM4_MOD_WWN              861
#define BSIM4_MOD_WWL              862
#define BSIM4_MOD_WMIN             863
#define BSIM4_MOD_WMAX             864
#define BSIM4_MOD_DWC              865
#define BSIM4_MOD_DLC              866
#define BSIM4_MOD_XL               867
#define BSIM4_MOD_XW               868
#define BSIM4_MOD_EM               869
#define BSIM4_MOD_EF               870
#define BSIM4_MOD_AF               871
#define BSIM4_MOD_KF               872
#define BSIM4_MOD_NJS              873
#define BSIM4_MOD_XTIS             874
#define BSIM4_MOD_PBSWGS           875
#define BSIM4_MOD_MJSWGS           876
#define BSIM4_MOD_CJSWGS           877
#define BSIM4_MOD_JSWS             878
#define BSIM4_MOD_LLC              879
#define BSIM4_MOD_LWC              880
#define BSIM4_MOD_LWLC             881
#define BSIM4_MOD_WLC              882
#define BSIM4_MOD_WWC              883
#define BSIM4_MOD_WWLC             884
#define BSIM4_MOD_DWJ              885
#define BSIM4_MOD_JSD              886
#define BSIM4_MOD_PBD              887
#define BSIM4_MOD_MJD              888
#define BSIM4_MOD_PBSWD            889
#define BSIM4_MOD_MJSWD            890
#define BSIM4_MOD_CJD              891
#define BSIM4_MOD_CJSWD            892
#define BSIM4_MOD_NJD              893
#define BSIM4_MOD_XTID             894
#define BSIM4_MOD_PBSWGD           895
#define BSIM4_MOD_MJSWGD           896
#define BSIM4_MOD_CJSWGD           897
#define BSIM4_MOD_JSWD             898
#define BSIM4_MOD_DLCIG            899

/* device questions */
#define BSIM4_DNODE                945
#define BSIM4_GNODEEXT             946
#define BSIM4_SNODE                947
#define BSIM4_BNODE                948
#define BSIM4_DNODEPRIME           949
#define BSIM4_GNODEPRIME           950
#define BSIM4_GNODEMIDE            951
#define BSIM4_GNODEMID             952
#define BSIM4_SNODEPRIME           953
#define BSIM4_BNODEPRIME           954
#define BSIM4_DBNODE               955
#define BSIM4_SBNODE               956
#define BSIM4_VBD                  957
#define BSIM4_VBS                  958
#define BSIM4_VGS                  959
#define BSIM4_VDS                  960
#define BSIM4_CD                   961
#define BSIM4_CBS                  962
#define BSIM4_CBD                  963
#define BSIM4_GM                   964
#define BSIM4_GDS                  965
#define BSIM4_GMBS                 966
#define BSIM4_GBD                  967
#define BSIM4_GBS                  968
#define BSIM4_QB                   969
#define BSIM4_CQB                  970
#define BSIM4_QG                   971
#define BSIM4_CQG                  972
#define BSIM4_QD                   973
#define BSIM4_CQD                  974
#define BSIM4_CGGB                 975
#define BSIM4_CGDB                 976
#define BSIM4_CGSB                 977
#define BSIM4_CBGB                 978
#define BSIM4_CAPBD                979
#define BSIM4_CQBD                 980
#define BSIM4_CAPBS                981
#define BSIM4_CQBS                 982
#define BSIM4_CDGB                 983
#define BSIM4_CDDB                 984
#define BSIM4_CDSB                 985
#define BSIM4_VON                  986
#define BSIM4_VDSAT                987
#define BSIM4_QBS                  988
#define BSIM4_QBD                  989
#define BSIM4_SOURCECONDUCT        990
#define BSIM4_DRAINCONDUCT         991
#define BSIM4_CBDB                 992
#define BSIM4_CBSB                 993
#define BSIM4_CSUB		   994
#define BSIM4_QINV		   995
#define BSIM4_IGIDL		   996
#define BSIM4_CSGB                 997
#define BSIM4_CSDB                 998
#define BSIM4_CSSB                 999
#define BSIM4_CGBB                 1000
#define BSIM4_CDBB                 1001
#define BSIM4_CSBB                 1002
#define BSIM4_CBBB                 1003
#define BSIM4_QS                   1004
#define BSIM4_IGISL		   1005
#define BSIM4_IGS		   1006
#define BSIM4_IGD		   1007
#define BSIM4_IGB		   1008
#define BSIM4_IGCS		   1009
#define BSIM4_IGCD		   1010


#
#include "bsim4ext.h"

#ifdef __STDC__
extern void BSIM4evaluate(double,double,double,BSIM4instance*,BSIM4model*,
        double*,double*,double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, CKTcircuit*);
extern int BSIM4debug(BSIM4model*, BSIM4instance*, CKTcircuit*, int);
extern int BSIM4checkModel(BSIM4model*, BSIM4instance*, CKTcircuit*);
#else /* stdc */
extern void BSIM4evaluate();
extern int BSIM4debug();
extern int BSIM4checkModel();
#endif /* stdc */

#endif /*BSIM4*/
