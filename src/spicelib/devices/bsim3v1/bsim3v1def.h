/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan
File: bsim3v1def.h
**********/

#ifndef BSIM3V1
#define BSIM3V1

#include "ifsim.h"
#include "gendefs.h"
#include "cktdefs.h"
#include "complex.h"
#include "noisedef.h"         

typedef struct sBSIM3V1instance
{
    struct sBSIM3V1model *BSIM3V1modPtr;
    struct sBSIM3V1instance *BSIM3V1nextInstance;
    IFuid BSIM3V1name;
    int BSIM3V1owner;      /* number of owner process */
    int BSIM3V1states;     /* index into state table for this device */

    int BSIM3V1dNode;
    int BSIM3V1gNode;
    int BSIM3V1sNode;
    int BSIM3V1bNode;
    int BSIM3V1dNodePrime;
    int BSIM3V1sNodePrime;
    int BSIM3V1qNode; /* MCJ */

    /* MCJ */
    double BSIM3V1ueff;
    double BSIM3V1thetavth; 
    double BSIM3V1von;
    double BSIM3V1vdsat;
    double BSIM3V1cgdo;
    double BSIM3V1cgso;

    double BSIM3V1l;
    double BSIM3V1w;
    double BSIM3V1drainArea;
    double BSIM3V1sourceArea;
    double BSIM3V1drainSquares;
    double BSIM3V1sourceSquares;
    double BSIM3V1drainPerimeter;
    double BSIM3V1sourcePerimeter;
    double BSIM3V1sourceConductance;
    double BSIM3V1drainConductance;
    double BSIM3V1m;

    double BSIM3V1icVBS;
    double BSIM3V1icVDS;
    double BSIM3V1icVGS;
    int BSIM3V1off;
    int BSIM3V1mode;
    int BSIM3V1nqsMod;

    /* OP point */
    double BSIM3V1qinv;
    double BSIM3V1cd;
    double BSIM3V1cbs;
    double BSIM3V1cbd;
    double BSIM3V1csub;
    double BSIM3V1gm;
    double BSIM3V1gds;
    double BSIM3V1gmbs;
    double BSIM3V1gbd;
    double BSIM3V1gbs;

    double BSIM3V1gbbs;
    double BSIM3V1gbgs;
    double BSIM3V1gbds;

    double BSIM3V1cggb;
    double BSIM3V1cgdb;
    double BSIM3V1cgsb;
    double BSIM3V1cbgb;
    double BSIM3V1cbdb;
    double BSIM3V1cbsb;
    double BSIM3V1cdgb;
    double BSIM3V1cddb;
    double BSIM3V1cdsb;
    double BSIM3V1capbd;
    double BSIM3V1capbs;

    double BSIM3V1cqgb;
    double BSIM3V1cqdb;
    double BSIM3V1cqsb;
    double BSIM3V1cqbb;

    double BSIM3V1gtau;
    double BSIM3V1gtg;
    double BSIM3V1gtd;
    double BSIM3V1gts;
    double BSIM3V1gtb;
    double BSIM3V1tconst;

    struct bsim3v1SizeDependParam  *pParam;

    unsigned BSIM3V1lGiven :1;
    unsigned BSIM3V1wGiven :1;
    unsigned BSIM3V1drainAreaGiven :1;
    unsigned BSIM3V1sourceAreaGiven    :1;
    unsigned BSIM3V1drainSquaresGiven  :1;
    unsigned BSIM3V1sourceSquaresGiven :1;
    unsigned BSIM3V1drainPerimeterGiven    :1;
    unsigned BSIM3V1sourcePerimeterGiven   :1;
    unsigned BSIM3V1dNodePrimeSet  :1;
    unsigned BSIM3V1sNodePrimeSet  :1;
    unsigned BSIM3V1icVBSGiven :1;
    unsigned BSIM3V1icVDSGiven :1;
    unsigned BSIM3V1icVGSGiven :1;
    unsigned BSIM3V1nqsModGiven :1;

    double *BSIM3V1DdPtr;
    double *BSIM3V1GgPtr;
    double *BSIM3V1SsPtr;
    double *BSIM3V1BbPtr;
    double *BSIM3V1DPdpPtr;
    double *BSIM3V1SPspPtr;
    double *BSIM3V1DdpPtr;
    double *BSIM3V1GbPtr;
    double *BSIM3V1GdpPtr;
    double *BSIM3V1GspPtr;
    double *BSIM3V1SspPtr;
    double *BSIM3V1BdpPtr;
    double *BSIM3V1BspPtr;
    double *BSIM3V1DPspPtr;
    double *BSIM3V1DPdPtr;
    double *BSIM3V1BgPtr;
    double *BSIM3V1DPgPtr;
    double *BSIM3V1SPgPtr;
    double *BSIM3V1SPsPtr;
    double *BSIM3V1DPbPtr;
    double *BSIM3V1SPbPtr;
    double *BSIM3V1SPdpPtr;

    double *BSIM3V1QqPtr;
    double *BSIM3V1QdpPtr;
    double *BSIM3V1QgPtr;
    double *BSIM3V1QspPtr;
    double *BSIM3V1QbPtr;
    double *BSIM3V1DPqPtr;
    double *BSIM3V1GqPtr;
    double *BSIM3V1SPqPtr;
    double *BSIM3V1BqPtr;

#define BSIM3V1vbd BSIM3V1states+ 0
#define BSIM3V1vbs BSIM3V1states+ 1
#define BSIM3V1vgs BSIM3V1states+ 2
#define BSIM3V1vds BSIM3V1states+ 3

#define BSIM3V1qb BSIM3V1states+ 4
#define BSIM3V1cqb BSIM3V1states+ 5
#define BSIM3V1qg BSIM3V1states+ 6
#define BSIM3V1cqg BSIM3V1states+ 7
#define BSIM3V1qd BSIM3V1states+ 8
#define BSIM3V1cqd BSIM3V1states+ 9

#define BSIM3V1qbs  BSIM3V1states+ 10
#define BSIM3V1qbd  BSIM3V1states+ 11

#define BSIM3V1qcheq BSIM3V1states+ 12
#define BSIM3V1cqcheq BSIM3V1states+ 13
#define BSIM3V1qcdump BSIM3V1states+ 14
#define BSIM3V1cqcdump BSIM3V1states+ 15

#define BSIM3V1tau BSIM3V1states+ 16
#define BSIM3V1qdef BSIM3V1states+ 17

#define BSIM3V1numStates 18


/* indices to the array of BSIM3V1 NOISE SOURCES */

#define BSIM3V1RDNOIZ       0
#define BSIM3V1RSNOIZ       1
#define BSIM3V1IDNOIZ       2
#define BSIM3V1FLNOIZ       3
#define BSIM3V1TOTNOIZ      4

#define BSIM3V1NSRCS        5     /* the number of MOSFET(3) noise sources */

#ifndef NONOISE
    double BSIM3V1nVar[NSTATVARS][BSIM3V1NSRCS];
#else /* NONOISE */
        double **BSIM3V1nVar;
#endif /* NONOISE */

} BSIM3V1instance ;

struct bsim3v1SizeDependParam
{
    double Width;
    double Length;

    double BSIM3V1cdsc;           
    double BSIM3V1cdscb;    
    double BSIM3V1cdscd;       
    double BSIM3V1cit;           
    double BSIM3V1nfactor;      
    double BSIM3V1xj;
    double BSIM3V1vsat;         
    double BSIM3V1at;         
    double BSIM3V1a0;   
    double BSIM3V1ags;      
    double BSIM3V1a1;         
    double BSIM3V1a2;         
    double BSIM3V1keta;     
    double BSIM3V1nsub;
    double BSIM3V1npeak;        
    double BSIM3V1ngate;        
    double BSIM3V1gamma1;      
    double BSIM3V1gamma2;     
    double BSIM3V1vbx;      
    double BSIM3V1vbi;       
    double BSIM3V1vbm;       
    double BSIM3V1vbsc;       
    double BSIM3V1xt;       
    double BSIM3V1phi;
    double BSIM3V1litl;
    double BSIM3V1k1;
    double BSIM3V1kt1;
    double BSIM3V1kt1l;
    double BSIM3V1kt2;
    double BSIM3V1k2;
    double BSIM3V1k3;
    double BSIM3V1k3b;
    double BSIM3V1w0;
    double BSIM3V1nlx;
    double BSIM3V1dvt0;      
    double BSIM3V1dvt1;      
    double BSIM3V1dvt2;      
    double BSIM3V1dvt0w;      
    double BSIM3V1dvt1w;      
    double BSIM3V1dvt2w;      
    double BSIM3V1drout;      
    double BSIM3V1dsub;      
    double BSIM3V1vth0;
    double BSIM3V1ua;
    double BSIM3V1ua1;
    double BSIM3V1ub;
    double BSIM3V1ub1;
    double BSIM3V1uc;
    double BSIM3V1uc1;
    double BSIM3V1u0;
    double BSIM3V1ute;
    double BSIM3V1voff;
    double BSIM3V1vfb;
    double BSIM3V1delta;
    double BSIM3V1rdsw;       
    double BSIM3V1rds0;       
    double BSIM3V1prwg;       
    double BSIM3V1prwb;       
    double BSIM3V1prt;       
    double BSIM3V1eta0;         
    double BSIM3V1etab;         
    double BSIM3V1pclm;      
    double BSIM3V1pdibl1;      
    double BSIM3V1pdibl2;      
    double BSIM3V1pdiblb;      
    double BSIM3V1pscbe1;       
    double BSIM3V1pscbe2;       
    double BSIM3V1pvag;       
    double BSIM3V1wr;
    double BSIM3V1dwg;
    double BSIM3V1dwb;
    double BSIM3V1b0;
    double BSIM3V1b1;
    double BSIM3V1alpha0;
    double BSIM3V1beta0;


    /* CV model */
    double BSIM3V1elm;
    double BSIM3V1cgsl;
    double BSIM3V1cgdl;
    double BSIM3V1ckappa;
    double BSIM3V1cf;
    double BSIM3V1clc;
    double BSIM3V1cle;
    double BSIM3V1vfbcv;


/* Pre-calculated constants */

    double BSIM3V1dw;
    double BSIM3V1dl;
    double BSIM3V1leff;
    double BSIM3V1weff;

    double BSIM3V1dwc;
    double BSIM3V1dlc;
    double BSIM3V1leffCV;
    double BSIM3V1weffCV;
    double BSIM3V1abulkCVfactor;
    double BSIM3V1cgso;
    double BSIM3V1cgdo;
    double BSIM3V1cgbo;

    double BSIM3V1u0temp;       
    double BSIM3V1vsattemp;   
    double BSIM3V1sqrtPhi;   
    double BSIM3V1phis3;   
    double BSIM3V1Xdep0;          
    double BSIM3V1sqrtXdep0;          
    double BSIM3V1theta0vb0;
    double BSIM3V1thetaRout; 

    double BSIM3V1cof1;
    double BSIM3V1cof2;
    double BSIM3V1cof3;
    double BSIM3V1cof4;
    double BSIM3V1cdep0;
    struct bsim3v1SizeDependParam  *pNext;
};


typedef struct sBSIM3V1model 
{
    int BSIM3V1modType;
    struct sBSIM3V1model *BSIM3V1nextModel;
    BSIM3V1instance *BSIM3V1instances;
    IFuid BSIM3V1modName; 
    int BSIM3V1type;

    int    BSIM3V1mobMod;
    int    BSIM3V1capMod;
    int    BSIM3V1nqsMod;
    int    BSIM3V1noiMod;
    int    BSIM3V1binUnit;
    int    BSIM3V1paramChk;
    double BSIM3V1version;             
    double BSIM3V1tox;             
    double BSIM3V1cdsc;           
    double BSIM3V1cdscb; 
    double BSIM3V1cdscd;          
    double BSIM3V1cit;           
    double BSIM3V1nfactor;      
    double BSIM3V1xj;
    double BSIM3V1vsat;         
    double BSIM3V1at;         
    double BSIM3V1a0;   
    double BSIM3V1ags;      
    double BSIM3V1a1;         
    double BSIM3V1a2;         
    double BSIM3V1keta;     
    double BSIM3V1nsub;
    double BSIM3V1npeak;        
    double BSIM3V1ngate;        
    double BSIM3V1gamma1;      
    double BSIM3V1gamma2;     
    double BSIM3V1vbx;      
    double BSIM3V1vbm;       
    double BSIM3V1xt;       
    double BSIM3V1k1;
    double BSIM3V1kt1;
    double BSIM3V1kt1l;
    double BSIM3V1kt2;
    double BSIM3V1k2;
    double BSIM3V1k3;
    double BSIM3V1k3b;
    double BSIM3V1w0;
    double BSIM3V1nlx;
    double BSIM3V1dvt0;      
    double BSIM3V1dvt1;      
    double BSIM3V1dvt2;      
    double BSIM3V1dvt0w;      
    double BSIM3V1dvt1w;      
    double BSIM3V1dvt2w;      
    double BSIM3V1drout;      
    double BSIM3V1dsub;      
    double BSIM3V1vth0;
    double BSIM3V1ua;
    double BSIM3V1ua1;
    double BSIM3V1ub;
    double BSIM3V1ub1;
    double BSIM3V1uc;
    double BSIM3V1uc1;
    double BSIM3V1u0;
    double BSIM3V1ute;
    double BSIM3V1voff;
    double BSIM3V1delta;
    double BSIM3V1rdsw;       
    double BSIM3V1prwg;
    double BSIM3V1prwb;
    double BSIM3V1prt;       
    double BSIM3V1eta0;         
    double BSIM3V1etab;         
    double BSIM3V1pclm;      
    double BSIM3V1pdibl1;      
    double BSIM3V1pdibl2;      
    double BSIM3V1pdiblb;
    double BSIM3V1pscbe1;       
    double BSIM3V1pscbe2;       
    double BSIM3V1pvag;       
    double BSIM3V1wr;
    double BSIM3V1dwg;
    double BSIM3V1dwb;
    double BSIM3V1b0;
    double BSIM3V1b1;
    double BSIM3V1alpha0;
    double BSIM3V1beta0;
/* serban */
    double BSIM3V1hdif;

    /* CV model */
    double BSIM3V1elm;
    double BSIM3V1cgsl;
    double BSIM3V1cgdl;
    double BSIM3V1ckappa;
    double BSIM3V1cf;
    double BSIM3V1vfbcv;
    double BSIM3V1clc;
    double BSIM3V1cle;
    double BSIM3V1dwc;
    double BSIM3V1dlc;

    /* Length Dependence */
    double BSIM3V1lcdsc;           
    double BSIM3V1lcdscb; 
    double BSIM3V1lcdscd;          
    double BSIM3V1lcit;           
    double BSIM3V1lnfactor;      
    double BSIM3V1lxj;
    double BSIM3V1lvsat;         
    double BSIM3V1lat;         
    double BSIM3V1la0;   
    double BSIM3V1lags;      
    double BSIM3V1la1;         
    double BSIM3V1la2;         
    double BSIM3V1lketa;     
    double BSIM3V1lnsub;
    double BSIM3V1lnpeak;        
    double BSIM3V1lngate;        
    double BSIM3V1lgamma1;      
    double BSIM3V1lgamma2;     
    double BSIM3V1lvbx;      
    double BSIM3V1lvbm;       
    double BSIM3V1lxt;       
    double BSIM3V1lk1;
    double BSIM3V1lkt1;
    double BSIM3V1lkt1l;
    double BSIM3V1lkt2;
    double BSIM3V1lk2;
    double BSIM3V1lk3;
    double BSIM3V1lk3b;
    double BSIM3V1lw0;
    double BSIM3V1lnlx;
    double BSIM3V1ldvt0;      
    double BSIM3V1ldvt1;      
    double BSIM3V1ldvt2;      
    double BSIM3V1ldvt0w;      
    double BSIM3V1ldvt1w;      
    double BSIM3V1ldvt2w;      
    double BSIM3V1ldrout;      
    double BSIM3V1ldsub;      
    double BSIM3V1lvth0;
    double BSIM3V1lua;
    double BSIM3V1lua1;
    double BSIM3V1lub;
    double BSIM3V1lub1;
    double BSIM3V1luc;
    double BSIM3V1luc1;
    double BSIM3V1lu0;
    double BSIM3V1lute;
    double BSIM3V1lvoff;
    double BSIM3V1ldelta;
    double BSIM3V1lrdsw;       
    double BSIM3V1lprwg;
    double BSIM3V1lprwb;
    double BSIM3V1lprt;       
    double BSIM3V1leta0;         
    double BSIM3V1letab;         
    double BSIM3V1lpclm;      
    double BSIM3V1lpdibl1;      
    double BSIM3V1lpdibl2;      
    double BSIM3V1lpdiblb;
    double BSIM3V1lpscbe1;       
    double BSIM3V1lpscbe2;       
    double BSIM3V1lpvag;       
    double BSIM3V1lwr;
    double BSIM3V1ldwg;
    double BSIM3V1ldwb;
    double BSIM3V1lb0;
    double BSIM3V1lb1;
    double BSIM3V1lalpha0;
    double BSIM3V1lbeta0;

    /* CV model */
    double BSIM3V1lelm;
    double BSIM3V1lcgsl;
    double BSIM3V1lcgdl;
    double BSIM3V1lckappa;
    double BSIM3V1lcf;
    double BSIM3V1lclc;
    double BSIM3V1lcle;
    double BSIM3V1lvfbcv;

    /* Width Dependence */
    double BSIM3V1wcdsc;           
    double BSIM3V1wcdscb; 
    double BSIM3V1wcdscd;          
    double BSIM3V1wcit;           
    double BSIM3V1wnfactor;      
    double BSIM3V1wxj;
    double BSIM3V1wvsat;         
    double BSIM3V1wat;         
    double BSIM3V1wa0;   
    double BSIM3V1wags;      
    double BSIM3V1wa1;         
    double BSIM3V1wa2;         
    double BSIM3V1wketa;     
    double BSIM3V1wnsub;
    double BSIM3V1wnpeak;        
    double BSIM3V1wngate;        
    double BSIM3V1wgamma1;      
    double BSIM3V1wgamma2;     
    double BSIM3V1wvbx;      
    double BSIM3V1wvbm;       
    double BSIM3V1wxt;       
    double BSIM3V1wk1;
    double BSIM3V1wkt1;
    double BSIM3V1wkt1l;
    double BSIM3V1wkt2;
    double BSIM3V1wk2;
    double BSIM3V1wk3;
    double BSIM3V1wk3b;
    double BSIM3V1ww0;
    double BSIM3V1wnlx;
    double BSIM3V1wdvt0;      
    double BSIM3V1wdvt1;      
    double BSIM3V1wdvt2;      
    double BSIM3V1wdvt0w;      
    double BSIM3V1wdvt1w;      
    double BSIM3V1wdvt2w;      
    double BSIM3V1wdrout;      
    double BSIM3V1wdsub;      
    double BSIM3V1wvth0;
    double BSIM3V1wua;
    double BSIM3V1wua1;
    double BSIM3V1wub;
    double BSIM3V1wub1;
    double BSIM3V1wuc;
    double BSIM3V1wuc1;
    double BSIM3V1wu0;
    double BSIM3V1wute;
    double BSIM3V1wvoff;
    double BSIM3V1wdelta;
    double BSIM3V1wrdsw;       
    double BSIM3V1wprwg;
    double BSIM3V1wprwb;
    double BSIM3V1wprt;       
    double BSIM3V1weta0;         
    double BSIM3V1wetab;         
    double BSIM3V1wpclm;      
    double BSIM3V1wpdibl1;      
    double BSIM3V1wpdibl2;      
    double BSIM3V1wpdiblb;
    double BSIM3V1wpscbe1;       
    double BSIM3V1wpscbe2;       
    double BSIM3V1wpvag;       
    double BSIM3V1wwr;
    double BSIM3V1wdwg;
    double BSIM3V1wdwb;
    double BSIM3V1wb0;
    double BSIM3V1wb1;
    double BSIM3V1walpha0;
    double BSIM3V1wbeta0;

    /* CV model */
    double BSIM3V1welm;
    double BSIM3V1wcgsl;
    double BSIM3V1wcgdl;
    double BSIM3V1wckappa;
    double BSIM3V1wcf;
    double BSIM3V1wclc;
    double BSIM3V1wcle;
    double BSIM3V1wvfbcv;

    /* Cross-term Dependence */
    double BSIM3V1pcdsc;           
    double BSIM3V1pcdscb; 
    double BSIM3V1pcdscd;          
    double BSIM3V1pcit;           
    double BSIM3V1pnfactor;      
    double BSIM3V1pxj;
    double BSIM3V1pvsat;         
    double BSIM3V1pat;         
    double BSIM3V1pa0;   
    double BSIM3V1pags;      
    double BSIM3V1pa1;         
    double BSIM3V1pa2;         
    double BSIM3V1pketa;     
    double BSIM3V1pnsub;
    double BSIM3V1pnpeak;        
    double BSIM3V1pngate;        
    double BSIM3V1pgamma1;      
    double BSIM3V1pgamma2;     
    double BSIM3V1pvbx;      
    double BSIM3V1pvbm;       
    double BSIM3V1pxt;       
    double BSIM3V1pk1;
    double BSIM3V1pkt1;
    double BSIM3V1pkt1l;
    double BSIM3V1pkt2;
    double BSIM3V1pk2;
    double BSIM3V1pk3;
    double BSIM3V1pk3b;
    double BSIM3V1pw0;
    double BSIM3V1pnlx;
    double BSIM3V1pdvt0;      
    double BSIM3V1pdvt1;      
    double BSIM3V1pdvt2;      
    double BSIM3V1pdvt0w;      
    double BSIM3V1pdvt1w;      
    double BSIM3V1pdvt2w;      
    double BSIM3V1pdrout;      
    double BSIM3V1pdsub;      
    double BSIM3V1pvth0;
    double BSIM3V1pua;
    double BSIM3V1pua1;
    double BSIM3V1pub;
    double BSIM3V1pub1;
    double BSIM3V1puc;
    double BSIM3V1puc1;
    double BSIM3V1pu0;
    double BSIM3V1pute;
    double BSIM3V1pvoff;
    double BSIM3V1pdelta;
    double BSIM3V1prdsw;
    double BSIM3V1pprwg;
    double BSIM3V1pprwb;
    double BSIM3V1pprt;       
    double BSIM3V1peta0;         
    double BSIM3V1petab;         
    double BSIM3V1ppclm;      
    double BSIM3V1ppdibl1;      
    double BSIM3V1ppdibl2;      
    double BSIM3V1ppdiblb;
    double BSIM3V1ppscbe1;       
    double BSIM3V1ppscbe2;       
    double BSIM3V1ppvag;       
    double BSIM3V1pwr;
    double BSIM3V1pdwg;
    double BSIM3V1pdwb;
    double BSIM3V1pb0;
    double BSIM3V1pb1;
    double BSIM3V1palpha0;
    double BSIM3V1pbeta0;

    /* CV model */
    double BSIM3V1pelm;
    double BSIM3V1pcgsl;
    double BSIM3V1pcgdl;
    double BSIM3V1pckappa;
    double BSIM3V1pcf;
    double BSIM3V1pclc;
    double BSIM3V1pcle;
    double BSIM3V1pvfbcv;

    double BSIM3V1tnom;
    double BSIM3V1cgso;
    double BSIM3V1cgdo;
    double BSIM3V1cgbo;
    double BSIM3V1xpart;
    double BSIM3V1cFringOut;
    double BSIM3V1cFringMax;

    double BSIM3V1sheetResistance;
    double BSIM3V1jctSatCurDensity;
    double BSIM3V1jctSidewallSatCurDensity;
    double BSIM3V1bulkJctPotential;
    double BSIM3V1bulkJctBotGradingCoeff;
    double BSIM3V1bulkJctSideGradingCoeff;
    double BSIM3V1bulkJctGateSideGradingCoeff;
    double BSIM3V1sidewallJctPotential;
    double BSIM3V1GatesidewallJctPotential;
    double BSIM3V1unitAreaJctCap;
    double BSIM3V1unitLengthSidewallJctCap;
    double BSIM3V1unitLengthGateSidewallJctCap;
    double BSIM3V1jctEmissionCoeff;
    double BSIM3V1jctTempExponent;

    double BSIM3V1Lint;
    double BSIM3V1Ll;
    double BSIM3V1Lln;
    double BSIM3V1Lw;
    double BSIM3V1Lwn;
    double BSIM3V1Lwl;
    double BSIM3V1Lmin;
    double BSIM3V1Lmax;

    double BSIM3V1Wint;
    double BSIM3V1Wl;
    double BSIM3V1Wln;
    double BSIM3V1Ww;
    double BSIM3V1Wwn;
    double BSIM3V1Wwl;
    double BSIM3V1Wmin;
    double BSIM3V1Wmax;


/* Pre-calculated constants */
    /* MCJ: move to size-dependent param. */
    double BSIM3V1vtm;   
    double BSIM3V1cox;
    double BSIM3V1cof1;
    double BSIM3V1cof2;
    double BSIM3V1cof3;
    double BSIM3V1cof4;
    double BSIM3V1vcrit;
    double BSIM3V1factor1;
    double BSIM3V1jctTempSatCurDensity;
    double BSIM3V1jctSidewallTempSatCurDensity;

    double BSIM3V1oxideTrapDensityA;      
    double BSIM3V1oxideTrapDensityB;     
    double BSIM3V1oxideTrapDensityC;  
    double BSIM3V1em;  
    double BSIM3V1ef;  
    double BSIM3V1af;  
    double BSIM3V1kf;  

    struct bsim3v1SizeDependParam *pSizeDependParamKnot;

    /* Flags */
    unsigned  BSIM3V1mobModGiven :1;
    unsigned  BSIM3V1binUnitGiven :1;
    unsigned  BSIM3V1capModGiven :1;
    unsigned  BSIM3V1paramChkGiven :1;
    unsigned  BSIM3V1nqsModGiven :1;
    unsigned  BSIM3V1noiModGiven :1;
    unsigned  BSIM3V1typeGiven   :1;
    unsigned  BSIM3V1toxGiven   :1;
    unsigned  BSIM3V1versionGiven   :1;

    unsigned  BSIM3V1cdscGiven   :1;
    unsigned  BSIM3V1cdscbGiven   :1;
    unsigned  BSIM3V1cdscdGiven   :1;
    unsigned  BSIM3V1citGiven   :1;
    unsigned  BSIM3V1nfactorGiven   :1;
    unsigned  BSIM3V1xjGiven   :1;
    unsigned  BSIM3V1vsatGiven   :1;
    unsigned  BSIM3V1atGiven   :1;
    unsigned  BSIM3V1a0Given   :1;
    unsigned  BSIM3V1agsGiven   :1;
    unsigned  BSIM3V1a1Given   :1;
    unsigned  BSIM3V1a2Given   :1;
    unsigned  BSIM3V1ketaGiven   :1;    
    unsigned  BSIM3V1nsubGiven   :1;
    unsigned  BSIM3V1npeakGiven   :1;
    unsigned  BSIM3V1ngateGiven   :1;
    unsigned  BSIM3V1gamma1Given   :1;
    unsigned  BSIM3V1gamma2Given   :1;
    unsigned  BSIM3V1vbxGiven   :1;
    unsigned  BSIM3V1vbmGiven   :1;
    unsigned  BSIM3V1xtGiven   :1;
    unsigned  BSIM3V1k1Given   :1;
    unsigned  BSIM3V1kt1Given   :1;
    unsigned  BSIM3V1kt1lGiven   :1;
    unsigned  BSIM3V1kt2Given   :1;
    unsigned  BSIM3V1k2Given   :1;
    unsigned  BSIM3V1k3Given   :1;
    unsigned  BSIM3V1k3bGiven   :1;
    unsigned  BSIM3V1w0Given   :1;
    unsigned  BSIM3V1nlxGiven   :1;
    unsigned  BSIM3V1dvt0Given   :1;   
    unsigned  BSIM3V1dvt1Given   :1;     
    unsigned  BSIM3V1dvt2Given   :1;     
    unsigned  BSIM3V1dvt0wGiven   :1;   
    unsigned  BSIM3V1dvt1wGiven   :1;     
    unsigned  BSIM3V1dvt2wGiven   :1;     
    unsigned  BSIM3V1droutGiven   :1;     
    unsigned  BSIM3V1dsubGiven   :1;     
    unsigned  BSIM3V1vth0Given   :1;
    unsigned  BSIM3V1uaGiven   :1;
    unsigned  BSIM3V1ua1Given   :1;
    unsigned  BSIM3V1ubGiven   :1;
    unsigned  BSIM3V1ub1Given   :1;
    unsigned  BSIM3V1ucGiven   :1;
    unsigned  BSIM3V1uc1Given   :1;
    unsigned  BSIM3V1u0Given   :1;
    unsigned  BSIM3V1uteGiven   :1;
    unsigned  BSIM3V1voffGiven   :1;
    unsigned  BSIM3V1rdswGiven   :1;      
    unsigned  BSIM3V1prwgGiven   :1;      
    unsigned  BSIM3V1prwbGiven   :1;      
    unsigned  BSIM3V1prtGiven   :1;      
    unsigned  BSIM3V1eta0Given   :1;    
    unsigned  BSIM3V1etabGiven   :1;    
    unsigned  BSIM3V1pclmGiven   :1;   
    unsigned  BSIM3V1pdibl1Given   :1;   
    unsigned  BSIM3V1pdibl2Given   :1;  
    unsigned  BSIM3V1pdiblbGiven   :1;  
    unsigned  BSIM3V1pscbe1Given   :1;    
    unsigned  BSIM3V1pscbe2Given   :1;    
    unsigned  BSIM3V1pvagGiven   :1;    
    unsigned  BSIM3V1deltaGiven  :1;     
    unsigned  BSIM3V1wrGiven   :1;
    unsigned  BSIM3V1dwgGiven   :1;
    unsigned  BSIM3V1dwbGiven   :1;
    unsigned  BSIM3V1b0Given   :1;
    unsigned  BSIM3V1b1Given   :1;
    unsigned  BSIM3V1alpha0Given   :1;
    unsigned  BSIM3V1beta0Given   :1;
    unsigned  BSIM3V1hdifGiven   :1;

    /* CV model */
    unsigned  BSIM3V1elmGiven  :1;     
    unsigned  BSIM3V1cgslGiven   :1;
    unsigned  BSIM3V1cgdlGiven   :1;
    unsigned  BSIM3V1ckappaGiven   :1;
    unsigned  BSIM3V1cfGiven   :1;
    unsigned  BSIM3V1vfbcvGiven   :1;
    unsigned  BSIM3V1clcGiven   :1;
    unsigned  BSIM3V1cleGiven   :1;
    unsigned  BSIM3V1dwcGiven   :1;
    unsigned  BSIM3V1dlcGiven   :1;


    /* Length dependence */
    unsigned  BSIM3V1lcdscGiven   :1;
    unsigned  BSIM3V1lcdscbGiven   :1;
    unsigned  BSIM3V1lcdscdGiven   :1;
    unsigned  BSIM3V1lcitGiven   :1;
    unsigned  BSIM3V1lnfactorGiven   :1;
    unsigned  BSIM3V1lxjGiven   :1;
    unsigned  BSIM3V1lvsatGiven   :1;
    unsigned  BSIM3V1latGiven   :1;
    unsigned  BSIM3V1la0Given   :1;
    unsigned  BSIM3V1lagsGiven   :1;
    unsigned  BSIM3V1la1Given   :1;
    unsigned  BSIM3V1la2Given   :1;
    unsigned  BSIM3V1lketaGiven   :1;    
    unsigned  BSIM3V1lnsubGiven   :1;
    unsigned  BSIM3V1lnpeakGiven   :1;
    unsigned  BSIM3V1lngateGiven   :1;
    unsigned  BSIM3V1lgamma1Given   :1;
    unsigned  BSIM3V1lgamma2Given   :1;
    unsigned  BSIM3V1lvbxGiven   :1;
    unsigned  BSIM3V1lvbmGiven   :1;
    unsigned  BSIM3V1lxtGiven   :1;
    unsigned  BSIM3V1lk1Given   :1;
    unsigned  BSIM3V1lkt1Given   :1;
    unsigned  BSIM3V1lkt1lGiven   :1;
    unsigned  BSIM3V1lkt2Given   :1;
    unsigned  BSIM3V1lk2Given   :1;
    unsigned  BSIM3V1lk3Given   :1;
    unsigned  BSIM3V1lk3bGiven   :1;
    unsigned  BSIM3V1lw0Given   :1;
    unsigned  BSIM3V1lnlxGiven   :1;
    unsigned  BSIM3V1ldvt0Given   :1;   
    unsigned  BSIM3V1ldvt1Given   :1;     
    unsigned  BSIM3V1ldvt2Given   :1;     
    unsigned  BSIM3V1ldvt0wGiven   :1;   
    unsigned  BSIM3V1ldvt1wGiven   :1;     
    unsigned  BSIM3V1ldvt2wGiven   :1;     
    unsigned  BSIM3V1ldroutGiven   :1;     
    unsigned  BSIM3V1ldsubGiven   :1;     
    unsigned  BSIM3V1lvth0Given   :1;
    unsigned  BSIM3V1luaGiven   :1;
    unsigned  BSIM3V1lua1Given   :1;
    unsigned  BSIM3V1lubGiven   :1;
    unsigned  BSIM3V1lub1Given   :1;
    unsigned  BSIM3V1lucGiven   :1;
    unsigned  BSIM3V1luc1Given   :1;
    unsigned  BSIM3V1lu0Given   :1;
    unsigned  BSIM3V1luteGiven   :1;
    unsigned  BSIM3V1lvoffGiven   :1;
    unsigned  BSIM3V1lrdswGiven   :1;      
    unsigned  BSIM3V1lprwgGiven   :1;      
    unsigned  BSIM3V1lprwbGiven   :1;      
    unsigned  BSIM3V1lprtGiven   :1;      
    unsigned  BSIM3V1leta0Given   :1;    
    unsigned  BSIM3V1letabGiven   :1;    
    unsigned  BSIM3V1lpclmGiven   :1;   
    unsigned  BSIM3V1lpdibl1Given   :1;   
    unsigned  BSIM3V1lpdibl2Given   :1;  
    unsigned  BSIM3V1lpdiblbGiven   :1;  
    unsigned  BSIM3V1lpscbe1Given   :1;    
    unsigned  BSIM3V1lpscbe2Given   :1;    
    unsigned  BSIM3V1lpvagGiven   :1;    
    unsigned  BSIM3V1ldeltaGiven  :1;     
    unsigned  BSIM3V1lwrGiven   :1;
    unsigned  BSIM3V1ldwgGiven   :1;
    unsigned  BSIM3V1ldwbGiven   :1;
    unsigned  BSIM3V1lb0Given   :1;
    unsigned  BSIM3V1lb1Given   :1;
    unsigned  BSIM3V1lalpha0Given   :1;
    unsigned  BSIM3V1lbeta0Given   :1;

    /* CV model */
    unsigned  BSIM3V1lelmGiven  :1;     
    unsigned  BSIM3V1lcgslGiven   :1;
    unsigned  BSIM3V1lcgdlGiven   :1;
    unsigned  BSIM3V1lckappaGiven   :1;
    unsigned  BSIM3V1lcfGiven   :1;
    unsigned  BSIM3V1lclcGiven   :1;
    unsigned  BSIM3V1lcleGiven   :1;
    unsigned  BSIM3V1lvfbcvGiven   :1;

    /* Width dependence */
    unsigned  BSIM3V1wcdscGiven   :1;
    unsigned  BSIM3V1wcdscbGiven   :1;
    unsigned  BSIM3V1wcdscdGiven   :1;
    unsigned  BSIM3V1wcitGiven   :1;
    unsigned  BSIM3V1wnfactorGiven   :1;
    unsigned  BSIM3V1wxjGiven   :1;
    unsigned  BSIM3V1wvsatGiven   :1;
    unsigned  BSIM3V1watGiven   :1;
    unsigned  BSIM3V1wa0Given   :1;
    unsigned  BSIM3V1wagsGiven   :1;
    unsigned  BSIM3V1wa1Given   :1;
    unsigned  BSIM3V1wa2Given   :1;
    unsigned  BSIM3V1wketaGiven   :1;    
    unsigned  BSIM3V1wnsubGiven   :1;
    unsigned  BSIM3V1wnpeakGiven   :1;
    unsigned  BSIM3V1wngateGiven   :1;
    unsigned  BSIM3V1wgamma1Given   :1;
    unsigned  BSIM3V1wgamma2Given   :1;
    unsigned  BSIM3V1wvbxGiven   :1;
    unsigned  BSIM3V1wvbmGiven   :1;
    unsigned  BSIM3V1wxtGiven   :1;
    unsigned  BSIM3V1wk1Given   :1;
    unsigned  BSIM3V1wkt1Given   :1;
    unsigned  BSIM3V1wkt1lGiven   :1;
    unsigned  BSIM3V1wkt2Given   :1;
    unsigned  BSIM3V1wk2Given   :1;
    unsigned  BSIM3V1wk3Given   :1;
    unsigned  BSIM3V1wk3bGiven   :1;
    unsigned  BSIM3V1ww0Given   :1;
    unsigned  BSIM3V1wnlxGiven   :1;
    unsigned  BSIM3V1wdvt0Given   :1;   
    unsigned  BSIM3V1wdvt1Given   :1;     
    unsigned  BSIM3V1wdvt2Given   :1;     
    unsigned  BSIM3V1wdvt0wGiven   :1;   
    unsigned  BSIM3V1wdvt1wGiven   :1;     
    unsigned  BSIM3V1wdvt2wGiven   :1;     
    unsigned  BSIM3V1wdroutGiven   :1;     
    unsigned  BSIM3V1wdsubGiven   :1;     
    unsigned  BSIM3V1wvth0Given   :1;
    unsigned  BSIM3V1wuaGiven   :1;
    unsigned  BSIM3V1wua1Given   :1;
    unsigned  BSIM3V1wubGiven   :1;
    unsigned  BSIM3V1wub1Given   :1;
    unsigned  BSIM3V1wucGiven   :1;
    unsigned  BSIM3V1wuc1Given   :1;
    unsigned  BSIM3V1wu0Given   :1;
    unsigned  BSIM3V1wuteGiven   :1;
    unsigned  BSIM3V1wvoffGiven   :1;
    unsigned  BSIM3V1wrdswGiven   :1;      
    unsigned  BSIM3V1wprwgGiven   :1;      
    unsigned  BSIM3V1wprwbGiven   :1;      
    unsigned  BSIM3V1wprtGiven   :1;      
    unsigned  BSIM3V1weta0Given   :1;    
    unsigned  BSIM3V1wetabGiven   :1;    
    unsigned  BSIM3V1wpclmGiven   :1;   
    unsigned  BSIM3V1wpdibl1Given   :1;   
    unsigned  BSIM3V1wpdibl2Given   :1;  
    unsigned  BSIM3V1wpdiblbGiven   :1;  
    unsigned  BSIM3V1wpscbe1Given   :1;    
    unsigned  BSIM3V1wpscbe2Given   :1;    
    unsigned  BSIM3V1wpvagGiven   :1;    
    unsigned  BSIM3V1wdeltaGiven  :1;     
    unsigned  BSIM3V1wwrGiven   :1;
    unsigned  BSIM3V1wdwgGiven   :1;
    unsigned  BSIM3V1wdwbGiven   :1;
    unsigned  BSIM3V1wb0Given   :1;
    unsigned  BSIM3V1wb1Given   :1;
    unsigned  BSIM3V1walpha0Given   :1;
    unsigned  BSIM3V1wbeta0Given   :1;

    /* CV model */
    unsigned  BSIM3V1welmGiven  :1;     
    unsigned  BSIM3V1wcgslGiven   :1;
    unsigned  BSIM3V1wcgdlGiven   :1;
    unsigned  BSIM3V1wckappaGiven   :1;
    unsigned  BSIM3V1wcfGiven   :1;
    unsigned  BSIM3V1wclcGiven   :1;
    unsigned  BSIM3V1wcleGiven   :1;
    unsigned  BSIM3V1wvfbcvGiven   :1;

    /* Cross-term dependence */
    unsigned  BSIM3V1pcdscGiven   :1;
    unsigned  BSIM3V1pcdscbGiven   :1;
    unsigned  BSIM3V1pcdscdGiven   :1;
    unsigned  BSIM3V1pcitGiven   :1;
    unsigned  BSIM3V1pnfactorGiven   :1;
    unsigned  BSIM3V1pxjGiven   :1;
    unsigned  BSIM3V1pvsatGiven   :1;
    unsigned  BSIM3V1patGiven   :1;
    unsigned  BSIM3V1pa0Given   :1;
    unsigned  BSIM3V1pagsGiven   :1;
    unsigned  BSIM3V1pa1Given   :1;
    unsigned  BSIM3V1pa2Given   :1;
    unsigned  BSIM3V1pketaGiven   :1;    
    unsigned  BSIM3V1pnsubGiven   :1;
    unsigned  BSIM3V1pnpeakGiven   :1;
    unsigned  BSIM3V1pngateGiven   :1;
    unsigned  BSIM3V1pgamma1Given   :1;
    unsigned  BSIM3V1pgamma2Given   :1;
    unsigned  BSIM3V1pvbxGiven   :1;
    unsigned  BSIM3V1pvbmGiven   :1;
    unsigned  BSIM3V1pxtGiven   :1;
    unsigned  BSIM3V1pk1Given   :1;
    unsigned  BSIM3V1pkt1Given   :1;
    unsigned  BSIM3V1pkt1lGiven   :1;
    unsigned  BSIM3V1pkt2Given   :1;
    unsigned  BSIM3V1pk2Given   :1;
    unsigned  BSIM3V1pk3Given   :1;
    unsigned  BSIM3V1pk3bGiven   :1;
    unsigned  BSIM3V1pw0Given   :1;
    unsigned  BSIM3V1pnlxGiven   :1;
    unsigned  BSIM3V1pdvt0Given   :1;   
    unsigned  BSIM3V1pdvt1Given   :1;     
    unsigned  BSIM3V1pdvt2Given   :1;     
    unsigned  BSIM3V1pdvt0wGiven   :1;   
    unsigned  BSIM3V1pdvt1wGiven   :1;     
    unsigned  BSIM3V1pdvt2wGiven   :1;     
    unsigned  BSIM3V1pdroutGiven   :1;     
    unsigned  BSIM3V1pdsubGiven   :1;     
    unsigned  BSIM3V1pvth0Given   :1;
    unsigned  BSIM3V1puaGiven   :1;
    unsigned  BSIM3V1pua1Given   :1;
    unsigned  BSIM3V1pubGiven   :1;
    unsigned  BSIM3V1pub1Given   :1;
    unsigned  BSIM3V1pucGiven   :1;
    unsigned  BSIM3V1puc1Given   :1;
    unsigned  BSIM3V1pu0Given   :1;
    unsigned  BSIM3V1puteGiven   :1;
    unsigned  BSIM3V1pvoffGiven   :1;
    unsigned  BSIM3V1prdswGiven   :1;      
    unsigned  BSIM3V1pprwgGiven   :1;      
    unsigned  BSIM3V1pprwbGiven   :1;      
    unsigned  BSIM3V1pprtGiven   :1;      
    unsigned  BSIM3V1peta0Given   :1;    
    unsigned  BSIM3V1petabGiven   :1;    
    unsigned  BSIM3V1ppclmGiven   :1;   
    unsigned  BSIM3V1ppdibl1Given   :1;   
    unsigned  BSIM3V1ppdibl2Given   :1;  
    unsigned  BSIM3V1ppdiblbGiven   :1;  
    unsigned  BSIM3V1ppscbe1Given   :1;    
    unsigned  BSIM3V1ppscbe2Given   :1;    
    unsigned  BSIM3V1ppvagGiven   :1;    
    unsigned  BSIM3V1pdeltaGiven  :1;     
    unsigned  BSIM3V1pwrGiven   :1;
    unsigned  BSIM3V1pdwgGiven   :1;
    unsigned  BSIM3V1pdwbGiven   :1;
    unsigned  BSIM3V1pb0Given   :1;
    unsigned  BSIM3V1pb1Given   :1;
    unsigned  BSIM3V1palpha0Given   :1;
    unsigned  BSIM3V1pbeta0Given   :1;

    /* CV model */
    unsigned  BSIM3V1pelmGiven  :1;     
    unsigned  BSIM3V1pcgslGiven   :1;
    unsigned  BSIM3V1pcgdlGiven   :1;
    unsigned  BSIM3V1pckappaGiven   :1;
    unsigned  BSIM3V1pcfGiven   :1;
    unsigned  BSIM3V1pclcGiven   :1;
    unsigned  BSIM3V1pcleGiven   :1;
    unsigned  BSIM3V1pvfbcvGiven   :1;

    unsigned  BSIM3V1useFringeGiven   :1;

    unsigned  BSIM3V1tnomGiven   :1;
    unsigned  BSIM3V1cgsoGiven   :1;
    unsigned  BSIM3V1cgdoGiven   :1;
    unsigned  BSIM3V1cgboGiven   :1;
    unsigned  BSIM3V1xpartGiven   :1;
    unsigned  BSIM3V1sheetResistanceGiven   :1;
    unsigned  BSIM3V1jctSatCurDensityGiven   :1;
    unsigned  BSIM3V1jctSidewallSatCurDensityGiven   :1;
    unsigned  BSIM3V1bulkJctPotentialGiven   :1;
    unsigned  BSIM3V1bulkJctBotGradingCoeffGiven   :1;
    unsigned  BSIM3V1sidewallJctPotentialGiven   :1;
    unsigned  BSIM3V1GatesidewallJctPotentialGiven   :1;
    unsigned  BSIM3V1bulkJctSideGradingCoeffGiven   :1;
    unsigned  BSIM3V1unitAreaJctCapGiven   :1;
    unsigned  BSIM3V1unitLengthSidewallJctCapGiven   :1;
    unsigned  BSIM3V1bulkJctGateSideGradingCoeffGiven   :1;
    unsigned  BSIM3V1unitLengthGateSidewallJctCapGiven   :1;
    unsigned  BSIM3V1jctEmissionCoeffGiven :1;
    unsigned  BSIM3V1jctTempExponentGiven	:1;

    unsigned  BSIM3V1oxideTrapDensityAGiven  :1;         
    unsigned  BSIM3V1oxideTrapDensityBGiven  :1;        
    unsigned  BSIM3V1oxideTrapDensityCGiven  :1;     
    unsigned  BSIM3V1emGiven  :1;     
    unsigned  BSIM3V1efGiven  :1;     
    unsigned  BSIM3V1afGiven  :1;     
    unsigned  BSIM3V1kfGiven  :1;     

    unsigned  BSIM3V1LintGiven   :1;
    unsigned  BSIM3V1LlGiven   :1;
    unsigned  BSIM3V1LlnGiven   :1;
    unsigned  BSIM3V1LwGiven   :1;
    unsigned  BSIM3V1LwnGiven   :1;
    unsigned  BSIM3V1LwlGiven   :1;
    unsigned  BSIM3V1LminGiven   :1;
    unsigned  BSIM3V1LmaxGiven   :1;

    unsigned  BSIM3V1WintGiven   :1;
    unsigned  BSIM3V1WlGiven   :1;
    unsigned  BSIM3V1WlnGiven   :1;
    unsigned  BSIM3V1WwGiven   :1;
    unsigned  BSIM3V1WwnGiven   :1;
    unsigned  BSIM3V1WwlGiven   :1;
    unsigned  BSIM3V1WminGiven   :1;
    unsigned  BSIM3V1WmaxGiven   :1;

} BSIM3V1model;


#ifndef NMOS
#define NMOS 1
#define PMOS -1
#endif /*NMOS*/


/* device parameters */
#define BSIM3V1_W 1
#define BSIM3V1_L 2
#define BSIM3V1_AS 3
#define BSIM3V1_AD 4
#define BSIM3V1_PS 5
#define BSIM3V1_PD 6
#define BSIM3V1_NRS 7
#define BSIM3V1_NRD 8
#define BSIM3V1_OFF 9
#define BSIM3V1_IC_VBS 10
#define BSIM3V1_IC_VDS 11
#define BSIM3V1_IC_VGS 12
#define BSIM3V1_IC 13
#define BSIM3V1_NQSMOD 14
#define BSIM3V1_M 20

/* model parameters */
#define BSIM3V1_MOD_CAPMOD          101
#define BSIM3V1_MOD_NQSMOD          102
#define BSIM3V1_MOD_MOBMOD          103    
#define BSIM3V1_MOD_NOIMOD          104    

#define BSIM3V1_MOD_TOX             105

#define BSIM3V1_MOD_CDSC            106
#define BSIM3V1_MOD_CDSCB           107
#define BSIM3V1_MOD_CIT             108
#define BSIM3V1_MOD_NFACTOR         109
#define BSIM3V1_MOD_XJ              110
#define BSIM3V1_MOD_VSAT            111
#define BSIM3V1_MOD_AT              112
#define BSIM3V1_MOD_A0              113
#define BSIM3V1_MOD_A1              114
#define BSIM3V1_MOD_A2              115
#define BSIM3V1_MOD_KETA            116   
#define BSIM3V1_MOD_NSUB            117
#define BSIM3V1_MOD_NPEAK           118
#define BSIM3V1_MOD_NGATE           120
#define BSIM3V1_MOD_GAMMA1          121
#define BSIM3V1_MOD_GAMMA2          122
#define BSIM3V1_MOD_VBX             123
#define BSIM3V1_MOD_BINUNIT         124    

#define BSIM3V1_MOD_VBM             125

#define BSIM3V1_MOD_XT              126
#define BSIM3V1_MOD_K1              129
#define BSIM3V1_MOD_KT1             130
#define BSIM3V1_MOD_KT1L            131
#define BSIM3V1_MOD_K2              132
#define BSIM3V1_MOD_KT2             133
#define BSIM3V1_MOD_K3              134
#define BSIM3V1_MOD_K3B             135
#define BSIM3V1_MOD_W0              136
#define BSIM3V1_MOD_NLX             137

#define BSIM3V1_MOD_DVT0            138
#define BSIM3V1_MOD_DVT1            139
#define BSIM3V1_MOD_DVT2            140

#define BSIM3V1_MOD_DVT0W           141
#define BSIM3V1_MOD_DVT1W           142
#define BSIM3V1_MOD_DVT2W           143

#define BSIM3V1_MOD_DROUT           144
#define BSIM3V1_MOD_DSUB            145
#define BSIM3V1_MOD_VTH0            146
#define BSIM3V1_MOD_UA              147
#define BSIM3V1_MOD_UA1             148
#define BSIM3V1_MOD_UB              149
#define BSIM3V1_MOD_UB1             150
#define BSIM3V1_MOD_UC              151
#define BSIM3V1_MOD_UC1             152
#define BSIM3V1_MOD_U0              153
#define BSIM3V1_MOD_UTE             154
#define BSIM3V1_MOD_VOFF            155
#define BSIM3V1_MOD_DELTA           156
#define BSIM3V1_MOD_RDSW            157
#define BSIM3V1_MOD_PRT             158
#define BSIM3V1_MOD_LDD             159
#define BSIM3V1_MOD_ETA             160
#define BSIM3V1_MOD_ETA0            161
#define BSIM3V1_MOD_ETAB            162
#define BSIM3V1_MOD_PCLM            163
#define BSIM3V1_MOD_PDIBL1          164
#define BSIM3V1_MOD_PDIBL2          165
#define BSIM3V1_MOD_PSCBE1          166
#define BSIM3V1_MOD_PSCBE2          167
#define BSIM3V1_MOD_PVAG            168
#define BSIM3V1_MOD_WR              169
#define BSIM3V1_MOD_DWG             170
#define BSIM3V1_MOD_DWB             171
#define BSIM3V1_MOD_B0              172
#define BSIM3V1_MOD_B1              173
#define BSIM3V1_MOD_ALPHA0          174
#define BSIM3V1_MOD_BETA0           175
#define BSIM3V1_MOD_PDIBLB          178

#define BSIM3V1_MOD_PRWG            179
#define BSIM3V1_MOD_PRWB            180

#define BSIM3V1_MOD_CDSCD           181
#define BSIM3V1_MOD_AGS             182

#define BSIM3V1_MOD_FRINGE          184
#define BSIM3V1_MOD_ELM             185
#define BSIM3V1_MOD_CGSL            186
#define BSIM3V1_MOD_CGDL            187
#define BSIM3V1_MOD_CKAPPA          188
#define BSIM3V1_MOD_CF              189
#define BSIM3V1_MOD_CLC             190
#define BSIM3V1_MOD_CLE             191
#define BSIM3V1_MOD_PARAMCHK        192
#define BSIM3V1_MOD_VERSION         193
#define BSIM3V1_MOD_VFBCV           194

#define BSIM3V1_MOD_HDIF            198

/* Length dependence */
#define BSIM3V1_MOD_LCDSC            201
#define BSIM3V1_MOD_LCDSCB           202
#define BSIM3V1_MOD_LCIT             203
#define BSIM3V1_MOD_LNFACTOR         204
#define BSIM3V1_MOD_LXJ              205
#define BSIM3V1_MOD_LVSAT            206
#define BSIM3V1_MOD_LAT              207
#define BSIM3V1_MOD_LA0              208
#define BSIM3V1_MOD_LA1              209
#define BSIM3V1_MOD_LA2              210
#define BSIM3V1_MOD_LKETA            211   
#define BSIM3V1_MOD_LNSUB            212
#define BSIM3V1_MOD_LNPEAK           213
#define BSIM3V1_MOD_LNGATE           215
#define BSIM3V1_MOD_LGAMMA1          216
#define BSIM3V1_MOD_LGAMMA2          217
#define BSIM3V1_MOD_LVBX             218

#define BSIM3V1_MOD_LVBM             220

#define BSIM3V1_MOD_LXT              222
#define BSIM3V1_MOD_LK1              225
#define BSIM3V1_MOD_LKT1             226
#define BSIM3V1_MOD_LKT1L            227
#define BSIM3V1_MOD_LK2              228
#define BSIM3V1_MOD_LKT2             229
#define BSIM3V1_MOD_LK3              230
#define BSIM3V1_MOD_LK3B             231
#define BSIM3V1_MOD_LW0              232
#define BSIM3V1_MOD_LNLX             233

#define BSIM3V1_MOD_LDVT0            234
#define BSIM3V1_MOD_LDVT1            235
#define BSIM3V1_MOD_LDVT2            236

#define BSIM3V1_MOD_LDVT0W           237
#define BSIM3V1_MOD_LDVT1W           238
#define BSIM3V1_MOD_LDVT2W           239

#define BSIM3V1_MOD_LDROUT           240
#define BSIM3V1_MOD_LDSUB            241
#define BSIM3V1_MOD_LVTH0            242
#define BSIM3V1_MOD_LUA              243
#define BSIM3V1_MOD_LUA1             244
#define BSIM3V1_MOD_LUB              245
#define BSIM3V1_MOD_LUB1             246
#define BSIM3V1_MOD_LUC              247
#define BSIM3V1_MOD_LUC1             248
#define BSIM3V1_MOD_LU0              249
#define BSIM3V1_MOD_LUTE             250
#define BSIM3V1_MOD_LVOFF            251
#define BSIM3V1_MOD_LDELTA           252
#define BSIM3V1_MOD_LRDSW            253
#define BSIM3V1_MOD_LPRT             254
#define BSIM3V1_MOD_LLDD             255
#define BSIM3V1_MOD_LETA             256
#define BSIM3V1_MOD_LETA0            257
#define BSIM3V1_MOD_LETAB            258
#define BSIM3V1_MOD_LPCLM            259
#define BSIM3V1_MOD_LPDIBL1          260
#define BSIM3V1_MOD_LPDIBL2          261
#define BSIM3V1_MOD_LPSCBE1          262
#define BSIM3V1_MOD_LPSCBE2          263
#define BSIM3V1_MOD_LPVAG            264
#define BSIM3V1_MOD_LWR              265
#define BSIM3V1_MOD_LDWG             266
#define BSIM3V1_MOD_LDWB             267
#define BSIM3V1_MOD_LB0              268
#define BSIM3V1_MOD_LB1              269
#define BSIM3V1_MOD_LALPHA0          270
#define BSIM3V1_MOD_LBETA0           271
#define BSIM3V1_MOD_LPDIBLB          274

#define BSIM3V1_MOD_LPRWG            275
#define BSIM3V1_MOD_LPRWB            276

#define BSIM3V1_MOD_LCDSCD           277
#define BSIM3V1_MOD_LAGS            278
                                    

#define BSIM3V1_MOD_LFRINGE          281
#define BSIM3V1_MOD_LELM             282
#define BSIM3V1_MOD_LCGSL            283
#define BSIM3V1_MOD_LCGDL            284
#define BSIM3V1_MOD_LCKAPPA          285
#define BSIM3V1_MOD_LCF              286
#define BSIM3V1_MOD_LCLC             287
#define BSIM3V1_MOD_LCLE             288
#define BSIM3V1_MOD_LVFBCV             289

/* Width dependence */
#define BSIM3V1_MOD_WCDSC            301
#define BSIM3V1_MOD_WCDSCB           302
#define BSIM3V1_MOD_WCIT             303
#define BSIM3V1_MOD_WNFACTOR         304
#define BSIM3V1_MOD_WXJ              305
#define BSIM3V1_MOD_WVSAT            306
#define BSIM3V1_MOD_WAT              307
#define BSIM3V1_MOD_WA0              308
#define BSIM3V1_MOD_WA1              309
#define BSIM3V1_MOD_WA2              310
#define BSIM3V1_MOD_WKETA            311   
#define BSIM3V1_MOD_WNSUB            312
#define BSIM3V1_MOD_WNPEAK           313
#define BSIM3V1_MOD_WNGATE           315
#define BSIM3V1_MOD_WGAMMA1          316
#define BSIM3V1_MOD_WGAMMA2          317
#define BSIM3V1_MOD_WVBX             318

#define BSIM3V1_MOD_WVBM             320

#define BSIM3V1_MOD_WXT              322
#define BSIM3V1_MOD_WK1              325
#define BSIM3V1_MOD_WKT1             326
#define BSIM3V1_MOD_WKT1L            327
#define BSIM3V1_MOD_WK2              328
#define BSIM3V1_MOD_WKT2             329
#define BSIM3V1_MOD_WK3              330
#define BSIM3V1_MOD_WK3B             331
#define BSIM3V1_MOD_WW0              332
#define BSIM3V1_MOD_WNLX             333

#define BSIM3V1_MOD_WDVT0            334
#define BSIM3V1_MOD_WDVT1            335
#define BSIM3V1_MOD_WDVT2            336

#define BSIM3V1_MOD_WDVT0W           337
#define BSIM3V1_MOD_WDVT1W           338
#define BSIM3V1_MOD_WDVT2W           339

#define BSIM3V1_MOD_WDROUT           340
#define BSIM3V1_MOD_WDSUB            341
#define BSIM3V1_MOD_WVTH0            342
#define BSIM3V1_MOD_WUA              343
#define BSIM3V1_MOD_WUA1             344
#define BSIM3V1_MOD_WUB              345
#define BSIM3V1_MOD_WUB1             346
#define BSIM3V1_MOD_WUC              347
#define BSIM3V1_MOD_WUC1             348
#define BSIM3V1_MOD_WU0              349
#define BSIM3V1_MOD_WUTE             350
#define BSIM3V1_MOD_WVOFF            351
#define BSIM3V1_MOD_WDELTA           352
#define BSIM3V1_MOD_WRDSW            353
#define BSIM3V1_MOD_WPRT             354
#define BSIM3V1_MOD_WLDD             355
#define BSIM3V1_MOD_WETA             356
#define BSIM3V1_MOD_WETA0            357
#define BSIM3V1_MOD_WETAB            358
#define BSIM3V1_MOD_WPCLM            359
#define BSIM3V1_MOD_WPDIBL1          360
#define BSIM3V1_MOD_WPDIBL2          361
#define BSIM3V1_MOD_WPSCBE1          362
#define BSIM3V1_MOD_WPSCBE2          363
#define BSIM3V1_MOD_WPVAG            364
#define BSIM3V1_MOD_WWR              365
#define BSIM3V1_MOD_WDWG             366
#define BSIM3V1_MOD_WDWB             367
#define BSIM3V1_MOD_WB0              368
#define BSIM3V1_MOD_WB1              369
#define BSIM3V1_MOD_WALPHA0          370
#define BSIM3V1_MOD_WBETA0           371
#define BSIM3V1_MOD_WPDIBLB          374

#define BSIM3V1_MOD_WPRWG            375
#define BSIM3V1_MOD_WPRWB            376

#define BSIM3V1_MOD_WCDSCD           377
#define BSIM3V1_MOD_WAGS             378


#define BSIM3V1_MOD_WFRINGE          381
#define BSIM3V1_MOD_WELM             382
#define BSIM3V1_MOD_WCGSL            383
#define BSIM3V1_MOD_WCGDL            384
#define BSIM3V1_MOD_WCKAPPA          385
#define BSIM3V1_MOD_WCF              386
#define BSIM3V1_MOD_WCLC             387
#define BSIM3V1_MOD_WCLE             388
#define BSIM3V1_MOD_WVFBCV             389

/* Cross-term dependence */
#define BSIM3V1_MOD_PCDSC            401
#define BSIM3V1_MOD_PCDSCB           402
#define BSIM3V1_MOD_PCIT             403
#define BSIM3V1_MOD_PNFACTOR         404
#define BSIM3V1_MOD_PXJ              405
#define BSIM3V1_MOD_PVSAT            406
#define BSIM3V1_MOD_PAT              407
#define BSIM3V1_MOD_PA0              408
#define BSIM3V1_MOD_PA1              409
#define BSIM3V1_MOD_PA2              410
#define BSIM3V1_MOD_PKETA            411   
#define BSIM3V1_MOD_PNSUB            412
#define BSIM3V1_MOD_PNPEAK           413
#define BSIM3V1_MOD_PNGATE           415
#define BSIM3V1_MOD_PGAMMA1          416
#define BSIM3V1_MOD_PGAMMA2          417
#define BSIM3V1_MOD_PVBX             418

#define BSIM3V1_MOD_PVBM             420

#define BSIM3V1_MOD_PXT              422
#define BSIM3V1_MOD_PK1              425
#define BSIM3V1_MOD_PKT1             426
#define BSIM3V1_MOD_PKT1L            427
#define BSIM3V1_MOD_PK2              428
#define BSIM3V1_MOD_PKT2             429
#define BSIM3V1_MOD_PK3              430
#define BSIM3V1_MOD_PK3B             431
#define BSIM3V1_MOD_PW0              432
#define BSIM3V1_MOD_PNLX             433

#define BSIM3V1_MOD_PDVT0            434
#define BSIM3V1_MOD_PDVT1            435
#define BSIM3V1_MOD_PDVT2            436

#define BSIM3V1_MOD_PDVT0W           437
#define BSIM3V1_MOD_PDVT1W           438
#define BSIM3V1_MOD_PDVT2W           439

#define BSIM3V1_MOD_PDROUT           440
#define BSIM3V1_MOD_PDSUB            441
#define BSIM3V1_MOD_PVTH0            442
#define BSIM3V1_MOD_PUA              443
#define BSIM3V1_MOD_PUA1             444
#define BSIM3V1_MOD_PUB              445
#define BSIM3V1_MOD_PUB1             446
#define BSIM3V1_MOD_PUC              447
#define BSIM3V1_MOD_PUC1             448
#define BSIM3V1_MOD_PU0              449
#define BSIM3V1_MOD_PUTE             450
#define BSIM3V1_MOD_PVOFF            451
#define BSIM3V1_MOD_PDELTA           452
#define BSIM3V1_MOD_PRDSW            453
#define BSIM3V1_MOD_PPRT             454
#define BSIM3V1_MOD_PLDD             455
#define BSIM3V1_MOD_PETA             456
#define BSIM3V1_MOD_PETA0            457
#define BSIM3V1_MOD_PETAB            458
#define BSIM3V1_MOD_PPCLM            459
#define BSIM3V1_MOD_PPDIBL1          460
#define BSIM3V1_MOD_PPDIBL2          461
#define BSIM3V1_MOD_PPSCBE1          462
#define BSIM3V1_MOD_PPSCBE2          463
#define BSIM3V1_MOD_PPVAG            464
#define BSIM3V1_MOD_PWR              465
#define BSIM3V1_MOD_PDWG             466
#define BSIM3V1_MOD_PDWB             467
#define BSIM3V1_MOD_PB0              468
#define BSIM3V1_MOD_PB1              469
#define BSIM3V1_MOD_PALPHA0          470
#define BSIM3V1_MOD_PBETA0           471
#define BSIM3V1_MOD_PPDIBLB          474

#define BSIM3V1_MOD_PPRWG            475
#define BSIM3V1_MOD_PPRWB            476

#define BSIM3V1_MOD_PCDSCD           477
#define BSIM3V1_MOD_PAGS             478

#define BSIM3V1_MOD_PFRINGE          481
#define BSIM3V1_MOD_PELM             482
#define BSIM3V1_MOD_PCGSL            483
#define BSIM3V1_MOD_PCGDL            484
#define BSIM3V1_MOD_PCKAPPA          485
#define BSIM3V1_MOD_PCF              486
#define BSIM3V1_MOD_PCLC             487
#define BSIM3V1_MOD_PCLE             488
#define BSIM3V1_MOD_PVFBCV           489

#define BSIM3V1_MOD_TNOM             501
#define BSIM3V1_MOD_CGSO             502
#define BSIM3V1_MOD_CGDO             503
#define BSIM3V1_MOD_CGBO             504
#define BSIM3V1_MOD_XPART            505

#define BSIM3V1_MOD_RSH              506
#define BSIM3V1_MOD_JS               507
#define BSIM3V1_MOD_PB               508
#define BSIM3V1_MOD_MJ               509
#define BSIM3V1_MOD_PBSW             510
#define BSIM3V1_MOD_MJSW             511
#define BSIM3V1_MOD_CJ               512
#define BSIM3V1_MOD_CJSW             513
#define BSIM3V1_MOD_NMOS             514
#define BSIM3V1_MOD_PMOS             515

#define BSIM3V1_MOD_NOIA             516
#define BSIM3V1_MOD_NOIB             517
#define BSIM3V1_MOD_NOIC             518

#define BSIM3V1_MOD_LINT             519
#define BSIM3V1_MOD_LL               520
#define BSIM3V1_MOD_LLN              521
#define BSIM3V1_MOD_LW               522
#define BSIM3V1_MOD_LWN              523
#define BSIM3V1_MOD_LWL              524
#define BSIM3V1_MOD_LMIN             525
#define BSIM3V1_MOD_LMAX             526

#define BSIM3V1_MOD_WINT             527
#define BSIM3V1_MOD_WL               528
#define BSIM3V1_MOD_WLN              529
#define BSIM3V1_MOD_WW               530
#define BSIM3V1_MOD_WWN              531
#define BSIM3V1_MOD_WWL              532
#define BSIM3V1_MOD_WMIN             533
#define BSIM3V1_MOD_WMAX             534

#define BSIM3V1_MOD_DWC              535
#define BSIM3V1_MOD_DLC              536

#define BSIM3V1_MOD_EM               537
#define BSIM3V1_MOD_EF               538
#define BSIM3V1_MOD_AF               539
#define BSIM3V1_MOD_KF               540

#define BSIM3V1_MOD_NJ               541
#define BSIM3V1_MOD_XTI              542

#define BSIM3V1_MOD_PBSWG            543
#define BSIM3V1_MOD_MJSWG            544
#define BSIM3V1_MOD_CJSWG            545
#define BSIM3V1_MOD_JSW              546

/* device questions */
#define BSIM3V1_DNODE                601
#define BSIM3V1_GNODE                602
#define BSIM3V1_SNODE                603
#define BSIM3V1_BNODE                604
#define BSIM3V1_DNODEPRIME           605
#define BSIM3V1_SNODEPRIME           606
#define BSIM3V1_VBD                  607
#define BSIM3V1_VBS                  608
#define BSIM3V1_VGS                  609
#define BSIM3V1_VDS                  610
#define BSIM3V1_CD                   611
#define BSIM3V1_CBS                  612
#define BSIM3V1_CBD                  613
#define BSIM3V1_GM                   614
#define BSIM3V1_GDS                  615
#define BSIM3V1_GMBS                 616
#define BSIM3V1_GBD                  617
#define BSIM3V1_GBS                  618
#define BSIM3V1_QB                   619
#define BSIM3V1_CQB                  620
#define BSIM3V1_QG                   621
#define BSIM3V1_CQG                  622
#define BSIM3V1_QD                   623
#define BSIM3V1_CQD                  624
#define BSIM3V1_CGG                  625
#define BSIM3V1_CGD                  626
#define BSIM3V1_CGS                  627
#define BSIM3V1_CBG                  628
#define BSIM3V1_CAPBD                629
#define BSIM3V1_CQBD                 630
#define BSIM3V1_CAPBS                631
#define BSIM3V1_CQBS                 632
#define BSIM3V1_CDG                  633
#define BSIM3V1_CDD                  634
#define BSIM3V1_CDS                  635
#define BSIM3V1_VON                  636
#define BSIM3V1_VDSAT                637
#define BSIM3V1_QBS                  638
#define BSIM3V1_QBD                  639
#define BSIM3V1_SOURCECONDUCT        640
#define BSIM3V1_DRAINCONDUCT         641
#define BSIM3V1_CBDB                 642
#define BSIM3V1_CBSB                 643

#include "bsim3v1ext.h"

#ifdef __STDC__
extern void BSIM3V1evaluate(double,double,double,BSIM3V1instance*,BSIM3V1model*,
        double*,double*,double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, CKTcircuit*);
extern int BSIM3V1debug(BSIM3V1model*, BSIM3V1instance*, CKTcircuit*, int);
extern int BSIM3V1checkModel(BSIM3V1model*, BSIM3V1instance*, CKTcircuit*);
#else /* stdc */
extern void BSIM3V1evaluate();
extern int BSIM3V1debug();
extern int BSIM3V1checkModel();
#endif /* stdc */

#endif /*BSIM3V1*/




