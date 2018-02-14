/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified: 2002 Paolo Nenzi.
File: bsim3v1def.h
**********/

#ifndef BSIM3v1
#define BSIM3v1

#include "ngspice/ifsim.h"
#include "ngspice/gendefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/noisedef.h"         

typedef struct sBSIM3v1instance
{

    struct GENinstance gen;

#define BSIM3v1modPtr(inst) ((struct sBSIM3v1model *)((inst)->gen.GENmodPtr))
#define BSIM3v1nextInstance(inst) ((struct sBSIM3v1instance *)((inst)->gen.GENnextInstance))
#define BSIM3v1name gen.GENname
#define BSIM3v1states gen.GENstate

    const int BSIM3v1dNode;
    const int BSIM3v1gNode;
    const int BSIM3v1sNode;
    const int BSIM3v1bNode;
    int BSIM3v1dNodePrime;
    int BSIM3v1sNodePrime;
    int BSIM3v1qNode; /* MCJ */

    /* MCJ */
    double BSIM3v1ueff;
    double BSIM3v1thetavth; 
    double BSIM3v1von;
    double BSIM3v1vdsat;
    double BSIM3v1cgdo;
    double BSIM3v1cgso;

    double BSIM3v1l;
    double BSIM3v1w;
    double BSIM3v1m;
    double BSIM3v1drainArea;
    double BSIM3v1sourceArea;
    double BSIM3v1drainSquares;
    double BSIM3v1sourceSquares;
    double BSIM3v1drainPerimeter;
    double BSIM3v1sourcePerimeter;
    double BSIM3v1sourceConductance;
    double BSIM3v1drainConductance;

    double BSIM3v1icVBS;
    double BSIM3v1icVDS;
    double BSIM3v1icVGS;
    int BSIM3v1off;
    int BSIM3v1mode;
    int BSIM3v1nqsMod;

    /* OP point */
    double BSIM3v1qinv;
    double BSIM3v1cd;
    double BSIM3v1cbs;
    double BSIM3v1cbd;
    double BSIM3v1csub;
    double BSIM3v1gm;
    double BSIM3v1gds;
    double BSIM3v1gmbs;
    double BSIM3v1gbd;
    double BSIM3v1gbs;

    double BSIM3v1gbbs;
    double BSIM3v1gbgs;
    double BSIM3v1gbds;

    double BSIM3v1cggb;
    double BSIM3v1cgdb;
    double BSIM3v1cgsb;
    double BSIM3v1cbgb;
    double BSIM3v1cbdb;
    double BSIM3v1cbsb;
    double BSIM3v1cdgb;
    double BSIM3v1cddb;
    double BSIM3v1cdsb;
    double BSIM3v1capbd;
    double BSIM3v1capbs;

    double BSIM3v1cqgb;
    double BSIM3v1cqdb;
    double BSIM3v1cqsb;
    double BSIM3v1cqbb;

    double BSIM3v1gtau;
    double BSIM3v1gtg;
    double BSIM3v1gtd;
    double BSIM3v1gts;
    double BSIM3v1gtb;
    double BSIM3v1tconst;

    struct bsim3v1SizeDependParam  *pParam;

    unsigned BSIM3v1lGiven :1;
    unsigned BSIM3v1wGiven :1;
    unsigned BSIM3v1mGiven :1;
    unsigned BSIM3v1drainAreaGiven :1;
    unsigned BSIM3v1sourceAreaGiven    :1;
    unsigned BSIM3v1drainSquaresGiven  :1;
    unsigned BSIM3v1sourceSquaresGiven :1;
    unsigned BSIM3v1drainPerimeterGiven    :1;
    unsigned BSIM3v1sourcePerimeterGiven   :1;
    unsigned BSIM3v1dNodePrimeSet  :1;
    unsigned BSIM3v1sNodePrimeSet  :1;
    unsigned BSIM3v1icVBSGiven :1;
    unsigned BSIM3v1icVDSGiven :1;
    unsigned BSIM3v1icVGSGiven :1;
    unsigned BSIM3v1nqsModGiven :1;

    double *BSIM3v1DdPtr;
    double *BSIM3v1GgPtr;
    double *BSIM3v1SsPtr;
    double *BSIM3v1BbPtr;
    double *BSIM3v1DPdpPtr;
    double *BSIM3v1SPspPtr;
    double *BSIM3v1DdpPtr;
    double *BSIM3v1GbPtr;
    double *BSIM3v1GdpPtr;
    double *BSIM3v1GspPtr;
    double *BSIM3v1SspPtr;
    double *BSIM3v1BdpPtr;
    double *BSIM3v1BspPtr;
    double *BSIM3v1DPspPtr;
    double *BSIM3v1DPdPtr;
    double *BSIM3v1BgPtr;
    double *BSIM3v1DPgPtr;
    double *BSIM3v1SPgPtr;
    double *BSIM3v1SPsPtr;
    double *BSIM3v1DPbPtr;
    double *BSIM3v1SPbPtr;
    double *BSIM3v1SPdpPtr;

    double *BSIM3v1QqPtr;
    double *BSIM3v1QdpPtr;
    double *BSIM3v1QgPtr;
    double *BSIM3v1QspPtr;
    double *BSIM3v1QbPtr;
    double *BSIM3v1DPqPtr;
    double *BSIM3v1GqPtr;
    double *BSIM3v1SPqPtr;
    double *BSIM3v1BqPtr;

#define BSIM3v1vbd BSIM3v1states+ 0
#define BSIM3v1vbs BSIM3v1states+ 1
#define BSIM3v1vgs BSIM3v1states+ 2
#define BSIM3v1vds BSIM3v1states+ 3

#define BSIM3v1qb BSIM3v1states+ 4
#define BSIM3v1cqb BSIM3v1states+ 5
#define BSIM3v1qg BSIM3v1states+ 6
#define BSIM3v1cqg BSIM3v1states+ 7
#define BSIM3v1qd BSIM3v1states+ 8
#define BSIM3v1cqd BSIM3v1states+ 9

#define BSIM3v1qbs  BSIM3v1states+ 10
#define BSIM3v1qbd  BSIM3v1states+ 11

#define BSIM3v1qcheq BSIM3v1states+ 12
#define BSIM3v1cqcheq BSIM3v1states+ 13
#define BSIM3v1qcdump BSIM3v1states+ 14
#define BSIM3v1cqcdump BSIM3v1states+ 15

#define BSIM3v1tau BSIM3v1states+ 16
#define BSIM3v1qdef BSIM3v1states+ 17

#define BSIM3v1numStates 18


/* indices to the array of BSIM3v1 NOISE SOURCES */

#define BSIM3v1RDNOIZ       0
#define BSIM3v1RSNOIZ       1
#define BSIM3v1IDNOIZ       2
#define BSIM3v1FLNOIZ       3
#define BSIM3v1TOTNOIZ      4

#define BSIM3v1NSRCS        5     /* the number of MOSFET(3) noise sources */

#ifndef NONOISE
    double BSIM3v1nVar[NSTATVARS][BSIM3v1NSRCS];
#else /* NONOISE */
        double **BSIM3v1nVar;
#endif /* NONOISE */

} BSIM3v1instance ;

struct bsim3v1SizeDependParam
{
    double Width;
    double Length;

    double BSIM3v1cdsc;           
    double BSIM3v1cdscb;    
    double BSIM3v1cdscd;       
    double BSIM3v1cit;           
    double BSIM3v1nfactor;      
    double BSIM3v1xj;
    double BSIM3v1vsat;         
    double BSIM3v1at;         
    double BSIM3v1a0;   
    double BSIM3v1ags;      
    double BSIM3v1a1;         
    double BSIM3v1a2;         
    double BSIM3v1keta;     
    double BSIM3v1nsub;
    double BSIM3v1npeak;        
    double BSIM3v1ngate;        
    double BSIM3v1gamma1;      
    double BSIM3v1gamma2;     
    double BSIM3v1vbx;      
    double BSIM3v1vbi;       
    double BSIM3v1vbm;       
    double BSIM3v1vbsc;       
    double BSIM3v1xt;       
    double BSIM3v1phi;
    double BSIM3v1litl;
    double BSIM3v1k1;
    double BSIM3v1kt1;
    double BSIM3v1kt1l;
    double BSIM3v1kt2;
    double BSIM3v1k2;
    double BSIM3v1k3;
    double BSIM3v1k3b;
    double BSIM3v1w0;
    double BSIM3v1nlx;
    double BSIM3v1dvt0;      
    double BSIM3v1dvt1;      
    double BSIM3v1dvt2;      
    double BSIM3v1dvt0w;      
    double BSIM3v1dvt1w;      
    double BSIM3v1dvt2w;      
    double BSIM3v1drout;      
    double BSIM3v1dsub;      
    double BSIM3v1vth0;
    double BSIM3v1ua;
    double BSIM3v1ua1;
    double BSIM3v1ub;
    double BSIM3v1ub1;
    double BSIM3v1uc;
    double BSIM3v1uc1;
    double BSIM3v1u0;
    double BSIM3v1ute;
    double BSIM3v1voff;
    double BSIM3v1vfb;
    double BSIM3v1delta;
    double BSIM3v1rdsw;       
    double BSIM3v1rds0;       
    double BSIM3v1prwg;       
    double BSIM3v1prwb;       
    double BSIM3v1prt;       
    double BSIM3v1eta0;         
    double BSIM3v1etab;         
    double BSIM3v1pclm;      
    double BSIM3v1pdibl1;      
    double BSIM3v1pdibl2;      
    double BSIM3v1pdiblb;      
    double BSIM3v1pscbe1;       
    double BSIM3v1pscbe2;       
    double BSIM3v1pvag;       
    double BSIM3v1wr;
    double BSIM3v1dwg;
    double BSIM3v1dwb;
    double BSIM3v1b0;
    double BSIM3v1b1;
    double BSIM3v1alpha0;
    double BSIM3v1beta0;


    /* CV model */
    double BSIM3v1elm;
    double BSIM3v1cgsl;
    double BSIM3v1cgdl;
    double BSIM3v1ckappa;
    double BSIM3v1cf;
    double BSIM3v1clc;
    double BSIM3v1cle;
    double BSIM3v1vfbcv;


/* Pre-calculated constants */

    double BSIM3v1dw;
    double BSIM3v1dl;
    double BSIM3v1leff;
    double BSIM3v1weff;

    double BSIM3v1dwc;
    double BSIM3v1dlc;
    double BSIM3v1leffCV;
    double BSIM3v1weffCV;
    double BSIM3v1abulkCVfactor;
    double BSIM3v1cgso;
    double BSIM3v1cgdo;
    double BSIM3v1cgbo;

    double BSIM3v1u0temp;       
    double BSIM3v1vsattemp;   
    double BSIM3v1sqrtPhi;   
    double BSIM3v1phis3;   
    double BSIM3v1Xdep0;          
    double BSIM3v1sqrtXdep0;          
    double BSIM3v1theta0vb0;
    double BSIM3v1thetaRout; 

    double BSIM3v1cof1;
    double BSIM3v1cof2;
    double BSIM3v1cof3;
    double BSIM3v1cof4;
    double BSIM3v1cdep0;
    struct bsim3v1SizeDependParam  *pNext;
};


typedef struct sBSIM3v1model 
{

    struct GENmodel gen;

#define BSIM3v1modType gen.GENmodType
#define BSIM3v1nextModel(inst) ((struct sBSIM3v1model *)((inst)->gen.GENnextModel))
#define BSIM3v1instances(inst) ((BSIM3v1instance *)((inst)->gen.GENinstances))
#define BSIM3v1modName gen.GENmodName

    int BSIM3v1type;

    int    BSIM3v1mobMod;
    int    BSIM3v1capMod;
    int    BSIM3v1nqsMod;
    int    BSIM3v1noiMod;
    int    BSIM3v1binUnit;
    int    BSIM3v1paramChk;
    double BSIM3v1version;             
    double BSIM3v1tox;             
    double BSIM3v1cdsc;           
    double BSIM3v1cdscb; 
    double BSIM3v1cdscd;          
    double BSIM3v1cit;           
    double BSIM3v1nfactor;      
    double BSIM3v1xj;
    double BSIM3v1vsat;         
    double BSIM3v1at;         
    double BSIM3v1a0;   
    double BSIM3v1ags;      
    double BSIM3v1a1;         
    double BSIM3v1a2;         
    double BSIM3v1keta;     
    double BSIM3v1nsub;
    double BSIM3v1npeak;        
    double BSIM3v1ngate;        
    double BSIM3v1gamma1;      
    double BSIM3v1gamma2;     
    double BSIM3v1vbx;      
    double BSIM3v1vbm;       
    double BSIM3v1xt;       
    double BSIM3v1k1;
    double BSIM3v1kt1;
    double BSIM3v1kt1l;
    double BSIM3v1kt2;
    double BSIM3v1k2;
    double BSIM3v1k3;
    double BSIM3v1k3b;
    double BSIM3v1w0;
    double BSIM3v1nlx;
    double BSIM3v1dvt0;      
    double BSIM3v1dvt1;      
    double BSIM3v1dvt2;      
    double BSIM3v1dvt0w;      
    double BSIM3v1dvt1w;      
    double BSIM3v1dvt2w;      
    double BSIM3v1drout;      
    double BSIM3v1dsub;      
    double BSIM3v1vth0;
    double BSIM3v1ua;
    double BSIM3v1ua1;
    double BSIM3v1ub;
    double BSIM3v1ub1;
    double BSIM3v1uc;
    double BSIM3v1uc1;
    double BSIM3v1u0;
    double BSIM3v1ute;
    double BSIM3v1voff;
    double BSIM3v1delta;
    double BSIM3v1rdsw;       
    double BSIM3v1prwg;
    double BSIM3v1prwb;
    double BSIM3v1prt;       
    double BSIM3v1eta0;         
    double BSIM3v1etab;         
    double BSIM3v1pclm;      
    double BSIM3v1pdibl1;      
    double BSIM3v1pdibl2;      
    double BSIM3v1pdiblb;
    double BSIM3v1pscbe1;       
    double BSIM3v1pscbe2;       
    double BSIM3v1pvag;       
    double BSIM3v1wr;
    double BSIM3v1dwg;
    double BSIM3v1dwb;
    double BSIM3v1b0;
    double BSIM3v1b1;
    double BSIM3v1alpha0;
    double BSIM3v1beta0;
/* serban */
    double BSIM3v1hdif;

    /* CV model */
    double BSIM3v1elm;
    double BSIM3v1cgsl;
    double BSIM3v1cgdl;
    double BSIM3v1ckappa;
    double BSIM3v1cf;
    double BSIM3v1vfbcv;
    double BSIM3v1clc;
    double BSIM3v1cle;
    double BSIM3v1dwc;
    double BSIM3v1dlc;

    /* Length Dependence */
    double BSIM3v1lcdsc;           
    double BSIM3v1lcdscb; 
    double BSIM3v1lcdscd;          
    double BSIM3v1lcit;           
    double BSIM3v1lnfactor;      
    double BSIM3v1lxj;
    double BSIM3v1lvsat;         
    double BSIM3v1lat;         
    double BSIM3v1la0;   
    double BSIM3v1lags;      
    double BSIM3v1la1;         
    double BSIM3v1la2;         
    double BSIM3v1lketa;     
    double BSIM3v1lnsub;
    double BSIM3v1lnpeak;        
    double BSIM3v1lngate;        
    double BSIM3v1lgamma1;      
    double BSIM3v1lgamma2;     
    double BSIM3v1lvbx;      
    double BSIM3v1lvbm;       
    double BSIM3v1lxt;       
    double BSIM3v1lk1;
    double BSIM3v1lkt1;
    double BSIM3v1lkt1l;
    double BSIM3v1lkt2;
    double BSIM3v1lk2;
    double BSIM3v1lk3;
    double BSIM3v1lk3b;
    double BSIM3v1lw0;
    double BSIM3v1lnlx;
    double BSIM3v1ldvt0;      
    double BSIM3v1ldvt1;      
    double BSIM3v1ldvt2;      
    double BSIM3v1ldvt0w;      
    double BSIM3v1ldvt1w;      
    double BSIM3v1ldvt2w;      
    double BSIM3v1ldrout;      
    double BSIM3v1ldsub;      
    double BSIM3v1lvth0;
    double BSIM3v1lua;
    double BSIM3v1lua1;
    double BSIM3v1lub;
    double BSIM3v1lub1;
    double BSIM3v1luc;
    double BSIM3v1luc1;
    double BSIM3v1lu0;
    double BSIM3v1lute;
    double BSIM3v1lvoff;
    double BSIM3v1ldelta;
    double BSIM3v1lrdsw;       
    double BSIM3v1lprwg;
    double BSIM3v1lprwb;
    double BSIM3v1lprt;       
    double BSIM3v1leta0;         
    double BSIM3v1letab;         
    double BSIM3v1lpclm;      
    double BSIM3v1lpdibl1;      
    double BSIM3v1lpdibl2;      
    double BSIM3v1lpdiblb;
    double BSIM3v1lpscbe1;       
    double BSIM3v1lpscbe2;       
    double BSIM3v1lpvag;       
    double BSIM3v1lwr;
    double BSIM3v1ldwg;
    double BSIM3v1ldwb;
    double BSIM3v1lb0;
    double BSIM3v1lb1;
    double BSIM3v1lalpha0;
    double BSIM3v1lbeta0;

    /* CV model */
    double BSIM3v1lelm;
    double BSIM3v1lcgsl;
    double BSIM3v1lcgdl;
    double BSIM3v1lckappa;
    double BSIM3v1lcf;
    double BSIM3v1lclc;
    double BSIM3v1lcle;
    double BSIM3v1lvfbcv;

    /* Width Dependence */
    double BSIM3v1wcdsc;           
    double BSIM3v1wcdscb; 
    double BSIM3v1wcdscd;          
    double BSIM3v1wcit;           
    double BSIM3v1wnfactor;      
    double BSIM3v1wxj;
    double BSIM3v1wvsat;         
    double BSIM3v1wat;         
    double BSIM3v1wa0;   
    double BSIM3v1wags;      
    double BSIM3v1wa1;         
    double BSIM3v1wa2;         
    double BSIM3v1wketa;     
    double BSIM3v1wnsub;
    double BSIM3v1wnpeak;        
    double BSIM3v1wngate;        
    double BSIM3v1wgamma1;      
    double BSIM3v1wgamma2;     
    double BSIM3v1wvbx;      
    double BSIM3v1wvbm;       
    double BSIM3v1wxt;       
    double BSIM3v1wk1;
    double BSIM3v1wkt1;
    double BSIM3v1wkt1l;
    double BSIM3v1wkt2;
    double BSIM3v1wk2;
    double BSIM3v1wk3;
    double BSIM3v1wk3b;
    double BSIM3v1ww0;
    double BSIM3v1wnlx;
    double BSIM3v1wdvt0;      
    double BSIM3v1wdvt1;      
    double BSIM3v1wdvt2;      
    double BSIM3v1wdvt0w;      
    double BSIM3v1wdvt1w;      
    double BSIM3v1wdvt2w;      
    double BSIM3v1wdrout;      
    double BSIM3v1wdsub;      
    double BSIM3v1wvth0;
    double BSIM3v1wua;
    double BSIM3v1wua1;
    double BSIM3v1wub;
    double BSIM3v1wub1;
    double BSIM3v1wuc;
    double BSIM3v1wuc1;
    double BSIM3v1wu0;
    double BSIM3v1wute;
    double BSIM3v1wvoff;
    double BSIM3v1wdelta;
    double BSIM3v1wrdsw;       
    double BSIM3v1wprwg;
    double BSIM3v1wprwb;
    double BSIM3v1wprt;       
    double BSIM3v1weta0;         
    double BSIM3v1wetab;         
    double BSIM3v1wpclm;      
    double BSIM3v1wpdibl1;      
    double BSIM3v1wpdibl2;      
    double BSIM3v1wpdiblb;
    double BSIM3v1wpscbe1;       
    double BSIM3v1wpscbe2;       
    double BSIM3v1wpvag;       
    double BSIM3v1wwr;
    double BSIM3v1wdwg;
    double BSIM3v1wdwb;
    double BSIM3v1wb0;
    double BSIM3v1wb1;
    double BSIM3v1walpha0;
    double BSIM3v1wbeta0;

    /* CV model */
    double BSIM3v1welm;
    double BSIM3v1wcgsl;
    double BSIM3v1wcgdl;
    double BSIM3v1wckappa;
    double BSIM3v1wcf;
    double BSIM3v1wclc;
    double BSIM3v1wcle;
    double BSIM3v1wvfbcv;

    /* Cross-term Dependence */
    double BSIM3v1pcdsc;           
    double BSIM3v1pcdscb; 
    double BSIM3v1pcdscd;          
    double BSIM3v1pcit;           
    double BSIM3v1pnfactor;      
    double BSIM3v1pxj;
    double BSIM3v1pvsat;         
    double BSIM3v1pat;         
    double BSIM3v1pa0;   
    double BSIM3v1pags;      
    double BSIM3v1pa1;         
    double BSIM3v1pa2;         
    double BSIM3v1pketa;     
    double BSIM3v1pnsub;
    double BSIM3v1pnpeak;        
    double BSIM3v1pngate;        
    double BSIM3v1pgamma1;      
    double BSIM3v1pgamma2;     
    double BSIM3v1pvbx;      
    double BSIM3v1pvbm;       
    double BSIM3v1pxt;       
    double BSIM3v1pk1;
    double BSIM3v1pkt1;
    double BSIM3v1pkt1l;
    double BSIM3v1pkt2;
    double BSIM3v1pk2;
    double BSIM3v1pk3;
    double BSIM3v1pk3b;
    double BSIM3v1pw0;
    double BSIM3v1pnlx;
    double BSIM3v1pdvt0;      
    double BSIM3v1pdvt1;      
    double BSIM3v1pdvt2;      
    double BSIM3v1pdvt0w;      
    double BSIM3v1pdvt1w;      
    double BSIM3v1pdvt2w;      
    double BSIM3v1pdrout;      
    double BSIM3v1pdsub;      
    double BSIM3v1pvth0;
    double BSIM3v1pua;
    double BSIM3v1pua1;
    double BSIM3v1pub;
    double BSIM3v1pub1;
    double BSIM3v1puc;
    double BSIM3v1puc1;
    double BSIM3v1pu0;
    double BSIM3v1pute;
    double BSIM3v1pvoff;
    double BSIM3v1pdelta;
    double BSIM3v1prdsw;
    double BSIM3v1pprwg;
    double BSIM3v1pprwb;
    double BSIM3v1pprt;       
    double BSIM3v1peta0;         
    double BSIM3v1petab;         
    double BSIM3v1ppclm;      
    double BSIM3v1ppdibl1;      
    double BSIM3v1ppdibl2;      
    double BSIM3v1ppdiblb;
    double BSIM3v1ppscbe1;       
    double BSIM3v1ppscbe2;       
    double BSIM3v1ppvag;       
    double BSIM3v1pwr;
    double BSIM3v1pdwg;
    double BSIM3v1pdwb;
    double BSIM3v1pb0;
    double BSIM3v1pb1;
    double BSIM3v1palpha0;
    double BSIM3v1pbeta0;

    /* CV model */
    double BSIM3v1pelm;
    double BSIM3v1pcgsl;
    double BSIM3v1pcgdl;
    double BSIM3v1pckappa;
    double BSIM3v1pcf;
    double BSIM3v1pclc;
    double BSIM3v1pcle;
    double BSIM3v1pvfbcv;

    double BSIM3v1tnom;
    double BSIM3v1cgso;
    double BSIM3v1cgdo;
    double BSIM3v1cgbo;
    double BSIM3v1xpart;
    double BSIM3v1cFringOut;
    double BSIM3v1cFringMax;

    double BSIM3v1sheetResistance;
    double BSIM3v1jctSatCurDensity;
    double BSIM3v1jctSidewallSatCurDensity;
    double BSIM3v1bulkJctPotential;
    double BSIM3v1bulkJctBotGradingCoeff;
    double BSIM3v1bulkJctSideGradingCoeff;
    double BSIM3v1bulkJctGateSideGradingCoeff;
    double BSIM3v1sidewallJctPotential;
    double BSIM3v1GatesidewallJctPotential;
    double BSIM3v1unitAreaJctCap;
    double BSIM3v1unitLengthSidewallJctCap;
    double BSIM3v1unitLengthGateSidewallJctCap;
    double BSIM3v1jctEmissionCoeff;
    double BSIM3v1jctTempExponent;

    double BSIM3v1Lint;
    double BSIM3v1Ll;
    double BSIM3v1Lln;
    double BSIM3v1Lw;
    double BSIM3v1Lwn;
    double BSIM3v1Lwl;
    double BSIM3v1Lmin;
    double BSIM3v1Lmax;

    double BSIM3v1Wint;
    double BSIM3v1Wl;
    double BSIM3v1Wln;
    double BSIM3v1Ww;
    double BSIM3v1Wwn;
    double BSIM3v1Wwl;
    double BSIM3v1Wmin;
    double BSIM3v1Wmax;


/* Pre-calculated constants */
    /* MCJ: move to size-dependent param. */
    double BSIM3v1vtm;   
    double BSIM3v1cox;
    double BSIM3v1cof1;
    double BSIM3v1cof2;
    double BSIM3v1cof3;
    double BSIM3v1cof4;
    double BSIM3v1vcrit;
    double BSIM3v1factor1;
    double BSIM3v1jctTempSatCurDensity;
    double BSIM3v1jctSidewallTempSatCurDensity;

    double BSIM3v1oxideTrapDensityA;      
    double BSIM3v1oxideTrapDensityB;     
    double BSIM3v1oxideTrapDensityC;  
    double BSIM3v1em;  
    double BSIM3v1ef;  
    double BSIM3v1af;  
    double BSIM3v1kf;  

    struct bsim3v1SizeDependParam *pSizeDependParamKnot;

    /* Flags */
    unsigned  BSIM3v1mobModGiven :1;
    unsigned  BSIM3v1binUnitGiven :1;
    unsigned  BSIM3v1capModGiven :1;
    unsigned  BSIM3v1paramChkGiven :1;
    unsigned  BSIM3v1nqsModGiven :1;
    unsigned  BSIM3v1noiModGiven :1;
    unsigned  BSIM3v1typeGiven   :1;
    unsigned  BSIM3v1toxGiven   :1;
    unsigned  BSIM3v1versionGiven   :1;

    unsigned  BSIM3v1cdscGiven   :1;
    unsigned  BSIM3v1cdscbGiven   :1;
    unsigned  BSIM3v1cdscdGiven   :1;
    unsigned  BSIM3v1citGiven   :1;
    unsigned  BSIM3v1nfactorGiven   :1;
    unsigned  BSIM3v1xjGiven   :1;
    unsigned  BSIM3v1vsatGiven   :1;
    unsigned  BSIM3v1atGiven   :1;
    unsigned  BSIM3v1a0Given   :1;
    unsigned  BSIM3v1agsGiven   :1;
    unsigned  BSIM3v1a1Given   :1;
    unsigned  BSIM3v1a2Given   :1;
    unsigned  BSIM3v1ketaGiven   :1;    
    unsigned  BSIM3v1nsubGiven   :1;
    unsigned  BSIM3v1npeakGiven   :1;
    unsigned  BSIM3v1ngateGiven   :1;
    unsigned  BSIM3v1gamma1Given   :1;
    unsigned  BSIM3v1gamma2Given   :1;
    unsigned  BSIM3v1vbxGiven   :1;
    unsigned  BSIM3v1vbmGiven   :1;
    unsigned  BSIM3v1xtGiven   :1;
    unsigned  BSIM3v1k1Given   :1;
    unsigned  BSIM3v1kt1Given   :1;
    unsigned  BSIM3v1kt1lGiven   :1;
    unsigned  BSIM3v1kt2Given   :1;
    unsigned  BSIM3v1k2Given   :1;
    unsigned  BSIM3v1k3Given   :1;
    unsigned  BSIM3v1k3bGiven   :1;
    unsigned  BSIM3v1w0Given   :1;
    unsigned  BSIM3v1nlxGiven   :1;
    unsigned  BSIM3v1dvt0Given   :1;   
    unsigned  BSIM3v1dvt1Given   :1;     
    unsigned  BSIM3v1dvt2Given   :1;     
    unsigned  BSIM3v1dvt0wGiven   :1;   
    unsigned  BSIM3v1dvt1wGiven   :1;     
    unsigned  BSIM3v1dvt2wGiven   :1;     
    unsigned  BSIM3v1droutGiven   :1;     
    unsigned  BSIM3v1dsubGiven   :1;     
    unsigned  BSIM3v1vth0Given   :1;
    unsigned  BSIM3v1uaGiven   :1;
    unsigned  BSIM3v1ua1Given   :1;
    unsigned  BSIM3v1ubGiven   :1;
    unsigned  BSIM3v1ub1Given   :1;
    unsigned  BSIM3v1ucGiven   :1;
    unsigned  BSIM3v1uc1Given   :1;
    unsigned  BSIM3v1u0Given   :1;
    unsigned  BSIM3v1uteGiven   :1;
    unsigned  BSIM3v1voffGiven   :1;
    unsigned  BSIM3v1rdswGiven   :1;      
    unsigned  BSIM3v1prwgGiven   :1;      
    unsigned  BSIM3v1prwbGiven   :1;      
    unsigned  BSIM3v1prtGiven   :1;      
    unsigned  BSIM3v1eta0Given   :1;    
    unsigned  BSIM3v1etabGiven   :1;    
    unsigned  BSIM3v1pclmGiven   :1;   
    unsigned  BSIM3v1pdibl1Given   :1;   
    unsigned  BSIM3v1pdibl2Given   :1;  
    unsigned  BSIM3v1pdiblbGiven   :1;  
    unsigned  BSIM3v1pscbe1Given   :1;    
    unsigned  BSIM3v1pscbe2Given   :1;    
    unsigned  BSIM3v1pvagGiven   :1;    
    unsigned  BSIM3v1deltaGiven  :1;     
    unsigned  BSIM3v1wrGiven   :1;
    unsigned  BSIM3v1dwgGiven   :1;
    unsigned  BSIM3v1dwbGiven   :1;
    unsigned  BSIM3v1b0Given   :1;
    unsigned  BSIM3v1b1Given   :1;
    unsigned  BSIM3v1alpha0Given   :1;
    unsigned  BSIM3v1beta0Given   :1;
    unsigned  BSIM3v1hdifGiven   :1;

    /* CV model */
    unsigned  BSIM3v1elmGiven  :1;     
    unsigned  BSIM3v1cgslGiven   :1;
    unsigned  BSIM3v1cgdlGiven   :1;
    unsigned  BSIM3v1ckappaGiven   :1;
    unsigned  BSIM3v1cfGiven   :1;
    unsigned  BSIM3v1vfbcvGiven   :1;
    unsigned  BSIM3v1clcGiven   :1;
    unsigned  BSIM3v1cleGiven   :1;
    unsigned  BSIM3v1dwcGiven   :1;
    unsigned  BSIM3v1dlcGiven   :1;


    /* Length dependence */
    unsigned  BSIM3v1lcdscGiven   :1;
    unsigned  BSIM3v1lcdscbGiven   :1;
    unsigned  BSIM3v1lcdscdGiven   :1;
    unsigned  BSIM3v1lcitGiven   :1;
    unsigned  BSIM3v1lnfactorGiven   :1;
    unsigned  BSIM3v1lxjGiven   :1;
    unsigned  BSIM3v1lvsatGiven   :1;
    unsigned  BSIM3v1latGiven   :1;
    unsigned  BSIM3v1la0Given   :1;
    unsigned  BSIM3v1lagsGiven   :1;
    unsigned  BSIM3v1la1Given   :1;
    unsigned  BSIM3v1la2Given   :1;
    unsigned  BSIM3v1lketaGiven   :1;    
    unsigned  BSIM3v1lnsubGiven   :1;
    unsigned  BSIM3v1lnpeakGiven   :1;
    unsigned  BSIM3v1lngateGiven   :1;
    unsigned  BSIM3v1lgamma1Given   :1;
    unsigned  BSIM3v1lgamma2Given   :1;
    unsigned  BSIM3v1lvbxGiven   :1;
    unsigned  BSIM3v1lvbmGiven   :1;
    unsigned  BSIM3v1lxtGiven   :1;
    unsigned  BSIM3v1lk1Given   :1;
    unsigned  BSIM3v1lkt1Given   :1;
    unsigned  BSIM3v1lkt1lGiven   :1;
    unsigned  BSIM3v1lkt2Given   :1;
    unsigned  BSIM3v1lk2Given   :1;
    unsigned  BSIM3v1lk3Given   :1;
    unsigned  BSIM3v1lk3bGiven   :1;
    unsigned  BSIM3v1lw0Given   :1;
    unsigned  BSIM3v1lnlxGiven   :1;
    unsigned  BSIM3v1ldvt0Given   :1;   
    unsigned  BSIM3v1ldvt1Given   :1;     
    unsigned  BSIM3v1ldvt2Given   :1;     
    unsigned  BSIM3v1ldvt0wGiven   :1;   
    unsigned  BSIM3v1ldvt1wGiven   :1;     
    unsigned  BSIM3v1ldvt2wGiven   :1;     
    unsigned  BSIM3v1ldroutGiven   :1;     
    unsigned  BSIM3v1ldsubGiven   :1;     
    unsigned  BSIM3v1lvth0Given   :1;
    unsigned  BSIM3v1luaGiven   :1;
    unsigned  BSIM3v1lua1Given   :1;
    unsigned  BSIM3v1lubGiven   :1;
    unsigned  BSIM3v1lub1Given   :1;
    unsigned  BSIM3v1lucGiven   :1;
    unsigned  BSIM3v1luc1Given   :1;
    unsigned  BSIM3v1lu0Given   :1;
    unsigned  BSIM3v1luteGiven   :1;
    unsigned  BSIM3v1lvoffGiven   :1;
    unsigned  BSIM3v1lrdswGiven   :1;      
    unsigned  BSIM3v1lprwgGiven   :1;      
    unsigned  BSIM3v1lprwbGiven   :1;      
    unsigned  BSIM3v1lprtGiven   :1;      
    unsigned  BSIM3v1leta0Given   :1;    
    unsigned  BSIM3v1letabGiven   :1;    
    unsigned  BSIM3v1lpclmGiven   :1;   
    unsigned  BSIM3v1lpdibl1Given   :1;   
    unsigned  BSIM3v1lpdibl2Given   :1;  
    unsigned  BSIM3v1lpdiblbGiven   :1;  
    unsigned  BSIM3v1lpscbe1Given   :1;    
    unsigned  BSIM3v1lpscbe2Given   :1;    
    unsigned  BSIM3v1lpvagGiven   :1;    
    unsigned  BSIM3v1ldeltaGiven  :1;     
    unsigned  BSIM3v1lwrGiven   :1;
    unsigned  BSIM3v1ldwgGiven   :1;
    unsigned  BSIM3v1ldwbGiven   :1;
    unsigned  BSIM3v1lb0Given   :1;
    unsigned  BSIM3v1lb1Given   :1;
    unsigned  BSIM3v1lalpha0Given   :1;
    unsigned  BSIM3v1lbeta0Given   :1;

    /* CV model */
    unsigned  BSIM3v1lelmGiven  :1;     
    unsigned  BSIM3v1lcgslGiven   :1;
    unsigned  BSIM3v1lcgdlGiven   :1;
    unsigned  BSIM3v1lckappaGiven   :1;
    unsigned  BSIM3v1lcfGiven   :1;
    unsigned  BSIM3v1lclcGiven   :1;
    unsigned  BSIM3v1lcleGiven   :1;
    unsigned  BSIM3v1lvfbcvGiven   :1;

    /* Width dependence */
    unsigned  BSIM3v1wcdscGiven   :1;
    unsigned  BSIM3v1wcdscbGiven   :1;
    unsigned  BSIM3v1wcdscdGiven   :1;
    unsigned  BSIM3v1wcitGiven   :1;
    unsigned  BSIM3v1wnfactorGiven   :1;
    unsigned  BSIM3v1wxjGiven   :1;
    unsigned  BSIM3v1wvsatGiven   :1;
    unsigned  BSIM3v1watGiven   :1;
    unsigned  BSIM3v1wa0Given   :1;
    unsigned  BSIM3v1wagsGiven   :1;
    unsigned  BSIM3v1wa1Given   :1;
    unsigned  BSIM3v1wa2Given   :1;
    unsigned  BSIM3v1wketaGiven   :1;    
    unsigned  BSIM3v1wnsubGiven   :1;
    unsigned  BSIM3v1wnpeakGiven   :1;
    unsigned  BSIM3v1wngateGiven   :1;
    unsigned  BSIM3v1wgamma1Given   :1;
    unsigned  BSIM3v1wgamma2Given   :1;
    unsigned  BSIM3v1wvbxGiven   :1;
    unsigned  BSIM3v1wvbmGiven   :1;
    unsigned  BSIM3v1wxtGiven   :1;
    unsigned  BSIM3v1wk1Given   :1;
    unsigned  BSIM3v1wkt1Given   :1;
    unsigned  BSIM3v1wkt1lGiven   :1;
    unsigned  BSIM3v1wkt2Given   :1;
    unsigned  BSIM3v1wk2Given   :1;
    unsigned  BSIM3v1wk3Given   :1;
    unsigned  BSIM3v1wk3bGiven   :1;
    unsigned  BSIM3v1ww0Given   :1;
    unsigned  BSIM3v1wnlxGiven   :1;
    unsigned  BSIM3v1wdvt0Given   :1;   
    unsigned  BSIM3v1wdvt1Given   :1;     
    unsigned  BSIM3v1wdvt2Given   :1;     
    unsigned  BSIM3v1wdvt0wGiven   :1;   
    unsigned  BSIM3v1wdvt1wGiven   :1;     
    unsigned  BSIM3v1wdvt2wGiven   :1;     
    unsigned  BSIM3v1wdroutGiven   :1;     
    unsigned  BSIM3v1wdsubGiven   :1;     
    unsigned  BSIM3v1wvth0Given   :1;
    unsigned  BSIM3v1wuaGiven   :1;
    unsigned  BSIM3v1wua1Given   :1;
    unsigned  BSIM3v1wubGiven   :1;
    unsigned  BSIM3v1wub1Given   :1;
    unsigned  BSIM3v1wucGiven   :1;
    unsigned  BSIM3v1wuc1Given   :1;
    unsigned  BSIM3v1wu0Given   :1;
    unsigned  BSIM3v1wuteGiven   :1;
    unsigned  BSIM3v1wvoffGiven   :1;
    unsigned  BSIM3v1wrdswGiven   :1;      
    unsigned  BSIM3v1wprwgGiven   :1;      
    unsigned  BSIM3v1wprwbGiven   :1;      
    unsigned  BSIM3v1wprtGiven   :1;      
    unsigned  BSIM3v1weta0Given   :1;    
    unsigned  BSIM3v1wetabGiven   :1;    
    unsigned  BSIM3v1wpclmGiven   :1;   
    unsigned  BSIM3v1wpdibl1Given   :1;   
    unsigned  BSIM3v1wpdibl2Given   :1;  
    unsigned  BSIM3v1wpdiblbGiven   :1;  
    unsigned  BSIM3v1wpscbe1Given   :1;    
    unsigned  BSIM3v1wpscbe2Given   :1;    
    unsigned  BSIM3v1wpvagGiven   :1;    
    unsigned  BSIM3v1wdeltaGiven  :1;     
    unsigned  BSIM3v1wwrGiven   :1;
    unsigned  BSIM3v1wdwgGiven   :1;
    unsigned  BSIM3v1wdwbGiven   :1;
    unsigned  BSIM3v1wb0Given   :1;
    unsigned  BSIM3v1wb1Given   :1;
    unsigned  BSIM3v1walpha0Given   :1;
    unsigned  BSIM3v1wbeta0Given   :1;

    /* CV model */
    unsigned  BSIM3v1welmGiven  :1;     
    unsigned  BSIM3v1wcgslGiven   :1;
    unsigned  BSIM3v1wcgdlGiven   :1;
    unsigned  BSIM3v1wckappaGiven   :1;
    unsigned  BSIM3v1wcfGiven   :1;
    unsigned  BSIM3v1wclcGiven   :1;
    unsigned  BSIM3v1wcleGiven   :1;
    unsigned  BSIM3v1wvfbcvGiven   :1;

    /* Cross-term dependence */
    unsigned  BSIM3v1pcdscGiven   :1;
    unsigned  BSIM3v1pcdscbGiven   :1;
    unsigned  BSIM3v1pcdscdGiven   :1;
    unsigned  BSIM3v1pcitGiven   :1;
    unsigned  BSIM3v1pnfactorGiven   :1;
    unsigned  BSIM3v1pxjGiven   :1;
    unsigned  BSIM3v1pvsatGiven   :1;
    unsigned  BSIM3v1patGiven   :1;
    unsigned  BSIM3v1pa0Given   :1;
    unsigned  BSIM3v1pagsGiven   :1;
    unsigned  BSIM3v1pa1Given   :1;
    unsigned  BSIM3v1pa2Given   :1;
    unsigned  BSIM3v1pketaGiven   :1;    
    unsigned  BSIM3v1pnsubGiven   :1;
    unsigned  BSIM3v1pnpeakGiven   :1;
    unsigned  BSIM3v1pngateGiven   :1;
    unsigned  BSIM3v1pgamma1Given   :1;
    unsigned  BSIM3v1pgamma2Given   :1;
    unsigned  BSIM3v1pvbxGiven   :1;
    unsigned  BSIM3v1pvbmGiven   :1;
    unsigned  BSIM3v1pxtGiven   :1;
    unsigned  BSIM3v1pk1Given   :1;
    unsigned  BSIM3v1pkt1Given   :1;
    unsigned  BSIM3v1pkt1lGiven   :1;
    unsigned  BSIM3v1pkt2Given   :1;
    unsigned  BSIM3v1pk2Given   :1;
    unsigned  BSIM3v1pk3Given   :1;
    unsigned  BSIM3v1pk3bGiven   :1;
    unsigned  BSIM3v1pw0Given   :1;
    unsigned  BSIM3v1pnlxGiven   :1;
    unsigned  BSIM3v1pdvt0Given   :1;   
    unsigned  BSIM3v1pdvt1Given   :1;     
    unsigned  BSIM3v1pdvt2Given   :1;     
    unsigned  BSIM3v1pdvt0wGiven   :1;   
    unsigned  BSIM3v1pdvt1wGiven   :1;     
    unsigned  BSIM3v1pdvt2wGiven   :1;     
    unsigned  BSIM3v1pdroutGiven   :1;     
    unsigned  BSIM3v1pdsubGiven   :1;     
    unsigned  BSIM3v1pvth0Given   :1;
    unsigned  BSIM3v1puaGiven   :1;
    unsigned  BSIM3v1pua1Given   :1;
    unsigned  BSIM3v1pubGiven   :1;
    unsigned  BSIM3v1pub1Given   :1;
    unsigned  BSIM3v1pucGiven   :1;
    unsigned  BSIM3v1puc1Given   :1;
    unsigned  BSIM3v1pu0Given   :1;
    unsigned  BSIM3v1puteGiven   :1;
    unsigned  BSIM3v1pvoffGiven   :1;
    unsigned  BSIM3v1prdswGiven   :1;      
    unsigned  BSIM3v1pprwgGiven   :1;      
    unsigned  BSIM3v1pprwbGiven   :1;      
    unsigned  BSIM3v1pprtGiven   :1;      
    unsigned  BSIM3v1peta0Given   :1;    
    unsigned  BSIM3v1petabGiven   :1;    
    unsigned  BSIM3v1ppclmGiven   :1;   
    unsigned  BSIM3v1ppdibl1Given   :1;   
    unsigned  BSIM3v1ppdibl2Given   :1;  
    unsigned  BSIM3v1ppdiblbGiven   :1;  
    unsigned  BSIM3v1ppscbe1Given   :1;    
    unsigned  BSIM3v1ppscbe2Given   :1;    
    unsigned  BSIM3v1ppvagGiven   :1;    
    unsigned  BSIM3v1pdeltaGiven  :1;     
    unsigned  BSIM3v1pwrGiven   :1;
    unsigned  BSIM3v1pdwgGiven   :1;
    unsigned  BSIM3v1pdwbGiven   :1;
    unsigned  BSIM3v1pb0Given   :1;
    unsigned  BSIM3v1pb1Given   :1;
    unsigned  BSIM3v1palpha0Given   :1;
    unsigned  BSIM3v1pbeta0Given   :1;

    /* CV model */
    unsigned  BSIM3v1pelmGiven  :1;     
    unsigned  BSIM3v1pcgslGiven   :1;
    unsigned  BSIM3v1pcgdlGiven   :1;
    unsigned  BSIM3v1pckappaGiven   :1;
    unsigned  BSIM3v1pcfGiven   :1;
    unsigned  BSIM3v1pclcGiven   :1;
    unsigned  BSIM3v1pcleGiven   :1;
    unsigned  BSIM3v1pvfbcvGiven   :1;

    unsigned  BSIM3v1useFringeGiven   :1;

    unsigned  BSIM3v1tnomGiven   :1;
    unsigned  BSIM3v1cgsoGiven   :1;
    unsigned  BSIM3v1cgdoGiven   :1;
    unsigned  BSIM3v1cgboGiven   :1;
    unsigned  BSIM3v1xpartGiven   :1;
    unsigned  BSIM3v1sheetResistanceGiven   :1;
    unsigned  BSIM3v1jctSatCurDensityGiven   :1;
    unsigned  BSIM3v1jctSidewallSatCurDensityGiven   :1;
    unsigned  BSIM3v1bulkJctPotentialGiven   :1;
    unsigned  BSIM3v1bulkJctBotGradingCoeffGiven   :1;
    unsigned  BSIM3v1sidewallJctPotentialGiven   :1;
    unsigned  BSIM3v1GatesidewallJctPotentialGiven   :1;
    unsigned  BSIM3v1bulkJctSideGradingCoeffGiven   :1;
    unsigned  BSIM3v1unitAreaJctCapGiven   :1;
    unsigned  BSIM3v1unitLengthSidewallJctCapGiven   :1;
    unsigned  BSIM3v1bulkJctGateSideGradingCoeffGiven   :1;
    unsigned  BSIM3v1unitLengthGateSidewallJctCapGiven   :1;
    unsigned  BSIM3v1jctEmissionCoeffGiven :1;
    unsigned  BSIM3v1jctTempExponentGiven	:1;

    unsigned  BSIM3v1oxideTrapDensityAGiven  :1;         
    unsigned  BSIM3v1oxideTrapDensityBGiven  :1;        
    unsigned  BSIM3v1oxideTrapDensityCGiven  :1;     
    unsigned  BSIM3v1emGiven  :1;     
    unsigned  BSIM3v1efGiven  :1;     
    unsigned  BSIM3v1afGiven  :1;     
    unsigned  BSIM3v1kfGiven  :1;     

    unsigned  BSIM3v1LintGiven   :1;
    unsigned  BSIM3v1LlGiven   :1;
    unsigned  BSIM3v1LlnGiven   :1;
    unsigned  BSIM3v1LwGiven   :1;
    unsigned  BSIM3v1LwnGiven   :1;
    unsigned  BSIM3v1LwlGiven   :1;
    unsigned  BSIM3v1LminGiven   :1;
    unsigned  BSIM3v1LmaxGiven   :1;

    unsigned  BSIM3v1WintGiven   :1;
    unsigned  BSIM3v1WlGiven   :1;
    unsigned  BSIM3v1WlnGiven   :1;
    unsigned  BSIM3v1WwGiven   :1;
    unsigned  BSIM3v1WwnGiven   :1;
    unsigned  BSIM3v1WwlGiven   :1;
    unsigned  BSIM3v1WminGiven   :1;
    unsigned  BSIM3v1WmaxGiven   :1;

} BSIM3v1model;


#ifndef NMOS
#define NMOS 1
#define PMOS -1
#endif /*NMOS*/


/* device parameters */
#define BSIM3v1_W 1
#define BSIM3v1_L 2
#define BSIM3v1_M 15
#define BSIM3v1_AS 3
#define BSIM3v1_AD 4
#define BSIM3v1_PS 5
#define BSIM3v1_PD 6
#define BSIM3v1_NRS 7
#define BSIM3v1_NRD 8
#define BSIM3v1_OFF 9
#define BSIM3v1_IC_VBS 10
#define BSIM3v1_IC_VDS 11
#define BSIM3v1_IC_VGS 12
#define BSIM3v1_IC 13
#define BSIM3v1_NQSMOD 14

/* model parameters */
#define BSIM3v1_MOD_CAPMOD          101
#define BSIM3v1_MOD_NQSMOD          102
#define BSIM3v1_MOD_MOBMOD          103    
#define BSIM3v1_MOD_NOIMOD          104    

#define BSIM3v1_MOD_TOX             105

#define BSIM3v1_MOD_CDSC            106
#define BSIM3v1_MOD_CDSCB           107
#define BSIM3v1_MOD_CIT             108
#define BSIM3v1_MOD_NFACTOR         109
#define BSIM3v1_MOD_XJ              110
#define BSIM3v1_MOD_VSAT            111
#define BSIM3v1_MOD_AT              112
#define BSIM3v1_MOD_A0              113
#define BSIM3v1_MOD_A1              114
#define BSIM3v1_MOD_A2              115
#define BSIM3v1_MOD_KETA            116   
#define BSIM3v1_MOD_NSUB            117
#define BSIM3v1_MOD_NPEAK           118
#define BSIM3v1_MOD_NGATE           120
#define BSIM3v1_MOD_GAMMA1          121
#define BSIM3v1_MOD_GAMMA2          122
#define BSIM3v1_MOD_VBX             123
#define BSIM3v1_MOD_BINUNIT         124    

#define BSIM3v1_MOD_VBM             125

#define BSIM3v1_MOD_XT              126
#define BSIM3v1_MOD_K1              129
#define BSIM3v1_MOD_KT1             130
#define BSIM3v1_MOD_KT1L            131
#define BSIM3v1_MOD_K2              132
#define BSIM3v1_MOD_KT2             133
#define BSIM3v1_MOD_K3              134
#define BSIM3v1_MOD_K3B             135
#define BSIM3v1_MOD_W0              136
#define BSIM3v1_MOD_NLX             137

#define BSIM3v1_MOD_DVT0            138
#define BSIM3v1_MOD_DVT1            139
#define BSIM3v1_MOD_DVT2            140

#define BSIM3v1_MOD_DVT0W           141
#define BSIM3v1_MOD_DVT1W           142
#define BSIM3v1_MOD_DVT2W           143

#define BSIM3v1_MOD_DROUT           144
#define BSIM3v1_MOD_DSUB            145
#define BSIM3v1_MOD_VTH0            146
#define BSIM3v1_MOD_UA              147
#define BSIM3v1_MOD_UA1             148
#define BSIM3v1_MOD_UB              149
#define BSIM3v1_MOD_UB1             150
#define BSIM3v1_MOD_UC              151
#define BSIM3v1_MOD_UC1             152
#define BSIM3v1_MOD_U0              153
#define BSIM3v1_MOD_UTE             154
#define BSIM3v1_MOD_VOFF            155
#define BSIM3v1_MOD_DELTA           156
#define BSIM3v1_MOD_RDSW            157
#define BSIM3v1_MOD_PRT             158
#define BSIM3v1_MOD_LDD             159
#define BSIM3v1_MOD_ETA             160
#define BSIM3v1_MOD_ETA0            161
#define BSIM3v1_MOD_ETAB            162
#define BSIM3v1_MOD_PCLM            163
#define BSIM3v1_MOD_PDIBL1          164
#define BSIM3v1_MOD_PDIBL2          165
#define BSIM3v1_MOD_PSCBE1          166
#define BSIM3v1_MOD_PSCBE2          167
#define BSIM3v1_MOD_PVAG            168
#define BSIM3v1_MOD_WR              169
#define BSIM3v1_MOD_DWG             170
#define BSIM3v1_MOD_DWB             171
#define BSIM3v1_MOD_B0              172
#define BSIM3v1_MOD_B1              173
#define BSIM3v1_MOD_ALPHA0          174
#define BSIM3v1_MOD_BETA0           175
#define BSIM3v1_MOD_PDIBLB          178

#define BSIM3v1_MOD_PRWG            179
#define BSIM3v1_MOD_PRWB            180

#define BSIM3v1_MOD_CDSCD           181
#define BSIM3v1_MOD_AGS             182

#define BSIM3v1_MOD_FRINGE          184
#define BSIM3v1_MOD_ELM             185
#define BSIM3v1_MOD_CGSL            186
#define BSIM3v1_MOD_CGDL            187
#define BSIM3v1_MOD_CKAPPA          188
#define BSIM3v1_MOD_CF              189
#define BSIM3v1_MOD_CLC             190
#define BSIM3v1_MOD_CLE             191
#define BSIM3v1_MOD_PARAMCHK        192
#define BSIM3v1_MOD_VERSION         193
#define BSIM3v1_MOD_VFBCV           194

#define BSIM3v1_MOD_HDIF            198

/* Length dependence */
#define BSIM3v1_MOD_LCDSC            201
#define BSIM3v1_MOD_LCDSCB           202
#define BSIM3v1_MOD_LCIT             203
#define BSIM3v1_MOD_LNFACTOR         204
#define BSIM3v1_MOD_LXJ              205
#define BSIM3v1_MOD_LVSAT            206
#define BSIM3v1_MOD_LAT              207
#define BSIM3v1_MOD_LA0              208
#define BSIM3v1_MOD_LA1              209
#define BSIM3v1_MOD_LA2              210
#define BSIM3v1_MOD_LKETA            211   
#define BSIM3v1_MOD_LNSUB            212
#define BSIM3v1_MOD_LNPEAK           213
#define BSIM3v1_MOD_LNGATE           215
#define BSIM3v1_MOD_LGAMMA1          216
#define BSIM3v1_MOD_LGAMMA2          217
#define BSIM3v1_MOD_LVBX             218

#define BSIM3v1_MOD_LVBM             220

#define BSIM3v1_MOD_LXT              222
#define BSIM3v1_MOD_LK1              225
#define BSIM3v1_MOD_LKT1             226
#define BSIM3v1_MOD_LKT1L            227
#define BSIM3v1_MOD_LK2              228
#define BSIM3v1_MOD_LKT2             229
#define BSIM3v1_MOD_LK3              230
#define BSIM3v1_MOD_LK3B             231
#define BSIM3v1_MOD_LW0              232
#define BSIM3v1_MOD_LNLX             233

#define BSIM3v1_MOD_LDVT0            234
#define BSIM3v1_MOD_LDVT1            235
#define BSIM3v1_MOD_LDVT2            236

#define BSIM3v1_MOD_LDVT0W           237
#define BSIM3v1_MOD_LDVT1W           238
#define BSIM3v1_MOD_LDVT2W           239

#define BSIM3v1_MOD_LDROUT           240
#define BSIM3v1_MOD_LDSUB            241
#define BSIM3v1_MOD_LVTH0            242
#define BSIM3v1_MOD_LUA              243
#define BSIM3v1_MOD_LUA1             244
#define BSIM3v1_MOD_LUB              245
#define BSIM3v1_MOD_LUB1             246
#define BSIM3v1_MOD_LUC              247
#define BSIM3v1_MOD_LUC1             248
#define BSIM3v1_MOD_LU0              249
#define BSIM3v1_MOD_LUTE             250
#define BSIM3v1_MOD_LVOFF            251
#define BSIM3v1_MOD_LDELTA           252
#define BSIM3v1_MOD_LRDSW            253
#define BSIM3v1_MOD_LPRT             254
#define BSIM3v1_MOD_LLDD             255
#define BSIM3v1_MOD_LETA             256
#define BSIM3v1_MOD_LETA0            257
#define BSIM3v1_MOD_LETAB            258
#define BSIM3v1_MOD_LPCLM            259
#define BSIM3v1_MOD_LPDIBL1          260
#define BSIM3v1_MOD_LPDIBL2          261
#define BSIM3v1_MOD_LPSCBE1          262
#define BSIM3v1_MOD_LPSCBE2          263
#define BSIM3v1_MOD_LPVAG            264
#define BSIM3v1_MOD_LWR              265
#define BSIM3v1_MOD_LDWG             266
#define BSIM3v1_MOD_LDWB             267
#define BSIM3v1_MOD_LB0              268
#define BSIM3v1_MOD_LB1              269
#define BSIM3v1_MOD_LALPHA0          270
#define BSIM3v1_MOD_LBETA0           271
#define BSIM3v1_MOD_LPDIBLB          274

#define BSIM3v1_MOD_LPRWG            275
#define BSIM3v1_MOD_LPRWB            276

#define BSIM3v1_MOD_LCDSCD           277
#define BSIM3v1_MOD_LAGS            278
                                    

#define BSIM3v1_MOD_LFRINGE          281
#define BSIM3v1_MOD_LELM             282
#define BSIM3v1_MOD_LCGSL            283
#define BSIM3v1_MOD_LCGDL            284
#define BSIM3v1_MOD_LCKAPPA          285
#define BSIM3v1_MOD_LCF              286
#define BSIM3v1_MOD_LCLC             287
#define BSIM3v1_MOD_LCLE             288
#define BSIM3v1_MOD_LVFBCV             289

/* Width dependence */
#define BSIM3v1_MOD_WCDSC            301
#define BSIM3v1_MOD_WCDSCB           302
#define BSIM3v1_MOD_WCIT             303
#define BSIM3v1_MOD_WNFACTOR         304
#define BSIM3v1_MOD_WXJ              305
#define BSIM3v1_MOD_WVSAT            306
#define BSIM3v1_MOD_WAT              307
#define BSIM3v1_MOD_WA0              308
#define BSIM3v1_MOD_WA1              309
#define BSIM3v1_MOD_WA2              310
#define BSIM3v1_MOD_WKETA            311   
#define BSIM3v1_MOD_WNSUB            312
#define BSIM3v1_MOD_WNPEAK           313
#define BSIM3v1_MOD_WNGATE           315
#define BSIM3v1_MOD_WGAMMA1          316
#define BSIM3v1_MOD_WGAMMA2          317
#define BSIM3v1_MOD_WVBX             318

#define BSIM3v1_MOD_WVBM             320

#define BSIM3v1_MOD_WXT              322
#define BSIM3v1_MOD_WK1              325
#define BSIM3v1_MOD_WKT1             326
#define BSIM3v1_MOD_WKT1L            327
#define BSIM3v1_MOD_WK2              328
#define BSIM3v1_MOD_WKT2             329
#define BSIM3v1_MOD_WK3              330
#define BSIM3v1_MOD_WK3B             331
#define BSIM3v1_MOD_WW0              332
#define BSIM3v1_MOD_WNLX             333

#define BSIM3v1_MOD_WDVT0            334
#define BSIM3v1_MOD_WDVT1            335
#define BSIM3v1_MOD_WDVT2            336

#define BSIM3v1_MOD_WDVT0W           337
#define BSIM3v1_MOD_WDVT1W           338
#define BSIM3v1_MOD_WDVT2W           339

#define BSIM3v1_MOD_WDROUT           340
#define BSIM3v1_MOD_WDSUB            341
#define BSIM3v1_MOD_WVTH0            342
#define BSIM3v1_MOD_WUA              343
#define BSIM3v1_MOD_WUA1             344
#define BSIM3v1_MOD_WUB              345
#define BSIM3v1_MOD_WUB1             346
#define BSIM3v1_MOD_WUC              347
#define BSIM3v1_MOD_WUC1             348
#define BSIM3v1_MOD_WU0              349
#define BSIM3v1_MOD_WUTE             350
#define BSIM3v1_MOD_WVOFF            351
#define BSIM3v1_MOD_WDELTA           352
#define BSIM3v1_MOD_WRDSW            353
#define BSIM3v1_MOD_WPRT             354
#define BSIM3v1_MOD_WLDD             355
#define BSIM3v1_MOD_WETA             356
#define BSIM3v1_MOD_WETA0            357
#define BSIM3v1_MOD_WETAB            358
#define BSIM3v1_MOD_WPCLM            359
#define BSIM3v1_MOD_WPDIBL1          360
#define BSIM3v1_MOD_WPDIBL2          361
#define BSIM3v1_MOD_WPSCBE1          362
#define BSIM3v1_MOD_WPSCBE2          363
#define BSIM3v1_MOD_WPVAG            364
#define BSIM3v1_MOD_WWR              365
#define BSIM3v1_MOD_WDWG             366
#define BSIM3v1_MOD_WDWB             367
#define BSIM3v1_MOD_WB0              368
#define BSIM3v1_MOD_WB1              369
#define BSIM3v1_MOD_WALPHA0          370
#define BSIM3v1_MOD_WBETA0           371
#define BSIM3v1_MOD_WPDIBLB          374

#define BSIM3v1_MOD_WPRWG            375
#define BSIM3v1_MOD_WPRWB            376

#define BSIM3v1_MOD_WCDSCD           377
#define BSIM3v1_MOD_WAGS             378


#define BSIM3v1_MOD_WFRINGE          381
#define BSIM3v1_MOD_WELM             382
#define BSIM3v1_MOD_WCGSL            383
#define BSIM3v1_MOD_WCGDL            384
#define BSIM3v1_MOD_WCKAPPA          385
#define BSIM3v1_MOD_WCF              386
#define BSIM3v1_MOD_WCLC             387
#define BSIM3v1_MOD_WCLE             388
#define BSIM3v1_MOD_WVFBCV             389

/* Cross-term dependence */
#define BSIM3v1_MOD_PCDSC            401
#define BSIM3v1_MOD_PCDSCB           402
#define BSIM3v1_MOD_PCIT             403
#define BSIM3v1_MOD_PNFACTOR         404
#define BSIM3v1_MOD_PXJ              405
#define BSIM3v1_MOD_PVSAT            406
#define BSIM3v1_MOD_PAT              407
#define BSIM3v1_MOD_PA0              408
#define BSIM3v1_MOD_PA1              409
#define BSIM3v1_MOD_PA2              410
#define BSIM3v1_MOD_PKETA            411   
#define BSIM3v1_MOD_PNSUB            412
#define BSIM3v1_MOD_PNPEAK           413
#define BSIM3v1_MOD_PNGATE           415
#define BSIM3v1_MOD_PGAMMA1          416
#define BSIM3v1_MOD_PGAMMA2          417
#define BSIM3v1_MOD_PVBX             418

#define BSIM3v1_MOD_PVBM             420

#define BSIM3v1_MOD_PXT              422
#define BSIM3v1_MOD_PK1              425
#define BSIM3v1_MOD_PKT1             426
#define BSIM3v1_MOD_PKT1L            427
#define BSIM3v1_MOD_PK2              428
#define BSIM3v1_MOD_PKT2             429
#define BSIM3v1_MOD_PK3              430
#define BSIM3v1_MOD_PK3B             431
#define BSIM3v1_MOD_PW0              432
#define BSIM3v1_MOD_PNLX             433

#define BSIM3v1_MOD_PDVT0            434
#define BSIM3v1_MOD_PDVT1            435
#define BSIM3v1_MOD_PDVT2            436

#define BSIM3v1_MOD_PDVT0W           437
#define BSIM3v1_MOD_PDVT1W           438
#define BSIM3v1_MOD_PDVT2W           439

#define BSIM3v1_MOD_PDROUT           440
#define BSIM3v1_MOD_PDSUB            441
#define BSIM3v1_MOD_PVTH0            442
#define BSIM3v1_MOD_PUA              443
#define BSIM3v1_MOD_PUA1             444
#define BSIM3v1_MOD_PUB              445
#define BSIM3v1_MOD_PUB1             446
#define BSIM3v1_MOD_PUC              447
#define BSIM3v1_MOD_PUC1             448
#define BSIM3v1_MOD_PU0              449
#define BSIM3v1_MOD_PUTE             450
#define BSIM3v1_MOD_PVOFF            451
#define BSIM3v1_MOD_PDELTA           452
#define BSIM3v1_MOD_PRDSW            453
#define BSIM3v1_MOD_PPRT             454
#define BSIM3v1_MOD_PLDD             455
#define BSIM3v1_MOD_PETA             456
#define BSIM3v1_MOD_PETA0            457
#define BSIM3v1_MOD_PETAB            458
#define BSIM3v1_MOD_PPCLM            459
#define BSIM3v1_MOD_PPDIBL1          460
#define BSIM3v1_MOD_PPDIBL2          461
#define BSIM3v1_MOD_PPSCBE1          462
#define BSIM3v1_MOD_PPSCBE2          463
#define BSIM3v1_MOD_PPVAG            464
#define BSIM3v1_MOD_PWR              465
#define BSIM3v1_MOD_PDWG             466
#define BSIM3v1_MOD_PDWB             467
#define BSIM3v1_MOD_PB0              468
#define BSIM3v1_MOD_PB1              469
#define BSIM3v1_MOD_PALPHA0          470
#define BSIM3v1_MOD_PBETA0           471
#define BSIM3v1_MOD_PPDIBLB          474

#define BSIM3v1_MOD_PPRWG            475
#define BSIM3v1_MOD_PPRWB            476

#define BSIM3v1_MOD_PCDSCD           477
#define BSIM3v1_MOD_PAGS             478

#define BSIM3v1_MOD_PFRINGE          481
#define BSIM3v1_MOD_PELM             482
#define BSIM3v1_MOD_PCGSL            483
#define BSIM3v1_MOD_PCGDL            484
#define BSIM3v1_MOD_PCKAPPA          485
#define BSIM3v1_MOD_PCF              486
#define BSIM3v1_MOD_PCLC             487
#define BSIM3v1_MOD_PCLE             488
#define BSIM3v1_MOD_PVFBCV           489

#define BSIM3v1_MOD_TNOM             501
#define BSIM3v1_MOD_CGSO             502
#define BSIM3v1_MOD_CGDO             503
#define BSIM3v1_MOD_CGBO             504
#define BSIM3v1_MOD_XPART            505

#define BSIM3v1_MOD_RSH              506
#define BSIM3v1_MOD_JS               507
#define BSIM3v1_MOD_PB               508
#define BSIM3v1_MOD_MJ               509
#define BSIM3v1_MOD_PBSW             510
#define BSIM3v1_MOD_MJSW             511
#define BSIM3v1_MOD_CJ               512
#define BSIM3v1_MOD_CJSW             513
#define BSIM3v1_MOD_NMOS             514
#define BSIM3v1_MOD_PMOS             515

#define BSIM3v1_MOD_NOIA             516
#define BSIM3v1_MOD_NOIB             517
#define BSIM3v1_MOD_NOIC             518

#define BSIM3v1_MOD_LINT             519
#define BSIM3v1_MOD_LL               520
#define BSIM3v1_MOD_LLN              521
#define BSIM3v1_MOD_LW               522
#define BSIM3v1_MOD_LWN              523
#define BSIM3v1_MOD_LWL              524
#define BSIM3v1_MOD_LMIN             525
#define BSIM3v1_MOD_LMAX             526

#define BSIM3v1_MOD_WINT             527
#define BSIM3v1_MOD_WL               528
#define BSIM3v1_MOD_WLN              529
#define BSIM3v1_MOD_WW               530
#define BSIM3v1_MOD_WWN              531
#define BSIM3v1_MOD_WWL              532
#define BSIM3v1_MOD_WMIN             533
#define BSIM3v1_MOD_WMAX             534

#define BSIM3v1_MOD_DWC              535
#define BSIM3v1_MOD_DLC              536

#define BSIM3v1_MOD_EM               537
#define BSIM3v1_MOD_EF               538
#define BSIM3v1_MOD_AF               539
#define BSIM3v1_MOD_KF               540

#define BSIM3v1_MOD_NJ               541
#define BSIM3v1_MOD_XTI              542

#define BSIM3v1_MOD_PBSWG            543
#define BSIM3v1_MOD_MJSWG            544
#define BSIM3v1_MOD_CJSWG            545
#define BSIM3v1_MOD_JSW              546

/* device questions */
#define BSIM3v1_DNODE                601
#define BSIM3v1_GNODE                602
#define BSIM3v1_SNODE                603
#define BSIM3v1_BNODE                604
#define BSIM3v1_DNODEPRIME           605
#define BSIM3v1_SNODEPRIME           606
#define BSIM3v1_VBD                  607
#define BSIM3v1_VBS                  608
#define BSIM3v1_VGS                  609
#define BSIM3v1_VDS                  610
#define BSIM3v1_CD                   611
#define BSIM3v1_CBS                  612
#define BSIM3v1_CBD                  613
#define BSIM3v1_GM                   614
#define BSIM3v1_GDS                  615
#define BSIM3v1_GMBS                 616
#define BSIM3v1_GBD                  617
#define BSIM3v1_GBS                  618
#define BSIM3v1_QB                   619
#define BSIM3v1_CQB                  620
#define BSIM3v1_QG                   621
#define BSIM3v1_CQG                  622
#define BSIM3v1_QD                   623
#define BSIM3v1_CQD                  624
#define BSIM3v1_CGG                  625
#define BSIM3v1_CGD                  626
#define BSIM3v1_CGS                  627
#define BSIM3v1_CBG                  628
#define BSIM3v1_CAPBD                629
#define BSIM3v1_CQBD                 630
#define BSIM3v1_CAPBS                631
#define BSIM3v1_CQBS                 632
#define BSIM3v1_CDG                  633
#define BSIM3v1_CDD                  634
#define BSIM3v1_CDS                  635
#define BSIM3v1_VON                  636
#define BSIM3v1_VDSAT                637
#define BSIM3v1_QBS                  638
#define BSIM3v1_QBD                  639
#define BSIM3v1_SOURCECONDUCT        640
#define BSIM3v1_DRAINCONDUCT         641
#define BSIM3v1_CBDB                 642
#define BSIM3v1_CBSB                 643

#include "bsim3v1ext.h"

extern void BSIM3v1evaluate(double,double,double,BSIM3v1instance*,BSIM3v1model*,
        double*,double*,double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, CKTcircuit*);
extern int BSIM3v1debug(BSIM3v1model*, BSIM3v1instance*, CKTcircuit*, int);
extern int BSIM3v1checkModel(BSIM3v1model*, BSIM3v1instance*, CKTcircuit*);

#endif /*BSIM3v1*/




