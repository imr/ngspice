/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Weidong Liu (1997-1998).
File: bsim3v2def.h
**********/

#ifndef BSIM3V2
#define BSIM3V2

#include "ifsim.h"
#include "gendefs.h"
#include "cktdefs.h"
#include "complex.h"
#include "noisedef.h"         

typedef struct sBSIM3V2instance
{
    struct sBSIM3V2model *BSIM3V2modPtr;
    struct sBSIM3V2instance *BSIM3V2nextInstance;
    IFuid BSIM3V2name;
    int BSIM3V2owner;    /* number of owner process */
    int BSIM3V2states;  /* index into state table for this device */

    int BSIM3V2dNode;
    int BSIM3V2gNode;
    int BSIM3V2sNode;
    int BSIM3V2bNode;
    int BSIM3V2dNodePrime;
    int BSIM3V2sNodePrime;
    int BSIM3V2qNode; /* MCJ */

    /* MCJ */
    double BSIM3V2ueff;
    double BSIM3V2thetavth; 
    double BSIM3V2von;
    double BSIM3V2vdsat;
    double BSIM3V2cgdo;
    double BSIM3V2cgso;
    double BSIM3V2vjsm;
    double BSIM3V2vjdm;

    double BSIM3V2l;
    double BSIM3V2w;
    double BSIM3V2drainArea;
    double BSIM3V2sourceArea;
    double BSIM3V2drainSquares;
    double BSIM3V2sourceSquares;
    double BSIM3V2drainPerimeter;
    double BSIM3V2sourcePerimeter;
    double BSIM3V2sourceConductance;
    double BSIM3V2drainConductance;

    double BSIM3V2icVBS;
    double BSIM3V2icVDS;
    double BSIM3V2icVGS;
    int BSIM3V2off;
    int BSIM3V2mode;
    int BSIM3V2nqsMod;

    /* OP point */
    double BSIM3V2qinv;
    double BSIM3V2cd;
    double BSIM3V2cbs;
    double BSIM3V2cbd;
    double BSIM3V2csub;
    double BSIM3V2gm;
    double BSIM3V2gds;
    double BSIM3V2gmbs;
    double BSIM3V2gbd;
    double BSIM3V2gbs;

    double BSIM3V2gbbs;
    double BSIM3V2gbgs;
    double BSIM3V2gbds;

    double BSIM3V2cggb;
    double BSIM3V2cgdb;
    double BSIM3V2cgsb;
    double BSIM3V2cbgb;
    double BSIM3V2cbdb;
    double BSIM3V2cbsb;
    double BSIM3V2cdgb;
    double BSIM3V2cddb;
    double BSIM3V2cdsb;
    double BSIM3V2capbd;
    double BSIM3V2capbs;

    double BSIM3V2cqgb;
    double BSIM3V2cqdb;
    double BSIM3V2cqsb;
    double BSIM3V2cqbb;

    double BSIM3V2qgate;
    double BSIM3V2qbulk;
    double BSIM3V2qdrn;

    double BSIM3V2gtau;
    double BSIM3V2gtg;
    double BSIM3V2gtd;
    double BSIM3V2gts;
    double BSIM3V2gtb;

    struct BSIM3V2SizeDependParam  *pParam;

    unsigned BSIM3V2lGiven :1;
    unsigned BSIM3V2wGiven :1;
    unsigned BSIM3V2drainAreaGiven :1;
    unsigned BSIM3V2sourceAreaGiven    :1;
    unsigned BSIM3V2drainSquaresGiven  :1;
    unsigned BSIM3V2sourceSquaresGiven :1;
    unsigned BSIM3V2drainPerimeterGiven    :1;
    unsigned BSIM3V2sourcePerimeterGiven   :1;
    unsigned BSIM3V2dNodePrimeSet  :1;
    unsigned BSIM3V2sNodePrimeSet  :1;
    unsigned BSIM3V2icVBSGiven :1;
    unsigned BSIM3V2icVDSGiven :1;
    unsigned BSIM3V2icVGSGiven :1;
    unsigned BSIM3V2nqsModGiven :1;

    double *BSIM3V2DdPtr;
    double *BSIM3V2GgPtr;
    double *BSIM3V2SsPtr;
    double *BSIM3V2BbPtr;
    double *BSIM3V2DPdpPtr;
    double *BSIM3V2SPspPtr;
    double *BSIM3V2DdpPtr;
    double *BSIM3V2GbPtr;
    double *BSIM3V2GdpPtr;
    double *BSIM3V2GspPtr;
    double *BSIM3V2SspPtr;
    double *BSIM3V2BdpPtr;
    double *BSIM3V2BspPtr;
    double *BSIM3V2DPspPtr;
    double *BSIM3V2DPdPtr;
    double *BSIM3V2BgPtr;
    double *BSIM3V2DPgPtr;
    double *BSIM3V2SPgPtr;
    double *BSIM3V2SPsPtr;
    double *BSIM3V2DPbPtr;
    double *BSIM3V2SPbPtr;
    double *BSIM3V2SPdpPtr;

    double *BSIM3V2QqPtr;
    double *BSIM3V2QdpPtr;
    double *BSIM3V2QgPtr;
    double *BSIM3V2QspPtr;
    double *BSIM3V2QbPtr;
    double *BSIM3V2DPqPtr;
    double *BSIM3V2GqPtr;
    double *BSIM3V2SPqPtr;
    double *BSIM3V2BqPtr;

  /* int BSIM3V2states; */     /* index into state table for this device */
#define BSIM3V2vbd BSIM3V2states+ 0
#define BSIM3V2vbs BSIM3V2states+ 1
#define BSIM3V2vgs BSIM3V2states+ 2
#define BSIM3V2vds BSIM3V2states+ 3

#define BSIM3V2qb BSIM3V2states+ 4
#define BSIM3V2cqb BSIM3V2states+ 5
#define BSIM3V2qg BSIM3V2states+ 6
#define BSIM3V2cqg BSIM3V2states+ 7
#define BSIM3V2qd BSIM3V2states+ 8
#define BSIM3V2cqd BSIM3V2states+ 9

#define BSIM3V2qbs  BSIM3V2states+ 10
#define BSIM3V2qbd  BSIM3V2states+ 11

#define BSIM3V2qcheq BSIM3V2states+ 12
#define BSIM3V2cqcheq BSIM3V2states+ 13
#define BSIM3V2qcdump BSIM3V2states+ 14
#define BSIM3V2cqcdump BSIM3V2states+ 15

#define BSIM3V2qdef BSIM3V2states+ 16

#define BSIM3V2numStates 17


/* indices to the array of BSIM3V2 NOISE SOURCES */

#define BSIM3V2RDNOIZ       0
#define BSIM3V2RSNOIZ       1
#define BSIM3V2IDNOIZ       2
#define BSIM3V2FLNOIZ       3
#define BSIM3V2TOTNOIZ      4

#define BSIM3V2NSRCS        5  /* the number of BSIM3V2 MOSFET noise sources */

#ifndef NONOISE
    double BSIM3V2nVar[NSTATVARS][BSIM3V2NSRCS];
#else /* NONOISE */
        double **BSIM3V2nVar;
#endif /* NONOISE */

} BSIM3V2instance ;

struct BSIM3V2SizeDependParam
{
    double Width;
    double Length;

    double BSIM3V2cdsc;           
    double BSIM3V2cdscb;    
    double BSIM3V2cdscd;       
    double BSIM3V2cit;           
    double BSIM3V2nfactor;      
    double BSIM3V2xj;
    double BSIM3V2vsat;         
    double BSIM3V2at;         
    double BSIM3V2a0;   
    double BSIM3V2ags;      
    double BSIM3V2a1;         
    double BSIM3V2a2;         
    double BSIM3V2keta;     
    double BSIM3V2nsub;
    double BSIM3V2npeak;        
    double BSIM3V2ngate;        
    double BSIM3V2gamma1;      
    double BSIM3V2gamma2;     
    double BSIM3V2vbx;      
    double BSIM3V2vbi;       
    double BSIM3V2vbm;       
    double BSIM3V2vbsc;       
    double BSIM3V2xt;       
    double BSIM3V2phi;
    double BSIM3V2litl;
    double BSIM3V2k1;
    double BSIM3V2kt1;
    double BSIM3V2kt1l;
    double BSIM3V2kt2;
    double BSIM3V2k2;
    double BSIM3V2k3;
    double BSIM3V2k3b;
    double BSIM3V2w0;
    double BSIM3V2nlx;
    double BSIM3V2dvt0;      
    double BSIM3V2dvt1;      
    double BSIM3V2dvt2;      
    double BSIM3V2dvt0w;      
    double BSIM3V2dvt1w;      
    double BSIM3V2dvt2w;      
    double BSIM3V2drout;      
    double BSIM3V2dsub;      
    double BSIM3V2vth0;
    double BSIM3V2ua;
    double BSIM3V2ua1;
    double BSIM3V2ub;
    double BSIM3V2ub1;
    double BSIM3V2uc;
    double BSIM3V2uc1;
    double BSIM3V2u0;
    double BSIM3V2ute;
    double BSIM3V2voff;
    double BSIM3V2vfb;
    double BSIM3V2delta;
    double BSIM3V2rdsw;       
    double BSIM3V2rds0;       
    double BSIM3V2prwg;       
    double BSIM3V2prwb;       
    double BSIM3V2prt;       
    double BSIM3V2eta0;         
    double BSIM3V2etab;         
    double BSIM3V2pclm;      
    double BSIM3V2pdibl1;      
    double BSIM3V2pdibl2;      
    double BSIM3V2pdiblb;      
    double BSIM3V2pscbe1;       
    double BSIM3V2pscbe2;       
    double BSIM3V2pvag;       
    double BSIM3V2wr;
    double BSIM3V2dwg;
    double BSIM3V2dwb;
    double BSIM3V2b0;
    double BSIM3V2b1;
    double BSIM3V2alpha0;
    double BSIM3V2alpha1;
    double BSIM3V2beta0;


    /* CV model */
    double BSIM3V2elm;
    double BSIM3V2cgsl;
    double BSIM3V2cgdl;
    double BSIM3V2ckappa;
    double BSIM3V2cf;
    double BSIM3V2clc;
    double BSIM3V2cle;
    double BSIM3V2vfbcv;
    double BSIM3V2noff;
    double BSIM3V2voffcv;
    double BSIM3V2acde;
    double BSIM3V2moin;


/* Pre-calculated constants */

    double BSIM3V2dw;
    double BSIM3V2dl;
    double BSIM3V2leff;
    double BSIM3V2weff;

    double BSIM3V2dwc;
    double BSIM3V2dlc;
    double BSIM3V2leffCV;
    double BSIM3V2weffCV;
    double BSIM3V2abulkCVfactor;
    double BSIM3V2cgso;
    double BSIM3V2cgdo;
    double BSIM3V2cgbo;
    double BSIM3V2tconst;

    double BSIM3V2u0temp;       
    double BSIM3V2vsattemp;   
    double BSIM3V2sqrtPhi;   
    double BSIM3V2phis3;   
    double BSIM3V2Xdep0;          
    double BSIM3V2sqrtXdep0;          
    double BSIM3V2theta0vb0;
    double BSIM3V2thetaRout; 

    double BSIM3V2cof1;
    double BSIM3V2cof2;
    double BSIM3V2cof3;
    double BSIM3V2cof4;
    double BSIM3V2cdep0;
    double BSIM3V2vfbzb;
    double BSIM3V2ldeb;
    double BSIM3V2k1ox;
    double BSIM3V2k2ox;

    struct BSIM3V2SizeDependParam  *pNext;
};


typedef struct sBSIM3V2model 
{
    int BSIM3V2modType;
    struct sBSIM3V2model *BSIM3V2nextModel;
    BSIM3V2instance *BSIM3V2instances;
    IFuid BSIM3V2modName; 
    int BSIM3V2type;

    int    BSIM3V2mobMod;
    int    BSIM3V2capMod;
    int    BSIM3V2noiMod;
    int    BSIM3V2binUnit;
    int    BSIM3V2paramChk;
    double BSIM3V2version;             
    double BSIM3V2tox;             
    double BSIM3V2toxm;
    double BSIM3V2cdsc;           
    double BSIM3V2cdscb; 
    double BSIM3V2cdscd;          
    double BSIM3V2cit;           
    double BSIM3V2nfactor;      
    double BSIM3V2xj;
    double BSIM3V2vsat;         
    double BSIM3V2at;         
    double BSIM3V2a0;   
    double BSIM3V2ags;      
    double BSIM3V2a1;         
    double BSIM3V2a2;         
    double BSIM3V2keta;     
    double BSIM3V2nsub;
    double BSIM3V2npeak;        
    double BSIM3V2ngate;        
    double BSIM3V2gamma1;      
    double BSIM3V2gamma2;     
    double BSIM3V2vbx;      
    double BSIM3V2vbm;       
    double BSIM3V2xt;       
    double BSIM3V2k1;
    double BSIM3V2kt1;
    double BSIM3V2kt1l;
    double BSIM3V2kt2;
    double BSIM3V2k2;
    double BSIM3V2k3;
    double BSIM3V2k3b;
    double BSIM3V2w0;
    double BSIM3V2nlx;
    double BSIM3V2dvt0;      
    double BSIM3V2dvt1;      
    double BSIM3V2dvt2;      
    double BSIM3V2dvt0w;      
    double BSIM3V2dvt1w;      
    double BSIM3V2dvt2w;      
    double BSIM3V2drout;      
    double BSIM3V2dsub;      
    double BSIM3V2vth0;
    double BSIM3V2ua;
    double BSIM3V2ua1;
    double BSIM3V2ub;
    double BSIM3V2ub1;
    double BSIM3V2uc;
    double BSIM3V2uc1;
    double BSIM3V2u0;
    double BSIM3V2ute;
    double BSIM3V2voff;
    double BSIM3V2delta;
    double BSIM3V2rdsw;       
    double BSIM3V2prwg;
    double BSIM3V2prwb;
    double BSIM3V2prt;       
    double BSIM3V2eta0;         
    double BSIM3V2etab;         
    double BSIM3V2pclm;      
    double BSIM3V2pdibl1;      
    double BSIM3V2pdibl2;      
    double BSIM3V2pdiblb;
    double BSIM3V2pscbe1;       
    double BSIM3V2pscbe2;       
    double BSIM3V2pvag;       
    double BSIM3V2wr;
    double BSIM3V2dwg;
    double BSIM3V2dwb;
    double BSIM3V2b0;
    double BSIM3V2b1;
    double BSIM3V2alpha0;
    double BSIM3V2alpha1;
    double BSIM3V2beta0;
    double BSIM3V2ijth;
    double BSIM3V2vfb;

    /* CV model */
    double BSIM3V2elm;
    double BSIM3V2cgsl;
    double BSIM3V2cgdl;
    double BSIM3V2ckappa;
    double BSIM3V2cf;
    double BSIM3V2vfbcv;
    double BSIM3V2clc;
    double BSIM3V2cle;
    double BSIM3V2dwc;
    double BSIM3V2dlc;
    double BSIM3V2noff;
    double BSIM3V2voffcv;
    double BSIM3V2acde;
    double BSIM3V2moin;
    double BSIM3V2tcj;
    double BSIM3V2tcjsw;
    double BSIM3V2tcjswg;
    double BSIM3V2tpb;
    double BSIM3V2tpbsw;
    double BSIM3V2tpbswg;

    /* Length Dependence */
    double BSIM3V2lcdsc;           
    double BSIM3V2lcdscb; 
    double BSIM3V2lcdscd;          
    double BSIM3V2lcit;           
    double BSIM3V2lnfactor;      
    double BSIM3V2lxj;
    double BSIM3V2lvsat;         
    double BSIM3V2lat;         
    double BSIM3V2la0;   
    double BSIM3V2lags;      
    double BSIM3V2la1;         
    double BSIM3V2la2;         
    double BSIM3V2lketa;     
    double BSIM3V2lnsub;
    double BSIM3V2lnpeak;        
    double BSIM3V2lngate;        
    double BSIM3V2lgamma1;      
    double BSIM3V2lgamma2;     
    double BSIM3V2lvbx;      
    double BSIM3V2lvbm;       
    double BSIM3V2lxt;       
    double BSIM3V2lk1;
    double BSIM3V2lkt1;
    double BSIM3V2lkt1l;
    double BSIM3V2lkt2;
    double BSIM3V2lk2;
    double BSIM3V2lk3;
    double BSIM3V2lk3b;
    double BSIM3V2lw0;
    double BSIM3V2lnlx;
    double BSIM3V2ldvt0;      
    double BSIM3V2ldvt1;      
    double BSIM3V2ldvt2;      
    double BSIM3V2ldvt0w;      
    double BSIM3V2ldvt1w;      
    double BSIM3V2ldvt2w;      
    double BSIM3V2ldrout;      
    double BSIM3V2ldsub;      
    double BSIM3V2lvth0;
    double BSIM3V2lua;
    double BSIM3V2lua1;
    double BSIM3V2lub;
    double BSIM3V2lub1;
    double BSIM3V2luc;
    double BSIM3V2luc1;
    double BSIM3V2lu0;
    double BSIM3V2lute;
    double BSIM3V2lvoff;
    double BSIM3V2ldelta;
    double BSIM3V2lrdsw;       
    double BSIM3V2lprwg;
    double BSIM3V2lprwb;
    double BSIM3V2lprt;       
    double BSIM3V2leta0;         
    double BSIM3V2letab;         
    double BSIM3V2lpclm;      
    double BSIM3V2lpdibl1;      
    double BSIM3V2lpdibl2;      
    double BSIM3V2lpdiblb;
    double BSIM3V2lpscbe1;       
    double BSIM3V2lpscbe2;       
    double BSIM3V2lpvag;       
    double BSIM3V2lwr;
    double BSIM3V2ldwg;
    double BSIM3V2ldwb;
    double BSIM3V2lb0;
    double BSIM3V2lb1;
    double BSIM3V2lalpha0;
    double BSIM3V2lalpha1;
    double BSIM3V2lbeta0;
    double BSIM3V2lvfb;

    /* CV model */
    double BSIM3V2lelm;
    double BSIM3V2lcgsl;
    double BSIM3V2lcgdl;
    double BSIM3V2lckappa;
    double BSIM3V2lcf;
    double BSIM3V2lclc;
    double BSIM3V2lcle;
    double BSIM3V2lvfbcv;
    double BSIM3V2lnoff;
    double BSIM3V2lvoffcv;
    double BSIM3V2lacde;
    double BSIM3V2lmoin;

    /* Width Dependence */
    double BSIM3V2wcdsc;           
    double BSIM3V2wcdscb; 
    double BSIM3V2wcdscd;          
    double BSIM3V2wcit;           
    double BSIM3V2wnfactor;      
    double BSIM3V2wxj;
    double BSIM3V2wvsat;         
    double BSIM3V2wat;         
    double BSIM3V2wa0;   
    double BSIM3V2wags;      
    double BSIM3V2wa1;         
    double BSIM3V2wa2;         
    double BSIM3V2wketa;     
    double BSIM3V2wnsub;
    double BSIM3V2wnpeak;        
    double BSIM3V2wngate;        
    double BSIM3V2wgamma1;      
    double BSIM3V2wgamma2;     
    double BSIM3V2wvbx;      
    double BSIM3V2wvbm;       
    double BSIM3V2wxt;       
    double BSIM3V2wk1;
    double BSIM3V2wkt1;
    double BSIM3V2wkt1l;
    double BSIM3V2wkt2;
    double BSIM3V2wk2;
    double BSIM3V2wk3;
    double BSIM3V2wk3b;
    double BSIM3V2ww0;
    double BSIM3V2wnlx;
    double BSIM3V2wdvt0;      
    double BSIM3V2wdvt1;      
    double BSIM3V2wdvt2;      
    double BSIM3V2wdvt0w;      
    double BSIM3V2wdvt1w;      
    double BSIM3V2wdvt2w;      
    double BSIM3V2wdrout;      
    double BSIM3V2wdsub;      
    double BSIM3V2wvth0;
    double BSIM3V2wua;
    double BSIM3V2wua1;
    double BSIM3V2wub;
    double BSIM3V2wub1;
    double BSIM3V2wuc;
    double BSIM3V2wuc1;
    double BSIM3V2wu0;
    double BSIM3V2wute;
    double BSIM3V2wvoff;
    double BSIM3V2wdelta;
    double BSIM3V2wrdsw;       
    double BSIM3V2wprwg;
    double BSIM3V2wprwb;
    double BSIM3V2wprt;       
    double BSIM3V2weta0;         
    double BSIM3V2wetab;         
    double BSIM3V2wpclm;      
    double BSIM3V2wpdibl1;      
    double BSIM3V2wpdibl2;      
    double BSIM3V2wpdiblb;
    double BSIM3V2wpscbe1;       
    double BSIM3V2wpscbe2;       
    double BSIM3V2wpvag;       
    double BSIM3V2wwr;
    double BSIM3V2wdwg;
    double BSIM3V2wdwb;
    double BSIM3V2wb0;
    double BSIM3V2wb1;
    double BSIM3V2walpha0;
    double BSIM3V2walpha1;
    double BSIM3V2wbeta0;
    double BSIM3V2wvfb;

    /* CV model */
    double BSIM3V2welm;
    double BSIM3V2wcgsl;
    double BSIM3V2wcgdl;
    double BSIM3V2wckappa;
    double BSIM3V2wcf;
    double BSIM3V2wclc;
    double BSIM3V2wcle;
    double BSIM3V2wvfbcv;
    double BSIM3V2wnoff;
    double BSIM3V2wvoffcv;
    double BSIM3V2wacde;
    double BSIM3V2wmoin;

    /* Cross-term Dependence */
    double BSIM3V2pcdsc;           
    double BSIM3V2pcdscb; 
    double BSIM3V2pcdscd;          
    double BSIM3V2pcit;           
    double BSIM3V2pnfactor;      
    double BSIM3V2pxj;
    double BSIM3V2pvsat;         
    double BSIM3V2pat;         
    double BSIM3V2pa0;   
    double BSIM3V2pags;      
    double BSIM3V2pa1;         
    double BSIM3V2pa2;         
    double BSIM3V2pketa;     
    double BSIM3V2pnsub;
    double BSIM3V2pnpeak;        
    double BSIM3V2pngate;        
    double BSIM3V2pgamma1;      
    double BSIM3V2pgamma2;     
    double BSIM3V2pvbx;      
    double BSIM3V2pvbm;       
    double BSIM3V2pxt;       
    double BSIM3V2pk1;
    double BSIM3V2pkt1;
    double BSIM3V2pkt1l;
    double BSIM3V2pkt2;
    double BSIM3V2pk2;
    double BSIM3V2pk3;
    double BSIM3V2pk3b;
    double BSIM3V2pw0;
    double BSIM3V2pnlx;
    double BSIM3V2pdvt0;      
    double BSIM3V2pdvt1;      
    double BSIM3V2pdvt2;      
    double BSIM3V2pdvt0w;      
    double BSIM3V2pdvt1w;      
    double BSIM3V2pdvt2w;      
    double BSIM3V2pdrout;      
    double BSIM3V2pdsub;      
    double BSIM3V2pvth0;
    double BSIM3V2pua;
    double BSIM3V2pua1;
    double BSIM3V2pub;
    double BSIM3V2pub1;
    double BSIM3V2puc;
    double BSIM3V2puc1;
    double BSIM3V2pu0;
    double BSIM3V2pute;
    double BSIM3V2pvoff;
    double BSIM3V2pdelta;
    double BSIM3V2prdsw;
    double BSIM3V2pprwg;
    double BSIM3V2pprwb;
    double BSIM3V2pprt;       
    double BSIM3V2peta0;         
    double BSIM3V2petab;         
    double BSIM3V2ppclm;      
    double BSIM3V2ppdibl1;      
    double BSIM3V2ppdibl2;      
    double BSIM3V2ppdiblb;
    double BSIM3V2ppscbe1;       
    double BSIM3V2ppscbe2;       
    double BSIM3V2ppvag;       
    double BSIM3V2pwr;
    double BSIM3V2pdwg;
    double BSIM3V2pdwb;
    double BSIM3V2pb0;
    double BSIM3V2pb1;
    double BSIM3V2palpha0;
    double BSIM3V2palpha1;
    double BSIM3V2pbeta0;
    double BSIM3V2pvfb;

    /* CV model */
    double BSIM3V2pelm;
    double BSIM3V2pcgsl;
    double BSIM3V2pcgdl;
    double BSIM3V2pckappa;
    double BSIM3V2pcf;
    double BSIM3V2pclc;
    double BSIM3V2pcle;
    double BSIM3V2pvfbcv;
    double BSIM3V2pnoff;
    double BSIM3V2pvoffcv;
    double BSIM3V2pacde;
    double BSIM3V2pmoin;

    double BSIM3V2tnom;
    double BSIM3V2cgso;
    double BSIM3V2cgdo;
    double BSIM3V2cgbo;
    double BSIM3V2xpart;
    double BSIM3V2cFringOut;
    double BSIM3V2cFringMax;

    double BSIM3V2sheetResistance;
    double BSIM3V2jctSatCurDensity;
    double BSIM3V2jctSidewallSatCurDensity;
    double BSIM3V2bulkJctPotential;
    double BSIM3V2bulkJctBotGradingCoeff;
    double BSIM3V2bulkJctSideGradingCoeff;
    double BSIM3V2bulkJctGateSideGradingCoeff;
    double BSIM3V2sidewallJctPotential;
    double BSIM3V2GatesidewallJctPotential;
    double BSIM3V2unitAreaJctCap;
    double BSIM3V2unitLengthSidewallJctCap;
    double BSIM3V2unitLengthGateSidewallJctCap;
    double BSIM3V2jctEmissionCoeff;
    double BSIM3V2jctTempExponent;

    double BSIM3V2Lint;
    double BSIM3V2Ll;
    double BSIM3V2Llc;
    double BSIM3V2Lln;
    double BSIM3V2Lw;
    double BSIM3V2Lwc;
    double BSIM3V2Lwn;
    double BSIM3V2Lwl;
    double BSIM3V2Lwlc;
    double BSIM3V2Lmin;
    double BSIM3V2Lmax;

    double BSIM3V2Wint;
    double BSIM3V2Wl;
    double BSIM3V2Wlc;
    double BSIM3V2Wln;
    double BSIM3V2Ww;
    double BSIM3V2Wwc;
    double BSIM3V2Wwn;
    double BSIM3V2Wwl;
    double BSIM3V2Wwlc;
    double BSIM3V2Wmin;
    double BSIM3V2Wmax;


/* Pre-calculated constants */
    /* MCJ: move to size-dependent param. */
    double BSIM3V2vtm;   
    double BSIM3V2cox;
    double BSIM3V2cof1;
    double BSIM3V2cof2;
    double BSIM3V2cof3;
    double BSIM3V2cof4;
    double BSIM3V2vcrit;
    double BSIM3V2factor1;
    double BSIM3V2PhiB;
    double BSIM3V2PhiBSW;
    double BSIM3V2PhiBSWG;
    double BSIM3V2jctTempSatCurDensity;
    double BSIM3V2jctSidewallTempSatCurDensity;

    double BSIM3V2oxideTrapDensityA;      
    double BSIM3V2oxideTrapDensityB;     
    double BSIM3V2oxideTrapDensityC;  
    double BSIM3V2em;  
    double BSIM3V2ef;  
    double BSIM3V2af;  
    double BSIM3V2kf;  

    struct BSIM3V2SizeDependParam *pSizeDependParamKnot;

    /* Flags */
    unsigned  BSIM3V2mobModGiven :1;
    unsigned  BSIM3V2binUnitGiven :1;
    unsigned  BSIM3V2capModGiven :1;
    unsigned  BSIM3V2paramChkGiven :1;
    unsigned  BSIM3V2noiModGiven :1;
    unsigned  BSIM3V2typeGiven   :1;
    unsigned  BSIM3V2toxGiven   :1;
    unsigned  BSIM3V2versionGiven   :1;
    unsigned  BSIM3V2toxmGiven   :1;
    unsigned  BSIM3V2cdscGiven   :1;
    unsigned  BSIM3V2cdscbGiven   :1;
    unsigned  BSIM3V2cdscdGiven   :1;
    unsigned  BSIM3V2citGiven   :1;
    unsigned  BSIM3V2nfactorGiven   :1;
    unsigned  BSIM3V2xjGiven   :1;
    unsigned  BSIM3V2vsatGiven   :1;
    unsigned  BSIM3V2atGiven   :1;
    unsigned  BSIM3V2a0Given   :1;
    unsigned  BSIM3V2agsGiven   :1;
    unsigned  BSIM3V2a1Given   :1;
    unsigned  BSIM3V2a2Given   :1;
    unsigned  BSIM3V2ketaGiven   :1;    
    unsigned  BSIM3V2nsubGiven   :1;
    unsigned  BSIM3V2npeakGiven   :1;
    unsigned  BSIM3V2ngateGiven   :1;
    unsigned  BSIM3V2gamma1Given   :1;
    unsigned  BSIM3V2gamma2Given   :1;
    unsigned  BSIM3V2vbxGiven   :1;
    unsigned  BSIM3V2vbmGiven   :1;
    unsigned  BSIM3V2xtGiven   :1;
    unsigned  BSIM3V2k1Given   :1;
    unsigned  BSIM3V2kt1Given   :1;
    unsigned  BSIM3V2kt1lGiven   :1;
    unsigned  BSIM3V2kt2Given   :1;
    unsigned  BSIM3V2k2Given   :1;
    unsigned  BSIM3V2k3Given   :1;
    unsigned  BSIM3V2k3bGiven   :1;
    unsigned  BSIM3V2w0Given   :1;
    unsigned  BSIM3V2nlxGiven   :1;
    unsigned  BSIM3V2dvt0Given   :1;   
    unsigned  BSIM3V2dvt1Given   :1;     
    unsigned  BSIM3V2dvt2Given   :1;     
    unsigned  BSIM3V2dvt0wGiven   :1;   
    unsigned  BSIM3V2dvt1wGiven   :1;     
    unsigned  BSIM3V2dvt2wGiven   :1;     
    unsigned  BSIM3V2droutGiven   :1;     
    unsigned  BSIM3V2dsubGiven   :1;     
    unsigned  BSIM3V2vth0Given   :1;
    unsigned  BSIM3V2uaGiven   :1;
    unsigned  BSIM3V2ua1Given   :1;
    unsigned  BSIM3V2ubGiven   :1;
    unsigned  BSIM3V2ub1Given   :1;
    unsigned  BSIM3V2ucGiven   :1;
    unsigned  BSIM3V2uc1Given   :1;
    unsigned  BSIM3V2u0Given   :1;
    unsigned  BSIM3V2uteGiven   :1;
    unsigned  BSIM3V2voffGiven   :1;
    unsigned  BSIM3V2rdswGiven   :1;      
    unsigned  BSIM3V2prwgGiven   :1;      
    unsigned  BSIM3V2prwbGiven   :1;      
    unsigned  BSIM3V2prtGiven   :1;      
    unsigned  BSIM3V2eta0Given   :1;    
    unsigned  BSIM3V2etabGiven   :1;    
    unsigned  BSIM3V2pclmGiven   :1;   
    unsigned  BSIM3V2pdibl1Given   :1;   
    unsigned  BSIM3V2pdibl2Given   :1;  
    unsigned  BSIM3V2pdiblbGiven   :1;  
    unsigned  BSIM3V2pscbe1Given   :1;    
    unsigned  BSIM3V2pscbe2Given   :1;    
    unsigned  BSIM3V2pvagGiven   :1;    
    unsigned  BSIM3V2deltaGiven  :1;     
    unsigned  BSIM3V2wrGiven   :1;
    unsigned  BSIM3V2dwgGiven   :1;
    unsigned  BSIM3V2dwbGiven   :1;
    unsigned  BSIM3V2b0Given   :1;
    unsigned  BSIM3V2b1Given   :1;
    unsigned  BSIM3V2alpha0Given   :1;
    unsigned  BSIM3V2alpha1Given   :1;
    unsigned  BSIM3V2beta0Given   :1;
    unsigned  BSIM3V2ijthGiven   :1;
    unsigned  BSIM3V2vfbGiven   :1;

    /* CV model */
    unsigned  BSIM3V2elmGiven  :1;     
    unsigned  BSIM3V2cgslGiven   :1;
    unsigned  BSIM3V2cgdlGiven   :1;
    unsigned  BSIM3V2ckappaGiven   :1;
    unsigned  BSIM3V2cfGiven   :1;
    unsigned  BSIM3V2vfbcvGiven   :1;
    unsigned  BSIM3V2clcGiven   :1;
    unsigned  BSIM3V2cleGiven   :1;
    unsigned  BSIM3V2dwcGiven   :1;
    unsigned  BSIM3V2dlcGiven   :1;
    unsigned  BSIM3V2noffGiven  :1;
    unsigned  BSIM3V2voffcvGiven :1;
    unsigned  BSIM3V2acdeGiven  :1;
    unsigned  BSIM3V2moinGiven  :1;
    unsigned  BSIM3V2tcjGiven   :1;
    unsigned  BSIM3V2tcjswGiven :1;
    unsigned  BSIM3V2tcjswgGiven :1;
    unsigned  BSIM3V2tpbGiven    :1;
    unsigned  BSIM3V2tpbswGiven  :1;
    unsigned  BSIM3V2tpbswgGiven :1;


    /* Length dependence */
    unsigned  BSIM3V2lcdscGiven   :1;
    unsigned  BSIM3V2lcdscbGiven   :1;
    unsigned  BSIM3V2lcdscdGiven   :1;
    unsigned  BSIM3V2lcitGiven   :1;
    unsigned  BSIM3V2lnfactorGiven   :1;
    unsigned  BSIM3V2lxjGiven   :1;
    unsigned  BSIM3V2lvsatGiven   :1;
    unsigned  BSIM3V2latGiven   :1;
    unsigned  BSIM3V2la0Given   :1;
    unsigned  BSIM3V2lagsGiven   :1;
    unsigned  BSIM3V2la1Given   :1;
    unsigned  BSIM3V2la2Given   :1;
    unsigned  BSIM3V2lketaGiven   :1;    
    unsigned  BSIM3V2lnsubGiven   :1;
    unsigned  BSIM3V2lnpeakGiven   :1;
    unsigned  BSIM3V2lngateGiven   :1;
    unsigned  BSIM3V2lgamma1Given   :1;
    unsigned  BSIM3V2lgamma2Given   :1;
    unsigned  BSIM3V2lvbxGiven   :1;
    unsigned  BSIM3V2lvbmGiven   :1;
    unsigned  BSIM3V2lxtGiven   :1;
    unsigned  BSIM3V2lk1Given   :1;
    unsigned  BSIM3V2lkt1Given   :1;
    unsigned  BSIM3V2lkt1lGiven   :1;
    unsigned  BSIM3V2lkt2Given   :1;
    unsigned  BSIM3V2lk2Given   :1;
    unsigned  BSIM3V2lk3Given   :1;
    unsigned  BSIM3V2lk3bGiven   :1;
    unsigned  BSIM3V2lw0Given   :1;
    unsigned  BSIM3V2lnlxGiven   :1;
    unsigned  BSIM3V2ldvt0Given   :1;   
    unsigned  BSIM3V2ldvt1Given   :1;     
    unsigned  BSIM3V2ldvt2Given   :1;     
    unsigned  BSIM3V2ldvt0wGiven   :1;   
    unsigned  BSIM3V2ldvt1wGiven   :1;     
    unsigned  BSIM3V2ldvt2wGiven   :1;     
    unsigned  BSIM3V2ldroutGiven   :1;     
    unsigned  BSIM3V2ldsubGiven   :1;     
    unsigned  BSIM3V2lvth0Given   :1;
    unsigned  BSIM3V2luaGiven   :1;
    unsigned  BSIM3V2lua1Given   :1;
    unsigned  BSIM3V2lubGiven   :1;
    unsigned  BSIM3V2lub1Given   :1;
    unsigned  BSIM3V2lucGiven   :1;
    unsigned  BSIM3V2luc1Given   :1;
    unsigned  BSIM3V2lu0Given   :1;
    unsigned  BSIM3V2luteGiven   :1;
    unsigned  BSIM3V2lvoffGiven   :1;
    unsigned  BSIM3V2lrdswGiven   :1;      
    unsigned  BSIM3V2lprwgGiven   :1;      
    unsigned  BSIM3V2lprwbGiven   :1;      
    unsigned  BSIM3V2lprtGiven   :1;      
    unsigned  BSIM3V2leta0Given   :1;    
    unsigned  BSIM3V2letabGiven   :1;    
    unsigned  BSIM3V2lpclmGiven   :1;   
    unsigned  BSIM3V2lpdibl1Given   :1;   
    unsigned  BSIM3V2lpdibl2Given   :1;  
    unsigned  BSIM3V2lpdiblbGiven   :1;  
    unsigned  BSIM3V2lpscbe1Given   :1;    
    unsigned  BSIM3V2lpscbe2Given   :1;    
    unsigned  BSIM3V2lpvagGiven   :1;    
    unsigned  BSIM3V2ldeltaGiven  :1;     
    unsigned  BSIM3V2lwrGiven   :1;
    unsigned  BSIM3V2ldwgGiven   :1;
    unsigned  BSIM3V2ldwbGiven   :1;
    unsigned  BSIM3V2lb0Given   :1;
    unsigned  BSIM3V2lb1Given   :1;
    unsigned  BSIM3V2lalpha0Given   :1;
    unsigned  BSIM3V2lalpha1Given   :1;
    unsigned  BSIM3V2lbeta0Given   :1;
    unsigned  BSIM3V2lvfbGiven   :1;

    /* CV model */
    unsigned  BSIM3V2lelmGiven  :1;     
    unsigned  BSIM3V2lcgslGiven   :1;
    unsigned  BSIM3V2lcgdlGiven   :1;
    unsigned  BSIM3V2lckappaGiven   :1;
    unsigned  BSIM3V2lcfGiven   :1;
    unsigned  BSIM3V2lclcGiven   :1;
    unsigned  BSIM3V2lcleGiven   :1;
    unsigned  BSIM3V2lvfbcvGiven   :1;
    unsigned  BSIM3V2lnoffGiven   :1;
    unsigned  BSIM3V2lvoffcvGiven :1;
    unsigned  BSIM3V2lacdeGiven   :1;
    unsigned  BSIM3V2lmoinGiven   :1;

    /* Width dependence */
    unsigned  BSIM3V2wcdscGiven   :1;
    unsigned  BSIM3V2wcdscbGiven   :1;
    unsigned  BSIM3V2wcdscdGiven   :1;
    unsigned  BSIM3V2wcitGiven   :1;
    unsigned  BSIM3V2wnfactorGiven   :1;
    unsigned  BSIM3V2wxjGiven   :1;
    unsigned  BSIM3V2wvsatGiven   :1;
    unsigned  BSIM3V2watGiven   :1;
    unsigned  BSIM3V2wa0Given   :1;
    unsigned  BSIM3V2wagsGiven   :1;
    unsigned  BSIM3V2wa1Given   :1;
    unsigned  BSIM3V2wa2Given   :1;
    unsigned  BSIM3V2wketaGiven   :1;    
    unsigned  BSIM3V2wnsubGiven   :1;
    unsigned  BSIM3V2wnpeakGiven   :1;
    unsigned  BSIM3V2wngateGiven   :1;
    unsigned  BSIM3V2wgamma1Given   :1;
    unsigned  BSIM3V2wgamma2Given   :1;
    unsigned  BSIM3V2wvbxGiven   :1;
    unsigned  BSIM3V2wvbmGiven   :1;
    unsigned  BSIM3V2wxtGiven   :1;
    unsigned  BSIM3V2wk1Given   :1;
    unsigned  BSIM3V2wkt1Given   :1;
    unsigned  BSIM3V2wkt1lGiven   :1;
    unsigned  BSIM3V2wkt2Given   :1;
    unsigned  BSIM3V2wk2Given   :1;
    unsigned  BSIM3V2wk3Given   :1;
    unsigned  BSIM3V2wk3bGiven   :1;
    unsigned  BSIM3V2ww0Given   :1;
    unsigned  BSIM3V2wnlxGiven   :1;
    unsigned  BSIM3V2wdvt0Given   :1;   
    unsigned  BSIM3V2wdvt1Given   :1;     
    unsigned  BSIM3V2wdvt2Given   :1;     
    unsigned  BSIM3V2wdvt0wGiven   :1;   
    unsigned  BSIM3V2wdvt1wGiven   :1;     
    unsigned  BSIM3V2wdvt2wGiven   :1;     
    unsigned  BSIM3V2wdroutGiven   :1;     
    unsigned  BSIM3V2wdsubGiven   :1;     
    unsigned  BSIM3V2wvth0Given   :1;
    unsigned  BSIM3V2wuaGiven   :1;
    unsigned  BSIM3V2wua1Given   :1;
    unsigned  BSIM3V2wubGiven   :1;
    unsigned  BSIM3V2wub1Given   :1;
    unsigned  BSIM3V2wucGiven   :1;
    unsigned  BSIM3V2wuc1Given   :1;
    unsigned  BSIM3V2wu0Given   :1;
    unsigned  BSIM3V2wuteGiven   :1;
    unsigned  BSIM3V2wvoffGiven   :1;
    unsigned  BSIM3V2wrdswGiven   :1;      
    unsigned  BSIM3V2wprwgGiven   :1;      
    unsigned  BSIM3V2wprwbGiven   :1;      
    unsigned  BSIM3V2wprtGiven   :1;      
    unsigned  BSIM3V2weta0Given   :1;    
    unsigned  BSIM3V2wetabGiven   :1;    
    unsigned  BSIM3V2wpclmGiven   :1;   
    unsigned  BSIM3V2wpdibl1Given   :1;   
    unsigned  BSIM3V2wpdibl2Given   :1;  
    unsigned  BSIM3V2wpdiblbGiven   :1;  
    unsigned  BSIM3V2wpscbe1Given   :1;    
    unsigned  BSIM3V2wpscbe2Given   :1;    
    unsigned  BSIM3V2wpvagGiven   :1;    
    unsigned  BSIM3V2wdeltaGiven  :1;     
    unsigned  BSIM3V2wwrGiven   :1;
    unsigned  BSIM3V2wdwgGiven   :1;
    unsigned  BSIM3V2wdwbGiven   :1;
    unsigned  BSIM3V2wb0Given   :1;
    unsigned  BSIM3V2wb1Given   :1;
    unsigned  BSIM3V2walpha0Given   :1;
    unsigned  BSIM3V2walpha1Given   :1;
    unsigned  BSIM3V2wbeta0Given   :1;
    unsigned  BSIM3V2wvfbGiven   :1;

    /* CV model */
    unsigned  BSIM3V2welmGiven  :1;     
    unsigned  BSIM3V2wcgslGiven   :1;
    unsigned  BSIM3V2wcgdlGiven   :1;
    unsigned  BSIM3V2wckappaGiven   :1;
    unsigned  BSIM3V2wcfGiven   :1;
    unsigned  BSIM3V2wclcGiven   :1;
    unsigned  BSIM3V2wcleGiven   :1;
    unsigned  BSIM3V2wvfbcvGiven   :1;
    unsigned  BSIM3V2wnoffGiven   :1;
    unsigned  BSIM3V2wvoffcvGiven :1;
    unsigned  BSIM3V2wacdeGiven   :1;
    unsigned  BSIM3V2wmoinGiven   :1;

    /* Cross-term dependence */
    unsigned  BSIM3V2pcdscGiven   :1;
    unsigned  BSIM3V2pcdscbGiven   :1;
    unsigned  BSIM3V2pcdscdGiven   :1;
    unsigned  BSIM3V2pcitGiven   :1;
    unsigned  BSIM3V2pnfactorGiven   :1;
    unsigned  BSIM3V2pxjGiven   :1;
    unsigned  BSIM3V2pvsatGiven   :1;
    unsigned  BSIM3V2patGiven   :1;
    unsigned  BSIM3V2pa0Given   :1;
    unsigned  BSIM3V2pagsGiven   :1;
    unsigned  BSIM3V2pa1Given   :1;
    unsigned  BSIM3V2pa2Given   :1;
    unsigned  BSIM3V2pketaGiven   :1;    
    unsigned  BSIM3V2pnsubGiven   :1;
    unsigned  BSIM3V2pnpeakGiven   :1;
    unsigned  BSIM3V2pngateGiven   :1;
    unsigned  BSIM3V2pgamma1Given   :1;
    unsigned  BSIM3V2pgamma2Given   :1;
    unsigned  BSIM3V2pvbxGiven   :1;
    unsigned  BSIM3V2pvbmGiven   :1;
    unsigned  BSIM3V2pxtGiven   :1;
    unsigned  BSIM3V2pk1Given   :1;
    unsigned  BSIM3V2pkt1Given   :1;
    unsigned  BSIM3V2pkt1lGiven   :1;
    unsigned  BSIM3V2pkt2Given   :1;
    unsigned  BSIM3V2pk2Given   :1;
    unsigned  BSIM3V2pk3Given   :1;
    unsigned  BSIM3V2pk3bGiven   :1;
    unsigned  BSIM3V2pw0Given   :1;
    unsigned  BSIM3V2pnlxGiven   :1;
    unsigned  BSIM3V2pdvt0Given   :1;   
    unsigned  BSIM3V2pdvt1Given   :1;     
    unsigned  BSIM3V2pdvt2Given   :1;     
    unsigned  BSIM3V2pdvt0wGiven   :1;   
    unsigned  BSIM3V2pdvt1wGiven   :1;     
    unsigned  BSIM3V2pdvt2wGiven   :1;     
    unsigned  BSIM3V2pdroutGiven   :1;     
    unsigned  BSIM3V2pdsubGiven   :1;     
    unsigned  BSIM3V2pvth0Given   :1;
    unsigned  BSIM3V2puaGiven   :1;
    unsigned  BSIM3V2pua1Given   :1;
    unsigned  BSIM3V2pubGiven   :1;
    unsigned  BSIM3V2pub1Given   :1;
    unsigned  BSIM3V2pucGiven   :1;
    unsigned  BSIM3V2puc1Given   :1;
    unsigned  BSIM3V2pu0Given   :1;
    unsigned  BSIM3V2puteGiven   :1;
    unsigned  BSIM3V2pvoffGiven   :1;
    unsigned  BSIM3V2prdswGiven   :1;      
    unsigned  BSIM3V2pprwgGiven   :1;      
    unsigned  BSIM3V2pprwbGiven   :1;      
    unsigned  BSIM3V2pprtGiven   :1;      
    unsigned  BSIM3V2peta0Given   :1;    
    unsigned  BSIM3V2petabGiven   :1;    
    unsigned  BSIM3V2ppclmGiven   :1;   
    unsigned  BSIM3V2ppdibl1Given   :1;   
    unsigned  BSIM3V2ppdibl2Given   :1;  
    unsigned  BSIM3V2ppdiblbGiven   :1;  
    unsigned  BSIM3V2ppscbe1Given   :1;    
    unsigned  BSIM3V2ppscbe2Given   :1;    
    unsigned  BSIM3V2ppvagGiven   :1;    
    unsigned  BSIM3V2pdeltaGiven  :1;     
    unsigned  BSIM3V2pwrGiven   :1;
    unsigned  BSIM3V2pdwgGiven   :1;
    unsigned  BSIM3V2pdwbGiven   :1;
    unsigned  BSIM3V2pb0Given   :1;
    unsigned  BSIM3V2pb1Given   :1;
    unsigned  BSIM3V2palpha0Given   :1;
    unsigned  BSIM3V2palpha1Given   :1;
    unsigned  BSIM3V2pbeta0Given   :1;
    unsigned  BSIM3V2pvfbGiven   :1;

    /* CV model */
    unsigned  BSIM3V2pelmGiven  :1;     
    unsigned  BSIM3V2pcgslGiven   :1;
    unsigned  BSIM3V2pcgdlGiven   :1;
    unsigned  BSIM3V2pckappaGiven   :1;
    unsigned  BSIM3V2pcfGiven   :1;
    unsigned  BSIM3V2pclcGiven   :1;
    unsigned  BSIM3V2pcleGiven   :1;
    unsigned  BSIM3V2pvfbcvGiven   :1;
    unsigned  BSIM3V2pnoffGiven   :1;
    unsigned  BSIM3V2pvoffcvGiven :1;
    unsigned  BSIM3V2pacdeGiven   :1;
    unsigned  BSIM3V2pmoinGiven   :1;

    unsigned  BSIM3V2useFringeGiven   :1;

    unsigned  BSIM3V2tnomGiven   :1;
    unsigned  BSIM3V2cgsoGiven   :1;
    unsigned  BSIM3V2cgdoGiven   :1;
    unsigned  BSIM3V2cgboGiven   :1;
    unsigned  BSIM3V2xpartGiven   :1;
    unsigned  BSIM3V2sheetResistanceGiven   :1;
    unsigned  BSIM3V2jctSatCurDensityGiven   :1;
    unsigned  BSIM3V2jctSidewallSatCurDensityGiven   :1;
    unsigned  BSIM3V2bulkJctPotentialGiven   :1;
    unsigned  BSIM3V2bulkJctBotGradingCoeffGiven   :1;
    unsigned  BSIM3V2sidewallJctPotentialGiven   :1;
    unsigned  BSIM3V2GatesidewallJctPotentialGiven   :1;
    unsigned  BSIM3V2bulkJctSideGradingCoeffGiven   :1;
    unsigned  BSIM3V2unitAreaJctCapGiven   :1;
    unsigned  BSIM3V2unitLengthSidewallJctCapGiven   :1;
    unsigned  BSIM3V2bulkJctGateSideGradingCoeffGiven   :1;
    unsigned  BSIM3V2unitLengthGateSidewallJctCapGiven   :1;
    unsigned  BSIM3V2jctEmissionCoeffGiven :1;
    unsigned  BSIM3V2jctTempExponentGiven	:1;

    unsigned  BSIM3V2oxideTrapDensityAGiven  :1;         
    unsigned  BSIM3V2oxideTrapDensityBGiven  :1;        
    unsigned  BSIM3V2oxideTrapDensityCGiven  :1;     
    unsigned  BSIM3V2emGiven  :1;     
    unsigned  BSIM3V2efGiven  :1;     
    unsigned  BSIM3V2afGiven  :1;     
    unsigned  BSIM3V2kfGiven  :1;     

    unsigned  BSIM3V2LintGiven   :1;
    unsigned  BSIM3V2LlGiven   :1;
    unsigned  BSIM3V2LlcGiven   :1;
    unsigned  BSIM3V2LlnGiven   :1;
    unsigned  BSIM3V2LwGiven   :1;
    unsigned  BSIM3V2LwcGiven   :1;
    unsigned  BSIM3V2LwnGiven   :1;
    unsigned  BSIM3V2LwlGiven   :1;
    unsigned  BSIM3V2LwlcGiven   :1;
    unsigned  BSIM3V2LminGiven   :1;
    unsigned  BSIM3V2LmaxGiven   :1;

    unsigned  BSIM3V2WintGiven   :1;
    unsigned  BSIM3V2WlGiven   :1;
    unsigned  BSIM3V2WlcGiven   :1;
    unsigned  BSIM3V2WlnGiven   :1;
    unsigned  BSIM3V2WwGiven   :1;
    unsigned  BSIM3V2WwcGiven   :1;
    unsigned  BSIM3V2WwnGiven   :1;
    unsigned  BSIM3V2WwlGiven   :1;
    unsigned  BSIM3V2WwlcGiven   :1;
    unsigned  BSIM3V2WminGiven   :1;
    unsigned  BSIM3V2WmaxGiven   :1;

} BSIM3V2model;


#ifndef NMOS
#define NMOS 1
#define PMOS -1
#endif /*NMOS*/


/* device parameters */
#define BSIM3V2_W 1
#define BSIM3V2_L 2
#define BSIM3V2_AS 3
#define BSIM3V2_AD 4
#define BSIM3V2_PS 5
#define BSIM3V2_PD 6
#define BSIM3V2_NRS 7
#define BSIM3V2_NRD 8
#define BSIM3V2_OFF 9
#define BSIM3V2_IC_VBS 10
#define BSIM3V2_IC_VDS 11
#define BSIM3V2_IC_VGS 12
#define BSIM3V2_IC 13
#define BSIM3V2_NQSMOD 14

/* model parameters */
#define BSIM3V2_MOD_CAPMOD          101
#define BSIM3V2_MOD_MOBMOD          103    
#define BSIM3V2_MOD_NOIMOD          104    

#define BSIM3V2_MOD_TOX             105

#define BSIM3V2_MOD_CDSC            106
#define BSIM3V2_MOD_CDSCB           107
#define BSIM3V2_MOD_CIT             108
#define BSIM3V2_MOD_NFACTOR         109
#define BSIM3V2_MOD_XJ              110
#define BSIM3V2_MOD_VSAT            111
#define BSIM3V2_MOD_AT              112
#define BSIM3V2_MOD_A0              113
#define BSIM3V2_MOD_A1              114
#define BSIM3V2_MOD_A2              115
#define BSIM3V2_MOD_KETA            116   
#define BSIM3V2_MOD_NSUB            117
#define BSIM3V2_MOD_NPEAK           118
#define BSIM3V2_MOD_NGATE           120
#define BSIM3V2_MOD_GAMMA1          121
#define BSIM3V2_MOD_GAMMA2          122
#define BSIM3V2_MOD_VBX             123
#define BSIM3V2_MOD_BINUNIT         124    

#define BSIM3V2_MOD_VBM             125

#define BSIM3V2_MOD_XT              126
#define BSIM3V2_MOD_K1              129
#define BSIM3V2_MOD_KT1             130
#define BSIM3V2_MOD_KT1L            131
#define BSIM3V2_MOD_K2              132
#define BSIM3V2_MOD_KT2             133
#define BSIM3V2_MOD_K3              134
#define BSIM3V2_MOD_K3B             135
#define BSIM3V2_MOD_W0              136
#define BSIM3V2_MOD_NLX             137

#define BSIM3V2_MOD_DVT0            138
#define BSIM3V2_MOD_DVT1            139
#define BSIM3V2_MOD_DVT2            140

#define BSIM3V2_MOD_DVT0W           141
#define BSIM3V2_MOD_DVT1W           142
#define BSIM3V2_MOD_DVT2W           143

#define BSIM3V2_MOD_DROUT           144
#define BSIM3V2_MOD_DSUB            145
#define BSIM3V2_MOD_VTH0            146
#define BSIM3V2_MOD_UA              147
#define BSIM3V2_MOD_UA1             148
#define BSIM3V2_MOD_UB              149
#define BSIM3V2_MOD_UB1             150
#define BSIM3V2_MOD_UC              151
#define BSIM3V2_MOD_UC1             152
#define BSIM3V2_MOD_U0              153
#define BSIM3V2_MOD_UTE             154
#define BSIM3V2_MOD_VOFF            155
#define BSIM3V2_MOD_DELTA           156
#define BSIM3V2_MOD_RDSW            157
#define BSIM3V2_MOD_PRT             158
#define BSIM3V2_MOD_LDD             159
#define BSIM3V2_MOD_ETA             160
#define BSIM3V2_MOD_ETA0            161
#define BSIM3V2_MOD_ETAB            162
#define BSIM3V2_MOD_PCLM            163
#define BSIM3V2_MOD_PDIBL1          164
#define BSIM3V2_MOD_PDIBL2          165
#define BSIM3V2_MOD_PSCBE1          166
#define BSIM3V2_MOD_PSCBE2          167
#define BSIM3V2_MOD_PVAG            168
#define BSIM3V2_MOD_WR              169
#define BSIM3V2_MOD_DWG             170
#define BSIM3V2_MOD_DWB             171
#define BSIM3V2_MOD_B0              172
#define BSIM3V2_MOD_B1              173
#define BSIM3V2_MOD_ALPHA0          174
#define BSIM3V2_MOD_BETA0           175
#define BSIM3V2_MOD_PDIBLB          178

#define BSIM3V2_MOD_PRWG            179
#define BSIM3V2_MOD_PRWB            180

#define BSIM3V2_MOD_CDSCD           181
#define BSIM3V2_MOD_AGS             182

#define BSIM3V2_MOD_FRINGE          184
#define BSIM3V2_MOD_ELM             185
#define BSIM3V2_MOD_CGSL            186
#define BSIM3V2_MOD_CGDL            187
#define BSIM3V2_MOD_CKAPPA          188
#define BSIM3V2_MOD_CF              189
#define BSIM3V2_MOD_CLC             190
#define BSIM3V2_MOD_CLE             191
#define BSIM3V2_MOD_PARAMCHK        192
#define BSIM3V2_MOD_VERSION         193
#define BSIM3V2_MOD_VFBCV           194
#define BSIM3V2_MOD_ACDE            195
#define BSIM3V2_MOD_MOIN            196
#define BSIM3V2_MOD_NOFF            197
#define BSIM3V2_MOD_IJTH            198
#define BSIM3V2_MOD_ALPHA1          199
#define BSIM3V2_MOD_VFB             200
#define BSIM3V2_MOD_TOXM            201
#define BSIM3V2_MOD_TCJ             202
#define BSIM3V2_MOD_TCJSW           203
#define BSIM3V2_MOD_TCJSWG          204
#define BSIM3V2_MOD_TPB             205
#define BSIM3V2_MOD_TPBSW           206
#define BSIM3V2_MOD_TPBSWG          207
#define BSIM3V2_MOD_VOFFCV          208

/* Length dependence */
#define BSIM3V2_MOD_LCDSC            251
#define BSIM3V2_MOD_LCDSCB           252
#define BSIM3V2_MOD_LCIT             253
#define BSIM3V2_MOD_LNFACTOR         254
#define BSIM3V2_MOD_LXJ              255
#define BSIM3V2_MOD_LVSAT            256
#define BSIM3V2_MOD_LAT              257
#define BSIM3V2_MOD_LA0              258
#define BSIM3V2_MOD_LA1              259
#define BSIM3V2_MOD_LA2              260
#define BSIM3V2_MOD_LKETA            261   
#define BSIM3V2_MOD_LNSUB            262
#define BSIM3V2_MOD_LNPEAK           263
#define BSIM3V2_MOD_LNGATE           265
#define BSIM3V2_MOD_LGAMMA1          266
#define BSIM3V2_MOD_LGAMMA2          267
#define BSIM3V2_MOD_LVBX             268

#define BSIM3V2_MOD_LVBM             270

#define BSIM3V2_MOD_LXT              272
#define BSIM3V2_MOD_LK1              275
#define BSIM3V2_MOD_LKT1             276
#define BSIM3V2_MOD_LKT1L            277
#define BSIM3V2_MOD_LK2              278
#define BSIM3V2_MOD_LKT2             279
#define BSIM3V2_MOD_LK3              280
#define BSIM3V2_MOD_LK3B             281
#define BSIM3V2_MOD_LW0              282
#define BSIM3V2_MOD_LNLX             283

#define BSIM3V2_MOD_LDVT0            284
#define BSIM3V2_MOD_LDVT1            285
#define BSIM3V2_MOD_LDVT2            286

#define BSIM3V2_MOD_LDVT0W           287
#define BSIM3V2_MOD_LDVT1W           288
#define BSIM3V2_MOD_LDVT2W           289

#define BSIM3V2_MOD_LDROUT           290
#define BSIM3V2_MOD_LDSUB            291
#define BSIM3V2_MOD_LVTH0            292
#define BSIM3V2_MOD_LUA              293
#define BSIM3V2_MOD_LUA1             294
#define BSIM3V2_MOD_LUB              295
#define BSIM3V2_MOD_LUB1             296
#define BSIM3V2_MOD_LUC              297
#define BSIM3V2_MOD_LUC1             298
#define BSIM3V2_MOD_LU0              299
#define BSIM3V2_MOD_LUTE             300
#define BSIM3V2_MOD_LVOFF            301
#define BSIM3V2_MOD_LDELTA           302
#define BSIM3V2_MOD_LRDSW            303
#define BSIM3V2_MOD_LPRT             304
#define BSIM3V2_MOD_LLDD             305
#define BSIM3V2_MOD_LETA             306
#define BSIM3V2_MOD_LETA0            307
#define BSIM3V2_MOD_LETAB            308
#define BSIM3V2_MOD_LPCLM            309
#define BSIM3V2_MOD_LPDIBL1          310
#define BSIM3V2_MOD_LPDIBL2          311
#define BSIM3V2_MOD_LPSCBE1          312
#define BSIM3V2_MOD_LPSCBE2          313
#define BSIM3V2_MOD_LPVAG            314
#define BSIM3V2_MOD_LWR              315
#define BSIM3V2_MOD_LDWG             316
#define BSIM3V2_MOD_LDWB             317
#define BSIM3V2_MOD_LB0              318
#define BSIM3V2_MOD_LB1              319
#define BSIM3V2_MOD_LALPHA0          320
#define BSIM3V2_MOD_LBETA0           321
#define BSIM3V2_MOD_LPDIBLB          324

#define BSIM3V2_MOD_LPRWG            325
#define BSIM3V2_MOD_LPRWB            326

#define BSIM3V2_MOD_LCDSCD           327
#define BSIM3V2_MOD_LAGS             328
                                    

#define BSIM3V2_MOD_LFRINGE          331
#define BSIM3V2_MOD_LELM             332
#define BSIM3V2_MOD_LCGSL            333
#define BSIM3V2_MOD_LCGDL            334
#define BSIM3V2_MOD_LCKAPPA          335
#define BSIM3V2_MOD_LCF              336
#define BSIM3V2_MOD_LCLC             337
#define BSIM3V2_MOD_LCLE             338
#define BSIM3V2_MOD_LVFBCV           339
#define BSIM3V2_MOD_LACDE            340
#define BSIM3V2_MOD_LMOIN            341
#define BSIM3V2_MOD_LNOFF            342
#define BSIM3V2_MOD_LALPHA1          344
#define BSIM3V2_MOD_LVFB             345
#define BSIM3V2_MOD_LVOFFCV          346

/* Width dependence */
#define BSIM3V2_MOD_WCDSC            381
#define BSIM3V2_MOD_WCDSCB           382
#define BSIM3V2_MOD_WCIT             383
#define BSIM3V2_MOD_WNFACTOR         384
#define BSIM3V2_MOD_WXJ              385
#define BSIM3V2_MOD_WVSAT            386
#define BSIM3V2_MOD_WAT              387
#define BSIM3V2_MOD_WA0              388
#define BSIM3V2_MOD_WA1              389
#define BSIM3V2_MOD_WA2              390
#define BSIM3V2_MOD_WKETA            391   
#define BSIM3V2_MOD_WNSUB            392
#define BSIM3V2_MOD_WNPEAK           393
#define BSIM3V2_MOD_WNGATE           395
#define BSIM3V2_MOD_WGAMMA1          396
#define BSIM3V2_MOD_WGAMMA2          397
#define BSIM3V2_MOD_WVBX             398

#define BSIM3V2_MOD_WVBM             400

#define BSIM3V2_MOD_WXT              402
#define BSIM3V2_MOD_WK1              405
#define BSIM3V2_MOD_WKT1             406
#define BSIM3V2_MOD_WKT1L            407
#define BSIM3V2_MOD_WK2              408
#define BSIM3V2_MOD_WKT2             409
#define BSIM3V2_MOD_WK3              410
#define BSIM3V2_MOD_WK3B             411
#define BSIM3V2_MOD_WW0              412
#define BSIM3V2_MOD_WNLX             413

#define BSIM3V2_MOD_WDVT0            414
#define BSIM3V2_MOD_WDVT1            415
#define BSIM3V2_MOD_WDVT2            416

#define BSIM3V2_MOD_WDVT0W           417
#define BSIM3V2_MOD_WDVT1W           418
#define BSIM3V2_MOD_WDVT2W           419

#define BSIM3V2_MOD_WDROUT           420
#define BSIM3V2_MOD_WDSUB            421
#define BSIM3V2_MOD_WVTH0            422
#define BSIM3V2_MOD_WUA              423
#define BSIM3V2_MOD_WUA1             424
#define BSIM3V2_MOD_WUB              425
#define BSIM3V2_MOD_WUB1             426
#define BSIM3V2_MOD_WUC              427
#define BSIM3V2_MOD_WUC1             428
#define BSIM3V2_MOD_WU0              429
#define BSIM3V2_MOD_WUTE             430
#define BSIM3V2_MOD_WVOFF            431
#define BSIM3V2_MOD_WDELTA           432
#define BSIM3V2_MOD_WRDSW            433
#define BSIM3V2_MOD_WPRT             434
#define BSIM3V2_MOD_WLDD             435
#define BSIM3V2_MOD_WETA             436
#define BSIM3V2_MOD_WETA0            437
#define BSIM3V2_MOD_WETAB            438
#define BSIM3V2_MOD_WPCLM            439
#define BSIM3V2_MOD_WPDIBL1          440
#define BSIM3V2_MOD_WPDIBL2          441
#define BSIM3V2_MOD_WPSCBE1          442
#define BSIM3V2_MOD_WPSCBE2          443
#define BSIM3V2_MOD_WPVAG            444
#define BSIM3V2_MOD_WWR              445
#define BSIM3V2_MOD_WDWG             446
#define BSIM3V2_MOD_WDWB             447
#define BSIM3V2_MOD_WB0              448
#define BSIM3V2_MOD_WB1              449
#define BSIM3V2_MOD_WALPHA0          450
#define BSIM3V2_MOD_WBETA0           451
#define BSIM3V2_MOD_WPDIBLB          454

#define BSIM3V2_MOD_WPRWG            455
#define BSIM3V2_MOD_WPRWB            456

#define BSIM3V2_MOD_WCDSCD           457
#define BSIM3V2_MOD_WAGS             458


#define BSIM3V2_MOD_WFRINGE          461
#define BSIM3V2_MOD_WELM             462
#define BSIM3V2_MOD_WCGSL            463
#define BSIM3V2_MOD_WCGDL            464
#define BSIM3V2_MOD_WCKAPPA          465
#define BSIM3V2_MOD_WCF              466
#define BSIM3V2_MOD_WCLC             467
#define BSIM3V2_MOD_WCLE             468
#define BSIM3V2_MOD_WVFBCV           469
#define BSIM3V2_MOD_WACDE            470
#define BSIM3V2_MOD_WMOIN            471
#define BSIM3V2_MOD_WNOFF            472
#define BSIM3V2_MOD_WALPHA1          474
#define BSIM3V2_MOD_WVFB             475
#define BSIM3V2_MOD_WVOFFCV          476

/* Cross-term dependence */
#define BSIM3V2_MOD_PCDSC            511
#define BSIM3V2_MOD_PCDSCB           512
#define BSIM3V2_MOD_PCIT             513
#define BSIM3V2_MOD_PNFACTOR         514
#define BSIM3V2_MOD_PXJ              515
#define BSIM3V2_MOD_PVSAT            516
#define BSIM3V2_MOD_PAT              517
#define BSIM3V2_MOD_PA0              518
#define BSIM3V2_MOD_PA1              519
#define BSIM3V2_MOD_PA2              520
#define BSIM3V2_MOD_PKETA            521   
#define BSIM3V2_MOD_PNSUB            522
#define BSIM3V2_MOD_PNPEAK           523
#define BSIM3V2_MOD_PNGATE           525
#define BSIM3V2_MOD_PGAMMA1          526
#define BSIM3V2_MOD_PGAMMA2          527
#define BSIM3V2_MOD_PVBX             528

#define BSIM3V2_MOD_PVBM             530

#define BSIM3V2_MOD_PXT              532
#define BSIM3V2_MOD_PK1              535
#define BSIM3V2_MOD_PKT1             536
#define BSIM3V2_MOD_PKT1L            537
#define BSIM3V2_MOD_PK2              538
#define BSIM3V2_MOD_PKT2             539
#define BSIM3V2_MOD_PK3              540
#define BSIM3V2_MOD_PK3B             541
#define BSIM3V2_MOD_PW0              542
#define BSIM3V2_MOD_PNLX             543

#define BSIM3V2_MOD_PDVT0            544
#define BSIM3V2_MOD_PDVT1            545
#define BSIM3V2_MOD_PDVT2            546

#define BSIM3V2_MOD_PDVT0W           547
#define BSIM3V2_MOD_PDVT1W           548
#define BSIM3V2_MOD_PDVT2W           549

#define BSIM3V2_MOD_PDROUT           550
#define BSIM3V2_MOD_PDSUB            551
#define BSIM3V2_MOD_PVTH0            552
#define BSIM3V2_MOD_PUA              553
#define BSIM3V2_MOD_PUA1             554
#define BSIM3V2_MOD_PUB              555
#define BSIM3V2_MOD_PUB1             556
#define BSIM3V2_MOD_PUC              557
#define BSIM3V2_MOD_PUC1             558
#define BSIM3V2_MOD_PU0              559
#define BSIM3V2_MOD_PUTE             560
#define BSIM3V2_MOD_PVOFF            561
#define BSIM3V2_MOD_PDELTA           562
#define BSIM3V2_MOD_PRDSW            563
#define BSIM3V2_MOD_PPRT             564
#define BSIM3V2_MOD_PLDD             565
#define BSIM3V2_MOD_PETA             566
#define BSIM3V2_MOD_PETA0            567
#define BSIM3V2_MOD_PETAB            568
#define BSIM3V2_MOD_PPCLM            569
#define BSIM3V2_MOD_PPDIBL1          570
#define BSIM3V2_MOD_PPDIBL2          571
#define BSIM3V2_MOD_PPSCBE1          572
#define BSIM3V2_MOD_PPSCBE2          573
#define BSIM3V2_MOD_PPVAG            574
#define BSIM3V2_MOD_PWR              575
#define BSIM3V2_MOD_PDWG             576
#define BSIM3V2_MOD_PDWB             577
#define BSIM3V2_MOD_PB0              578
#define BSIM3V2_MOD_PB1              579
#define BSIM3V2_MOD_PALPHA0          580
#define BSIM3V2_MOD_PBETA0           581
#define BSIM3V2_MOD_PPDIBLB          584

#define BSIM3V2_MOD_PPRWG            585
#define BSIM3V2_MOD_PPRWB            586

#define BSIM3V2_MOD_PCDSCD           587
#define BSIM3V2_MOD_PAGS             588

#define BSIM3V2_MOD_PFRINGE          591
#define BSIM3V2_MOD_PELM             592
#define BSIM3V2_MOD_PCGSL            593
#define BSIM3V2_MOD_PCGDL            594
#define BSIM3V2_MOD_PCKAPPA          595
#define BSIM3V2_MOD_PCF              596
#define BSIM3V2_MOD_PCLC             597
#define BSIM3V2_MOD_PCLE             598
#define BSIM3V2_MOD_PVFBCV           599
#define BSIM3V2_MOD_PACDE            600
#define BSIM3V2_MOD_PMOIN            601
#define BSIM3V2_MOD_PNOFF            602
#define BSIM3V2_MOD_PALPHA1          604
#define BSIM3V2_MOD_PVFB             605
#define BSIM3V2_MOD_PVOFFCV          606

#define BSIM3V2_MOD_TNOM             651
#define BSIM3V2_MOD_CGSO             652
#define BSIM3V2_MOD_CGDO             653
#define BSIM3V2_MOD_CGBO             654
#define BSIM3V2_MOD_XPART            655

#define BSIM3V2_MOD_RSH              656
#define BSIM3V2_MOD_JS               657
#define BSIM3V2_MOD_PB               658
#define BSIM3V2_MOD_MJ               659
#define BSIM3V2_MOD_PBSW             660
#define BSIM3V2_MOD_MJSW             661
#define BSIM3V2_MOD_CJ               662
#define BSIM3V2_MOD_CJSW             663
#define BSIM3V2_MOD_NMOS             664
#define BSIM3V2_MOD_PMOS             665

#define BSIM3V2_MOD_NOIA             666
#define BSIM3V2_MOD_NOIB             667
#define BSIM3V2_MOD_NOIC             668

#define BSIM3V2_MOD_LINT             669
#define BSIM3V2_MOD_LL               670
#define BSIM3V2_MOD_LLN              671
#define BSIM3V2_MOD_LW               672
#define BSIM3V2_MOD_LWN              673
#define BSIM3V2_MOD_LWL              674
#define BSIM3V2_MOD_LMIN             675
#define BSIM3V2_MOD_LMAX             676

#define BSIM3V2_MOD_WINT             677
#define BSIM3V2_MOD_WL               678
#define BSIM3V2_MOD_WLN              679
#define BSIM3V2_MOD_WW               680
#define BSIM3V2_MOD_WWN              681
#define BSIM3V2_MOD_WWL              682
#define BSIM3V2_MOD_WMIN             683
#define BSIM3V2_MOD_WMAX             684

#define BSIM3V2_MOD_DWC              685
#define BSIM3V2_MOD_DLC              686

#define BSIM3V2_MOD_EM               687
#define BSIM3V2_MOD_EF               688
#define BSIM3V2_MOD_AF               689
#define BSIM3V2_MOD_KF               690

#define BSIM3V2_MOD_NJ               691
#define BSIM3V2_MOD_XTI              692

#define BSIM3V2_MOD_PBSWG            693
#define BSIM3V2_MOD_MJSWG            694
#define BSIM3V2_MOD_CJSWG            695
#define BSIM3V2_MOD_JSW              696

#define BSIM3V2_MOD_LLC              697
#define BSIM3V2_MOD_LWC              698
#define BSIM3V2_MOD_LWLC             699

#define BSIM3V2_MOD_WLC              700
#define BSIM3V2_MOD_WWC              701
#define BSIM3V2_MOD_WWLC             702

/* device questions */
#define BSIM3V2_DNODE                751
#define BSIM3V2_GNODE                752
#define BSIM3V2_SNODE                753
#define BSIM3V2_BNODE                754
#define BSIM3V2_DNODEPRIME           755
#define BSIM3V2_SNODEPRIME           756
#define BSIM3V2_VBD                  757
#define BSIM3V2_VBS                  758
#define BSIM3V2_VGS                  759
#define BSIM3V2_VDS                  760
#define BSIM3V2_CD                   761
#define BSIM3V2_CBS                  762
#define BSIM3V2_CBD                  763
#define BSIM3V2_GM                   764
#define BSIM3V2_GDS                  765
#define BSIM3V2_GMBS                 766
#define BSIM3V2_GBD                  767
#define BSIM3V2_GBS                  768
#define BSIM3V2_QB                   769
#define BSIM3V2_CQB                  770
#define BSIM3V2_QG                   771
#define BSIM3V2_CQG                  772
#define BSIM3V2_QD                   773
#define BSIM3V2_CQD                  774
#define BSIM3V2_CGG                  775
#define BSIM3V2_CGD                  776
#define BSIM3V2_CGS                  777
#define BSIM3V2_CBG                  778
#define BSIM3V2_CAPBD                779
#define BSIM3V2_CQBD                 780
#define BSIM3V2_CAPBS                781
#define BSIM3V2_CQBS                 782
#define BSIM3V2_CDG                  783
#define BSIM3V2_CDD                  784
#define BSIM3V2_CDS                  785
#define BSIM3V2_VON                  786
#define BSIM3V2_VDSAT                787
#define BSIM3V2_QBS                  788
#define BSIM3V2_QBD                  789
#define BSIM3V2_SOURCECONDUCT        790
#define BSIM3V2_DRAINCONDUCT         791
#define BSIM3V2_CBDB                 792
#define BSIM3V2_CBSB                 793


#include "bsim3v2ext.h"

#ifdef __STDC__
extern void BSIM3V2evaluate(double,double,double,BSIM3V2instance*,BSIM3V2model*,
        double*,double*,double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, CKTcircuit*);
extern int BSIM3V2debug(BSIM3V2model*, BSIM3V2instance*, CKTcircuit*, int);
extern int BSIM3V2checkModel(BSIM3V2model*, BSIM3V2instance*, CKTcircuit*);
#else /* stdc */
extern void BSIM3V2evaluate();
extern int BSIM3V2debug();
extern int BSIM3V2checkModel();
#endif /* stdc */

#endif /*BSIM3V2*/

