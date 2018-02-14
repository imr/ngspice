/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan
File: bsim3v0def.h
**********/

#ifndef BSIM3v0
#define BSIM3v0

#include "ngspice/ifsim.h"
#include "ngspice/gendefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/noisedef.h"         

typedef struct sBSIM3v0instance
{

    struct GENinstance gen;

#define BSIM3v0modPtr(inst) ((struct sBSIM3v0model *)((inst)->gen.GENmodPtr))
#define BSIM3v0nextInstance(inst) ((struct sBSIM3v0instance *)((inst)->gen.GENnextInstance))
#define BSIM3v0name gen.GENname
#define BSIM3v0states gen.GENstate

    const int BSIM3v0dNode;
    const int BSIM3v0gNode;
    const int BSIM3v0sNode;
    const int BSIM3v0bNode;
    int BSIM3v0dNodePrime;
    int BSIM3v0sNodePrime;
    int BSIM3v0qNode; /* MCJ */

    /* MCJ */
    double BSIM3v0ueff;
    double BSIM3v0thetavth; 
    double BSIM3v0von;
    double BSIM3v0vdsat;
    double BSIM3v0cgdo;
    double BSIM3v0cgso;

    double BSIM3v0l;
    double BSIM3v0w;
    double BSIM3v0m;
    double BSIM3v0drainArea;
    double BSIM3v0sourceArea;
    double BSIM3v0drainSquares;
    double BSIM3v0sourceSquares;
    double BSIM3v0drainPerimeter;
    double BSIM3v0sourcePerimeter;
    double BSIM3v0sourceConductance;
    double BSIM3v0drainConductance;

    double BSIM3v0icVBS;
    double BSIM3v0icVDS;
    double BSIM3v0icVGS;
    int BSIM3v0off;
    int BSIM3v0mode;
    int BSIM3v0nqsMod;

    /* OP point */
    double BSIM3v0qinv;
    double BSIM3v0cd;
    double BSIM3v0cbs;
    double BSIM3v0cbd;
    double BSIM3v0csub;
    double BSIM3v0gm;
    double BSIM3v0gds;
    double BSIM3v0gmbs;
    double BSIM3v0gbd;
    double BSIM3v0gbs;

    double BSIM3v0gbbs;
    double BSIM3v0gbgs;
    double BSIM3v0gbds;

    double BSIM3v0cggb;
    double BSIM3v0cgdb;
    double BSIM3v0cgsb;
    double BSIM3v0cbgb;
    double BSIM3v0cbdb;
    double BSIM3v0cbsb;
    double BSIM3v0cdgb;
    double BSIM3v0cddb;
    double BSIM3v0cdsb;
    double BSIM3v0capbd;
    double BSIM3v0capbs;

    double BSIM3v0cqgb;
    double BSIM3v0cqdb;
    double BSIM3v0cqsb;
    double BSIM3v0cqbb;

    double BSIM3v0gtau;
    double BSIM3v0gtg;
    double BSIM3v0gtd;
    double BSIM3v0gts;
    double BSIM3v0gtb;
    double BSIM3v0tconst;

    struct bsim3v0SizeDependParam  *pParam;

    unsigned BSIM3v0lGiven :1;
    unsigned BSIM3v0wGiven :1;
    unsigned BSIM3v0mGiven :1;
    unsigned BSIM3v0drainAreaGiven :1;
    unsigned BSIM3v0sourceAreaGiven    :1;
    unsigned BSIM3v0drainSquaresGiven  :1;
    unsigned BSIM3v0sourceSquaresGiven :1;
    unsigned BSIM3v0drainPerimeterGiven    :1;
    unsigned BSIM3v0sourcePerimeterGiven   :1;
    unsigned BSIM3v0dNodePrimeSet  :1;
    unsigned BSIM3v0sNodePrimeSet  :1;
    unsigned BSIM3v0icVBSGiven :1;
    unsigned BSIM3v0icVDSGiven :1;
    unsigned BSIM3v0icVGSGiven :1;
    unsigned BSIM3v0nqsModGiven :1;

    double *BSIM3v0DdPtr;
    double *BSIM3v0GgPtr;
    double *BSIM3v0SsPtr;
    double *BSIM3v0BbPtr;
    double *BSIM3v0DPdpPtr;
    double *BSIM3v0SPspPtr;
    double *BSIM3v0DdpPtr;
    double *BSIM3v0GbPtr;
    double *BSIM3v0GdpPtr;
    double *BSIM3v0GspPtr;
    double *BSIM3v0SspPtr;
    double *BSIM3v0BdpPtr;
    double *BSIM3v0BspPtr;
    double *BSIM3v0DPspPtr;
    double *BSIM3v0DPdPtr;
    double *BSIM3v0BgPtr;
    double *BSIM3v0DPgPtr;
    double *BSIM3v0SPgPtr;
    double *BSIM3v0SPsPtr;
    double *BSIM3v0DPbPtr;
    double *BSIM3v0SPbPtr;
    double *BSIM3v0SPdpPtr;

    double *BSIM3v0QqPtr;
    double *BSIM3v0QdpPtr;
    double *BSIM3v0QgPtr;
    double *BSIM3v0QspPtr;
    double *BSIM3v0QbPtr;
    double *BSIM3v0DPqPtr;
    double *BSIM3v0GqPtr;
    double *BSIM3v0SPqPtr;
    double *BSIM3v0BqPtr;

#define BSIM3v0vbd BSIM3v0states+ 0
#define BSIM3v0vbs BSIM3v0states+ 1
#define BSIM3v0vgs BSIM3v0states+ 2
#define BSIM3v0vds BSIM3v0states+ 3

#define BSIM3v0qb BSIM3v0states+ 4
#define BSIM3v0cqb BSIM3v0states+ 5
#define BSIM3v0qg BSIM3v0states+ 6
#define BSIM3v0cqg BSIM3v0states+ 7
#define BSIM3v0qd BSIM3v0states+ 8
#define BSIM3v0cqd BSIM3v0states+ 9

#define BSIM3v0qbs  BSIM3v0states+ 10
#define BSIM3v0qbd  BSIM3v0states+ 11

#define BSIM3v0qcheq BSIM3v0states+ 12
#define BSIM3v0cqcheq BSIM3v0states+ 13
#define BSIM3v0qcdump BSIM3v0states+ 14
#define BSIM3v0cqcdump BSIM3v0states+ 15

#define BSIM3v0tau BSIM3v0states+ 16
#define BSIM3v0qdef BSIM3v0states+ 17

#define BSIM3v0numStates 18


/* indices to the array of BSIM3v0 NOISE SOURCES */

#define BSIM3v0RDNOIZ       0
#define BSIM3v0RSNOIZ       1
#define BSIM3v0IDNOIZ       2
#define BSIM3v0FLNOIZ       3
#define BSIM3v0TOTNOIZ      4

#define BSIM3v0NSRCS        5     /* the number of MOSFET(3) noise sources */

#ifndef NONOISE
    double BSIM3v0nVar[NSTATVARS][BSIM3v0NSRCS];
#else /* NONOISE */
        double **BSIM3v0nVar;
#endif /* NONOISE */

} BSIM3v0instance ;

struct bsim3v0SizeDependParam
{
    double Width;
    double Length;

    double BSIM3v0cdsc;           
    double BSIM3v0cdscb;    
    double BSIM3v0cdscd;       
    double BSIM3v0cit;           
    double BSIM3v0nfactor;      
    double BSIM3v0xj;
    double BSIM3v0vsat;         
    double BSIM3v0at;         
    double BSIM3v0a0;   
    double BSIM3v0ags;      
    double BSIM3v0a1;         
    double BSIM3v0a2;         
    double BSIM3v0keta;     
    double BSIM3v0nsub;
    double BSIM3v0npeak;        
    double BSIM3v0ngate;        
    double BSIM3v0gamma1;      
    double BSIM3v0gamma2;     
    double BSIM3v0vbx;      
    double BSIM3v0vbi;       
    double BSIM3v0vbm;       
    double BSIM3v0vbsc;       
    double BSIM3v0xt;       
    double BSIM3v0phi;
    double BSIM3v0litl;
    double BSIM3v0k1;
    double BSIM3v0kt1;
    double BSIM3v0kt1l;
    double BSIM3v0kt2;
    double BSIM3v0k2;
    double BSIM3v0k3;
    double BSIM3v0k3b;
    double BSIM3v0w0;
    double BSIM3v0nlx;
    double BSIM3v0dvt0;      
    double BSIM3v0dvt1;      
    double BSIM3v0dvt2;      
    double BSIM3v0dvt0w;      
    double BSIM3v0dvt1w;      
    double BSIM3v0dvt2w;      
    double BSIM3v0drout;      
    double BSIM3v0dsub;      
    double BSIM3v0vth0;
    double BSIM3v0ua;
    double BSIM3v0ua1;
    double BSIM3v0ub;
    double BSIM3v0ub1;
    double BSIM3v0uc;
    double BSIM3v0uc1;
    double BSIM3v0u0;
    double BSIM3v0ute;
    double BSIM3v0voff;
    double BSIM3v0vfb;
    double BSIM3v0delta;
    double BSIM3v0rdsw;       
    double BSIM3v0rds0;       
    double BSIM3v0prwg;       
    double BSIM3v0prwb;       
    double BSIM3v0prt;       
    double BSIM3v0eta0;         
    double BSIM3v0etab;         
    double BSIM3v0pclm;      
    double BSIM3v0pdibl1;      
    double BSIM3v0pdibl2;      
    double BSIM3v0pdiblb;      
    double BSIM3v0pscbe1;       
    double BSIM3v0pscbe2;       
    double BSIM3v0pvag;       
    double BSIM3v0wr;
    double BSIM3v0dwg;
    double BSIM3v0dwb;
    double BSIM3v0b0;
    double BSIM3v0b1;
    double BSIM3v0alpha0;
    double BSIM3v0beta0;


    /* CV model */
    double BSIM3v0elm;
    double BSIM3v0cgsl;
    double BSIM3v0cgdl;
    double BSIM3v0ckappa;
    double BSIM3v0cf;
    double BSIM3v0clc;
    double BSIM3v0cle;


/* Pre-calculated constants */

    double BSIM3v0dw;
    double BSIM3v0dl;
    double BSIM3v0leff;
    double BSIM3v0weff;

    double BSIM3v0dwc;
    double BSIM3v0dlc;
    double BSIM3v0leffCV;
    double BSIM3v0weffCV;
    double BSIM3v0abulkCVfactor;
    double BSIM3v0cgso;
    double BSIM3v0cgdo;
    double BSIM3v0cgbo;

    double BSIM3v0u0temp;       
    double BSIM3v0vsattemp;   
    double BSIM3v0sqrtPhi;   
    double BSIM3v0phis3;   
    double BSIM3v0Xdep0;          
    double BSIM3v0sqrtXdep0;          
    double BSIM3v0theta0vb0;
    double BSIM3v0thetaRout; 

    double BSIM3v0cof1;
    double BSIM3v0cof2;
    double BSIM3v0cof3;
    double BSIM3v0cof4;
    double BSIM3v0cdep0;
    struct bsim3v0SizeDependParam  *pNext;
};


typedef struct sBSIM3v0model 
{

    struct GENmodel gen;

#define BSIM3v0modType gen.GENmodType
#define BSIM3v0nextModel(inst) ((struct sBSIM3v0model *)((inst)->gen.GENnextModel))
#define BSIM3v0instances(inst) ((BSIM3v0instance *)((inst)->gen.GENinstances))
#define BSIM3v0modName gen.GENmodName

    int BSIM3v0type;

    int    BSIM3v0mobMod;
    int    BSIM3v0capMod;
    int    BSIM3v0nqsMod;
    int    BSIM3v0noiMod;
    int    BSIM3v0binUnit;
    double BSIM3v0tox;             
    double BSIM3v0cdsc;           
    double BSIM3v0cdscb; 
    double BSIM3v0cdscd;          
    double BSIM3v0cit;           
    double BSIM3v0nfactor;      
    double BSIM3v0xj;
    double BSIM3v0vsat;         
    double BSIM3v0at;         
    double BSIM3v0a0;   
    double BSIM3v0ags;      
    double BSIM3v0a1;         
    double BSIM3v0a2;         
    double BSIM3v0keta;     
    double BSIM3v0nsub;
    double BSIM3v0npeak;        
    double BSIM3v0ngate;        
    double BSIM3v0gamma1;      
    double BSIM3v0gamma2;     
    double BSIM3v0vbx;      
    double BSIM3v0vbm;       
    double BSIM3v0xt;       
    double BSIM3v0k1;
    double BSIM3v0kt1;
    double BSIM3v0kt1l;
    double BSIM3v0kt2;
    double BSIM3v0k2;
    double BSIM3v0k3;
    double BSIM3v0k3b;
    double BSIM3v0w0;
    double BSIM3v0nlx;
    double BSIM3v0dvt0;      
    double BSIM3v0dvt1;      
    double BSIM3v0dvt2;      
    double BSIM3v0dvt0w;      
    double BSIM3v0dvt1w;      
    double BSIM3v0dvt2w;      
    double BSIM3v0drout;      
    double BSIM3v0dsub;      
    double BSIM3v0vth0;
    double BSIM3v0ua;
    double BSIM3v0ua1;
    double BSIM3v0ub;
    double BSIM3v0ub1;
    double BSIM3v0uc;
    double BSIM3v0uc1;
    double BSIM3v0u0;
    double BSIM3v0ute;
    double BSIM3v0voff;
    double BSIM3v0delta;
    double BSIM3v0rdsw;       
    double BSIM3v0prwg;
    double BSIM3v0prwb;
    double BSIM3v0prt;       
    double BSIM3v0eta0;         
    double BSIM3v0etab;         
    double BSIM3v0pclm;      
    double BSIM3v0pdibl1;      
    double BSIM3v0pdibl2;      
    double BSIM3v0pdiblb;
    double BSIM3v0pscbe1;       
    double BSIM3v0pscbe2;       
    double BSIM3v0pvag;       
    double BSIM3v0wr;
    double BSIM3v0dwg;
    double BSIM3v0dwb;
    double BSIM3v0b0;
    double BSIM3v0b1;
    double BSIM3v0alpha0;
    double BSIM3v0beta0;

    /* CV model */
    double BSIM3v0elm;
    double BSIM3v0cgsl;
    double BSIM3v0cgdl;
    double BSIM3v0ckappa;
    double BSIM3v0cf;
    double BSIM3v0clc;
    double BSIM3v0cle;
    double BSIM3v0dwc;
    double BSIM3v0dlc;

    /* Length Dependence */
    double BSIM3v0lcdsc;           
    double BSIM3v0lcdscb; 
    double BSIM3v0lcdscd;          
    double BSIM3v0lcit;           
    double BSIM3v0lnfactor;      
    double BSIM3v0lxj;
    double BSIM3v0lvsat;         
    double BSIM3v0lat;         
    double BSIM3v0la0;   
    double BSIM3v0lags;      
    double BSIM3v0la1;         
    double BSIM3v0la2;         
    double BSIM3v0lketa;     
    double BSIM3v0lnsub;
    double BSIM3v0lnpeak;        
    double BSIM3v0lngate;        
    double BSIM3v0lgamma1;      
    double BSIM3v0lgamma2;     
    double BSIM3v0lvbx;      
    double BSIM3v0lvbm;       
    double BSIM3v0lxt;       
    double BSIM3v0lk1;
    double BSIM3v0lkt1;
    double BSIM3v0lkt1l;
    double BSIM3v0lkt2;
    double BSIM3v0lk2;
    double BSIM3v0lk3;
    double BSIM3v0lk3b;
    double BSIM3v0lw0;
    double BSIM3v0lnlx;
    double BSIM3v0ldvt0;      
    double BSIM3v0ldvt1;      
    double BSIM3v0ldvt2;      
    double BSIM3v0ldvt0w;      
    double BSIM3v0ldvt1w;      
    double BSIM3v0ldvt2w;      
    double BSIM3v0ldrout;      
    double BSIM3v0ldsub;      
    double BSIM3v0lvth0;
    double BSIM3v0lua;
    double BSIM3v0lua1;
    double BSIM3v0lub;
    double BSIM3v0lub1;
    double BSIM3v0luc;
    double BSIM3v0luc1;
    double BSIM3v0lu0;
    double BSIM3v0lute;
    double BSIM3v0lvoff;
    double BSIM3v0ldelta;
    double BSIM3v0lrdsw;       
    double BSIM3v0lprwg;
    double BSIM3v0lprwb;
    double BSIM3v0lprt;       
    double BSIM3v0leta0;         
    double BSIM3v0letab;         
    double BSIM3v0lpclm;      
    double BSIM3v0lpdibl1;      
    double BSIM3v0lpdibl2;      
    double BSIM3v0lpdiblb;
    double BSIM3v0lpscbe1;       
    double BSIM3v0lpscbe2;       
    double BSIM3v0lpvag;       
    double BSIM3v0lwr;
    double BSIM3v0ldwg;
    double BSIM3v0ldwb;
    double BSIM3v0lb0;
    double BSIM3v0lb1;
    double BSIM3v0lalpha0;
    double BSIM3v0lbeta0;

    /* CV model */
    double BSIM3v0lelm;
    double BSIM3v0lcgsl;
    double BSIM3v0lcgdl;
    double BSIM3v0lckappa;
    double BSIM3v0lcf;
    double BSIM3v0lclc;
    double BSIM3v0lcle;

    /* Width Dependence */
    double BSIM3v0wcdsc;           
    double BSIM3v0wcdscb; 
    double BSIM3v0wcdscd;          
    double BSIM3v0wcit;           
    double BSIM3v0wnfactor;      
    double BSIM3v0wxj;
    double BSIM3v0wvsat;         
    double BSIM3v0wat;         
    double BSIM3v0wa0;   
    double BSIM3v0wags;      
    double BSIM3v0wa1;         
    double BSIM3v0wa2;         
    double BSIM3v0wketa;     
    double BSIM3v0wnsub;
    double BSIM3v0wnpeak;        
    double BSIM3v0wngate;        
    double BSIM3v0wgamma1;      
    double BSIM3v0wgamma2;     
    double BSIM3v0wvbx;      
    double BSIM3v0wvbm;       
    double BSIM3v0wxt;       
    double BSIM3v0wk1;
    double BSIM3v0wkt1;
    double BSIM3v0wkt1l;
    double BSIM3v0wkt2;
    double BSIM3v0wk2;
    double BSIM3v0wk3;
    double BSIM3v0wk3b;
    double BSIM3v0ww0;
    double BSIM3v0wnlx;
    double BSIM3v0wdvt0;      
    double BSIM3v0wdvt1;      
    double BSIM3v0wdvt2;      
    double BSIM3v0wdvt0w;      
    double BSIM3v0wdvt1w;      
    double BSIM3v0wdvt2w;      
    double BSIM3v0wdrout;      
    double BSIM3v0wdsub;      
    double BSIM3v0wvth0;
    double BSIM3v0wua;
    double BSIM3v0wua1;
    double BSIM3v0wub;
    double BSIM3v0wub1;
    double BSIM3v0wuc;
    double BSIM3v0wuc1;
    double BSIM3v0wu0;
    double BSIM3v0wute;
    double BSIM3v0wvoff;
    double BSIM3v0wdelta;
    double BSIM3v0wrdsw;       
    double BSIM3v0wprwg;
    double BSIM3v0wprwb;
    double BSIM3v0wprt;       
    double BSIM3v0weta0;         
    double BSIM3v0wetab;         
    double BSIM3v0wpclm;      
    double BSIM3v0wpdibl1;      
    double BSIM3v0wpdibl2;      
    double BSIM3v0wpdiblb;
    double BSIM3v0wpscbe1;       
    double BSIM3v0wpscbe2;       
    double BSIM3v0wpvag;       
    double BSIM3v0wwr;
    double BSIM3v0wdwg;
    double BSIM3v0wdwb;
    double BSIM3v0wb0;
    double BSIM3v0wb1;
    double BSIM3v0walpha0;
    double BSIM3v0wbeta0;

    /* CV model */
    double BSIM3v0welm;
    double BSIM3v0wcgsl;
    double BSIM3v0wcgdl;
    double BSIM3v0wckappa;
    double BSIM3v0wcf;
    double BSIM3v0wclc;
    double BSIM3v0wcle;

    /* Cross-term Dependence */
    double BSIM3v0pcdsc;           
    double BSIM3v0pcdscb; 
    double BSIM3v0pcdscd;          
    double BSIM3v0pcit;           
    double BSIM3v0pnfactor;      
    double BSIM3v0pxj;
    double BSIM3v0pvsat;         
    double BSIM3v0pat;         
    double BSIM3v0pa0;   
    double BSIM3v0pags;      
    double BSIM3v0pa1;         
    double BSIM3v0pa2;         
    double BSIM3v0pketa;     
    double BSIM3v0pnsub;
    double BSIM3v0pnpeak;        
    double BSIM3v0pngate;        
    double BSIM3v0pgamma1;      
    double BSIM3v0pgamma2;     
    double BSIM3v0pvbx;      
    double BSIM3v0pvbm;       
    double BSIM3v0pxt;       
    double BSIM3v0pk1;
    double BSIM3v0pkt1;
    double BSIM3v0pkt1l;
    double BSIM3v0pkt2;
    double BSIM3v0pk2;
    double BSIM3v0pk3;
    double BSIM3v0pk3b;
    double BSIM3v0pw0;
    double BSIM3v0pnlx;
    double BSIM3v0pdvt0;      
    double BSIM3v0pdvt1;      
    double BSIM3v0pdvt2;      
    double BSIM3v0pdvt0w;      
    double BSIM3v0pdvt1w;      
    double BSIM3v0pdvt2w;      
    double BSIM3v0pdrout;      
    double BSIM3v0pdsub;      
    double BSIM3v0pvth0;
    double BSIM3v0pua;
    double BSIM3v0pua1;
    double BSIM3v0pub;
    double BSIM3v0pub1;
    double BSIM3v0puc;
    double BSIM3v0puc1;
    double BSIM3v0pu0;
    double BSIM3v0pute;
    double BSIM3v0pvoff;
    double BSIM3v0pdelta;
    double BSIM3v0prdsw;
    double BSIM3v0pprwg;
    double BSIM3v0pprwb;
    double BSIM3v0pprt;       
    double BSIM3v0peta0;         
    double BSIM3v0petab;         
    double BSIM3v0ppclm;      
    double BSIM3v0ppdibl1;      
    double BSIM3v0ppdibl2;      
    double BSIM3v0ppdiblb;
    double BSIM3v0ppscbe1;       
    double BSIM3v0ppscbe2;       
    double BSIM3v0ppvag;       
    double BSIM3v0pwr;
    double BSIM3v0pdwg;
    double BSIM3v0pdwb;
    double BSIM3v0pb0;
    double BSIM3v0pb1;
    double BSIM3v0palpha0;
    double BSIM3v0pbeta0;

    /* CV model */
    double BSIM3v0pelm;
    double BSIM3v0pcgsl;
    double BSIM3v0pcgdl;
    double BSIM3v0pckappa;
    double BSIM3v0pcf;
    double BSIM3v0pclc;
    double BSIM3v0pcle;

    double BSIM3v0tnom;
    double BSIM3v0cgso;
    double BSIM3v0cgdo;
    double BSIM3v0cgbo;
    double BSIM3v0xpart;
    double BSIM3v0cFringOut;
    double BSIM3v0cFringMax;

    double BSIM3v0sheetResistance;
    double BSIM3v0jctSatCurDensity;
    double BSIM3v0bulkJctPotential;
    double BSIM3v0bulkJctBotGradingCoeff;
    double BSIM3v0bulkJctSideGradingCoeff;
    double BSIM3v0sidewallJctPotential;
    double BSIM3v0unitAreaJctCap;
    double BSIM3v0unitLengthSidewallJctCap;

    double BSIM3v0Lint;
    double BSIM3v0Ll;
    double BSIM3v0Lln;
    double BSIM3v0Lw;
    double BSIM3v0Lwn;
    double BSIM3v0Lwl;
    double BSIM3v0Lmin;
    double BSIM3v0Lmax;

    double BSIM3v0Wint;
    double BSIM3v0Wl;
    double BSIM3v0Wln;
    double BSIM3v0Ww;
    double BSIM3v0Wwn;
    double BSIM3v0Wwl;
    double BSIM3v0Wmin;
    double BSIM3v0Wmax;


/* Pre-calculated constants */
    /* MCJ: move to size-dependent param. */
    double BSIM3v0vtm;   
    double BSIM3v0cox;
    double BSIM3v0cof1;
    double BSIM3v0cof2;
    double BSIM3v0cof3;
    double BSIM3v0cof4;
    double BSIM3v0vcrit;
    double BSIM3v0factor1;

    double BSIM3v0oxideTrapDensityA;      
    double BSIM3v0oxideTrapDensityB;     
    double BSIM3v0oxideTrapDensityC;  
    double BSIM3v0em;  
    double BSIM3v0ef;  
    double BSIM3v0af;  
    double BSIM3v0kf;  

    struct bsim3v0SizeDependParam *pSizeDependParamKnot;

    /* Flags */
    unsigned  BSIM3v0mobModGiven :1;
    unsigned  BSIM3v0binUnitGiven :1;
    unsigned  BSIM3v0capModGiven :1;
    unsigned  BSIM3v0nqsModGiven :1;
    unsigned  BSIM3v0noiModGiven :1;
    unsigned  BSIM3v0typeGiven   :1;
    unsigned  BSIM3v0toxGiven   :1;

    unsigned  BSIM3v0cdscGiven   :1;
    unsigned  BSIM3v0cdscbGiven   :1;
    unsigned  BSIM3v0cdscdGiven   :1;
    unsigned  BSIM3v0citGiven   :1;
    unsigned  BSIM3v0nfactorGiven   :1;
    unsigned  BSIM3v0xjGiven   :1;
    unsigned  BSIM3v0vsatGiven   :1;
    unsigned  BSIM3v0atGiven   :1;
    unsigned  BSIM3v0a0Given   :1;
    unsigned  BSIM3v0agsGiven   :1;
    unsigned  BSIM3v0a1Given   :1;
    unsigned  BSIM3v0a2Given   :1;
    unsigned  BSIM3v0ketaGiven   :1;    
    unsigned  BSIM3v0nsubGiven   :1;
    unsigned  BSIM3v0npeakGiven   :1;
    unsigned  BSIM3v0ngateGiven   :1;
    unsigned  BSIM3v0gamma1Given   :1;
    unsigned  BSIM3v0gamma2Given   :1;
    unsigned  BSIM3v0vbxGiven   :1;
    unsigned  BSIM3v0vbmGiven   :1;
    unsigned  BSIM3v0xtGiven   :1;
    unsigned  BSIM3v0k1Given   :1;
    unsigned  BSIM3v0kt1Given   :1;
    unsigned  BSIM3v0kt1lGiven   :1;
    unsigned  BSIM3v0kt2Given   :1;
    unsigned  BSIM3v0k2Given   :1;
    unsigned  BSIM3v0k3Given   :1;
    unsigned  BSIM3v0k3bGiven   :1;
    unsigned  BSIM3v0w0Given   :1;
    unsigned  BSIM3v0nlxGiven   :1;
    unsigned  BSIM3v0dvt0Given   :1;   
    unsigned  BSIM3v0dvt1Given   :1;     
    unsigned  BSIM3v0dvt2Given   :1;     
    unsigned  BSIM3v0dvt0wGiven   :1;   
    unsigned  BSIM3v0dvt1wGiven   :1;     
    unsigned  BSIM3v0dvt2wGiven   :1;     
    unsigned  BSIM3v0droutGiven   :1;     
    unsigned  BSIM3v0dsubGiven   :1;     
    unsigned  BSIM3v0vth0Given   :1;
    unsigned  BSIM3v0uaGiven   :1;
    unsigned  BSIM3v0ua1Given   :1;
    unsigned  BSIM3v0ubGiven   :1;
    unsigned  BSIM3v0ub1Given   :1;
    unsigned  BSIM3v0ucGiven   :1;
    unsigned  BSIM3v0uc1Given   :1;
    unsigned  BSIM3v0u0Given   :1;
    unsigned  BSIM3v0uteGiven   :1;
    unsigned  BSIM3v0voffGiven   :1;
    unsigned  BSIM3v0rdswGiven   :1;      
    unsigned  BSIM3v0prwgGiven   :1;      
    unsigned  BSIM3v0prwbGiven   :1;      
    unsigned  BSIM3v0prtGiven   :1;      
    unsigned  BSIM3v0eta0Given   :1;    
    unsigned  BSIM3v0etabGiven   :1;    
    unsigned  BSIM3v0pclmGiven   :1;   
    unsigned  BSIM3v0pdibl1Given   :1;   
    unsigned  BSIM3v0pdibl2Given   :1;  
    unsigned  BSIM3v0pdiblbGiven   :1;  
    unsigned  BSIM3v0pscbe1Given   :1;    
    unsigned  BSIM3v0pscbe2Given   :1;    
    unsigned  BSIM3v0pvagGiven   :1;    
    unsigned  BSIM3v0deltaGiven  :1;     
    unsigned  BSIM3v0wrGiven   :1;
    unsigned  BSIM3v0dwgGiven   :1;
    unsigned  BSIM3v0dwbGiven   :1;
    unsigned  BSIM3v0b0Given   :1;
    unsigned  BSIM3v0b1Given   :1;
    unsigned  BSIM3v0alpha0Given   :1;
    unsigned  BSIM3v0beta0Given   :1;

    /* CV model */
    unsigned  BSIM3v0elmGiven  :1;     
    unsigned  BSIM3v0cgslGiven   :1;
    unsigned  BSIM3v0cgdlGiven   :1;
    unsigned  BSIM3v0ckappaGiven   :1;
    unsigned  BSIM3v0cfGiven   :1;
    unsigned  BSIM3v0clcGiven   :1;
    unsigned  BSIM3v0cleGiven   :1;
    unsigned  BSIM3v0dwcGiven   :1;
    unsigned  BSIM3v0dlcGiven   :1;


    /* Length dependence */
    unsigned  BSIM3v0lcdscGiven   :1;
    unsigned  BSIM3v0lcdscbGiven   :1;
    unsigned  BSIM3v0lcdscdGiven   :1;
    unsigned  BSIM3v0lcitGiven   :1;
    unsigned  BSIM3v0lnfactorGiven   :1;
    unsigned  BSIM3v0lxjGiven   :1;
    unsigned  BSIM3v0lvsatGiven   :1;
    unsigned  BSIM3v0latGiven   :1;
    unsigned  BSIM3v0la0Given   :1;
    unsigned  BSIM3v0lagsGiven   :1;
    unsigned  BSIM3v0la1Given   :1;
    unsigned  BSIM3v0la2Given   :1;
    unsigned  BSIM3v0lketaGiven   :1;    
    unsigned  BSIM3v0lnsubGiven   :1;
    unsigned  BSIM3v0lnpeakGiven   :1;
    unsigned  BSIM3v0lngateGiven   :1;
    unsigned  BSIM3v0lgamma1Given   :1;
    unsigned  BSIM3v0lgamma2Given   :1;
    unsigned  BSIM3v0lvbxGiven   :1;
    unsigned  BSIM3v0lvbmGiven   :1;
    unsigned  BSIM3v0lxtGiven   :1;
    unsigned  BSIM3v0lk1Given   :1;
    unsigned  BSIM3v0lkt1Given   :1;
    unsigned  BSIM3v0lkt1lGiven   :1;
    unsigned  BSIM3v0lkt2Given   :1;
    unsigned  BSIM3v0lk2Given   :1;
    unsigned  BSIM3v0lk3Given   :1;
    unsigned  BSIM3v0lk3bGiven   :1;
    unsigned  BSIM3v0lw0Given   :1;
    unsigned  BSIM3v0lnlxGiven   :1;
    unsigned  BSIM3v0ldvt0Given   :1;   
    unsigned  BSIM3v0ldvt1Given   :1;     
    unsigned  BSIM3v0ldvt2Given   :1;     
    unsigned  BSIM3v0ldvt0wGiven   :1;   
    unsigned  BSIM3v0ldvt1wGiven   :1;     
    unsigned  BSIM3v0ldvt2wGiven   :1;     
    unsigned  BSIM3v0ldroutGiven   :1;     
    unsigned  BSIM3v0ldsubGiven   :1;     
    unsigned  BSIM3v0lvth0Given   :1;
    unsigned  BSIM3v0luaGiven   :1;
    unsigned  BSIM3v0lua1Given   :1;
    unsigned  BSIM3v0lubGiven   :1;
    unsigned  BSIM3v0lub1Given   :1;
    unsigned  BSIM3v0lucGiven   :1;
    unsigned  BSIM3v0luc1Given   :1;
    unsigned  BSIM3v0lu0Given   :1;
    unsigned  BSIM3v0luteGiven   :1;
    unsigned  BSIM3v0lvoffGiven   :1;
    unsigned  BSIM3v0lrdswGiven   :1;      
    unsigned  BSIM3v0lprwgGiven   :1;      
    unsigned  BSIM3v0lprwbGiven   :1;      
    unsigned  BSIM3v0lprtGiven   :1;      
    unsigned  BSIM3v0leta0Given   :1;    
    unsigned  BSIM3v0letabGiven   :1;    
    unsigned  BSIM3v0lpclmGiven   :1;   
    unsigned  BSIM3v0lpdibl1Given   :1;   
    unsigned  BSIM3v0lpdibl2Given   :1;  
    unsigned  BSIM3v0lpdiblbGiven   :1;  
    unsigned  BSIM3v0lpscbe1Given   :1;    
    unsigned  BSIM3v0lpscbe2Given   :1;    
    unsigned  BSIM3v0lpvagGiven   :1;    
    unsigned  BSIM3v0ldeltaGiven  :1;     
    unsigned  BSIM3v0lwrGiven   :1;
    unsigned  BSIM3v0ldwgGiven   :1;
    unsigned  BSIM3v0ldwbGiven   :1;
    unsigned  BSIM3v0lb0Given   :1;
    unsigned  BSIM3v0lb1Given   :1;
    unsigned  BSIM3v0lalpha0Given   :1;
    unsigned  BSIM3v0lbeta0Given   :1;

    /* CV model */
    unsigned  BSIM3v0lelmGiven  :1;     
    unsigned  BSIM3v0lcgslGiven   :1;
    unsigned  BSIM3v0lcgdlGiven   :1;
    unsigned  BSIM3v0lckappaGiven   :1;
    unsigned  BSIM3v0lcfGiven   :1;
    unsigned  BSIM3v0lclcGiven   :1;
    unsigned  BSIM3v0lcleGiven   :1;

    /* Width dependence */
    unsigned  BSIM3v0wcdscGiven   :1;
    unsigned  BSIM3v0wcdscbGiven   :1;
    unsigned  BSIM3v0wcdscdGiven   :1;
    unsigned  BSIM3v0wcitGiven   :1;
    unsigned  BSIM3v0wnfactorGiven   :1;
    unsigned  BSIM3v0wxjGiven   :1;
    unsigned  BSIM3v0wvsatGiven   :1;
    unsigned  BSIM3v0watGiven   :1;
    unsigned  BSIM3v0wa0Given   :1;
    unsigned  BSIM3v0wagsGiven   :1;
    unsigned  BSIM3v0wa1Given   :1;
    unsigned  BSIM3v0wa2Given   :1;
    unsigned  BSIM3v0wketaGiven   :1;    
    unsigned  BSIM3v0wnsubGiven   :1;
    unsigned  BSIM3v0wnpeakGiven   :1;
    unsigned  BSIM3v0wngateGiven   :1;
    unsigned  BSIM3v0wgamma1Given   :1;
    unsigned  BSIM3v0wgamma2Given   :1;
    unsigned  BSIM3v0wvbxGiven   :1;
    unsigned  BSIM3v0wvbmGiven   :1;
    unsigned  BSIM3v0wxtGiven   :1;
    unsigned  BSIM3v0wk1Given   :1;
    unsigned  BSIM3v0wkt1Given   :1;
    unsigned  BSIM3v0wkt1lGiven   :1;
    unsigned  BSIM3v0wkt2Given   :1;
    unsigned  BSIM3v0wk2Given   :1;
    unsigned  BSIM3v0wk3Given   :1;
    unsigned  BSIM3v0wk3bGiven   :1;
    unsigned  BSIM3v0ww0Given   :1;
    unsigned  BSIM3v0wnlxGiven   :1;
    unsigned  BSIM3v0wdvt0Given   :1;   
    unsigned  BSIM3v0wdvt1Given   :1;     
    unsigned  BSIM3v0wdvt2Given   :1;     
    unsigned  BSIM3v0wdvt0wGiven   :1;   
    unsigned  BSIM3v0wdvt1wGiven   :1;     
    unsigned  BSIM3v0wdvt2wGiven   :1;     
    unsigned  BSIM3v0wdroutGiven   :1;     
    unsigned  BSIM3v0wdsubGiven   :1;     
    unsigned  BSIM3v0wvth0Given   :1;
    unsigned  BSIM3v0wuaGiven   :1;
    unsigned  BSIM3v0wua1Given   :1;
    unsigned  BSIM3v0wubGiven   :1;
    unsigned  BSIM3v0wub1Given   :1;
    unsigned  BSIM3v0wucGiven   :1;
    unsigned  BSIM3v0wuc1Given   :1;
    unsigned  BSIM3v0wu0Given   :1;
    unsigned  BSIM3v0wuteGiven   :1;
    unsigned  BSIM3v0wvoffGiven   :1;
    unsigned  BSIM3v0wrdswGiven   :1;      
    unsigned  BSIM3v0wprwgGiven   :1;      
    unsigned  BSIM3v0wprwbGiven   :1;      
    unsigned  BSIM3v0wprtGiven   :1;      
    unsigned  BSIM3v0weta0Given   :1;    
    unsigned  BSIM3v0wetabGiven   :1;    
    unsigned  BSIM3v0wpclmGiven   :1;   
    unsigned  BSIM3v0wpdibl1Given   :1;   
    unsigned  BSIM3v0wpdibl2Given   :1;  
    unsigned  BSIM3v0wpdiblbGiven   :1;  
    unsigned  BSIM3v0wpscbe1Given   :1;    
    unsigned  BSIM3v0wpscbe2Given   :1;    
    unsigned  BSIM3v0wpvagGiven   :1;    
    unsigned  BSIM3v0wdeltaGiven  :1;     
    unsigned  BSIM3v0wwrGiven   :1;
    unsigned  BSIM3v0wdwgGiven   :1;
    unsigned  BSIM3v0wdwbGiven   :1;
    unsigned  BSIM3v0wb0Given   :1;
    unsigned  BSIM3v0wb1Given   :1;
    unsigned  BSIM3v0walpha0Given   :1;
    unsigned  BSIM3v0wbeta0Given   :1;

    /* CV model */
    unsigned  BSIM3v0welmGiven  :1;     
    unsigned  BSIM3v0wcgslGiven   :1;
    unsigned  BSIM3v0wcgdlGiven   :1;
    unsigned  BSIM3v0wckappaGiven   :1;
    unsigned  BSIM3v0wcfGiven   :1;
    unsigned  BSIM3v0wclcGiven   :1;
    unsigned  BSIM3v0wcleGiven   :1;

    /* Cross-term dependence */
    unsigned  BSIM3v0pcdscGiven   :1;
    unsigned  BSIM3v0pcdscbGiven   :1;
    unsigned  BSIM3v0pcdscdGiven   :1;
    unsigned  BSIM3v0pcitGiven   :1;
    unsigned  BSIM3v0pnfactorGiven   :1;
    unsigned  BSIM3v0pxjGiven   :1;
    unsigned  BSIM3v0pvsatGiven   :1;
    unsigned  BSIM3v0patGiven   :1;
    unsigned  BSIM3v0pa0Given   :1;
    unsigned  BSIM3v0pagsGiven   :1;
    unsigned  BSIM3v0pa1Given   :1;
    unsigned  BSIM3v0pa2Given   :1;
    unsigned  BSIM3v0pketaGiven   :1;    
    unsigned  BSIM3v0pnsubGiven   :1;
    unsigned  BSIM3v0pnpeakGiven   :1;
    unsigned  BSIM3v0pngateGiven   :1;
    unsigned  BSIM3v0pgamma1Given   :1;
    unsigned  BSIM3v0pgamma2Given   :1;
    unsigned  BSIM3v0pvbxGiven   :1;
    unsigned  BSIM3v0pvbmGiven   :1;
    unsigned  BSIM3v0pxtGiven   :1;
    unsigned  BSIM3v0pk1Given   :1;
    unsigned  BSIM3v0pkt1Given   :1;
    unsigned  BSIM3v0pkt1lGiven   :1;
    unsigned  BSIM3v0pkt2Given   :1;
    unsigned  BSIM3v0pk2Given   :1;
    unsigned  BSIM3v0pk3Given   :1;
    unsigned  BSIM3v0pk3bGiven   :1;
    unsigned  BSIM3v0pw0Given   :1;
    unsigned  BSIM3v0pnlxGiven   :1;
    unsigned  BSIM3v0pdvt0Given   :1;   
    unsigned  BSIM3v0pdvt1Given   :1;     
    unsigned  BSIM3v0pdvt2Given   :1;     
    unsigned  BSIM3v0pdvt0wGiven   :1;   
    unsigned  BSIM3v0pdvt1wGiven   :1;     
    unsigned  BSIM3v0pdvt2wGiven   :1;     
    unsigned  BSIM3v0pdroutGiven   :1;     
    unsigned  BSIM3v0pdsubGiven   :1;     
    unsigned  BSIM3v0pvth0Given   :1;
    unsigned  BSIM3v0puaGiven   :1;
    unsigned  BSIM3v0pua1Given   :1;
    unsigned  BSIM3v0pubGiven   :1;
    unsigned  BSIM3v0pub1Given   :1;
    unsigned  BSIM3v0pucGiven   :1;
    unsigned  BSIM3v0puc1Given   :1;
    unsigned  BSIM3v0pu0Given   :1;
    unsigned  BSIM3v0puteGiven   :1;
    unsigned  BSIM3v0pvoffGiven   :1;
    unsigned  BSIM3v0prdswGiven   :1;      
    unsigned  BSIM3v0pprwgGiven   :1;      
    unsigned  BSIM3v0pprwbGiven   :1;      
    unsigned  BSIM3v0pprtGiven   :1;      
    unsigned  BSIM3v0peta0Given   :1;    
    unsigned  BSIM3v0petabGiven   :1;    
    unsigned  BSIM3v0ppclmGiven   :1;   
    unsigned  BSIM3v0ppdibl1Given   :1;   
    unsigned  BSIM3v0ppdibl2Given   :1;  
    unsigned  BSIM3v0ppdiblbGiven   :1;  
    unsigned  BSIM3v0ppscbe1Given   :1;    
    unsigned  BSIM3v0ppscbe2Given   :1;    
    unsigned  BSIM3v0ppvagGiven   :1;    
    unsigned  BSIM3v0pdeltaGiven  :1;     
    unsigned  BSIM3v0pwrGiven   :1;
    unsigned  BSIM3v0pdwgGiven   :1;
    unsigned  BSIM3v0pdwbGiven   :1;
    unsigned  BSIM3v0pb0Given   :1;
    unsigned  BSIM3v0pb1Given   :1;
    unsigned  BSIM3v0palpha0Given   :1;
    unsigned  BSIM3v0pbeta0Given   :1;

    /* CV model */
    unsigned  BSIM3v0pelmGiven  :1;     
    unsigned  BSIM3v0pcgslGiven   :1;
    unsigned  BSIM3v0pcgdlGiven   :1;
    unsigned  BSIM3v0pckappaGiven   :1;
    unsigned  BSIM3v0pcfGiven   :1;
    unsigned  BSIM3v0pclcGiven   :1;
    unsigned  BSIM3v0pcleGiven   :1;

    unsigned  BSIM3v0useFringeGiven   :1;

    unsigned  BSIM3v0tnomGiven   :1;
    unsigned  BSIM3v0cgsoGiven   :1;
    unsigned  BSIM3v0cgdoGiven   :1;
    unsigned  BSIM3v0cgboGiven   :1;
    unsigned  BSIM3v0xpartGiven   :1;
    unsigned  BSIM3v0sheetResistanceGiven   :1;
    unsigned  BSIM3v0jctSatCurDensityGiven   :1;
    unsigned  BSIM3v0bulkJctPotentialGiven   :1;
    unsigned  BSIM3v0bulkJctBotGradingCoeffGiven   :1;
    unsigned  BSIM3v0sidewallJctPotentialGiven   :1;
    unsigned  BSIM3v0bulkJctSideGradingCoeffGiven   :1;
    unsigned  BSIM3v0unitAreaJctCapGiven   :1;
    unsigned  BSIM3v0unitLengthSidewallJctCapGiven   :1;

    unsigned  BSIM3v0oxideTrapDensityAGiven  :1;         
    unsigned  BSIM3v0oxideTrapDensityBGiven  :1;        
    unsigned  BSIM3v0oxideTrapDensityCGiven  :1;     
    unsigned  BSIM3v0emGiven  :1;     
    unsigned  BSIM3v0efGiven  :1;     
    unsigned  BSIM3v0afGiven  :1;     
    unsigned  BSIM3v0kfGiven  :1;     

    unsigned  BSIM3v0LintGiven   :1;
    unsigned  BSIM3v0LlGiven   :1;
    unsigned  BSIM3v0LlnGiven   :1;
    unsigned  BSIM3v0LwGiven   :1;
    unsigned  BSIM3v0LwnGiven   :1;
    unsigned  BSIM3v0LwlGiven   :1;
    unsigned  BSIM3v0LminGiven   :1;
    unsigned  BSIM3v0LmaxGiven   :1;

    unsigned  BSIM3v0WintGiven   :1;
    unsigned  BSIM3v0WlGiven   :1;
    unsigned  BSIM3v0WlnGiven   :1;
    unsigned  BSIM3v0WwGiven   :1;
    unsigned  BSIM3v0WwnGiven   :1;
    unsigned  BSIM3v0WwlGiven   :1;
    unsigned  BSIM3v0WminGiven   :1;
    unsigned  BSIM3v0WmaxGiven   :1;

} BSIM3v0model;


#ifndef NMOS
#define NMOS 1
#define PMOS -1
#endif /*NMOS*/


/* device parameters */
#define BSIM3v0_W 1
#define BSIM3v0_L 2
#define BSIM3v0_M 15
#define BSIM3v0_AS 3
#define BSIM3v0_AD 4
#define BSIM3v0_PS 5
#define BSIM3v0_PD 6
#define BSIM3v0_NRS 7
#define BSIM3v0_NRD 8
#define BSIM3v0_OFF 9
#define BSIM3v0_IC_VBS 10
#define BSIM3v0_IC_VDS 11
#define BSIM3v0_IC_VGS 12
#define BSIM3v0_IC 13
#define BSIM3v0_NQSMOD 14

/* model parameters */
#define BSIM3v0_MOD_CAPMOD          101
#define BSIM3v0_MOD_NQSMOD          102
#define BSIM3v0_MOD_MOBMOD          103    
#define BSIM3v0_MOD_NOIMOD          104    

#define BSIM3v0_MOD_TOX             105

#define BSIM3v0_MOD_CDSC            106
#define BSIM3v0_MOD_CDSCB           107
#define BSIM3v0_MOD_CIT             108
#define BSIM3v0_MOD_NFACTOR         109
#define BSIM3v0_MOD_XJ              110
#define BSIM3v0_MOD_VSAT            111
#define BSIM3v0_MOD_AT              112
#define BSIM3v0_MOD_A0              113
#define BSIM3v0_MOD_A1              114
#define BSIM3v0_MOD_A2              115
#define BSIM3v0_MOD_KETA            116   
#define BSIM3v0_MOD_NSUB            117
#define BSIM3v0_MOD_NPEAK           118
#define BSIM3v0_MOD_NGATE           120
#define BSIM3v0_MOD_GAMMA1          121
#define BSIM3v0_MOD_GAMMA2          122
#define BSIM3v0_MOD_VBX             123
#define BSIM3v0_MOD_BINUNIT         124    

#define BSIM3v0_MOD_VBM             125

#define BSIM3v0_MOD_XT              126
#define BSIM3v0_MOD_K1              129
#define BSIM3v0_MOD_KT1             130
#define BSIM3v0_MOD_KT1L            131
#define BSIM3v0_MOD_K2              132
#define BSIM3v0_MOD_KT2             133
#define BSIM3v0_MOD_K3              134
#define BSIM3v0_MOD_K3B             135
#define BSIM3v0_MOD_W0              136
#define BSIM3v0_MOD_NLX             137

#define BSIM3v0_MOD_DVT0            138
#define BSIM3v0_MOD_DVT1            139
#define BSIM3v0_MOD_DVT2            140

#define BSIM3v0_MOD_DVT0W           141
#define BSIM3v0_MOD_DVT1W           142
#define BSIM3v0_MOD_DVT2W           143

#define BSIM3v0_MOD_DROUT           144
#define BSIM3v0_MOD_DSUB            145
#define BSIM3v0_MOD_VTH0            146
#define BSIM3v0_MOD_UA              147
#define BSIM3v0_MOD_UA1             148
#define BSIM3v0_MOD_UB              149
#define BSIM3v0_MOD_UB1             150
#define BSIM3v0_MOD_UC              151
#define BSIM3v0_MOD_UC1             152
#define BSIM3v0_MOD_U0              153
#define BSIM3v0_MOD_UTE             154
#define BSIM3v0_MOD_VOFF            155
#define BSIM3v0_MOD_DELTA           156
#define BSIM3v0_MOD_RDSW            157
#define BSIM3v0_MOD_PRT             158
#define BSIM3v0_MOD_LDD             159
#define BSIM3v0_MOD_ETA             160
#define BSIM3v0_MOD_ETA0            161
#define BSIM3v0_MOD_ETAB            162
#define BSIM3v0_MOD_PCLM            163
#define BSIM3v0_MOD_PDIBL1          164
#define BSIM3v0_MOD_PDIBL2          165
#define BSIM3v0_MOD_PSCBE1          166
#define BSIM3v0_MOD_PSCBE2          167
#define BSIM3v0_MOD_PVAG            168
#define BSIM3v0_MOD_WR              169
#define BSIM3v0_MOD_DWG             170
#define BSIM3v0_MOD_DWB             171
#define BSIM3v0_MOD_B0              172
#define BSIM3v0_MOD_B1              173
#define BSIM3v0_MOD_ALPHA0          174
#define BSIM3v0_MOD_BETA0           175
#define BSIM3v0_MOD_PDIBLB          178

#define BSIM3v0_MOD_PRWG            179
#define BSIM3v0_MOD_PRWB            180

#define BSIM3v0_MOD_CDSCD           181
#define BSIM3v0_MOD_AGS             182

#define BSIM3v0_MOD_FRINGE          184
#define BSIM3v0_MOD_ELM             185
#define BSIM3v0_MOD_CGSL            186
#define BSIM3v0_MOD_CGDL            187
#define BSIM3v0_MOD_CKAPPA          188
#define BSIM3v0_MOD_CF              189
#define BSIM3v0_MOD_CLC             190
#define BSIM3v0_MOD_CLE             191

/* Length dependence */
#define BSIM3v0_MOD_LCDSC            201
#define BSIM3v0_MOD_LCDSCB           202
#define BSIM3v0_MOD_LCIT             203
#define BSIM3v0_MOD_LNFACTOR         204
#define BSIM3v0_MOD_LXJ              205
#define BSIM3v0_MOD_LVSAT            206
#define BSIM3v0_MOD_LAT              207
#define BSIM3v0_MOD_LA0              208
#define BSIM3v0_MOD_LA1              209
#define BSIM3v0_MOD_LA2              210
#define BSIM3v0_MOD_LKETA            211   
#define BSIM3v0_MOD_LNSUB            212
#define BSIM3v0_MOD_LNPEAK           213
#define BSIM3v0_MOD_LNGATE           215
#define BSIM3v0_MOD_LGAMMA1          216
#define BSIM3v0_MOD_LGAMMA2          217
#define BSIM3v0_MOD_LVBX             218

#define BSIM3v0_MOD_LVBM             220

#define BSIM3v0_MOD_LXT              222
#define BSIM3v0_MOD_LK1              225
#define BSIM3v0_MOD_LKT1             226
#define BSIM3v0_MOD_LKT1L            227
#define BSIM3v0_MOD_LK2              228
#define BSIM3v0_MOD_LKT2             229
#define BSIM3v0_MOD_LK3              230
#define BSIM3v0_MOD_LK3B             231
#define BSIM3v0_MOD_LW0              232
#define BSIM3v0_MOD_LNLX             233

#define BSIM3v0_MOD_LDVT0            234
#define BSIM3v0_MOD_LDVT1            235
#define BSIM3v0_MOD_LDVT2            236

#define BSIM3v0_MOD_LDVT0W           237
#define BSIM3v0_MOD_LDVT1W           238
#define BSIM3v0_MOD_LDVT2W           239

#define BSIM3v0_MOD_LDROUT           240
#define BSIM3v0_MOD_LDSUB            241
#define BSIM3v0_MOD_LVTH0            242
#define BSIM3v0_MOD_LUA              243
#define BSIM3v0_MOD_LUA1             244
#define BSIM3v0_MOD_LUB              245
#define BSIM3v0_MOD_LUB1             246
#define BSIM3v0_MOD_LUC              247
#define BSIM3v0_MOD_LUC1             248
#define BSIM3v0_MOD_LU0              249
#define BSIM3v0_MOD_LUTE             250
#define BSIM3v0_MOD_LVOFF            251
#define BSIM3v0_MOD_LDELTA           252
#define BSIM3v0_MOD_LRDSW            253
#define BSIM3v0_MOD_LPRT             254
#define BSIM3v0_MOD_LLDD             255
#define BSIM3v0_MOD_LETA             256
#define BSIM3v0_MOD_LETA0            257
#define BSIM3v0_MOD_LETAB            258
#define BSIM3v0_MOD_LPCLM            259
#define BSIM3v0_MOD_LPDIBL1          260
#define BSIM3v0_MOD_LPDIBL2          261
#define BSIM3v0_MOD_LPSCBE1          262
#define BSIM3v0_MOD_LPSCBE2          263
#define BSIM3v0_MOD_LPVAG            264
#define BSIM3v0_MOD_LWR              265
#define BSIM3v0_MOD_LDWG             266
#define BSIM3v0_MOD_LDWB             267
#define BSIM3v0_MOD_LB0              268
#define BSIM3v0_MOD_LB1              269
#define BSIM3v0_MOD_LALPHA0          270
#define BSIM3v0_MOD_LBETA0           271
#define BSIM3v0_MOD_LPDIBLB          274

#define BSIM3v0_MOD_LPRWG            275
#define BSIM3v0_MOD_LPRWB            276

#define BSIM3v0_MOD_LCDSCD           277
#define BSIM3v0_MOD_LAGS            278
                                    

#define BSIM3v0_MOD_LFRINGE          281
#define BSIM3v0_MOD_LELM             282
#define BSIM3v0_MOD_LCGSL            283
#define BSIM3v0_MOD_LCGDL            284
#define BSIM3v0_MOD_LCKAPPA          285
#define BSIM3v0_MOD_LCF              286
#define BSIM3v0_MOD_LCLC             287
#define BSIM3v0_MOD_LCLE             288

/* Width dependence */
#define BSIM3v0_MOD_WCDSC            301
#define BSIM3v0_MOD_WCDSCB           302
#define BSIM3v0_MOD_WCIT             303
#define BSIM3v0_MOD_WNFACTOR         304
#define BSIM3v0_MOD_WXJ              305
#define BSIM3v0_MOD_WVSAT            306
#define BSIM3v0_MOD_WAT              307
#define BSIM3v0_MOD_WA0              308
#define BSIM3v0_MOD_WA1              309
#define BSIM3v0_MOD_WA2              310
#define BSIM3v0_MOD_WKETA            311   
#define BSIM3v0_MOD_WNSUB            312
#define BSIM3v0_MOD_WNPEAK           313
#define BSIM3v0_MOD_WNGATE           315
#define BSIM3v0_MOD_WGAMMA1          316
#define BSIM3v0_MOD_WGAMMA2          317
#define BSIM3v0_MOD_WVBX             318

#define BSIM3v0_MOD_WVBM             320

#define BSIM3v0_MOD_WXT              322
#define BSIM3v0_MOD_WK1              325
#define BSIM3v0_MOD_WKT1             326
#define BSIM3v0_MOD_WKT1L            327
#define BSIM3v0_MOD_WK2              328
#define BSIM3v0_MOD_WKT2             329
#define BSIM3v0_MOD_WK3              330
#define BSIM3v0_MOD_WK3B             331
#define BSIM3v0_MOD_WW0              332
#define BSIM3v0_MOD_WNLX             333

#define BSIM3v0_MOD_WDVT0            334
#define BSIM3v0_MOD_WDVT1            335
#define BSIM3v0_MOD_WDVT2            336

#define BSIM3v0_MOD_WDVT0W           337
#define BSIM3v0_MOD_WDVT1W           338
#define BSIM3v0_MOD_WDVT2W           339

#define BSIM3v0_MOD_WDROUT           340
#define BSIM3v0_MOD_WDSUB            341
#define BSIM3v0_MOD_WVTH0            342
#define BSIM3v0_MOD_WUA              343
#define BSIM3v0_MOD_WUA1             344
#define BSIM3v0_MOD_WUB              345
#define BSIM3v0_MOD_WUB1             346
#define BSIM3v0_MOD_WUC              347
#define BSIM3v0_MOD_WUC1             348
#define BSIM3v0_MOD_WU0              349
#define BSIM3v0_MOD_WUTE             350
#define BSIM3v0_MOD_WVOFF            351
#define BSIM3v0_MOD_WDELTA           352
#define BSIM3v0_MOD_WRDSW            353
#define BSIM3v0_MOD_WPRT             354
#define BSIM3v0_MOD_WLDD             355
#define BSIM3v0_MOD_WETA             356
#define BSIM3v0_MOD_WETA0            357
#define BSIM3v0_MOD_WETAB            358
#define BSIM3v0_MOD_WPCLM            359
#define BSIM3v0_MOD_WPDIBL1          360
#define BSIM3v0_MOD_WPDIBL2          361
#define BSIM3v0_MOD_WPSCBE1          362
#define BSIM3v0_MOD_WPSCBE2          363
#define BSIM3v0_MOD_WPVAG            364
#define BSIM3v0_MOD_WWR              365
#define BSIM3v0_MOD_WDWG             366
#define BSIM3v0_MOD_WDWB             367
#define BSIM3v0_MOD_WB0              368
#define BSIM3v0_MOD_WB1              369
#define BSIM3v0_MOD_WALPHA0          370
#define BSIM3v0_MOD_WBETA0           371
#define BSIM3v0_MOD_WPDIBLB          374

#define BSIM3v0_MOD_WPRWG            375
#define BSIM3v0_MOD_WPRWB            376

#define BSIM3v0_MOD_WCDSCD           377
#define BSIM3v0_MOD_WAGS             378


#define BSIM3v0_MOD_WFRINGE          381
#define BSIM3v0_MOD_WELM             382
#define BSIM3v0_MOD_WCGSL            383
#define BSIM3v0_MOD_WCGDL            384
#define BSIM3v0_MOD_WCKAPPA          385
#define BSIM3v0_MOD_WCF              386
#define BSIM3v0_MOD_WCLC             387
#define BSIM3v0_MOD_WCLE             388

/* Cross-term dependence */
#define BSIM3v0_MOD_PCDSC            401
#define BSIM3v0_MOD_PCDSCB           402
#define BSIM3v0_MOD_PCIT             403
#define BSIM3v0_MOD_PNFACTOR         404
#define BSIM3v0_MOD_PXJ              405
#define BSIM3v0_MOD_PVSAT            406
#define BSIM3v0_MOD_PAT              407
#define BSIM3v0_MOD_PA0              408
#define BSIM3v0_MOD_PA1              409
#define BSIM3v0_MOD_PA2              410
#define BSIM3v0_MOD_PKETA            411   
#define BSIM3v0_MOD_PNSUB            412
#define BSIM3v0_MOD_PNPEAK           413
#define BSIM3v0_MOD_PNGATE           415
#define BSIM3v0_MOD_PGAMMA1          416
#define BSIM3v0_MOD_PGAMMA2          417
#define BSIM3v0_MOD_PVBX             418

#define BSIM3v0_MOD_PVBM             420

#define BSIM3v0_MOD_PXT              422
#define BSIM3v0_MOD_PK1              425
#define BSIM3v0_MOD_PKT1             426
#define BSIM3v0_MOD_PKT1L            427
#define BSIM3v0_MOD_PK2              428
#define BSIM3v0_MOD_PKT2             429
#define BSIM3v0_MOD_PK3              430
#define BSIM3v0_MOD_PK3B             431
#define BSIM3v0_MOD_PW0              432
#define BSIM3v0_MOD_PNLX             433

#define BSIM3v0_MOD_PDVT0            434
#define BSIM3v0_MOD_PDVT1            435
#define BSIM3v0_MOD_PDVT2            436

#define BSIM3v0_MOD_PDVT0W           437
#define BSIM3v0_MOD_PDVT1W           438
#define BSIM3v0_MOD_PDVT2W           439

#define BSIM3v0_MOD_PDROUT           440
#define BSIM3v0_MOD_PDSUB            441
#define BSIM3v0_MOD_PVTH0            442
#define BSIM3v0_MOD_PUA              443
#define BSIM3v0_MOD_PUA1             444
#define BSIM3v0_MOD_PUB              445
#define BSIM3v0_MOD_PUB1             446
#define BSIM3v0_MOD_PUC              447
#define BSIM3v0_MOD_PUC1             448
#define BSIM3v0_MOD_PU0              449
#define BSIM3v0_MOD_PUTE             450
#define BSIM3v0_MOD_PVOFF            451
#define BSIM3v0_MOD_PDELTA           452
#define BSIM3v0_MOD_PRDSW            453
#define BSIM3v0_MOD_PPRT             454
#define BSIM3v0_MOD_PLDD             455
#define BSIM3v0_MOD_PETA             456
#define BSIM3v0_MOD_PETA0            457
#define BSIM3v0_MOD_PETAB            458
#define BSIM3v0_MOD_PPCLM            459
#define BSIM3v0_MOD_PPDIBL1          460
#define BSIM3v0_MOD_PPDIBL2          461
#define BSIM3v0_MOD_PPSCBE1          462
#define BSIM3v0_MOD_PPSCBE2          463
#define BSIM3v0_MOD_PPVAG            464
#define BSIM3v0_MOD_PWR              465
#define BSIM3v0_MOD_PDWG             466
#define BSIM3v0_MOD_PDWB             467
#define BSIM3v0_MOD_PB0              468
#define BSIM3v0_MOD_PB1              469
#define BSIM3v0_MOD_PALPHA0          470
#define BSIM3v0_MOD_PBETA0           471
#define BSIM3v0_MOD_PPDIBLB          474

#define BSIM3v0_MOD_PPRWG            475
#define BSIM3v0_MOD_PPRWB            476

#define BSIM3v0_MOD_PCDSCD           477
#define BSIM3v0_MOD_PAGS             478

#define BSIM3v0_MOD_PFRINGE          481
#define BSIM3v0_MOD_PELM             482
#define BSIM3v0_MOD_PCGSL            483
#define BSIM3v0_MOD_PCGDL            484
#define BSIM3v0_MOD_PCKAPPA          485
#define BSIM3v0_MOD_PCF              486
#define BSIM3v0_MOD_PCLC             487
#define BSIM3v0_MOD_PCLE             488

#define BSIM3v0_MOD_TNOM             501
#define BSIM3v0_MOD_CGSO             502
#define BSIM3v0_MOD_CGDO             503
#define BSIM3v0_MOD_CGBO             504
#define BSIM3v0_MOD_XPART            505

#define BSIM3v0_MOD_RSH              506
#define BSIM3v0_MOD_JS               507
#define BSIM3v0_MOD_PB               508
#define BSIM3v0_MOD_MJ               509
#define BSIM3v0_MOD_PBSW             510
#define BSIM3v0_MOD_MJSW             511
#define BSIM3v0_MOD_CJ               512
#define BSIM3v0_MOD_CJSW             513
#define BSIM3v0_MOD_NMOS             514
#define BSIM3v0_MOD_PMOS             515

#define BSIM3v0_MOD_NOIA             516
#define BSIM3v0_MOD_NOIB             517
#define BSIM3v0_MOD_NOIC             518

#define BSIM3v0_MOD_LINT             519
#define BSIM3v0_MOD_LL               520
#define BSIM3v0_MOD_LLN              521
#define BSIM3v0_MOD_LW               522
#define BSIM3v0_MOD_LWN              523
#define BSIM3v0_MOD_LWL              524
#define BSIM3v0_MOD_LMIN             525
#define BSIM3v0_MOD_LMAX             526

#define BSIM3v0_MOD_WINT             527
#define BSIM3v0_MOD_WL               528
#define BSIM3v0_MOD_WLN              529
#define BSIM3v0_MOD_WW               530
#define BSIM3v0_MOD_WWN              531
#define BSIM3v0_MOD_WWL              532
#define BSIM3v0_MOD_WMIN             533
#define BSIM3v0_MOD_WMAX             534

#define BSIM3v0_MOD_DWC              535
#define BSIM3v0_MOD_DLC              536

#define BSIM3v0_MOD_EM               537
#define BSIM3v0_MOD_EF               538
#define BSIM3v0_MOD_AF               539
#define BSIM3v0_MOD_KF               540


/* device questions */
#define BSIM3v0_DNODE                601
#define BSIM3v0_GNODE                602
#define BSIM3v0_SNODE                603
#define BSIM3v0_BNODE                604
#define BSIM3v0_DNODEPRIME           605
#define BSIM3v0_SNODEPRIME           606
#define BSIM3v0_VBD                  607
#define BSIM3v0_VBS                  608
#define BSIM3v0_VGS                  609
#define BSIM3v0_VDS                  610
#define BSIM3v0_CD                   611
#define BSIM3v0_CBS                  612
#define BSIM3v0_CBD                  613
#define BSIM3v0_GM                   614
#define BSIM3v0_GDS                  615
#define BSIM3v0_GMBS                 616
#define BSIM3v0_GBD                  617
#define BSIM3v0_GBS                  618
#define BSIM3v0_QB                   619
#define BSIM3v0_CQB                  620
#define BSIM3v0_QG                   621
#define BSIM3v0_CQG                  622
#define BSIM3v0_QD                   623
#define BSIM3v0_CQD                  624
#define BSIM3v0_CGG                  625
#define BSIM3v0_CGD                  626
#define BSIM3v0_CGS                  627
#define BSIM3v0_CBG                  628
#define BSIM3v0_CAPBD                629
#define BSIM3v0_CQBD                 630
#define BSIM3v0_CAPBS                631
#define BSIM3v0_CQBS                 632
#define BSIM3v0_CDG                  633
#define BSIM3v0_CDD                  634
#define BSIM3v0_CDS                  635
#define BSIM3v0_VON                  636
#define BSIM3v0_VDSAT                637
#define BSIM3v0_QBS                  638
#define BSIM3v0_QBD                  639
#define BSIM3v0_SOURCECONDUCT        640
#define BSIM3v0_DRAINCONDUCT         641
#define BSIM3v0_CBDB                 642
#define BSIM3v0_CBSB                 643

#include "bsim3v0ext.h"

extern void BSIM3v0evaluate(double,double,double,BSIM3v0instance*,BSIM3v0model*,
        double*,double*,double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, CKTcircuit*);

#endif /*BSIM3v0*/



