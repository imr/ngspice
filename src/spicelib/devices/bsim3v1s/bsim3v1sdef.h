/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan
Modified by Paolo Nenzi 2002
File: bsim3v1sdef.h
**********/

#ifndef BSIM3v1S
#define BSIM3v1S

#include "ifsim.h"
#include "gendefs.h"
#include "cktdefs.h"
#include "complex.h"
#include "noisedef.h"         

typedef struct sBSIM3v1Sinstance
{
    struct sBSIM3v1Smodel *BSIM3v1SmodPtr;
    struct sBSIM3v1Sinstance *BSIM3v1SnextInstance;
    IFuid BSIM3v1Sname;
    int BSIM3v1Sowner;      /* number of owner process */
    int BSIM3v1Sstates;     /* index into state table for this device */

    int BSIM3v1SdNode;
    int BSIM3v1SgNode;
    int BSIM3v1SsNode;
    int BSIM3v1SbNode;
    int BSIM3v1SdNodePrime;
    int BSIM3v1SsNodePrime;
    int BSIM3v1SqNode; /* MCJ */

    /* MCJ */
    double BSIM3v1Sueff;
    double BSIM3v1Sthetavth; 
    double BSIM3v1Svon;
    double BSIM3v1Svdsat;
    double BSIM3v1Scgdo;
    double BSIM3v1Scgso;

    double BSIM3v1Sl;
    double BSIM3v1Sw;
    double BSIM3v1SdrainArea;
    double BSIM3v1SsourceArea;
    double BSIM3v1SdrainSquares;
    double BSIM3v1SsourceSquares;
    double BSIM3v1SdrainPerimeter;
    double BSIM3v1SsourcePerimeter;
    double BSIM3v1SsourceConductance;
    double BSIM3v1SdrainConductance;
    double BSIM3v1Sm;

    double BSIM3v1SicVBS;
    double BSIM3v1SicVDS;
    double BSIM3v1SicVGS;
    int BSIM3v1Soff;
    int BSIM3v1Smode;
    int BSIM3v1SnqsMod;

    /* OP point */
    double BSIM3v1Sqinv;
    double BSIM3v1Scd;
    double BSIM3v1Scbs;
    double BSIM3v1Scbd;
    double BSIM3v1Scsub;
    double BSIM3v1Sgm;
    double BSIM3v1Sgds;
    double BSIM3v1Sgmbs;
    double BSIM3v1Sgbd;
    double BSIM3v1Sgbs;

    double BSIM3v1Sgbbs;
    double BSIM3v1Sgbgs;
    double BSIM3v1Sgbds;

    double BSIM3v1Scggb;
    double BSIM3v1Scgdb;
    double BSIM3v1Scgsb;
    double BSIM3v1Scbgb;
    double BSIM3v1Scbdb;
    double BSIM3v1Scbsb;
    double BSIM3v1Scdgb;
    double BSIM3v1Scddb;
    double BSIM3v1Scdsb;
    double BSIM3v1Scapbd;
    double BSIM3v1Scapbs;

    double BSIM3v1Scqgb;
    double BSIM3v1Scqdb;
    double BSIM3v1Scqsb;
    double BSIM3v1Scqbb;

    double BSIM3v1Sgtau;
    double BSIM3v1Sgtg;
    double BSIM3v1Sgtd;
    double BSIM3v1Sgts;
    double BSIM3v1Sgtb;
    double BSIM3v1Stconst;

    struct bsim3v1sSizeDependParam  *pParam;

    unsigned BSIM3v1SlGiven :1;
    unsigned BSIM3v1SwGiven :1;
    unsigned BSIM3v1SdrainAreaGiven :1;
    unsigned BSIM3v1SsourceAreaGiven    :1;
    unsigned BSIM3v1SdrainSquaresGiven  :1;
    unsigned BSIM3v1SsourceSquaresGiven :1;
    unsigned BSIM3v1SdrainPerimeterGiven    :1;
    unsigned BSIM3v1SsourcePerimeterGiven   :1;
    unsigned BSIM3v1SdNodePrimeSet  :1;
    unsigned BSIM3v1SsNodePrimeSet  :1;
    unsigned BSIM3v1SicVBSGiven :1;
    unsigned BSIM3v1SicVDSGiven :1;
    unsigned BSIM3v1SicVGSGiven :1;
    unsigned BSIM3v1SnqsModGiven :1;

    double *BSIM3v1SDdPtr;
    double *BSIM3v1SGgPtr;
    double *BSIM3v1SSsPtr;
    double *BSIM3v1SBbPtr;
    double *BSIM3v1SDPdpPtr;
    double *BSIM3v1SSPspPtr;
    double *BSIM3v1SDdpPtr;
    double *BSIM3v1SGbPtr;
    double *BSIM3v1SGdpPtr;
    double *BSIM3v1SGspPtr;
    double *BSIM3v1SSspPtr;
    double *BSIM3v1SBdpPtr;
    double *BSIM3v1SBspPtr;
    double *BSIM3v1SDPspPtr;
    double *BSIM3v1SDPdPtr;
    double *BSIM3v1SBgPtr;
    double *BSIM3v1SDPgPtr;
    double *BSIM3v1SSPgPtr;
    double *BSIM3v1SSPsPtr;
    double *BSIM3v1SDPbPtr;
    double *BSIM3v1SSPbPtr;
    double *BSIM3v1SSPdpPtr;

    double *BSIM3v1SQqPtr;
    double *BSIM3v1SQdpPtr;
    double *BSIM3v1SQgPtr;
    double *BSIM3v1SQspPtr;
    double *BSIM3v1SQbPtr;
    double *BSIM3v1SDPqPtr;
    double *BSIM3v1SGqPtr;
    double *BSIM3v1SSPqPtr;
    double *BSIM3v1SBqPtr;

#define BSIM3v1Svbd BSIM3v1Sstates+ 0
#define BSIM3v1Svbs BSIM3v1Sstates+ 1
#define BSIM3v1Svgs BSIM3v1Sstates+ 2
#define BSIM3v1Svds BSIM3v1Sstates+ 3

#define BSIM3v1Sqb BSIM3v1Sstates+ 4
#define BSIM3v1Scqb BSIM3v1Sstates+ 5
#define BSIM3v1Sqg BSIM3v1Sstates+ 6
#define BSIM3v1Scqg BSIM3v1Sstates+ 7
#define BSIM3v1Sqd BSIM3v1Sstates+ 8
#define BSIM3v1Scqd BSIM3v1Sstates+ 9

#define BSIM3v1Sqbs  BSIM3v1Sstates+ 10
#define BSIM3v1Sqbd  BSIM3v1Sstates+ 11

#define BSIM3v1Sqcheq BSIM3v1Sstates+ 12
#define BSIM3v1Scqcheq BSIM3v1Sstates+ 13
#define BSIM3v1Sqcdump BSIM3v1Sstates+ 14
#define BSIM3v1Scqcdump BSIM3v1Sstates+ 15

#define BSIM3v1Stau BSIM3v1Sstates+ 16
#define BSIM3v1Sqdef BSIM3v1Sstates+ 17

#define BSIM3v1SnumStates 18


/* indices to the array of BSIM3v1S NOISE SOURCES */

#define BSIM3v1SRDNOIZ       0
#define BSIM3v1SRSNOIZ       1
#define BSIM3v1SIDNOIZ       2
#define BSIM3v1SFLNOIZ       3
#define BSIM3v1STOTNOIZ      4

#define BSIM3v1SNSRCS        5     /* the number of MOSFET(3) noise sources */

#ifndef NONOISE
    double BSIM3v1SnVar[NSTATVARS][BSIM3v1SNSRCS];
#else /* NONOISE */
        double **BSIM3v1SnVar;
#endif /* NONOISE */

} BSIM3v1Sinstance ;

struct bsim3v1sSizeDependParam
{
    double Width;
    double Length;

    double BSIM3v1Scdsc;           
    double BSIM3v1Scdscb;    
    double BSIM3v1Scdscd;       
    double BSIM3v1Scit;           
    double BSIM3v1Snfactor;      
    double BSIM3v1Sxj;
    double BSIM3v1Svsat;         
    double BSIM3v1Sat;         
    double BSIM3v1Sa0;   
    double BSIM3v1Sags;      
    double BSIM3v1Sa1;         
    double BSIM3v1Sa2;         
    double BSIM3v1Sketa;     
    double BSIM3v1Snsub;
    double BSIM3v1Snpeak;        
    double BSIM3v1Sngate;        
    double BSIM3v1Sgamma1;      
    double BSIM3v1Sgamma2;     
    double BSIM3v1Svbx;      
    double BSIM3v1Svbi;       
    double BSIM3v1Svbm;       
    double BSIM3v1Svbsc;       
    double BSIM3v1Sxt;       
    double BSIM3v1Sphi;
    double BSIM3v1Slitl;
    double BSIM3v1Sk1;
    double BSIM3v1Skt1;
    double BSIM3v1Skt1l;
    double BSIM3v1Skt2;
    double BSIM3v1Sk2;
    double BSIM3v1Sk3;
    double BSIM3v1Sk3b;
    double BSIM3v1Sw0;
    double BSIM3v1Snlx;
    double BSIM3v1Sdvt0;      
    double BSIM3v1Sdvt1;      
    double BSIM3v1Sdvt2;      
    double BSIM3v1Sdvt0w;      
    double BSIM3v1Sdvt1w;      
    double BSIM3v1Sdvt2w;      
    double BSIM3v1Sdrout;      
    double BSIM3v1Sdsub;      
    double BSIM3v1Svth0;
    double BSIM3v1Sua;
    double BSIM3v1Sua1;
    double BSIM3v1Sub;
    double BSIM3v1Sub1;
    double BSIM3v1Suc;
    double BSIM3v1Suc1;
    double BSIM3v1Su0;
    double BSIM3v1Sute;
    double BSIM3v1Svoff;
    double BSIM3v1Svfb;
    double BSIM3v1Sdelta;
    double BSIM3v1Srdsw;       
    double BSIM3v1Srds0;       
    double BSIM3v1Sprwg;       
    double BSIM3v1Sprwb;       
    double BSIM3v1Sprt;       
    double BSIM3v1Seta0;         
    double BSIM3v1Setab;         
    double BSIM3v1Spclm;      
    double BSIM3v1Spdibl1;      
    double BSIM3v1Spdibl2;      
    double BSIM3v1Spdiblb;      
    double BSIM3v1Spscbe1;       
    double BSIM3v1Spscbe2;       
    double BSIM3v1Spvag;       
    double BSIM3v1Swr;
    double BSIM3v1Sdwg;
    double BSIM3v1Sdwb;
    double BSIM3v1Sb0;
    double BSIM3v1Sb1;
    double BSIM3v1Salpha0;
    double BSIM3v1Sbeta0;


    /* CV model */
    double BSIM3v1Selm;
    double BSIM3v1Scgsl;
    double BSIM3v1Scgdl;
    double BSIM3v1Sckappa;
    double BSIM3v1Scf;
    double BSIM3v1Sclc;
    double BSIM3v1Scle;
    double BSIM3v1Svfbcv;


/* Pre-calculated constants */

    double BSIM3v1Sdw;
    double BSIM3v1Sdl;
    double BSIM3v1Sleff;
    double BSIM3v1Sweff;

    double BSIM3v1Sdwc;
    double BSIM3v1Sdlc;
    double BSIM3v1SleffCV;
    double BSIM3v1SweffCV;
    double BSIM3v1SabulkCVfactor;
    double BSIM3v1Scgso;
    double BSIM3v1Scgdo;
    double BSIM3v1Scgbo;

    double BSIM3v1Su0temp;       
    double BSIM3v1Svsattemp;   
    double BSIM3v1SsqrtPhi;   
    double BSIM3v1Sphis3;   
    double BSIM3v1SXdep0;          
    double BSIM3v1SsqrtXdep0;          
    double BSIM3v1Stheta0vb0;
    double BSIM3v1SthetaRout; 

    double BSIM3v1Scof1;
    double BSIM3v1Scof2;
    double BSIM3v1Scof3;
    double BSIM3v1Scof4;
    double BSIM3v1Scdep0;
    struct bsim3v1sSizeDependParam  *pNext;
};


typedef struct sBSIM3v1Smodel 
{
    int BSIM3v1SmodType;
    struct sBSIM3v1Smodel *BSIM3v1SnextModel;
    BSIM3v1Sinstance *BSIM3v1Sinstances;
    IFuid BSIM3v1SmodName; 
    int BSIM3v1Stype;

    int    BSIM3v1SmobMod;
    int    BSIM3v1ScapMod;
    int    BSIM3v1SnqsMod;
    int    BSIM3v1SnoiMod;
    int    BSIM3v1SbinUnit;
    int    BSIM3v1SparamChk;
    double BSIM3v1Sversion;             
    double BSIM3v1Stox;             
    double BSIM3v1Scdsc;           
    double BSIM3v1Scdscb; 
    double BSIM3v1Scdscd;          
    double BSIM3v1Scit;           
    double BSIM3v1Snfactor;      
    double BSIM3v1Sxj;
    double BSIM3v1Svsat;         
    double BSIM3v1Sat;         
    double BSIM3v1Sa0;   
    double BSIM3v1Sags;      
    double BSIM3v1Sa1;         
    double BSIM3v1Sa2;         
    double BSIM3v1Sketa;     
    double BSIM3v1Snsub;
    double BSIM3v1Snpeak;        
    double BSIM3v1Sngate;        
    double BSIM3v1Sgamma1;      
    double BSIM3v1Sgamma2;     
    double BSIM3v1Svbx;      
    double BSIM3v1Svbm;       
    double BSIM3v1Sxt;       
    double BSIM3v1Sk1;
    double BSIM3v1Skt1;
    double BSIM3v1Skt1l;
    double BSIM3v1Skt2;
    double BSIM3v1Sk2;
    double BSIM3v1Sk3;
    double BSIM3v1Sk3b;
    double BSIM3v1Sw0;
    double BSIM3v1Snlx;
    double BSIM3v1Sdvt0;      
    double BSIM3v1Sdvt1;      
    double BSIM3v1Sdvt2;      
    double BSIM3v1Sdvt0w;      
    double BSIM3v1Sdvt1w;      
    double BSIM3v1Sdvt2w;      
    double BSIM3v1Sdrout;      
    double BSIM3v1Sdsub;      
    double BSIM3v1Svth0;
    double BSIM3v1Sua;
    double BSIM3v1Sua1;
    double BSIM3v1Sub;
    double BSIM3v1Sub1;
    double BSIM3v1Suc;
    double BSIM3v1Suc1;
    double BSIM3v1Su0;
    double BSIM3v1Sute;
    double BSIM3v1Svoff;
    double BSIM3v1Sdelta;
    double BSIM3v1Srdsw;       
    double BSIM3v1Sprwg;
    double BSIM3v1Sprwb;
    double BSIM3v1Sprt;       
    double BSIM3v1Seta0;         
    double BSIM3v1Setab;         
    double BSIM3v1Spclm;      
    double BSIM3v1Spdibl1;      
    double BSIM3v1Spdibl2;      
    double BSIM3v1Spdiblb;
    double BSIM3v1Spscbe1;       
    double BSIM3v1Spscbe2;       
    double BSIM3v1Spvag;       
    double BSIM3v1Swr;
    double BSIM3v1Sdwg;
    double BSIM3v1Sdwb;
    double BSIM3v1Sb0;
    double BSIM3v1Sb1;
    double BSIM3v1Salpha0;
    double BSIM3v1Sbeta0;
/* serban */
    double BSIM3v1Shdif;

    /* CV model */
    double BSIM3v1Selm;
    double BSIM3v1Scgsl;
    double BSIM3v1Scgdl;
    double BSIM3v1Sckappa;
    double BSIM3v1Scf;
    double BSIM3v1Svfbcv;
    double BSIM3v1Sclc;
    double BSIM3v1Scle;
    double BSIM3v1Sdwc;
    double BSIM3v1Sdlc;

    /* Length Dependence */
    double BSIM3v1Slcdsc;           
    double BSIM3v1Slcdscb; 
    double BSIM3v1Slcdscd;          
    double BSIM3v1Slcit;           
    double BSIM3v1Slnfactor;      
    double BSIM3v1Slxj;
    double BSIM3v1Slvsat;         
    double BSIM3v1Slat;         
    double BSIM3v1Sla0;   
    double BSIM3v1Slags;      
    double BSIM3v1Sla1;         
    double BSIM3v1Sla2;         
    double BSIM3v1Slketa;     
    double BSIM3v1Slnsub;
    double BSIM3v1Slnpeak;        
    double BSIM3v1Slngate;        
    double BSIM3v1Slgamma1;      
    double BSIM3v1Slgamma2;     
    double BSIM3v1Slvbx;      
    double BSIM3v1Slvbm;       
    double BSIM3v1Slxt;       
    double BSIM3v1Slk1;
    double BSIM3v1Slkt1;
    double BSIM3v1Slkt1l;
    double BSIM3v1Slkt2;
    double BSIM3v1Slk2;
    double BSIM3v1Slk3;
    double BSIM3v1Slk3b;
    double BSIM3v1Slw0;
    double BSIM3v1Slnlx;
    double BSIM3v1Sldvt0;      
    double BSIM3v1Sldvt1;      
    double BSIM3v1Sldvt2;      
    double BSIM3v1Sldvt0w;      
    double BSIM3v1Sldvt1w;      
    double BSIM3v1Sldvt2w;      
    double BSIM3v1Sldrout;      
    double BSIM3v1Sldsub;      
    double BSIM3v1Slvth0;
    double BSIM3v1Slua;
    double BSIM3v1Slua1;
    double BSIM3v1Slub;
    double BSIM3v1Slub1;
    double BSIM3v1Sluc;
    double BSIM3v1Sluc1;
    double BSIM3v1Slu0;
    double BSIM3v1Slute;
    double BSIM3v1Slvoff;
    double BSIM3v1Sldelta;
    double BSIM3v1Slrdsw;       
    double BSIM3v1Slprwg;
    double BSIM3v1Slprwb;
    double BSIM3v1Slprt;       
    double BSIM3v1Sleta0;         
    double BSIM3v1Sletab;         
    double BSIM3v1Slpclm;      
    double BSIM3v1Slpdibl1;      
    double BSIM3v1Slpdibl2;      
    double BSIM3v1Slpdiblb;
    double BSIM3v1Slpscbe1;       
    double BSIM3v1Slpscbe2;       
    double BSIM3v1Slpvag;       
    double BSIM3v1Slwr;
    double BSIM3v1Sldwg;
    double BSIM3v1Sldwb;
    double BSIM3v1Slb0;
    double BSIM3v1Slb1;
    double BSIM3v1Slalpha0;
    double BSIM3v1Slbeta0;

    /* CV model */
    double BSIM3v1Slelm;
    double BSIM3v1Slcgsl;
    double BSIM3v1Slcgdl;
    double BSIM3v1Slckappa;
    double BSIM3v1Slcf;
    double BSIM3v1Slclc;
    double BSIM3v1Slcle;
    double BSIM3v1Slvfbcv;

    /* Width Dependence */
    double BSIM3v1Swcdsc;           
    double BSIM3v1Swcdscb; 
    double BSIM3v1Swcdscd;          
    double BSIM3v1Swcit;           
    double BSIM3v1Swnfactor;      
    double BSIM3v1Swxj;
    double BSIM3v1Swvsat;         
    double BSIM3v1Swat;         
    double BSIM3v1Swa0;   
    double BSIM3v1Swags;      
    double BSIM3v1Swa1;         
    double BSIM3v1Swa2;         
    double BSIM3v1Swketa;     
    double BSIM3v1Swnsub;
    double BSIM3v1Swnpeak;        
    double BSIM3v1Swngate;        
    double BSIM3v1Swgamma1;      
    double BSIM3v1Swgamma2;     
    double BSIM3v1Swvbx;      
    double BSIM3v1Swvbm;       
    double BSIM3v1Swxt;       
    double BSIM3v1Swk1;
    double BSIM3v1Swkt1;
    double BSIM3v1Swkt1l;
    double BSIM3v1Swkt2;
    double BSIM3v1Swk2;
    double BSIM3v1Swk3;
    double BSIM3v1Swk3b;
    double BSIM3v1Sww0;
    double BSIM3v1Swnlx;
    double BSIM3v1Swdvt0;      
    double BSIM3v1Swdvt1;      
    double BSIM3v1Swdvt2;      
    double BSIM3v1Swdvt0w;      
    double BSIM3v1Swdvt1w;      
    double BSIM3v1Swdvt2w;      
    double BSIM3v1Swdrout;      
    double BSIM3v1Swdsub;      
    double BSIM3v1Swvth0;
    double BSIM3v1Swua;
    double BSIM3v1Swua1;
    double BSIM3v1Swub;
    double BSIM3v1Swub1;
    double BSIM3v1Swuc;
    double BSIM3v1Swuc1;
    double BSIM3v1Swu0;
    double BSIM3v1Swute;
    double BSIM3v1Swvoff;
    double BSIM3v1Swdelta;
    double BSIM3v1Swrdsw;       
    double BSIM3v1Swprwg;
    double BSIM3v1Swprwb;
    double BSIM3v1Swprt;       
    double BSIM3v1Sweta0;         
    double BSIM3v1Swetab;         
    double BSIM3v1Swpclm;      
    double BSIM3v1Swpdibl1;      
    double BSIM3v1Swpdibl2;      
    double BSIM3v1Swpdiblb;
    double BSIM3v1Swpscbe1;       
    double BSIM3v1Swpscbe2;       
    double BSIM3v1Swpvag;       
    double BSIM3v1Swwr;
    double BSIM3v1Swdwg;
    double BSIM3v1Swdwb;
    double BSIM3v1Swb0;
    double BSIM3v1Swb1;
    double BSIM3v1Swalpha0;
    double BSIM3v1Swbeta0;

    /* CV model */
    double BSIM3v1Swelm;
    double BSIM3v1Swcgsl;
    double BSIM3v1Swcgdl;
    double BSIM3v1Swckappa;
    double BSIM3v1Swcf;
    double BSIM3v1Swclc;
    double BSIM3v1Swcle;
    double BSIM3v1Swvfbcv;

    /* Cross-term Dependence */
    double BSIM3v1Spcdsc;           
    double BSIM3v1Spcdscb; 
    double BSIM3v1Spcdscd;          
    double BSIM3v1Spcit;           
    double BSIM3v1Spnfactor;      
    double BSIM3v1Spxj;
    double BSIM3v1Spvsat;         
    double BSIM3v1Spat;         
    double BSIM3v1Spa0;   
    double BSIM3v1Spags;      
    double BSIM3v1Spa1;         
    double BSIM3v1Spa2;         
    double BSIM3v1Spketa;     
    double BSIM3v1Spnsub;
    double BSIM3v1Spnpeak;        
    double BSIM3v1Spngate;        
    double BSIM3v1Spgamma1;      
    double BSIM3v1Spgamma2;     
    double BSIM3v1Spvbx;      
    double BSIM3v1Spvbm;       
    double BSIM3v1Spxt;       
    double BSIM3v1Spk1;
    double BSIM3v1Spkt1;
    double BSIM3v1Spkt1l;
    double BSIM3v1Spkt2;
    double BSIM3v1Spk2;
    double BSIM3v1Spk3;
    double BSIM3v1Spk3b;
    double BSIM3v1Spw0;
    double BSIM3v1Spnlx;
    double BSIM3v1Spdvt0;      
    double BSIM3v1Spdvt1;      
    double BSIM3v1Spdvt2;      
    double BSIM3v1Spdvt0w;      
    double BSIM3v1Spdvt1w;      
    double BSIM3v1Spdvt2w;      
    double BSIM3v1Spdrout;      
    double BSIM3v1Spdsub;      
    double BSIM3v1Spvth0;
    double BSIM3v1Spua;
    double BSIM3v1Spua1;
    double BSIM3v1Spub;
    double BSIM3v1Spub1;
    double BSIM3v1Spuc;
    double BSIM3v1Spuc1;
    double BSIM3v1Spu0;
    double BSIM3v1Spute;
    double BSIM3v1Spvoff;
    double BSIM3v1Spdelta;
    double BSIM3v1Sprdsw;
    double BSIM3v1Spprwg;
    double BSIM3v1Spprwb;
    double BSIM3v1Spprt;       
    double BSIM3v1Speta0;         
    double BSIM3v1Spetab;         
    double BSIM3v1Sppclm;      
    double BSIM3v1Sppdibl1;      
    double BSIM3v1Sppdibl2;      
    double BSIM3v1Sppdiblb;
    double BSIM3v1Sppscbe1;       
    double BSIM3v1Sppscbe2;       
    double BSIM3v1Sppvag;       
    double BSIM3v1Spwr;
    double BSIM3v1Spdwg;
    double BSIM3v1Spdwb;
    double BSIM3v1Spb0;
    double BSIM3v1Spb1;
    double BSIM3v1Spalpha0;
    double BSIM3v1Spbeta0;

    /* CV model */
    double BSIM3v1Spelm;
    double BSIM3v1Spcgsl;
    double BSIM3v1Spcgdl;
    double BSIM3v1Spckappa;
    double BSIM3v1Spcf;
    double BSIM3v1Spclc;
    double BSIM3v1Spcle;
    double BSIM3v1Spvfbcv;

    double BSIM3v1Stnom;
    double BSIM3v1Scgso;
    double BSIM3v1Scgdo;
    double BSIM3v1Scgbo;
    double BSIM3v1Sxpart;
    double BSIM3v1ScFringOut;
    double BSIM3v1ScFringMax;

    double BSIM3v1SsheetResistance;
    double BSIM3v1SjctSatCurDensity;
    double BSIM3v1SjctSidewallSatCurDensity;
    double BSIM3v1SbulkJctPotential;
    double BSIM3v1SbulkJctBotGradingCoeff;
    double BSIM3v1SbulkJctSideGradingCoeff;
    double BSIM3v1SbulkJctGateSideGradingCoeff;
    double BSIM3v1SsidewallJctPotential;
    double BSIM3v1SGatesidewallJctPotential;
    double BSIM3v1SunitAreaJctCap;
    double BSIM3v1SunitLengthSidewallJctCap;
    double BSIM3v1SunitLengthGateSidewallJctCap;
    double BSIM3v1SjctEmissionCoeff;
    double BSIM3v1SjctTempExponent;

    double BSIM3v1SLint;
    double BSIM3v1SLl;
    double BSIM3v1SLln;
    double BSIM3v1SLw;
    double BSIM3v1SLwn;
    double BSIM3v1SLwl;
    double BSIM3v1SLmin;
    double BSIM3v1SLmax;

    double BSIM3v1SWint;
    double BSIM3v1SWl;
    double BSIM3v1SWln;
    double BSIM3v1SWw;
    double BSIM3v1SWwn;
    double BSIM3v1SWwl;
    double BSIM3v1SWmin;
    double BSIM3v1SWmax;


/* Pre-calculated constants */
    /* MCJ: move to size-dependent param. */
    double BSIM3v1Svtm;   
    double BSIM3v1Scox;
    double BSIM3v1Scof1;
    double BSIM3v1Scof2;
    double BSIM3v1Scof3;
    double BSIM3v1Scof4;
    double BSIM3v1Svcrit;
    double BSIM3v1Sfactor1;
    double BSIM3v1SjctTempSatCurDensity;
    double BSIM3v1SjctSidewallTempSatCurDensity;

    double BSIM3v1SoxideTrapDensityA;      
    double BSIM3v1SoxideTrapDensityB;     
    double BSIM3v1SoxideTrapDensityC;  
    double BSIM3v1Sem;  
    double BSIM3v1Sef;  
    double BSIM3v1Saf;  
    double BSIM3v1Skf;  

    struct bsim3v1sSizeDependParam *pSizeDependParamKnot;

    /* Flags */
    unsigned  BSIM3v1SmobModGiven :1;
    unsigned  BSIM3v1SbinUnitGiven :1;
    unsigned  BSIM3v1ScapModGiven :1;
    unsigned  BSIM3v1SparamChkGiven :1;
    unsigned  BSIM3v1SnqsModGiven :1;
    unsigned  BSIM3v1SnoiModGiven :1;
    unsigned  BSIM3v1StypeGiven   :1;
    unsigned  BSIM3v1StoxGiven   :1;
    unsigned  BSIM3v1SversionGiven   :1;

    unsigned  BSIM3v1ScdscGiven   :1;
    unsigned  BSIM3v1ScdscbGiven   :1;
    unsigned  BSIM3v1ScdscdGiven   :1;
    unsigned  BSIM3v1ScitGiven   :1;
    unsigned  BSIM3v1SnfactorGiven   :1;
    unsigned  BSIM3v1SxjGiven   :1;
    unsigned  BSIM3v1SvsatGiven   :1;
    unsigned  BSIM3v1SatGiven   :1;
    unsigned  BSIM3v1Sa0Given   :1;
    unsigned  BSIM3v1SagsGiven   :1;
    unsigned  BSIM3v1Sa1Given   :1;
    unsigned  BSIM3v1Sa2Given   :1;
    unsigned  BSIM3v1SketaGiven   :1;    
    unsigned  BSIM3v1SnsubGiven   :1;
    unsigned  BSIM3v1SnpeakGiven   :1;
    unsigned  BSIM3v1SngateGiven   :1;
    unsigned  BSIM3v1Sgamma1Given   :1;
    unsigned  BSIM3v1Sgamma2Given   :1;
    unsigned  BSIM3v1SvbxGiven   :1;
    unsigned  BSIM3v1SvbmGiven   :1;
    unsigned  BSIM3v1SxtGiven   :1;
    unsigned  BSIM3v1Sk1Given   :1;
    unsigned  BSIM3v1Skt1Given   :1;
    unsigned  BSIM3v1Skt1lGiven   :1;
    unsigned  BSIM3v1Skt2Given   :1;
    unsigned  BSIM3v1Sk2Given   :1;
    unsigned  BSIM3v1Sk3Given   :1;
    unsigned  BSIM3v1Sk3bGiven   :1;
    unsigned  BSIM3v1Sw0Given   :1;
    unsigned  BSIM3v1SnlxGiven   :1;
    unsigned  BSIM3v1Sdvt0Given   :1;   
    unsigned  BSIM3v1Sdvt1Given   :1;     
    unsigned  BSIM3v1Sdvt2Given   :1;     
    unsigned  BSIM3v1Sdvt0wGiven   :1;   
    unsigned  BSIM3v1Sdvt1wGiven   :1;     
    unsigned  BSIM3v1Sdvt2wGiven   :1;     
    unsigned  BSIM3v1SdroutGiven   :1;     
    unsigned  BSIM3v1SdsubGiven   :1;     
    unsigned  BSIM3v1Svth0Given   :1;
    unsigned  BSIM3v1SuaGiven   :1;
    unsigned  BSIM3v1Sua1Given   :1;
    unsigned  BSIM3v1SubGiven   :1;
    unsigned  BSIM3v1Sub1Given   :1;
    unsigned  BSIM3v1SucGiven   :1;
    unsigned  BSIM3v1Suc1Given   :1;
    unsigned  BSIM3v1Su0Given   :1;
    unsigned  BSIM3v1SuteGiven   :1;
    unsigned  BSIM3v1SvoffGiven   :1;
    unsigned  BSIM3v1SrdswGiven   :1;      
    unsigned  BSIM3v1SprwgGiven   :1;      
    unsigned  BSIM3v1SprwbGiven   :1;      
    unsigned  BSIM3v1SprtGiven   :1;      
    unsigned  BSIM3v1Seta0Given   :1;    
    unsigned  BSIM3v1SetabGiven   :1;    
    unsigned  BSIM3v1SpclmGiven   :1;   
    unsigned  BSIM3v1Spdibl1Given   :1;   
    unsigned  BSIM3v1Spdibl2Given   :1;  
    unsigned  BSIM3v1SpdiblbGiven   :1;  
    unsigned  BSIM3v1Spscbe1Given   :1;    
    unsigned  BSIM3v1Spscbe2Given   :1;    
    unsigned  BSIM3v1SpvagGiven   :1;    
    unsigned  BSIM3v1SdeltaGiven  :1;     
    unsigned  BSIM3v1SwrGiven   :1;
    unsigned  BSIM3v1SdwgGiven   :1;
    unsigned  BSIM3v1SdwbGiven   :1;
    unsigned  BSIM3v1Sb0Given   :1;
    unsigned  BSIM3v1Sb1Given   :1;
    unsigned  BSIM3v1Salpha0Given   :1;
    unsigned  BSIM3v1Sbeta0Given   :1;
    unsigned  BSIM3v1ShdifGiven   :1;

    /* CV model */
    unsigned  BSIM3v1SelmGiven  :1;     
    unsigned  BSIM3v1ScgslGiven   :1;
    unsigned  BSIM3v1ScgdlGiven   :1;
    unsigned  BSIM3v1SckappaGiven   :1;
    unsigned  BSIM3v1ScfGiven   :1;
    unsigned  BSIM3v1SvfbcvGiven   :1;
    unsigned  BSIM3v1SclcGiven   :1;
    unsigned  BSIM3v1ScleGiven   :1;
    unsigned  BSIM3v1SdwcGiven   :1;
    unsigned  BSIM3v1SdlcGiven   :1;


    /* Length dependence */
    unsigned  BSIM3v1SlcdscGiven   :1;
    unsigned  BSIM3v1SlcdscbGiven   :1;
    unsigned  BSIM3v1SlcdscdGiven   :1;
    unsigned  BSIM3v1SlcitGiven   :1;
    unsigned  BSIM3v1SlnfactorGiven   :1;
    unsigned  BSIM3v1SlxjGiven   :1;
    unsigned  BSIM3v1SlvsatGiven   :1;
    unsigned  BSIM3v1SlatGiven   :1;
    unsigned  BSIM3v1Sla0Given   :1;
    unsigned  BSIM3v1SlagsGiven   :1;
    unsigned  BSIM3v1Sla1Given   :1;
    unsigned  BSIM3v1Sla2Given   :1;
    unsigned  BSIM3v1SlketaGiven   :1;    
    unsigned  BSIM3v1SlnsubGiven   :1;
    unsigned  BSIM3v1SlnpeakGiven   :1;
    unsigned  BSIM3v1SlngateGiven   :1;
    unsigned  BSIM3v1Slgamma1Given   :1;
    unsigned  BSIM3v1Slgamma2Given   :1;
    unsigned  BSIM3v1SlvbxGiven   :1;
    unsigned  BSIM3v1SlvbmGiven   :1;
    unsigned  BSIM3v1SlxtGiven   :1;
    unsigned  BSIM3v1Slk1Given   :1;
    unsigned  BSIM3v1Slkt1Given   :1;
    unsigned  BSIM3v1Slkt1lGiven   :1;
    unsigned  BSIM3v1Slkt2Given   :1;
    unsigned  BSIM3v1Slk2Given   :1;
    unsigned  BSIM3v1Slk3Given   :1;
    unsigned  BSIM3v1Slk3bGiven   :1;
    unsigned  BSIM3v1Slw0Given   :1;
    unsigned  BSIM3v1SlnlxGiven   :1;
    unsigned  BSIM3v1Sldvt0Given   :1;   
    unsigned  BSIM3v1Sldvt1Given   :1;     
    unsigned  BSIM3v1Sldvt2Given   :1;     
    unsigned  BSIM3v1Sldvt0wGiven   :1;   
    unsigned  BSIM3v1Sldvt1wGiven   :1;     
    unsigned  BSIM3v1Sldvt2wGiven   :1;     
    unsigned  BSIM3v1SldroutGiven   :1;     
    unsigned  BSIM3v1SldsubGiven   :1;     
    unsigned  BSIM3v1Slvth0Given   :1;
    unsigned  BSIM3v1SluaGiven   :1;
    unsigned  BSIM3v1Slua1Given   :1;
    unsigned  BSIM3v1SlubGiven   :1;
    unsigned  BSIM3v1Slub1Given   :1;
    unsigned  BSIM3v1SlucGiven   :1;
    unsigned  BSIM3v1Sluc1Given   :1;
    unsigned  BSIM3v1Slu0Given   :1;
    unsigned  BSIM3v1SluteGiven   :1;
    unsigned  BSIM3v1SlvoffGiven   :1;
    unsigned  BSIM3v1SlrdswGiven   :1;      
    unsigned  BSIM3v1SlprwgGiven   :1;      
    unsigned  BSIM3v1SlprwbGiven   :1;      
    unsigned  BSIM3v1SlprtGiven   :1;      
    unsigned  BSIM3v1Sleta0Given   :1;    
    unsigned  BSIM3v1SletabGiven   :1;    
    unsigned  BSIM3v1SlpclmGiven   :1;   
    unsigned  BSIM3v1Slpdibl1Given   :1;   
    unsigned  BSIM3v1Slpdibl2Given   :1;  
    unsigned  BSIM3v1SlpdiblbGiven   :1;  
    unsigned  BSIM3v1Slpscbe1Given   :1;    
    unsigned  BSIM3v1Slpscbe2Given   :1;    
    unsigned  BSIM3v1SlpvagGiven   :1;    
    unsigned  BSIM3v1SldeltaGiven  :1;     
    unsigned  BSIM3v1SlwrGiven   :1;
    unsigned  BSIM3v1SldwgGiven   :1;
    unsigned  BSIM3v1SldwbGiven   :1;
    unsigned  BSIM3v1Slb0Given   :1;
    unsigned  BSIM3v1Slb1Given   :1;
    unsigned  BSIM3v1Slalpha0Given   :1;
    unsigned  BSIM3v1Slbeta0Given   :1;

    /* CV model */
    unsigned  BSIM3v1SlelmGiven  :1;     
    unsigned  BSIM3v1SlcgslGiven   :1;
    unsigned  BSIM3v1SlcgdlGiven   :1;
    unsigned  BSIM3v1SlckappaGiven   :1;
    unsigned  BSIM3v1SlcfGiven   :1;
    unsigned  BSIM3v1SlclcGiven   :1;
    unsigned  BSIM3v1SlcleGiven   :1;
    unsigned  BSIM3v1SlvfbcvGiven   :1;

    /* Width dependence */
    unsigned  BSIM3v1SwcdscGiven   :1;
    unsigned  BSIM3v1SwcdscbGiven   :1;
    unsigned  BSIM3v1SwcdscdGiven   :1;
    unsigned  BSIM3v1SwcitGiven   :1;
    unsigned  BSIM3v1SwnfactorGiven   :1;
    unsigned  BSIM3v1SwxjGiven   :1;
    unsigned  BSIM3v1SwvsatGiven   :1;
    unsigned  BSIM3v1SwatGiven   :1;
    unsigned  BSIM3v1Swa0Given   :1;
    unsigned  BSIM3v1SwagsGiven   :1;
    unsigned  BSIM3v1Swa1Given   :1;
    unsigned  BSIM3v1Swa2Given   :1;
    unsigned  BSIM3v1SwketaGiven   :1;    
    unsigned  BSIM3v1SwnsubGiven   :1;
    unsigned  BSIM3v1SwnpeakGiven   :1;
    unsigned  BSIM3v1SwngateGiven   :1;
    unsigned  BSIM3v1Swgamma1Given   :1;
    unsigned  BSIM3v1Swgamma2Given   :1;
    unsigned  BSIM3v1SwvbxGiven   :1;
    unsigned  BSIM3v1SwvbmGiven   :1;
    unsigned  BSIM3v1SwxtGiven   :1;
    unsigned  BSIM3v1Swk1Given   :1;
    unsigned  BSIM3v1Swkt1Given   :1;
    unsigned  BSIM3v1Swkt1lGiven   :1;
    unsigned  BSIM3v1Swkt2Given   :1;
    unsigned  BSIM3v1Swk2Given   :1;
    unsigned  BSIM3v1Swk3Given   :1;
    unsigned  BSIM3v1Swk3bGiven   :1;
    unsigned  BSIM3v1Sww0Given   :1;
    unsigned  BSIM3v1SwnlxGiven   :1;
    unsigned  BSIM3v1Swdvt0Given   :1;   
    unsigned  BSIM3v1Swdvt1Given   :1;     
    unsigned  BSIM3v1Swdvt2Given   :1;     
    unsigned  BSIM3v1Swdvt0wGiven   :1;   
    unsigned  BSIM3v1Swdvt1wGiven   :1;     
    unsigned  BSIM3v1Swdvt2wGiven   :1;     
    unsigned  BSIM3v1SwdroutGiven   :1;     
    unsigned  BSIM3v1SwdsubGiven   :1;     
    unsigned  BSIM3v1Swvth0Given   :1;
    unsigned  BSIM3v1SwuaGiven   :1;
    unsigned  BSIM3v1Swua1Given   :1;
    unsigned  BSIM3v1SwubGiven   :1;
    unsigned  BSIM3v1Swub1Given   :1;
    unsigned  BSIM3v1SwucGiven   :1;
    unsigned  BSIM3v1Swuc1Given   :1;
    unsigned  BSIM3v1Swu0Given   :1;
    unsigned  BSIM3v1SwuteGiven   :1;
    unsigned  BSIM3v1SwvoffGiven   :1;
    unsigned  BSIM3v1SwrdswGiven   :1;      
    unsigned  BSIM3v1SwprwgGiven   :1;      
    unsigned  BSIM3v1SwprwbGiven   :1;      
    unsigned  BSIM3v1SwprtGiven   :1;      
    unsigned  BSIM3v1Sweta0Given   :1;    
    unsigned  BSIM3v1SwetabGiven   :1;    
    unsigned  BSIM3v1SwpclmGiven   :1;   
    unsigned  BSIM3v1Swpdibl1Given   :1;   
    unsigned  BSIM3v1Swpdibl2Given   :1;  
    unsigned  BSIM3v1SwpdiblbGiven   :1;  
    unsigned  BSIM3v1Swpscbe1Given   :1;    
    unsigned  BSIM3v1Swpscbe2Given   :1;    
    unsigned  BSIM3v1SwpvagGiven   :1;    
    unsigned  BSIM3v1SwdeltaGiven  :1;     
    unsigned  BSIM3v1SwwrGiven   :1;
    unsigned  BSIM3v1SwdwgGiven   :1;
    unsigned  BSIM3v1SwdwbGiven   :1;
    unsigned  BSIM3v1Swb0Given   :1;
    unsigned  BSIM3v1Swb1Given   :1;
    unsigned  BSIM3v1Swalpha0Given   :1;
    unsigned  BSIM3v1Swbeta0Given   :1;

    /* CV model */
    unsigned  BSIM3v1SwelmGiven  :1;     
    unsigned  BSIM3v1SwcgslGiven   :1;
    unsigned  BSIM3v1SwcgdlGiven   :1;
    unsigned  BSIM3v1SwckappaGiven   :1;
    unsigned  BSIM3v1SwcfGiven   :1;
    unsigned  BSIM3v1SwclcGiven   :1;
    unsigned  BSIM3v1SwcleGiven   :1;
    unsigned  BSIM3v1SwvfbcvGiven   :1;

    /* Cross-term dependence */
    unsigned  BSIM3v1SpcdscGiven   :1;
    unsigned  BSIM3v1SpcdscbGiven   :1;
    unsigned  BSIM3v1SpcdscdGiven   :1;
    unsigned  BSIM3v1SpcitGiven   :1;
    unsigned  BSIM3v1SpnfactorGiven   :1;
    unsigned  BSIM3v1SpxjGiven   :1;
    unsigned  BSIM3v1SpvsatGiven   :1;
    unsigned  BSIM3v1SpatGiven   :1;
    unsigned  BSIM3v1Spa0Given   :1;
    unsigned  BSIM3v1SpagsGiven   :1;
    unsigned  BSIM3v1Spa1Given   :1;
    unsigned  BSIM3v1Spa2Given   :1;
    unsigned  BSIM3v1SpketaGiven   :1;    
    unsigned  BSIM3v1SpnsubGiven   :1;
    unsigned  BSIM3v1SpnpeakGiven   :1;
    unsigned  BSIM3v1SpngateGiven   :1;
    unsigned  BSIM3v1Spgamma1Given   :1;
    unsigned  BSIM3v1Spgamma2Given   :1;
    unsigned  BSIM3v1SpvbxGiven   :1;
    unsigned  BSIM3v1SpvbmGiven   :1;
    unsigned  BSIM3v1SpxtGiven   :1;
    unsigned  BSIM3v1Spk1Given   :1;
    unsigned  BSIM3v1Spkt1Given   :1;
    unsigned  BSIM3v1Spkt1lGiven   :1;
    unsigned  BSIM3v1Spkt2Given   :1;
    unsigned  BSIM3v1Spk2Given   :1;
    unsigned  BSIM3v1Spk3Given   :1;
    unsigned  BSIM3v1Spk3bGiven   :1;
    unsigned  BSIM3v1Spw0Given   :1;
    unsigned  BSIM3v1SpnlxGiven   :1;
    unsigned  BSIM3v1Spdvt0Given   :1;   
    unsigned  BSIM3v1Spdvt1Given   :1;     
    unsigned  BSIM3v1Spdvt2Given   :1;     
    unsigned  BSIM3v1Spdvt0wGiven   :1;   
    unsigned  BSIM3v1Spdvt1wGiven   :1;     
    unsigned  BSIM3v1Spdvt2wGiven   :1;     
    unsigned  BSIM3v1SpdroutGiven   :1;     
    unsigned  BSIM3v1SpdsubGiven   :1;     
    unsigned  BSIM3v1Spvth0Given   :1;
    unsigned  BSIM3v1SpuaGiven   :1;
    unsigned  BSIM3v1Spua1Given   :1;
    unsigned  BSIM3v1SpubGiven   :1;
    unsigned  BSIM3v1Spub1Given   :1;
    unsigned  BSIM3v1SpucGiven   :1;
    unsigned  BSIM3v1Spuc1Given   :1;
    unsigned  BSIM3v1Spu0Given   :1;
    unsigned  BSIM3v1SputeGiven   :1;
    unsigned  BSIM3v1SpvoffGiven   :1;
    unsigned  BSIM3v1SprdswGiven   :1;      
    unsigned  BSIM3v1SpprwgGiven   :1;      
    unsigned  BSIM3v1SpprwbGiven   :1;      
    unsigned  BSIM3v1SpprtGiven   :1;      
    unsigned  BSIM3v1Speta0Given   :1;    
    unsigned  BSIM3v1SpetabGiven   :1;    
    unsigned  BSIM3v1SppclmGiven   :1;   
    unsigned  BSIM3v1Sppdibl1Given   :1;   
    unsigned  BSIM3v1Sppdibl2Given   :1;  
    unsigned  BSIM3v1SppdiblbGiven   :1;  
    unsigned  BSIM3v1Sppscbe1Given   :1;    
    unsigned  BSIM3v1Sppscbe2Given   :1;    
    unsigned  BSIM3v1SppvagGiven   :1;    
    unsigned  BSIM3v1SpdeltaGiven  :1;     
    unsigned  BSIM3v1SpwrGiven   :1;
    unsigned  BSIM3v1SpdwgGiven   :1;
    unsigned  BSIM3v1SpdwbGiven   :1;
    unsigned  BSIM3v1Spb0Given   :1;
    unsigned  BSIM3v1Spb1Given   :1;
    unsigned  BSIM3v1Spalpha0Given   :1;
    unsigned  BSIM3v1Spbeta0Given   :1;

    /* CV model */
    unsigned  BSIM3v1SpelmGiven  :1;     
    unsigned  BSIM3v1SpcgslGiven   :1;
    unsigned  BSIM3v1SpcgdlGiven   :1;
    unsigned  BSIM3v1SpckappaGiven   :1;
    unsigned  BSIM3v1SpcfGiven   :1;
    unsigned  BSIM3v1SpclcGiven   :1;
    unsigned  BSIM3v1SpcleGiven   :1;
    unsigned  BSIM3v1SpvfbcvGiven   :1;

    unsigned  BSIM3v1SuseFringeGiven   :1;

    unsigned  BSIM3v1StnomGiven   :1;
    unsigned  BSIM3v1ScgsoGiven   :1;
    unsigned  BSIM3v1ScgdoGiven   :1;
    unsigned  BSIM3v1ScgboGiven   :1;
    unsigned  BSIM3v1SxpartGiven   :1;
    unsigned  BSIM3v1SsheetResistanceGiven   :1;
    unsigned  BSIM3v1SjctSatCurDensityGiven   :1;
    unsigned  BSIM3v1SjctSidewallSatCurDensityGiven   :1;
    unsigned  BSIM3v1SbulkJctPotentialGiven   :1;
    unsigned  BSIM3v1SbulkJctBotGradingCoeffGiven   :1;
    unsigned  BSIM3v1SsidewallJctPotentialGiven   :1;
    unsigned  BSIM3v1SGatesidewallJctPotentialGiven   :1;
    unsigned  BSIM3v1SbulkJctSideGradingCoeffGiven   :1;
    unsigned  BSIM3v1SunitAreaJctCapGiven   :1;
    unsigned  BSIM3v1SunitLengthSidewallJctCapGiven   :1;
    unsigned  BSIM3v1SbulkJctGateSideGradingCoeffGiven   :1;
    unsigned  BSIM3v1SunitLengthGateSidewallJctCapGiven   :1;
    unsigned  BSIM3v1SjctEmissionCoeffGiven :1;
    unsigned  BSIM3v1SjctTempExponentGiven	:1;

    unsigned  BSIM3v1SoxideTrapDensityAGiven  :1;         
    unsigned  BSIM3v1SoxideTrapDensityBGiven  :1;        
    unsigned  BSIM3v1SoxideTrapDensityCGiven  :1;     
    unsigned  BSIM3v1SemGiven  :1;     
    unsigned  BSIM3v1SefGiven  :1;     
    unsigned  BSIM3v1SafGiven  :1;     
    unsigned  BSIM3v1SkfGiven  :1;     

    unsigned  BSIM3v1SLintGiven   :1;
    unsigned  BSIM3v1SLlGiven   :1;
    unsigned  BSIM3v1SLlnGiven   :1;
    unsigned  BSIM3v1SLwGiven   :1;
    unsigned  BSIM3v1SLwnGiven   :1;
    unsigned  BSIM3v1SLwlGiven   :1;
    unsigned  BSIM3v1SLminGiven   :1;
    unsigned  BSIM3v1SLmaxGiven   :1;

    unsigned  BSIM3v1SWintGiven   :1;
    unsigned  BSIM3v1SWlGiven   :1;
    unsigned  BSIM3v1SWlnGiven   :1;
    unsigned  BSIM3v1SWwGiven   :1;
    unsigned  BSIM3v1SWwnGiven   :1;
    unsigned  BSIM3v1SWwlGiven   :1;
    unsigned  BSIM3v1SWminGiven   :1;
    unsigned  BSIM3v1SWmaxGiven   :1;

} BSIM3v1Smodel;


#ifndef NMOS
#define NMOS 1
#define PMOS -1
#endif /*NMOS*/


/* device parameters */
#define BSIM3v1S_W 1
#define BSIM3v1S_L 2
#define BSIM3v1S_AS 3
#define BSIM3v1S_AD 4
#define BSIM3v1S_PS 5
#define BSIM3v1S_PD 6
#define BSIM3v1S_NRS 7
#define BSIM3v1S_NRD 8
#define BSIM3v1S_OFF 9
#define BSIM3v1S_IC_VBS 10
#define BSIM3v1S_IC_VDS 11
#define BSIM3v1S_IC_VGS 12
#define BSIM3v1S_IC 13
#define BSIM3v1S_NQSMOD 14
#define BSIM3v1S_M 20

/* model parameters */
#define BSIM3v1S_MOD_CAPMOD          101
#define BSIM3v1S_MOD_NQSMOD          102
#define BSIM3v1S_MOD_MOBMOD          103    
#define BSIM3v1S_MOD_NOIMOD          104    

#define BSIM3v1S_MOD_TOX             105

#define BSIM3v1S_MOD_CDSC            106
#define BSIM3v1S_MOD_CDSCB           107
#define BSIM3v1S_MOD_CIT             108
#define BSIM3v1S_MOD_NFACTOR         109
#define BSIM3v1S_MOD_XJ              110
#define BSIM3v1S_MOD_VSAT            111
#define BSIM3v1S_MOD_AT              112
#define BSIM3v1S_MOD_A0              113
#define BSIM3v1S_MOD_A1              114
#define BSIM3v1S_MOD_A2              115
#define BSIM3v1S_MOD_KETA            116   
#define BSIM3v1S_MOD_NSUB            117
#define BSIM3v1S_MOD_NPEAK           118
#define BSIM3v1S_MOD_NGATE           120
#define BSIM3v1S_MOD_GAMMA1          121
#define BSIM3v1S_MOD_GAMMA2          122
#define BSIM3v1S_MOD_VBX             123
#define BSIM3v1S_MOD_BINUNIT         124    

#define BSIM3v1S_MOD_VBM             125

#define BSIM3v1S_MOD_XT              126
#define BSIM3v1S_MOD_K1              129
#define BSIM3v1S_MOD_KT1             130
#define BSIM3v1S_MOD_KT1L            131
#define BSIM3v1S_MOD_K2              132
#define BSIM3v1S_MOD_KT2             133
#define BSIM3v1S_MOD_K3              134
#define BSIM3v1S_MOD_K3B             135
#define BSIM3v1S_MOD_W0              136
#define BSIM3v1S_MOD_NLX             137

#define BSIM3v1S_MOD_DVT0            138
#define BSIM3v1S_MOD_DVT1            139
#define BSIM3v1S_MOD_DVT2            140

#define BSIM3v1S_MOD_DVT0W           141
#define BSIM3v1S_MOD_DVT1W           142
#define BSIM3v1S_MOD_DVT2W           143

#define BSIM3v1S_MOD_DROUT           144
#define BSIM3v1S_MOD_DSUB            145
#define BSIM3v1S_MOD_VTH0            146
#define BSIM3v1S_MOD_UA              147
#define BSIM3v1S_MOD_UA1             148
#define BSIM3v1S_MOD_UB              149
#define BSIM3v1S_MOD_UB1             150
#define BSIM3v1S_MOD_UC              151
#define BSIM3v1S_MOD_UC1             152
#define BSIM3v1S_MOD_U0              153
#define BSIM3v1S_MOD_UTE             154
#define BSIM3v1S_MOD_VOFF            155
#define BSIM3v1S_MOD_DELTA           156
#define BSIM3v1S_MOD_RDSW            157
#define BSIM3v1S_MOD_PRT             158
#define BSIM3v1S_MOD_LDD             159
#define BSIM3v1S_MOD_ETA             160
#define BSIM3v1S_MOD_ETA0            161
#define BSIM3v1S_MOD_ETAB            162
#define BSIM3v1S_MOD_PCLM            163
#define BSIM3v1S_MOD_PDIBL1          164
#define BSIM3v1S_MOD_PDIBL2          165
#define BSIM3v1S_MOD_PSCBE1          166
#define BSIM3v1S_MOD_PSCBE2          167
#define BSIM3v1S_MOD_PVAG            168
#define BSIM3v1S_MOD_WR              169
#define BSIM3v1S_MOD_DWG             170
#define BSIM3v1S_MOD_DWB             171
#define BSIM3v1S_MOD_B0              172
#define BSIM3v1S_MOD_B1              173
#define BSIM3v1S_MOD_ALPHA0          174
#define BSIM3v1S_MOD_BETA0           175
#define BSIM3v1S_MOD_PDIBLB          178

#define BSIM3v1S_MOD_PRWG            179
#define BSIM3v1S_MOD_PRWB            180

#define BSIM3v1S_MOD_CDSCD           181
#define BSIM3v1S_MOD_AGS             182

#define BSIM3v1S_MOD_FRINGE          184
#define BSIM3v1S_MOD_ELM             185
#define BSIM3v1S_MOD_CGSL            186
#define BSIM3v1S_MOD_CGDL            187
#define BSIM3v1S_MOD_CKAPPA          188
#define BSIM3v1S_MOD_CF              189
#define BSIM3v1S_MOD_CLC             190
#define BSIM3v1S_MOD_CLE             191
#define BSIM3v1S_MOD_PARAMCHK        192
#define BSIM3v1S_MOD_VERSION         193
#define BSIM3v1S_MOD_VFBCV           194

#define BSIM3v1S_MOD_HDIF            198

/* Length dependence */
#define BSIM3v1S_MOD_LCDSC            201
#define BSIM3v1S_MOD_LCDSCB           202
#define BSIM3v1S_MOD_LCIT             203
#define BSIM3v1S_MOD_LNFACTOR         204
#define BSIM3v1S_MOD_LXJ              205
#define BSIM3v1S_MOD_LVSAT            206
#define BSIM3v1S_MOD_LAT              207
#define BSIM3v1S_MOD_LA0              208
#define BSIM3v1S_MOD_LA1              209
#define BSIM3v1S_MOD_LA2              210
#define BSIM3v1S_MOD_LKETA            211   
#define BSIM3v1S_MOD_LNSUB            212
#define BSIM3v1S_MOD_LNPEAK           213
#define BSIM3v1S_MOD_LNGATE           215
#define BSIM3v1S_MOD_LGAMMA1          216
#define BSIM3v1S_MOD_LGAMMA2          217
#define BSIM3v1S_MOD_LVBX             218

#define BSIM3v1S_MOD_LVBM             220

#define BSIM3v1S_MOD_LXT              222
#define BSIM3v1S_MOD_LK1              225
#define BSIM3v1S_MOD_LKT1             226
#define BSIM3v1S_MOD_LKT1L            227
#define BSIM3v1S_MOD_LK2              228
#define BSIM3v1S_MOD_LKT2             229
#define BSIM3v1S_MOD_LK3              230
#define BSIM3v1S_MOD_LK3B             231
#define BSIM3v1S_MOD_LW0              232
#define BSIM3v1S_MOD_LNLX             233

#define BSIM3v1S_MOD_LDVT0            234
#define BSIM3v1S_MOD_LDVT1            235
#define BSIM3v1S_MOD_LDVT2            236

#define BSIM3v1S_MOD_LDVT0W           237
#define BSIM3v1S_MOD_LDVT1W           238
#define BSIM3v1S_MOD_LDVT2W           239

#define BSIM3v1S_MOD_LDROUT           240
#define BSIM3v1S_MOD_LDSUB            241
#define BSIM3v1S_MOD_LVTH0            242
#define BSIM3v1S_MOD_LUA              243
#define BSIM3v1S_MOD_LUA1             244
#define BSIM3v1S_MOD_LUB              245
#define BSIM3v1S_MOD_LUB1             246
#define BSIM3v1S_MOD_LUC              247
#define BSIM3v1S_MOD_LUC1             248
#define BSIM3v1S_MOD_LU0              249
#define BSIM3v1S_MOD_LUTE             250
#define BSIM3v1S_MOD_LVOFF            251
#define BSIM3v1S_MOD_LDELTA           252
#define BSIM3v1S_MOD_LRDSW            253
#define BSIM3v1S_MOD_LPRT             254
#define BSIM3v1S_MOD_LLDD             255
#define BSIM3v1S_MOD_LETA             256
#define BSIM3v1S_MOD_LETA0            257
#define BSIM3v1S_MOD_LETAB            258
#define BSIM3v1S_MOD_LPCLM            259
#define BSIM3v1S_MOD_LPDIBL1          260
#define BSIM3v1S_MOD_LPDIBL2          261
#define BSIM3v1S_MOD_LPSCBE1          262
#define BSIM3v1S_MOD_LPSCBE2          263
#define BSIM3v1S_MOD_LPVAG            264
#define BSIM3v1S_MOD_LWR              265
#define BSIM3v1S_MOD_LDWG             266
#define BSIM3v1S_MOD_LDWB             267
#define BSIM3v1S_MOD_LB0              268
#define BSIM3v1S_MOD_LB1              269
#define BSIM3v1S_MOD_LALPHA0          270
#define BSIM3v1S_MOD_LBETA0           271
#define BSIM3v1S_MOD_LPDIBLB          274

#define BSIM3v1S_MOD_LPRWG            275
#define BSIM3v1S_MOD_LPRWB            276

#define BSIM3v1S_MOD_LCDSCD           277
#define BSIM3v1S_MOD_LAGS            278
                                    

#define BSIM3v1S_MOD_LFRINGE          281
#define BSIM3v1S_MOD_LELM             282
#define BSIM3v1S_MOD_LCGSL            283
#define BSIM3v1S_MOD_LCGDL            284
#define BSIM3v1S_MOD_LCKAPPA          285
#define BSIM3v1S_MOD_LCF              286
#define BSIM3v1S_MOD_LCLC             287
#define BSIM3v1S_MOD_LCLE             288
#define BSIM3v1S_MOD_LVFBCV             289

/* Width dependence */
#define BSIM3v1S_MOD_WCDSC            301
#define BSIM3v1S_MOD_WCDSCB           302
#define BSIM3v1S_MOD_WCIT             303
#define BSIM3v1S_MOD_WNFACTOR         304
#define BSIM3v1S_MOD_WXJ              305
#define BSIM3v1S_MOD_WVSAT            306
#define BSIM3v1S_MOD_WAT              307
#define BSIM3v1S_MOD_WA0              308
#define BSIM3v1S_MOD_WA1              309
#define BSIM3v1S_MOD_WA2              310
#define BSIM3v1S_MOD_WKETA            311   
#define BSIM3v1S_MOD_WNSUB            312
#define BSIM3v1S_MOD_WNPEAK           313
#define BSIM3v1S_MOD_WNGATE           315
#define BSIM3v1S_MOD_WGAMMA1          316
#define BSIM3v1S_MOD_WGAMMA2          317
#define BSIM3v1S_MOD_WVBX             318

#define BSIM3v1S_MOD_WVBM             320

#define BSIM3v1S_MOD_WXT              322
#define BSIM3v1S_MOD_WK1              325
#define BSIM3v1S_MOD_WKT1             326
#define BSIM3v1S_MOD_WKT1L            327
#define BSIM3v1S_MOD_WK2              328
#define BSIM3v1S_MOD_WKT2             329
#define BSIM3v1S_MOD_WK3              330
#define BSIM3v1S_MOD_WK3B             331
#define BSIM3v1S_MOD_WW0              332
#define BSIM3v1S_MOD_WNLX             333

#define BSIM3v1S_MOD_WDVT0            334
#define BSIM3v1S_MOD_WDVT1            335
#define BSIM3v1S_MOD_WDVT2            336

#define BSIM3v1S_MOD_WDVT0W           337
#define BSIM3v1S_MOD_WDVT1W           338
#define BSIM3v1S_MOD_WDVT2W           339

#define BSIM3v1S_MOD_WDROUT           340
#define BSIM3v1S_MOD_WDSUB            341
#define BSIM3v1S_MOD_WVTH0            342
#define BSIM3v1S_MOD_WUA              343
#define BSIM3v1S_MOD_WUA1             344
#define BSIM3v1S_MOD_WUB              345
#define BSIM3v1S_MOD_WUB1             346
#define BSIM3v1S_MOD_WUC              347
#define BSIM3v1S_MOD_WUC1             348
#define BSIM3v1S_MOD_WU0              349
#define BSIM3v1S_MOD_WUTE             350
#define BSIM3v1S_MOD_WVOFF            351
#define BSIM3v1S_MOD_WDELTA           352
#define BSIM3v1S_MOD_WRDSW            353
#define BSIM3v1S_MOD_WPRT             354
#define BSIM3v1S_MOD_WLDD             355
#define BSIM3v1S_MOD_WETA             356
#define BSIM3v1S_MOD_WETA0            357
#define BSIM3v1S_MOD_WETAB            358
#define BSIM3v1S_MOD_WPCLM            359
#define BSIM3v1S_MOD_WPDIBL1          360
#define BSIM3v1S_MOD_WPDIBL2          361
#define BSIM3v1S_MOD_WPSCBE1          362
#define BSIM3v1S_MOD_WPSCBE2          363
#define BSIM3v1S_MOD_WPVAG            364
#define BSIM3v1S_MOD_WWR              365
#define BSIM3v1S_MOD_WDWG             366
#define BSIM3v1S_MOD_WDWB             367
#define BSIM3v1S_MOD_WB0              368
#define BSIM3v1S_MOD_WB1              369
#define BSIM3v1S_MOD_WALPHA0          370
#define BSIM3v1S_MOD_WBETA0           371
#define BSIM3v1S_MOD_WPDIBLB          374

#define BSIM3v1S_MOD_WPRWG            375
#define BSIM3v1S_MOD_WPRWB            376

#define BSIM3v1S_MOD_WCDSCD           377
#define BSIM3v1S_MOD_WAGS             378


#define BSIM3v1S_MOD_WFRINGE          381
#define BSIM3v1S_MOD_WELM             382
#define BSIM3v1S_MOD_WCGSL            383
#define BSIM3v1S_MOD_WCGDL            384
#define BSIM3v1S_MOD_WCKAPPA          385
#define BSIM3v1S_MOD_WCF              386
#define BSIM3v1S_MOD_WCLC             387
#define BSIM3v1S_MOD_WCLE             388
#define BSIM3v1S_MOD_WVFBCV             389

/* Cross-term dependence */
#define BSIM3v1S_MOD_PCDSC            401
#define BSIM3v1S_MOD_PCDSCB           402
#define BSIM3v1S_MOD_PCIT             403
#define BSIM3v1S_MOD_PNFACTOR         404
#define BSIM3v1S_MOD_PXJ              405
#define BSIM3v1S_MOD_PVSAT            406
#define BSIM3v1S_MOD_PAT              407
#define BSIM3v1S_MOD_PA0              408
#define BSIM3v1S_MOD_PA1              409
#define BSIM3v1S_MOD_PA2              410
#define BSIM3v1S_MOD_PKETA            411   
#define BSIM3v1S_MOD_PNSUB            412
#define BSIM3v1S_MOD_PNPEAK           413
#define BSIM3v1S_MOD_PNGATE           415
#define BSIM3v1S_MOD_PGAMMA1          416
#define BSIM3v1S_MOD_PGAMMA2          417
#define BSIM3v1S_MOD_PVBX             418

#define BSIM3v1S_MOD_PVBM             420

#define BSIM3v1S_MOD_PXT              422
#define BSIM3v1S_MOD_PK1              425
#define BSIM3v1S_MOD_PKT1             426
#define BSIM3v1S_MOD_PKT1L            427
#define BSIM3v1S_MOD_PK2              428
#define BSIM3v1S_MOD_PKT2             429
#define BSIM3v1S_MOD_PK3              430
#define BSIM3v1S_MOD_PK3B             431
#define BSIM3v1S_MOD_PW0              432
#define BSIM3v1S_MOD_PNLX             433

#define BSIM3v1S_MOD_PDVT0            434
#define BSIM3v1S_MOD_PDVT1            435
#define BSIM3v1S_MOD_PDVT2            436

#define BSIM3v1S_MOD_PDVT0W           437
#define BSIM3v1S_MOD_PDVT1W           438
#define BSIM3v1S_MOD_PDVT2W           439

#define BSIM3v1S_MOD_PDROUT           440
#define BSIM3v1S_MOD_PDSUB            441
#define BSIM3v1S_MOD_PVTH0            442
#define BSIM3v1S_MOD_PUA              443
#define BSIM3v1S_MOD_PUA1             444
#define BSIM3v1S_MOD_PUB              445
#define BSIM3v1S_MOD_PUB1             446
#define BSIM3v1S_MOD_PUC              447
#define BSIM3v1S_MOD_PUC1             448
#define BSIM3v1S_MOD_PU0              449
#define BSIM3v1S_MOD_PUTE             450
#define BSIM3v1S_MOD_PVOFF            451
#define BSIM3v1S_MOD_PDELTA           452
#define BSIM3v1S_MOD_PRDSW            453
#define BSIM3v1S_MOD_PPRT             454
#define BSIM3v1S_MOD_PLDD             455
#define BSIM3v1S_MOD_PETA             456
#define BSIM3v1S_MOD_PETA0            457
#define BSIM3v1S_MOD_PETAB            458
#define BSIM3v1S_MOD_PPCLM            459
#define BSIM3v1S_MOD_PPDIBL1          460
#define BSIM3v1S_MOD_PPDIBL2          461
#define BSIM3v1S_MOD_PPSCBE1          462
#define BSIM3v1S_MOD_PPSCBE2          463
#define BSIM3v1S_MOD_PPVAG            464
#define BSIM3v1S_MOD_PWR              465
#define BSIM3v1S_MOD_PDWG             466
#define BSIM3v1S_MOD_PDWB             467
#define BSIM3v1S_MOD_PB0              468
#define BSIM3v1S_MOD_PB1              469
#define BSIM3v1S_MOD_PALPHA0          470
#define BSIM3v1S_MOD_PBETA0           471
#define BSIM3v1S_MOD_PPDIBLB          474

#define BSIM3v1S_MOD_PPRWG            475
#define BSIM3v1S_MOD_PPRWB            476

#define BSIM3v1S_MOD_PCDSCD           477
#define BSIM3v1S_MOD_PAGS             478

#define BSIM3v1S_MOD_PFRINGE          481
#define BSIM3v1S_MOD_PELM             482
#define BSIM3v1S_MOD_PCGSL            483
#define BSIM3v1S_MOD_PCGDL            484
#define BSIM3v1S_MOD_PCKAPPA          485
#define BSIM3v1S_MOD_PCF              486
#define BSIM3v1S_MOD_PCLC             487
#define BSIM3v1S_MOD_PCLE             488
#define BSIM3v1S_MOD_PVFBCV           489

#define BSIM3v1S_MOD_TNOM             501
#define BSIM3v1S_MOD_CGSO             502
#define BSIM3v1S_MOD_CGDO             503
#define BSIM3v1S_MOD_CGBO             504
#define BSIM3v1S_MOD_XPART            505

#define BSIM3v1S_MOD_RSH              506
#define BSIM3v1S_MOD_JS               507
#define BSIM3v1S_MOD_PB               508
#define BSIM3v1S_MOD_MJ               509
#define BSIM3v1S_MOD_PBSW             510
#define BSIM3v1S_MOD_MJSW             511
#define BSIM3v1S_MOD_CJ               512
#define BSIM3v1S_MOD_CJSW             513
#define BSIM3v1S_MOD_NMOS             514
#define BSIM3v1S_MOD_PMOS             515

#define BSIM3v1S_MOD_NOIA             516
#define BSIM3v1S_MOD_NOIB             517
#define BSIM3v1S_MOD_NOIC             518

#define BSIM3v1S_MOD_LINT             519
#define BSIM3v1S_MOD_LL               520
#define BSIM3v1S_MOD_LLN              521
#define BSIM3v1S_MOD_LW               522
#define BSIM3v1S_MOD_LWN              523
#define BSIM3v1S_MOD_LWL              524
#define BSIM3v1S_MOD_LMIN             525
#define BSIM3v1S_MOD_LMAX             526

#define BSIM3v1S_MOD_WINT             527
#define BSIM3v1S_MOD_WL               528
#define BSIM3v1S_MOD_WLN              529
#define BSIM3v1S_MOD_WW               530
#define BSIM3v1S_MOD_WWN              531
#define BSIM3v1S_MOD_WWL              532
#define BSIM3v1S_MOD_WMIN             533
#define BSIM3v1S_MOD_WMAX             534

#define BSIM3v1S_MOD_DWC              535
#define BSIM3v1S_MOD_DLC              536

#define BSIM3v1S_MOD_EM               537
#define BSIM3v1S_MOD_EF               538
#define BSIM3v1S_MOD_AF               539
#define BSIM3v1S_MOD_KF               540

#define BSIM3v1S_MOD_NJ               541
#define BSIM3v1S_MOD_XTI              542

#define BSIM3v1S_MOD_PBSWG            543
#define BSIM3v1S_MOD_MJSWG            544
#define BSIM3v1S_MOD_CJSWG            545
#define BSIM3v1S_MOD_JSW              546

/* device questions */
#define BSIM3v1S_DNODE                601
#define BSIM3v1S_GNODE                602
#define BSIM3v1S_SNODE                603
#define BSIM3v1S_BNODE                604
#define BSIM3v1S_DNODEPRIME           605
#define BSIM3v1S_SNODEPRIME           606
#define BSIM3v1S_VBD                  607
#define BSIM3v1S_VBS                  608
#define BSIM3v1S_VGS                  609
#define BSIM3v1S_VDS                  610
#define BSIM3v1S_CD                   611
#define BSIM3v1S_CBS                  612
#define BSIM3v1S_CBD                  613
#define BSIM3v1S_GM                   614
#define BSIM3v1S_GDS                  615
#define BSIM3v1S_GMBS                 616
#define BSIM3v1S_GBD                  617
#define BSIM3v1S_GBS                  618
#define BSIM3v1S_QB                   619
#define BSIM3v1S_CQB                  620
#define BSIM3v1S_QG                   621
#define BSIM3v1S_CQG                  622
#define BSIM3v1S_QD                   623
#define BSIM3v1S_CQD                  624
#define BSIM3v1S_CGG                  625
#define BSIM3v1S_CGD                  626
#define BSIM3v1S_CGS                  627
#define BSIM3v1S_CBG                  628
#define BSIM3v1S_CAPBD                629
#define BSIM3v1S_CQBD                 630
#define BSIM3v1S_CAPBS                631
#define BSIM3v1S_CQBS                 632
#define BSIM3v1S_CDG                  633
#define BSIM3v1S_CDD                  634
#define BSIM3v1S_CDS                  635
#define BSIM3v1S_VON                  636
#define BSIM3v1S_VDSAT                637
#define BSIM3v1S_QBS                  638
#define BSIM3v1S_QBD                  639
#define BSIM3v1S_SOURCECONDUCT        640
#define BSIM3v1S_DRAINCONDUCT         641
#define BSIM3v1S_CBDB                 642
#define BSIM3v1S_CBSB                 643

#include "bsim3v1sext.h"

extern void BSIM3v1Sevaluate(double,double,double,BSIM3v1Sinstance*,BSIM3v1Smodel*,
        double*,double*,double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, CKTcircuit*);
extern int BSIM3v1Sdebug(BSIM3v1Smodel*, BSIM3v1Sinstance*, CKTcircuit*, int);
extern int BSIM3v1ScheckModel(BSIM3v1Smodel*, BSIM3v1Sinstance*, CKTcircuit*);


#endif /*BSIM3v1S*/




