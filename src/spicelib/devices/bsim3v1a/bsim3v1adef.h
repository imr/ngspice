/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan
File: bsim3v1adef.h
**********/

#ifndef BSIM3v1A
#define BSIM3v1A

#include "ifsim.h"
#include "gendefs.h"
#include "cktdefs.h"
#include "complex.h"
#include "noisedef.h"         

typedef struct sBSIM3v1Ainstance
{
    struct sBSIM3v1Amodel *BSIM3v1AmodPtr;
    struct sBSIM3v1Ainstance *BSIM3v1AnextInstance;
    IFuid BSIM3v1Aname;
    int BSIM3v1Aowner;      /* number of owner process */  
    int BSIM3v1Astates;     /* index into state table for this device */

    int BSIM3v1AdNode;
    int BSIM3v1AgNode;
    int BSIM3v1AsNode;
    int BSIM3v1AbNode;
    int BSIM3v1AdNodePrime;
    int BSIM3v1AsNodePrime;
    int BSIM3v1AqNode; /* MCJ */

    /* MCJ */
    double BSIM3v1Aueff;
    double BSIM3v1Athetavth; 
    double BSIM3v1Avon;
    double BSIM3v1Avdsat;
    double BSIM3v1Acgdo;
    double BSIM3v1Acgso;

    double BSIM3v1Al;
    double BSIM3v1Aw;
    double BSIM3v1Am;
    double BSIM3v1AdrainArea;
    double BSIM3v1AsourceArea;
    double BSIM3v1AdrainSquares;
    double BSIM3v1AsourceSquares;
    double BSIM3v1AdrainPerimeter;
    double BSIM3v1AsourcePerimeter;
    double BSIM3v1AsourceConductance;
    double BSIM3v1AdrainConductance;

    double BSIM3v1AicVBS;
    double BSIM3v1AicVDS;
    double BSIM3v1AicVGS;
    int BSIM3v1Aoff;
    int BSIM3v1Amode;
    int BSIM3v1AnqsMod;

    /* OP point */
    double BSIM3v1Aqinv;
    double BSIM3v1Acd;
    double BSIM3v1Acbs;
    double BSIM3v1Acbd;
    double BSIM3v1Acsub;
    double BSIM3v1Agm;
    double BSIM3v1Agds;
    double BSIM3v1Agmbs;
    double BSIM3v1Agbd;
    double BSIM3v1Agbs;

    double BSIM3v1Agbbs;
    double BSIM3v1Agbgs;
    double BSIM3v1Agbds;

    double BSIM3v1Acggb;
    double BSIM3v1Acgdb;
    double BSIM3v1Acgsb;
    double BSIM3v1Acbgb;
    double BSIM3v1Acbdb;
    double BSIM3v1Acbsb;
    double BSIM3v1Acdgb;
    double BSIM3v1Acddb;
    double BSIM3v1Acdsb;
    double BSIM3v1Acapbd;
    double BSIM3v1Acapbs;

    double BSIM3v1Acqgb;
    double BSIM3v1Acqdb;
    double BSIM3v1Acqsb;
    double BSIM3v1Acqbb;

    double BSIM3v1Agtau;
    double BSIM3v1Agtg;
    double BSIM3v1Agtd;
    double BSIM3v1Agts;
    double BSIM3v1Agtb;
    double BSIM3v1Atconst;

    struct bsim3v1aSizeDependParam  *pParam;

    unsigned BSIM3v1AlGiven :1;
    unsigned BSIM3v1AwGiven :1;
    unsigned BSIM3v1AmGiven :1;
    unsigned BSIM3v1AdrainAreaGiven :1;
    unsigned BSIM3v1AsourceAreaGiven    :1;
    unsigned BSIM3v1AdrainSquaresGiven  :1;
    unsigned BSIM3v1AsourceSquaresGiven :1;
    unsigned BSIM3v1AdrainPerimeterGiven    :1;
    unsigned BSIM3v1AsourcePerimeterGiven   :1;
    unsigned BSIM3v1AdNodePrimeSet  :1;
    unsigned BSIM3v1AsNodePrimeSet  :1;
    unsigned BSIM3v1AicVBSGiven :1;
    unsigned BSIM3v1AicVDSGiven :1;
    unsigned BSIM3v1AicVGSGiven :1;
    unsigned BSIM3v1AnqsModGiven :1;

    double *BSIM3v1ADdPtr;
    double *BSIM3v1AGgPtr;
    double *BSIM3v1ASsPtr;
    double *BSIM3v1ABbPtr;
    double *BSIM3v1ADPdpPtr;
    double *BSIM3v1ASPspPtr;
    double *BSIM3v1ADdpPtr;
    double *BSIM3v1AGbPtr;
    double *BSIM3v1AGdpPtr;
    double *BSIM3v1AGspPtr;
    double *BSIM3v1ASspPtr;
    double *BSIM3v1ABdpPtr;
    double *BSIM3v1ABspPtr;
    double *BSIM3v1ADPspPtr;
    double *BSIM3v1ADPdPtr;
    double *BSIM3v1ABgPtr;
    double *BSIM3v1ADPgPtr;
    double *BSIM3v1ASPgPtr;
    double *BSIM3v1ASPsPtr;
    double *BSIM3v1ADPbPtr;
    double *BSIM3v1ASPbPtr;
    double *BSIM3v1ASPdpPtr;

    double *BSIM3v1AQqPtr;
    double *BSIM3v1AQdpPtr;
    double *BSIM3v1AQgPtr;
    double *BSIM3v1AQspPtr;
    double *BSIM3v1AQbPtr;
    double *BSIM3v1ADPqPtr;
    double *BSIM3v1AGqPtr;
    double *BSIM3v1ASPqPtr;
    double *BSIM3v1ABqPtr;


#define BSIM3v1Avbd BSIM3v1Astates+ 0
#define BSIM3v1Avbs BSIM3v1Astates+ 1
#define BSIM3v1Avgs BSIM3v1Astates+ 2
#define BSIM3v1Avds BSIM3v1Astates+ 3

#define BSIM3v1Aqb BSIM3v1Astates+ 4
#define BSIM3v1Acqb BSIM3v1Astates+ 5
#define BSIM3v1Aqg BSIM3v1Astates+ 6
#define BSIM3v1Acqg BSIM3v1Astates+ 7
#define BSIM3v1Aqd BSIM3v1Astates+ 8
#define BSIM3v1Acqd BSIM3v1Astates+ 9

#define BSIM3v1Aqbs  BSIM3v1Astates+ 10
#define BSIM3v1Aqbd  BSIM3v1Astates+ 11

#define BSIM3v1Aqcheq BSIM3v1Astates+ 12
#define BSIM3v1Acqcheq BSIM3v1Astates+ 13
#define BSIM3v1Aqcdump BSIM3v1Astates+ 14
#define BSIM3v1Acqcdump BSIM3v1Astates+ 15

#define BSIM3v1Atau BSIM3v1Astates+ 16
#define BSIM3v1Aqdef BSIM3v1Astates+ 17

#define BSIM3v1AnumStates 18


/* indices to the array of BSIM3v1A NOISE SOURCES */

#define BSIM3v1ARDNOIZ       0
#define BSIM3v1ARSNOIZ       1
#define BSIM3v1AIDNOIZ       2
#define BSIM3v1AFLNOIZ       3
#define BSIM3v1ATOTNOIZ      4

#define BSIM3v1ANSRCS        5     /* the number of MOSFET(3) noise sources */

#ifndef NONOISE
    double BSIM3v1AnVar[NSTATVARS][BSIM3v1ANSRCS];
#else /* NONOISE */
        double **BSIM3v1AnVar;
#endif /* NONOISE */

} BSIM3v1Ainstance ;

struct bsim3v1aSizeDependParam
{
    double Width;
    double Length;

    double BSIM3v1Acdsc;           
    double BSIM3v1Acdscb;    
    double BSIM3v1Acdscd;       
    double BSIM3v1Acit;           
    double BSIM3v1Anfactor;      
    double BSIM3v1Axj;
    double BSIM3v1Avsat;         
    double BSIM3v1Aat;         
    double BSIM3v1Aa0;   
    double BSIM3v1Aags;      
    double BSIM3v1Aa1;         
    double BSIM3v1Aa2;         
    double BSIM3v1Aketa;     
    double BSIM3v1Ansub;
    double BSIM3v1Anpeak;        
    double BSIM3v1Angate;        
    double BSIM3v1Agamma1;      
    double BSIM3v1Agamma2;     
    double BSIM3v1Avbx;      
    double BSIM3v1Avbi;       
    double BSIM3v1Avbm;       
    double BSIM3v1Avbsc;       
    double BSIM3v1Axt;       
    double BSIM3v1Aphi;
    double BSIM3v1Alitl;
    double BSIM3v1Ak1;
    double BSIM3v1Akt1;
    double BSIM3v1Akt1l;
    double BSIM3v1Akt2;
    double BSIM3v1Ak2;
    double BSIM3v1Ak3;
    double BSIM3v1Ak3b;
    double BSIM3v1Aw0;
    double BSIM3v1Anlx;
    double BSIM3v1Advt0;      
    double BSIM3v1Advt1;      
    double BSIM3v1Advt2;      
    double BSIM3v1Advt0w;      
    double BSIM3v1Advt1w;      
    double BSIM3v1Advt2w;      
    double BSIM3v1Adrout;      
    double BSIM3v1Adsub;      
    double BSIM3v1Avth0;
    double BSIM3v1Aua;
    double BSIM3v1Aua1;
    double BSIM3v1Aub;
    double BSIM3v1Aub1;
    double BSIM3v1Auc;
    double BSIM3v1Auc1;
    double BSIM3v1Au0;
    double BSIM3v1Aute;
    double BSIM3v1Avoff;
    double BSIM3v1Avfb;
    double BSIM3v1Adelta;
    double BSIM3v1Ardsw;       
    double BSIM3v1Ards0;       
    double BSIM3v1Aprwg;       
    double BSIM3v1Aprwb;       
    double BSIM3v1Aprt;       
    double BSIM3v1Aeta0;         
    double BSIM3v1Aetab;         
    double BSIM3v1Apclm;      
    double BSIM3v1Apdibl1;      
    double BSIM3v1Apdibl2;      
    double BSIM3v1Apdiblb;      
    double BSIM3v1Apscbe1;       
    double BSIM3v1Apscbe2;       
    double BSIM3v1Apvag;       
    double BSIM3v1Awr;
    double BSIM3v1Adwg;
    double BSIM3v1Adwb;
    double BSIM3v1Ab0;
    double BSIM3v1Ab1;
    double BSIM3v1Aalpha0;
    double BSIM3v1Abeta0;


    /* CV model */
    double BSIM3v1Aelm;
    double BSIM3v1Acgsl;
    double BSIM3v1Acgdl;
    double BSIM3v1Ackappa;
    double BSIM3v1Acf;
    double BSIM3v1Aclc;
    double BSIM3v1Acle;


/* Pre-calculated constants */

    double BSIM3v1Adw;
    double BSIM3v1Adl;
    double BSIM3v1Aleff;
    double BSIM3v1Aweff;

    double BSIM3v1Adwc;
    double BSIM3v1Adlc;
    double BSIM3v1AleffCV;
    double BSIM3v1AweffCV;
    double BSIM3v1AabulkCVfactor;
    double BSIM3v1Acgso;
    double BSIM3v1Acgdo;
    double BSIM3v1Acgbo;

    double BSIM3v1Au0temp;       
    double BSIM3v1Avsattemp;   
    double BSIM3v1AsqrtPhi;   
    double BSIM3v1Aphis3;   
    double BSIM3v1AXdep0;          
    double BSIM3v1AsqrtXdep0;          
    double BSIM3v1Atheta0vb0;
    double BSIM3v1AthetaRout; 

    double BSIM3v1Acof1;
    double BSIM3v1Acof2;
    double BSIM3v1Acof3;
    double BSIM3v1Acof4;
    double BSIM3v1Acdep0;
    struct bsim3v1aSizeDependParam  *pNext;
};


typedef struct sBSIM3v1Amodel 
{
    int BSIM3v1AmodType;
    struct sBSIM3v1Amodel *BSIM3v1AnextModel;
    BSIM3v1Ainstance *BSIM3v1Ainstances;
    IFuid BSIM3v1AmodName; 
    int BSIM3v1Atype;

    int    BSIM3v1AmobMod;
    int    BSIM3v1AcapMod;
    int    BSIM3v1AnqsMod;
    int    BSIM3v1AnoiMod;
    int    BSIM3v1AbinUnit;
    double BSIM3v1Atox;             
    double BSIM3v1Acdsc;           
    double BSIM3v1Acdscb; 
    double BSIM3v1Acdscd;          
    double BSIM3v1Acit;           
    double BSIM3v1Anfactor;      
    double BSIM3v1Axj;
    double BSIM3v1Avsat;         
    double BSIM3v1Aat;         
    double BSIM3v1Aa0;   
    double BSIM3v1Aags;      
    double BSIM3v1Aa1;         
    double BSIM3v1Aa2;         
    double BSIM3v1Aketa;     
    double BSIM3v1Ansub;
    double BSIM3v1Anpeak;        
    double BSIM3v1Angate;        
    double BSIM3v1Agamma1;      
    double BSIM3v1Agamma2;     
    double BSIM3v1Avbx;      
    double BSIM3v1Avbm;       
    double BSIM3v1Axt;       
    double BSIM3v1Ak1;
    double BSIM3v1Akt1;
    double BSIM3v1Akt1l;
    double BSIM3v1Akt2;
    double BSIM3v1Ak2;
    double BSIM3v1Ak3;
    double BSIM3v1Ak3b;
    double BSIM3v1Aw0;
    double BSIM3v1Anlx;
    double BSIM3v1Advt0;      
    double BSIM3v1Advt1;      
    double BSIM3v1Advt2;      
    double BSIM3v1Advt0w;      
    double BSIM3v1Advt1w;      
    double BSIM3v1Advt2w;      
    double BSIM3v1Adrout;      
    double BSIM3v1Adsub;      
    double BSIM3v1Avth0;
    double BSIM3v1Aua;
    double BSIM3v1Aua1;
    double BSIM3v1Aub;
    double BSIM3v1Aub1;
    double BSIM3v1Auc;
    double BSIM3v1Auc1;
    double BSIM3v1Au0;
    double BSIM3v1Aute;
    double BSIM3v1Avoff;
    double BSIM3v1Adelta;
    double BSIM3v1Ardsw;       
    double BSIM3v1Aprwg;
    double BSIM3v1Aprwb;
    double BSIM3v1Aprt;       
    double BSIM3v1Aeta0;         
    double BSIM3v1Aetab;         
    double BSIM3v1Apclm;      
    double BSIM3v1Apdibl1;      
    double BSIM3v1Apdibl2;      
    double BSIM3v1Apdiblb;
    double BSIM3v1Apscbe1;       
    double BSIM3v1Apscbe2;       
    double BSIM3v1Apvag;       
    double BSIM3v1Awr;
    double BSIM3v1Adwg;
    double BSIM3v1Adwb;
    double BSIM3v1Ab0;
    double BSIM3v1Ab1;
    double BSIM3v1Aalpha0;
    double BSIM3v1Abeta0;

    /* CV model */
    double BSIM3v1Aelm;
    double BSIM3v1Acgsl;
    double BSIM3v1Acgdl;
    double BSIM3v1Ackappa;
    double BSIM3v1Acf;
    double BSIM3v1Aclc;
    double BSIM3v1Acle;
    double BSIM3v1Adwc;
    double BSIM3v1Adlc;

    /* Length Dependence */
    double BSIM3v1Alcdsc;           
    double BSIM3v1Alcdscb; 
    double BSIM3v1Alcdscd;          
    double BSIM3v1Alcit;           
    double BSIM3v1Alnfactor;      
    double BSIM3v1Alxj;
    double BSIM3v1Alvsat;         
    double BSIM3v1Alat;         
    double BSIM3v1Ala0;   
    double BSIM3v1Alags;      
    double BSIM3v1Ala1;         
    double BSIM3v1Ala2;         
    double BSIM3v1Alketa;     
    double BSIM3v1Alnsub;
    double BSIM3v1Alnpeak;        
    double BSIM3v1Alngate;        
    double BSIM3v1Algamma1;      
    double BSIM3v1Algamma2;     
    double BSIM3v1Alvbx;      
    double BSIM3v1Alvbm;       
    double BSIM3v1Alxt;       
    double BSIM3v1Alk1;
    double BSIM3v1Alkt1;
    double BSIM3v1Alkt1l;
    double BSIM3v1Alkt2;
    double BSIM3v1Alk2;
    double BSIM3v1Alk3;
    double BSIM3v1Alk3b;
    double BSIM3v1Alw0;
    double BSIM3v1Alnlx;
    double BSIM3v1Aldvt0;      
    double BSIM3v1Aldvt1;      
    double BSIM3v1Aldvt2;      
    double BSIM3v1Aldvt0w;      
    double BSIM3v1Aldvt1w;      
    double BSIM3v1Aldvt2w;      
    double BSIM3v1Aldrout;      
    double BSIM3v1Aldsub;      
    double BSIM3v1Alvth0;
    double BSIM3v1Alua;
    double BSIM3v1Alua1;
    double BSIM3v1Alub;
    double BSIM3v1Alub1;
    double BSIM3v1Aluc;
    double BSIM3v1Aluc1;
    double BSIM3v1Alu0;
    double BSIM3v1Alute;
    double BSIM3v1Alvoff;
    double BSIM3v1Aldelta;
    double BSIM3v1Alrdsw;       
    double BSIM3v1Alprwg;
    double BSIM3v1Alprwb;
    double BSIM3v1Alprt;       
    double BSIM3v1Aleta0;         
    double BSIM3v1Aletab;         
    double BSIM3v1Alpclm;      
    double BSIM3v1Alpdibl1;      
    double BSIM3v1Alpdibl2;      
    double BSIM3v1Alpdiblb;
    double BSIM3v1Alpscbe1;       
    double BSIM3v1Alpscbe2;       
    double BSIM3v1Alpvag;       
    double BSIM3v1Alwr;
    double BSIM3v1Aldwg;
    double BSIM3v1Aldwb;
    double BSIM3v1Alb0;
    double BSIM3v1Alb1;
    double BSIM3v1Alalpha0;
    double BSIM3v1Albeta0;

    /* CV model */
    double BSIM3v1Alelm;
    double BSIM3v1Alcgsl;
    double BSIM3v1Alcgdl;
    double BSIM3v1Alckappa;
    double BSIM3v1Alcf;
    double BSIM3v1Alclc;
    double BSIM3v1Alcle;

    /* Width Dependence */
    double BSIM3v1Awcdsc;           
    double BSIM3v1Awcdscb; 
    double BSIM3v1Awcdscd;          
    double BSIM3v1Awcit;           
    double BSIM3v1Awnfactor;      
    double BSIM3v1Awxj;
    double BSIM3v1Awvsat;         
    double BSIM3v1Awat;         
    double BSIM3v1Awa0;   
    double BSIM3v1Awags;      
    double BSIM3v1Awa1;         
    double BSIM3v1Awa2;         
    double BSIM3v1Awketa;     
    double BSIM3v1Awnsub;
    double BSIM3v1Awnpeak;        
    double BSIM3v1Awngate;        
    double BSIM3v1Awgamma1;      
    double BSIM3v1Awgamma2;     
    double BSIM3v1Awvbx;      
    double BSIM3v1Awvbm;       
    double BSIM3v1Awxt;       
    double BSIM3v1Awk1;
    double BSIM3v1Awkt1;
    double BSIM3v1Awkt1l;
    double BSIM3v1Awkt2;
    double BSIM3v1Awk2;
    double BSIM3v1Awk3;
    double BSIM3v1Awk3b;
    double BSIM3v1Aww0;
    double BSIM3v1Awnlx;
    double BSIM3v1Awdvt0;      
    double BSIM3v1Awdvt1;      
    double BSIM3v1Awdvt2;      
    double BSIM3v1Awdvt0w;      
    double BSIM3v1Awdvt1w;      
    double BSIM3v1Awdvt2w;      
    double BSIM3v1Awdrout;      
    double BSIM3v1Awdsub;      
    double BSIM3v1Awvth0;
    double BSIM3v1Awua;
    double BSIM3v1Awua1;
    double BSIM3v1Awub;
    double BSIM3v1Awub1;
    double BSIM3v1Awuc;
    double BSIM3v1Awuc1;
    double BSIM3v1Awu0;
    double BSIM3v1Awute;
    double BSIM3v1Awvoff;
    double BSIM3v1Awdelta;
    double BSIM3v1Awrdsw;       
    double BSIM3v1Awprwg;
    double BSIM3v1Awprwb;
    double BSIM3v1Awprt;       
    double BSIM3v1Aweta0;         
    double BSIM3v1Awetab;         
    double BSIM3v1Awpclm;      
    double BSIM3v1Awpdibl1;      
    double BSIM3v1Awpdibl2;      
    double BSIM3v1Awpdiblb;
    double BSIM3v1Awpscbe1;       
    double BSIM3v1Awpscbe2;       
    double BSIM3v1Awpvag;       
    double BSIM3v1Awwr;
    double BSIM3v1Awdwg;
    double BSIM3v1Awdwb;
    double BSIM3v1Awb0;
    double BSIM3v1Awb1;
    double BSIM3v1Awalpha0;
    double BSIM3v1Awbeta0;

    /* CV model */
    double BSIM3v1Awelm;
    double BSIM3v1Awcgsl;
    double BSIM3v1Awcgdl;
    double BSIM3v1Awckappa;
    double BSIM3v1Awcf;
    double BSIM3v1Awclc;
    double BSIM3v1Awcle;

    /* Cross-term Dependence */
    double BSIM3v1Apcdsc;           
    double BSIM3v1Apcdscb; 
    double BSIM3v1Apcdscd;          
    double BSIM3v1Apcit;           
    double BSIM3v1Apnfactor;      
    double BSIM3v1Apxj;
    double BSIM3v1Apvsat;         
    double BSIM3v1Apat;         
    double BSIM3v1Apa0;   
    double BSIM3v1Apags;      
    double BSIM3v1Apa1;         
    double BSIM3v1Apa2;         
    double BSIM3v1Apketa;     
    double BSIM3v1Apnsub;
    double BSIM3v1Apnpeak;        
    double BSIM3v1Apngate;        
    double BSIM3v1Apgamma1;      
    double BSIM3v1Apgamma2;     
    double BSIM3v1Apvbx;      
    double BSIM3v1Apvbm;       
    double BSIM3v1Apxt;       
    double BSIM3v1Apk1;
    double BSIM3v1Apkt1;
    double BSIM3v1Apkt1l;
    double BSIM3v1Apkt2;
    double BSIM3v1Apk2;
    double BSIM3v1Apk3;
    double BSIM3v1Apk3b;
    double BSIM3v1Apw0;
    double BSIM3v1Apnlx;
    double BSIM3v1Apdvt0;      
    double BSIM3v1Apdvt1;      
    double BSIM3v1Apdvt2;      
    double BSIM3v1Apdvt0w;      
    double BSIM3v1Apdvt1w;      
    double BSIM3v1Apdvt2w;      
    double BSIM3v1Apdrout;      
    double BSIM3v1Apdsub;      
    double BSIM3v1Apvth0;
    double BSIM3v1Apua;
    double BSIM3v1Apua1;
    double BSIM3v1Apub;
    double BSIM3v1Apub1;
    double BSIM3v1Apuc;
    double BSIM3v1Apuc1;
    double BSIM3v1Apu0;
    double BSIM3v1Apute;
    double BSIM3v1Apvoff;
    double BSIM3v1Apdelta;
    double BSIM3v1Aprdsw;
    double BSIM3v1Apprwg;
    double BSIM3v1Apprwb;
    double BSIM3v1Apprt;       
    double BSIM3v1Apeta0;         
    double BSIM3v1Apetab;         
    double BSIM3v1Appclm;      
    double BSIM3v1Appdibl1;      
    double BSIM3v1Appdibl2;      
    double BSIM3v1Appdiblb;
    double BSIM3v1Appscbe1;       
    double BSIM3v1Appscbe2;       
    double BSIM3v1Appvag;       
    double BSIM3v1Apwr;
    double BSIM3v1Apdwg;
    double BSIM3v1Apdwb;
    double BSIM3v1Apb0;
    double BSIM3v1Apb1;
    double BSIM3v1Apalpha0;
    double BSIM3v1Apbeta0;

    /* CV model */
    double BSIM3v1Apelm;
    double BSIM3v1Apcgsl;
    double BSIM3v1Apcgdl;
    double BSIM3v1Apckappa;
    double BSIM3v1Apcf;
    double BSIM3v1Apclc;
    double BSIM3v1Apcle;

    double BSIM3v1Atnom;
    double BSIM3v1Acgso;
    double BSIM3v1Acgdo;
    double BSIM3v1Acgbo;
    double BSIM3v1Axpart;
    double BSIM3v1AcFringOut;
    double BSIM3v1AcFringMax;

    double BSIM3v1AsheetResistance;
    double BSIM3v1AjctSatCurDensity;
    double BSIM3v1AbulkJctPotential;
    double BSIM3v1AbulkJctBotGradingCoeff;
    double BSIM3v1AbulkJctSideGradingCoeff;
    double BSIM3v1AsidewallJctPotential;
    double BSIM3v1AunitAreaJctCap;
    double BSIM3v1AunitLengthSidewallJctCap;

    double BSIM3v1ALint;
    double BSIM3v1ALl;
    double BSIM3v1ALln;
    double BSIM3v1ALw;
    double BSIM3v1ALwn;
    double BSIM3v1ALwl;
    double BSIM3v1ALmin;
    double BSIM3v1ALmax;

    double BSIM3v1AWint;
    double BSIM3v1AWl;
    double BSIM3v1AWln;
    double BSIM3v1AWw;
    double BSIM3v1AWwn;
    double BSIM3v1AWwl;
    double BSIM3v1AWmin;
    double BSIM3v1AWmax;


/* Pre-calculated constants */
    /* MCJ: move to size-dependent param. */
    double BSIM3v1Avtm;   
    double BSIM3v1Acox;
    double BSIM3v1Acof1;
    double BSIM3v1Acof2;
    double BSIM3v1Acof3;
    double BSIM3v1Acof4;
    double BSIM3v1Avcrit;
    double BSIM3v1Afactor1;

    double BSIM3v1AoxideTrapDensityA;      
    double BSIM3v1AoxideTrapDensityB;     
    double BSIM3v1AoxideTrapDensityC;  
    double BSIM3v1Aem;  
    double BSIM3v1Aef;  
    double BSIM3v1Aaf;  
    double BSIM3v1Akf;  

    struct bsim3v1aSizeDependParam *pSizeDependParamKnot;

    /* Flags */
    unsigned  BSIM3v1AmobModGiven :1;
    unsigned  BSIM3v1AbinUnitGiven :1;
    unsigned  BSIM3v1AcapModGiven :1;
    unsigned  BSIM3v1AnqsModGiven :1;
    unsigned  BSIM3v1AnoiModGiven :1;
    unsigned  BSIM3v1AtypeGiven   :1;
    unsigned  BSIM3v1AtoxGiven   :1;

    unsigned  BSIM3v1AcdscGiven   :1;
    unsigned  BSIM3v1AcdscbGiven   :1;
    unsigned  BSIM3v1AcdscdGiven   :1;
    unsigned  BSIM3v1AcitGiven   :1;
    unsigned  BSIM3v1AnfactorGiven   :1;
    unsigned  BSIM3v1AxjGiven   :1;
    unsigned  BSIM3v1AvsatGiven   :1;
    unsigned  BSIM3v1AatGiven   :1;
    unsigned  BSIM3v1Aa0Given   :1;
    unsigned  BSIM3v1AagsGiven   :1;
    unsigned  BSIM3v1Aa1Given   :1;
    unsigned  BSIM3v1Aa2Given   :1;
    unsigned  BSIM3v1AketaGiven   :1;    
    unsigned  BSIM3v1AnsubGiven   :1;
    unsigned  BSIM3v1AnpeakGiven   :1;
    unsigned  BSIM3v1AngateGiven   :1;
    unsigned  BSIM3v1Agamma1Given   :1;
    unsigned  BSIM3v1Agamma2Given   :1;
    unsigned  BSIM3v1AvbxGiven   :1;
    unsigned  BSIM3v1AvbmGiven   :1;
    unsigned  BSIM3v1AxtGiven   :1;
    unsigned  BSIM3v1Ak1Given   :1;
    unsigned  BSIM3v1Akt1Given   :1;
    unsigned  BSIM3v1Akt1lGiven   :1;
    unsigned  BSIM3v1Akt2Given   :1;
    unsigned  BSIM3v1Ak2Given   :1;
    unsigned  BSIM3v1Ak3Given   :1;
    unsigned  BSIM3v1Ak3bGiven   :1;
    unsigned  BSIM3v1Aw0Given   :1;
    unsigned  BSIM3v1AnlxGiven   :1;
    unsigned  BSIM3v1Advt0Given   :1;   
    unsigned  BSIM3v1Advt1Given   :1;     
    unsigned  BSIM3v1Advt2Given   :1;     
    unsigned  BSIM3v1Advt0wGiven   :1;   
    unsigned  BSIM3v1Advt1wGiven   :1;     
    unsigned  BSIM3v1Advt2wGiven   :1;     
    unsigned  BSIM3v1AdroutGiven   :1;     
    unsigned  BSIM3v1AdsubGiven   :1;     
    unsigned  BSIM3v1Avth0Given   :1;
    unsigned  BSIM3v1AuaGiven   :1;
    unsigned  BSIM3v1Aua1Given   :1;
    unsigned  BSIM3v1AubGiven   :1;
    unsigned  BSIM3v1Aub1Given   :1;
    unsigned  BSIM3v1AucGiven   :1;
    unsigned  BSIM3v1Auc1Given   :1;
    unsigned  BSIM3v1Au0Given   :1;
    unsigned  BSIM3v1AuteGiven   :1;
    unsigned  BSIM3v1AvoffGiven   :1;
    unsigned  BSIM3v1ArdswGiven   :1;      
    unsigned  BSIM3v1AprwgGiven   :1;      
    unsigned  BSIM3v1AprwbGiven   :1;      
    unsigned  BSIM3v1AprtGiven   :1;      
    unsigned  BSIM3v1Aeta0Given   :1;    
    unsigned  BSIM3v1AetabGiven   :1;    
    unsigned  BSIM3v1ApclmGiven   :1;   
    unsigned  BSIM3v1Apdibl1Given   :1;   
    unsigned  BSIM3v1Apdibl2Given   :1;  
    unsigned  BSIM3v1ApdiblbGiven   :1;  
    unsigned  BSIM3v1Apscbe1Given   :1;    
    unsigned  BSIM3v1Apscbe2Given   :1;    
    unsigned  BSIM3v1ApvagGiven   :1;    
    unsigned  BSIM3v1AdeltaGiven  :1;     
    unsigned  BSIM3v1AwrGiven   :1;
    unsigned  BSIM3v1AdwgGiven   :1;
    unsigned  BSIM3v1AdwbGiven   :1;
    unsigned  BSIM3v1Ab0Given   :1;
    unsigned  BSIM3v1Ab1Given   :1;
    unsigned  BSIM3v1Aalpha0Given   :1;
    unsigned  BSIM3v1Abeta0Given   :1;

    /* CV model */
    unsigned  BSIM3v1AelmGiven  :1;     
    unsigned  BSIM3v1AcgslGiven   :1;
    unsigned  BSIM3v1AcgdlGiven   :1;
    unsigned  BSIM3v1AckappaGiven   :1;
    unsigned  BSIM3v1AcfGiven   :1;
    unsigned  BSIM3v1AclcGiven   :1;
    unsigned  BSIM3v1AcleGiven   :1;
    unsigned  BSIM3v1AdwcGiven   :1;
    unsigned  BSIM3v1AdlcGiven   :1;


    /* Length dependence */
    unsigned  BSIM3v1AlcdscGiven   :1;
    unsigned  BSIM3v1AlcdscbGiven   :1;
    unsigned  BSIM3v1AlcdscdGiven   :1;
    unsigned  BSIM3v1AlcitGiven   :1;
    unsigned  BSIM3v1AlnfactorGiven   :1;
    unsigned  BSIM3v1AlxjGiven   :1;
    unsigned  BSIM3v1AlvsatGiven   :1;
    unsigned  BSIM3v1AlatGiven   :1;
    unsigned  BSIM3v1Ala0Given   :1;
    unsigned  BSIM3v1AlagsGiven   :1;
    unsigned  BSIM3v1Ala1Given   :1;
    unsigned  BSIM3v1Ala2Given   :1;
    unsigned  BSIM3v1AlketaGiven   :1;    
    unsigned  BSIM3v1AlnsubGiven   :1;
    unsigned  BSIM3v1AlnpeakGiven   :1;
    unsigned  BSIM3v1AlngateGiven   :1;
    unsigned  BSIM3v1Algamma1Given   :1;
    unsigned  BSIM3v1Algamma2Given   :1;
    unsigned  BSIM3v1AlvbxGiven   :1;
    unsigned  BSIM3v1AlvbmGiven   :1;
    unsigned  BSIM3v1AlxtGiven   :1;
    unsigned  BSIM3v1Alk1Given   :1;
    unsigned  BSIM3v1Alkt1Given   :1;
    unsigned  BSIM3v1Alkt1lGiven   :1;
    unsigned  BSIM3v1Alkt2Given   :1;
    unsigned  BSIM3v1Alk2Given   :1;
    unsigned  BSIM3v1Alk3Given   :1;
    unsigned  BSIM3v1Alk3bGiven   :1;
    unsigned  BSIM3v1Alw0Given   :1;
    unsigned  BSIM3v1AlnlxGiven   :1;
    unsigned  BSIM3v1Aldvt0Given   :1;   
    unsigned  BSIM3v1Aldvt1Given   :1;     
    unsigned  BSIM3v1Aldvt2Given   :1;     
    unsigned  BSIM3v1Aldvt0wGiven   :1;   
    unsigned  BSIM3v1Aldvt1wGiven   :1;     
    unsigned  BSIM3v1Aldvt2wGiven   :1;     
    unsigned  BSIM3v1AldroutGiven   :1;     
    unsigned  BSIM3v1AldsubGiven   :1;     
    unsigned  BSIM3v1Alvth0Given   :1;
    unsigned  BSIM3v1AluaGiven   :1;
    unsigned  BSIM3v1Alua1Given   :1;
    unsigned  BSIM3v1AlubGiven   :1;
    unsigned  BSIM3v1Alub1Given   :1;
    unsigned  BSIM3v1AlucGiven   :1;
    unsigned  BSIM3v1Aluc1Given   :1;
    unsigned  BSIM3v1Alu0Given   :1;
    unsigned  BSIM3v1AluteGiven   :1;
    unsigned  BSIM3v1AlvoffGiven   :1;
    unsigned  BSIM3v1AlrdswGiven   :1;      
    unsigned  BSIM3v1AlprwgGiven   :1;      
    unsigned  BSIM3v1AlprwbGiven   :1;      
    unsigned  BSIM3v1AlprtGiven   :1;      
    unsigned  BSIM3v1Aleta0Given   :1;    
    unsigned  BSIM3v1AletabGiven   :1;    
    unsigned  BSIM3v1AlpclmGiven   :1;   
    unsigned  BSIM3v1Alpdibl1Given   :1;   
    unsigned  BSIM3v1Alpdibl2Given   :1;  
    unsigned  BSIM3v1AlpdiblbGiven   :1;  
    unsigned  BSIM3v1Alpscbe1Given   :1;    
    unsigned  BSIM3v1Alpscbe2Given   :1;    
    unsigned  BSIM3v1AlpvagGiven   :1;    
    unsigned  BSIM3v1AldeltaGiven  :1;     
    unsigned  BSIM3v1AlwrGiven   :1;
    unsigned  BSIM3v1AldwgGiven   :1;
    unsigned  BSIM3v1AldwbGiven   :1;
    unsigned  BSIM3v1Alb0Given   :1;
    unsigned  BSIM3v1Alb1Given   :1;
    unsigned  BSIM3v1Alalpha0Given   :1;
    unsigned  BSIM3v1Albeta0Given   :1;

    /* CV model */
    unsigned  BSIM3v1AlelmGiven  :1;     
    unsigned  BSIM3v1AlcgslGiven   :1;
    unsigned  BSIM3v1AlcgdlGiven   :1;
    unsigned  BSIM3v1AlckappaGiven   :1;
    unsigned  BSIM3v1AlcfGiven   :1;
    unsigned  BSIM3v1AlclcGiven   :1;
    unsigned  BSIM3v1AlcleGiven   :1;

    /* Width dependence */
    unsigned  BSIM3v1AwcdscGiven   :1;
    unsigned  BSIM3v1AwcdscbGiven   :1;
    unsigned  BSIM3v1AwcdscdGiven   :1;
    unsigned  BSIM3v1AwcitGiven   :1;
    unsigned  BSIM3v1AwnfactorGiven   :1;
    unsigned  BSIM3v1AwxjGiven   :1;
    unsigned  BSIM3v1AwvsatGiven   :1;
    unsigned  BSIM3v1AwatGiven   :1;
    unsigned  BSIM3v1Awa0Given   :1;
    unsigned  BSIM3v1AwagsGiven   :1;
    unsigned  BSIM3v1Awa1Given   :1;
    unsigned  BSIM3v1Awa2Given   :1;
    unsigned  BSIM3v1AwketaGiven   :1;    
    unsigned  BSIM3v1AwnsubGiven   :1;
    unsigned  BSIM3v1AwnpeakGiven   :1;
    unsigned  BSIM3v1AwngateGiven   :1;
    unsigned  BSIM3v1Awgamma1Given   :1;
    unsigned  BSIM3v1Awgamma2Given   :1;
    unsigned  BSIM3v1AwvbxGiven   :1;
    unsigned  BSIM3v1AwvbmGiven   :1;
    unsigned  BSIM3v1AwxtGiven   :1;
    unsigned  BSIM3v1Awk1Given   :1;
    unsigned  BSIM3v1Awkt1Given   :1;
    unsigned  BSIM3v1Awkt1lGiven   :1;
    unsigned  BSIM3v1Awkt2Given   :1;
    unsigned  BSIM3v1Awk2Given   :1;
    unsigned  BSIM3v1Awk3Given   :1;
    unsigned  BSIM3v1Awk3bGiven   :1;
    unsigned  BSIM3v1Aww0Given   :1;
    unsigned  BSIM3v1AwnlxGiven   :1;
    unsigned  BSIM3v1Awdvt0Given   :1;   
    unsigned  BSIM3v1Awdvt1Given   :1;     
    unsigned  BSIM3v1Awdvt2Given   :1;     
    unsigned  BSIM3v1Awdvt0wGiven   :1;   
    unsigned  BSIM3v1Awdvt1wGiven   :1;     
    unsigned  BSIM3v1Awdvt2wGiven   :1;     
    unsigned  BSIM3v1AwdroutGiven   :1;     
    unsigned  BSIM3v1AwdsubGiven   :1;     
    unsigned  BSIM3v1Awvth0Given   :1;
    unsigned  BSIM3v1AwuaGiven   :1;
    unsigned  BSIM3v1Awua1Given   :1;
    unsigned  BSIM3v1AwubGiven   :1;
    unsigned  BSIM3v1Awub1Given   :1;
    unsigned  BSIM3v1AwucGiven   :1;
    unsigned  BSIM3v1Awuc1Given   :1;
    unsigned  BSIM3v1Awu0Given   :1;
    unsigned  BSIM3v1AwuteGiven   :1;
    unsigned  BSIM3v1AwvoffGiven   :1;
    unsigned  BSIM3v1AwrdswGiven   :1;      
    unsigned  BSIM3v1AwprwgGiven   :1;      
    unsigned  BSIM3v1AwprwbGiven   :1;      
    unsigned  BSIM3v1AwprtGiven   :1;      
    unsigned  BSIM3v1Aweta0Given   :1;    
    unsigned  BSIM3v1AwetabGiven   :1;    
    unsigned  BSIM3v1AwpclmGiven   :1;   
    unsigned  BSIM3v1Awpdibl1Given   :1;   
    unsigned  BSIM3v1Awpdibl2Given   :1;  
    unsigned  BSIM3v1AwpdiblbGiven   :1;  
    unsigned  BSIM3v1Awpscbe1Given   :1;    
    unsigned  BSIM3v1Awpscbe2Given   :1;    
    unsigned  BSIM3v1AwpvagGiven   :1;    
    unsigned  BSIM3v1AwdeltaGiven  :1;     
    unsigned  BSIM3v1AwwrGiven   :1;
    unsigned  BSIM3v1AwdwgGiven   :1;
    unsigned  BSIM3v1AwdwbGiven   :1;
    unsigned  BSIM3v1Awb0Given   :1;
    unsigned  BSIM3v1Awb1Given   :1;
    unsigned  BSIM3v1Awalpha0Given   :1;
    unsigned  BSIM3v1Awbeta0Given   :1;

    /* CV model */
    unsigned  BSIM3v1AwelmGiven  :1;     
    unsigned  BSIM3v1AwcgslGiven   :1;
    unsigned  BSIM3v1AwcgdlGiven   :1;
    unsigned  BSIM3v1AwckappaGiven   :1;
    unsigned  BSIM3v1AwcfGiven   :1;
    unsigned  BSIM3v1AwclcGiven   :1;
    unsigned  BSIM3v1AwcleGiven   :1;

    /* Cross-term dependence */
    unsigned  BSIM3v1ApcdscGiven   :1;
    unsigned  BSIM3v1ApcdscbGiven   :1;
    unsigned  BSIM3v1ApcdscdGiven   :1;
    unsigned  BSIM3v1ApcitGiven   :1;
    unsigned  BSIM3v1ApnfactorGiven   :1;
    unsigned  BSIM3v1ApxjGiven   :1;
    unsigned  BSIM3v1ApvsatGiven   :1;
    unsigned  BSIM3v1ApatGiven   :1;
    unsigned  BSIM3v1Apa0Given   :1;
    unsigned  BSIM3v1ApagsGiven   :1;
    unsigned  BSIM3v1Apa1Given   :1;
    unsigned  BSIM3v1Apa2Given   :1;
    unsigned  BSIM3v1ApketaGiven   :1;    
    unsigned  BSIM3v1ApnsubGiven   :1;
    unsigned  BSIM3v1ApnpeakGiven   :1;
    unsigned  BSIM3v1ApngateGiven   :1;
    unsigned  BSIM3v1Apgamma1Given   :1;
    unsigned  BSIM3v1Apgamma2Given   :1;
    unsigned  BSIM3v1ApvbxGiven   :1;
    unsigned  BSIM3v1ApvbmGiven   :1;
    unsigned  BSIM3v1ApxtGiven   :1;
    unsigned  BSIM3v1Apk1Given   :1;
    unsigned  BSIM3v1Apkt1Given   :1;
    unsigned  BSIM3v1Apkt1lGiven   :1;
    unsigned  BSIM3v1Apkt2Given   :1;
    unsigned  BSIM3v1Apk2Given   :1;
    unsigned  BSIM3v1Apk3Given   :1;
    unsigned  BSIM3v1Apk3bGiven   :1;
    unsigned  BSIM3v1Apw0Given   :1;
    unsigned  BSIM3v1ApnlxGiven   :1;
    unsigned  BSIM3v1Apdvt0Given   :1;   
    unsigned  BSIM3v1Apdvt1Given   :1;     
    unsigned  BSIM3v1Apdvt2Given   :1;     
    unsigned  BSIM3v1Apdvt0wGiven   :1;   
    unsigned  BSIM3v1Apdvt1wGiven   :1;     
    unsigned  BSIM3v1Apdvt2wGiven   :1;     
    unsigned  BSIM3v1ApdroutGiven   :1;     
    unsigned  BSIM3v1ApdsubGiven   :1;     
    unsigned  BSIM3v1Apvth0Given   :1;
    unsigned  BSIM3v1ApuaGiven   :1;
    unsigned  BSIM3v1Apua1Given   :1;
    unsigned  BSIM3v1ApubGiven   :1;
    unsigned  BSIM3v1Apub1Given   :1;
    unsigned  BSIM3v1ApucGiven   :1;
    unsigned  BSIM3v1Apuc1Given   :1;
    unsigned  BSIM3v1Apu0Given   :1;
    unsigned  BSIM3v1AputeGiven   :1;
    unsigned  BSIM3v1ApvoffGiven   :1;
    unsigned  BSIM3v1AprdswGiven   :1;      
    unsigned  BSIM3v1ApprwgGiven   :1;      
    unsigned  BSIM3v1ApprwbGiven   :1;      
    unsigned  BSIM3v1ApprtGiven   :1;      
    unsigned  BSIM3v1Apeta0Given   :1;    
    unsigned  BSIM3v1ApetabGiven   :1;    
    unsigned  BSIM3v1AppclmGiven   :1;   
    unsigned  BSIM3v1Appdibl1Given   :1;   
    unsigned  BSIM3v1Appdibl2Given   :1;  
    unsigned  BSIM3v1AppdiblbGiven   :1;  
    unsigned  BSIM3v1Appscbe1Given   :1;    
    unsigned  BSIM3v1Appscbe2Given   :1;    
    unsigned  BSIM3v1AppvagGiven   :1;    
    unsigned  BSIM3v1ApdeltaGiven  :1;     
    unsigned  BSIM3v1ApwrGiven   :1;
    unsigned  BSIM3v1ApdwgGiven   :1;
    unsigned  BSIM3v1ApdwbGiven   :1;
    unsigned  BSIM3v1Apb0Given   :1;
    unsigned  BSIM3v1Apb1Given   :1;
    unsigned  BSIM3v1Apalpha0Given   :1;
    unsigned  BSIM3v1Apbeta0Given   :1;

    /* CV model */
    unsigned  BSIM3v1ApelmGiven  :1;     
    unsigned  BSIM3v1ApcgslGiven   :1;
    unsigned  BSIM3v1ApcgdlGiven   :1;
    unsigned  BSIM3v1ApckappaGiven   :1;
    unsigned  BSIM3v1ApcfGiven   :1;
    unsigned  BSIM3v1ApclcGiven   :1;
    unsigned  BSIM3v1ApcleGiven   :1;

    unsigned  BSIM3v1AuseFringeGiven   :1;

    unsigned  BSIM3v1AtnomGiven   :1;
    unsigned  BSIM3v1AcgsoGiven   :1;
    unsigned  BSIM3v1AcgdoGiven   :1;
    unsigned  BSIM3v1AcgboGiven   :1;
    unsigned  BSIM3v1AxpartGiven   :1;
    unsigned  BSIM3v1AsheetResistanceGiven   :1;
    unsigned  BSIM3v1AjctSatCurDensityGiven   :1;
    unsigned  BSIM3v1AbulkJctPotentialGiven   :1;
    unsigned  BSIM3v1AbulkJctBotGradingCoeffGiven   :1;
    unsigned  BSIM3v1AsidewallJctPotentialGiven   :1;
    unsigned  BSIM3v1AbulkJctSideGradingCoeffGiven   :1;
    unsigned  BSIM3v1AunitAreaJctCapGiven   :1;
    unsigned  BSIM3v1AunitLengthSidewallJctCapGiven   :1;

    unsigned  BSIM3v1AoxideTrapDensityAGiven  :1;         
    unsigned  BSIM3v1AoxideTrapDensityBGiven  :1;        
    unsigned  BSIM3v1AoxideTrapDensityCGiven  :1;     
    unsigned  BSIM3v1AemGiven  :1;     
    unsigned  BSIM3v1AefGiven  :1;     
    unsigned  BSIM3v1AafGiven  :1;     
    unsigned  BSIM3v1AkfGiven  :1;     

    unsigned  BSIM3v1ALintGiven   :1;
    unsigned  BSIM3v1ALlGiven   :1;
    unsigned  BSIM3v1ALlnGiven   :1;
    unsigned  BSIM3v1ALwGiven   :1;
    unsigned  BSIM3v1ALwnGiven   :1;
    unsigned  BSIM3v1ALwlGiven   :1;
    unsigned  BSIM3v1ALminGiven   :1;
    unsigned  BSIM3v1ALmaxGiven   :1;

    unsigned  BSIM3v1AWintGiven   :1;
    unsigned  BSIM3v1AWlGiven   :1;
    unsigned  BSIM3v1AWlnGiven   :1;
    unsigned  BSIM3v1AWwGiven   :1;
    unsigned  BSIM3v1AWwnGiven   :1;
    unsigned  BSIM3v1AWwlGiven   :1;
    unsigned  BSIM3v1AWminGiven   :1;
    unsigned  BSIM3v1AWmaxGiven   :1;

} BSIM3v1Amodel;


#ifndef NMOS
#define NMOS 1
#define PMOS -1
#endif /*NMOS*/


/* device parameters */
#define BSIM3v1A_W 1
#define BSIM3v1A_L 2
#define BSIM3v1A_M 15
#define BSIM3v1A_AS 3
#define BSIM3v1A_AD 4
#define BSIM3v1A_PS 5
#define BSIM3v1A_PD 6
#define BSIM3v1A_NRS 7
#define BSIM3v1A_NRD 8
#define BSIM3v1A_OFF 9
#define BSIM3v1A_IC_VBS 10
#define BSIM3v1A_IC_VDS 11
#define BSIM3v1A_IC_VGS 12
#define BSIM3v1A_IC 13
#define BSIM3v1A_NQSMOD 14

/* model parameters */
#define BSIM3v1A_MOD_CAPMOD          101
#define BSIM3v1A_MOD_NQSMOD          102
#define BSIM3v1A_MOD_MOBMOD          103    
#define BSIM3v1A_MOD_NOIMOD          104    

#define BSIM3v1A_MOD_TOX             105

#define BSIM3v1A_MOD_CDSC            106
#define BSIM3v1A_MOD_CDSCB           107
#define BSIM3v1A_MOD_CIT             108
#define BSIM3v1A_MOD_NFACTOR         109
#define BSIM3v1A_MOD_XJ              110
#define BSIM3v1A_MOD_VSAT            111
#define BSIM3v1A_MOD_AT              112
#define BSIM3v1A_MOD_A0              113
#define BSIM3v1A_MOD_A1              114
#define BSIM3v1A_MOD_A2              115
#define BSIM3v1A_MOD_KETA            116   
#define BSIM3v1A_MOD_NSUB            117
#define BSIM3v1A_MOD_NPEAK           118
#define BSIM3v1A_MOD_NGATE           120
#define BSIM3v1A_MOD_GAMMA1          121
#define BSIM3v1A_MOD_GAMMA2          122
#define BSIM3v1A_MOD_VBX             123
#define BSIM3v1A_MOD_BINUNIT         124    

#define BSIM3v1A_MOD_VBM             125

#define BSIM3v1A_MOD_XT              126
#define BSIM3v1A_MOD_K1              129
#define BSIM3v1A_MOD_KT1             130
#define BSIM3v1A_MOD_KT1L            131
#define BSIM3v1A_MOD_K2              132
#define BSIM3v1A_MOD_KT2             133
#define BSIM3v1A_MOD_K3              134
#define BSIM3v1A_MOD_K3B             135
#define BSIM3v1A_MOD_W0              136
#define BSIM3v1A_MOD_NLX             137

#define BSIM3v1A_MOD_DVT0            138
#define BSIM3v1A_MOD_DVT1            139
#define BSIM3v1A_MOD_DVT2            140

#define BSIM3v1A_MOD_DVT0W           141
#define BSIM3v1A_MOD_DVT1W           142
#define BSIM3v1A_MOD_DVT2W           143

#define BSIM3v1A_MOD_DROUT           144
#define BSIM3v1A_MOD_DSUB            145
#define BSIM3v1A_MOD_VTH0            146
#define BSIM3v1A_MOD_UA              147
#define BSIM3v1A_MOD_UA1             148
#define BSIM3v1A_MOD_UB              149
#define BSIM3v1A_MOD_UB1             150
#define BSIM3v1A_MOD_UC              151
#define BSIM3v1A_MOD_UC1             152
#define BSIM3v1A_MOD_U0              153
#define BSIM3v1A_MOD_UTE             154
#define BSIM3v1A_MOD_VOFF            155
#define BSIM3v1A_MOD_DELTA           156
#define BSIM3v1A_MOD_RDSW            157
#define BSIM3v1A_MOD_PRT             158
#define BSIM3v1A_MOD_LDD             159
#define BSIM3v1A_MOD_ETA             160
#define BSIM3v1A_MOD_ETA0            161
#define BSIM3v1A_MOD_ETAB            162
#define BSIM3v1A_MOD_PCLM            163
#define BSIM3v1A_MOD_PDIBL1          164
#define BSIM3v1A_MOD_PDIBL2          165
#define BSIM3v1A_MOD_PSCBE1          166
#define BSIM3v1A_MOD_PSCBE2          167
#define BSIM3v1A_MOD_PVAG            168
#define BSIM3v1A_MOD_WR              169
#define BSIM3v1A_MOD_DWG             170
#define BSIM3v1A_MOD_DWB             171
#define BSIM3v1A_MOD_B0              172
#define BSIM3v1A_MOD_B1              173
#define BSIM3v1A_MOD_ALPHA0          174
#define BSIM3v1A_MOD_BETA0           175
#define BSIM3v1A_MOD_PDIBLB          178

#define BSIM3v1A_MOD_PRWG            179
#define BSIM3v1A_MOD_PRWB            180

#define BSIM3v1A_MOD_CDSCD           181
#define BSIM3v1A_MOD_AGS             182

#define BSIM3v1A_MOD_FRINGE          184
#define BSIM3v1A_MOD_ELM             185
#define BSIM3v1A_MOD_CGSL            186
#define BSIM3v1A_MOD_CGDL            187
#define BSIM3v1A_MOD_CKAPPA          188
#define BSIM3v1A_MOD_CF              189
#define BSIM3v1A_MOD_CLC             190
#define BSIM3v1A_MOD_CLE             191

/* Length dependence */
#define BSIM3v1A_MOD_LCDSC            201
#define BSIM3v1A_MOD_LCDSCB           202
#define BSIM3v1A_MOD_LCIT             203
#define BSIM3v1A_MOD_LNFACTOR         204
#define BSIM3v1A_MOD_LXJ              205
#define BSIM3v1A_MOD_LVSAT            206
#define BSIM3v1A_MOD_LAT              207
#define BSIM3v1A_MOD_LA0              208
#define BSIM3v1A_MOD_LA1              209
#define BSIM3v1A_MOD_LA2              210
#define BSIM3v1A_MOD_LKETA            211   
#define BSIM3v1A_MOD_LNSUB            212
#define BSIM3v1A_MOD_LNPEAK           213
#define BSIM3v1A_MOD_LNGATE           215
#define BSIM3v1A_MOD_LGAMMA1          216
#define BSIM3v1A_MOD_LGAMMA2          217
#define BSIM3v1A_MOD_LVBX             218

#define BSIM3v1A_MOD_LVBM             220

#define BSIM3v1A_MOD_LXT              222
#define BSIM3v1A_MOD_LK1              225
#define BSIM3v1A_MOD_LKT1             226
#define BSIM3v1A_MOD_LKT1L            227
#define BSIM3v1A_MOD_LK2              228
#define BSIM3v1A_MOD_LKT2             229
#define BSIM3v1A_MOD_LK3              230
#define BSIM3v1A_MOD_LK3B             231
#define BSIM3v1A_MOD_LW0              232
#define BSIM3v1A_MOD_LNLX             233

#define BSIM3v1A_MOD_LDVT0            234
#define BSIM3v1A_MOD_LDVT1            235
#define BSIM3v1A_MOD_LDVT2            236

#define BSIM3v1A_MOD_LDVT0W           237
#define BSIM3v1A_MOD_LDVT1W           238
#define BSIM3v1A_MOD_LDVT2W           239

#define BSIM3v1A_MOD_LDROUT           240
#define BSIM3v1A_MOD_LDSUB            241
#define BSIM3v1A_MOD_LVTH0            242
#define BSIM3v1A_MOD_LUA              243
#define BSIM3v1A_MOD_LUA1             244
#define BSIM3v1A_MOD_LUB              245
#define BSIM3v1A_MOD_LUB1             246
#define BSIM3v1A_MOD_LUC              247
#define BSIM3v1A_MOD_LUC1             248
#define BSIM3v1A_MOD_LU0              249
#define BSIM3v1A_MOD_LUTE             250
#define BSIM3v1A_MOD_LVOFF            251
#define BSIM3v1A_MOD_LDELTA           252
#define BSIM3v1A_MOD_LRDSW            253
#define BSIM3v1A_MOD_LPRT             254
#define BSIM3v1A_MOD_LLDD             255
#define BSIM3v1A_MOD_LETA             256
#define BSIM3v1A_MOD_LETA0            257
#define BSIM3v1A_MOD_LETAB            258
#define BSIM3v1A_MOD_LPCLM            259
#define BSIM3v1A_MOD_LPDIBL1          260
#define BSIM3v1A_MOD_LPDIBL2          261
#define BSIM3v1A_MOD_LPSCBE1          262
#define BSIM3v1A_MOD_LPSCBE2          263
#define BSIM3v1A_MOD_LPVAG            264
#define BSIM3v1A_MOD_LWR              265
#define BSIM3v1A_MOD_LDWG             266
#define BSIM3v1A_MOD_LDWB             267
#define BSIM3v1A_MOD_LB0              268
#define BSIM3v1A_MOD_LB1              269
#define BSIM3v1A_MOD_LALPHA0          270
#define BSIM3v1A_MOD_LBETA0           271
#define BSIM3v1A_MOD_LPDIBLB          274

#define BSIM3v1A_MOD_LPRWG            275
#define BSIM3v1A_MOD_LPRWB            276

#define BSIM3v1A_MOD_LCDSCD           277
#define BSIM3v1A_MOD_LAGS            278
                                    

#define BSIM3v1A_MOD_LFRINGE          281
#define BSIM3v1A_MOD_LELM             282
#define BSIM3v1A_MOD_LCGSL            283
#define BSIM3v1A_MOD_LCGDL            284
#define BSIM3v1A_MOD_LCKAPPA          285
#define BSIM3v1A_MOD_LCF              286
#define BSIM3v1A_MOD_LCLC             287
#define BSIM3v1A_MOD_LCLE             288

/* Width dependence */
#define BSIM3v1A_MOD_WCDSC            301
#define BSIM3v1A_MOD_WCDSCB           302
#define BSIM3v1A_MOD_WCIT             303
#define BSIM3v1A_MOD_WNFACTOR         304
#define BSIM3v1A_MOD_WXJ              305
#define BSIM3v1A_MOD_WVSAT            306
#define BSIM3v1A_MOD_WAT              307
#define BSIM3v1A_MOD_WA0              308
#define BSIM3v1A_MOD_WA1              309
#define BSIM3v1A_MOD_WA2              310
#define BSIM3v1A_MOD_WKETA            311   
#define BSIM3v1A_MOD_WNSUB            312
#define BSIM3v1A_MOD_WNPEAK           313
#define BSIM3v1A_MOD_WNGATE           315
#define BSIM3v1A_MOD_WGAMMA1          316
#define BSIM3v1A_MOD_WGAMMA2          317
#define BSIM3v1A_MOD_WVBX             318

#define BSIM3v1A_MOD_WVBM             320

#define BSIM3v1A_MOD_WXT              322
#define BSIM3v1A_MOD_WK1              325
#define BSIM3v1A_MOD_WKT1             326
#define BSIM3v1A_MOD_WKT1L            327
#define BSIM3v1A_MOD_WK2              328
#define BSIM3v1A_MOD_WKT2             329
#define BSIM3v1A_MOD_WK3              330
#define BSIM3v1A_MOD_WK3B             331
#define BSIM3v1A_MOD_WW0              332
#define BSIM3v1A_MOD_WNLX             333

#define BSIM3v1A_MOD_WDVT0            334
#define BSIM3v1A_MOD_WDVT1            335
#define BSIM3v1A_MOD_WDVT2            336

#define BSIM3v1A_MOD_WDVT0W           337
#define BSIM3v1A_MOD_WDVT1W           338
#define BSIM3v1A_MOD_WDVT2W           339

#define BSIM3v1A_MOD_WDROUT           340
#define BSIM3v1A_MOD_WDSUB            341
#define BSIM3v1A_MOD_WVTH0            342
#define BSIM3v1A_MOD_WUA              343
#define BSIM3v1A_MOD_WUA1             344
#define BSIM3v1A_MOD_WUB              345
#define BSIM3v1A_MOD_WUB1             346
#define BSIM3v1A_MOD_WUC              347
#define BSIM3v1A_MOD_WUC1             348
#define BSIM3v1A_MOD_WU0              349
#define BSIM3v1A_MOD_WUTE             350
#define BSIM3v1A_MOD_WVOFF            351
#define BSIM3v1A_MOD_WDELTA           352
#define BSIM3v1A_MOD_WRDSW            353
#define BSIM3v1A_MOD_WPRT             354
#define BSIM3v1A_MOD_WLDD             355
#define BSIM3v1A_MOD_WETA             356
#define BSIM3v1A_MOD_WETA0            357
#define BSIM3v1A_MOD_WETAB            358
#define BSIM3v1A_MOD_WPCLM            359
#define BSIM3v1A_MOD_WPDIBL1          360
#define BSIM3v1A_MOD_WPDIBL2          361
#define BSIM3v1A_MOD_WPSCBE1          362
#define BSIM3v1A_MOD_WPSCBE2          363
#define BSIM3v1A_MOD_WPVAG            364
#define BSIM3v1A_MOD_WWR              365
#define BSIM3v1A_MOD_WDWG             366
#define BSIM3v1A_MOD_WDWB             367
#define BSIM3v1A_MOD_WB0              368
#define BSIM3v1A_MOD_WB1              369
#define BSIM3v1A_MOD_WALPHA0          370
#define BSIM3v1A_MOD_WBETA0           371
#define BSIM3v1A_MOD_WPDIBLB          374

#define BSIM3v1A_MOD_WPRWG            375
#define BSIM3v1A_MOD_WPRWB            376

#define BSIM3v1A_MOD_WCDSCD           377
#define BSIM3v1A_MOD_WAGS             378


#define BSIM3v1A_MOD_WFRINGE          381
#define BSIM3v1A_MOD_WELM             382
#define BSIM3v1A_MOD_WCGSL            383
#define BSIM3v1A_MOD_WCGDL            384
#define BSIM3v1A_MOD_WCKAPPA          385
#define BSIM3v1A_MOD_WCF              386
#define BSIM3v1A_MOD_WCLC             387
#define BSIM3v1A_MOD_WCLE             388

/* Cross-term dependence */
#define BSIM3v1A_MOD_PCDSC            401
#define BSIM3v1A_MOD_PCDSCB           402
#define BSIM3v1A_MOD_PCIT             403
#define BSIM3v1A_MOD_PNFACTOR         404
#define BSIM3v1A_MOD_PXJ              405
#define BSIM3v1A_MOD_PVSAT            406
#define BSIM3v1A_MOD_PAT              407
#define BSIM3v1A_MOD_PA0              408
#define BSIM3v1A_MOD_PA1              409
#define BSIM3v1A_MOD_PA2              410
#define BSIM3v1A_MOD_PKETA            411   
#define BSIM3v1A_MOD_PNSUB            412
#define BSIM3v1A_MOD_PNPEAK           413
#define BSIM3v1A_MOD_PNGATE           415
#define BSIM3v1A_MOD_PGAMMA1          416
#define BSIM3v1A_MOD_PGAMMA2          417
#define BSIM3v1A_MOD_PVBX             418

#define BSIM3v1A_MOD_PVBM             420

#define BSIM3v1A_MOD_PXT              422
#define BSIM3v1A_MOD_PK1              425
#define BSIM3v1A_MOD_PKT1             426
#define BSIM3v1A_MOD_PKT1L            427
#define BSIM3v1A_MOD_PK2              428
#define BSIM3v1A_MOD_PKT2             429
#define BSIM3v1A_MOD_PK3              430
#define BSIM3v1A_MOD_PK3B             431
#define BSIM3v1A_MOD_PW0              432
#define BSIM3v1A_MOD_PNLX             433

#define BSIM3v1A_MOD_PDVT0            434
#define BSIM3v1A_MOD_PDVT1            435
#define BSIM3v1A_MOD_PDVT2            436

#define BSIM3v1A_MOD_PDVT0W           437
#define BSIM3v1A_MOD_PDVT1W           438
#define BSIM3v1A_MOD_PDVT2W           439

#define BSIM3v1A_MOD_PDROUT           440
#define BSIM3v1A_MOD_PDSUB            441
#define BSIM3v1A_MOD_PVTH0            442
#define BSIM3v1A_MOD_PUA              443
#define BSIM3v1A_MOD_PUA1             444
#define BSIM3v1A_MOD_PUB              445
#define BSIM3v1A_MOD_PUB1             446
#define BSIM3v1A_MOD_PUC              447
#define BSIM3v1A_MOD_PUC1             448
#define BSIM3v1A_MOD_PU0              449
#define BSIM3v1A_MOD_PUTE             450
#define BSIM3v1A_MOD_PVOFF            451
#define BSIM3v1A_MOD_PDELTA           452
#define BSIM3v1A_MOD_PRDSW            453
#define BSIM3v1A_MOD_PPRT             454
#define BSIM3v1A_MOD_PLDD             455
#define BSIM3v1A_MOD_PETA             456
#define BSIM3v1A_MOD_PETA0            457
#define BSIM3v1A_MOD_PETAB            458
#define BSIM3v1A_MOD_PPCLM            459
#define BSIM3v1A_MOD_PPDIBL1          460
#define BSIM3v1A_MOD_PPDIBL2          461
#define BSIM3v1A_MOD_PPSCBE1          462
#define BSIM3v1A_MOD_PPSCBE2          463
#define BSIM3v1A_MOD_PPVAG            464
#define BSIM3v1A_MOD_PWR              465
#define BSIM3v1A_MOD_PDWG             466
#define BSIM3v1A_MOD_PDWB             467
#define BSIM3v1A_MOD_PB0              468
#define BSIM3v1A_MOD_PB1              469
#define BSIM3v1A_MOD_PALPHA0          470
#define BSIM3v1A_MOD_PBETA0           471
#define BSIM3v1A_MOD_PPDIBLB          474

#define BSIM3v1A_MOD_PPRWG            475
#define BSIM3v1A_MOD_PPRWB            476

#define BSIM3v1A_MOD_PCDSCD           477
#define BSIM3v1A_MOD_PAGS             478

#define BSIM3v1A_MOD_PFRINGE          481
#define BSIM3v1A_MOD_PELM             482
#define BSIM3v1A_MOD_PCGSL            483
#define BSIM3v1A_MOD_PCGDL            484
#define BSIM3v1A_MOD_PCKAPPA          485
#define BSIM3v1A_MOD_PCF              486
#define BSIM3v1A_MOD_PCLC             487
#define BSIM3v1A_MOD_PCLE             488

#define BSIM3v1A_MOD_TNOM             501
#define BSIM3v1A_MOD_CGSO             502
#define BSIM3v1A_MOD_CGDO             503
#define BSIM3v1A_MOD_CGBO             504
#define BSIM3v1A_MOD_XPART            505

#define BSIM3v1A_MOD_RSH              506
#define BSIM3v1A_MOD_JS               507
#define BSIM3v1A_MOD_PB               508
#define BSIM3v1A_MOD_MJ               509
#define BSIM3v1A_MOD_PBSW             510
#define BSIM3v1A_MOD_MJSW             511
#define BSIM3v1A_MOD_CJ               512
#define BSIM3v1A_MOD_CJSW             513
#define BSIM3v1A_MOD_NMOS             514
#define BSIM3v1A_MOD_PMOS             515

#define BSIM3v1A_MOD_NOIA             516
#define BSIM3v1A_MOD_NOIB             517
#define BSIM3v1A_MOD_NOIC             518

#define BSIM3v1A_MOD_LINT             519
#define BSIM3v1A_MOD_LL               520
#define BSIM3v1A_MOD_LLN              521
#define BSIM3v1A_MOD_LW               522
#define BSIM3v1A_MOD_LWN              523
#define BSIM3v1A_MOD_LWL              524
#define BSIM3v1A_MOD_LMIN             525
#define BSIM3v1A_MOD_LMAX             526

#define BSIM3v1A_MOD_WINT             527
#define BSIM3v1A_MOD_WL               528
#define BSIM3v1A_MOD_WLN              529
#define BSIM3v1A_MOD_WW               530
#define BSIM3v1A_MOD_WWN              531
#define BSIM3v1A_MOD_WWL              532
#define BSIM3v1A_MOD_WMIN             533
#define BSIM3v1A_MOD_WMAX             534

#define BSIM3v1A_MOD_DWC              535
#define BSIM3v1A_MOD_DLC              536

#define BSIM3v1A_MOD_EM               537
#define BSIM3v1A_MOD_EF               538
#define BSIM3v1A_MOD_AF               539
#define BSIM3v1A_MOD_KF               540


/* device questions */
#define BSIM3v1A_DNODE                601
#define BSIM3v1A_GNODE                602
#define BSIM3v1A_SNODE                603
#define BSIM3v1A_BNODE                604
#define BSIM3v1A_DNODEPRIME           605
#define BSIM3v1A_SNODEPRIME           606
#define BSIM3v1A_VBD                  607
#define BSIM3v1A_VBS                  608
#define BSIM3v1A_VGS                  609
#define BSIM3v1A_VDS                  610
#define BSIM3v1A_CD                   611
#define BSIM3v1A_CBS                  612
#define BSIM3v1A_CBD                  613
#define BSIM3v1A_GM                   614
#define BSIM3v1A_GDS                  615
#define BSIM3v1A_GMBS                 616
#define BSIM3v1A_GBD                  617
#define BSIM3v1A_GBS                  618
#define BSIM3v1A_QB                   619
#define BSIM3v1A_CQB                  620
#define BSIM3v1A_QG                   621
#define BSIM3v1A_CQG                  622
#define BSIM3v1A_QD                   623
#define BSIM3v1A_CQD                  624
#define BSIM3v1A_CGG                  625
#define BSIM3v1A_CGD                  626
#define BSIM3v1A_CGS                  627
#define BSIM3v1A_CBG                  628
#define BSIM3v1A_CAPBD                629
#define BSIM3v1A_CQBD                 630
#define BSIM3v1A_CAPBS                631
#define BSIM3v1A_CQBS                 632
#define BSIM3v1A_CDG                  633
#define BSIM3v1A_CDD                  634
#define BSIM3v1A_CDS                  635
#define BSIM3v1A_VON                  636
#define BSIM3v1A_VDSAT                637
#define BSIM3v1A_QBS                  638
#define BSIM3v1A_QBD                  639
#define BSIM3v1A_SOURCECONDUCT        640
#define BSIM3v1A_DRAINCONDUCT         641
#define BSIM3v1A_CBDB                 642
#define BSIM3v1A_CBSB                 643

#include "bsim3v1aext.h"

extern void BSIM3v1Aevaluate(double,double,double,BSIM3v1Ainstance*,BSIM3v1Amodel*,
        double*,double*,double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, CKTcircuit*);
#endif /*BSIM3v1A*/



