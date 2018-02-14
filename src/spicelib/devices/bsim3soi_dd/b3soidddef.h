/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: Weidong Liu and Pin Su       Feb 1999
Author: 1998 Samuel Fung
Modified by Pin Su, Wei Jin 99/9/27
File: b3soidddef.h
Modified by Paolo Nenzi 2002
**********/

#ifndef B3SOIDD
#define B3SOIDD

#define SOICODE
/*  #define BULKCODE  */

#include "ngspice/ifsim.h"
#include "ngspice/gendefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/noisedef.h"         

typedef struct sB3SOIDDinstance
{

    struct GENinstance gen;

#define B3SOIDDmodPtr(inst) ((struct sB3SOIDDmodel *)((inst)->gen.GENmodPtr))
#define B3SOIDDnextInstance(inst) ((struct sB3SOIDDinstance *)((inst)->gen.GENnextInstance))
#define B3SOIDDname gen.GENname
#define B3SOIDDstates gen.GENstate

    const int B3SOIDDdNode;
    const int B3SOIDDgNode;
    const int B3SOIDDsNode;
    const int B3SOIDDeNode;
    const int B3SOIDDbNodeExt;
    const int B3SOIDDtempNodeExt;
    const int B3SOIDDpNodeExt;
    int B3SOIDDbNode;
    int B3SOIDDtempNode;
    int B3SOIDDpNode;  
    int B3SOIDDdNodePrime;
    int B3SOIDDsNodePrime;

    int B3SOIDDvbsNode;
    /* for Debug */
    int B3SOIDDidsNode;
    int B3SOIDDicNode;
    int B3SOIDDibsNode;
    int B3SOIDDibdNode;
    int B3SOIDDiiiNode;
    int B3SOIDDigidlNode;
    int B3SOIDDitunNode;
    int B3SOIDDibpNode;
    int B3SOIDDabeffNode;
    int B3SOIDDvbs0effNode;
    int B3SOIDDvbseffNode;
    int B3SOIDDxcNode;
    int B3SOIDDcbbNode;
    int B3SOIDDcbdNode;
    int B3SOIDDcbeNode;
    int B3SOIDDcbgNode;
    int B3SOIDDqbNode;
    int B3SOIDDqbfNode;
    int B3SOIDDqjsNode;
    int B3SOIDDqjdNode;

/* clean up last */
    int B3SOIDDgmNode;
    int B3SOIDDgmbsNode;
    int B3SOIDDgdsNode;
    int B3SOIDDgmeNode;
    int B3SOIDDqgNode;
    int B3SOIDDqdNode;
    int B3SOIDDqeNode;
    int B3SOIDDiterations;
    int B3SOIDDvbs0teffNode;
    int B3SOIDDvthNode;
    int B3SOIDDvgsteffNode;
    int B3SOIDDxcsatNode;
    int B3SOIDDqaccNode;
    int B3SOIDDqsub0Node;
    int B3SOIDDqsubs1Node;
    int B3SOIDDqsubs2Node;
    int B3SOIDDvcscvNode;
    int B3SOIDDvdscvNode;
    int B3SOIDDdum1Node;
    int B3SOIDDdum2Node;
    int B3SOIDDdum3Node;
    int B3SOIDDdum4Node;
    int B3SOIDDdum5Node;
/* end clean up last */

    double B3SOIDDphi;
    double B3SOIDDvtm;
    double B3SOIDDni;
    double B3SOIDDueff;
    double B3SOIDDthetavth; 
    double B3SOIDDvon;
    double B3SOIDDvbsdio;
    double B3SOIDDvdsat;
    double B3SOIDDcgdo;
    double B3SOIDDcgso;
    double B3SOIDDcgeo;

    double B3SOIDDids;
    double B3SOIDDic;
    double B3SOIDDibs;
    double B3SOIDDibd;
    double B3SOIDDiii;
    double B3SOIDDigidl;
    double B3SOIDDitun;
    double B3SOIDDibp;
    double B3SOIDDabeff;
    double B3SOIDDvbs0eff;
    double B3SOIDDvbseff;
    double B3SOIDDxc;
    double B3SOIDDcbg;
    double B3SOIDDcbb;
    double B3SOIDDcbd;
    double B3SOIDDqb;
    double B3SOIDDqbf;
    double B3SOIDDqjs;
    double B3SOIDDqjd;
    double B3SOIDDminIsub;
    int    B3SOIDDfloat;

/* clean up last */
    double B3SOIDDdum1;
    double B3SOIDDdum2;
    double B3SOIDDdum3;
    double B3SOIDDdum4;
    double B3SOIDDdum5;
/* end clean up last */

    double B3SOIDDl;
    double B3SOIDDw;
    double B3SOIDDm;
    double B3SOIDDdrainArea;
    double B3SOIDDsourceArea;
    double B3SOIDDdrainSquares;
    double B3SOIDDsourceSquares;
    double B3SOIDDdrainPerimeter;
    double B3SOIDDsourcePerimeter;
    double B3SOIDDsourceConductance;
    double B3SOIDDdrainConductance;

    double B3SOIDDicVBS;
    double B3SOIDDicVDS;
    double B3SOIDDicVGS;
    double B3SOIDDicVES;
    double B3SOIDDicVPS;
    int B3SOIDDbjtoff;
    int B3SOIDDbodyMod;
    int B3SOIDDdebugMod;
    double B3SOIDDrth0;
    double B3SOIDDcth0;
    double B3SOIDDbodySquares;
    double B3SOIDDrbodyext;

    double B3SOIDDcsbox;
    double B3SOIDDcdbox;
    double B3SOIDDcsmin;
    double B3SOIDDcdmin;
    double B3SOIDDst4;
    double B3SOIDDdt4;

    int B3SOIDDoff;
    int B3SOIDDmode;

    /* OP point */
    double B3SOIDDqinv;
    double B3SOIDDcd;
    double B3SOIDDcjs;
    double B3SOIDDcjd;
    double B3SOIDDcbody;
    double B3SOIDDcbodcon;
    double B3SOIDDcth;
    double B3SOIDDcsubstrate;

    double B3SOIDDgm;
    double B3SOIDDgme;
    double B3SOIDDcb;
    double B3SOIDDcdrain;
    double B3SOIDDgds;
    double B3SOIDDgmbs;
    double B3SOIDDgmT;

    double B3SOIDDgbbs;
    double B3SOIDDgbgs;
    double B3SOIDDgbds;
    double B3SOIDDgbes;
    double B3SOIDDgbps;
    double B3SOIDDgbT;

    double B3SOIDDgjsd;
    double B3SOIDDgjsb;
    double B3SOIDDgjsg;
    double B3SOIDDgjsT;

    double B3SOIDDgjdb;
    double B3SOIDDgjdd;
    double B3SOIDDgjdg;
    double B3SOIDDgjde;
    double B3SOIDDgjdT;

    double B3SOIDDgbpbs;
    double B3SOIDDgbpgs;
    double B3SOIDDgbpds;
    double B3SOIDDgbpes;
    double B3SOIDDgbpps;
    double B3SOIDDgbpT;

    double B3SOIDDgtempb;
    double B3SOIDDgtempg;
    double B3SOIDDgtempd;
    double B3SOIDDgtempe;
    double B3SOIDDgtempT;

    double B3SOIDDcggb;
    double B3SOIDDcgdb;
    double B3SOIDDcgsb;
    double B3SOIDDcgeb;
    double B3SOIDDcgT;

    double B3SOIDDcbgb;
    double B3SOIDDcbdb;
    double B3SOIDDcbsb;
    double B3SOIDDcbeb;
    double B3SOIDDcbT;

    double B3SOIDDcdgb;
    double B3SOIDDcddb;
    double B3SOIDDcdsb;
    double B3SOIDDcdeb;
    double B3SOIDDcdT;

    double B3SOIDDcegb;
    double B3SOIDDcedb;
    double B3SOIDDcesb;
    double B3SOIDDceeb;
    double B3SOIDDceT;

    double B3SOIDDqse;
    double B3SOIDDgcse;
    double B3SOIDDqde;
    double B3SOIDDgcde;

    struct b3soiddSizeDependParam  *pParam;

    unsigned B3SOIDDlGiven :1;
    unsigned B3SOIDDwGiven :1;
    unsigned B3SOIDDmGiven :1;
    unsigned B3SOIDDdrainAreaGiven :1;
    unsigned B3SOIDDsourceAreaGiven    :1;
    unsigned B3SOIDDdrainSquaresGiven  :1;
    unsigned B3SOIDDsourceSquaresGiven :1;
    unsigned B3SOIDDdrainPerimeterGiven    :1;
    unsigned B3SOIDDsourcePerimeterGiven   :1;
    unsigned B3SOIDDdNodePrimeSet  :1;
    unsigned B3SOIDDsNodePrimeSet  :1;
    unsigned B3SOIDDicVBSGiven :1;
    unsigned B3SOIDDicVDSGiven :1;
    unsigned B3SOIDDicVGSGiven :1;
    unsigned B3SOIDDicVESGiven :1;
    unsigned B3SOIDDicVPSGiven :1;
    unsigned B3SOIDDbjtoffGiven :1;
    unsigned B3SOIDDdebugModGiven :1;
    unsigned B3SOIDDrth0Given :1;
    unsigned B3SOIDDcth0Given :1;
    unsigned B3SOIDDbodySquaresGiven :1;
    unsigned B3SOIDDoffGiven :1;

    double *B3SOIDDEePtr;
    double *B3SOIDDEbPtr;
    double *B3SOIDDBePtr;
    double *B3SOIDDEgPtr;
    double *B3SOIDDEdpPtr;
    double *B3SOIDDEspPtr;
    double *B3SOIDDTemptempPtr;
    double *B3SOIDDTempdpPtr;
    double *B3SOIDDTempspPtr;
    double *B3SOIDDTempgPtr;
    double *B3SOIDDTempbPtr;
    double *B3SOIDDTempePtr;
    double *B3SOIDDGtempPtr;
    double *B3SOIDDDPtempPtr;
    double *B3SOIDDSPtempPtr;
    double *B3SOIDDEtempPtr;
    double *B3SOIDDBtempPtr;
    double *B3SOIDDPtempPtr;
    double *B3SOIDDBpPtr;
    double *B3SOIDDPbPtr;
    double *B3SOIDDPpPtr;
    double *B3SOIDDPgPtr;
    double *B3SOIDDPdpPtr;
    double *B3SOIDDPspPtr;
    double *B3SOIDDPePtr;
    double *B3SOIDDDPePtr;
    double *B3SOIDDSPePtr;
    double *B3SOIDDGePtr;
    double *B3SOIDDDdPtr;
    double *B3SOIDDGgPtr;
    double *B3SOIDDSsPtr;
    double *B3SOIDDBbPtr;
    double *B3SOIDDDPdpPtr;
    double *B3SOIDDSPspPtr;
    double *B3SOIDDDdpPtr;
    double *B3SOIDDGbPtr;
    double *B3SOIDDGdpPtr;
    double *B3SOIDDGspPtr;
    double *B3SOIDDSspPtr;
    double *B3SOIDDBdpPtr;
    double *B3SOIDDBspPtr;
    double *B3SOIDDDPspPtr;
    double *B3SOIDDDPdPtr;
    double *B3SOIDDBgPtr;
    double *B3SOIDDDPgPtr;
    double *B3SOIDDSPgPtr;
    double *B3SOIDDSPsPtr;
    double *B3SOIDDDPbPtr;
    double *B3SOIDDSPbPtr;
    double *B3SOIDDSPdpPtr;

    double *B3SOIDDVbsPtr;
    /* Debug */
    double *B3SOIDDIdsPtr;
    double *B3SOIDDIcPtr;
    double *B3SOIDDIbsPtr;
    double *B3SOIDDIbdPtr;
    double *B3SOIDDIiiPtr;
    double *B3SOIDDIgidlPtr;
    double *B3SOIDDItunPtr;
    double *B3SOIDDIbpPtr;
    double *B3SOIDDAbeffPtr;
    double *B3SOIDDVbs0effPtr;
    double *B3SOIDDVbseffPtr;
    double *B3SOIDDXcPtr;
    double *B3SOIDDCbbPtr;
    double *B3SOIDDCbdPtr;
    double *B3SOIDDCbgPtr;
    double *B3SOIDDqbPtr;
    double *B3SOIDDQbfPtr;
    double *B3SOIDDQjsPtr;
    double *B3SOIDDQjdPtr;

    /* clean up last */
    double *B3SOIDDGmPtr;
    double *B3SOIDDGmbsPtr;
    double *B3SOIDDGdsPtr;
    double *B3SOIDDGmePtr;
    double *B3SOIDDVbs0teffPtr;
    double *B3SOIDDVthPtr;
    double *B3SOIDDVgsteffPtr;
    double *B3SOIDDXcsatPtr;
    double *B3SOIDDQaccPtr;
    double *B3SOIDDQsub0Ptr;
    double *B3SOIDDQsubs1Ptr;
    double *B3SOIDDQsubs2Ptr;
    double *B3SOIDDVdscvPtr;
    double *B3SOIDDVcscvPtr;
    double *B3SOIDDCbePtr;
    double *B3SOIDDqgPtr;
    double *B3SOIDDqdPtr;
    double *B3SOIDDqePtr;
    double *B3SOIDDDum1Ptr;
    double *B3SOIDDDum2Ptr;
    double *B3SOIDDDum3Ptr;
    double *B3SOIDDDum4Ptr;
    double *B3SOIDDDum5Ptr;
    /* End clean up last */

#define B3SOIDDvbd B3SOIDDstates+ 0
#define B3SOIDDvbs B3SOIDDstates+ 1
#define B3SOIDDvgs B3SOIDDstates+ 2
#define B3SOIDDvds B3SOIDDstates+ 3
#define B3SOIDDves B3SOIDDstates+ 4
#define B3SOIDDvps B3SOIDDstates+ 5

#define B3SOIDDvg B3SOIDDstates+ 6
#define B3SOIDDvd B3SOIDDstates+ 7
#define B3SOIDDvs B3SOIDDstates+ 8
#define B3SOIDDvp B3SOIDDstates+ 9
#define B3SOIDDve B3SOIDDstates+ 10
#define B3SOIDDdeltemp B3SOIDDstates+ 11

#define B3SOIDDqb B3SOIDDstates+ 12
#define B3SOIDDcqb B3SOIDDstates+ 13
#define B3SOIDDqg B3SOIDDstates+ 14
#define B3SOIDDcqg B3SOIDDstates+ 15
#define B3SOIDDqd B3SOIDDstates+ 16
#define B3SOIDDcqd B3SOIDDstates+ 17
#define B3SOIDDqe B3SOIDDstates+ 18
#define B3SOIDDcqe B3SOIDDstates+ 19

#define B3SOIDDqbs  B3SOIDDstates+ 20
#define B3SOIDDqbd  B3SOIDDstates+ 21
#define B3SOIDDqbe  B3SOIDDstates+ 22

#define B3SOIDDqth B3SOIDDstates+ 23
#define B3SOIDDcqth B3SOIDDstates+ 24

#define B3SOIDDnumStates 25


/* indices to the array of B3SOIDD NOISE SOURCES */

#define B3SOIDDRDNOIZ       0
#define B3SOIDDRSNOIZ       1
#define B3SOIDDIDNOIZ       2
#define B3SOIDDFLNOIZ       3
#define B3SOIDDFBNOIZ       4
#define B3SOIDDTOTNOIZ      5

#define B3SOIDDNSRCS        6     /* the number of MOSFET(3) noise sources */

#ifndef NONOISE
    double B3SOIDDnVar[NSTATVARS][B3SOIDDNSRCS];
#else /* NONOISE */
        double **B3SOIDDnVar;
#endif /* NONOISE */

} B3SOIDDinstance ;

struct b3soiddSizeDependParam
{
    double Width;
    double Length;
    double Rth0;
    double Cth0;

    double B3SOIDDcdsc;           
    double B3SOIDDcdscb;    
    double B3SOIDDcdscd;       
    double B3SOIDDcit;           
    double B3SOIDDnfactor;      
    double B3SOIDDvsat;         
    double B3SOIDDat;         
    double B3SOIDDa0;   
    double B3SOIDDags;      
    double B3SOIDDa1;         
    double B3SOIDDa2;         
    double B3SOIDDketa;     
    double B3SOIDDnpeak;        
    double B3SOIDDnsub;
    double B3SOIDDngate;        
    double B3SOIDDgamma1;      
    double B3SOIDDgamma2;     
    double B3SOIDDvbx;      
    double B3SOIDDvbi;       
    double B3SOIDDvbm;       
    double B3SOIDDvbsc;       
    double B3SOIDDxt;       
    double B3SOIDDphi;
    double B3SOIDDlitl;
    double B3SOIDDk1;
    double B3SOIDDkt1;
    double B3SOIDDkt1l;
    double B3SOIDDkt2;
    double B3SOIDDk2;
    double B3SOIDDk3;
    double B3SOIDDk3b;
    double B3SOIDDw0;
    double B3SOIDDnlx;
    double B3SOIDDdvt0;      
    double B3SOIDDdvt1;      
    double B3SOIDDdvt2;      
    double B3SOIDDdvt0w;      
    double B3SOIDDdvt1w;      
    double B3SOIDDdvt2w;      
    double B3SOIDDdrout;      
    double B3SOIDDdsub;      
    double B3SOIDDvth0;
    double B3SOIDDua;
    double B3SOIDDua1;
    double B3SOIDDub;
    double B3SOIDDub1;
    double B3SOIDDuc;
    double B3SOIDDuc1;
    double B3SOIDDu0;
    double B3SOIDDute;
    double B3SOIDDvoff;
    double B3SOIDDvfb;
    double B3SOIDDuatemp;
    double B3SOIDDubtemp;
    double B3SOIDDuctemp;
    double B3SOIDDrbody;
    double B3SOIDDrth;
    double B3SOIDDcth;
    double B3SOIDDrds0denom;
    double B3SOIDDvfbb;
    double B3SOIDDjbjt;
    double B3SOIDDjdif;
    double B3SOIDDjrec;
    double B3SOIDDjtun;
    double B3SOIDDcsesw;
    double B3SOIDDcdesw;

   /* Added */
    double B3SOIDDsdt1;
    double B3SOIDDst2;
    double B3SOIDDst3;
    double B3SOIDDdt2;
    double B3SOIDDdt3;
  /* Added */

    double B3SOIDDdelta;
    double B3SOIDDrdsw;       
    double B3SOIDDrds0;       
    double B3SOIDDprwg;       
    double B3SOIDDprwb;       
    double B3SOIDDprt;       
    double B3SOIDDeta0;         
    double B3SOIDDetab;         
    double B3SOIDDpclm;      
    double B3SOIDDpdibl1;      
    double B3SOIDDpdibl2;      
    double B3SOIDDpdiblb;      
    double B3SOIDDpvag;       
    double B3SOIDDwr;
    double B3SOIDDdwg;
    double B3SOIDDdwb;
    double B3SOIDDb0;
    double B3SOIDDb1;
    double B3SOIDDalpha0;
    double B3SOIDDalpha1;
    double B3SOIDDbeta0;


    /* CV model */
    double B3SOIDDcgsl;
    double B3SOIDDcgdl;
    double B3SOIDDckappa;
    double B3SOIDDcf;
    double B3SOIDDclc;
    double B3SOIDDcle;

/* Added for binning - START0 */
    double B3SOIDDvbsa;
    double B3SOIDDdelp;
    double B3SOIDDkb1;
    double B3SOIDDkb3;
    double B3SOIDDdvbd0;
    double B3SOIDDdvbd1;
    double B3SOIDDabp;
    double B3SOIDDmxc;
    double B3SOIDDadice0;
    double B3SOIDDaii;
    double B3SOIDDbii;
    double B3SOIDDcii;
    double B3SOIDDdii;
    double B3SOIDDagidl;
    double B3SOIDDbgidl;
    double B3SOIDDngidl;
    double B3SOIDDntun;
    double B3SOIDDndiode;
    double B3SOIDDisbjt;
    double B3SOIDDisdif;
    double B3SOIDDisrec;
    double B3SOIDDistun;
    double B3SOIDDedl;
    double B3SOIDDkbjt1;
    double B3SOIDDvsdfb;
    double B3SOIDDvsdth;
/* Added for binning - END0 */

/* Pre-calculated constants */

    double B3SOIDDdw;
    double B3SOIDDdl;
    double B3SOIDDleff;
    double B3SOIDDweff;

    double B3SOIDDdwc;
    double B3SOIDDdlc;
    double B3SOIDDleffCV;
    double B3SOIDDweffCV;
    double B3SOIDDabulkCVfactor;
    double B3SOIDDcgso;
    double B3SOIDDcgdo;
    double B3SOIDDcgeo;

    double B3SOIDDu0temp;       
    double B3SOIDDvsattemp;   
    double B3SOIDDsqrtPhi;   
    double B3SOIDDphis3;   
    double B3SOIDDXdep0;          
    double B3SOIDDsqrtXdep0;          
    double B3SOIDDtheta0vb0;
    double B3SOIDDthetaRout; 

    double B3SOIDDcof1;
    double B3SOIDDcof2;
    double B3SOIDDcof3;
    double B3SOIDDcof4;
    double B3SOIDDcdep0;
    struct b3soiddSizeDependParam  *pNext;
};


typedef struct sB3SOIDDmodel 
{

    struct GENmodel gen;

#define B3SOIDDmodType gen.GENmodType
#define B3SOIDDnextModel(inst) ((struct sB3SOIDDmodel *)((inst)->gen.GENnextModel))
#define B3SOIDDinstances(inst) ((B3SOIDDinstance *)((inst)->gen.GENinstances))
#define B3SOIDDmodName gen.GENmodName

    int B3SOIDDtype;

    int    B3SOIDDmobMod;
    int    B3SOIDDcapMod;
    int    B3SOIDDnoiMod;
    int    B3SOIDDshMod;
    int    B3SOIDDbinUnit;
    int    B3SOIDDparamChk;
    double B3SOIDDversion;             
    double B3SOIDDtox;             
    double B3SOIDDcdsc;           
    double B3SOIDDcdscb; 
    double B3SOIDDcdscd;          
    double B3SOIDDcit;           
    double B3SOIDDnfactor;      
    double B3SOIDDvsat;         
    double B3SOIDDat;         
    double B3SOIDDa0;   
    double B3SOIDDags;      
    double B3SOIDDa1;         
    double B3SOIDDa2;         
    double B3SOIDDketa;     
    double B3SOIDDnsub;
    double B3SOIDDnpeak;        
    double B3SOIDDngate;        
    double B3SOIDDgamma1;      
    double B3SOIDDgamma2;     
    double B3SOIDDvbx;      
    double B3SOIDDvbm;       
    double B3SOIDDxt;       
    double B3SOIDDk1;
    double B3SOIDDkt1;
    double B3SOIDDkt1l;
    double B3SOIDDkt2;
    double B3SOIDDk2;
    double B3SOIDDk3;
    double B3SOIDDk3b;
    double B3SOIDDw0;
    double B3SOIDDnlx;
    double B3SOIDDdvt0;      
    double B3SOIDDdvt1;      
    double B3SOIDDdvt2;      
    double B3SOIDDdvt0w;      
    double B3SOIDDdvt1w;      
    double B3SOIDDdvt2w;      
    double B3SOIDDdrout;      
    double B3SOIDDdsub;      
    double B3SOIDDvth0;
    double B3SOIDDua;
    double B3SOIDDua1;
    double B3SOIDDub;
    double B3SOIDDub1;
    double B3SOIDDuc;
    double B3SOIDDuc1;
    double B3SOIDDu0;
    double B3SOIDDute;
    double B3SOIDDvoff;
    double B3SOIDDdelta;
    double B3SOIDDrdsw;       
    double B3SOIDDprwg;
    double B3SOIDDprwb;
    double B3SOIDDprt;       
    double B3SOIDDeta0;         
    double B3SOIDDetab;         
    double B3SOIDDpclm;      
    double B3SOIDDpdibl1;      
    double B3SOIDDpdibl2;      
    double B3SOIDDpdiblb;
    double B3SOIDDpvag;       
    double B3SOIDDwr;
    double B3SOIDDdwg;
    double B3SOIDDdwb;
    double B3SOIDDb0;
    double B3SOIDDb1;
    double B3SOIDDalpha0;
    double B3SOIDDalpha1;
    double B3SOIDDbeta0;
    double B3SOIDDtbox;
    double B3SOIDDtsi;
    double B3SOIDDxj;
    double B3SOIDDkb1;
    double B3SOIDDkb3;
    double B3SOIDDdvbd0;
    double B3SOIDDdvbd1;
    double B3SOIDDvbsa;
    double B3SOIDDdelp;
    double B3SOIDDrbody;
    double B3SOIDDrbsh;
    double B3SOIDDadice0;
    double B3SOIDDabp;
    double B3SOIDDmxc;
    double B3SOIDDrth0;
    double B3SOIDDcth0;
    double B3SOIDDaii;
    double B3SOIDDbii;
    double B3SOIDDcii;
    double B3SOIDDdii;
    double B3SOIDDngidl;
    double B3SOIDDagidl;
    double B3SOIDDbgidl;
    double B3SOIDDndiode;
    double B3SOIDDntun;
    double B3SOIDDisbjt;
    double B3SOIDDisdif;
    double B3SOIDDisrec;
    double B3SOIDDistun;
    double B3SOIDDxbjt;
    double B3SOIDDxdif;
    double B3SOIDDxrec;
    double B3SOIDDxtun;
    double B3SOIDDedl;
    double B3SOIDDkbjt1;
    double B3SOIDDtt;
    double B3SOIDDvsdfb;
    double B3SOIDDvsdth;
    double B3SOIDDcsdmin;
    double B3SOIDDasd;

    /* CV model */
    double B3SOIDDcgsl;
    double B3SOIDDcgdl;
    double B3SOIDDckappa;
    double B3SOIDDcf;
    double B3SOIDDclc;
    double B3SOIDDcle;
    double B3SOIDDdwc;
    double B3SOIDDdlc;

    double B3SOIDDtnom;
    double B3SOIDDcgso;
    double B3SOIDDcgdo;
    double B3SOIDDcgeo;

    double B3SOIDDxpart;
    double B3SOIDDcFringOut;
    double B3SOIDDcFringMax;

    double B3SOIDDsheetResistance;
    double B3SOIDDbodyJctGateSideGradingCoeff;
    double B3SOIDDGatesidewallJctPotential;
    double B3SOIDDunitLengthGateSidewallJctCap;
    double B3SOIDDcsdesw;

    double B3SOIDDLint;
    double B3SOIDDLl;
    double B3SOIDDLln;
    double B3SOIDDLw;
    double B3SOIDDLwn;
    double B3SOIDDLwl;
    double B3SOIDDLmin;
    double B3SOIDDLmax;

    double B3SOIDDWint;
    double B3SOIDDWl;
    double B3SOIDDWln;
    double B3SOIDDWw;
    double B3SOIDDWwn;
    double B3SOIDDWwl;
    double B3SOIDDWmin;
    double B3SOIDDWmax;

/* Added for binning - START1 */
    /* Length Dependence */
    double B3SOIDDlnpeak;        
    double B3SOIDDlnsub;
    double B3SOIDDlngate;        
    double B3SOIDDlvth0;
    double B3SOIDDlk1;
    double B3SOIDDlk2;
    double B3SOIDDlk3;
    double B3SOIDDlk3b;
    double B3SOIDDlvbsa;
    double B3SOIDDldelp;
    double B3SOIDDlkb1;
    double B3SOIDDlkb3;
    double B3SOIDDldvbd0;
    double B3SOIDDldvbd1;
    double B3SOIDDlw0;
    double B3SOIDDlnlx;
    double B3SOIDDldvt0;      
    double B3SOIDDldvt1;      
    double B3SOIDDldvt2;      
    double B3SOIDDldvt0w;      
    double B3SOIDDldvt1w;      
    double B3SOIDDldvt2w;      
    double B3SOIDDlu0;
    double B3SOIDDlua;
    double B3SOIDDlub;
    double B3SOIDDluc;
    double B3SOIDDlvsat;         
    double B3SOIDDla0;   
    double B3SOIDDlags;      
    double B3SOIDDlb0;
    double B3SOIDDlb1;
    double B3SOIDDlketa;
    double B3SOIDDlabp;
    double B3SOIDDlmxc;
    double B3SOIDDladice0;
    double B3SOIDDla1;         
    double B3SOIDDla2;         
    double B3SOIDDlrdsw;       
    double B3SOIDDlprwb;
    double B3SOIDDlprwg;
    double B3SOIDDlwr;
    double B3SOIDDlnfactor;      
    double B3SOIDDldwg;
    double B3SOIDDldwb;
    double B3SOIDDlvoff;
    double B3SOIDDleta0;         
    double B3SOIDDletab;         
    double B3SOIDDldsub;      
    double B3SOIDDlcit;           
    double B3SOIDDlcdsc;           
    double B3SOIDDlcdscb; 
    double B3SOIDDlcdscd;          
    double B3SOIDDlpclm;      
    double B3SOIDDlpdibl1;      
    double B3SOIDDlpdibl2;      
    double B3SOIDDlpdiblb;      
    double B3SOIDDldrout;      
    double B3SOIDDlpvag;       
    double B3SOIDDldelta;
    double B3SOIDDlaii;
    double B3SOIDDlbii;
    double B3SOIDDlcii;
    double B3SOIDDldii;
    double B3SOIDDlalpha0;
    double B3SOIDDlalpha1;
    double B3SOIDDlbeta0;
    double B3SOIDDlagidl;
    double B3SOIDDlbgidl;
    double B3SOIDDlngidl;
    double B3SOIDDlntun;
    double B3SOIDDlndiode;
    double B3SOIDDlisbjt;       
    double B3SOIDDlisdif;
    double B3SOIDDlisrec;       
    double B3SOIDDlistun;       
    double B3SOIDDledl;       
    double B3SOIDDlkbjt1;       
    /* CV model */
    double B3SOIDDlvsdfb;
    double B3SOIDDlvsdth;

    /* Width Dependence */
    double B3SOIDDwnpeak;        
    double B3SOIDDwnsub;
    double B3SOIDDwngate;        
    double B3SOIDDwvth0;
    double B3SOIDDwk1;
    double B3SOIDDwk2;
    double B3SOIDDwk3;
    double B3SOIDDwk3b;
    double B3SOIDDwvbsa;
    double B3SOIDDwdelp;
    double B3SOIDDwkb1;
    double B3SOIDDwkb3;
    double B3SOIDDwdvbd0;
    double B3SOIDDwdvbd1;
    double B3SOIDDww0;
    double B3SOIDDwnlx;
    double B3SOIDDwdvt0;      
    double B3SOIDDwdvt1;      
    double B3SOIDDwdvt2;      
    double B3SOIDDwdvt0w;      
    double B3SOIDDwdvt1w;      
    double B3SOIDDwdvt2w;      
    double B3SOIDDwu0;
    double B3SOIDDwua;
    double B3SOIDDwub;
    double B3SOIDDwuc;
    double B3SOIDDwvsat;         
    double B3SOIDDwa0;   
    double B3SOIDDwags;      
    double B3SOIDDwb0;
    double B3SOIDDwb1;
    double B3SOIDDwketa;
    double B3SOIDDwabp;
    double B3SOIDDwmxc;
    double B3SOIDDwadice0;
    double B3SOIDDwa1;         
    double B3SOIDDwa2;         
    double B3SOIDDwrdsw;       
    double B3SOIDDwprwb;
    double B3SOIDDwprwg;
    double B3SOIDDwwr;
    double B3SOIDDwnfactor;      
    double B3SOIDDwdwg;
    double B3SOIDDwdwb;
    double B3SOIDDwvoff;
    double B3SOIDDweta0;         
    double B3SOIDDwetab;         
    double B3SOIDDwdsub;      
    double B3SOIDDwcit;           
    double B3SOIDDwcdsc;           
    double B3SOIDDwcdscb; 
    double B3SOIDDwcdscd;          
    double B3SOIDDwpclm;      
    double B3SOIDDwpdibl1;      
    double B3SOIDDwpdibl2;      
    double B3SOIDDwpdiblb;      
    double B3SOIDDwdrout;      
    double B3SOIDDwpvag;       
    double B3SOIDDwdelta;
    double B3SOIDDwaii;
    double B3SOIDDwbii;
    double B3SOIDDwcii;
    double B3SOIDDwdii;
    double B3SOIDDwalpha0;
    double B3SOIDDwalpha1;
    double B3SOIDDwbeta0;
    double B3SOIDDwagidl;
    double B3SOIDDwbgidl;
    double B3SOIDDwngidl;
    double B3SOIDDwntun;
    double B3SOIDDwndiode;
    double B3SOIDDwisbjt;       
    double B3SOIDDwisdif;
    double B3SOIDDwisrec;       
    double B3SOIDDwistun;       
    double B3SOIDDwedl;       
    double B3SOIDDwkbjt1;       
    /* CV model */
    double B3SOIDDwvsdfb;
    double B3SOIDDwvsdth;

    /* Cross-term Dependence */
    double B3SOIDDpnpeak;        
    double B3SOIDDpnsub;
    double B3SOIDDpngate;        
    double B3SOIDDpvth0;
    double B3SOIDDpk1;
    double B3SOIDDpk2;
    double B3SOIDDpk3;
    double B3SOIDDpk3b;
    double B3SOIDDpvbsa;
    double B3SOIDDpdelp;
    double B3SOIDDpkb1;
    double B3SOIDDpkb3;
    double B3SOIDDpdvbd0;
    double B3SOIDDpdvbd1;
    double B3SOIDDpw0;
    double B3SOIDDpnlx;
    double B3SOIDDpdvt0;      
    double B3SOIDDpdvt1;      
    double B3SOIDDpdvt2;      
    double B3SOIDDpdvt0w;      
    double B3SOIDDpdvt1w;      
    double B3SOIDDpdvt2w;      
    double B3SOIDDpu0;
    double B3SOIDDpua;
    double B3SOIDDpub;
    double B3SOIDDpuc;
    double B3SOIDDpvsat;         
    double B3SOIDDpa0;   
    double B3SOIDDpags;      
    double B3SOIDDpb0;
    double B3SOIDDpb1;
    double B3SOIDDpketa;
    double B3SOIDDpabp;
    double B3SOIDDpmxc;
    double B3SOIDDpadice0;
    double B3SOIDDpa1;         
    double B3SOIDDpa2;         
    double B3SOIDDprdsw;       
    double B3SOIDDpprwb;
    double B3SOIDDpprwg;
    double B3SOIDDpwr;
    double B3SOIDDpnfactor;      
    double B3SOIDDpdwg;
    double B3SOIDDpdwb;
    double B3SOIDDpvoff;
    double B3SOIDDpeta0;         
    double B3SOIDDpetab;         
    double B3SOIDDpdsub;      
    double B3SOIDDpcit;           
    double B3SOIDDpcdsc;           
    double B3SOIDDpcdscb; 
    double B3SOIDDpcdscd;          
    double B3SOIDDppclm;      
    double B3SOIDDppdibl1;      
    double B3SOIDDppdibl2;      
    double B3SOIDDppdiblb;      
    double B3SOIDDpdrout;      
    double B3SOIDDppvag;       
    double B3SOIDDpdelta;
    double B3SOIDDpaii;
    double B3SOIDDpbii;
    double B3SOIDDpcii;
    double B3SOIDDpdii;
    double B3SOIDDpalpha0;
    double B3SOIDDpalpha1;
    double B3SOIDDpbeta0;
    double B3SOIDDpagidl;
    double B3SOIDDpbgidl;
    double B3SOIDDpngidl;
    double B3SOIDDpntun;
    double B3SOIDDpndiode;
    double B3SOIDDpisbjt;       
    double B3SOIDDpisdif;
    double B3SOIDDpisrec;       
    double B3SOIDDpistun;       
    double B3SOIDDpedl;       
    double B3SOIDDpkbjt1;       
    /* CV model */
    double B3SOIDDpvsdfb;
    double B3SOIDDpvsdth;
/* Added for binning - END1 */

/* Pre-calculated constants */
    double B3SOIDDcbox;
    double B3SOIDDcsi;
    double B3SOIDDcsieff;
    double B3SOIDDcoxt;
    double B3SOIDDcboxt;
    double B3SOIDDcsit;
    double B3SOIDDnfb;
    double B3SOIDDadice;
    double B3SOIDDqsi;
    double B3SOIDDqsieff;
    double B3SOIDDeg0;

    /* MCJ: move to size-dependent param. */
    double B3SOIDDvtm;   
    double B3SOIDDcox;
    double B3SOIDDcof1;
    double B3SOIDDcof2;
    double B3SOIDDcof3;
    double B3SOIDDcof4;
    double B3SOIDDvcrit;
    double B3SOIDDfactor1;

    double B3SOIDDoxideTrapDensityA;      
    double B3SOIDDoxideTrapDensityB;     
    double B3SOIDDoxideTrapDensityC;  
    double B3SOIDDem;  
    double B3SOIDDef;  
    double B3SOIDDaf;  
    double B3SOIDDkf;  
    double B3SOIDDnoif;  

    struct b3soiddSizeDependParam *pSizeDependParamKnot;

    /* Flags */

    unsigned B3SOIDDtboxGiven:1;
    unsigned B3SOIDDtsiGiven :1;
    unsigned B3SOIDDxjGiven :1;
    unsigned B3SOIDDkb1Given :1;
    unsigned B3SOIDDkb3Given :1;
    unsigned B3SOIDDdvbd0Given :1;
    unsigned B3SOIDDdvbd1Given :1;
    unsigned B3SOIDDvbsaGiven :1;
    unsigned B3SOIDDdelpGiven :1;
    unsigned B3SOIDDrbodyGiven :1;
    unsigned B3SOIDDrbshGiven :1;
    unsigned B3SOIDDadice0Given :1;
    unsigned B3SOIDDabpGiven :1;
    unsigned B3SOIDDmxcGiven :1;
    unsigned B3SOIDDrth0Given :1;
    unsigned B3SOIDDcth0Given :1;
    unsigned B3SOIDDaiiGiven :1;
    unsigned B3SOIDDbiiGiven :1;
    unsigned B3SOIDDciiGiven :1;
    unsigned B3SOIDDdiiGiven :1;
    unsigned B3SOIDDngidlGiven :1;
    unsigned B3SOIDDagidlGiven :1;
    unsigned B3SOIDDbgidlGiven :1;
    unsigned B3SOIDDndiodeGiven :1;
    unsigned B3SOIDDntunGiven :1;
    unsigned B3SOIDDisbjtGiven :1;
    unsigned B3SOIDDisdifGiven :1;
    unsigned B3SOIDDisrecGiven :1;
    unsigned B3SOIDDistunGiven :1;
    unsigned B3SOIDDxbjtGiven :1;
    unsigned B3SOIDDxdifGiven :1;
    unsigned B3SOIDDxrecGiven :1;
    unsigned B3SOIDDxtunGiven :1;
    unsigned B3SOIDDedlGiven :1;
    unsigned B3SOIDDkbjt1Given :1;
    unsigned B3SOIDDttGiven :1;
    unsigned B3SOIDDvsdfbGiven :1;
    unsigned B3SOIDDvsdthGiven :1;
    unsigned B3SOIDDasdGiven :1;
    unsigned B3SOIDDcsdminGiven :1;

    unsigned  B3SOIDDmobModGiven :1;
    unsigned  B3SOIDDbinUnitGiven :1;
    unsigned  B3SOIDDcapModGiven :1;
    unsigned  B3SOIDDparamChkGiven :1;
    unsigned  B3SOIDDnoiModGiven :1;
    unsigned  B3SOIDDshModGiven :1;
    unsigned  B3SOIDDtypeGiven   :1;
    unsigned  B3SOIDDtoxGiven   :1;
    unsigned  B3SOIDDversionGiven   :1;

    unsigned  B3SOIDDcdscGiven   :1;
    unsigned  B3SOIDDcdscbGiven   :1;
    unsigned  B3SOIDDcdscdGiven   :1;
    unsigned  B3SOIDDcitGiven   :1;
    unsigned  B3SOIDDnfactorGiven   :1;
    unsigned  B3SOIDDvsatGiven   :1;
    unsigned  B3SOIDDatGiven   :1;
    unsigned  B3SOIDDa0Given   :1;
    unsigned  B3SOIDDagsGiven   :1;
    unsigned  B3SOIDDa1Given   :1;
    unsigned  B3SOIDDa2Given   :1;
    unsigned  B3SOIDDketaGiven   :1;    
    unsigned  B3SOIDDnsubGiven   :1;
    unsigned  B3SOIDDnpeakGiven   :1;
    unsigned  B3SOIDDngateGiven   :1;
    unsigned  B3SOIDDgamma1Given   :1;
    unsigned  B3SOIDDgamma2Given   :1;
    unsigned  B3SOIDDvbxGiven   :1;
    unsigned  B3SOIDDvbmGiven   :1;
    unsigned  B3SOIDDxtGiven   :1;
    unsigned  B3SOIDDk1Given   :1;
    unsigned  B3SOIDDkt1Given   :1;
    unsigned  B3SOIDDkt1lGiven   :1;
    unsigned  B3SOIDDkt2Given   :1;
    unsigned  B3SOIDDk2Given   :1;
    unsigned  B3SOIDDk3Given   :1;
    unsigned  B3SOIDDk3bGiven   :1;
    unsigned  B3SOIDDw0Given   :1;
    unsigned  B3SOIDDnlxGiven   :1;
    unsigned  B3SOIDDdvt0Given   :1;   
    unsigned  B3SOIDDdvt1Given   :1;     
    unsigned  B3SOIDDdvt2Given   :1;     
    unsigned  B3SOIDDdvt0wGiven   :1;   
    unsigned  B3SOIDDdvt1wGiven   :1;     
    unsigned  B3SOIDDdvt2wGiven   :1;     
    unsigned  B3SOIDDdroutGiven   :1;     
    unsigned  B3SOIDDdsubGiven   :1;     
    unsigned  B3SOIDDvth0Given   :1;
    unsigned  B3SOIDDuaGiven   :1;
    unsigned  B3SOIDDua1Given   :1;
    unsigned  B3SOIDDubGiven   :1;
    unsigned  B3SOIDDub1Given   :1;
    unsigned  B3SOIDDucGiven   :1;
    unsigned  B3SOIDDuc1Given   :1;
    unsigned  B3SOIDDu0Given   :1;
    unsigned  B3SOIDDuteGiven   :1;
    unsigned  B3SOIDDvoffGiven   :1;
    unsigned  B3SOIDDrdswGiven   :1;      
    unsigned  B3SOIDDprwgGiven   :1;      
    unsigned  B3SOIDDprwbGiven   :1;      
    unsigned  B3SOIDDprtGiven   :1;      
    unsigned  B3SOIDDeta0Given   :1;    
    unsigned  B3SOIDDetabGiven   :1;    
    unsigned  B3SOIDDpclmGiven   :1;   
    unsigned  B3SOIDDpdibl1Given   :1;   
    unsigned  B3SOIDDpdibl2Given   :1;  
    unsigned  B3SOIDDpdiblbGiven   :1;  
    unsigned  B3SOIDDpvagGiven   :1;    
    unsigned  B3SOIDDdeltaGiven  :1;     
    unsigned  B3SOIDDwrGiven   :1;
    unsigned  B3SOIDDdwgGiven   :1;
    unsigned  B3SOIDDdwbGiven   :1;
    unsigned  B3SOIDDb0Given   :1;
    unsigned  B3SOIDDb1Given   :1;
    unsigned  B3SOIDDalpha0Given   :1;
    unsigned  B3SOIDDalpha1Given   :1;
    unsigned  B3SOIDDbeta0Given   :1;

    /* CV model */
    unsigned  B3SOIDDcgslGiven   :1;
    unsigned  B3SOIDDcgdlGiven   :1;
    unsigned  B3SOIDDckappaGiven   :1;
    unsigned  B3SOIDDcfGiven   :1;
    unsigned  B3SOIDDclcGiven   :1;
    unsigned  B3SOIDDcleGiven   :1;
    unsigned  B3SOIDDdwcGiven   :1;
    unsigned  B3SOIDDdlcGiven   :1;

/* Added for binning - START2 */
    /* Length Dependence */
    unsigned  B3SOIDDlnpeakGiven   :1;        
    unsigned  B3SOIDDlnsubGiven   :1;
    unsigned  B3SOIDDlngateGiven   :1;        
    unsigned  B3SOIDDlvth0Given   :1;
    unsigned  B3SOIDDlk1Given   :1;
    unsigned  B3SOIDDlk2Given   :1;
    unsigned  B3SOIDDlk3Given   :1;
    unsigned  B3SOIDDlk3bGiven   :1;
    unsigned  B3SOIDDlvbsaGiven   :1;
    unsigned  B3SOIDDldelpGiven   :1;
    unsigned  B3SOIDDlkb1Given   :1;
    unsigned  B3SOIDDlkb3Given   :1;
    unsigned  B3SOIDDldvbd0Given   :1;
    unsigned  B3SOIDDldvbd1Given   :1;
    unsigned  B3SOIDDlw0Given   :1;
    unsigned  B3SOIDDlnlxGiven   :1;
    unsigned  B3SOIDDldvt0Given   :1;      
    unsigned  B3SOIDDldvt1Given   :1;      
    unsigned  B3SOIDDldvt2Given   :1;      
    unsigned  B3SOIDDldvt0wGiven   :1;      
    unsigned  B3SOIDDldvt1wGiven   :1;      
    unsigned  B3SOIDDldvt2wGiven   :1;      
    unsigned  B3SOIDDlu0Given   :1;
    unsigned  B3SOIDDluaGiven   :1;
    unsigned  B3SOIDDlubGiven   :1;
    unsigned  B3SOIDDlucGiven   :1;
    unsigned  B3SOIDDlvsatGiven   :1;         
    unsigned  B3SOIDDla0Given   :1;   
    unsigned  B3SOIDDlagsGiven   :1;      
    unsigned  B3SOIDDlb0Given   :1;
    unsigned  B3SOIDDlb1Given   :1;
    unsigned  B3SOIDDlketaGiven   :1;
    unsigned  B3SOIDDlabpGiven   :1;
    unsigned  B3SOIDDlmxcGiven   :1;
    unsigned  B3SOIDDladice0Given   :1;
    unsigned  B3SOIDDla1Given   :1;         
    unsigned  B3SOIDDla2Given   :1;         
    unsigned  B3SOIDDlrdswGiven   :1;       
    unsigned  B3SOIDDlprwbGiven   :1;
    unsigned  B3SOIDDlprwgGiven   :1;
    unsigned  B3SOIDDlwrGiven   :1;
    unsigned  B3SOIDDlnfactorGiven   :1;      
    unsigned  B3SOIDDldwgGiven   :1;
    unsigned  B3SOIDDldwbGiven   :1;
    unsigned  B3SOIDDlvoffGiven   :1;
    unsigned  B3SOIDDleta0Given   :1;         
    unsigned  B3SOIDDletabGiven   :1;         
    unsigned  B3SOIDDldsubGiven   :1;      
    unsigned  B3SOIDDlcitGiven   :1;           
    unsigned  B3SOIDDlcdscGiven   :1;           
    unsigned  B3SOIDDlcdscbGiven   :1; 
    unsigned  B3SOIDDlcdscdGiven   :1;          
    unsigned  B3SOIDDlpclmGiven   :1;      
    unsigned  B3SOIDDlpdibl1Given   :1;      
    unsigned  B3SOIDDlpdibl2Given   :1;      
    unsigned  B3SOIDDlpdiblbGiven   :1;      
    unsigned  B3SOIDDldroutGiven   :1;      
    unsigned  B3SOIDDlpvagGiven   :1;       
    unsigned  B3SOIDDldeltaGiven   :1;
    unsigned  B3SOIDDlaiiGiven   :1;
    unsigned  B3SOIDDlbiiGiven   :1;
    unsigned  B3SOIDDlciiGiven   :1;
    unsigned  B3SOIDDldiiGiven   :1;
    unsigned  B3SOIDDlalpha0Given   :1;
    unsigned  B3SOIDDlalpha1Given   :1;
    unsigned  B3SOIDDlbeta0Given   :1;
    unsigned  B3SOIDDlagidlGiven   :1;
    unsigned  B3SOIDDlbgidlGiven   :1;
    unsigned  B3SOIDDlngidlGiven   :1;
    unsigned  B3SOIDDlntunGiven   :1;
    unsigned  B3SOIDDlndiodeGiven   :1;
    unsigned  B3SOIDDlisbjtGiven   :1;       
    unsigned  B3SOIDDlisdifGiven   :1;
    unsigned  B3SOIDDlisrecGiven   :1;       
    unsigned  B3SOIDDlistunGiven   :1;       
    unsigned  B3SOIDDledlGiven   :1;       
    unsigned  B3SOIDDlkbjt1Given   :1;       
    /* CV model */
    unsigned  B3SOIDDlvsdfbGiven   :1;
    unsigned  B3SOIDDlvsdthGiven   :1;

    /* Width Dependence */
    unsigned  B3SOIDDwnpeakGiven   :1;        
    unsigned  B3SOIDDwnsubGiven   :1;
    unsigned  B3SOIDDwngateGiven   :1;        
    unsigned  B3SOIDDwvth0Given   :1;
    unsigned  B3SOIDDwk1Given   :1;
    unsigned  B3SOIDDwk2Given   :1;
    unsigned  B3SOIDDwk3Given   :1;
    unsigned  B3SOIDDwk3bGiven   :1;
    unsigned  B3SOIDDwvbsaGiven   :1;
    unsigned  B3SOIDDwdelpGiven   :1;
    unsigned  B3SOIDDwkb1Given   :1;
    unsigned  B3SOIDDwkb3Given   :1;
    unsigned  B3SOIDDwdvbd0Given   :1;
    unsigned  B3SOIDDwdvbd1Given   :1;
    unsigned  B3SOIDDww0Given   :1;
    unsigned  B3SOIDDwnlxGiven   :1;
    unsigned  B3SOIDDwdvt0Given   :1;      
    unsigned  B3SOIDDwdvt1Given   :1;      
    unsigned  B3SOIDDwdvt2Given   :1;      
    unsigned  B3SOIDDwdvt0wGiven   :1;      
    unsigned  B3SOIDDwdvt1wGiven   :1;      
    unsigned  B3SOIDDwdvt2wGiven   :1;      
    unsigned  B3SOIDDwu0Given   :1;
    unsigned  B3SOIDDwuaGiven   :1;
    unsigned  B3SOIDDwubGiven   :1;
    unsigned  B3SOIDDwucGiven   :1;
    unsigned  B3SOIDDwvsatGiven   :1;         
    unsigned  B3SOIDDwa0Given   :1;   
    unsigned  B3SOIDDwagsGiven   :1;      
    unsigned  B3SOIDDwb0Given   :1;
    unsigned  B3SOIDDwb1Given   :1;
    unsigned  B3SOIDDwketaGiven   :1;
    unsigned  B3SOIDDwabpGiven   :1;
    unsigned  B3SOIDDwmxcGiven   :1;
    unsigned  B3SOIDDwadice0Given   :1;
    unsigned  B3SOIDDwa1Given   :1;         
    unsigned  B3SOIDDwa2Given   :1;         
    unsigned  B3SOIDDwrdswGiven   :1;       
    unsigned  B3SOIDDwprwbGiven   :1;
    unsigned  B3SOIDDwprwgGiven   :1;
    unsigned  B3SOIDDwwrGiven   :1;
    unsigned  B3SOIDDwnfactorGiven   :1;      
    unsigned  B3SOIDDwdwgGiven   :1;
    unsigned  B3SOIDDwdwbGiven   :1;
    unsigned  B3SOIDDwvoffGiven   :1;
    unsigned  B3SOIDDweta0Given   :1;         
    unsigned  B3SOIDDwetabGiven   :1;         
    unsigned  B3SOIDDwdsubGiven   :1;      
    unsigned  B3SOIDDwcitGiven   :1;           
    unsigned  B3SOIDDwcdscGiven   :1;           
    unsigned  B3SOIDDwcdscbGiven   :1; 
    unsigned  B3SOIDDwcdscdGiven   :1;          
    unsigned  B3SOIDDwpclmGiven   :1;      
    unsigned  B3SOIDDwpdibl1Given   :1;      
    unsigned  B3SOIDDwpdibl2Given   :1;      
    unsigned  B3SOIDDwpdiblbGiven   :1;      
    unsigned  B3SOIDDwdroutGiven   :1;      
    unsigned  B3SOIDDwpvagGiven   :1;       
    unsigned  B3SOIDDwdeltaGiven   :1;
    unsigned  B3SOIDDwaiiGiven   :1;
    unsigned  B3SOIDDwbiiGiven   :1;
    unsigned  B3SOIDDwciiGiven   :1;
    unsigned  B3SOIDDwdiiGiven   :1;
    unsigned  B3SOIDDwalpha0Given   :1;
    unsigned  B3SOIDDwalpha1Given   :1;
    unsigned  B3SOIDDwbeta0Given   :1;
    unsigned  B3SOIDDwagidlGiven   :1;
    unsigned  B3SOIDDwbgidlGiven   :1;
    unsigned  B3SOIDDwngidlGiven   :1;
    unsigned  B3SOIDDwntunGiven   :1;
    unsigned  B3SOIDDwndiodeGiven   :1;
    unsigned  B3SOIDDwisbjtGiven   :1;       
    unsigned  B3SOIDDwisdifGiven   :1;
    unsigned  B3SOIDDwisrecGiven   :1;       
    unsigned  B3SOIDDwistunGiven   :1;       
    unsigned  B3SOIDDwedlGiven   :1;       
    unsigned  B3SOIDDwkbjt1Given   :1;       
    /* CV model */
    unsigned  B3SOIDDwvsdfbGiven   :1;
    unsigned  B3SOIDDwvsdthGiven   :1;

    /* Cross-term Dependence */
    unsigned  B3SOIDDpnpeakGiven   :1;        
    unsigned  B3SOIDDpnsubGiven   :1;
    unsigned  B3SOIDDpngateGiven   :1;        
    unsigned  B3SOIDDpvth0Given   :1;
    unsigned  B3SOIDDpk1Given   :1;
    unsigned  B3SOIDDpk2Given   :1;
    unsigned  B3SOIDDpk3Given   :1;
    unsigned  B3SOIDDpk3bGiven   :1;
    unsigned  B3SOIDDpvbsaGiven   :1;
    unsigned  B3SOIDDpdelpGiven   :1;
    unsigned  B3SOIDDpkb1Given   :1;
    unsigned  B3SOIDDpkb3Given   :1;
    unsigned  B3SOIDDpdvbd0Given   :1;
    unsigned  B3SOIDDpdvbd1Given   :1;
    unsigned  B3SOIDDpw0Given   :1;
    unsigned  B3SOIDDpnlxGiven   :1;
    unsigned  B3SOIDDpdvt0Given   :1;      
    unsigned  B3SOIDDpdvt1Given   :1;      
    unsigned  B3SOIDDpdvt2Given   :1;      
    unsigned  B3SOIDDpdvt0wGiven   :1;      
    unsigned  B3SOIDDpdvt1wGiven   :1;      
    unsigned  B3SOIDDpdvt2wGiven   :1;      
    unsigned  B3SOIDDpu0Given   :1;
    unsigned  B3SOIDDpuaGiven   :1;
    unsigned  B3SOIDDpubGiven   :1;
    unsigned  B3SOIDDpucGiven   :1;
    unsigned  B3SOIDDpvsatGiven   :1;         
    unsigned  B3SOIDDpa0Given   :1;   
    unsigned  B3SOIDDpagsGiven   :1;      
    unsigned  B3SOIDDpb0Given   :1;
    unsigned  B3SOIDDpb1Given   :1;
    unsigned  B3SOIDDpketaGiven   :1;
    unsigned  B3SOIDDpabpGiven   :1;
    unsigned  B3SOIDDpmxcGiven   :1;
    unsigned  B3SOIDDpadice0Given   :1;
    unsigned  B3SOIDDpa1Given   :1;         
    unsigned  B3SOIDDpa2Given   :1;         
    unsigned  B3SOIDDprdswGiven   :1;       
    unsigned  B3SOIDDpprwbGiven   :1;
    unsigned  B3SOIDDpprwgGiven   :1;
    unsigned  B3SOIDDpwrGiven   :1;
    unsigned  B3SOIDDpnfactorGiven   :1;      
    unsigned  B3SOIDDpdwgGiven   :1;
    unsigned  B3SOIDDpdwbGiven   :1;
    unsigned  B3SOIDDpvoffGiven   :1;
    unsigned  B3SOIDDpeta0Given   :1;         
    unsigned  B3SOIDDpetabGiven   :1;         
    unsigned  B3SOIDDpdsubGiven   :1;      
    unsigned  B3SOIDDpcitGiven   :1;           
    unsigned  B3SOIDDpcdscGiven   :1;           
    unsigned  B3SOIDDpcdscbGiven   :1; 
    unsigned  B3SOIDDpcdscdGiven   :1;          
    unsigned  B3SOIDDppclmGiven   :1;      
    unsigned  B3SOIDDppdibl1Given   :1;      
    unsigned  B3SOIDDppdibl2Given   :1;      
    unsigned  B3SOIDDppdiblbGiven   :1;      
    unsigned  B3SOIDDpdroutGiven   :1;      
    unsigned  B3SOIDDppvagGiven   :1;       
    unsigned  B3SOIDDpdeltaGiven   :1;
    unsigned  B3SOIDDpaiiGiven   :1;
    unsigned  B3SOIDDpbiiGiven   :1;
    unsigned  B3SOIDDpciiGiven   :1;
    unsigned  B3SOIDDpdiiGiven   :1;
    unsigned  B3SOIDDpalpha0Given   :1;
    unsigned  B3SOIDDpalpha1Given   :1;
    unsigned  B3SOIDDpbeta0Given   :1;
    unsigned  B3SOIDDpagidlGiven   :1;
    unsigned  B3SOIDDpbgidlGiven   :1;
    unsigned  B3SOIDDpngidlGiven   :1;
    unsigned  B3SOIDDpntunGiven   :1;
    unsigned  B3SOIDDpndiodeGiven   :1;
    unsigned  B3SOIDDpisbjtGiven   :1;       
    unsigned  B3SOIDDpisdifGiven   :1;
    unsigned  B3SOIDDpisrecGiven   :1;       
    unsigned  B3SOIDDpistunGiven   :1;       
    unsigned  B3SOIDDpedlGiven   :1;       
    unsigned  B3SOIDDpkbjt1Given   :1;       
    /* CV model */
    unsigned  B3SOIDDpvsdfbGiven   :1;
    unsigned  B3SOIDDpvsdthGiven   :1;
/* Added for binning - END2 */

    unsigned  B3SOIDDuseFringeGiven   :1;

    unsigned  B3SOIDDtnomGiven   :1;
    unsigned  B3SOIDDcgsoGiven   :1;
    unsigned  B3SOIDDcgdoGiven   :1;
    unsigned  B3SOIDDcgeoGiven   :1;
    unsigned  B3SOIDDxpartGiven   :1;
    unsigned  B3SOIDDsheetResistanceGiven   :1;
    unsigned  B3SOIDDGatesidewallJctPotentialGiven   :1;
    unsigned  B3SOIDDbodyJctGateSideGradingCoeffGiven   :1;
    unsigned  B3SOIDDunitLengthGateSidewallJctCapGiven   :1;
    unsigned  B3SOIDDcsdeswGiven :1;

    unsigned  B3SOIDDoxideTrapDensityAGiven  :1;         
    unsigned  B3SOIDDoxideTrapDensityBGiven  :1;        
    unsigned  B3SOIDDoxideTrapDensityCGiven  :1;     
    unsigned  B3SOIDDemGiven  :1;     
    unsigned  B3SOIDDefGiven  :1;     
    unsigned  B3SOIDDafGiven  :1;     
    unsigned  B3SOIDDkfGiven  :1;     
    unsigned  B3SOIDDnoifGiven  :1;     

    unsigned  B3SOIDDLintGiven   :1;
    unsigned  B3SOIDDLlGiven   :1;
    unsigned  B3SOIDDLlnGiven   :1;
    unsigned  B3SOIDDLwGiven   :1;
    unsigned  B3SOIDDLwnGiven   :1;
    unsigned  B3SOIDDLwlGiven   :1;
    unsigned  B3SOIDDLminGiven   :1;
    unsigned  B3SOIDDLmaxGiven   :1;

    unsigned  B3SOIDDWintGiven   :1;
    unsigned  B3SOIDDWlGiven   :1;
    unsigned  B3SOIDDWlnGiven   :1;
    unsigned  B3SOIDDWwGiven   :1;
    unsigned  B3SOIDDWwnGiven   :1;
    unsigned  B3SOIDDWwlGiven   :1;
    unsigned  B3SOIDDWminGiven   :1;
    unsigned  B3SOIDDWmaxGiven   :1;

} B3SOIDDmodel;


#ifndef NMOS
#define NMOS 1
#define PMOS -1
#endif /*NMOS*/


/* device parameters */
#define B3SOIDD_W 1
#define B3SOIDD_L 2
#define B3SOIDD_M 22
#define B3SOIDD_AS 3
#define B3SOIDD_AD 4
#define B3SOIDD_PS 5
#define B3SOIDD_PD 6
#define B3SOIDD_NRS 7
#define B3SOIDD_NRD 8
#define B3SOIDD_OFF 9
#define B3SOIDD_IC_VBS 10
#define B3SOIDD_IC_VDS 11
#define B3SOIDD_IC_VGS 12
#define B3SOIDD_IC_VES 13
#define B3SOIDD_IC_VPS 14
#define B3SOIDD_BJTOFF 15
#define B3SOIDD_RTH0 16
#define B3SOIDD_CTH0 17
#define B3SOIDD_NRB 18
#define B3SOIDD_IC 19
#define B3SOIDD_NQSMOD 20
#define B3SOIDD_DEBUG  21

/* model parameters */
#define B3SOIDD_MOD_CAPMOD          101
#define B3SOIDD_MOD_NQSMOD          102
#define B3SOIDD_MOD_MOBMOD          103    
#define B3SOIDD_MOD_NOIMOD          104    
#define B3SOIDD_MOD_SHMOD           105
#define B3SOIDD_MOD_DDMOD           106

#define B3SOIDD_MOD_TOX             107



#define B3SOIDD_MOD_CDSC            108
#define B3SOIDD_MOD_CDSCB           109
#define B3SOIDD_MOD_CIT             110
#define B3SOIDD_MOD_NFACTOR         111
#define B3SOIDD_MOD_XJ              112
#define B3SOIDD_MOD_VSAT            113
#define B3SOIDD_MOD_AT              114
#define B3SOIDD_MOD_A0              115
#define B3SOIDD_MOD_A1              116
#define B3SOIDD_MOD_A2              117
#define B3SOIDD_MOD_KETA            118   
#define B3SOIDD_MOD_NSUB            119
#define B3SOIDD_MOD_NPEAK           120
#define B3SOIDD_MOD_NGATE           121
#define B3SOIDD_MOD_GAMMA1          122
#define B3SOIDD_MOD_GAMMA2          123
#define B3SOIDD_MOD_VBX             124
#define B3SOIDD_MOD_BINUNIT         125    

#define B3SOIDD_MOD_VBM             126

#define B3SOIDD_MOD_XT              127
#define B3SOIDD_MOD_K1              129
#define B3SOIDD_MOD_KT1             130
#define B3SOIDD_MOD_KT1L            131
#define B3SOIDD_MOD_K2              132
#define B3SOIDD_MOD_KT2             133
#define B3SOIDD_MOD_K3              134
#define B3SOIDD_MOD_K3B             135
#define B3SOIDD_MOD_W0              136
#define B3SOIDD_MOD_NLX             137

#define B3SOIDD_MOD_DVT0            138
#define B3SOIDD_MOD_DVT1            139
#define B3SOIDD_MOD_DVT2            140

#define B3SOIDD_MOD_DVT0W           141
#define B3SOIDD_MOD_DVT1W           142
#define B3SOIDD_MOD_DVT2W           143

#define B3SOIDD_MOD_DROUT           144
#define B3SOIDD_MOD_DSUB            145
#define B3SOIDD_MOD_VTH0            146
#define B3SOIDD_MOD_UA              147
#define B3SOIDD_MOD_UA1             148
#define B3SOIDD_MOD_UB              149
#define B3SOIDD_MOD_UB1             150
#define B3SOIDD_MOD_UC              151
#define B3SOIDD_MOD_UC1             152
#define B3SOIDD_MOD_U0              153
#define B3SOIDD_MOD_UTE             154
#define B3SOIDD_MOD_VOFF            155
#define B3SOIDD_MOD_DELTA           156
#define B3SOIDD_MOD_RDSW            157
#define B3SOIDD_MOD_PRT             158
#define B3SOIDD_MOD_LDD             159
#define B3SOIDD_MOD_ETA             160
#define B3SOIDD_MOD_ETA0            161
#define B3SOIDD_MOD_ETAB            162
#define B3SOIDD_MOD_PCLM            163
#define B3SOIDD_MOD_PDIBL1          164
#define B3SOIDD_MOD_PDIBL2          165
#define B3SOIDD_MOD_PSCBE1          166
#define B3SOIDD_MOD_PSCBE2          167
#define B3SOIDD_MOD_PVAG            168
#define B3SOIDD_MOD_WR              169
#define B3SOIDD_MOD_DWG             170
#define B3SOIDD_MOD_DWB             171
#define B3SOIDD_MOD_B0              172
#define B3SOIDD_MOD_B1              173
#define B3SOIDD_MOD_ALPHA0          174
#define B3SOIDD_MOD_BETA0           175
#define B3SOIDD_MOD_PDIBLB          178

#define B3SOIDD_MOD_PRWG            179
#define B3SOIDD_MOD_PRWB            180

#define B3SOIDD_MOD_CDSCD           181
#define B3SOIDD_MOD_AGS             182

#define B3SOIDD_MOD_FRINGE          184
#define B3SOIDD_MOD_CGSL            186
#define B3SOIDD_MOD_CGDL            187
#define B3SOIDD_MOD_CKAPPA          188
#define B3SOIDD_MOD_CF              189
#define B3SOIDD_MOD_CLC             190
#define B3SOIDD_MOD_CLE             191
#define B3SOIDD_MOD_PARAMCHK        192
#define B3SOIDD_MOD_VERSION         193

#define B3SOIDD_MOD_TBOX            195
#define B3SOIDD_MOD_TSI             196
#define B3SOIDD_MOD_KB1             197
#define B3SOIDD_MOD_KB3             198
#define B3SOIDD_MOD_DVBD0           199
#define B3SOIDD_MOD_DVBD1           200
#define B3SOIDD_MOD_DELP            201
#define B3SOIDD_MOD_VBSA            202
#define B3SOIDD_MOD_RBODY           204
#define B3SOIDD_MOD_ADICE0          205
#define B3SOIDD_MOD_ABP             206
#define B3SOIDD_MOD_MXC             207
#define B3SOIDD_MOD_RTH0            208
#define B3SOIDD_MOD_CTH0            209
#define B3SOIDD_MOD_AII             210
#define B3SOIDD_MOD_BII             211
#define B3SOIDD_MOD_CII             212
#define B3SOIDD_MOD_DII             213
#define B3SOIDD_MOD_ALPHA1	    214
#define B3SOIDD_MOD_NGIDL           215
#define B3SOIDD_MOD_AGIDL           216
#define B3SOIDD_MOD_BGIDL           217
#define B3SOIDD_MOD_NDIODE          218
#define B3SOIDD_MOD_LDIOF           219
#define B3SOIDD_MOD_LDIOR           220
#define B3SOIDD_MOD_NTUN            221
#define B3SOIDD_MOD_ISBJT           222
#define B3SOIDD_MOD_ISDIF           223
#define B3SOIDD_MOD_ISREC           224
#define B3SOIDD_MOD_ISTUN           225
#define B3SOIDD_MOD_XBJT            226
#define B3SOIDD_MOD_XDIF            227
#define B3SOIDD_MOD_XREC            228
#define B3SOIDD_MOD_XTUN            229
#define B3SOIDD_MOD_EDL             230
#define B3SOIDD_MOD_KBJT1           231
#define B3SOIDD_MOD_TT              232
#define B3SOIDD_MOD_VSDTH           233
#define B3SOIDD_MOD_VSDFB           234
#define B3SOIDD_MOD_ASD             235
#define B3SOIDD_MOD_CSDMIN          236
#define B3SOIDD_MOD_RBSH            237

/* Added for binning - START3 */
/* Length dependence */
#define B3SOIDD_MOD_LNPEAK           301
#define B3SOIDD_MOD_LNSUB            302
#define B3SOIDD_MOD_LNGATE           303
#define B3SOIDD_MOD_LVTH0            304
#define B3SOIDD_MOD_LK1              305     
#define B3SOIDD_MOD_LK2              306
#define B3SOIDD_MOD_LK3              307
#define B3SOIDD_MOD_LK3B             308
#define B3SOIDD_MOD_LVBSA            309
#define B3SOIDD_MOD_LDELP            310
#define B3SOIDD_MOD_LKB1             311
#define B3SOIDD_MOD_LKB3             312
#define B3SOIDD_MOD_LDVBD0           313
#define B3SOIDD_MOD_LDVBD1           314
#define B3SOIDD_MOD_LW0              315
#define B3SOIDD_MOD_LNLX             316
#define B3SOIDD_MOD_LDVT0            317
#define B3SOIDD_MOD_LDVT1            318
#define B3SOIDD_MOD_LDVT2            319
#define B3SOIDD_MOD_LDVT0W           320
#define B3SOIDD_MOD_LDVT1W           321
#define B3SOIDD_MOD_LDVT2W           322
#define B3SOIDD_MOD_LU0              323
#define B3SOIDD_MOD_LUA              324
#define B3SOIDD_MOD_LUB              325
#define B3SOIDD_MOD_LUC              326
#define B3SOIDD_MOD_LVSAT            327
#define B3SOIDD_MOD_LA0              328
#define B3SOIDD_MOD_LAGS             329
#define B3SOIDD_MOD_LB0              330
#define B3SOIDD_MOD_LB1              331
#define B3SOIDD_MOD_LKETA            332   
#define B3SOIDD_MOD_LABP             333   
#define B3SOIDD_MOD_LMXC             334   
#define B3SOIDD_MOD_LADICE0          335   
#define B3SOIDD_MOD_LA1              336
#define B3SOIDD_MOD_LA2              337
#define B3SOIDD_MOD_LRDSW            338
#define B3SOIDD_MOD_LPRWB            339
#define B3SOIDD_MOD_LPRWG            340
#define B3SOIDD_MOD_LWR              341
#define B3SOIDD_MOD_LNFACTOR         342
#define B3SOIDD_MOD_LDWG             343
#define B3SOIDD_MOD_LDWB             344
#define B3SOIDD_MOD_LVOFF            345
#define B3SOIDD_MOD_LETA0            346
#define B3SOIDD_MOD_LETAB            347
#define B3SOIDD_MOD_LDSUB            348
#define B3SOIDD_MOD_LCIT             349
#define B3SOIDD_MOD_LCDSC            350
#define B3SOIDD_MOD_LCDSCB           351
#define B3SOIDD_MOD_LCDSCD           352
#define B3SOIDD_MOD_LPCLM            353
#define B3SOIDD_MOD_LPDIBL1          354
#define B3SOIDD_MOD_LPDIBL2          355
#define B3SOIDD_MOD_LPDIBLB          356
#define B3SOIDD_MOD_LDROUT           357
#define B3SOIDD_MOD_LPVAG            358
#define B3SOIDD_MOD_LDELTA           359
#define B3SOIDD_MOD_LAII             360
#define B3SOIDD_MOD_LBII             361
#define B3SOIDD_MOD_LCII             362
#define B3SOIDD_MOD_LDII             363
#define B3SOIDD_MOD_LALPHA0          364
#define B3SOIDD_MOD_LALPHA1          365
#define B3SOIDD_MOD_LBETA0           366
#define B3SOIDD_MOD_LAGIDL           367
#define B3SOIDD_MOD_LBGIDL           368
#define B3SOIDD_MOD_LNGIDL           369
#define B3SOIDD_MOD_LNTUN            370
#define B3SOIDD_MOD_LNDIODE          371
#define B3SOIDD_MOD_LISBJT           372
#define B3SOIDD_MOD_LISDIF           373
#define B3SOIDD_MOD_LISREC           374
#define B3SOIDD_MOD_LISTUN           375
#define B3SOIDD_MOD_LEDL             376
#define B3SOIDD_MOD_LKBJT1           377
#define B3SOIDD_MOD_LVSDFB           378
#define B3SOIDD_MOD_LVSDTH           379

/* Width dependence */
#define B3SOIDD_MOD_WNPEAK           401
#define B3SOIDD_MOD_WNSUB            402
#define B3SOIDD_MOD_WNGATE           403
#define B3SOIDD_MOD_WVTH0            404
#define B3SOIDD_MOD_WK1              405     
#define B3SOIDD_MOD_WK2              406
#define B3SOIDD_MOD_WK3              407
#define B3SOIDD_MOD_WK3B             408
#define B3SOIDD_MOD_WVBSA            409
#define B3SOIDD_MOD_WDELP            410
#define B3SOIDD_MOD_WKB1             411
#define B3SOIDD_MOD_WKB3             412
#define B3SOIDD_MOD_WDVBD0           413
#define B3SOIDD_MOD_WDVBD1           414
#define B3SOIDD_MOD_WW0              415
#define B3SOIDD_MOD_WNLX             416
#define B3SOIDD_MOD_WDVT0            417
#define B3SOIDD_MOD_WDVT1            418
#define B3SOIDD_MOD_WDVT2            419
#define B3SOIDD_MOD_WDVT0W           420
#define B3SOIDD_MOD_WDVT1W           421
#define B3SOIDD_MOD_WDVT2W           422
#define B3SOIDD_MOD_WU0              423
#define B3SOIDD_MOD_WUA              424
#define B3SOIDD_MOD_WUB              425
#define B3SOIDD_MOD_WUC              426
#define B3SOIDD_MOD_WVSAT            427
#define B3SOIDD_MOD_WA0              428
#define B3SOIDD_MOD_WAGS             429
#define B3SOIDD_MOD_WB0              430
#define B3SOIDD_MOD_WB1              431
#define B3SOIDD_MOD_WKETA            432   
#define B3SOIDD_MOD_WABP             433   
#define B3SOIDD_MOD_WMXC             434   
#define B3SOIDD_MOD_WADICE0          435   
#define B3SOIDD_MOD_WA1              436
#define B3SOIDD_MOD_WA2              437
#define B3SOIDD_MOD_WRDSW            438
#define B3SOIDD_MOD_WPRWB            439
#define B3SOIDD_MOD_WPRWG            440
#define B3SOIDD_MOD_WWR              441
#define B3SOIDD_MOD_WNFACTOR         442
#define B3SOIDD_MOD_WDWG             443
#define B3SOIDD_MOD_WDWB             444
#define B3SOIDD_MOD_WVOFF            445
#define B3SOIDD_MOD_WETA0            446
#define B3SOIDD_MOD_WETAB            447
#define B3SOIDD_MOD_WDSUB            448
#define B3SOIDD_MOD_WCIT             449
#define B3SOIDD_MOD_WCDSC            450
#define B3SOIDD_MOD_WCDSCB           451
#define B3SOIDD_MOD_WCDSCD           452
#define B3SOIDD_MOD_WPCLM            453
#define B3SOIDD_MOD_WPDIBL1          454
#define B3SOIDD_MOD_WPDIBL2          455
#define B3SOIDD_MOD_WPDIBLB          456
#define B3SOIDD_MOD_WDROUT           457
#define B3SOIDD_MOD_WPVAG            458
#define B3SOIDD_MOD_WDELTA           459
#define B3SOIDD_MOD_WAII             460
#define B3SOIDD_MOD_WBII             461
#define B3SOIDD_MOD_WCII             462
#define B3SOIDD_MOD_WDII             463
#define B3SOIDD_MOD_WALPHA0          464
#define B3SOIDD_MOD_WALPHA1          465
#define B3SOIDD_MOD_WBETA0           466
#define B3SOIDD_MOD_WAGIDL           467
#define B3SOIDD_MOD_WBGIDL           468
#define B3SOIDD_MOD_WNGIDL           469
#define B3SOIDD_MOD_WNTUN            470
#define B3SOIDD_MOD_WNDIODE          471
#define B3SOIDD_MOD_WISBJT           472
#define B3SOIDD_MOD_WISDIF           473
#define B3SOIDD_MOD_WISREC           474
#define B3SOIDD_MOD_WISTUN           475
#define B3SOIDD_MOD_WEDL             476
#define B3SOIDD_MOD_WKBJT1           477
#define B3SOIDD_MOD_WVSDFB           478
#define B3SOIDD_MOD_WVSDTH           479

/* Cross-term dependence */
#define B3SOIDD_MOD_PNPEAK           501
#define B3SOIDD_MOD_PNSUB            502
#define B3SOIDD_MOD_PNGATE           503
#define B3SOIDD_MOD_PVTH0            504
#define B3SOIDD_MOD_PK1              505     
#define B3SOIDD_MOD_PK2              506
#define B3SOIDD_MOD_PK3              507
#define B3SOIDD_MOD_PK3B             508
#define B3SOIDD_MOD_PVBSA            509
#define B3SOIDD_MOD_PDELP            510
#define B3SOIDD_MOD_PKB1             511
#define B3SOIDD_MOD_PKB3             512
#define B3SOIDD_MOD_PDVBD0           513
#define B3SOIDD_MOD_PDVBD1           514
#define B3SOIDD_MOD_PW0              515
#define B3SOIDD_MOD_PNLX             516
#define B3SOIDD_MOD_PDVT0            517
#define B3SOIDD_MOD_PDVT1            518
#define B3SOIDD_MOD_PDVT2            519
#define B3SOIDD_MOD_PDVT0W           520
#define B3SOIDD_MOD_PDVT1W           521
#define B3SOIDD_MOD_PDVT2W           522
#define B3SOIDD_MOD_PU0              523
#define B3SOIDD_MOD_PUA              524
#define B3SOIDD_MOD_PUB              525
#define B3SOIDD_MOD_PUC              526
#define B3SOIDD_MOD_PVSAT            527
#define B3SOIDD_MOD_PA0              528
#define B3SOIDD_MOD_PAGS             529
#define B3SOIDD_MOD_PB0              530
#define B3SOIDD_MOD_PB1              531
#define B3SOIDD_MOD_PKETA            532   
#define B3SOIDD_MOD_PABP             533   
#define B3SOIDD_MOD_PMXC             534   
#define B3SOIDD_MOD_PADICE0          535   
#define B3SOIDD_MOD_PA1              536
#define B3SOIDD_MOD_PA2              537
#define B3SOIDD_MOD_PRDSW            538
#define B3SOIDD_MOD_PPRWB            539
#define B3SOIDD_MOD_PPRWG            540
#define B3SOIDD_MOD_PWR              541
#define B3SOIDD_MOD_PNFACTOR         542
#define B3SOIDD_MOD_PDWG             543
#define B3SOIDD_MOD_PDWB             544
#define B3SOIDD_MOD_PVOFF            545
#define B3SOIDD_MOD_PETA0            546
#define B3SOIDD_MOD_PETAB            547
#define B3SOIDD_MOD_PDSUB            548
#define B3SOIDD_MOD_PCIT             549
#define B3SOIDD_MOD_PCDSC            550
#define B3SOIDD_MOD_PCDSCB           551
#define B3SOIDD_MOD_PCDSCD           552
#define B3SOIDD_MOD_PPCLM            553
#define B3SOIDD_MOD_PPDIBL1          554
#define B3SOIDD_MOD_PPDIBL2          555
#define B3SOIDD_MOD_PPDIBLB          556
#define B3SOIDD_MOD_PDROUT           557
#define B3SOIDD_MOD_PPVAG            558
#define B3SOIDD_MOD_PDELTA           559
#define B3SOIDD_MOD_PAII             560
#define B3SOIDD_MOD_PBII             561
#define B3SOIDD_MOD_PCII             562
#define B3SOIDD_MOD_PDII             563
#define B3SOIDD_MOD_PALPHA0          564
#define B3SOIDD_MOD_PALPHA1          565
#define B3SOIDD_MOD_PBETA0           566
#define B3SOIDD_MOD_PAGIDL           567
#define B3SOIDD_MOD_PBGIDL           568
#define B3SOIDD_MOD_PNGIDL           569
#define B3SOIDD_MOD_PNTUN            570
#define B3SOIDD_MOD_PNDIODE          571
#define B3SOIDD_MOD_PISBJT           572
#define B3SOIDD_MOD_PISDIF           573
#define B3SOIDD_MOD_PISREC           574
#define B3SOIDD_MOD_PISTUN           575
#define B3SOIDD_MOD_PEDL             576
#define B3SOIDD_MOD_PKBJT1           577
#define B3SOIDD_MOD_PVSDFB           578
#define B3SOIDD_MOD_PVSDTH           579
/* Added for binning - END3 */

#define B3SOIDD_MOD_TNOM             701
#define B3SOIDD_MOD_CGSO             702
#define B3SOIDD_MOD_CGDO             703
#define B3SOIDD_MOD_CGEO             704
#define B3SOIDD_MOD_XPART            705

#define B3SOIDD_MOD_RSH              706
#define B3SOIDD_MOD_NMOS             814
#define B3SOIDD_MOD_PMOS             815

#define B3SOIDD_MOD_NOIA             816
#define B3SOIDD_MOD_NOIB             817
#define B3SOIDD_MOD_NOIC             818

#define B3SOIDD_MOD_LINT             819
#define B3SOIDD_MOD_LL               820
#define B3SOIDD_MOD_LLN              821
#define B3SOIDD_MOD_LW               822
#define B3SOIDD_MOD_LWN              823
#define B3SOIDD_MOD_LWL              824

#define B3SOIDD_MOD_WINT             827
#define B3SOIDD_MOD_WL               828
#define B3SOIDD_MOD_WLN              829
#define B3SOIDD_MOD_WW               830
#define B3SOIDD_MOD_WWN              831
#define B3SOIDD_MOD_WWL              832

#define B3SOIDD_MOD_DWC              835
#define B3SOIDD_MOD_DLC              836

#define B3SOIDD_MOD_EM               837
#define B3SOIDD_MOD_EF               838
#define B3SOIDD_MOD_AF               839
#define B3SOIDD_MOD_KF               840
#define B3SOIDD_MOD_NOIF             841


#define B3SOIDD_MOD_PBSWG            843
#define B3SOIDD_MOD_MJSWG            844
#define B3SOIDD_MOD_CJSWG            845
#define B3SOIDD_MOD_CSDESW           846

/* device questions */
#define B3SOIDD_DNODE                901
#define B3SOIDD_GNODE                902
#define B3SOIDD_SNODE                903
#define B3SOIDD_BNODE                904
#define B3SOIDD_ENODE                905
#define B3SOIDD_DNODEPRIME           906
#define B3SOIDD_SNODEPRIME           907
#define B3SOIDD_VBD                  908
#define B3SOIDD_VBS                  909
#define B3SOIDD_VGS                  910
#define B3SOIDD_VES                  911
#define B3SOIDD_VDS                  912
#define B3SOIDD_CD                   913
#define B3SOIDD_CBS                  914
#define B3SOIDD_CBD                  915
#define B3SOIDD_GM                   916
#define B3SOIDD_GDS                  917
#define B3SOIDD_GMBS                 918
#define B3SOIDD_GBD                  919
#define B3SOIDD_GBS                  920
#define B3SOIDD_QB                   921
#define B3SOIDD_CQB                  922
#define B3SOIDD_QG                   923
#define B3SOIDD_CQG                  924
#define B3SOIDD_QD                   925
#define B3SOIDD_CQD                  926
#define B3SOIDD_CGG                  927
#define B3SOIDD_CGD                  928
#define B3SOIDD_CGS                  929
#define B3SOIDD_CBG                  930
#define B3SOIDD_CAPBD                931
#define B3SOIDD_CQBD                 932
#define B3SOIDD_CAPBS                933
#define B3SOIDD_CQBS                 934
#define B3SOIDD_CDG                  935
#define B3SOIDD_CDD                  936
#define B3SOIDD_CDS                  937
#define B3SOIDD_VON                  938
#define B3SOIDD_VDSAT                939
#define B3SOIDD_QBS                  940
#define B3SOIDD_QBD                  941
#define B3SOIDD_SOURCECONDUCT        942
#define B3SOIDD_DRAINCONDUCT         943
#define B3SOIDD_CBDB                 944
#define B3SOIDD_CBSB                 945
#define B3SOIDD_GMID                 946


#include "b3soiddext.h"

extern void B3SOIDDevaluate(double,double,double,B3SOIDDinstance*,B3SOIDDmodel*,
        double*,double*,double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, CKTcircuit*);
extern int B3SOIDDdebug(B3SOIDDmodel*, B3SOIDDinstance*, CKTcircuit*, int);
extern int B3SOIDDcheckModel(B3SOIDDmodel*, B3SOIDDinstance*, CKTcircuit*);

#endif /*B3SOIDD*/

