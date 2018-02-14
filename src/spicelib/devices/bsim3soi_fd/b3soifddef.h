/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: Weidong Liu and Pin Su         Feb 1999
Author: 1998 Samuel Fung
Modified by Pin Su, Wei Jin 99/9/27
Modified by Paolo Nenzi 2002
File: b3soifddef.h
**********/

#ifndef B3SOIFD
#define B3SOIFD

#define SOICODE
/*  #define BULKCODE  */

#include "ngspice/ifsim.h"
#include "ngspice/gendefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/noisedef.h"         

typedef struct sB3SOIFDinstance
{

    struct GENinstance gen;

#define B3SOIFDmodPtr(inst) ((struct sB3SOIFDmodel *)((inst)->gen.GENmodPtr))
#define B3SOIFDnextInstance(inst) ((struct sB3SOIFDinstance *)((inst)->gen.GENnextInstance))
#define B3SOIFDname gen.GENname
#define B3SOIFDstates gen.GENstate

    const int B3SOIFDdNode;
    const int B3SOIFDgNode;
    const int B3SOIFDsNode;
    const int B3SOIFDeNode;
    const int B3SOIFDbNodeExt;
    const int B3SOIFDtempNodeExt;
    const int B3SOIFDpNodeExt;
    int B3SOIFDbNode;
    int B3SOIFDtempNode;
    int B3SOIFDpNode;  
    int B3SOIFDdNodePrime;
    int B3SOIFDsNodePrime;

    int B3SOIFDvbsNode;
    /* for Debug */
    int B3SOIFDidsNode;
    int B3SOIFDicNode;
    int B3SOIFDibsNode;
    int B3SOIFDibdNode;
    int B3SOIFDiiiNode;
    int B3SOIFDigidlNode;
    int B3SOIFDitunNode;
    int B3SOIFDibpNode;
    int B3SOIFDabeffNode;
    int B3SOIFDvbs0effNode;
    int B3SOIFDvbseffNode;
    int B3SOIFDxcNode;
    int B3SOIFDcbbNode;
    int B3SOIFDcbdNode;
    int B3SOIFDcbeNode;
    int B3SOIFDcbgNode;
    int B3SOIFDqbNode;
    int B3SOIFDqbfNode;
    int B3SOIFDqjsNode;
    int B3SOIFDqjdNode;

/* clean up last */
    int B3SOIFDgmNode;
    int B3SOIFDgmbsNode;
    int B3SOIFDgdsNode;
    int B3SOIFDgmeNode;
    int B3SOIFDqgNode;
    int B3SOIFDqdNode;
    int B3SOIFDqeNode;
    int B3SOIFDiterations;
    int B3SOIFDvbs0teffNode;
    int B3SOIFDvthNode;
    int B3SOIFDvgsteffNode;
    int B3SOIFDxcsatNode;
    int B3SOIFDqaccNode;
    int B3SOIFDqsub0Node;
    int B3SOIFDqsubs1Node;
    int B3SOIFDqsubs2Node;
    int B3SOIFDvcscvNode;
    int B3SOIFDvdscvNode;
    int B3SOIFDdum1Node;
    int B3SOIFDdum2Node;
    int B3SOIFDdum3Node;
    int B3SOIFDdum4Node;
    int B3SOIFDdum5Node;
/* end clean up last */

    double B3SOIFDphi;
    double B3SOIFDvtm;
    double B3SOIFDni;
    double B3SOIFDueff;
    double B3SOIFDthetavth; 
    double B3SOIFDvon;
    double B3SOIFDvbsdio;
    double B3SOIFDvdsat;
    double B3SOIFDcgdo;
    double B3SOIFDcgso;
    double B3SOIFDcgeo;

    double B3SOIFDids;
    double B3SOIFDic;
    double B3SOIFDibs;
    double B3SOIFDibd;
    double B3SOIFDiii;
    double B3SOIFDigidl;
    double B3SOIFDitun;
    double B3SOIFDibp;
    double B3SOIFDabeff;
    double B3SOIFDvbs0eff;
    double B3SOIFDvbseff;
    double B3SOIFDxc;
    double B3SOIFDcbg;
    double B3SOIFDcbb;
    double B3SOIFDcbd;
    double B3SOIFDqb;
    double B3SOIFDqbf;
    double B3SOIFDqjs;
    double B3SOIFDqjd;
    double B3SOIFDminIsub;
    int    B3SOIFDfloat;

/* clean up last */
    double B3SOIFDdum1;
    double B3SOIFDdum2;
    double B3SOIFDdum3;
    double B3SOIFDdum4;
    double B3SOIFDdum5;
/* end clean up last */

    double B3SOIFDl;
    double B3SOIFDw;
    double B3SOIFDm;
    double B3SOIFDdrainArea;
    double B3SOIFDsourceArea;
    double B3SOIFDdrainSquares;
    double B3SOIFDsourceSquares;
    double B3SOIFDdrainPerimeter;
    double B3SOIFDsourcePerimeter;
    double B3SOIFDsourceConductance;
    double B3SOIFDdrainConductance;

    double B3SOIFDicVBS;
    double B3SOIFDicVDS;
    double B3SOIFDicVGS;
    double B3SOIFDicVES;
    double B3SOIFDicVPS;
    int B3SOIFDbjtoff;
    int B3SOIFDbodyMod;
    int B3SOIFDdebugMod;
    double B3SOIFDrth0;
    double B3SOIFDcth0;
    double B3SOIFDbodySquares;
    double B3SOIFDrbodyext;

    double B3SOIFDcsbox;
    double B3SOIFDcdbox;
    double B3SOIFDcsmin;
    double B3SOIFDcdmin;
    double B3SOIFDst4;
    double B3SOIFDdt4;

    int B3SOIFDoff;
    int B3SOIFDmode;

    /* OP point */
    double B3SOIFDqinv;
    double B3SOIFDcd;
    double B3SOIFDcjs;
    double B3SOIFDcjd;
    double B3SOIFDcbody;
    double B3SOIFDcbodcon;
    double B3SOIFDcth;
    double B3SOIFDcsubstrate;

    double B3SOIFDgm;
    double B3SOIFDgme;
    double B3SOIFDcb;
    double B3SOIFDcdrain;
    double B3SOIFDgds;
    double B3SOIFDgmbs;
    double B3SOIFDgmT;

    double B3SOIFDgbbs;
    double B3SOIFDgbgs;
    double B3SOIFDgbds;
    double B3SOIFDgbes;
    double B3SOIFDgbps;
    double B3SOIFDgbT;

    double B3SOIFDgjsd;
    double B3SOIFDgjsb;
    double B3SOIFDgjsg;
    double B3SOIFDgjsT;

    double B3SOIFDgjdb;
    double B3SOIFDgjdd;
    double B3SOIFDgjdg;
    double B3SOIFDgjde;
    double B3SOIFDgjdT;

    double B3SOIFDgbpbs;
    double B3SOIFDgbpgs;
    double B3SOIFDgbpds;
    double B3SOIFDgbpes;
    double B3SOIFDgbpps;
    double B3SOIFDgbpT;

    double B3SOIFDgtempb;
    double B3SOIFDgtempg;
    double B3SOIFDgtempd;
    double B3SOIFDgtempe;
    double B3SOIFDgtempT;

    double B3SOIFDcggb;
    double B3SOIFDcgdb;
    double B3SOIFDcgsb;
    double B3SOIFDcgeb;
    double B3SOIFDcgT;

    double B3SOIFDcbgb;
    double B3SOIFDcbdb;
    double B3SOIFDcbsb;
    double B3SOIFDcbeb;
    double B3SOIFDcbT;

    double B3SOIFDcdgb;
    double B3SOIFDcddb;
    double B3SOIFDcdsb;
    double B3SOIFDcdeb;
    double B3SOIFDcdT;

    double B3SOIFDcegb;
    double B3SOIFDcedb;
    double B3SOIFDcesb;
    double B3SOIFDceeb;
    double B3SOIFDceT;

    double B3SOIFDqse;
    double B3SOIFDgcse;
    double B3SOIFDqde;
    double B3SOIFDgcde;

    struct b3soifdSizeDependParam  *pParam;

    unsigned B3SOIFDlGiven :1;
    unsigned B3SOIFDwGiven :1;
    unsigned B3SOIFDmGiven :1;
    unsigned B3SOIFDdrainAreaGiven :1;
    unsigned B3SOIFDsourceAreaGiven    :1;
    unsigned B3SOIFDdrainSquaresGiven  :1;
    unsigned B3SOIFDsourceSquaresGiven :1;
    unsigned B3SOIFDdrainPerimeterGiven    :1;
    unsigned B3SOIFDsourcePerimeterGiven   :1;
    unsigned B3SOIFDdNodePrimeSet  :1;
    unsigned B3SOIFDsNodePrimeSet  :1;
    unsigned B3SOIFDicVBSGiven :1;
    unsigned B3SOIFDicVDSGiven :1;
    unsigned B3SOIFDicVGSGiven :1;
    unsigned B3SOIFDicVESGiven :1;
    unsigned B3SOIFDicVPSGiven :1;
    unsigned B3SOIFDbjtoffGiven :1;
    unsigned B3SOIFDdebugModGiven :1;
    unsigned B3SOIFDrth0Given :1;
    unsigned B3SOIFDcth0Given :1;
    unsigned B3SOIFDbodySquaresGiven :1;
    unsigned B3SOIFDoffGiven :1;
    
    double *B3SOIFDEePtr;
    double *B3SOIFDEbPtr;
    double *B3SOIFDBePtr;
    double *B3SOIFDEgPtr;
    double *B3SOIFDEdpPtr;
    double *B3SOIFDEspPtr;
    double *B3SOIFDTemptempPtr;
    double *B3SOIFDTempdpPtr;
    double *B3SOIFDTempspPtr;
    double *B3SOIFDTempgPtr;
    double *B3SOIFDTempbPtr;
    double *B3SOIFDTempePtr;
    double *B3SOIFDGtempPtr;
    double *B3SOIFDDPtempPtr;
    double *B3SOIFDSPtempPtr;
    double *B3SOIFDEtempPtr;
    double *B3SOIFDBtempPtr;
    double *B3SOIFDPtempPtr;
    double *B3SOIFDBpPtr;
    double *B3SOIFDPbPtr;
    double *B3SOIFDPpPtr;
    double *B3SOIFDPgPtr;
    double *B3SOIFDPdpPtr;
    double *B3SOIFDPspPtr;
    double *B3SOIFDPePtr;
    double *B3SOIFDDPePtr;
    double *B3SOIFDSPePtr;
    double *B3SOIFDGePtr;
    double *B3SOIFDDdPtr;
    double *B3SOIFDGgPtr;
    double *B3SOIFDSsPtr;
    double *B3SOIFDBbPtr;
    double *B3SOIFDDPdpPtr;
    double *B3SOIFDSPspPtr;
    double *B3SOIFDDdpPtr;
    double *B3SOIFDGbPtr;
    double *B3SOIFDGdpPtr;
    double *B3SOIFDGspPtr;
    double *B3SOIFDSspPtr;
    double *B3SOIFDBdpPtr;
    double *B3SOIFDBspPtr;
    double *B3SOIFDDPspPtr;
    double *B3SOIFDDPdPtr;
    double *B3SOIFDBgPtr;
    double *B3SOIFDDPgPtr;
    double *B3SOIFDSPgPtr;
    double *B3SOIFDSPsPtr;
    double *B3SOIFDDPbPtr;
    double *B3SOIFDSPbPtr;
    double *B3SOIFDSPdpPtr;

    double *B3SOIFDVbsPtr;
    /* Debug */
    double *B3SOIFDIdsPtr;
    double *B3SOIFDIcPtr;
    double *B3SOIFDIbsPtr;
    double *B3SOIFDIbdPtr;
    double *B3SOIFDIiiPtr;
    double *B3SOIFDIgidlPtr;
    double *B3SOIFDItunPtr;
    double *B3SOIFDIbpPtr;
    double *B3SOIFDAbeffPtr;
    double *B3SOIFDVbs0effPtr;
    double *B3SOIFDVbseffPtr;
    double *B3SOIFDXcPtr;
    double *B3SOIFDCbbPtr;
    double *B3SOIFDCbdPtr;
    double *B3SOIFDCbgPtr;
    double *B3SOIFDqbPtr;
    double *B3SOIFDQbfPtr;
    double *B3SOIFDQjsPtr;
    double *B3SOIFDQjdPtr;

    /* clean up last */
    double *B3SOIFDGmPtr;
    double *B3SOIFDGmbsPtr;
    double *B3SOIFDGdsPtr;
    double *B3SOIFDGmePtr;
    double *B3SOIFDVbs0teffPtr;
    double *B3SOIFDVthPtr;
    double *B3SOIFDVgsteffPtr;
    double *B3SOIFDXcsatPtr;
    double *B3SOIFDQaccPtr;
    double *B3SOIFDQsub0Ptr;
    double *B3SOIFDQsubs1Ptr;
    double *B3SOIFDQsubs2Ptr;
    double *B3SOIFDVdscvPtr;
    double *B3SOIFDVcscvPtr;
    double *B3SOIFDCbePtr;
    double *B3SOIFDqgPtr;
    double *B3SOIFDqdPtr;
    double *B3SOIFDqePtr;
    double *B3SOIFDDum1Ptr;
    double *B3SOIFDDum2Ptr;
    double *B3SOIFDDum3Ptr;
    double *B3SOIFDDum4Ptr;
    double *B3SOIFDDum5Ptr;
    /* End clean up last */

#define B3SOIFDvbd B3SOIFDstates+ 0
#define B3SOIFDvbs B3SOIFDstates+ 1
#define B3SOIFDvgs B3SOIFDstates+ 2
#define B3SOIFDvds B3SOIFDstates+ 3
#define B3SOIFDves B3SOIFDstates+ 4
#define B3SOIFDvps B3SOIFDstates+ 5

#define B3SOIFDvg B3SOIFDstates+ 6
#define B3SOIFDvd B3SOIFDstates+ 7
#define B3SOIFDvs B3SOIFDstates+ 8
#define B3SOIFDvp B3SOIFDstates+ 9
#define B3SOIFDve B3SOIFDstates+ 10
#define B3SOIFDdeltemp B3SOIFDstates+ 11

#define B3SOIFDqb B3SOIFDstates+ 12
#define B3SOIFDcqb B3SOIFDstates+ 13
#define B3SOIFDqg B3SOIFDstates+ 14
#define B3SOIFDcqg B3SOIFDstates+ 15
#define B3SOIFDqd B3SOIFDstates+ 16
#define B3SOIFDcqd B3SOIFDstates+ 17
#define B3SOIFDqe B3SOIFDstates+ 18
#define B3SOIFDcqe B3SOIFDstates+ 19

#define B3SOIFDqbs  B3SOIFDstates+ 20
#define B3SOIFDqbd  B3SOIFDstates+ 21
#define B3SOIFDqbe  B3SOIFDstates+ 22

#define B3SOIFDqth B3SOIFDstates+ 23
#define B3SOIFDcqth B3SOIFDstates+ 24

#define B3SOIFDnumStates 25


/* indices to the array of B3SOIFD NOISE SOURCES */

#define B3SOIFDRDNOIZ       0
#define B3SOIFDRSNOIZ       1
#define B3SOIFDIDNOIZ       2
#define B3SOIFDFLNOIZ       3
#define B3SOIFDFBNOIZ       4
#define B3SOIFDTOTNOIZ      5

#define B3SOIFDNSRCS        6     /* the number of MOSFET(3) noise sources */

#ifndef NONOISE
    double B3SOIFDnVar[NSTATVARS][B3SOIFDNSRCS];
#else /* NONOISE */
        double **B3SOIFDnVar;
#endif /* NONOISE */

} B3SOIFDinstance ;

struct b3soifdSizeDependParam
{
    double Width;
    double Length;
    double Rth0;
    double Cth0;

    double B3SOIFDcdsc;           
    double B3SOIFDcdscb;    
    double B3SOIFDcdscd;       
    double B3SOIFDcit;           
    double B3SOIFDnfactor;      
    double B3SOIFDvsat;         
    double B3SOIFDat;         
    double B3SOIFDa0;   
    double B3SOIFDags;      
    double B3SOIFDa1;         
    double B3SOIFDa2;         
    double B3SOIFDketa;     
    double B3SOIFDnpeak;        
    double B3SOIFDnsub;
    double B3SOIFDngate;        
    double B3SOIFDgamma1;      
    double B3SOIFDgamma2;     
    double B3SOIFDvbx;      
    double B3SOIFDvbi;       
    double B3SOIFDvbm;       
    double B3SOIFDvbsc;       
    double B3SOIFDxt;       
    double B3SOIFDphi;
    double B3SOIFDlitl;
    double B3SOIFDk1;
    double B3SOIFDkt1;
    double B3SOIFDkt1l;
    double B3SOIFDkt2;
    double B3SOIFDk2;
    double B3SOIFDk3;
    double B3SOIFDk3b;
    double B3SOIFDw0;
    double B3SOIFDnlx;
    double B3SOIFDdvt0;      
    double B3SOIFDdvt1;      
    double B3SOIFDdvt2;      
    double B3SOIFDdvt0w;      
    double B3SOIFDdvt1w;      
    double B3SOIFDdvt2w;      
    double B3SOIFDdrout;      
    double B3SOIFDdsub;      
    double B3SOIFDvth0;
    double B3SOIFDua;
    double B3SOIFDua1;
    double B3SOIFDub;
    double B3SOIFDub1;
    double B3SOIFDuc;
    double B3SOIFDuc1;
    double B3SOIFDu0;
    double B3SOIFDute;
    double B3SOIFDvoff;
    double B3SOIFDvfb;
    double B3SOIFDuatemp;
    double B3SOIFDubtemp;
    double B3SOIFDuctemp;
    double B3SOIFDrbody;
    double B3SOIFDrth;
    double B3SOIFDcth;
    double B3SOIFDrds0denom;
    double B3SOIFDvfbb;
    double B3SOIFDjbjt;
    double B3SOIFDjdif;
    double B3SOIFDjrec;
    double B3SOIFDjtun;
    double B3SOIFDcsesw;
    double B3SOIFDcdesw;

  /* Added */
    double B3SOIFDsdt1;
    double B3SOIFDst2;
    double B3SOIFDst3;
    double B3SOIFDdt2;
    double B3SOIFDdt3;
  /* Added */

    double B3SOIFDdelta;
    double B3SOIFDrdsw;       
    double B3SOIFDrds0;       
    double B3SOIFDprwg;       
    double B3SOIFDprwb;       
    double B3SOIFDprt;       
    double B3SOIFDeta0;         
    double B3SOIFDetab;         
    double B3SOIFDpclm;      
    double B3SOIFDpdibl1;      
    double B3SOIFDpdibl2;      
    double B3SOIFDpdiblb;      
    double B3SOIFDpvag;       
    double B3SOIFDwr;
    double B3SOIFDdwg;
    double B3SOIFDdwb;
    double B3SOIFDb0;
    double B3SOIFDb1;
    double B3SOIFDalpha0;
    double B3SOIFDalpha1;
    double B3SOIFDbeta0;


    /* CV model */
    double B3SOIFDcgsl;
    double B3SOIFDcgdl;
    double B3SOIFDckappa;
    double B3SOIFDcf;
    double B3SOIFDclc;
    double B3SOIFDcle;

/* Added for binning - START0 */
    double B3SOIFDvbsa;
    double B3SOIFDdelp;
    double B3SOIFDkb1;
    double B3SOIFDkb3;
    double B3SOIFDdvbd0;
    double B3SOIFDdvbd1;
    double B3SOIFDabp;
    double B3SOIFDmxc;
    double B3SOIFDadice0;
    double B3SOIFDaii;
    double B3SOIFDbii;
    double B3SOIFDcii;
    double B3SOIFDdii;
    double B3SOIFDagidl;
    double B3SOIFDbgidl;
    double B3SOIFDngidl;
    double B3SOIFDntun;
    double B3SOIFDndiode;
    double B3SOIFDisbjt;
    double B3SOIFDisdif;
    double B3SOIFDisrec;
    double B3SOIFDistun;
    double B3SOIFDedl;
    double B3SOIFDkbjt1;
    double B3SOIFDvsdfb;
    double B3SOIFDvsdth;
/* Added for binning - END0 */

/* Pre-calculated constants */

    double B3SOIFDdw;
    double B3SOIFDdl;
    double B3SOIFDleff;
    double B3SOIFDweff;

    double B3SOIFDdwc;
    double B3SOIFDdlc;
    double B3SOIFDleffCV;
    double B3SOIFDweffCV;
    double B3SOIFDabulkCVfactor;
    double B3SOIFDcgso;
    double B3SOIFDcgdo;
    double B3SOIFDcgeo;

    double B3SOIFDu0temp;       
    double B3SOIFDvsattemp;   
    double B3SOIFDsqrtPhi;   
    double B3SOIFDphis3;   
    double B3SOIFDXdep0;          
    double B3SOIFDsqrtXdep0;          
    double B3SOIFDtheta0vb0;
    double B3SOIFDthetaRout; 

    double B3SOIFDcof1;
    double B3SOIFDcof2;
    double B3SOIFDcof3;
    double B3SOIFDcof4;
    double B3SOIFDcdep0;
    struct b3soifdSizeDependParam  *pNext;
};


typedef struct sB3SOIFDmodel 
{

    struct GENmodel gen;

#define B3SOIFDmodType gen.GENmodType
#define B3SOIFDnextModel(inst) ((struct sB3SOIFDmodel *)((inst)->gen.GENnextModel))
#define B3SOIFDinstances(inst) ((B3SOIFDinstance *)((inst)->gen.GENinstances))
#define B3SOIFDmodName gen.GENmodName

    int B3SOIFDtype;

    int    B3SOIFDmobMod;
    int    B3SOIFDcapMod;
    int    B3SOIFDnoiMod;
    int    B3SOIFDshMod;
    int    B3SOIFDbinUnit;
    int    B3SOIFDparamChk;
    double B3SOIFDversion;             
    double B3SOIFDtox;             
    double B3SOIFDcdsc;           
    double B3SOIFDcdscb; 
    double B3SOIFDcdscd;          
    double B3SOIFDcit;           
    double B3SOIFDnfactor;      
    double B3SOIFDvsat;         
    double B3SOIFDat;         
    double B3SOIFDa0;   
    double B3SOIFDags;      
    double B3SOIFDa1;         
    double B3SOIFDa2;         
    double B3SOIFDketa;     
    double B3SOIFDnsub;
    double B3SOIFDnpeak;        
    double B3SOIFDngate;        
    double B3SOIFDgamma1;      
    double B3SOIFDgamma2;     
    double B3SOIFDvbx;      
    double B3SOIFDvbm;       
    double B3SOIFDxt;       
    double B3SOIFDk1;
    double B3SOIFDkt1;
    double B3SOIFDkt1l;
    double B3SOIFDkt2;
    double B3SOIFDk2;
    double B3SOIFDk3;
    double B3SOIFDk3b;
    double B3SOIFDw0;
    double B3SOIFDnlx;
    double B3SOIFDdvt0;      
    double B3SOIFDdvt1;      
    double B3SOIFDdvt2;      
    double B3SOIFDdvt0w;      
    double B3SOIFDdvt1w;      
    double B3SOIFDdvt2w;      
    double B3SOIFDdrout;      
    double B3SOIFDdsub;      
    double B3SOIFDvth0;
    double B3SOIFDua;
    double B3SOIFDua1;
    double B3SOIFDub;
    double B3SOIFDub1;
    double B3SOIFDuc;
    double B3SOIFDuc1;
    double B3SOIFDu0;
    double B3SOIFDute;
    double B3SOIFDvoff;
    double B3SOIFDdelta;
    double B3SOIFDrdsw;       
    double B3SOIFDprwg;
    double B3SOIFDprwb;
    double B3SOIFDprt;       
    double B3SOIFDeta0;         
    double B3SOIFDetab;         
    double B3SOIFDpclm;      
    double B3SOIFDpdibl1;      
    double B3SOIFDpdibl2;      
    double B3SOIFDpdiblb;
    double B3SOIFDpvag;       
    double B3SOIFDwr;
    double B3SOIFDdwg;
    double B3SOIFDdwb;
    double B3SOIFDb0;
    double B3SOIFDb1;
    double B3SOIFDalpha0;
    double B3SOIFDalpha1;
    double B3SOIFDbeta0;
    double B3SOIFDtbox;
    double B3SOIFDtsi;
    double B3SOIFDxj;
    double B3SOIFDkb1;
    double B3SOIFDkb3;
    double B3SOIFDdvbd0;
    double B3SOIFDdvbd1;
    double B3SOIFDvbsa;
    double B3SOIFDdelp;
    double B3SOIFDrbody;
    double B3SOIFDrbsh;
    double B3SOIFDadice0;
    double B3SOIFDabp;
    double B3SOIFDmxc;
    double B3SOIFDrth0;
    double B3SOIFDcth0;
    double B3SOIFDaii;
    double B3SOIFDbii;
    double B3SOIFDcii;
    double B3SOIFDdii;
    double B3SOIFDngidl;
    double B3SOIFDagidl;
    double B3SOIFDbgidl;
    double B3SOIFDndiode;
    double B3SOIFDntun;
    double B3SOIFDisbjt;
    double B3SOIFDisdif;
    double B3SOIFDisrec;
    double B3SOIFDistun;
    double B3SOIFDxbjt;
    double B3SOIFDxdif;
    double B3SOIFDxrec;
    double B3SOIFDxtun;
    double B3SOIFDedl;
    double B3SOIFDkbjt1;
    double B3SOIFDtt;
    double B3SOIFDvsdfb;
    double B3SOIFDvsdth;
    double B3SOIFDcsdmin;
    double B3SOIFDasd;

    /* CV model */
    double B3SOIFDcgsl;
    double B3SOIFDcgdl;
    double B3SOIFDckappa;
    double B3SOIFDcf;
    double B3SOIFDclc;
    double B3SOIFDcle;
    double B3SOIFDdwc;
    double B3SOIFDdlc;

    double B3SOIFDtnom;
    double B3SOIFDcgso;
    double B3SOIFDcgdo;
    double B3SOIFDcgeo;
    double B3SOIFDxpart;
    double B3SOIFDcFringOut;
    double B3SOIFDcFringMax;

    double B3SOIFDsheetResistance;
    double B3SOIFDbodyJctGateSideGradingCoeff;
    double B3SOIFDGatesidewallJctPotential;
    double B3SOIFDunitLengthGateSidewallJctCap;
    double B3SOIFDcsdesw;

    double B3SOIFDLint;
    double B3SOIFDLl;
    double B3SOIFDLln;
    double B3SOIFDLw;
    double B3SOIFDLwn;
    double B3SOIFDLwl;
    double B3SOIFDLmin;
    double B3SOIFDLmax;

    double B3SOIFDWint;
    double B3SOIFDWl;
    double B3SOIFDWln;
    double B3SOIFDWw;
    double B3SOIFDWwn;
    double B3SOIFDWwl;
    double B3SOIFDWmin;
    double B3SOIFDWmax;

/* Added for binning - START1 */
    /* Length Dependence */
    double B3SOIFDlnpeak;        
    double B3SOIFDlnsub;
    double B3SOIFDlngate;        
    double B3SOIFDlvth0;
    double B3SOIFDlk1;
    double B3SOIFDlk2;
    double B3SOIFDlk3;
    double B3SOIFDlk3b;
    double B3SOIFDlvbsa;
    double B3SOIFDldelp;
    double B3SOIFDlkb1;
    double B3SOIFDlkb3;
    double B3SOIFDldvbd0;
    double B3SOIFDldvbd1;
    double B3SOIFDlw0;
    double B3SOIFDlnlx;
    double B3SOIFDldvt0;      
    double B3SOIFDldvt1;      
    double B3SOIFDldvt2;      
    double B3SOIFDldvt0w;      
    double B3SOIFDldvt1w;      
    double B3SOIFDldvt2w;      
    double B3SOIFDlu0;
    double B3SOIFDlua;
    double B3SOIFDlub;
    double B3SOIFDluc;
    double B3SOIFDlvsat;         
    double B3SOIFDla0;   
    double B3SOIFDlags;      
    double B3SOIFDlb0;
    double B3SOIFDlb1;
    double B3SOIFDlketa;
    double B3SOIFDlabp;
    double B3SOIFDlmxc;
    double B3SOIFDladice0;
    double B3SOIFDla1;         
    double B3SOIFDla2;         
    double B3SOIFDlrdsw;       
    double B3SOIFDlprwb;
    double B3SOIFDlprwg;
    double B3SOIFDlwr;
    double B3SOIFDlnfactor;      
    double B3SOIFDldwg;
    double B3SOIFDldwb;
    double B3SOIFDlvoff;
    double B3SOIFDleta0;         
    double B3SOIFDletab;         
    double B3SOIFDldsub;      
    double B3SOIFDlcit;           
    double B3SOIFDlcdsc;           
    double B3SOIFDlcdscb; 
    double B3SOIFDlcdscd;          
    double B3SOIFDlpclm;      
    double B3SOIFDlpdibl1;      
    double B3SOIFDlpdibl2;      
    double B3SOIFDlpdiblb;      
    double B3SOIFDldrout;      
    double B3SOIFDlpvag;       
    double B3SOIFDldelta;
    double B3SOIFDlaii;
    double B3SOIFDlbii;
    double B3SOIFDlcii;
    double B3SOIFDldii;
    double B3SOIFDlalpha0;
    double B3SOIFDlalpha1;
    double B3SOIFDlbeta0;
    double B3SOIFDlagidl;
    double B3SOIFDlbgidl;
    double B3SOIFDlngidl;
    double B3SOIFDlntun;
    double B3SOIFDlndiode;
    double B3SOIFDlisbjt;       
    double B3SOIFDlisdif;
    double B3SOIFDlisrec;       
    double B3SOIFDlistun;       
    double B3SOIFDledl;       
    double B3SOIFDlkbjt1;       
    /* CV model */
    double B3SOIFDlvsdfb;
    double B3SOIFDlvsdth;

    /* Width Dependence */
    double B3SOIFDwnpeak;        
    double B3SOIFDwnsub;
    double B3SOIFDwngate;        
    double B3SOIFDwvth0;
    double B3SOIFDwk1;
    double B3SOIFDwk2;
    double B3SOIFDwk3;
    double B3SOIFDwk3b;
    double B3SOIFDwvbsa;
    double B3SOIFDwdelp;
    double B3SOIFDwkb1;
    double B3SOIFDwkb3;
    double B3SOIFDwdvbd0;
    double B3SOIFDwdvbd1;
    double B3SOIFDww0;
    double B3SOIFDwnlx;
    double B3SOIFDwdvt0;      
    double B3SOIFDwdvt1;      
    double B3SOIFDwdvt2;      
    double B3SOIFDwdvt0w;      
    double B3SOIFDwdvt1w;      
    double B3SOIFDwdvt2w;      
    double B3SOIFDwu0;
    double B3SOIFDwua;
    double B3SOIFDwub;
    double B3SOIFDwuc;
    double B3SOIFDwvsat;         
    double B3SOIFDwa0;   
    double B3SOIFDwags;      
    double B3SOIFDwb0;
    double B3SOIFDwb1;
    double B3SOIFDwketa;
    double B3SOIFDwabp;
    double B3SOIFDwmxc;
    double B3SOIFDwadice0;
    double B3SOIFDwa1;         
    double B3SOIFDwa2;         
    double B3SOIFDwrdsw;       
    double B3SOIFDwprwb;
    double B3SOIFDwprwg;
    double B3SOIFDwwr;
    double B3SOIFDwnfactor;      
    double B3SOIFDwdwg;
    double B3SOIFDwdwb;
    double B3SOIFDwvoff;
    double B3SOIFDweta0;         
    double B3SOIFDwetab;         
    double B3SOIFDwdsub;      
    double B3SOIFDwcit;           
    double B3SOIFDwcdsc;           
    double B3SOIFDwcdscb; 
    double B3SOIFDwcdscd;          
    double B3SOIFDwpclm;      
    double B3SOIFDwpdibl1;      
    double B3SOIFDwpdibl2;      
    double B3SOIFDwpdiblb;      
    double B3SOIFDwdrout;      
    double B3SOIFDwpvag;       
    double B3SOIFDwdelta;
    double B3SOIFDwaii;
    double B3SOIFDwbii;
    double B3SOIFDwcii;
    double B3SOIFDwdii;
    double B3SOIFDwalpha0;
    double B3SOIFDwalpha1;
    double B3SOIFDwbeta0;
    double B3SOIFDwagidl;
    double B3SOIFDwbgidl;
    double B3SOIFDwngidl;
    double B3SOIFDwntun;
    double B3SOIFDwndiode;
    double B3SOIFDwisbjt;       
    double B3SOIFDwisdif;
    double B3SOIFDwisrec;       
    double B3SOIFDwistun;       
    double B3SOIFDwedl;       
    double B3SOIFDwkbjt1;       
    /* CV model */
    double B3SOIFDwvsdfb;
    double B3SOIFDwvsdth;

    /* Cross-term Dependence */
    double B3SOIFDpnpeak;        
    double B3SOIFDpnsub;
    double B3SOIFDpngate;        
    double B3SOIFDpvth0;
    double B3SOIFDpk1;
    double B3SOIFDpk2;
    double B3SOIFDpk3;
    double B3SOIFDpk3b;
    double B3SOIFDpvbsa;
    double B3SOIFDpdelp;
    double B3SOIFDpkb1;
    double B3SOIFDpkb3;
    double B3SOIFDpdvbd0;
    double B3SOIFDpdvbd1;
    double B3SOIFDpw0;
    double B3SOIFDpnlx;
    double B3SOIFDpdvt0;      
    double B3SOIFDpdvt1;      
    double B3SOIFDpdvt2;      
    double B3SOIFDpdvt0w;      
    double B3SOIFDpdvt1w;      
    double B3SOIFDpdvt2w;      
    double B3SOIFDpu0;
    double B3SOIFDpua;
    double B3SOIFDpub;
    double B3SOIFDpuc;
    double B3SOIFDpvsat;         
    double B3SOIFDpa0;   
    double B3SOIFDpags;      
    double B3SOIFDpb0;
    double B3SOIFDpb1;
    double B3SOIFDpketa;
    double B3SOIFDpabp;
    double B3SOIFDpmxc;
    double B3SOIFDpadice0;
    double B3SOIFDpa1;         
    double B3SOIFDpa2;         
    double B3SOIFDprdsw;       
    double B3SOIFDpprwb;
    double B3SOIFDpprwg;
    double B3SOIFDpwr;
    double B3SOIFDpnfactor;      
    double B3SOIFDpdwg;
    double B3SOIFDpdwb;
    double B3SOIFDpvoff;
    double B3SOIFDpeta0;         
    double B3SOIFDpetab;         
    double B3SOIFDpdsub;      
    double B3SOIFDpcit;           
    double B3SOIFDpcdsc;           
    double B3SOIFDpcdscb; 
    double B3SOIFDpcdscd;          
    double B3SOIFDppclm;      
    double B3SOIFDppdibl1;      
    double B3SOIFDppdibl2;      
    double B3SOIFDppdiblb;      
    double B3SOIFDpdrout;      
    double B3SOIFDppvag;       
    double B3SOIFDpdelta;
    double B3SOIFDpaii;
    double B3SOIFDpbii;
    double B3SOIFDpcii;
    double B3SOIFDpdii;
    double B3SOIFDpalpha0;
    double B3SOIFDpalpha1;
    double B3SOIFDpbeta0;
    double B3SOIFDpagidl;
    double B3SOIFDpbgidl;
    double B3SOIFDpngidl;
    double B3SOIFDpntun;
    double B3SOIFDpndiode;
    double B3SOIFDpisbjt;       
    double B3SOIFDpisdif;
    double B3SOIFDpisrec;       
    double B3SOIFDpistun;       
    double B3SOIFDpedl;       
    double B3SOIFDpkbjt1;       
    /* CV model */
    double B3SOIFDpvsdfb;
    double B3SOIFDpvsdth;
/* Added for binning - END1 */

/* Pre-calculated constants */
    double B3SOIFDcbox;
    double B3SOIFDcsi;
    double B3SOIFDcsieff;
    double B3SOIFDcoxt;
    double B3SOIFDcboxt;
    double B3SOIFDcsit;
    double B3SOIFDnfb;
    double B3SOIFDadice;
    double B3SOIFDqsi;
    double B3SOIFDqsieff;
    double B3SOIFDeg0;

    /* MCJ: move to size-dependent param. */
    double B3SOIFDvtm;   
    double B3SOIFDcox;
    double B3SOIFDcof1;
    double B3SOIFDcof2;
    double B3SOIFDcof3;
    double B3SOIFDcof4;
    double B3SOIFDvcrit;
    double B3SOIFDfactor1;

    double B3SOIFDoxideTrapDensityA;      
    double B3SOIFDoxideTrapDensityB;     
    double B3SOIFDoxideTrapDensityC;  
    double B3SOIFDem;  
    double B3SOIFDef;  
    double B3SOIFDaf;  
    double B3SOIFDkf;  
    double B3SOIFDnoif;  

    struct b3soifdSizeDependParam *pSizeDependParamKnot;

    /* Flags */

    unsigned B3SOIFDtboxGiven:1;
    unsigned B3SOIFDtsiGiven :1;
    unsigned B3SOIFDxjGiven :1;
    unsigned B3SOIFDkb1Given :1;
    unsigned B3SOIFDkb3Given :1;
    unsigned B3SOIFDdvbd0Given :1;
    unsigned B3SOIFDdvbd1Given :1;
    unsigned B3SOIFDvbsaGiven :1;
    unsigned B3SOIFDdelpGiven :1;
    unsigned B3SOIFDrbodyGiven :1;
    unsigned B3SOIFDrbshGiven :1;
    unsigned B3SOIFDadice0Given :1;
    unsigned B3SOIFDabpGiven :1;
    unsigned B3SOIFDmxcGiven :1;
    unsigned B3SOIFDrth0Given :1;
    unsigned B3SOIFDcth0Given :1;
    unsigned B3SOIFDaiiGiven :1;
    unsigned B3SOIFDbiiGiven :1;
    unsigned B3SOIFDciiGiven :1;
    unsigned B3SOIFDdiiGiven :1;
    unsigned B3SOIFDngidlGiven :1;
    unsigned B3SOIFDagidlGiven :1;
    unsigned B3SOIFDbgidlGiven :1;
    unsigned B3SOIFDndiodeGiven :1;
    unsigned B3SOIFDntunGiven :1;
    unsigned B3SOIFDisbjtGiven :1;
    unsigned B3SOIFDisdifGiven :1;
    unsigned B3SOIFDisrecGiven :1;
    unsigned B3SOIFDistunGiven :1;
    unsigned B3SOIFDxbjtGiven :1;
    unsigned B3SOIFDxdifGiven :1;
    unsigned B3SOIFDxrecGiven :1;
    unsigned B3SOIFDxtunGiven :1;
    unsigned B3SOIFDedlGiven :1;
    unsigned B3SOIFDkbjt1Given :1;
    unsigned B3SOIFDttGiven :1;
    unsigned B3SOIFDvsdfbGiven :1;
    unsigned B3SOIFDvsdthGiven :1;
    unsigned B3SOIFDasdGiven :1;
    unsigned B3SOIFDcsdminGiven :1;

    unsigned  B3SOIFDmobModGiven :1;
    unsigned  B3SOIFDbinUnitGiven :1;
    unsigned  B3SOIFDcapModGiven :1;
    unsigned  B3SOIFDparamChkGiven :1;
    unsigned  B3SOIFDnoiModGiven :1;
    unsigned  B3SOIFDshModGiven :1;
    unsigned  B3SOIFDtypeGiven   :1;
    unsigned  B3SOIFDtoxGiven   :1;
    unsigned  B3SOIFDversionGiven   :1;

    unsigned  B3SOIFDcdscGiven   :1;
    unsigned  B3SOIFDcdscbGiven   :1;
    unsigned  B3SOIFDcdscdGiven   :1;
    unsigned  B3SOIFDcitGiven   :1;
    unsigned  B3SOIFDnfactorGiven   :1;
    unsigned  B3SOIFDvsatGiven   :1;
    unsigned  B3SOIFDatGiven   :1;
    unsigned  B3SOIFDa0Given   :1;
    unsigned  B3SOIFDagsGiven   :1;
    unsigned  B3SOIFDa1Given   :1;
    unsigned  B3SOIFDa2Given   :1;
    unsigned  B3SOIFDketaGiven   :1;    
    unsigned  B3SOIFDnsubGiven   :1;
    unsigned  B3SOIFDnpeakGiven   :1;
    unsigned  B3SOIFDngateGiven   :1;
    unsigned  B3SOIFDgamma1Given   :1;
    unsigned  B3SOIFDgamma2Given   :1;
    unsigned  B3SOIFDvbxGiven   :1;
    unsigned  B3SOIFDvbmGiven   :1;
    unsigned  B3SOIFDxtGiven   :1;
    unsigned  B3SOIFDk1Given   :1;
    unsigned  B3SOIFDkt1Given   :1;
    unsigned  B3SOIFDkt1lGiven   :1;
    unsigned  B3SOIFDkt2Given   :1;
    unsigned  B3SOIFDk2Given   :1;
    unsigned  B3SOIFDk3Given   :1;
    unsigned  B3SOIFDk3bGiven   :1;
    unsigned  B3SOIFDw0Given   :1;
    unsigned  B3SOIFDnlxGiven   :1;
    unsigned  B3SOIFDdvt0Given   :1;   
    unsigned  B3SOIFDdvt1Given   :1;     
    unsigned  B3SOIFDdvt2Given   :1;     
    unsigned  B3SOIFDdvt0wGiven   :1;   
    unsigned  B3SOIFDdvt1wGiven   :1;     
    unsigned  B3SOIFDdvt2wGiven   :1;     
    unsigned  B3SOIFDdroutGiven   :1;     
    unsigned  B3SOIFDdsubGiven   :1;     
    unsigned  B3SOIFDvth0Given   :1;
    unsigned  B3SOIFDuaGiven   :1;
    unsigned  B3SOIFDua1Given   :1;
    unsigned  B3SOIFDubGiven   :1;
    unsigned  B3SOIFDub1Given   :1;
    unsigned  B3SOIFDucGiven   :1;
    unsigned  B3SOIFDuc1Given   :1;
    unsigned  B3SOIFDu0Given   :1;
    unsigned  B3SOIFDuteGiven   :1;
    unsigned  B3SOIFDvoffGiven   :1;
    unsigned  B3SOIFDrdswGiven   :1;      
    unsigned  B3SOIFDprwgGiven   :1;      
    unsigned  B3SOIFDprwbGiven   :1;      
    unsigned  B3SOIFDprtGiven   :1;      
    unsigned  B3SOIFDeta0Given   :1;    
    unsigned  B3SOIFDetabGiven   :1;    
    unsigned  B3SOIFDpclmGiven   :1;   
    unsigned  B3SOIFDpdibl1Given   :1;   
    unsigned  B3SOIFDpdibl2Given   :1;  
    unsigned  B3SOIFDpdiblbGiven   :1;  
    unsigned  B3SOIFDpvagGiven   :1;    
    unsigned  B3SOIFDdeltaGiven  :1;     
    unsigned  B3SOIFDwrGiven   :1;
    unsigned  B3SOIFDdwgGiven   :1;
    unsigned  B3SOIFDdwbGiven   :1;
    unsigned  B3SOIFDb0Given   :1;
    unsigned  B3SOIFDb1Given   :1;
    unsigned  B3SOIFDalpha0Given   :1;
    unsigned  B3SOIFDalpha1Given   :1;
    unsigned  B3SOIFDbeta0Given   :1;

    /* CV model */
    unsigned  B3SOIFDcgslGiven   :1;
    unsigned  B3SOIFDcgdlGiven   :1;
    unsigned  B3SOIFDckappaGiven   :1;
    unsigned  B3SOIFDcfGiven   :1;
    unsigned  B3SOIFDclcGiven   :1;
    unsigned  B3SOIFDcleGiven   :1;
    unsigned  B3SOIFDdwcGiven   :1;
    unsigned  B3SOIFDdlcGiven   :1;

/* Added for binning - START2 */
    /* Length Dependence */
    unsigned  B3SOIFDlnpeakGiven   :1;        
    unsigned  B3SOIFDlnsubGiven   :1;
    unsigned  B3SOIFDlngateGiven   :1;        
    unsigned  B3SOIFDlvth0Given   :1;
    unsigned  B3SOIFDlk1Given   :1;
    unsigned  B3SOIFDlk2Given   :1;
    unsigned  B3SOIFDlk3Given   :1;
    unsigned  B3SOIFDlk3bGiven   :1;
    unsigned  B3SOIFDlvbsaGiven   :1;
    unsigned  B3SOIFDldelpGiven   :1;
    unsigned  B3SOIFDlkb1Given   :1;
    unsigned  B3SOIFDlkb3Given   :1;
    unsigned  B3SOIFDldvbd0Given   :1;
    unsigned  B3SOIFDldvbd1Given   :1;
    unsigned  B3SOIFDlw0Given   :1;
    unsigned  B3SOIFDlnlxGiven   :1;
    unsigned  B3SOIFDldvt0Given   :1;      
    unsigned  B3SOIFDldvt1Given   :1;      
    unsigned  B3SOIFDldvt2Given   :1;      
    unsigned  B3SOIFDldvt0wGiven   :1;      
    unsigned  B3SOIFDldvt1wGiven   :1;      
    unsigned  B3SOIFDldvt2wGiven   :1;      
    unsigned  B3SOIFDlu0Given   :1;
    unsigned  B3SOIFDluaGiven   :1;
    unsigned  B3SOIFDlubGiven   :1;
    unsigned  B3SOIFDlucGiven   :1;
    unsigned  B3SOIFDlvsatGiven   :1;         
    unsigned  B3SOIFDla0Given   :1;   
    unsigned  B3SOIFDlagsGiven   :1;      
    unsigned  B3SOIFDlb0Given   :1;
    unsigned  B3SOIFDlb1Given   :1;
    unsigned  B3SOIFDlketaGiven   :1;
    unsigned  B3SOIFDlabpGiven   :1;
    unsigned  B3SOIFDlmxcGiven   :1;
    unsigned  B3SOIFDladice0Given   :1;
    unsigned  B3SOIFDla1Given   :1;         
    unsigned  B3SOIFDla2Given   :1;         
    unsigned  B3SOIFDlrdswGiven   :1;       
    unsigned  B3SOIFDlprwbGiven   :1;
    unsigned  B3SOIFDlprwgGiven   :1;
    unsigned  B3SOIFDlwrGiven   :1;
    unsigned  B3SOIFDlnfactorGiven   :1;      
    unsigned  B3SOIFDldwgGiven   :1;
    unsigned  B3SOIFDldwbGiven   :1;
    unsigned  B3SOIFDlvoffGiven   :1;
    unsigned  B3SOIFDleta0Given   :1;         
    unsigned  B3SOIFDletabGiven   :1;         
    unsigned  B3SOIFDldsubGiven   :1;      
    unsigned  B3SOIFDlcitGiven   :1;           
    unsigned  B3SOIFDlcdscGiven   :1;           
    unsigned  B3SOIFDlcdscbGiven   :1; 
    unsigned  B3SOIFDlcdscdGiven   :1;          
    unsigned  B3SOIFDlpclmGiven   :1;      
    unsigned  B3SOIFDlpdibl1Given   :1;      
    unsigned  B3SOIFDlpdibl2Given   :1;      
    unsigned  B3SOIFDlpdiblbGiven   :1;      
    unsigned  B3SOIFDldroutGiven   :1;      
    unsigned  B3SOIFDlpvagGiven   :1;       
    unsigned  B3SOIFDldeltaGiven   :1;
    unsigned  B3SOIFDlaiiGiven   :1;
    unsigned  B3SOIFDlbiiGiven   :1;
    unsigned  B3SOIFDlciiGiven   :1;
    unsigned  B3SOIFDldiiGiven   :1;
    unsigned  B3SOIFDlalpha0Given   :1;
    unsigned  B3SOIFDlalpha1Given   :1;
    unsigned  B3SOIFDlbeta0Given   :1;
    unsigned  B3SOIFDlagidlGiven   :1;
    unsigned  B3SOIFDlbgidlGiven   :1;
    unsigned  B3SOIFDlngidlGiven   :1;
    unsigned  B3SOIFDlntunGiven   :1;
    unsigned  B3SOIFDlndiodeGiven   :1;
    unsigned  B3SOIFDlisbjtGiven   :1;       
    unsigned  B3SOIFDlisdifGiven   :1;
    unsigned  B3SOIFDlisrecGiven   :1;       
    unsigned  B3SOIFDlistunGiven   :1;       
    unsigned  B3SOIFDledlGiven   :1;       
    unsigned  B3SOIFDlkbjt1Given   :1;       
    /* CV model */
    unsigned  B3SOIFDlvsdfbGiven   :1;
    unsigned  B3SOIFDlvsdthGiven   :1;

    /* Width Dependence */
    unsigned  B3SOIFDwnpeakGiven   :1;        
    unsigned  B3SOIFDwnsubGiven   :1;
    unsigned  B3SOIFDwngateGiven   :1;        
    unsigned  B3SOIFDwvth0Given   :1;
    unsigned  B3SOIFDwk1Given   :1;
    unsigned  B3SOIFDwk2Given   :1;
    unsigned  B3SOIFDwk3Given   :1;
    unsigned  B3SOIFDwk3bGiven   :1;
    unsigned  B3SOIFDwvbsaGiven   :1;
    unsigned  B3SOIFDwdelpGiven   :1;
    unsigned  B3SOIFDwkb1Given   :1;
    unsigned  B3SOIFDwkb3Given   :1;
    unsigned  B3SOIFDwdvbd0Given   :1;
    unsigned  B3SOIFDwdvbd1Given   :1;
    unsigned  B3SOIFDww0Given   :1;
    unsigned  B3SOIFDwnlxGiven   :1;
    unsigned  B3SOIFDwdvt0Given   :1;      
    unsigned  B3SOIFDwdvt1Given   :1;      
    unsigned  B3SOIFDwdvt2Given   :1;      
    unsigned  B3SOIFDwdvt0wGiven   :1;      
    unsigned  B3SOIFDwdvt1wGiven   :1;      
    unsigned  B3SOIFDwdvt2wGiven   :1;      
    unsigned  B3SOIFDwu0Given   :1;
    unsigned  B3SOIFDwuaGiven   :1;
    unsigned  B3SOIFDwubGiven   :1;
    unsigned  B3SOIFDwucGiven   :1;
    unsigned  B3SOIFDwvsatGiven   :1;         
    unsigned  B3SOIFDwa0Given   :1;   
    unsigned  B3SOIFDwagsGiven   :1;      
    unsigned  B3SOIFDwb0Given   :1;
    unsigned  B3SOIFDwb1Given   :1;
    unsigned  B3SOIFDwketaGiven   :1;
    unsigned  B3SOIFDwabpGiven   :1;
    unsigned  B3SOIFDwmxcGiven   :1;
    unsigned  B3SOIFDwadice0Given   :1;
    unsigned  B3SOIFDwa1Given   :1;         
    unsigned  B3SOIFDwa2Given   :1;         
    unsigned  B3SOIFDwrdswGiven   :1;       
    unsigned  B3SOIFDwprwbGiven   :1;
    unsigned  B3SOIFDwprwgGiven   :1;
    unsigned  B3SOIFDwwrGiven   :1;
    unsigned  B3SOIFDwnfactorGiven   :1;      
    unsigned  B3SOIFDwdwgGiven   :1;
    unsigned  B3SOIFDwdwbGiven   :1;
    unsigned  B3SOIFDwvoffGiven   :1;
    unsigned  B3SOIFDweta0Given   :1;         
    unsigned  B3SOIFDwetabGiven   :1;         
    unsigned  B3SOIFDwdsubGiven   :1;      
    unsigned  B3SOIFDwcitGiven   :1;           
    unsigned  B3SOIFDwcdscGiven   :1;           
    unsigned  B3SOIFDwcdscbGiven   :1; 
    unsigned  B3SOIFDwcdscdGiven   :1;          
    unsigned  B3SOIFDwpclmGiven   :1;      
    unsigned  B3SOIFDwpdibl1Given   :1;      
    unsigned  B3SOIFDwpdibl2Given   :1;      
    unsigned  B3SOIFDwpdiblbGiven   :1;      
    unsigned  B3SOIFDwdroutGiven   :1;      
    unsigned  B3SOIFDwpvagGiven   :1;       
    unsigned  B3SOIFDwdeltaGiven   :1;
    unsigned  B3SOIFDwaiiGiven   :1;
    unsigned  B3SOIFDwbiiGiven   :1;
    unsigned  B3SOIFDwciiGiven   :1;
    unsigned  B3SOIFDwdiiGiven   :1;
    unsigned  B3SOIFDwalpha0Given   :1;
    unsigned  B3SOIFDwalpha1Given   :1;
    unsigned  B3SOIFDwbeta0Given   :1;
    unsigned  B3SOIFDwagidlGiven   :1;
    unsigned  B3SOIFDwbgidlGiven   :1;
    unsigned  B3SOIFDwngidlGiven   :1;
    unsigned  B3SOIFDwntunGiven   :1;
    unsigned  B3SOIFDwndiodeGiven   :1;
    unsigned  B3SOIFDwisbjtGiven   :1;       
    unsigned  B3SOIFDwisdifGiven   :1;
    unsigned  B3SOIFDwisrecGiven   :1;       
    unsigned  B3SOIFDwistunGiven   :1;       
    unsigned  B3SOIFDwedlGiven   :1;       
    unsigned  B3SOIFDwkbjt1Given   :1;       
    /* CV model */
    unsigned  B3SOIFDwvsdfbGiven   :1;
    unsigned  B3SOIFDwvsdthGiven   :1;

    /* Cross-term Dependence */
    unsigned  B3SOIFDpnpeakGiven   :1;        
    unsigned  B3SOIFDpnsubGiven   :1;
    unsigned  B3SOIFDpngateGiven   :1;        
    unsigned  B3SOIFDpvth0Given   :1;
    unsigned  B3SOIFDpk1Given   :1;
    unsigned  B3SOIFDpk2Given   :1;
    unsigned  B3SOIFDpk3Given   :1;
    unsigned  B3SOIFDpk3bGiven   :1;
    unsigned  B3SOIFDpvbsaGiven   :1;
    unsigned  B3SOIFDpdelpGiven   :1;
    unsigned  B3SOIFDpkb1Given   :1;
    unsigned  B3SOIFDpkb3Given   :1;
    unsigned  B3SOIFDpdvbd0Given   :1;
    unsigned  B3SOIFDpdvbd1Given   :1;
    unsigned  B3SOIFDpw0Given   :1;
    unsigned  B3SOIFDpnlxGiven   :1;
    unsigned  B3SOIFDpdvt0Given   :1;      
    unsigned  B3SOIFDpdvt1Given   :1;      
    unsigned  B3SOIFDpdvt2Given   :1;      
    unsigned  B3SOIFDpdvt0wGiven   :1;      
    unsigned  B3SOIFDpdvt1wGiven   :1;      
    unsigned  B3SOIFDpdvt2wGiven   :1;      
    unsigned  B3SOIFDpu0Given   :1;
    unsigned  B3SOIFDpuaGiven   :1;
    unsigned  B3SOIFDpubGiven   :1;
    unsigned  B3SOIFDpucGiven   :1;
    unsigned  B3SOIFDpvsatGiven   :1;         
    unsigned  B3SOIFDpa0Given   :1;   
    unsigned  B3SOIFDpagsGiven   :1;      
    unsigned  B3SOIFDpb0Given   :1;
    unsigned  B3SOIFDpb1Given   :1;
    unsigned  B3SOIFDpketaGiven   :1;
    unsigned  B3SOIFDpabpGiven   :1;
    unsigned  B3SOIFDpmxcGiven   :1;
    unsigned  B3SOIFDpadice0Given   :1;
    unsigned  B3SOIFDpa1Given   :1;         
    unsigned  B3SOIFDpa2Given   :1;         
    unsigned  B3SOIFDprdswGiven   :1;       
    unsigned  B3SOIFDpprwbGiven   :1;
    unsigned  B3SOIFDpprwgGiven   :1;
    unsigned  B3SOIFDpwrGiven   :1;
    unsigned  B3SOIFDpnfactorGiven   :1;      
    unsigned  B3SOIFDpdwgGiven   :1;
    unsigned  B3SOIFDpdwbGiven   :1;
    unsigned  B3SOIFDpvoffGiven   :1;
    unsigned  B3SOIFDpeta0Given   :1;         
    unsigned  B3SOIFDpetabGiven   :1;         
    unsigned  B3SOIFDpdsubGiven   :1;      
    unsigned  B3SOIFDpcitGiven   :1;           
    unsigned  B3SOIFDpcdscGiven   :1;           
    unsigned  B3SOIFDpcdscbGiven   :1; 
    unsigned  B3SOIFDpcdscdGiven   :1;          
    unsigned  B3SOIFDppclmGiven   :1;      
    unsigned  B3SOIFDppdibl1Given   :1;      
    unsigned  B3SOIFDppdibl2Given   :1;      
    unsigned  B3SOIFDppdiblbGiven   :1;      
    unsigned  B3SOIFDpdroutGiven   :1;      
    unsigned  B3SOIFDppvagGiven   :1;       
    unsigned  B3SOIFDpdeltaGiven   :1;
    unsigned  B3SOIFDpaiiGiven   :1;
    unsigned  B3SOIFDpbiiGiven   :1;
    unsigned  B3SOIFDpciiGiven   :1;
    unsigned  B3SOIFDpdiiGiven   :1;
    unsigned  B3SOIFDpalpha0Given   :1;
    unsigned  B3SOIFDpalpha1Given   :1;
    unsigned  B3SOIFDpbeta0Given   :1;
    unsigned  B3SOIFDpagidlGiven   :1;
    unsigned  B3SOIFDpbgidlGiven   :1;
    unsigned  B3SOIFDpngidlGiven   :1;
    unsigned  B3SOIFDpntunGiven   :1;
    unsigned  B3SOIFDpndiodeGiven   :1;
    unsigned  B3SOIFDpisbjtGiven   :1;       
    unsigned  B3SOIFDpisdifGiven   :1;
    unsigned  B3SOIFDpisrecGiven   :1;       
    unsigned  B3SOIFDpistunGiven   :1;       
    unsigned  B3SOIFDpedlGiven   :1;       
    unsigned  B3SOIFDpkbjt1Given   :1;       
    /* CV model */
    unsigned  B3SOIFDpvsdfbGiven   :1;
    unsigned  B3SOIFDpvsdthGiven   :1;
/* Added for binning - END2 */

    unsigned  B3SOIFDuseFringeGiven   :1;

    unsigned  B3SOIFDtnomGiven   :1;
    unsigned  B3SOIFDcgsoGiven   :1;
    unsigned  B3SOIFDcgdoGiven   :1;
    unsigned  B3SOIFDcgeoGiven   :1;
    unsigned  B3SOIFDxpartGiven   :1;
    unsigned  B3SOIFDsheetResistanceGiven   :1;
    unsigned  B3SOIFDGatesidewallJctPotentialGiven   :1;
    unsigned  B3SOIFDbodyJctGateSideGradingCoeffGiven   :1;
    unsigned  B3SOIFDunitLengthGateSidewallJctCapGiven   :1;
    unsigned  B3SOIFDcsdeswGiven :1;

    unsigned  B3SOIFDoxideTrapDensityAGiven  :1;         
    unsigned  B3SOIFDoxideTrapDensityBGiven  :1;        
    unsigned  B3SOIFDoxideTrapDensityCGiven  :1;     
    unsigned  B3SOIFDemGiven  :1;     
    unsigned  B3SOIFDefGiven  :1;     
    unsigned  B3SOIFDafGiven  :1;     
    unsigned  B3SOIFDkfGiven  :1;     
    unsigned  B3SOIFDnoifGiven  :1;     

    unsigned  B3SOIFDLintGiven   :1;
    unsigned  B3SOIFDLlGiven   :1;
    unsigned  B3SOIFDLlnGiven   :1;
    unsigned  B3SOIFDLwGiven   :1;
    unsigned  B3SOIFDLwnGiven   :1;
    unsigned  B3SOIFDLwlGiven   :1;
    unsigned  B3SOIFDLminGiven   :1;
    unsigned  B3SOIFDLmaxGiven   :1;

    unsigned  B3SOIFDWintGiven   :1;
    unsigned  B3SOIFDWlGiven   :1;
    unsigned  B3SOIFDWlnGiven   :1;
    unsigned  B3SOIFDWwGiven   :1;
    unsigned  B3SOIFDWwnGiven   :1;
    unsigned  B3SOIFDWwlGiven   :1;
    unsigned  B3SOIFDWminGiven   :1;
    unsigned  B3SOIFDWmaxGiven   :1;

} B3SOIFDmodel;


#ifndef NMOS
#define NMOS 1
#define PMOS -1
#endif /*NMOS*/


/* device parameters */
#define B3SOIFD_W 1
#define B3SOIFD_L 2
#define B3SOIFD_M 22
#define B3SOIFD_AS 3
#define B3SOIFD_AD 4
#define B3SOIFD_PS 5
#define B3SOIFD_PD 6
#define B3SOIFD_NRS 7
#define B3SOIFD_NRD 8
#define B3SOIFD_OFF 9
#define B3SOIFD_IC_VBS 10
#define B3SOIFD_IC_VDS 11
#define B3SOIFD_IC_VGS 12
#define B3SOIFD_IC_VES 13
#define B3SOIFD_IC_VPS 14
#define B3SOIFD_BJTOFF 15
#define B3SOIFD_RTH0 16
#define B3SOIFD_CTH0 17
#define B3SOIFD_NRB 18
#define B3SOIFD_IC 19
#define B3SOIFD_NQSMOD 20
#define B3SOIFD_DEBUG  21

/* model parameters */
#define B3SOIFD_MOD_CAPMOD          101
#define B3SOIFD_MOD_NQSMOD          102
#define B3SOIFD_MOD_MOBMOD          103    
#define B3SOIFD_MOD_NOIMOD          104    
#define B3SOIFD_MOD_SHMOD           105
#define B3SOIFD_MOD_DDMOD           106

#define B3SOIFD_MOD_TOX             107



#define B3SOIFD_MOD_CDSC            108
#define B3SOIFD_MOD_CDSCB           109
#define B3SOIFD_MOD_CIT             110
#define B3SOIFD_MOD_NFACTOR         111
#define B3SOIFD_MOD_XJ              112
#define B3SOIFD_MOD_VSAT            113
#define B3SOIFD_MOD_AT              114
#define B3SOIFD_MOD_A0              115
#define B3SOIFD_MOD_A1              116
#define B3SOIFD_MOD_A2              117
#define B3SOIFD_MOD_KETA            118   
#define B3SOIFD_MOD_NSUB            119
#define B3SOIFD_MOD_NPEAK           120
#define B3SOIFD_MOD_NGATE           121
#define B3SOIFD_MOD_GAMMA1          122
#define B3SOIFD_MOD_GAMMA2          123
#define B3SOIFD_MOD_VBX             124
#define B3SOIFD_MOD_BINUNIT         125    

#define B3SOIFD_MOD_VBM             126

#define B3SOIFD_MOD_XT              127
#define B3SOIFD_MOD_K1              129
#define B3SOIFD_MOD_KT1             130
#define B3SOIFD_MOD_KT1L            131
#define B3SOIFD_MOD_K2              132
#define B3SOIFD_MOD_KT2             133
#define B3SOIFD_MOD_K3              134
#define B3SOIFD_MOD_K3B             135
#define B3SOIFD_MOD_W0              136
#define B3SOIFD_MOD_NLX             137

#define B3SOIFD_MOD_DVT0            138
#define B3SOIFD_MOD_DVT1            139
#define B3SOIFD_MOD_DVT2            140

#define B3SOIFD_MOD_DVT0W           141
#define B3SOIFD_MOD_DVT1W           142
#define B3SOIFD_MOD_DVT2W           143

#define B3SOIFD_MOD_DROUT           144
#define B3SOIFD_MOD_DSUB            145
#define B3SOIFD_MOD_VTH0            146
#define B3SOIFD_MOD_UA              147
#define B3SOIFD_MOD_UA1             148
#define B3SOIFD_MOD_UB              149
#define B3SOIFD_MOD_UB1             150
#define B3SOIFD_MOD_UC              151
#define B3SOIFD_MOD_UC1             152
#define B3SOIFD_MOD_U0              153
#define B3SOIFD_MOD_UTE             154
#define B3SOIFD_MOD_VOFF            155
#define B3SOIFD_MOD_DELTA           156
#define B3SOIFD_MOD_RDSW            157
#define B3SOIFD_MOD_PRT             158
#define B3SOIFD_MOD_LDD             159
#define B3SOIFD_MOD_ETA             160
#define B3SOIFD_MOD_ETA0            161
#define B3SOIFD_MOD_ETAB            162
#define B3SOIFD_MOD_PCLM            163
#define B3SOIFD_MOD_PDIBL1          164
#define B3SOIFD_MOD_PDIBL2          165
#define B3SOIFD_MOD_PSCBE1          166
#define B3SOIFD_MOD_PSCBE2          167
#define B3SOIFD_MOD_PVAG            168
#define B3SOIFD_MOD_WR              169
#define B3SOIFD_MOD_DWG             170
#define B3SOIFD_MOD_DWB             171
#define B3SOIFD_MOD_B0              172
#define B3SOIFD_MOD_B1              173
#define B3SOIFD_MOD_ALPHA0          174
#define B3SOIFD_MOD_BETA0           175
#define B3SOIFD_MOD_PDIBLB          178

#define B3SOIFD_MOD_PRWG            179
#define B3SOIFD_MOD_PRWB            180

#define B3SOIFD_MOD_CDSCD           181
#define B3SOIFD_MOD_AGS             182

#define B3SOIFD_MOD_FRINGE          184
#define B3SOIFD_MOD_CGSL            186
#define B3SOIFD_MOD_CGDL            187
#define B3SOIFD_MOD_CKAPPA          188
#define B3SOIFD_MOD_CF              189
#define B3SOIFD_MOD_CLC             190
#define B3SOIFD_MOD_CLE             191
#define B3SOIFD_MOD_PARAMCHK        192
#define B3SOIFD_MOD_VERSION         193

#define B3SOIFD_MOD_TBOX            195
#define B3SOIFD_MOD_TSI             196
#define B3SOIFD_MOD_KB1             197
#define B3SOIFD_MOD_KB3             198
#define B3SOIFD_MOD_DVBD0           199
#define B3SOIFD_MOD_DVBD1           200
#define B3SOIFD_MOD_DELP            201
#define B3SOIFD_MOD_VBSA            202
#define B3SOIFD_MOD_RBODY           204
#define B3SOIFD_MOD_ADICE0          205
#define B3SOIFD_MOD_ABP             206
#define B3SOIFD_MOD_MXC             207
#define B3SOIFD_MOD_RTH0            208
#define B3SOIFD_MOD_CTH0            209
#define B3SOIFD_MOD_AII             210
#define B3SOIFD_MOD_BII             211
#define B3SOIFD_MOD_CII             212
#define B3SOIFD_MOD_DII             213
#define B3SOIFD_MOD_ALPHA1	  214
#define B3SOIFD_MOD_NGIDL           215
#define B3SOIFD_MOD_AGIDL           216
#define B3SOIFD_MOD_BGIDL           217
#define B3SOIFD_MOD_NDIODE          218
#define B3SOIFD_MOD_LDIOF           219
#define B3SOIFD_MOD_LDIOR           220
#define B3SOIFD_MOD_NTUN            221
#define B3SOIFD_MOD_ISBJT           222
#define B3SOIFD_MOD_ISDIF           223
#define B3SOIFD_MOD_ISREC           224
#define B3SOIFD_MOD_ISTUN           225
#define B3SOIFD_MOD_XBJT            226
#define B3SOIFD_MOD_XDIF            227
#define B3SOIFD_MOD_XREC            228
#define B3SOIFD_MOD_XTUN            229
#define B3SOIFD_MOD_EDL             230
#define B3SOIFD_MOD_KBJT1           231
#define B3SOIFD_MOD_TT              232
#define B3SOIFD_MOD_VSDTH           233
#define B3SOIFD_MOD_VSDFB           234
#define B3SOIFD_MOD_ASD             235
#define B3SOIFD_MOD_CSDMIN          236
#define B3SOIFD_MOD_RBSH            237

/* Added for binning - START3 */
/* Length dependence */
#define B3SOIFD_MOD_LNPEAK           301
#define B3SOIFD_MOD_LNSUB            302
#define B3SOIFD_MOD_LNGATE           303
#define B3SOIFD_MOD_LVTH0            304
#define B3SOIFD_MOD_LK1              305     
#define B3SOIFD_MOD_LK2              306
#define B3SOIFD_MOD_LK3              307
#define B3SOIFD_MOD_LK3B             308
#define B3SOIFD_MOD_LVBSA            309
#define B3SOIFD_MOD_LDELP            310
#define B3SOIFD_MOD_LKB1             311
#define B3SOIFD_MOD_LKB3             312
#define B3SOIFD_MOD_LDVBD0           313
#define B3SOIFD_MOD_LDVBD1           314
#define B3SOIFD_MOD_LW0              315
#define B3SOIFD_MOD_LNLX             316
#define B3SOIFD_MOD_LDVT0            317
#define B3SOIFD_MOD_LDVT1            318
#define B3SOIFD_MOD_LDVT2            319
#define B3SOIFD_MOD_LDVT0W           320
#define B3SOIFD_MOD_LDVT1W           321
#define B3SOIFD_MOD_LDVT2W           322
#define B3SOIFD_MOD_LU0              323
#define B3SOIFD_MOD_LUA              324
#define B3SOIFD_MOD_LUB              325
#define B3SOIFD_MOD_LUC              326
#define B3SOIFD_MOD_LVSAT            327
#define B3SOIFD_MOD_LA0              328
#define B3SOIFD_MOD_LAGS             329
#define B3SOIFD_MOD_LB0              330
#define B3SOIFD_MOD_LB1              331
#define B3SOIFD_MOD_LKETA            332   
#define B3SOIFD_MOD_LABP             333   
#define B3SOIFD_MOD_LMXC             334   
#define B3SOIFD_MOD_LADICE0          335   
#define B3SOIFD_MOD_LA1              336
#define B3SOIFD_MOD_LA2              337
#define B3SOIFD_MOD_LRDSW            338
#define B3SOIFD_MOD_LPRWB            339
#define B3SOIFD_MOD_LPRWG            340
#define B3SOIFD_MOD_LWR              341
#define B3SOIFD_MOD_LNFACTOR         342
#define B3SOIFD_MOD_LDWG             343
#define B3SOIFD_MOD_LDWB             344
#define B3SOIFD_MOD_LVOFF            345
#define B3SOIFD_MOD_LETA0            346
#define B3SOIFD_MOD_LETAB            347
#define B3SOIFD_MOD_LDSUB            348
#define B3SOIFD_MOD_LCIT             349
#define B3SOIFD_MOD_LCDSC            350
#define B3SOIFD_MOD_LCDSCB           351
#define B3SOIFD_MOD_LCDSCD           352
#define B3SOIFD_MOD_LPCLM            353
#define B3SOIFD_MOD_LPDIBL1          354
#define B3SOIFD_MOD_LPDIBL2          355
#define B3SOIFD_MOD_LPDIBLB          356
#define B3SOIFD_MOD_LDROUT           357
#define B3SOIFD_MOD_LPVAG            358
#define B3SOIFD_MOD_LDELTA           359
#define B3SOIFD_MOD_LAII             360
#define B3SOIFD_MOD_LBII             361
#define B3SOIFD_MOD_LCII             362
#define B3SOIFD_MOD_LDII             363
#define B3SOIFD_MOD_LALPHA0          364
#define B3SOIFD_MOD_LALPHA1          365
#define B3SOIFD_MOD_LBETA0           366
#define B3SOIFD_MOD_LAGIDL           367
#define B3SOIFD_MOD_LBGIDL           368
#define B3SOIFD_MOD_LNGIDL           369
#define B3SOIFD_MOD_LNTUN            370
#define B3SOIFD_MOD_LNDIODE          371
#define B3SOIFD_MOD_LISBJT           372
#define B3SOIFD_MOD_LISDIF           373
#define B3SOIFD_MOD_LISREC           374
#define B3SOIFD_MOD_LISTUN           375
#define B3SOIFD_MOD_LEDL             376
#define B3SOIFD_MOD_LKBJT1           377
#define B3SOIFD_MOD_LVSDFB           378
#define B3SOIFD_MOD_LVSDTH           379

/* Width dependence */
#define B3SOIFD_MOD_WNPEAK           401
#define B3SOIFD_MOD_WNSUB            402
#define B3SOIFD_MOD_WNGATE           403
#define B3SOIFD_MOD_WVTH0            404
#define B3SOIFD_MOD_WK1              405     
#define B3SOIFD_MOD_WK2              406
#define B3SOIFD_MOD_WK3              407
#define B3SOIFD_MOD_WK3B             408
#define B3SOIFD_MOD_WVBSA            409
#define B3SOIFD_MOD_WDELP            410
#define B3SOIFD_MOD_WKB1             411
#define B3SOIFD_MOD_WKB3             412
#define B3SOIFD_MOD_WDVBD0           413
#define B3SOIFD_MOD_WDVBD1           414
#define B3SOIFD_MOD_WW0              415
#define B3SOIFD_MOD_WNLX             416
#define B3SOIFD_MOD_WDVT0            417
#define B3SOIFD_MOD_WDVT1            418
#define B3SOIFD_MOD_WDVT2            419
#define B3SOIFD_MOD_WDVT0W           420
#define B3SOIFD_MOD_WDVT1W           421
#define B3SOIFD_MOD_WDVT2W           422
#define B3SOIFD_MOD_WU0              423
#define B3SOIFD_MOD_WUA              424
#define B3SOIFD_MOD_WUB              425
#define B3SOIFD_MOD_WUC              426
#define B3SOIFD_MOD_WVSAT            427
#define B3SOIFD_MOD_WA0              428
#define B3SOIFD_MOD_WAGS             429
#define B3SOIFD_MOD_WB0              430
#define B3SOIFD_MOD_WB1              431
#define B3SOIFD_MOD_WKETA            432   
#define B3SOIFD_MOD_WABP             433   
#define B3SOIFD_MOD_WMXC             434   
#define B3SOIFD_MOD_WADICE0          435   
#define B3SOIFD_MOD_WA1              436
#define B3SOIFD_MOD_WA2              437
#define B3SOIFD_MOD_WRDSW            438
#define B3SOIFD_MOD_WPRWB            439
#define B3SOIFD_MOD_WPRWG            440
#define B3SOIFD_MOD_WWR              441
#define B3SOIFD_MOD_WNFACTOR         442
#define B3SOIFD_MOD_WDWG             443
#define B3SOIFD_MOD_WDWB             444
#define B3SOIFD_MOD_WVOFF            445
#define B3SOIFD_MOD_WETA0            446
#define B3SOIFD_MOD_WETAB            447
#define B3SOIFD_MOD_WDSUB            448
#define B3SOIFD_MOD_WCIT             449
#define B3SOIFD_MOD_WCDSC            450
#define B3SOIFD_MOD_WCDSCB           451
#define B3SOIFD_MOD_WCDSCD           452
#define B3SOIFD_MOD_WPCLM            453
#define B3SOIFD_MOD_WPDIBL1          454
#define B3SOIFD_MOD_WPDIBL2          455
#define B3SOIFD_MOD_WPDIBLB          456
#define B3SOIFD_MOD_WDROUT           457
#define B3SOIFD_MOD_WPVAG            458
#define B3SOIFD_MOD_WDELTA           459
#define B3SOIFD_MOD_WAII             460
#define B3SOIFD_MOD_WBII             461
#define B3SOIFD_MOD_WCII             462
#define B3SOIFD_MOD_WDII             463
#define B3SOIFD_MOD_WALPHA0          464
#define B3SOIFD_MOD_WALPHA1          465
#define B3SOIFD_MOD_WBETA0           466
#define B3SOIFD_MOD_WAGIDL           467
#define B3SOIFD_MOD_WBGIDL           468
#define B3SOIFD_MOD_WNGIDL           469
#define B3SOIFD_MOD_WNTUN            470
#define B3SOIFD_MOD_WNDIODE          471
#define B3SOIFD_MOD_WISBJT           472
#define B3SOIFD_MOD_WISDIF           473
#define B3SOIFD_MOD_WISREC           474
#define B3SOIFD_MOD_WISTUN           475
#define B3SOIFD_MOD_WEDL             476
#define B3SOIFD_MOD_WKBJT1           477
#define B3SOIFD_MOD_WVSDFB           478
#define B3SOIFD_MOD_WVSDTH           479

/* Cross-term dependence */
#define B3SOIFD_MOD_PNPEAK           501
#define B3SOIFD_MOD_PNSUB            502
#define B3SOIFD_MOD_PNGATE           503
#define B3SOIFD_MOD_PVTH0            504
#define B3SOIFD_MOD_PK1              505     
#define B3SOIFD_MOD_PK2              506
#define B3SOIFD_MOD_PK3              507
#define B3SOIFD_MOD_PK3B             508
#define B3SOIFD_MOD_PVBSA            509
#define B3SOIFD_MOD_PDELP            510
#define B3SOIFD_MOD_PKB1             511
#define B3SOIFD_MOD_PKB3             512
#define B3SOIFD_MOD_PDVBD0           513
#define B3SOIFD_MOD_PDVBD1           514
#define B3SOIFD_MOD_PW0              515
#define B3SOIFD_MOD_PNLX             516
#define B3SOIFD_MOD_PDVT0            517
#define B3SOIFD_MOD_PDVT1            518
#define B3SOIFD_MOD_PDVT2            519
#define B3SOIFD_MOD_PDVT0W           520
#define B3SOIFD_MOD_PDVT1W           521
#define B3SOIFD_MOD_PDVT2W           522
#define B3SOIFD_MOD_PU0              523
#define B3SOIFD_MOD_PUA              524
#define B3SOIFD_MOD_PUB              525
#define B3SOIFD_MOD_PUC              526
#define B3SOIFD_MOD_PVSAT            527
#define B3SOIFD_MOD_PA0              528
#define B3SOIFD_MOD_PAGS             529
#define B3SOIFD_MOD_PB0              530
#define B3SOIFD_MOD_PB1              531
#define B3SOIFD_MOD_PKETA            532   
#define B3SOIFD_MOD_PABP             533   
#define B3SOIFD_MOD_PMXC             534   
#define B3SOIFD_MOD_PADICE0          535   
#define B3SOIFD_MOD_PA1              536
#define B3SOIFD_MOD_PA2              537
#define B3SOIFD_MOD_PRDSW            538
#define B3SOIFD_MOD_PPRWB            539
#define B3SOIFD_MOD_PPRWG            540
#define B3SOIFD_MOD_PWR              541
#define B3SOIFD_MOD_PNFACTOR         542
#define B3SOIFD_MOD_PDWG             543
#define B3SOIFD_MOD_PDWB             544
#define B3SOIFD_MOD_PVOFF            545
#define B3SOIFD_MOD_PETA0            546
#define B3SOIFD_MOD_PETAB            547
#define B3SOIFD_MOD_PDSUB            548
#define B3SOIFD_MOD_PCIT             549
#define B3SOIFD_MOD_PCDSC            550
#define B3SOIFD_MOD_PCDSCB           551
#define B3SOIFD_MOD_PCDSCD           552
#define B3SOIFD_MOD_PPCLM            553
#define B3SOIFD_MOD_PPDIBL1          554
#define B3SOIFD_MOD_PPDIBL2          555
#define B3SOIFD_MOD_PPDIBLB          556
#define B3SOIFD_MOD_PDROUT           557
#define B3SOIFD_MOD_PPVAG            558
#define B3SOIFD_MOD_PDELTA           559
#define B3SOIFD_MOD_PAII             560
#define B3SOIFD_MOD_PBII             561
#define B3SOIFD_MOD_PCII             562
#define B3SOIFD_MOD_PDII             563
#define B3SOIFD_MOD_PALPHA0          564
#define B3SOIFD_MOD_PALPHA1          565
#define B3SOIFD_MOD_PBETA0           566
#define B3SOIFD_MOD_PAGIDL           567
#define B3SOIFD_MOD_PBGIDL           568
#define B3SOIFD_MOD_PNGIDL           569
#define B3SOIFD_MOD_PNTUN            570
#define B3SOIFD_MOD_PNDIODE          571
#define B3SOIFD_MOD_PISBJT           572
#define B3SOIFD_MOD_PISDIF           573
#define B3SOIFD_MOD_PISREC           574
#define B3SOIFD_MOD_PISTUN           575
#define B3SOIFD_MOD_PEDL             576
#define B3SOIFD_MOD_PKBJT1           577
#define B3SOIFD_MOD_PVSDFB           578
#define B3SOIFD_MOD_PVSDTH           579
/* Added for binning - END3 */

#define B3SOIFD_MOD_TNOM             701
#define B3SOIFD_MOD_CGSO             702
#define B3SOIFD_MOD_CGDO             703
#define B3SOIFD_MOD_CGEO             704
#define B3SOIFD_MOD_XPART            705

#define B3SOIFD_MOD_RSH              706
#define B3SOIFD_MOD_NMOS             814
#define B3SOIFD_MOD_PMOS             815

#define B3SOIFD_MOD_NOIA             816
#define B3SOIFD_MOD_NOIB             817
#define B3SOIFD_MOD_NOIC             818

#define B3SOIFD_MOD_LINT             819
#define B3SOIFD_MOD_LL               820
#define B3SOIFD_MOD_LLN              821
#define B3SOIFD_MOD_LW               822
#define B3SOIFD_MOD_LWN              823
#define B3SOIFD_MOD_LWL              824

#define B3SOIFD_MOD_WINT             827
#define B3SOIFD_MOD_WL               828
#define B3SOIFD_MOD_WLN              829
#define B3SOIFD_MOD_WW               830
#define B3SOIFD_MOD_WWN              831
#define B3SOIFD_MOD_WWL              832

#define B3SOIFD_MOD_DWC              835
#define B3SOIFD_MOD_DLC              836

#define B3SOIFD_MOD_EM               837
#define B3SOIFD_MOD_EF               838
#define B3SOIFD_MOD_AF               839
#define B3SOIFD_MOD_KF               840
#define B3SOIFD_MOD_NOIF             841


#define B3SOIFD_MOD_PBSWG            843
#define B3SOIFD_MOD_MJSWG            844
#define B3SOIFD_MOD_CJSWG            845
#define B3SOIFD_MOD_CSDESW           846

/* device questions */
#define B3SOIFD_DNODE                901
#define B3SOIFD_GNODE                902
#define B3SOIFD_SNODE                903
#define B3SOIFD_BNODE                904
#define B3SOIFD_ENODE                905
#define B3SOIFD_DNODEPRIME           906
#define B3SOIFD_SNODEPRIME           907
#define B3SOIFD_VBD                  908
#define B3SOIFD_VBS                  909
#define B3SOIFD_VGS                  910
#define B3SOIFD_VES                  911
#define B3SOIFD_VDS                  912
#define B3SOIFD_CD                   913
#define B3SOIFD_CBS                  914
#define B3SOIFD_CBD                  915
#define B3SOIFD_GM                   916
#define B3SOIFD_GDS                  917
#define B3SOIFD_GMBS                 918
#define B3SOIFD_GBD                  919
#define B3SOIFD_GBS                  920
#define B3SOIFD_QB                   921
#define B3SOIFD_CQB                  922
#define B3SOIFD_QG                   923
#define B3SOIFD_CQG                  924
#define B3SOIFD_QD                   925
#define B3SOIFD_CQD                  926
#define B3SOIFD_CGG                  927
#define B3SOIFD_CGD                  928
#define B3SOIFD_CGS                  929
#define B3SOIFD_CBG                  930
#define B3SOIFD_CAPBD                931
#define B3SOIFD_CQBD                 932
#define B3SOIFD_CAPBS                933
#define B3SOIFD_CQBS                 934
#define B3SOIFD_CDG                  935
#define B3SOIFD_CDD                  936
#define B3SOIFD_CDS                  937
#define B3SOIFD_VON                  938
#define B3SOIFD_VDSAT                939
#define B3SOIFD_QBS                  940
#define B3SOIFD_QBD                  941
#define B3SOIFD_SOURCECONDUCT        942
#define B3SOIFD_DRAINCONDUCT         943
#define B3SOIFD_CBDB                 944
#define B3SOIFD_CBSB                 945
#define B3SOIFD_GMID                 946


#include "b3soifdext.h"

extern void B3SOIFDevaluate(double,double,double,B3SOIFDinstance*,B3SOIFDmodel*,
        double*,double*,double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, CKTcircuit*);
extern int B3SOIFDdebug(B3SOIFDmodel*, B3SOIFDinstance*, CKTcircuit*, int);
extern int B3SOIFDcheckModel(B3SOIFDmodel*, B3SOIFDinstance*, CKTcircuit*);

#endif /*B3SOIFD*/

