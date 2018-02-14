/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung
File: b3soipddef.h
Modified by Pin Su and Jan Feng	99/2/15
Modified by Pin Su 99/4/30
Modified by Pin Su and Wei Jin 99/9/27
Modified by Pin Su 00/3/1
Modified by Pin Su 01/2/15
Modified by Pin Su and Hui Wan 02/3/5
Modified by Paolo Nenzi 2002
**********/

#ifndef B3SOIPD
#define B3SOIPD

#define SOICODE
/*  #define BULKCODE  */

#include "ngspice/ifsim.h"
#include "ngspice/gendefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/noisedef.h"         

typedef struct sB3SOIPDinstance
{

    struct GENinstance gen;

#define B3SOIPDmodPtr(inst) ((struct sB3SOIPDmodel *)((inst)->gen.GENmodPtr))
#define B3SOIPDnextInstance(inst) ((struct sB3SOIPDinstance *)((inst)->gen.GENnextInstance))
#define B3SOIPDname gen.GENname
#define B3SOIPDstates gen.GENstate

    const int B3SOIPDdNode;
    const int B3SOIPDgNode;
    const int B3SOIPDsNode;
    const int B3SOIPDeNode;
    const int B3SOIPDbNodeExt;
    const int B3SOIPDtempNodeExt;
    const int B3SOIPDpNodeExt;
    int B3SOIPDpNode;
    int B3SOIPDbNode;
    int B3SOIPDtempNode;
    int B3SOIPDdNodePrime;
    int B3SOIPDsNodePrime;

    int B3SOIPDvbsNode;
    /* for Debug */
    int B3SOIPDidsNode;
    int B3SOIPDicNode;
    int B3SOIPDibsNode;
    int B3SOIPDibdNode;
    int B3SOIPDiiiNode;
    int B3SOIPDigNode;
    int B3SOIPDgiggNode;
    int B3SOIPDgigdNode;
    int B3SOIPDgigbNode;
    int B3SOIPDigidlNode;
    int B3SOIPDitunNode;
    int B3SOIPDibpNode;
    int B3SOIPDcbbNode;
    int B3SOIPDcbdNode;
    int B3SOIPDcbgNode;

    int B3SOIPDqbfNode;
    int B3SOIPDqjsNode;
    int B3SOIPDqjdNode;


    double B3SOIPDphi;
    double B3SOIPDvtm;
    double B3SOIPDni;
    double B3SOIPDueff;
    double B3SOIPDthetavth; 
    double B3SOIPDvon;
    double B3SOIPDvdsat;
    double B3SOIPDcgdo;
    double B3SOIPDcgso;
    double B3SOIPDcgeo;

    double B3SOIPDids;
    double B3SOIPDic;
    double B3SOIPDibs;
    double B3SOIPDibd;
    double B3SOIPDiii;
    double B3SOIPDig;
    double B3SOIPDgigg;
    double B3SOIPDgigd;
    double B3SOIPDgigb;
    double B3SOIPDigidl;
    double B3SOIPDitun;
    double B3SOIPDibp;
    double B3SOIPDabeff;
    double B3SOIPDvbseff;
    double B3SOIPDcbg;
    double B3SOIPDcbb;
    double B3SOIPDcbd;
    double B3SOIPDqb;
    double B3SOIPDqbf;
    double B3SOIPDqjs;
    double B3SOIPDqjd;
    int    B3SOIPDfloat;

    double B3SOIPDl;
    double B3SOIPDw;
    double B3SOIPDm;
    double B3SOIPDdrainArea;
    double B3SOIPDsourceArea;
    double B3SOIPDdrainSquares;
    double B3SOIPDsourceSquares;
    double B3SOIPDdrainPerimeter;
    double B3SOIPDsourcePerimeter;
    double B3SOIPDsourceConductance;
    double B3SOIPDdrainConductance;

    double B3SOIPDicVBS;
    double B3SOIPDicVDS;
    double B3SOIPDicVGS;
    double B3SOIPDicVES;
    double B3SOIPDicVPS;
    int B3SOIPDbjtoff;
    int B3SOIPDbodyMod;
    int B3SOIPDdebugMod;
    double B3SOIPDrth0;
    double B3SOIPDcth0;
    double B3SOIPDbodySquares;
    double B3SOIPDrbodyext;
    double B3SOIPDfrbody;


/* v2.0 release */
    double B3SOIPDnbc;   
    double B3SOIPDnseg;
    double B3SOIPDpdbcp;
    double B3SOIPDpsbcp;
    double B3SOIPDagbcp;
    double B3SOIPDaebcp;
    double B3SOIPDvbsusr;
    int B3SOIPDtnodeout;

/* Deleted from pParam and moved to here */
    double B3SOIPDcsesw;
    double B3SOIPDcdesw;
    double B3SOIPDcsbox;
    double B3SOIPDcdbox;
    double B3SOIPDcsmin;
    double B3SOIPDcdmin;
    double B3SOIPDst4;
    double B3SOIPDdt4;

    int B3SOIPDoff;
    int B3SOIPDmode;

    /* OP point */
    double B3SOIPDqinv;
    double B3SOIPDcd;
    double B3SOIPDcjs;
    double B3SOIPDcjd;
    double B3SOIPDcbody;
/* v2.2 release */
    double B3SOIPDcgate;
    double B3SOIPDgigs;
    double B3SOIPDgigT;

    double B3SOIPDcbodcon;
    double B3SOIPDcth;
    double B3SOIPDcsubstrate;

    double B3SOIPDgm;
    double B3SOIPDcb;
    double B3SOIPDcdrain;
    double B3SOIPDgds;
    double B3SOIPDgmbs;
    double B3SOIPDgmT;

    double B3SOIPDgbbs;
    double B3SOIPDgbgs;
    double B3SOIPDgbds;
    double B3SOIPDgbps;
    double B3SOIPDgbT;

    double B3SOIPDgjsd;
    double B3SOIPDgjsb;
    double B3SOIPDgjsg;
    double B3SOIPDgjsT;

    double B3SOIPDgjdb;
    double B3SOIPDgjdd;
    double B3SOIPDgjdg;
    double B3SOIPDgjdT;

    double B3SOIPDgbpbs;
    double B3SOIPDgbpps;
    double B3SOIPDgbpT;

    double B3SOIPDgtempb;
    double B3SOIPDgtempg;
    double B3SOIPDgtempd;
    double B3SOIPDgtempT;

    double B3SOIPDcggb;
    double B3SOIPDcgdb;
    double B3SOIPDcgsb;
    double B3SOIPDcgT;

    double B3SOIPDcbgb;
    double B3SOIPDcbdb;
    double B3SOIPDcbsb;
    double B3SOIPDcbeb;
    double B3SOIPDcbT;

    double B3SOIPDcdgb;
    double B3SOIPDcddb;
    double B3SOIPDcdsb;
    double B3SOIPDcdeb;
    double B3SOIPDcdT;

    double B3SOIPDceeb;
    double B3SOIPDceT;

    double B3SOIPDqse;
    double B3SOIPDgcse;
    double B3SOIPDqde;
    double B3SOIPDgcde;
    double B3SOIPDrds; /* v2.2.3 */
    double B3SOIPDVgsteff; /* v2.2.3 */
    double B3SOIPDVdseff;  /* v2.2.3 */
    double B3SOIPDAbovVgst2Vtm; /* v2.2.3 */

    struct b3soipdSizeDependParam  *pParam;

    unsigned B3SOIPDlGiven :1;
    unsigned B3SOIPDwGiven :1;
    unsigned B3SOIPDmGiven :1;
    unsigned B3SOIPDdrainAreaGiven :1;
    unsigned B3SOIPDsourceAreaGiven    :1;
    unsigned B3SOIPDdrainSquaresGiven  :1;
    unsigned B3SOIPDsourceSquaresGiven :1;
    unsigned B3SOIPDdrainPerimeterGiven    :1;
    unsigned B3SOIPDsourcePerimeterGiven   :1;
    unsigned B3SOIPDdNodePrimeSet  :1;
    unsigned B3SOIPDsNodePrimeSet  :1;
    unsigned B3SOIPDicVBSGiven :1;
    unsigned B3SOIPDicVDSGiven :1;
    unsigned B3SOIPDicVGSGiven :1;
    unsigned B3SOIPDicVESGiven :1;
    unsigned B3SOIPDicVPSGiven :1;
    unsigned B3SOIPDbjtoffGiven :1;
    unsigned B3SOIPDdebugModGiven :1;
    unsigned B3SOIPDrth0Given :1;
    unsigned B3SOIPDcth0Given :1;
    unsigned B3SOIPDbodySquaresGiven :1;
    unsigned B3SOIPDfrbodyGiven:  1;

/* v2.0 release */
    unsigned B3SOIPDnbcGiven :1;   
    unsigned B3SOIPDnsegGiven :1;
    unsigned B3SOIPDpdbcpGiven :1;
    unsigned B3SOIPDpsbcpGiven :1;
    unsigned B3SOIPDagbcpGiven :1;
    unsigned B3SOIPDaebcpGiven :1;
    unsigned B3SOIPDvbsusrGiven :1;
    unsigned B3SOIPDtnodeoutGiven :1;
    unsigned B3SOIPDoffGiven :1;

    double *B3SOIPDGePtr;
    double *B3SOIPDDPePtr;
    double *B3SOIPDSPePtr;

    double *B3SOIPDEePtr;
    double *B3SOIPDEbPtr;
    double *B3SOIPDBePtr;
    double *B3SOIPDEgPtr;
    double *B3SOIPDEdpPtr;
    double *B3SOIPDEspPtr;
    double *B3SOIPDTemptempPtr;
    double *B3SOIPDTempdpPtr;
    double *B3SOIPDTempspPtr;
    double *B3SOIPDTempgPtr;
    double *B3SOIPDTempbPtr;
    double *B3SOIPDGtempPtr;
    double *B3SOIPDDPtempPtr;
    double *B3SOIPDSPtempPtr;
    double *B3SOIPDEtempPtr;
    double *B3SOIPDBtempPtr;
    double *B3SOIPDPtempPtr;
    double *B3SOIPDBpPtr;
    double *B3SOIPDPbPtr;
    double *B3SOIPDPpPtr;
    double *B3SOIPDDdPtr;
    double *B3SOIPDGgPtr;
    double *B3SOIPDSsPtr;
    double *B3SOIPDBbPtr;
    double *B3SOIPDDPdpPtr;
    double *B3SOIPDSPspPtr;
    double *B3SOIPDDdpPtr;
    double *B3SOIPDGbPtr;
    double *B3SOIPDGdpPtr;
    double *B3SOIPDGspPtr;
    double *B3SOIPDSspPtr;
    double *B3SOIPDBdpPtr;
    double *B3SOIPDBspPtr;
    double *B3SOIPDDPspPtr;
    double *B3SOIPDDPdPtr;
    double *B3SOIPDBgPtr;
    double *B3SOIPDDPgPtr;
    double *B3SOIPDSPgPtr;
    double *B3SOIPDSPsPtr;
    double *B3SOIPDDPbPtr;
    double *B3SOIPDSPbPtr;
    double *B3SOIPDSPdpPtr;

    double *B3SOIPDVbsPtr;
    /* Debug */
    double *B3SOIPDIdsPtr;
    double *B3SOIPDIcPtr;
    double *B3SOIPDIbsPtr;
    double *B3SOIPDIbdPtr;
    double *B3SOIPDIiiPtr;
    double *B3SOIPDIgPtr;
    double *B3SOIPDGiggPtr;
    double *B3SOIPDGigdPtr;
    double *B3SOIPDGigbPtr;
    double *B3SOIPDIgidlPtr;
    double *B3SOIPDItunPtr;
    double *B3SOIPDIbpPtr;
    double *B3SOIPDCbbPtr;
    double *B3SOIPDCbdPtr;
    double *B3SOIPDCbgPtr;
    double *B3SOIPDqbPtr;
    double *B3SOIPDQbfPtr;
    double *B3SOIPDQjsPtr;
    double *B3SOIPDQjdPtr;


#define B3SOIPDvbd B3SOIPDstates+ 0
#define B3SOIPDvbs B3SOIPDstates+ 1
#define B3SOIPDvgs B3SOIPDstates+ 2
#define B3SOIPDvds B3SOIPDstates+ 3
#define B3SOIPDves B3SOIPDstates+ 4
#define B3SOIPDvps B3SOIPDstates+ 5

#define B3SOIPDvg B3SOIPDstates+ 6
#define B3SOIPDvd B3SOIPDstates+ 7
#define B3SOIPDvs B3SOIPDstates+ 8
#define B3SOIPDvp B3SOIPDstates+ 9
#define B3SOIPDve B3SOIPDstates+ 10
#define B3SOIPDdeltemp B3SOIPDstates+ 11

#define B3SOIPDqb B3SOIPDstates+ 12
#define B3SOIPDcqb B3SOIPDstates+ 13
#define B3SOIPDqg B3SOIPDstates+ 14
#define B3SOIPDcqg B3SOIPDstates+ 15
#define B3SOIPDqd B3SOIPDstates+ 16
#define B3SOIPDcqd B3SOIPDstates+ 17
#define B3SOIPDqe B3SOIPDstates+ 18
#define B3SOIPDcqe B3SOIPDstates+ 19

#define B3SOIPDqbs  B3SOIPDstates+ 20
#define B3SOIPDqbd  B3SOIPDstates+ 21
#define B3SOIPDqbe  B3SOIPDstates+ 22

#define B3SOIPDqth B3SOIPDstates+ 23
#define B3SOIPDcqth B3SOIPDstates+ 24

#define B3SOIPDnumStates 25


/* indices to the array of B3SOIPD NOISE SOURCES */

#define B3SOIPDRDNOIZ       0
#define B3SOIPDRSNOIZ       1
#define B3SOIPDIDNOIZ       2
#define B3SOIPDFLNOIZ       3
#define B3SOIPDFBNOIZ       4
#define B3SOIPDTOTNOIZ      5

#define B3SOIPDNSRCS        6     /* the number of MOSFET(3) noise sources */

#ifndef NONOISE
    double B3SOIPDnVar[NSTATVARS][B3SOIPDNSRCS];
#else /* NONOISE */
        double **B3SOIPDnVar;
#endif /* NONOISE */

} B3SOIPDinstance ;

struct b3soipdSizeDependParam
{
    double Width;
    double Length;
    double Rth0;
    double Cth0;

    double B3SOIPDcdsc;           
    double B3SOIPDcdscb;    
    double B3SOIPDcdscd;       
    double B3SOIPDcit;           
    double B3SOIPDnfactor;      
    double B3SOIPDvsat;         
    double B3SOIPDat;         
    double B3SOIPDa0;   
    double B3SOIPDags;      
    double B3SOIPDa1;         
    double B3SOIPDa2;         
    double B3SOIPDketa;     
    double B3SOIPDnpeak;        
    double B3SOIPDnsub;
    double B3SOIPDngate;        
    double B3SOIPDgamma1;      
    double B3SOIPDgamma2;     
    double B3SOIPDvbx;      
    double B3SOIPDvbi;       
    double B3SOIPDvbm;       
    double B3SOIPDvbsc;       
    double B3SOIPDxt;       
    double B3SOIPDphi;
    double B3SOIPDlitl;
    double B3SOIPDk1;
    double B3SOIPDkt1;
    double B3SOIPDkt1l;
    double B3SOIPDkt2;
    double B3SOIPDk2;
    double B3SOIPDk3;
    double B3SOIPDk3b;
    double B3SOIPDw0;
    double B3SOIPDnlx;
    double B3SOIPDdvt0;      
    double B3SOIPDdvt1;      
    double B3SOIPDdvt2;      
    double B3SOIPDdvt0w;      
    double B3SOIPDdvt1w;      
    double B3SOIPDdvt2w;      
    double B3SOIPDdrout;      
    double B3SOIPDdsub;      
    double B3SOIPDvth0;
    double B3SOIPDua;
    double B3SOIPDua1;
    double B3SOIPDub;
    double B3SOIPDub1;
    double B3SOIPDuc;
    double B3SOIPDuc1;
    double B3SOIPDu0;
    double B3SOIPDute;
    double B3SOIPDvoff;
    double B3SOIPDvfb;
    double B3SOIPDuatemp;
    double B3SOIPDubtemp;
    double B3SOIPDuctemp;
    double B3SOIPDrbody;
    double B3SOIPDrth;
    double B3SOIPDcth;
    double B3SOIPDrds0denom;
    double B3SOIPDvfbb;
    double B3SOIPDjbjt;
    double B3SOIPDjdif;
    double B3SOIPDjrec;
    double B3SOIPDjtun;
    double B3SOIPDsdt1;
    double B3SOIPDst2;
    double B3SOIPDst3;
    double B3SOIPDdt2;
    double B3SOIPDdt3;
    double B3SOIPDdelta;
    double B3SOIPDrdsw;       
    double B3SOIPDrds0;       
    double B3SOIPDprwg;       
    double B3SOIPDprwb;       
    double B3SOIPDprt;       
    double B3SOIPDeta0;         
    double B3SOIPDetab;         
    double B3SOIPDpclm;      
    double B3SOIPDpdibl1;      
    double B3SOIPDpdibl2;      
    double B3SOIPDpdiblb;      
    double B3SOIPDpvag;       
    double B3SOIPDwr;
    double B3SOIPDdwg;
    double B3SOIPDdwb;
    double B3SOIPDb0;
    double B3SOIPDb1;
    double B3SOIPDalpha0;
    double B3SOIPDbeta0;


    /* CV model */
    double B3SOIPDcgsl;
    double B3SOIPDcgdl;
    double B3SOIPDckappa;
    double B3SOIPDcf;
    double B3SOIPDclc;
    double B3SOIPDcle;

/* Added for binning - START0 */
    double B3SOIPDkb1;
    double B3SOIPDk1w1;
    double B3SOIPDk1w2;
    double B3SOIPDketas;
    double B3SOIPDfbjtii;
    double B3SOIPDbeta1;
    double B3SOIPDbeta2;
    double B3SOIPDvdsatii0;
    double B3SOIPDlii;
    double B3SOIPDesatii;
    double B3SOIPDsii0;
    double B3SOIPDsii1;
    double B3SOIPDsii2;
    double B3SOIPDsiid;
    double B3SOIPDagidl;
    double B3SOIPDbgidl;
    double B3SOIPDngidl;
    double B3SOIPDntun;
    double B3SOIPDndiode;
    double B3SOIPDnrecf0;
    double B3SOIPDnrecr0;
    double B3SOIPDisbjt;
    double B3SOIPDisdif;
    double B3SOIPDisrec;
    double B3SOIPDistun;
    double B3SOIPDvrec0;
    double B3SOIPDvtun0;
    double B3SOIPDnbjt;
    double B3SOIPDlbjt0;
    double B3SOIPDvabjt;
    double B3SOIPDaely;
    double B3SOIPDvsdfb;
    double B3SOIPDvsdth;
    double B3SOIPDdelvt;
/* Added by binning - END0 */

/* Pre-calculated constants */

    double B3SOIPDdw;
    double B3SOIPDdl;
    double B3SOIPDleff;
    double B3SOIPDweff;

    double B3SOIPDdwc;
    double B3SOIPDdlc;
    double B3SOIPDleffCV;
    double B3SOIPDweffCV;
    double B3SOIPDabulkCVfactor;
    double B3SOIPDcgso;
    double B3SOIPDcgdo;
    double B3SOIPDcgeo;

    double B3SOIPDu0temp;       
    double B3SOIPDvsattemp;   
    double B3SOIPDsqrtPhi;   
    double B3SOIPDphis3;   
    double B3SOIPDXdep0;          
    double B3SOIPDsqrtXdep0;          
    double B3SOIPDtheta0vb0;
    double B3SOIPDthetaRout; 


/* v2.2 release */
    double B3SOIPDoxideRatio;


/* v2.0 release */
    double B3SOIPDk1eff;
    double B3SOIPDwdios;
    double B3SOIPDwdiod;
    double B3SOIPDwdiodCV;
    double B3SOIPDwdiosCV;
    double B3SOIPDarfabjt;
    double B3SOIPDlratio;
    double B3SOIPDlratiodif;
    double B3SOIPDvearly;
    double B3SOIPDahli;
    double B3SOIPDahli0;
    double B3SOIPDvfbzb;
    double B3SOIPDldeb;
    double B3SOIPDacde;
    double B3SOIPDmoin;
    double B3SOIPDleffCVb;
    double B3SOIPDleffCVbg;

    double B3SOIPDcof1;
    double B3SOIPDcof2;
    double B3SOIPDcof3;
    double B3SOIPDcof4;
    double B3SOIPDcdep0;
    struct b3soipdSizeDependParam  *pNext;
};


typedef struct sB3SOIPDmodel 
{

    struct GENmodel gen;

#define B3SOIPDmodType gen.GENmodType
#define B3SOIPDnextModel(inst) ((struct sB3SOIPDmodel *)((inst)->gen.GENnextModel))
#define B3SOIPDinstances(inst) ((B3SOIPDinstance *)((inst)->gen.GENinstances))
#define B3SOIPDmodName gen.GENmodName

    int B3SOIPDtype;

    int    B3SOIPDmobMod;
    int    B3SOIPDcapMod;
    int    B3SOIPDnoiMod;
    int    B3SOIPDshMod;
    int    B3SOIPDbinUnit;
    int    B3SOIPDparamChk;
    double B3SOIPDversion;             
    double B3SOIPDtox;             
    double B3SOIPDdtoxcv; /* v2.2.3 */
    double B3SOIPDcdsc;           
    double B3SOIPDcdscb; 
    double B3SOIPDcdscd;          
    double B3SOIPDcit;           
    double B3SOIPDnfactor;      
    double B3SOIPDvsat;         
    double B3SOIPDat;         
    double B3SOIPDa0;   
    double B3SOIPDags;      
    double B3SOIPDa1;         
    double B3SOIPDa2;         
    double B3SOIPDketa;     
    double B3SOIPDnsub;
    double B3SOIPDnpeak;        
    double B3SOIPDngate;        
    double B3SOIPDgamma1;      
    double B3SOIPDgamma2;     
    double B3SOIPDvbx;      
    double B3SOIPDvbm;       
    double B3SOIPDxt;       
    double B3SOIPDk1;
    double B3SOIPDkt1;
    double B3SOIPDkt1l;
    double B3SOIPDkt2;
    double B3SOIPDk2;
    double B3SOIPDk3;
    double B3SOIPDk3b;
    double B3SOIPDw0;
    double B3SOIPDnlx;
    double B3SOIPDdvt0;      
    double B3SOIPDdvt1;      
    double B3SOIPDdvt2;      
    double B3SOIPDdvt0w;      
    double B3SOIPDdvt1w;      
    double B3SOIPDdvt2w;      
    double B3SOIPDdrout;      
    double B3SOIPDdsub;      
    double B3SOIPDvth0;
    double B3SOIPDua;
    double B3SOIPDua1;
    double B3SOIPDub;
    double B3SOIPDub1;
    double B3SOIPDuc;
    double B3SOIPDuc1;
    double B3SOIPDu0;
    double B3SOIPDute;
    double B3SOIPDvoff;
    double B3SOIPDdelta;
    double B3SOIPDrdsw;       
    double B3SOIPDprwg;
    double B3SOIPDprwb;
    double B3SOIPDprt;       
    double B3SOIPDeta0;         
    double B3SOIPDetab;         
    double B3SOIPDpclm;      
    double B3SOIPDpdibl1;      
    double B3SOIPDpdibl2;      
    double B3SOIPDpdiblb;
    double B3SOIPDpvag;       
    double B3SOIPDwr;
    double B3SOIPDdwg;
    double B3SOIPDdwb;
    double B3SOIPDb0;
    double B3SOIPDb1;
    double B3SOIPDalpha0;
    double B3SOIPDtbox;
    double B3SOIPDtsi;
    double B3SOIPDxj;
    double B3SOIPDkb1;
    double B3SOIPDrth0;
    double B3SOIPDcth0;
    double B3SOIPDngidl;
    double B3SOIPDagidl;
    double B3SOIPDbgidl;
    double B3SOIPDndiode;
    double B3SOIPDistun;
    double B3SOIPDxbjt;
    double B3SOIPDxdif;
    double B3SOIPDxrec;
    double B3SOIPDxtun;


/* v2.2 release */
    double B3SOIPDwth0;
    double B3SOIPDrhalo;
    double B3SOIPDntox;
    double B3SOIPDtoxref;
    double B3SOIPDebg;
    double B3SOIPDvevb;
    double B3SOIPDalphaGB1;
    double B3SOIPDbetaGB1;
    double B3SOIPDvgb1;
    double B3SOIPDvecb;
    double B3SOIPDalphaGB2;
    double B3SOIPDbetaGB2;
    double B3SOIPDvgb2;
    double B3SOIPDtoxqm;
    double B3SOIPDvoxh;
    double B3SOIPDdeltavox;
    int B3SOIPDigMod;


/* v2.0 release */
    double B3SOIPDk1w1;  
    double B3SOIPDk1w2;
    double B3SOIPDketas;
    double B3SOIPDdwbc;
    double B3SOIPDbeta0;
    double B3SOIPDbeta1;
    double B3SOIPDbeta2;
    double B3SOIPDvdsatii0;
    double B3SOIPDtii;
    double B3SOIPDlii;
    double B3SOIPDsii0;
    double B3SOIPDsii1;
    double B3SOIPDsii2;
    double B3SOIPDsiid;
    double B3SOIPDfbjtii;
    double B3SOIPDesatii;
    double B3SOIPDntun;
    double B3SOIPDnrecf0;
    double B3SOIPDnrecr0;
    double B3SOIPDisbjt;
    double B3SOIPDisdif;
    double B3SOIPDisrec;
    double B3SOIPDln;
    double B3SOIPDvrec0;
    double B3SOIPDvtun0;
    double B3SOIPDnbjt;
    double B3SOIPDlbjt0;
    double B3SOIPDldif0;
    double B3SOIPDvabjt;
    double B3SOIPDaely;
    double B3SOIPDahli;
    double B3SOIPDrbody;
    double B3SOIPDrbsh;
    double B3SOIPDtt;
    double B3SOIPDndif;
    double B3SOIPDvsdfb;
    double B3SOIPDvsdth;
    double B3SOIPDcsdmin;
    double B3SOIPDasd;
    double B3SOIPDntrecf;
    double B3SOIPDntrecr;
    double B3SOIPDdlcb;
    double B3SOIPDfbody;
    double B3SOIPDtcjswg;
    double B3SOIPDtpbswg;
    double B3SOIPDacde;
    double B3SOIPDmoin;
    double B3SOIPDdelvt;
    double B3SOIPDdlbg;

    /* CV model */
    double B3SOIPDcgsl;
    double B3SOIPDcgdl;
    double B3SOIPDckappa;
    double B3SOIPDcf;
    double B3SOIPDclc;
    double B3SOIPDcle;
    double B3SOIPDdwc;
    double B3SOIPDdlc;

    double B3SOIPDtnom;
    double B3SOIPDcgso;
    double B3SOIPDcgdo;
    double B3SOIPDcgeo;
    double B3SOIPDxpart;
    double B3SOIPDcFringOut;
    double B3SOIPDcFringMax;

    double B3SOIPDsheetResistance;
    double B3SOIPDbodyJctGateSideGradingCoeff;
    double B3SOIPDGatesidewallJctPotential;
    double B3SOIPDunitLengthGateSidewallJctCap;
    double B3SOIPDcsdesw;

    double B3SOIPDLint;
    double B3SOIPDLl;
    double B3SOIPDLlc; /* v2.2.3 */
    double B3SOIPDLln;
    double B3SOIPDLw;
    double B3SOIPDLwc; /* v2.2.3 */
    double B3SOIPDLwn;
    double B3SOIPDLwl;
    double B3SOIPDLwlc; /* v2.2.3 */
    double B3SOIPDLmin;
    double B3SOIPDLmax;

    double B3SOIPDWint;
    double B3SOIPDWl;
    double B3SOIPDWlc; /* v2.2.3 */
    double B3SOIPDWln;
    double B3SOIPDWw;
    double B3SOIPDWwc; /* v2.2.3 */
    double B3SOIPDWwn;
    double B3SOIPDWwl;
    double B3SOIPDWwlc; /* v2.2.3 */
    double B3SOIPDWmin;
    double B3SOIPDWmax;

/* Added for binning - START1 */
    /* Length Dependence */
    double B3SOIPDlnpeak;        
    double B3SOIPDlnsub;
    double B3SOIPDlngate;        
    double B3SOIPDlvth0;
    double B3SOIPDlk1;
    double B3SOIPDlk1w1;
    double B3SOIPDlk1w2;
    double B3SOIPDlk2;
    double B3SOIPDlk3;
    double B3SOIPDlk3b;
    double B3SOIPDlkb1;
    double B3SOIPDlw0;
    double B3SOIPDlnlx;
    double B3SOIPDldvt0;      
    double B3SOIPDldvt1;      
    double B3SOIPDldvt2;      
    double B3SOIPDldvt0w;      
    double B3SOIPDldvt1w;      
    double B3SOIPDldvt2w;      
    double B3SOIPDlu0;
    double B3SOIPDlua;
    double B3SOIPDlub;
    double B3SOIPDluc;
    double B3SOIPDlvsat;         
    double B3SOIPDla0;   
    double B3SOIPDlags;      
    double B3SOIPDlb0;
    double B3SOIPDlb1;
    double B3SOIPDlketa;
    double B3SOIPDlketas;
    double B3SOIPDla1;         
    double B3SOIPDla2;         
    double B3SOIPDlrdsw;       
    double B3SOIPDlprwb;
    double B3SOIPDlprwg;
    double B3SOIPDlwr;
    double B3SOIPDlnfactor;      
    double B3SOIPDldwg;
    double B3SOIPDldwb;
    double B3SOIPDlvoff;
    double B3SOIPDleta0;         
    double B3SOIPDletab;         
    double B3SOIPDldsub;      
    double B3SOIPDlcit;           
    double B3SOIPDlcdsc;           
    double B3SOIPDlcdscb; 
    double B3SOIPDlcdscd;          
    double B3SOIPDlpclm;      
    double B3SOIPDlpdibl1;      
    double B3SOIPDlpdibl2;      
    double B3SOIPDlpdiblb;      
    double B3SOIPDldrout;      
    double B3SOIPDlpvag;       
    double B3SOIPDldelta;
    double B3SOIPDlalpha0;
    double B3SOIPDlfbjtii;
    double B3SOIPDlbeta0;
    double B3SOIPDlbeta1;
    double B3SOIPDlbeta2;         
    double B3SOIPDlvdsatii0;     
    double B3SOIPDllii;      
    double B3SOIPDlesatii;     
    double B3SOIPDlsii0;      
    double B3SOIPDlsii1;       
    double B3SOIPDlsii2;       
    double B3SOIPDlsiid;
    double B3SOIPDlagidl;
    double B3SOIPDlbgidl;
    double B3SOIPDlngidl;
    double B3SOIPDlntun;
    double B3SOIPDlndiode;
    double B3SOIPDlnrecf0;
    double B3SOIPDlnrecr0;
    double B3SOIPDlisbjt;       
    double B3SOIPDlisdif;
    double B3SOIPDlisrec;       
    double B3SOIPDlistun;       
    double B3SOIPDlvrec0;
    double B3SOIPDlvtun0;
    double B3SOIPDlnbjt;
    double B3SOIPDllbjt0;
    double B3SOIPDlvabjt;
    double B3SOIPDlaely;
    double B3SOIPDlahli;
    /* CV model */
    double B3SOIPDlvsdfb;
    double B3SOIPDlvsdth;
    double B3SOIPDldelvt;
    double B3SOIPDlacde;
    double B3SOIPDlmoin;

    /* Width Dependence */
    double B3SOIPDwnpeak;        
    double B3SOIPDwnsub;
    double B3SOIPDwngate;        
    double B3SOIPDwvth0;
    double B3SOIPDwk1;
    double B3SOIPDwk1w1;
    double B3SOIPDwk1w2;
    double B3SOIPDwk2;
    double B3SOIPDwk3;
    double B3SOIPDwk3b;
    double B3SOIPDwkb1;
    double B3SOIPDww0;
    double B3SOIPDwnlx;
    double B3SOIPDwdvt0;      
    double B3SOIPDwdvt1;      
    double B3SOIPDwdvt2;      
    double B3SOIPDwdvt0w;      
    double B3SOIPDwdvt1w;      
    double B3SOIPDwdvt2w;      
    double B3SOIPDwu0;
    double B3SOIPDwua;
    double B3SOIPDwub;
    double B3SOIPDwuc;
    double B3SOIPDwvsat;         
    double B3SOIPDwa0;   
    double B3SOIPDwags;      
    double B3SOIPDwb0;
    double B3SOIPDwb1;
    double B3SOIPDwketa;
    double B3SOIPDwketas;
    double B3SOIPDwa1;         
    double B3SOIPDwa2;         
    double B3SOIPDwrdsw;       
    double B3SOIPDwprwb;
    double B3SOIPDwprwg;
    double B3SOIPDwwr;
    double B3SOIPDwnfactor;      
    double B3SOIPDwdwg;
    double B3SOIPDwdwb;
    double B3SOIPDwvoff;
    double B3SOIPDweta0;         
    double B3SOIPDwetab;         
    double B3SOIPDwdsub;      
    double B3SOIPDwcit;           
    double B3SOIPDwcdsc;           
    double B3SOIPDwcdscb; 
    double B3SOIPDwcdscd;          
    double B3SOIPDwpclm;      
    double B3SOIPDwpdibl1;      
    double B3SOIPDwpdibl2;      
    double B3SOIPDwpdiblb;      
    double B3SOIPDwdrout;      
    double B3SOIPDwpvag;       
    double B3SOIPDwdelta;
    double B3SOIPDwalpha0;
    double B3SOIPDwfbjtii;
    double B3SOIPDwbeta0;
    double B3SOIPDwbeta1;
    double B3SOIPDwbeta2;         
    double B3SOIPDwvdsatii0;     
    double B3SOIPDwlii;      
    double B3SOIPDwesatii;     
    double B3SOIPDwsii0;      
    double B3SOIPDwsii1;       
    double B3SOIPDwsii2;       
    double B3SOIPDwsiid;
    double B3SOIPDwagidl;
    double B3SOIPDwbgidl;
    double B3SOIPDwngidl;
    double B3SOIPDwntun;
    double B3SOIPDwndiode;
    double B3SOIPDwnrecf0;
    double B3SOIPDwnrecr0;
    double B3SOIPDwisbjt;       
    double B3SOIPDwisdif;
    double B3SOIPDwisrec;       
    double B3SOIPDwistun;       
    double B3SOIPDwvrec0;
    double B3SOIPDwvtun0;
    double B3SOIPDwnbjt;
    double B3SOIPDwlbjt0;
    double B3SOIPDwvabjt;
    double B3SOIPDwaely;
    double B3SOIPDwahli;
    /* CV model */
    double B3SOIPDwvsdfb;
    double B3SOIPDwvsdth;
    double B3SOIPDwdelvt;
    double B3SOIPDwacde;
    double B3SOIPDwmoin;

    /* Cross-term Dependence */
    double B3SOIPDpnpeak;        
    double B3SOIPDpnsub;
    double B3SOIPDpngate;        
    double B3SOIPDpvth0;
    double B3SOIPDpk1;
    double B3SOIPDpk1w1;
    double B3SOIPDpk1w2;
    double B3SOIPDpk2;
    double B3SOIPDpk3;
    double B3SOIPDpk3b;
    double B3SOIPDpkb1;
    double B3SOIPDpw0;
    double B3SOIPDpnlx;
    double B3SOIPDpdvt0;      
    double B3SOIPDpdvt1;      
    double B3SOIPDpdvt2;      
    double B3SOIPDpdvt0w;      
    double B3SOIPDpdvt1w;      
    double B3SOIPDpdvt2w;      
    double B3SOIPDpu0;
    double B3SOIPDpua;
    double B3SOIPDpub;
    double B3SOIPDpuc;
    double B3SOIPDpvsat;         
    double B3SOIPDpa0;   
    double B3SOIPDpags;      
    double B3SOIPDpb0;
    double B3SOIPDpb1;
    double B3SOIPDpketa;
    double B3SOIPDpketas;
    double B3SOIPDpa1;         
    double B3SOIPDpa2;         
    double B3SOIPDprdsw;       
    double B3SOIPDpprwb;
    double B3SOIPDpprwg;
    double B3SOIPDpwr;
    double B3SOIPDpnfactor;      
    double B3SOIPDpdwg;
    double B3SOIPDpdwb;
    double B3SOIPDpvoff;
    double B3SOIPDpeta0;         
    double B3SOIPDpetab;         
    double B3SOIPDpdsub;      
    double B3SOIPDpcit;           
    double B3SOIPDpcdsc;           
    double B3SOIPDpcdscb; 
    double B3SOIPDpcdscd;          
    double B3SOIPDppclm;      
    double B3SOIPDppdibl1;      
    double B3SOIPDppdibl2;      
    double B3SOIPDppdiblb;      
    double B3SOIPDpdrout;      
    double B3SOIPDppvag;       
    double B3SOIPDpdelta;
    double B3SOIPDpalpha0;
    double B3SOIPDpfbjtii;
    double B3SOIPDpbeta0;
    double B3SOIPDpbeta1;
    double B3SOIPDpbeta2;         
    double B3SOIPDpvdsatii0;     
    double B3SOIPDplii;      
    double B3SOIPDpesatii;     
    double B3SOIPDpsii0;      
    double B3SOIPDpsii1;       
    double B3SOIPDpsii2;       
    double B3SOIPDpsiid;
    double B3SOIPDpagidl;
    double B3SOIPDpbgidl;
    double B3SOIPDpngidl;
    double B3SOIPDpntun;
    double B3SOIPDpndiode;
    double B3SOIPDpnrecf0;
    double B3SOIPDpnrecr0;
    double B3SOIPDpisbjt;       
    double B3SOIPDpisdif;
    double B3SOIPDpisrec;       
    double B3SOIPDpistun;       
    double B3SOIPDpvrec0;
    double B3SOIPDpvtun0;
    double B3SOIPDpnbjt;
    double B3SOIPDplbjt0;
    double B3SOIPDpvabjt;
    double B3SOIPDpaely;
    double B3SOIPDpahli;
    /* CV model */
    double B3SOIPDpvsdfb;
    double B3SOIPDpvsdth;
    double B3SOIPDpdelvt;
    double B3SOIPDpacde;
    double B3SOIPDpmoin;
/* Added for binning - END1 */

/* Pre-calculated constants */
    double B3SOIPDcbox;
    double B3SOIPDcsi;
    double B3SOIPDcsieff;
    double B3SOIPDcoxt;
    double B3SOIPDnfb;
    double B3SOIPDadice;
    double B3SOIPDqsi;
    double B3SOIPDqsieff;
    double B3SOIPDeg0;

    /* MCJ: move to size-dependent param. */
    double B3SOIPDvtm;   
    double B3SOIPDcox;
    double B3SOIPDcof1;
    double B3SOIPDcof2;
    double B3SOIPDcof3;
    double B3SOIPDcof4;
    double B3SOIPDvcrit;
    double B3SOIPDfactor1;

    double B3SOIPDoxideTrapDensityA;      
    double B3SOIPDoxideTrapDensityB;     
    double B3SOIPDoxideTrapDensityC;  
    double B3SOIPDem;  
    double B3SOIPDef;  
    double B3SOIPDaf;  
    double B3SOIPDkf;  
    double B3SOIPDnoif;  

    struct b3soipdSizeDependParam *pSizeDependParamKnot;

    /* Flags */

    unsigned B3SOIPDtboxGiven:1;
    unsigned B3SOIPDtsiGiven :1;
    unsigned B3SOIPDxjGiven :1;
    unsigned B3SOIPDkb1Given :1;
    unsigned B3SOIPDrth0Given :1;
    unsigned B3SOIPDcth0Given :1;
    unsigned B3SOIPDngidlGiven :1;
    unsigned B3SOIPDagidlGiven :1;
    unsigned B3SOIPDbgidlGiven :1;
    unsigned B3SOIPDndiodeGiven :1;
    unsigned B3SOIPDxbjtGiven :1;
    unsigned B3SOIPDxdifGiven :1;
    unsigned B3SOIPDxrecGiven :1;
    unsigned B3SOIPDxtunGiven :1;
    unsigned B3SOIPDttGiven :1;
    unsigned B3SOIPDvsdfbGiven :1;
    unsigned B3SOIPDvsdthGiven :1;
    unsigned B3SOIPDasdGiven :1;
    unsigned B3SOIPDcsdminGiven :1;

    unsigned  B3SOIPDmobModGiven :1;
    unsigned  B3SOIPDbinUnitGiven :1;
    unsigned  B3SOIPDcapModGiven :1;
    unsigned  B3SOIPDparamChkGiven :1;
    unsigned  B3SOIPDnoiModGiven :1;
    unsigned  B3SOIPDshModGiven :1;
    unsigned  B3SOIPDtypeGiven   :1;
    unsigned  B3SOIPDtoxGiven   :1;
    unsigned  B3SOIPDdtoxcvGiven   :1; /* v2.2.3 */
    unsigned  B3SOIPDversionGiven   :1;

    unsigned  B3SOIPDcdscGiven   :1;
    unsigned  B3SOIPDcdscbGiven   :1;
    unsigned  B3SOIPDcdscdGiven   :1;
    unsigned  B3SOIPDcitGiven   :1;
    unsigned  B3SOIPDnfactorGiven   :1;
    unsigned  B3SOIPDvsatGiven   :1;
    unsigned  B3SOIPDatGiven   :1;
    unsigned  B3SOIPDa0Given   :1;
    unsigned  B3SOIPDagsGiven   :1;
    unsigned  B3SOIPDa1Given   :1;
    unsigned  B3SOIPDa2Given   :1;
    unsigned  B3SOIPDketaGiven   :1;    
    unsigned  B3SOIPDnsubGiven   :1;
    unsigned  B3SOIPDnpeakGiven   :1;
    unsigned  B3SOIPDngateGiven   :1;
    unsigned  B3SOIPDgamma1Given   :1;
    unsigned  B3SOIPDgamma2Given   :1;
    unsigned  B3SOIPDvbxGiven   :1;
    unsigned  B3SOIPDvbmGiven   :1;
    unsigned  B3SOIPDxtGiven   :1;
    unsigned  B3SOIPDk1Given   :1;
    unsigned  B3SOIPDkt1Given   :1;
    unsigned  B3SOIPDkt1lGiven   :1;
    unsigned  B3SOIPDkt2Given   :1;
    unsigned  B3SOIPDk2Given   :1;
    unsigned  B3SOIPDk3Given   :1;
    unsigned  B3SOIPDk3bGiven   :1;
    unsigned  B3SOIPDw0Given   :1;
    unsigned  B3SOIPDnlxGiven   :1;
    unsigned  B3SOIPDdvt0Given   :1;   
    unsigned  B3SOIPDdvt1Given   :1;     
    unsigned  B3SOIPDdvt2Given   :1;     
    unsigned  B3SOIPDdvt0wGiven   :1;   
    unsigned  B3SOIPDdvt1wGiven   :1;     
    unsigned  B3SOIPDdvt2wGiven   :1;     
    unsigned  B3SOIPDdroutGiven   :1;     
    unsigned  B3SOIPDdsubGiven   :1;     
    unsigned  B3SOIPDvth0Given   :1;
    unsigned  B3SOIPDuaGiven   :1;
    unsigned  B3SOIPDua1Given   :1;
    unsigned  B3SOIPDubGiven   :1;
    unsigned  B3SOIPDub1Given   :1;
    unsigned  B3SOIPDucGiven   :1;
    unsigned  B3SOIPDuc1Given   :1;
    unsigned  B3SOIPDu0Given   :1;
    unsigned  B3SOIPDuteGiven   :1;
    unsigned  B3SOIPDvoffGiven   :1;
    unsigned  B3SOIPDrdswGiven   :1;      
    unsigned  B3SOIPDprwgGiven   :1;      
    unsigned  B3SOIPDprwbGiven   :1;      
    unsigned  B3SOIPDprtGiven   :1;      
    unsigned  B3SOIPDeta0Given   :1;    
    unsigned  B3SOIPDetabGiven   :1;    
    unsigned  B3SOIPDpclmGiven   :1;   
    unsigned  B3SOIPDpdibl1Given   :1;   
    unsigned  B3SOIPDpdibl2Given   :1;  
    unsigned  B3SOIPDpdiblbGiven   :1;  
    unsigned  B3SOIPDpvagGiven   :1;    
    unsigned  B3SOIPDdeltaGiven  :1;     
    unsigned  B3SOIPDwrGiven   :1;
    unsigned  B3SOIPDdwgGiven   :1;
    unsigned  B3SOIPDdwbGiven   :1;
    unsigned  B3SOIPDb0Given   :1;
    unsigned  B3SOIPDb1Given   :1;
    unsigned  B3SOIPDalpha0Given   :1;


/* v2.2 release */
    unsigned  B3SOIPDwth0Given   :1;
    unsigned  B3SOIPDrhaloGiven  :1;
    unsigned  B3SOIPDntoxGiven   :1;
    unsigned  B3SOIPDtoxrefGiven   :1;
    unsigned  B3SOIPDebgGiven     :1;
    unsigned  B3SOIPDvevbGiven   :1;
    unsigned  B3SOIPDalphaGB1Given :1;
    unsigned  B3SOIPDbetaGB1Given  :1;
    unsigned  B3SOIPDvgb1Given        :1;
    unsigned  B3SOIPDvecbGiven        :1;
    unsigned  B3SOIPDalphaGB2Given    :1;
    unsigned  B3SOIPDbetaGB2Given     :1;
    unsigned  B3SOIPDvgb2Given        :1;
    unsigned  B3SOIPDtoxqmGiven  :1;
    unsigned  B3SOIPDigModGiven  :1;
    unsigned  B3SOIPDvoxhGiven   :1;
    unsigned  B3SOIPDdeltavoxGiven :1;


/* v2.0 release */
    unsigned  B3SOIPDk1w1Given   :1;   
    unsigned  B3SOIPDk1w2Given   :1;
    unsigned  B3SOIPDketasGiven  :1;
    unsigned  B3SOIPDdwbcGiven  :1;
    unsigned  B3SOIPDbeta0Given  :1;
    unsigned  B3SOIPDbeta1Given  :1;
    unsigned  B3SOIPDbeta2Given  :1;
    unsigned  B3SOIPDvdsatii0Given  :1;
    unsigned  B3SOIPDtiiGiven  :1;
    unsigned  B3SOIPDliiGiven  :1;
    unsigned  B3SOIPDsii0Given  :1;
    unsigned  B3SOIPDsii1Given  :1;
    unsigned  B3SOIPDsii2Given  :1;
    unsigned  B3SOIPDsiidGiven  :1;
    unsigned  B3SOIPDfbjtiiGiven :1;
    unsigned  B3SOIPDesatiiGiven :1;
    unsigned  B3SOIPDntunGiven  :1;
    unsigned  B3SOIPDnrecf0Given  :1;
    unsigned  B3SOIPDnrecr0Given  :1;
    unsigned  B3SOIPDisbjtGiven  :1;
    unsigned  B3SOIPDisdifGiven  :1;
    unsigned  B3SOIPDisrecGiven  :1;
    unsigned  B3SOIPDistunGiven  :1;
    unsigned  B3SOIPDlnGiven  :1;
    unsigned  B3SOIPDvrec0Given  :1;
    unsigned  B3SOIPDvtun0Given  :1;
    unsigned  B3SOIPDnbjtGiven  :1;
    unsigned  B3SOIPDlbjt0Given  :1;
    unsigned  B3SOIPDldif0Given  :1;
    unsigned  B3SOIPDvabjtGiven  :1;
    unsigned  B3SOIPDaelyGiven  :1;
    unsigned  B3SOIPDahliGiven  :1;
    unsigned  B3SOIPDrbodyGiven :1;
    unsigned  B3SOIPDrbshGiven  :1;
    unsigned  B3SOIPDndifGiven  :1;
    unsigned  B3SOIPDntrecfGiven  :1;
    unsigned  B3SOIPDntrecrGiven  :1;
    unsigned  B3SOIPDdlcbGiven    :1;
    unsigned  B3SOIPDfbodyGiven   :1;
    unsigned  B3SOIPDtcjswgGiven  :1;
    unsigned  B3SOIPDtpbswgGiven  :1;
    unsigned  B3SOIPDacdeGiven  :1;
    unsigned  B3SOIPDmoinGiven  :1;
    unsigned  B3SOIPDdelvtGiven  :1;
    unsigned  B3SOIPDdlbgGiven  :1;

    
    /* CV model */
    unsigned  B3SOIPDcgslGiven   :1;
    unsigned  B3SOIPDcgdlGiven   :1;
    unsigned  B3SOIPDckappaGiven   :1;
    unsigned  B3SOIPDcfGiven   :1;
    unsigned  B3SOIPDclcGiven   :1;
    unsigned  B3SOIPDcleGiven   :1;
    unsigned  B3SOIPDdwcGiven   :1;
    unsigned  B3SOIPDdlcGiven   :1;

/* Added for binning - START2 */
    /* Length Dependence */
    unsigned  B3SOIPDlnpeakGiven   :1;        
    unsigned  B3SOIPDlnsubGiven   :1;
    unsigned  B3SOIPDlngateGiven   :1;        
    unsigned  B3SOIPDlvth0Given   :1;
    unsigned  B3SOIPDlk1Given   :1;
    unsigned  B3SOIPDlk1w1Given   :1;
    unsigned  B3SOIPDlk1w2Given   :1;
    unsigned  B3SOIPDlk2Given   :1;
    unsigned  B3SOIPDlk3Given   :1;
    unsigned  B3SOIPDlk3bGiven   :1;
    unsigned  B3SOIPDlkb1Given   :1;
    unsigned  B3SOIPDlw0Given   :1;
    unsigned  B3SOIPDlnlxGiven   :1;
    unsigned  B3SOIPDldvt0Given   :1;      
    unsigned  B3SOIPDldvt1Given   :1;      
    unsigned  B3SOIPDldvt2Given   :1;      
    unsigned  B3SOIPDldvt0wGiven   :1;      
    unsigned  B3SOIPDldvt1wGiven   :1;      
    unsigned  B3SOIPDldvt2wGiven   :1;      
    unsigned  B3SOIPDlu0Given   :1;
    unsigned  B3SOIPDluaGiven   :1;
    unsigned  B3SOIPDlubGiven   :1;
    unsigned  B3SOIPDlucGiven   :1;
    unsigned  B3SOIPDlvsatGiven   :1;         
    unsigned  B3SOIPDla0Given   :1;   
    unsigned  B3SOIPDlagsGiven   :1;      
    unsigned  B3SOIPDlb0Given   :1;
    unsigned  B3SOIPDlb1Given   :1;
    unsigned  B3SOIPDlketaGiven   :1;
    unsigned  B3SOIPDlketasGiven   :1;
    unsigned  B3SOIPDla1Given   :1;         
    unsigned  B3SOIPDla2Given   :1;         
    unsigned  B3SOIPDlrdswGiven   :1;       
    unsigned  B3SOIPDlprwbGiven   :1;
    unsigned  B3SOIPDlprwgGiven   :1;
    unsigned  B3SOIPDlwrGiven   :1;
    unsigned  B3SOIPDlnfactorGiven   :1;      
    unsigned  B3SOIPDldwgGiven   :1;
    unsigned  B3SOIPDldwbGiven   :1;
    unsigned  B3SOIPDlvoffGiven   :1;
    unsigned  B3SOIPDleta0Given   :1;         
    unsigned  B3SOIPDletabGiven   :1;         
    unsigned  B3SOIPDldsubGiven   :1;      
    unsigned  B3SOIPDlcitGiven   :1;           
    unsigned  B3SOIPDlcdscGiven   :1;           
    unsigned  B3SOIPDlcdscbGiven   :1; 
    unsigned  B3SOIPDlcdscdGiven   :1;          
    unsigned  B3SOIPDlpclmGiven   :1;      
    unsigned  B3SOIPDlpdibl1Given   :1;      
    unsigned  B3SOIPDlpdibl2Given   :1;      
    unsigned  B3SOIPDlpdiblbGiven   :1;      
    unsigned  B3SOIPDldroutGiven   :1;      
    unsigned  B3SOIPDlpvagGiven   :1;       
    unsigned  B3SOIPDldeltaGiven   :1;
    unsigned  B3SOIPDlalpha0Given   :1;
    unsigned  B3SOIPDlfbjtiiGiven   :1;
    unsigned  B3SOIPDlbeta0Given   :1;
    unsigned  B3SOIPDlbeta1Given   :1;
    unsigned  B3SOIPDlbeta2Given   :1;         
    unsigned  B3SOIPDlvdsatii0Given   :1;     
    unsigned  B3SOIPDlliiGiven   :1;      
    unsigned  B3SOIPDlesatiiGiven   :1;     
    unsigned  B3SOIPDlsii0Given   :1;      
    unsigned  B3SOIPDlsii1Given   :1;       
    unsigned  B3SOIPDlsii2Given   :1;       
    unsigned  B3SOIPDlsiidGiven   :1;
    unsigned  B3SOIPDlagidlGiven   :1;
    unsigned  B3SOIPDlbgidlGiven   :1;
    unsigned  B3SOIPDlngidlGiven   :1;
    unsigned  B3SOIPDlntunGiven   :1;
    unsigned  B3SOIPDlndiodeGiven   :1;
    unsigned  B3SOIPDlnrecf0Given   :1;
    unsigned  B3SOIPDlnrecr0Given   :1;
    unsigned  B3SOIPDlisbjtGiven   :1;       
    unsigned  B3SOIPDlisdifGiven   :1;
    unsigned  B3SOIPDlisrecGiven   :1;       
    unsigned  B3SOIPDlistunGiven   :1;       
    unsigned  B3SOIPDlvrec0Given   :1;
    unsigned  B3SOIPDlvtun0Given   :1;
    unsigned  B3SOIPDlnbjtGiven   :1;
    unsigned  B3SOIPDllbjt0Given   :1;
    unsigned  B3SOIPDlvabjtGiven   :1;
    unsigned  B3SOIPDlaelyGiven   :1;
    unsigned  B3SOIPDlahliGiven   :1;
    /* CV model */
    unsigned  B3SOIPDlvsdfbGiven   :1;
    unsigned  B3SOIPDlvsdthGiven   :1;
    unsigned  B3SOIPDldelvtGiven   :1;
    unsigned  B3SOIPDlacdeGiven   :1;
    unsigned  B3SOIPDlmoinGiven   :1;

    /* Width Dependence */
    unsigned  B3SOIPDwnpeakGiven   :1;        
    unsigned  B3SOIPDwnsubGiven   :1;
    unsigned  B3SOIPDwngateGiven   :1;        
    unsigned  B3SOIPDwvth0Given   :1;
    unsigned  B3SOIPDwk1Given   :1;
    unsigned  B3SOIPDwk1w1Given   :1;
    unsigned  B3SOIPDwk1w2Given   :1;
    unsigned  B3SOIPDwk2Given   :1;
    unsigned  B3SOIPDwk3Given   :1;
    unsigned  B3SOIPDwk3bGiven   :1;
    unsigned  B3SOIPDwkb1Given   :1;
    unsigned  B3SOIPDww0Given   :1;
    unsigned  B3SOIPDwnlxGiven   :1;
    unsigned  B3SOIPDwdvt0Given   :1;      
    unsigned  B3SOIPDwdvt1Given   :1;      
    unsigned  B3SOIPDwdvt2Given   :1;      
    unsigned  B3SOIPDwdvt0wGiven   :1;      
    unsigned  B3SOIPDwdvt1wGiven   :1;      
    unsigned  B3SOIPDwdvt2wGiven   :1;      
    unsigned  B3SOIPDwu0Given   :1;
    unsigned  B3SOIPDwuaGiven   :1;
    unsigned  B3SOIPDwubGiven   :1;
    unsigned  B3SOIPDwucGiven   :1;
    unsigned  B3SOIPDwvsatGiven   :1;         
    unsigned  B3SOIPDwa0Given   :1;   
    unsigned  B3SOIPDwagsGiven   :1;      
    unsigned  B3SOIPDwb0Given   :1;
    unsigned  B3SOIPDwb1Given   :1;
    unsigned  B3SOIPDwketaGiven   :1;
    unsigned  B3SOIPDwketasGiven   :1;
    unsigned  B3SOIPDwa1Given   :1;         
    unsigned  B3SOIPDwa2Given   :1;         
    unsigned  B3SOIPDwrdswGiven   :1;       
    unsigned  B3SOIPDwprwbGiven   :1;
    unsigned  B3SOIPDwprwgGiven   :1;
    unsigned  B3SOIPDwwrGiven   :1;
    unsigned  B3SOIPDwnfactorGiven   :1;      
    unsigned  B3SOIPDwdwgGiven   :1;
    unsigned  B3SOIPDwdwbGiven   :1;
    unsigned  B3SOIPDwvoffGiven   :1;
    unsigned  B3SOIPDweta0Given   :1;         
    unsigned  B3SOIPDwetabGiven   :1;         
    unsigned  B3SOIPDwdsubGiven   :1;      
    unsigned  B3SOIPDwcitGiven   :1;           
    unsigned  B3SOIPDwcdscGiven   :1;           
    unsigned  B3SOIPDwcdscbGiven   :1; 
    unsigned  B3SOIPDwcdscdGiven   :1;          
    unsigned  B3SOIPDwpclmGiven   :1;      
    unsigned  B3SOIPDwpdibl1Given   :1;      
    unsigned  B3SOIPDwpdibl2Given   :1;      
    unsigned  B3SOIPDwpdiblbGiven   :1;      
    unsigned  B3SOIPDwdroutGiven   :1;      
    unsigned  B3SOIPDwpvagGiven   :1;       
    unsigned  B3SOIPDwdeltaGiven   :1;
    unsigned  B3SOIPDwalpha0Given   :1;
    unsigned  B3SOIPDwfbjtiiGiven   :1;
    unsigned  B3SOIPDwbeta0Given   :1;
    unsigned  B3SOIPDwbeta1Given   :1;
    unsigned  B3SOIPDwbeta2Given   :1;         
    unsigned  B3SOIPDwvdsatii0Given   :1;     
    unsigned  B3SOIPDwliiGiven   :1;      
    unsigned  B3SOIPDwesatiiGiven   :1;     
    unsigned  B3SOIPDwsii0Given   :1;      
    unsigned  B3SOIPDwsii1Given   :1;       
    unsigned  B3SOIPDwsii2Given   :1;       
    unsigned  B3SOIPDwsiidGiven   :1;
    unsigned  B3SOIPDwagidlGiven   :1;
    unsigned  B3SOIPDwbgidlGiven   :1;
    unsigned  B3SOIPDwngidlGiven   :1;
    unsigned  B3SOIPDwntunGiven   :1;
    unsigned  B3SOIPDwndiodeGiven   :1;
    unsigned  B3SOIPDwnrecf0Given   :1;
    unsigned  B3SOIPDwnrecr0Given   :1;
    unsigned  B3SOIPDwisbjtGiven   :1;       
    unsigned  B3SOIPDwisdifGiven   :1;
    unsigned  B3SOIPDwisrecGiven   :1;       
    unsigned  B3SOIPDwistunGiven   :1;       
    unsigned  B3SOIPDwvrec0Given   :1;
    unsigned  B3SOIPDwvtun0Given   :1;
    unsigned  B3SOIPDwnbjtGiven   :1;
    unsigned  B3SOIPDwlbjt0Given   :1;
    unsigned  B3SOIPDwvabjtGiven   :1;
    unsigned  B3SOIPDwaelyGiven   :1;
    unsigned  B3SOIPDwahliGiven   :1;
    /* CV model */
    unsigned  B3SOIPDwvsdfbGiven   :1;
    unsigned  B3SOIPDwvsdthGiven   :1;
    unsigned  B3SOIPDwdelvtGiven   :1;
    unsigned  B3SOIPDwacdeGiven   :1;
    unsigned  B3SOIPDwmoinGiven   :1;

    /* Cross-term Dependence */
    unsigned  B3SOIPDpnpeakGiven   :1;        
    unsigned  B3SOIPDpnsubGiven   :1;
    unsigned  B3SOIPDpngateGiven   :1;        
    unsigned  B3SOIPDpvth0Given   :1;
    unsigned  B3SOIPDpk1Given   :1;
    unsigned  B3SOIPDpk1w1Given   :1;
    unsigned  B3SOIPDpk1w2Given   :1;
    unsigned  B3SOIPDpk2Given   :1;
    unsigned  B3SOIPDpk3Given   :1;
    unsigned  B3SOIPDpk3bGiven   :1;
    unsigned  B3SOIPDpkb1Given   :1;
    unsigned  B3SOIPDpw0Given   :1;
    unsigned  B3SOIPDpnlxGiven   :1;
    unsigned  B3SOIPDpdvt0Given   :1;      
    unsigned  B3SOIPDpdvt1Given   :1;      
    unsigned  B3SOIPDpdvt2Given   :1;      
    unsigned  B3SOIPDpdvt0wGiven   :1;      
    unsigned  B3SOIPDpdvt1wGiven   :1;      
    unsigned  B3SOIPDpdvt2wGiven   :1;      
    unsigned  B3SOIPDpu0Given   :1;
    unsigned  B3SOIPDpuaGiven   :1;
    unsigned  B3SOIPDpubGiven   :1;
    unsigned  B3SOIPDpucGiven   :1;
    unsigned  B3SOIPDpvsatGiven   :1;         
    unsigned  B3SOIPDpa0Given   :1;   
    unsigned  B3SOIPDpagsGiven   :1;      
    unsigned  B3SOIPDpb0Given   :1;
    unsigned  B3SOIPDpb1Given   :1;
    unsigned  B3SOIPDpketaGiven   :1;
    unsigned  B3SOIPDpketasGiven   :1;
    unsigned  B3SOIPDpa1Given   :1;         
    unsigned  B3SOIPDpa2Given   :1;         
    unsigned  B3SOIPDprdswGiven   :1;       
    unsigned  B3SOIPDpprwbGiven   :1;
    unsigned  B3SOIPDpprwgGiven   :1;
    unsigned  B3SOIPDpwrGiven   :1;
    unsigned  B3SOIPDpnfactorGiven   :1;      
    unsigned  B3SOIPDpdwgGiven   :1;
    unsigned  B3SOIPDpdwbGiven   :1;
    unsigned  B3SOIPDpvoffGiven   :1;
    unsigned  B3SOIPDpeta0Given   :1;         
    unsigned  B3SOIPDpetabGiven   :1;         
    unsigned  B3SOIPDpdsubGiven   :1;      
    unsigned  B3SOIPDpcitGiven   :1;           
    unsigned  B3SOIPDpcdscGiven   :1;           
    unsigned  B3SOIPDpcdscbGiven   :1; 
    unsigned  B3SOIPDpcdscdGiven   :1;          
    unsigned  B3SOIPDppclmGiven   :1;      
    unsigned  B3SOIPDppdibl1Given   :1;      
    unsigned  B3SOIPDppdibl2Given   :1;      
    unsigned  B3SOIPDppdiblbGiven   :1;      
    unsigned  B3SOIPDpdroutGiven   :1;      
    unsigned  B3SOIPDppvagGiven   :1;       
    unsigned  B3SOIPDpdeltaGiven   :1;
    unsigned  B3SOIPDpalpha0Given   :1;
    unsigned  B3SOIPDpfbjtiiGiven   :1;
    unsigned  B3SOIPDpbeta0Given   :1;
    unsigned  B3SOIPDpbeta1Given   :1;
    unsigned  B3SOIPDpbeta2Given   :1;         
    unsigned  B3SOIPDpvdsatii0Given   :1;     
    unsigned  B3SOIPDpliiGiven   :1;      
    unsigned  B3SOIPDpesatiiGiven   :1;     
    unsigned  B3SOIPDpsii0Given   :1;      
    unsigned  B3SOIPDpsii1Given   :1;       
    unsigned  B3SOIPDpsii2Given   :1;       
    unsigned  B3SOIPDpsiidGiven   :1;
    unsigned  B3SOIPDpagidlGiven   :1;
    unsigned  B3SOIPDpbgidlGiven   :1;
    unsigned  B3SOIPDpngidlGiven   :1;
    unsigned  B3SOIPDpntunGiven   :1;
    unsigned  B3SOIPDpndiodeGiven   :1;
    unsigned  B3SOIPDpnrecf0Given   :1;
    unsigned  B3SOIPDpnrecr0Given   :1;
    unsigned  B3SOIPDpisbjtGiven   :1;       
    unsigned  B3SOIPDpisdifGiven   :1;
    unsigned  B3SOIPDpisrecGiven   :1;       
    unsigned  B3SOIPDpistunGiven   :1;       
    unsigned  B3SOIPDpvrec0Given   :1;
    unsigned  B3SOIPDpvtun0Given   :1;
    unsigned  B3SOIPDpnbjtGiven   :1;
    unsigned  B3SOIPDplbjt0Given   :1;
    unsigned  B3SOIPDpvabjtGiven   :1;
    unsigned  B3SOIPDpaelyGiven   :1;
    unsigned  B3SOIPDpahliGiven   :1;
    /* CV model */
    unsigned  B3SOIPDpvsdfbGiven   :1;
    unsigned  B3SOIPDpvsdthGiven   :1;
    unsigned  B3SOIPDpdelvtGiven   :1;
    unsigned  B3SOIPDpacdeGiven   :1;
    unsigned  B3SOIPDpmoinGiven   :1;
/* Added for binning - END2 */

    unsigned  B3SOIPDuseFringeGiven   :1;

    unsigned  B3SOIPDtnomGiven   :1;
    unsigned  B3SOIPDcgsoGiven   :1;
    unsigned  B3SOIPDcgdoGiven   :1;
    unsigned  B3SOIPDcgeoGiven   :1;
    unsigned  B3SOIPDxpartGiven   :1;
    unsigned  B3SOIPDsheetResistanceGiven   :1;
    unsigned  B3SOIPDGatesidewallJctPotentialGiven   :1;
    unsigned  B3SOIPDbodyJctGateSideGradingCoeffGiven   :1;
    unsigned  B3SOIPDunitLengthGateSidewallJctCapGiven   :1;
    unsigned  B3SOIPDcsdeswGiven :1;

    unsigned  B3SOIPDoxideTrapDensityAGiven  :1;         
    unsigned  B3SOIPDoxideTrapDensityBGiven  :1;        
    unsigned  B3SOIPDoxideTrapDensityCGiven  :1;     
    unsigned  B3SOIPDemGiven  :1;     
    unsigned  B3SOIPDefGiven  :1;     
    unsigned  B3SOIPDafGiven  :1;     
    unsigned  B3SOIPDkfGiven  :1;     
    unsigned  B3SOIPDnoifGiven  :1;     

    unsigned  B3SOIPDLintGiven   :1;
    unsigned  B3SOIPDLlGiven   :1;
    unsigned  B3SOIPDLlcGiven   :1; /* v2.2.3 */
    unsigned  B3SOIPDLlnGiven   :1;
    unsigned  B3SOIPDLwGiven   :1;
    unsigned  B3SOIPDLwcGiven   :1; /* v2.2.3 */
    unsigned  B3SOIPDLwnGiven   :1;
    unsigned  B3SOIPDLwlGiven   :1;
    unsigned  B3SOIPDLwlcGiven   :1; /* v2.2.3 */
    unsigned  B3SOIPDLminGiven   :1;
    unsigned  B3SOIPDLmaxGiven   :1;

    unsigned  B3SOIPDWintGiven   :1;
    unsigned  B3SOIPDWlGiven   :1;
    unsigned  B3SOIPDWlcGiven   :1; /* v2.2.3 */
    unsigned  B3SOIPDWlnGiven   :1;
    unsigned  B3SOIPDWwGiven   :1;
    unsigned  B3SOIPDWwcGiven   :1; /* v2.2.3 */
    unsigned  B3SOIPDWwnGiven   :1;
    unsigned  B3SOIPDWwlGiven   :1;
    unsigned  B3SOIPDWwlcGiven   :1;  /* v2.2.3 */
    unsigned  B3SOIPDWminGiven   :1;
    unsigned  B3SOIPDWmaxGiven   :1;

} B3SOIPDmodel;


#ifndef NMOS
#define NMOS 1
#define PMOS -1
#endif /*NMOS*/


/* device parameters */
#define B3SOIPD_W 1
#define B3SOIPD_L 2
#define B3SOIPD_M 31
#define B3SOIPD_AS 3
#define B3SOIPD_AD 4
#define B3SOIPD_PS 5
#define B3SOIPD_PD 6
#define B3SOIPD_NRS 7
#define B3SOIPD_NRD 8
#define B3SOIPD_OFF 9
#define B3SOIPD_IC_VBS 10
#define B3SOIPD_IC_VDS 11
#define B3SOIPD_IC_VGS 12
#define B3SOIPD_IC_VES 13
#define B3SOIPD_IC_VPS 14
#define B3SOIPD_BJTOFF 15
#define B3SOIPD_RTH0 16
#define B3SOIPD_CTH0 17
#define B3SOIPD_NRB 18
#define B3SOIPD_IC 19
#define B3SOIPD_NQSMOD 20
#define B3SOIPD_DEBUG  21

/* v2.0 release */
#define B3SOIPD_NBC    22  
#define B3SOIPD_NSEG   23
#define B3SOIPD_PDBCP  24
#define B3SOIPD_PSBCP  25
#define B3SOIPD_AGBCP  26
#define B3SOIPD_AEBCP  27
#define B3SOIPD_VBSUSR 28
#define B3SOIPD_TNODEOUT 29

/* v2.2.2 */
#define B3SOIPD_FRBODY 30


/* model parameters */
#define B3SOIPD_MOD_CAPMOD          101
#define B3SOIPD_MOD_NQSMOD          102
#define B3SOIPD_MOD_MOBMOD          103    
#define B3SOIPD_MOD_NOIMOD          104    
#define B3SOIPD_MOD_SHMOD           105
#define B3SOIPD_MOD_DDMOD           106

#define B3SOIPD_MOD_TOX             107

#define B3SOIPD_MOD_CDSC            108
#define B3SOIPD_MOD_CDSCB           109
#define B3SOIPD_MOD_CIT             110
#define B3SOIPD_MOD_NFACTOR         111
#define B3SOIPD_MOD_XJ              112
#define B3SOIPD_MOD_VSAT            113
#define B3SOIPD_MOD_AT              114
#define B3SOIPD_MOD_A0              115
#define B3SOIPD_MOD_A1              116
#define B3SOIPD_MOD_A2              117
#define B3SOIPD_MOD_KETA            118   
#define B3SOIPD_MOD_NSUB            119
#define B3SOIPD_MOD_NPEAK           120
#define B3SOIPD_MOD_NGATE           121
#define B3SOIPD_MOD_GAMMA1          122
#define B3SOIPD_MOD_GAMMA2          123
#define B3SOIPD_MOD_VBX             124
#define B3SOIPD_MOD_BINUNIT         125    

#define B3SOIPD_MOD_VBM             126

#define B3SOIPD_MOD_XT              127
#define B3SOIPD_MOD_K1              129
#define B3SOIPD_MOD_KT1             130
#define B3SOIPD_MOD_KT1L            131
#define B3SOIPD_MOD_K2              132
#define B3SOIPD_MOD_KT2             133
#define B3SOIPD_MOD_K3              134
#define B3SOIPD_MOD_K3B             135
#define B3SOIPD_MOD_W0              136
#define B3SOIPD_MOD_NLX             137

#define B3SOIPD_MOD_DVT0            138
#define B3SOIPD_MOD_DVT1            139
#define B3SOIPD_MOD_DVT2            140

#define B3SOIPD_MOD_DVT0W           141
#define B3SOIPD_MOD_DVT1W           142
#define B3SOIPD_MOD_DVT2W           143

#define B3SOIPD_MOD_DROUT           144
#define B3SOIPD_MOD_DSUB            145
#define B3SOIPD_MOD_VTH0            146
#define B3SOIPD_MOD_UA              147
#define B3SOIPD_MOD_UA1             148
#define B3SOIPD_MOD_UB              149
#define B3SOIPD_MOD_UB1             150
#define B3SOIPD_MOD_UC              151
#define B3SOIPD_MOD_UC1             152
#define B3SOIPD_MOD_U0              153
#define B3SOIPD_MOD_UTE             154
#define B3SOIPD_MOD_VOFF            155
#define B3SOIPD_MOD_DELTA           156
#define B3SOIPD_MOD_RDSW            157
#define B3SOIPD_MOD_PRT             158
#define B3SOIPD_MOD_LDD             159
#define B3SOIPD_MOD_ETA             160
#define B3SOIPD_MOD_ETA0            161
#define B3SOIPD_MOD_ETAB            162
#define B3SOIPD_MOD_PCLM            163
#define B3SOIPD_MOD_PDIBL1          164
#define B3SOIPD_MOD_PDIBL2          165
#define B3SOIPD_MOD_PSCBE1          166
#define B3SOIPD_MOD_PSCBE2          167
#define B3SOIPD_MOD_PVAG            168
#define B3SOIPD_MOD_WR              169
#define B3SOIPD_MOD_DWG             170
#define B3SOIPD_MOD_DWB             171
#define B3SOIPD_MOD_B0              172
#define B3SOIPD_MOD_B1              173
#define B3SOIPD_MOD_ALPHA0          174
#define B3SOIPD_MOD_PDIBLB          178

#define B3SOIPD_MOD_PRWG            179
#define B3SOIPD_MOD_PRWB            180

#define B3SOIPD_MOD_CDSCD           181
#define B3SOIPD_MOD_AGS             182

#define B3SOIPD_MOD_FRINGE          184
#define B3SOIPD_MOD_CGSL            186
#define B3SOIPD_MOD_CGDL            187
#define B3SOIPD_MOD_CKAPPA          188
#define B3SOIPD_MOD_CF              189
#define B3SOIPD_MOD_CLC             190
#define B3SOIPD_MOD_CLE             191
#define B3SOIPD_MOD_PARAMCHK        192
#define B3SOIPD_MOD_VERSION         193

#define B3SOIPD_MOD_TBOX            195
#define B3SOIPD_MOD_TSI             196
#define B3SOIPD_MOD_KB1             197
#define B3SOIPD_MOD_KB3             198
#define B3SOIPD_MOD_DVBD0           199
#define B3SOIPD_MOD_DVBD1           200
#define B3SOIPD_MOD_DELP            201
#define B3SOIPD_MOD_VBSA            202
#define B3SOIPD_MOD_RBODY           204
#define B3SOIPD_MOD_ADICE0          205
#define B3SOIPD_MOD_ABP             206
#define B3SOIPD_MOD_MXC             207
#define B3SOIPD_MOD_RTH0            208
#define B3SOIPD_MOD_CTH0            209
#define B3SOIPD_MOD_ALPHA1	  214
#define B3SOIPD_MOD_NGIDL           215
#define B3SOIPD_MOD_AGIDL           216
#define B3SOIPD_MOD_BGIDL           217
#define B3SOIPD_MOD_NDIODE          218
#define B3SOIPD_MOD_LDIOF           219
#define B3SOIPD_MOD_LDIOR           220
#define B3SOIPD_MOD_NTUN            221
#define B3SOIPD_MOD_ISBJT           222
#define B3SOIPD_MOD_ISDIF           223
#define B3SOIPD_MOD_ISREC           224
#define B3SOIPD_MOD_ISTUN           225
#define B3SOIPD_MOD_XBJT            226
#define B3SOIPD_MOD_XDIF            227
#define B3SOIPD_MOD_XREC            228
#define B3SOIPD_MOD_XTUN            229
#define B3SOIPD_MOD_TT              232
#define B3SOIPD_MOD_VSDTH           233
#define B3SOIPD_MOD_VSDFB           234
#define B3SOIPD_MOD_ASD             235
#define B3SOIPD_MOD_CSDMIN          236
#define B3SOIPD_MOD_RBSH            237
#define B3SOIPD_MOD_ESATII          238


/* v2.0 release */
#define B3SOIPD_MOD_K1W1            239     
#define B3SOIPD_MOD_K1W2            240
#define B3SOIPD_MOD_KETAS           241
#define B3SOIPD_MOD_DWBC            242
#define B3SOIPD_MOD_BETA0           243
#define B3SOIPD_MOD_BETA1           244
#define B3SOIPD_MOD_BETA2           245
#define B3SOIPD_MOD_VDSATII0        246
#define B3SOIPD_MOD_TII             247
#define B3SOIPD_MOD_LII             248
#define B3SOIPD_MOD_SII0            249
#define B3SOIPD_MOD_SII1            250
#define B3SOIPD_MOD_SII2            251
#define B3SOIPD_MOD_SIID            252
#define B3SOIPD_MOD_FBJTII          253
#define B3SOIPD_MOD_NRECF0          255
#define B3SOIPD_MOD_NRECR0          256
#define B3SOIPD_MOD_LN              257
#define B3SOIPD_MOD_VREC0           258
#define B3SOIPD_MOD_VTUN0           259
#define B3SOIPD_MOD_NBJT            260
#define B3SOIPD_MOD_LBJT0           261
#define B3SOIPD_MOD_VABJT           262
#define B3SOIPD_MOD_AELY            263
#define B3SOIPD_MOD_AHLI            264
#define B3SOIPD_MOD_NTRECF          265
#define B3SOIPD_MOD_NTRECR          266
#define B3SOIPD_MOD_DLCB            267
#define B3SOIPD_MOD_FBODY           268
#define B3SOIPD_MOD_NDIF            269
#define B3SOIPD_MOD_TCJSWG          270
#define B3SOIPD_MOD_TPBSWG          271
#define B3SOIPD_MOD_ACDE            272
#define B3SOIPD_MOD_MOIN            273
#define B3SOIPD_MOD_DELVT           274
#define B3SOIPD_MOD_DLBG            275
#define B3SOIPD_MOD_LDIF0           276


/* v2.2 release */
#define B3SOIPD_MOD_WTH0            277
#define B3SOIPD_MOD_RHALO           278
#define B3SOIPD_MOD_NTOX            279
#define B3SOIPD_MOD_TOXREF          280
#define B3SOIPD_MOD_EBG             281
#define B3SOIPD_MOD_VEVB            282
#define B3SOIPD_MOD_ALPHAGB1        283
#define B3SOIPD_MOD_BETAGB1         284
#define B3SOIPD_MOD_VGB1            285
#define B3SOIPD_MOD_VECB            286
#define B3SOIPD_MOD_ALPHAGB2        287
#define B3SOIPD_MOD_BETAGB2         288
#define B3SOIPD_MOD_VGB2            289
#define B3SOIPD_MOD_TOXQM           290
#define B3SOIPD_MOD_IGMOD           291
#define B3SOIPD_MOD_VOXH            292
#define B3SOIPD_MOD_DELTAVOX        293


/* Added for binning - START3 */
/* Length dependence */
#define B3SOIPD_MOD_LNPEAK           301
#define B3SOIPD_MOD_LNSUB            302
#define B3SOIPD_MOD_LNGATE           303
#define B3SOIPD_MOD_LVTH0            304
#define B3SOIPD_MOD_LK1              305     
#define B3SOIPD_MOD_LK1W1            306
#define B3SOIPD_MOD_LK1W2            307
#define B3SOIPD_MOD_LK2              308
#define B3SOIPD_MOD_LK3              309
#define B3SOIPD_MOD_LK3B             310
#define B3SOIPD_MOD_LKB1             311
#define B3SOIPD_MOD_LW0              312
#define B3SOIPD_MOD_LNLX             313
#define B3SOIPD_MOD_LDVT0            314
#define B3SOIPD_MOD_LDVT1            315
#define B3SOIPD_MOD_LDVT2            316
#define B3SOIPD_MOD_LDVT0W           317
#define B3SOIPD_MOD_LDVT1W           318
#define B3SOIPD_MOD_LDVT2W           319
#define B3SOIPD_MOD_LU0              320
#define B3SOIPD_MOD_LUA              321
#define B3SOIPD_MOD_LUB              322
#define B3SOIPD_MOD_LUC              323
#define B3SOIPD_MOD_LVSAT            324
#define B3SOIPD_MOD_LA0              325
#define B3SOIPD_MOD_LAGS             326
#define B3SOIPD_MOD_LB0              327
#define B3SOIPD_MOD_LB1              328
#define B3SOIPD_MOD_LKETA            329   
#define B3SOIPD_MOD_LKETAS           330   
#define B3SOIPD_MOD_LA1              331
#define B3SOIPD_MOD_LA2              332
#define B3SOIPD_MOD_LRDSW            333
#define B3SOIPD_MOD_LPRWB            334
#define B3SOIPD_MOD_LPRWG            335
#define B3SOIPD_MOD_LWR              336
#define B3SOIPD_MOD_LNFACTOR         337
#define B3SOIPD_MOD_LDWG             338
#define B3SOIPD_MOD_LDWB             339
#define B3SOIPD_MOD_LVOFF            340
#define B3SOIPD_MOD_LETA0            341
#define B3SOIPD_MOD_LETAB            342
#define B3SOIPD_MOD_LDSUB            343
#define B3SOIPD_MOD_LCIT             344
#define B3SOIPD_MOD_LCDSC            345
#define B3SOIPD_MOD_LCDSCB           346
#define B3SOIPD_MOD_LCDSCD           347
#define B3SOIPD_MOD_LPCLM            348
#define B3SOIPD_MOD_LPDIBL1          349
#define B3SOIPD_MOD_LPDIBL2          350
#define B3SOIPD_MOD_LPDIBLB          351
#define B3SOIPD_MOD_LDROUT           352
#define B3SOIPD_MOD_LPVAG            353
#define B3SOIPD_MOD_LDELTA           354
#define B3SOIPD_MOD_LALPHA0          355
#define B3SOIPD_MOD_LFBJTII          356
#define B3SOIPD_MOD_LBETA0           357
#define B3SOIPD_MOD_LBETA1           358
#define B3SOIPD_MOD_LBETA2           359
#define B3SOIPD_MOD_LVDSATII0        360
#define B3SOIPD_MOD_LLII             361
#define B3SOIPD_MOD_LESATII          362
#define B3SOIPD_MOD_LSII0            363
#define B3SOIPD_MOD_LSII1            364
#define B3SOIPD_MOD_LSII2            365
#define B3SOIPD_MOD_LSIID            366
#define B3SOIPD_MOD_LAGIDL           367
#define B3SOIPD_MOD_LBGIDL           368
#define B3SOIPD_MOD_LNGIDL           369
#define B3SOIPD_MOD_LNTUN            370
#define B3SOIPD_MOD_LNDIODE          371
#define B3SOIPD_MOD_LNRECF0          372
#define B3SOIPD_MOD_LNRECR0          373
#define B3SOIPD_MOD_LISBJT           374
#define B3SOIPD_MOD_LISDIF           375
#define B3SOIPD_MOD_LISREC           376
#define B3SOIPD_MOD_LISTUN           377
#define B3SOIPD_MOD_LVREC0           378
#define B3SOIPD_MOD_LVTUN0           379
#define B3SOIPD_MOD_LNBJT            380
#define B3SOIPD_MOD_LLBJT0           381
#define B3SOIPD_MOD_LVABJT           382
#define B3SOIPD_MOD_LAELY            383
#define B3SOIPD_MOD_LAHLI            384
#define B3SOIPD_MOD_LVSDFB           385
#define B3SOIPD_MOD_LVSDTH           386
#define B3SOIPD_MOD_LDELVT           387
#define B3SOIPD_MOD_LACDE            388
#define B3SOIPD_MOD_LMOIN            389

/* Width dependence */
#define B3SOIPD_MOD_WNPEAK           401
#define B3SOIPD_MOD_WNSUB            402
#define B3SOIPD_MOD_WNGATE           403
#define B3SOIPD_MOD_WVTH0            404
#define B3SOIPD_MOD_WK1              405     
#define B3SOIPD_MOD_WK1W1            406
#define B3SOIPD_MOD_WK1W2            407
#define B3SOIPD_MOD_WK2              408
#define B3SOIPD_MOD_WK3              409
#define B3SOIPD_MOD_WK3B             410
#define B3SOIPD_MOD_WKB1             411
#define B3SOIPD_MOD_WW0              412
#define B3SOIPD_MOD_WNLX             413
#define B3SOIPD_MOD_WDVT0            414
#define B3SOIPD_MOD_WDVT1            415
#define B3SOIPD_MOD_WDVT2            416
#define B3SOIPD_MOD_WDVT0W           417
#define B3SOIPD_MOD_WDVT1W           418
#define B3SOIPD_MOD_WDVT2W           419
#define B3SOIPD_MOD_WU0              420
#define B3SOIPD_MOD_WUA              421
#define B3SOIPD_MOD_WUB              422
#define B3SOIPD_MOD_WUC              423
#define B3SOIPD_MOD_WVSAT            424
#define B3SOIPD_MOD_WA0              425
#define B3SOIPD_MOD_WAGS             426
#define B3SOIPD_MOD_WB0              427
#define B3SOIPD_MOD_WB1              428
#define B3SOIPD_MOD_WKETA            429   
#define B3SOIPD_MOD_WKETAS           430   
#define B3SOIPD_MOD_WA1              431
#define B3SOIPD_MOD_WA2              432
#define B3SOIPD_MOD_WRDSW            433
#define B3SOIPD_MOD_WPRWB            434
#define B3SOIPD_MOD_WPRWG            435
#define B3SOIPD_MOD_WWR              436
#define B3SOIPD_MOD_WNFACTOR         437
#define B3SOIPD_MOD_WDWG             438
#define B3SOIPD_MOD_WDWB             439
#define B3SOIPD_MOD_WVOFF            440
#define B3SOIPD_MOD_WETA0            441
#define B3SOIPD_MOD_WETAB            442
#define B3SOIPD_MOD_WDSUB            443
#define B3SOIPD_MOD_WCIT             444
#define B3SOIPD_MOD_WCDSC            445
#define B3SOIPD_MOD_WCDSCB           446
#define B3SOIPD_MOD_WCDSCD           447
#define B3SOIPD_MOD_WPCLM            448
#define B3SOIPD_MOD_WPDIBL1          449
#define B3SOIPD_MOD_WPDIBL2          450
#define B3SOIPD_MOD_WPDIBLB          451
#define B3SOIPD_MOD_WDROUT           452
#define B3SOIPD_MOD_WPVAG            453
#define B3SOIPD_MOD_WDELTA           454
#define B3SOIPD_MOD_WALPHA0          455
#define B3SOIPD_MOD_WFBJTII          456
#define B3SOIPD_MOD_WBETA0           457
#define B3SOIPD_MOD_WBETA1           458
#define B3SOIPD_MOD_WBETA2           459
#define B3SOIPD_MOD_WVDSATII0        460
#define B3SOIPD_MOD_WLII             461
#define B3SOIPD_MOD_WESATII          462
#define B3SOIPD_MOD_WSII0            463
#define B3SOIPD_MOD_WSII1            464
#define B3SOIPD_MOD_WSII2            465
#define B3SOIPD_MOD_WSIID            466
#define B3SOIPD_MOD_WAGIDL           467
#define B3SOIPD_MOD_WBGIDL           468
#define B3SOIPD_MOD_WNGIDL           469
#define B3SOIPD_MOD_WNTUN            470
#define B3SOIPD_MOD_WNDIODE          471
#define B3SOIPD_MOD_WNRECF0          472
#define B3SOIPD_MOD_WNRECR0          473
#define B3SOIPD_MOD_WISBJT           474
#define B3SOIPD_MOD_WISDIF           475
#define B3SOIPD_MOD_WISREC           476
#define B3SOIPD_MOD_WISTUN           477
#define B3SOIPD_MOD_WVREC0           478
#define B3SOIPD_MOD_WVTUN0           479
#define B3SOIPD_MOD_WNBJT            480
#define B3SOIPD_MOD_WLBJT0           481
#define B3SOIPD_MOD_WVABJT           482
#define B3SOIPD_MOD_WAELY            483
#define B3SOIPD_MOD_WAHLI            484
#define B3SOIPD_MOD_WVSDFB           485
#define B3SOIPD_MOD_WVSDTH           486
#define B3SOIPD_MOD_WDELVT           487
#define B3SOIPD_MOD_WACDE            488
#define B3SOIPD_MOD_WMOIN            489

/* Cross-term dependence */
#define B3SOIPD_MOD_PNPEAK           501
#define B3SOIPD_MOD_PNSUB            502
#define B3SOIPD_MOD_PNGATE           503
#define B3SOIPD_MOD_PVTH0            504
#define B3SOIPD_MOD_PK1              505     
#define B3SOIPD_MOD_PK1W1            506
#define B3SOIPD_MOD_PK1W2            507
#define B3SOIPD_MOD_PK2              508
#define B3SOIPD_MOD_PK3              509
#define B3SOIPD_MOD_PK3B             510
#define B3SOIPD_MOD_PKB1             511
#define B3SOIPD_MOD_PW0              512
#define B3SOIPD_MOD_PNLX             513
#define B3SOIPD_MOD_PDVT0            514
#define B3SOIPD_MOD_PDVT1            515
#define B3SOIPD_MOD_PDVT2            516
#define B3SOIPD_MOD_PDVT0W           517
#define B3SOIPD_MOD_PDVT1W           518
#define B3SOIPD_MOD_PDVT2W           519
#define B3SOIPD_MOD_PU0              520
#define B3SOIPD_MOD_PUA              521
#define B3SOIPD_MOD_PUB              522
#define B3SOIPD_MOD_PUC              523
#define B3SOIPD_MOD_PVSAT            524
#define B3SOIPD_MOD_PA0              525
#define B3SOIPD_MOD_PAGS             526
#define B3SOIPD_MOD_PB0              527
#define B3SOIPD_MOD_PB1              528
#define B3SOIPD_MOD_PKETA            529   
#define B3SOIPD_MOD_PKETAS           530   
#define B3SOIPD_MOD_PA1              531
#define B3SOIPD_MOD_PA2              532
#define B3SOIPD_MOD_PRDSW            533
#define B3SOIPD_MOD_PPRWB            534
#define B3SOIPD_MOD_PPRWG            535
#define B3SOIPD_MOD_PWR              536
#define B3SOIPD_MOD_PNFACTOR         537
#define B3SOIPD_MOD_PDWG             538
#define B3SOIPD_MOD_PDWB             539
#define B3SOIPD_MOD_PVOFF            540
#define B3SOIPD_MOD_PETA0            541
#define B3SOIPD_MOD_PETAB            542
#define B3SOIPD_MOD_PDSUB            543
#define B3SOIPD_MOD_PCIT             544
#define B3SOIPD_MOD_PCDSC            545
#define B3SOIPD_MOD_PCDSCB           546
#define B3SOIPD_MOD_PCDSCD           547
#define B3SOIPD_MOD_PPCLM            548
#define B3SOIPD_MOD_PPDIBL1          549
#define B3SOIPD_MOD_PPDIBL2          550
#define B3SOIPD_MOD_PPDIBLB          551
#define B3SOIPD_MOD_PDROUT           552
#define B3SOIPD_MOD_PPVAG            553
#define B3SOIPD_MOD_PDELTA           554
#define B3SOIPD_MOD_PALPHA0          555
#define B3SOIPD_MOD_PFBJTII          556
#define B3SOIPD_MOD_PBETA0           557
#define B3SOIPD_MOD_PBETA1           558
#define B3SOIPD_MOD_PBETA2           559
#define B3SOIPD_MOD_PVDSATII0        560
#define B3SOIPD_MOD_PLII             561
#define B3SOIPD_MOD_PESATII          562
#define B3SOIPD_MOD_PSII0            563
#define B3SOIPD_MOD_PSII1            564
#define B3SOIPD_MOD_PSII2            565
#define B3SOIPD_MOD_PSIID            566
#define B3SOIPD_MOD_PAGIDL           567
#define B3SOIPD_MOD_PBGIDL           568
#define B3SOIPD_MOD_PNGIDL           569
#define B3SOIPD_MOD_PNTUN            570
#define B3SOIPD_MOD_PNDIODE          571
#define B3SOIPD_MOD_PNRECF0          572
#define B3SOIPD_MOD_PNRECR0          573
#define B3SOIPD_MOD_PISBJT           574
#define B3SOIPD_MOD_PISDIF           575
#define B3SOIPD_MOD_PISREC           576
#define B3SOIPD_MOD_PISTUN           577
#define B3SOIPD_MOD_PVREC0           578
#define B3SOIPD_MOD_PVTUN0           579
#define B3SOIPD_MOD_PNBJT            580
#define B3SOIPD_MOD_PLBJT0           581
#define B3SOIPD_MOD_PVABJT           582
#define B3SOIPD_MOD_PAELY            583
#define B3SOIPD_MOD_PAHLI            584
#define B3SOIPD_MOD_PVSDFB           585
#define B3SOIPD_MOD_PVSDTH           586
#define B3SOIPD_MOD_PDELVT           587
#define B3SOIPD_MOD_PACDE            588
#define B3SOIPD_MOD_PMOIN            589
/* Added for binning - END3 */

#define B3SOIPD_MOD_TNOM             701
#define B3SOIPD_MOD_CGSO             702
#define B3SOIPD_MOD_CGDO             703
#define B3SOIPD_MOD_CGEO             704
#define B3SOIPD_MOD_XPART            705

#define B3SOIPD_MOD_RSH              706
#define B3SOIPD_MOD_NMOS             814
#define B3SOIPD_MOD_PMOS             815

#define B3SOIPD_MOD_NOIA             816
#define B3SOIPD_MOD_NOIB             817
#define B3SOIPD_MOD_NOIC             818

#define B3SOIPD_MOD_LINT             819
#define B3SOIPD_MOD_LL               820
#define B3SOIPD_MOD_LLN              821
#define B3SOIPD_MOD_LW               822
#define B3SOIPD_MOD_LWN              823
#define B3SOIPD_MOD_LWL              824

#define B3SOIPD_MOD_WINT             827
#define B3SOIPD_MOD_WL               828
#define B3SOIPD_MOD_WLN              829
#define B3SOIPD_MOD_WW               830
#define B3SOIPD_MOD_WWN              831
#define B3SOIPD_MOD_WWL              832

/* v2.2.3 */
#define B3SOIPD_MOD_LWLC             847
#define B3SOIPD_MOD_LLC              848
#define B3SOIPD_MOD_LWC              849
#define B3SOIPD_MOD_WWLC             850  
#define B3SOIPD_MOD_WLC              851
#define B3SOIPD_MOD_WWC              852
#define B3SOIPD_MOD_DTOXCV           853 

#define B3SOIPD_MOD_DWC              835
#define B3SOIPD_MOD_DLC              836

#define B3SOIPD_MOD_EM               837
#define B3SOIPD_MOD_EF               838
#define B3SOIPD_MOD_AF               839
#define B3SOIPD_MOD_KF               840
#define B3SOIPD_MOD_NOIF             841


#define B3SOIPD_MOD_PBSWG            843
#define B3SOIPD_MOD_MJSWG            844
#define B3SOIPD_MOD_CJSWG            845
#define B3SOIPD_MOD_CSDESW           846

/* device questions */
#define B3SOIPD_DNODE                901
#define B3SOIPD_GNODE                902
#define B3SOIPD_SNODE                903
#define B3SOIPD_BNODE                904
#define B3SOIPD_ENODE                905
#define B3SOIPD_DNODEPRIME           906
#define B3SOIPD_SNODEPRIME           907
#define B3SOIPD_VBD                  908
#define B3SOIPD_VBS                  909
#define B3SOIPD_VGS                  910
#define B3SOIPD_VES                  911
#define B3SOIPD_VDS                  912
#define B3SOIPD_CD                   913
#define B3SOIPD_CBS                  914
#define B3SOIPD_CBD                  915
#define B3SOIPD_GM                   916
#define B3SOIPD_GDS                  917
#define B3SOIPD_GMBS                 918
#define B3SOIPD_GBD                  919
#define B3SOIPD_GBS                  920
#define B3SOIPD_QB                   921
#define B3SOIPD_CQB                  922
#define B3SOIPD_QG                   923
#define B3SOIPD_CQG                  924
#define B3SOIPD_QD                   925
#define B3SOIPD_CQD                  926
#define B3SOIPD_CGG                  927
#define B3SOIPD_CGD                  928
#define B3SOIPD_CGS                  929
#define B3SOIPD_CBG                  930
#define B3SOIPD_CAPBD                931
#define B3SOIPD_CQBD                 932
#define B3SOIPD_CAPBS                933
#define B3SOIPD_CQBS                 934
#define B3SOIPD_CDG                  935
#define B3SOIPD_CDD                  936
#define B3SOIPD_CDS                  937
#define B3SOIPD_VON                  938
#define B3SOIPD_VDSAT                939
#define B3SOIPD_QBS                  940
#define B3SOIPD_QBD                  941
#define B3SOIPD_SOURCECONDUCT        942
#define B3SOIPD_DRAINCONDUCT         943
#define B3SOIPD_CBDB                 944
#define B3SOIPD_CBSB                 945
#define B3SOIPD_GMID                 946


#include "b3soipdext.h"

extern void B3SOIPDevaluate(double,double,double,B3SOIPDinstance*,B3SOIPDmodel*,
        double*,double*,double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, CKTcircuit*);
extern int B3SOIPDdebug(B3SOIPDmodel*, B3SOIPDinstance*, CKTcircuit*, int);
extern int B3SOIPDcheckModel(B3SOIPDmodel*, B3SOIPDinstance*, CKTcircuit*);
#endif /*B3SOIPD*/

