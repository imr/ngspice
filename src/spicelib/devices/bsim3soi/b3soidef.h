/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung
File: b3soidef.h
Modified by Pin Su and Jan Feng	99/2/15
Modified by Pin Su 99/4/30
Modified by Pin Su and Wei Jin 99/9/27
Modified by Pin Su 00/3/1
Modified by Pin Su 01/2/15
Modified by Pin Su and Hui Wan 02/3/5
Modified by Pin Su 02/5/20
Modified by Paolo Nenzi 2002
**********/

#ifndef B3SOI
#define B3SOI

#define SOICODE
/*  #define BULKCODE  */

#include "ifsim.h"
#include "gendefs.h"
#include "cktdefs.h"
#include "complex.h"
#include "noisedef.h"         

typedef struct sB3SOIinstance
{
    struct sB3SOImodel *B3SOImodPtr;
    struct sB3SOIinstance *B3SOInextInstance;
    IFuid B3SOIname;
    int B3SOIowner;      /* number of owner process */ 
    int B3SOIstates;     /* index into state table for this device */

    int B3SOIdNode;
    int B3SOIgNode;
    int B3SOIsNode;
    int B3SOIeNode;
    int B3SOIpNode;
    int B3SOIbNode;
    int B3SOItempNode;
    int B3SOIdNodePrime;
    int B3SOIsNodePrime;

    int B3SOIvbsNode;
    /* for Debug */
    int B3SOIidsNode;
    int B3SOIicNode;
    int B3SOIibsNode;
    int B3SOIibdNode;
    int B3SOIiiiNode;
    int B3SOIigNode;
    int B3SOIgiggNode;
    int B3SOIgigdNode;
    int B3SOIgigbNode;
    int B3SOIigidlNode;
    int B3SOIitunNode;
    int B3SOIibpNode;
    int B3SOIcbbNode;
    int B3SOIcbdNode;
    int B3SOIcbgNode;

    int B3SOIqbfNode;
    int B3SOIqjsNode;
    int B3SOIqjdNode;

    double B3SOIphi;
    double B3SOIvtm;
    double B3SOIni;
    double B3SOIueff;
    double B3SOIthetavth; 
    double B3SOIvon;
    double B3SOIvdsat;
    double B3SOIcgdo;
    double B3SOIcgso;
    double B3SOIcgeo;

    double B3SOIids;
    double B3SOIic;
    double B3SOIibs;
    double B3SOIibd;
    double B3SOIiii;
    double B3SOIig;
    double B3SOIgigg;
    double B3SOIgigd;
    double B3SOIgigb;
    double B3SOIgige; /* v3.0 */
    double B3SOIigidl;
    double B3SOIitun;
    double B3SOIibp;
    double B3SOIabeff;
    double B3SOIvbseff;
    double B3SOIcbg;
    double B3SOIcbb;
    double B3SOIcbd;
    double B3SOIqb;
    double B3SOIqbf;
    double B3SOIqjs;
    double B3SOIqjd;
    int    B3SOIfloat;

    double B3SOIl;
    double B3SOIw;
    double B3SOIm;
    double B3SOIdrainArea;
    double B3SOIsourceArea;
    double B3SOIdrainSquares;
    double B3SOIsourceSquares;
    double B3SOIdrainPerimeter;
    double B3SOIsourcePerimeter;
    double B3SOIsourceConductance;
    double B3SOIdrainConductance;

    double B3SOIicVBS;
    double B3SOIicVDS;
    double B3SOIicVGS;
    double B3SOIicVES;
    double B3SOIicVPS;
    int B3SOIbjtoff;
    int B3SOIbodyMod;
    int B3SOIdebugMod;
    double B3SOIrth0;
    double B3SOIcth0;
    double B3SOIbodySquares;
    double B3SOIrbodyext;
    double B3SOIfrbody;


/* v2.0 release */
    double B3SOInbc;   
    double B3SOInseg;
    double B3SOIpdbcp;
    double B3SOIpsbcp;
    double B3SOIagbcp;
    double B3SOIaebcp;
    double B3SOIvbsusr;
    int B3SOItnodeout;

/* Deleted from pParam and moved to here */
    double B3SOIcsesw;
    double B3SOIcdesw;
    double B3SOIcsbox;
    double B3SOIcdbox;
    double B3SOIcsmin;
    double B3SOIcdmin;
    double B3SOIst4;
    double B3SOIdt4;

    int B3SOIoff;
    int B3SOImode;

    /* OP point */
    double B3SOIqinv;
    double B3SOIcd;
    double B3SOIcjs;
    double B3SOIcjd;
    double B3SOIcbody;
/* v2.2 release */
    double B3SOIcgate;
    double B3SOIgigs;
    double B3SOIgigT;

    double B3SOIcbodcon;
    double B3SOIcth;
    double B3SOIcsubstrate;

    double B3SOIgm;
    double B3SOIgme;   /* v3.0 */
    double B3SOIcb;
    double B3SOIcdrain;
    double B3SOIgds;
    double B3SOIgmbs;
    double B3SOIgmT;

    double B3SOIgbbs;
    double B3SOIgbgs;
    double B3SOIgbds;
    double B3SOIgbes; /* v3.0 */
    double B3SOIgbps;
    double B3SOIgbT;

/* v3.0 */
    double B3SOIIgcs;
    double B3SOIgIgcsg;
    double B3SOIgIgcsd;
    double B3SOIgIgcss;
    double B3SOIgIgcsb;
    double B3SOIIgcd;
    double B3SOIgIgcdg;
    double B3SOIgIgcdd;
    double B3SOIgIgcds;
    double B3SOIgIgcdb;

    double B3SOIIgs;
    double B3SOIgIgsg;
    double B3SOIgIgss;
    double B3SOIIgd;
    double B3SOIgIgdg;
    double B3SOIgIgdd;


    double B3SOIgjsd;
    double B3SOIgjsb;
    double B3SOIgjsg;
    double B3SOIgjsT;

    double B3SOIgjdb;
    double B3SOIgjdd;
    double B3SOIgjdg;
    double B3SOIgjde; /* v3.0 */
    double B3SOIgjdT;

    double B3SOIgbpbs;
    double B3SOIgbpps;
    double B3SOIgbpT;

    double B3SOIgtempb;
    double B3SOIgtempg;
    double B3SOIgtempd;
    double B3SOIgtempe; /* v3.0 */
    double B3SOIgtempT;

    double B3SOIcggb;
    double B3SOIcgdb;
    double B3SOIcgsb;
    double B3SOIcgT;

    double B3SOIcbgb;
    double B3SOIcbdb;
    double B3SOIcbsb;
    double B3SOIcbeb;
    double B3SOIcbT;

    double B3SOIcdgb;
    double B3SOIcddb;
    double B3SOIcdsb;
    double B3SOIcdeb;
    double B3SOIcdT;

    double B3SOIceeb;
    double B3SOIceT;

    double B3SOIqse;
    double B3SOIgcse;
    double B3SOIqde;
    double B3SOIgcde;
    double B3SOIrds; /* v2.2.3 */
    double B3SOIVgsteff; /* v2.2.3 */
    double B3SOIVdseff;  /* v2.2.3 */
    double B3SOIAbovVgst2Vtm; /* v2.2.3 */

    struct b3soiSizeDependParam  *pParam;

    unsigned B3SOIlGiven :1;
    unsigned B3SOIwGiven :1;
    unsigned B3SOImGiven :1;
    unsigned B3SOIdrainAreaGiven :1;
    unsigned B3SOIsourceAreaGiven    :1;
    unsigned B3SOIdrainSquaresGiven  :1;
    unsigned B3SOIsourceSquaresGiven :1;
    unsigned B3SOIdrainPerimeterGiven    :1;
    unsigned B3SOIsourcePerimeterGiven   :1;
    unsigned B3SOIdNodePrimeSet  :1;
    unsigned B3SOIsNodePrimeSet  :1;
    unsigned B3SOIicVBSGiven :1;
    unsigned B3SOIicVDSGiven :1;
    unsigned B3SOIicVGSGiven :1;
    unsigned B3SOIicVESGiven :1;
    unsigned B3SOIicVPSGiven :1;
    unsigned B3SOIbjtoffGiven :1;
    unsigned B3SOIdebugModGiven :1;
    unsigned B3SOIrth0Given :1;
    unsigned B3SOIcth0Given :1;
    unsigned B3SOIbodySquaresGiven :1;
    unsigned B3SOIfrbodyGiven:  1;

/* v2.0 release */
    unsigned B3SOInbcGiven :1;   
    unsigned B3SOInsegGiven :1;
    unsigned B3SOIpdbcpGiven :1;
    unsigned B3SOIpsbcpGiven :1;
    unsigned B3SOIagbcpGiven :1;
    unsigned B3SOIaebcpGiven :1;
    unsigned B3SOIvbsusrGiven :1;
    unsigned B3SOItnodeoutGiven :1;
    unsigned B3SOIoffGiven :1;

    double *B3SOIGePtr;
    double *B3SOIDPePtr;
    double *B3SOISPePtr;

    double *B3SOIEePtr;
    double *B3SOIEbPtr;
    double *B3SOIBePtr;
    double *B3SOIEgPtr;
    double *B3SOIEdpPtr;
    double *B3SOIEspPtr;
    double *B3SOITemptempPtr;
    double *B3SOITempdpPtr;
    double *B3SOITempspPtr;
    double *B3SOITempgPtr;
    double *B3SOITempbPtr;
    double *B3SOITempePtr;   /* v3.0 */
    double *B3SOIGtempPtr;
    double *B3SOIDPtempPtr;
    double *B3SOISPtempPtr;
    double *B3SOIEtempPtr;
    double *B3SOIBtempPtr;
    double *B3SOIPtempPtr;
    double *B3SOIBpPtr;
    double *B3SOIPbPtr;
    double *B3SOIPpPtr;
    double *B3SOIDdPtr;
    double *B3SOIGgPtr;
    double *B3SOISsPtr;
    double *B3SOIBbPtr;
    double *B3SOIDPdpPtr;
    double *B3SOISPspPtr;
    double *B3SOIDdpPtr;
    double *B3SOIGbPtr;
    double *B3SOIGdpPtr;
    double *B3SOIGspPtr;
    double *B3SOISspPtr;
    double *B3SOIBdpPtr;
    double *B3SOIBspPtr;
    double *B3SOIDPspPtr;
    double *B3SOIDPdPtr;
    double *B3SOIBgPtr;
    double *B3SOIDPgPtr;
    double *B3SOISPgPtr;
    double *B3SOISPsPtr;
    double *B3SOIDPbPtr;
    double *B3SOISPbPtr;
    double *B3SOISPdpPtr;

    double *B3SOIVbsPtr;
    /* Debug */
    double *B3SOIIdsPtr;
    double *B3SOIIcPtr;
    double *B3SOIIbsPtr;
    double *B3SOIIbdPtr;
    double *B3SOIIiiPtr;
    double *B3SOIIgPtr;
    double *B3SOIGiggPtr;
    double *B3SOIGigdPtr;
    double *B3SOIGigbPtr;
    double *B3SOIIgidlPtr;
    double *B3SOIItunPtr;
    double *B3SOIIbpPtr;
    double *B3SOICbbPtr;
    double *B3SOICbdPtr;
    double *B3SOICbgPtr;
    double *B3SOIqbPtr;
    double *B3SOIQbfPtr;
    double *B3SOIQjsPtr;
    double *B3SOIQjdPtr;


#define B3SOIvbd B3SOIstates+ 0
#define B3SOIvbs B3SOIstates+ 1
#define B3SOIvgs B3SOIstates+ 2
#define B3SOIvds B3SOIstates+ 3
#define B3SOIves B3SOIstates+ 4
#define B3SOIvps B3SOIstates+ 5

#define B3SOIvg B3SOIstates+ 6
#define B3SOIvd B3SOIstates+ 7
#define B3SOIvs B3SOIstates+ 8
#define B3SOIvp B3SOIstates+ 9
#define B3SOIve B3SOIstates+ 10
#define B3SOIdeltemp B3SOIstates+ 11

#define B3SOIqb B3SOIstates+ 12
#define B3SOIcqb B3SOIstates+ 13
#define B3SOIqg B3SOIstates+ 14
#define B3SOIcqg B3SOIstates+ 15
#define B3SOIqd B3SOIstates+ 16
#define B3SOIcqd B3SOIstates+ 17
#define B3SOIqe B3SOIstates+ 18
#define B3SOIcqe B3SOIstates+ 19

#define B3SOIqbs  B3SOIstates+ 20
#define B3SOIqbd  B3SOIstates+ 21
#define B3SOIqbe  B3SOIstates+ 22

#define B3SOIqth B3SOIstates+ 23
#define B3SOIcqth B3SOIstates+ 24

#define B3SOInumStates 25


/* indices to the array of B3SOI NOISE SOURCES */

#define B3SOIRDNOIZ       0
#define B3SOIRSNOIZ       1
#define B3SOIIDNOIZ       2
#define B3SOIFLNOIZ       3
#define B3SOIFBNOIZ       4
#define B3SOITOTNOIZ      5

#define B3SOINSRCS        6     /* the number of MOSFET(3) noise sources */

#ifndef NONOISE
    double B3SOInVar[NSTATVARS][B3SOINSRCS];
#else /* NONOISE */
        double **B3SOInVar;
#endif /* NONOISE */

} B3SOIinstance ;

struct b3soiSizeDependParam
{
    double Width;
    double Length;
    double Rth0;
    double Cth0;

    double B3SOIcdsc;           
    double B3SOIcdscb;    
    double B3SOIcdscd;       
    double B3SOIcit;           
    double B3SOInfactor;      
    double B3SOIvsat;         
    double B3SOIat;         
    double B3SOIa0;   
    double B3SOIags;      
    double B3SOIa1;         
    double B3SOIa2;         
    double B3SOIketa;     
    double B3SOInpeak;        
    double B3SOInsub;
    double B3SOIngate;        
    double B3SOIgamma1;      
    double B3SOIgamma2;     
    double B3SOIvbx;      
    double B3SOIvbi;       
    double B3SOIvbm;       
    double B3SOIvbsc;       
    double B3SOIxt;       
    double B3SOIphi;
    double B3SOIlitl;
    double B3SOIk1;
    double B3SOIkt1;
    double B3SOIkt1l;
    double B3SOIkt2;
    double B3SOIk2;
    double B3SOIk3;
    double B3SOIk3b;
    double B3SOIw0;
    double B3SOInlx;
    double B3SOIdvt0;      
    double B3SOIdvt1;      
    double B3SOIdvt2;      
    double B3SOIdvt0w;      
    double B3SOIdvt1w;      
    double B3SOIdvt2w;      
    double B3SOIdrout;      
    double B3SOIdsub;      
    double B3SOIvth0;
    double B3SOIua;
    double B3SOIua1;
    double B3SOIub;
    double B3SOIub1;
    double B3SOIuc;
    double B3SOIuc1;
    double B3SOIu0;
    double B3SOIute;
    double B3SOIvoff;
    double B3SOIvfb;
    double B3SOIuatemp;
    double B3SOIubtemp;
    double B3SOIuctemp;
    double B3SOIrbody;
    double B3SOIrth;
    double B3SOIcth;
    double B3SOIrds0denom;
    double B3SOIvfbb;
    double B3SOIjbjt;
    double B3SOIjdif;
    double B3SOIjrec;
    double B3SOIjtun;
    double B3SOIsdt1;
    double B3SOIst2;
    double B3SOIst3;
    double B3SOIdt2;
    double B3SOIdt3;
    double B3SOIdelta;
    double B3SOIrdsw;       
    double B3SOIrds0;       
    double B3SOIprwg;       
    double B3SOIprwb;       
    double B3SOIprt;       
    double B3SOIeta0;         
    double B3SOIetab;         
    double B3SOIpclm;      
    double B3SOIpdibl1;      
    double B3SOIpdibl2;      
    double B3SOIpdiblb;      
    double B3SOIpvag;       
    double B3SOIwr;
    double B3SOIdwg;
    double B3SOIdwb;
    double B3SOIb0;
    double B3SOIb1;
    double B3SOIalpha0;
    double B3SOIbeta0;
/* v3.0 */
    double B3SOIaigc;
    double B3SOIbigc;
    double B3SOIcigc;
    double B3SOIaigsd;
    double B3SOIbigsd;
    double B3SOIcigsd;
    double B3SOInigc;
    double B3SOIpigcd;
    double B3SOIpoxedge;
    double B3SOIdlcig;

    /* CV model */
    double B3SOIcgsl;
    double B3SOIcgdl;
    double B3SOIckappa;
    double B3SOIcf;
    double B3SOIclc;
    double B3SOIcle;


/* Added for binning - START0 */
    double B3SOIkb1;
    double B3SOIk1w1;
    double B3SOIk1w2;
    double B3SOIketas;
    double B3SOIfbjtii;
    double B3SOIbeta1;
    double B3SOIbeta2;
    double B3SOIvdsatii0;
    double B3SOIlii;
    double B3SOIesatii;
    double B3SOIsii0;
    double B3SOIsii1;
    double B3SOIsii2;
    double B3SOIsiid;
    double B3SOIagidl;
    double B3SOIbgidl;
    double B3SOIngidl;
    double B3SOIntun;
    double B3SOIndiode;
    double B3SOInrecf0;
    double B3SOInrecr0;
    double B3SOIisbjt;
    double B3SOIisdif;
    double B3SOIisrec;
    double B3SOIistun;
    double B3SOIvrec0;
    double B3SOIvtun0;
    double B3SOInbjt;
    double B3SOIlbjt0;
    double B3SOIvabjt;
    double B3SOIaely;
    double B3SOIvsdfb;
    double B3SOIvsdth;
    double B3SOIdelvt;
/* Added by binning - END0 */

/* Pre-calculated constants */

    double B3SOIdw;
    double B3SOIdl;
    double B3SOIleff;
    double B3SOIweff;

    double B3SOIdwc;
    double B3SOIdlc;
    double B3SOIleffCV;
    double B3SOIweffCV;
    double B3SOIabulkCVfactor;
    double B3SOIcgso;
    double B3SOIcgdo;
    double B3SOIcgeo;

    double B3SOIu0temp;       
    double B3SOIvsattemp;   
    double B3SOIsqrtPhi;   
    double B3SOIphis3;   
    double B3SOIXdep0;          
    double B3SOIsqrtXdep0;          
    double B3SOItheta0vb0;
    double B3SOIthetaRout; 


/* v2.2 release */
    double B3SOIoxideRatio;


/* v2.0 release */
    double B3SOIk1eff;
    double B3SOIwdios;
    double B3SOIwdiod;
    double B3SOIwdiodCV;
    double B3SOIwdiosCV;
    double B3SOIarfabjt;
    double B3SOIlratio;
    double B3SOIlratiodif;
    double B3SOIvearly;
    double B3SOIahli;
    double B3SOIahli0;
    double B3SOIvfbzb;
    double B3SOIldeb;
    double B3SOIacde;
    double B3SOImoin;
    double B3SOIleffCVb;
    double B3SOIleffCVbg;

    double B3SOIcof1;
    double B3SOIcof2;
    double B3SOIcof3;
    double B3SOIcof4;
    double B3SOIcdep0;
/* v3.0 */
    double B3SOIToxRatio;
    double B3SOIAechvb;
    double B3SOIBechvb;
    double B3SOIToxRatioEdge;
    double B3SOIAechvbEdge;
    double B3SOIBechvbEdge;
    double B3SOIvfbsd;

    struct b3soiSizeDependParam  *pNext;
};


typedef struct sB3SOImodel 
{
    int B3SOImodType;
    struct sB3SOImodel *B3SOInextModel;
    B3SOIinstance *B3SOIinstances;
    IFuid B3SOImodName; 
    int B3SOItype;

    int    B3SOImobMod;
    int    B3SOIcapMod;
    int    B3SOInoiMod;
    int    B3SOIshMod;
    int    B3SOIbinUnit;
    int    B3SOIparamChk;
    double B3SOIversion;             
    double B3SOItox;             
    double B3SOIdtoxcv; /* v2.2.3 */
    double B3SOIcdsc;           
    double B3SOIcdscb; 
    double B3SOIcdscd;          
    double B3SOIcit;           
    double B3SOInfactor;      
    double B3SOIvsat;         
    double B3SOIat;         
    double B3SOIa0;   
    double B3SOIags;      
    double B3SOIa1;         
    double B3SOIa2;         
    double B3SOIketa;     
    double B3SOInsub;
    double B3SOInpeak;        
    double B3SOIngate;        
    double B3SOIgamma1;      
    double B3SOIgamma2;     
    double B3SOIvbx;      
    double B3SOIvbm;       
    double B3SOIxt;       
    double B3SOIk1;
    double B3SOIkt1;
    double B3SOIkt1l;
    double B3SOIkt2;
    double B3SOIk2;
    double B3SOIk3;
    double B3SOIk3b;
    double B3SOIw0;
    double B3SOInlx;
    double B3SOIdvt0;      
    double B3SOIdvt1;      
    double B3SOIdvt2;      
    double B3SOIdvt0w;      
    double B3SOIdvt1w;      
    double B3SOIdvt2w;      
    double B3SOIdrout;      
    double B3SOIdsub;      
    double B3SOIvth0;
    double B3SOIua;
    double B3SOIua1;
    double B3SOIub;
    double B3SOIub1;
    double B3SOIuc;
    double B3SOIuc1;
    double B3SOIu0;
    double B3SOIute;
    double B3SOIvoff;
    double B3SOIdelta;
    double B3SOIrdsw;       
    double B3SOIprwg;
    double B3SOIprwb;
    double B3SOIprt;       
    double B3SOIeta0;         
    double B3SOIetab;         
    double B3SOIpclm;      
    double B3SOIpdibl1;      
    double B3SOIpdibl2;      
    double B3SOIpdiblb;
    double B3SOIpvag;       
    double B3SOIwr;
    double B3SOIdwg;
    double B3SOIdwb;
    double B3SOIb0;
    double B3SOIb1;
    double B3SOIalpha0;
    double B3SOItbox;
    double B3SOItsi;
    double B3SOIxj;
    double B3SOIkb1;
    double B3SOIrth0;
    double B3SOIcth0;
    double B3SOIngidl;
    double B3SOIagidl;
    double B3SOIbgidl;
    double B3SOIndiode;
    double B3SOIistun;
    double B3SOIxbjt;
    double B3SOIxdif;
    double B3SOIxrec;
    double B3SOIxtun;


/* v3.0 */
    double B3SOIsoiMod;
    double B3SOIvbsa;
    double B3SOInofffd;
    double B3SOIvofffd;
    double B3SOIk1b;
    double B3SOIk2b;
    double B3SOIdk2b;
    double B3SOIdvbd0;
    double B3SOIdvbd1;
    double B3SOImoinFD;

/* v3.0 */
    int B3SOIigbMod;
    int B3SOIigcMod;
    double B3SOIaigc;
    double B3SOIbigc;
    double B3SOIcigc;
    double B3SOIaigsd;
    double B3SOIbigsd;
    double B3SOIcigsd;
    double B3SOInigc;
    double B3SOIpigcd;
    double B3SOIpoxedge;
    double B3SOIdlcig;

/* v2.2 release */
    double B3SOIwth0;
    double B3SOIrhalo;
    double B3SOIntox;
    double B3SOItoxref;
    double B3SOIebg;
    double B3SOIvevb;
    double B3SOIalphaGB1;
    double B3SOIbetaGB1;
    double B3SOIvgb1;
    double B3SOIvecb;
    double B3SOIalphaGB2;
    double B3SOIbetaGB2;
    double B3SOIvgb2;
    double B3SOItoxqm;
    double B3SOIvoxh;
    double B3SOIdeltavox;


/* v2.0 release */
    double B3SOIk1w1;  
    double B3SOIk1w2;
    double B3SOIketas;
    double B3SOIdwbc;
    double B3SOIbeta0;
    double B3SOIbeta1;
    double B3SOIbeta2;
    double B3SOIvdsatii0;
    double B3SOItii;
    double B3SOIlii;
    double B3SOIsii0;
    double B3SOIsii1;
    double B3SOIsii2;
    double B3SOIsiid;
    double B3SOIfbjtii;
    double B3SOIesatii;
    double B3SOIntun;
    double B3SOInrecf0;
    double B3SOInrecr0;
    double B3SOIisbjt;
    double B3SOIisdif;
    double B3SOIisrec;
    double B3SOIln;
    double B3SOIvrec0;
    double B3SOIvtun0;
    double B3SOInbjt;
    double B3SOIlbjt0;
    double B3SOIldif0;
    double B3SOIvabjt;
    double B3SOIaely;
    double B3SOIahli;
    double B3SOIrbody;
    double B3SOIrbsh;
    double B3SOItt;
    double B3SOIndif;
    double B3SOIvsdfb;
    double B3SOIvsdth;
    double B3SOIcsdmin;
    double B3SOIasd;
    double B3SOIntrecf;
    double B3SOIntrecr;
    double B3SOIdlcb;
    double B3SOIfbody;
    double B3SOItcjswg;
    double B3SOItpbswg;
    double B3SOIacde;
    double B3SOImoin;
    double B3SOIdelvt;
    double B3SOIdlbg;

    /* CV model */
    double B3SOIcgsl;
    double B3SOIcgdl;
    double B3SOIckappa;
    double B3SOIcf;
    double B3SOIclc;
    double B3SOIcle;
    double B3SOIdwc;
    double B3SOIdlc;

    double B3SOItnom;
    double B3SOIcgso;
    double B3SOIcgdo;
    double B3SOIcgeo;
    double B3SOIxpart;
    double B3SOIcFringOut;
    double B3SOIcFringMax;

    double B3SOIsheetResistance;
    double B3SOIbodyJctGateSideGradingCoeff;
    double B3SOIGatesidewallJctPotential;
    double B3SOIunitLengthGateSidewallJctCap;
    double B3SOIcsdesw;

    double B3SOILint;
    double B3SOILl;
    double B3SOILlc; /* v2.2.3 */
    double B3SOILln;
    double B3SOILw;
    double B3SOILwc; /* v2.2.3 */
    double B3SOILwn;
    double B3SOILwl;
    double B3SOILwlc; /* v2.2.3 */
    double B3SOILmin;
    double B3SOILmax;

    double B3SOIWint;
    double B3SOIWl;
    double B3SOIWlc; /* v2.2.3 */
    double B3SOIWln;
    double B3SOIWw;
    double B3SOIWwc; /* v2.2.3 */
    double B3SOIWwn;
    double B3SOIWwl;
    double B3SOIWwlc; /* v2.2.3 */
    double B3SOIWmin;
    double B3SOIWmax;

/* Added for binning - START1 */
    /* Length Dependence */
/* v3.0 */
    double B3SOIlaigc;
    double B3SOIlbigc;
    double B3SOIlcigc;
    double B3SOIlaigsd;
    double B3SOIlbigsd;
    double B3SOIlcigsd;
    double B3SOIlnigc;
    double B3SOIlpigcd;
    double B3SOIlpoxedge;

    double B3SOIlnpeak;        
    double B3SOIlnsub;
    double B3SOIlngate;        
    double B3SOIlvth0;
    double B3SOIlk1;
    double B3SOIlk1w1;
    double B3SOIlk1w2;
    double B3SOIlk2;
    double B3SOIlk3;
    double B3SOIlk3b;
    double B3SOIlkb1;
    double B3SOIlw0;
    double B3SOIlnlx;
    double B3SOIldvt0;      
    double B3SOIldvt1;      
    double B3SOIldvt2;      
    double B3SOIldvt0w;      
    double B3SOIldvt1w;      
    double B3SOIldvt2w;      
    double B3SOIlu0;
    double B3SOIlua;
    double B3SOIlub;
    double B3SOIluc;
    double B3SOIlvsat;         
    double B3SOIla0;   
    double B3SOIlags;      
    double B3SOIlb0;
    double B3SOIlb1;
    double B3SOIlketa;
    double B3SOIlketas;
    double B3SOIla1;         
    double B3SOIla2;         
    double B3SOIlrdsw;       
    double B3SOIlprwb;
    double B3SOIlprwg;
    double B3SOIlwr;
    double B3SOIlnfactor;      
    double B3SOIldwg;
    double B3SOIldwb;
    double B3SOIlvoff;
    double B3SOIleta0;         
    double B3SOIletab;         
    double B3SOIldsub;      
    double B3SOIlcit;           
    double B3SOIlcdsc;           
    double B3SOIlcdscb; 
    double B3SOIlcdscd;          
    double B3SOIlpclm;      
    double B3SOIlpdibl1;      
    double B3SOIlpdibl2;      
    double B3SOIlpdiblb;      
    double B3SOIldrout;      
    double B3SOIlpvag;       
    double B3SOIldelta;
    double B3SOIlalpha0;
    double B3SOIlfbjtii;
    double B3SOIlbeta0;
    double B3SOIlbeta1;
    double B3SOIlbeta2;         
    double B3SOIlvdsatii0;     
    double B3SOIllii;      
    double B3SOIlesatii;     
    double B3SOIlsii0;      
    double B3SOIlsii1;       
    double B3SOIlsii2;       
    double B3SOIlsiid;
    double B3SOIlagidl;
    double B3SOIlbgidl;
    double B3SOIlngidl;
    double B3SOIlntun;
    double B3SOIlndiode;
    double B3SOIlnrecf0;
    double B3SOIlnrecr0;
    double B3SOIlisbjt;       
    double B3SOIlisdif;
    double B3SOIlisrec;       
    double B3SOIlistun;       
    double B3SOIlvrec0;
    double B3SOIlvtun0;
    double B3SOIlnbjt;
    double B3SOIllbjt0;
    double B3SOIlvabjt;
    double B3SOIlaely;
    double B3SOIlahli;
    /* CV model */
    double B3SOIlvsdfb;
    double B3SOIlvsdth;
    double B3SOIldelvt;
    double B3SOIlacde;
    double B3SOIlmoin;

    /* Width Dependence */
/* v3.0 */
    double B3SOIwaigc;
    double B3SOIwbigc;
    double B3SOIwcigc;
    double B3SOIwaigsd;
    double B3SOIwbigsd;
    double B3SOIwcigsd;
    double B3SOIwnigc;
    double B3SOIwpigcd;
    double B3SOIwpoxedge;

    double B3SOIwnpeak;        
    double B3SOIwnsub;
    double B3SOIwngate;        
    double B3SOIwvth0;
    double B3SOIwk1;
    double B3SOIwk1w1;
    double B3SOIwk1w2;
    double B3SOIwk2;
    double B3SOIwk3;
    double B3SOIwk3b;
    double B3SOIwkb1;
    double B3SOIww0;
    double B3SOIwnlx;
    double B3SOIwdvt0;      
    double B3SOIwdvt1;      
    double B3SOIwdvt2;      
    double B3SOIwdvt0w;      
    double B3SOIwdvt1w;      
    double B3SOIwdvt2w;      
    double B3SOIwu0;
    double B3SOIwua;
    double B3SOIwub;
    double B3SOIwuc;
    double B3SOIwvsat;         
    double B3SOIwa0;   
    double B3SOIwags;      
    double B3SOIwb0;
    double B3SOIwb1;
    double B3SOIwketa;
    double B3SOIwketas;
    double B3SOIwa1;         
    double B3SOIwa2;         
    double B3SOIwrdsw;       
    double B3SOIwprwb;
    double B3SOIwprwg;
    double B3SOIwwr;
    double B3SOIwnfactor;      
    double B3SOIwdwg;
    double B3SOIwdwb;
    double B3SOIwvoff;
    double B3SOIweta0;         
    double B3SOIwetab;         
    double B3SOIwdsub;      
    double B3SOIwcit;           
    double B3SOIwcdsc;           
    double B3SOIwcdscb; 
    double B3SOIwcdscd;          
    double B3SOIwpclm;      
    double B3SOIwpdibl1;      
    double B3SOIwpdibl2;      
    double B3SOIwpdiblb;      
    double B3SOIwdrout;      
    double B3SOIwpvag;       
    double B3SOIwdelta;
    double B3SOIwalpha0;
    double B3SOIwfbjtii;
    double B3SOIwbeta0;
    double B3SOIwbeta1;
    double B3SOIwbeta2;         
    double B3SOIwvdsatii0;     
    double B3SOIwlii;      
    double B3SOIwesatii;     
    double B3SOIwsii0;      
    double B3SOIwsii1;       
    double B3SOIwsii2;       
    double B3SOIwsiid;
    double B3SOIwagidl;
    double B3SOIwbgidl;
    double B3SOIwngidl;
    double B3SOIwntun;
    double B3SOIwndiode;
    double B3SOIwnrecf0;
    double B3SOIwnrecr0;
    double B3SOIwisbjt;       
    double B3SOIwisdif;
    double B3SOIwisrec;       
    double B3SOIwistun;       
    double B3SOIwvrec0;
    double B3SOIwvtun0;
    double B3SOIwnbjt;
    double B3SOIwlbjt0;
    double B3SOIwvabjt;
    double B3SOIwaely;
    double B3SOIwahli;
    /* CV model */
    double B3SOIwvsdfb;
    double B3SOIwvsdth;
    double B3SOIwdelvt;
    double B3SOIwacde;
    double B3SOIwmoin;

    /* Cross-term Dependence */
/* v3.0 */
    double B3SOIpaigc;
    double B3SOIpbigc;
    double B3SOIpcigc;
    double B3SOIpaigsd;
    double B3SOIpbigsd;
    double B3SOIpcigsd;
    double B3SOIpnigc;
    double B3SOIppigcd;
    double B3SOIppoxedge;

    double B3SOIpnpeak;        
    double B3SOIpnsub;
    double B3SOIpngate;        
    double B3SOIpvth0;
    double B3SOIpk1;
    double B3SOIpk1w1;
    double B3SOIpk1w2;
    double B3SOIpk2;
    double B3SOIpk3;
    double B3SOIpk3b;
    double B3SOIpkb1;
    double B3SOIpw0;
    double B3SOIpnlx;
    double B3SOIpdvt0;      
    double B3SOIpdvt1;      
    double B3SOIpdvt2;      
    double B3SOIpdvt0w;      
    double B3SOIpdvt1w;      
    double B3SOIpdvt2w;      
    double B3SOIpu0;
    double B3SOIpua;
    double B3SOIpub;
    double B3SOIpuc;
    double B3SOIpvsat;         
    double B3SOIpa0;   
    double B3SOIpags;      
    double B3SOIpb0;
    double B3SOIpb1;
    double B3SOIpketa;
    double B3SOIpketas;
    double B3SOIpa1;         
    double B3SOIpa2;         
    double B3SOIprdsw;       
    double B3SOIpprwb;
    double B3SOIpprwg;
    double B3SOIpwr;
    double B3SOIpnfactor;      
    double B3SOIpdwg;
    double B3SOIpdwb;
    double B3SOIpvoff;
    double B3SOIpeta0;         
    double B3SOIpetab;         
    double B3SOIpdsub;      
    double B3SOIpcit;           
    double B3SOIpcdsc;           
    double B3SOIpcdscb; 
    double B3SOIpcdscd;          
    double B3SOIppclm;      
    double B3SOIppdibl1;      
    double B3SOIppdibl2;      
    double B3SOIppdiblb;      
    double B3SOIpdrout;      
    double B3SOIppvag;       
    double B3SOIpdelta;
    double B3SOIpalpha0;
    double B3SOIpfbjtii;
    double B3SOIpbeta0;
    double B3SOIpbeta1;
    double B3SOIpbeta2;         
    double B3SOIpvdsatii0;     
    double B3SOIplii;      
    double B3SOIpesatii;     
    double B3SOIpsii0;      
    double B3SOIpsii1;       
    double B3SOIpsii2;       
    double B3SOIpsiid;
    double B3SOIpagidl;
    double B3SOIpbgidl;
    double B3SOIpngidl;
    double B3SOIpntun;
    double B3SOIpndiode;
    double B3SOIpnrecf0;
    double B3SOIpnrecr0;
    double B3SOIpisbjt;       
    double B3SOIpisdif;
    double B3SOIpisrec;       
    double B3SOIpistun;       
    double B3SOIpvrec0;
    double B3SOIpvtun0;
    double B3SOIpnbjt;
    double B3SOIplbjt0;
    double B3SOIpvabjt;
    double B3SOIpaely;
    double B3SOIpahli;
    /* CV model */
    double B3SOIpvsdfb;
    double B3SOIpvsdth;
    double B3SOIpdelvt;
    double B3SOIpacde;
    double B3SOIpmoin;
/* Added for binning - END1 */

/* Pre-calculated constants */
    double B3SOIcbox;
    double B3SOIcsi;
    double B3SOIcsieff;
    double B3SOIcoxt;
    double B3SOInfb;
    double B3SOIadice;
    double B3SOIqsi;
    double B3SOIqsieff;
    double B3SOIeg0;

    /* MCJ: move to size-dependent param. */
    double B3SOIvtm;   
    double B3SOIcox;
    double B3SOIcof1;
    double B3SOIcof2;
    double B3SOIcof3;
    double B3SOIcof4;
    double B3SOIvcrit;
    double B3SOIfactor1;

    double B3SOIoxideTrapDensityA;      
    double B3SOIoxideTrapDensityB;     
    double B3SOIoxideTrapDensityC;  
    double B3SOIem;  
    double B3SOIef;  
    double B3SOIaf;  
    double B3SOIkf;  
    double B3SOInoif;  

    struct b3soiSizeDependParam *pSizeDependParamKnot;

    /* Flags */

/* v3.0 */
    unsigned B3SOIsoimodGiven: 1;
    unsigned B3SOIvbsaGiven  : 1;
    unsigned B3SOInofffdGiven: 1;
    unsigned B3SOIvofffdGiven: 1;
    unsigned B3SOIk1bGiven:    1;
    unsigned B3SOIk2bGiven:    1;
    unsigned B3SOIdk2bGiven:    1;
    unsigned B3SOIdvbd0Given:    1;
    unsigned B3SOIdvbd1Given:    1;
    unsigned B3SOImoinFDGiven:   1;


    unsigned B3SOItboxGiven:1;
    unsigned B3SOItsiGiven :1;
    unsigned B3SOIxjGiven :1;
    unsigned B3SOIkb1Given :1;
    unsigned B3SOIrth0Given :1;
    unsigned B3SOIcth0Given :1;
    unsigned B3SOIngidlGiven :1;
    unsigned B3SOIagidlGiven :1;
    unsigned B3SOIbgidlGiven :1;
    unsigned B3SOIndiodeGiven :1;
    unsigned B3SOIxbjtGiven :1;
    unsigned B3SOIxdifGiven :1;
    unsigned B3SOIxrecGiven :1;
    unsigned B3SOIxtunGiven :1;
    unsigned B3SOIttGiven :1;
    unsigned B3SOIvsdfbGiven :1;
    unsigned B3SOIvsdthGiven :1;
    unsigned B3SOIasdGiven :1;
    unsigned B3SOIcsdminGiven :1;

    unsigned  B3SOImobModGiven :1;
    unsigned  B3SOIbinUnitGiven :1;
    unsigned  B3SOIcapModGiven :1;
    unsigned  B3SOIparamChkGiven :1;
    unsigned  B3SOInoiModGiven :1;
    unsigned  B3SOIshModGiven :1;
    unsigned  B3SOItypeGiven   :1;
    unsigned  B3SOItoxGiven   :1;
    unsigned  B3SOIdtoxcvGiven   :1; /* v2.2.3 */
    unsigned  B3SOIversionGiven   :1;

    unsigned  B3SOIcdscGiven   :1;
    unsigned  B3SOIcdscbGiven   :1;
    unsigned  B3SOIcdscdGiven   :1;
    unsigned  B3SOIcitGiven   :1;
    unsigned  B3SOInfactorGiven   :1;
    unsigned  B3SOIvsatGiven   :1;
    unsigned  B3SOIatGiven   :1;
    unsigned  B3SOIa0Given   :1;
    unsigned  B3SOIagsGiven   :1;
    unsigned  B3SOIa1Given   :1;
    unsigned  B3SOIa2Given   :1;
    unsigned  B3SOIketaGiven   :1;    
    unsigned  B3SOInsubGiven   :1;
    unsigned  B3SOInpeakGiven   :1;
    unsigned  B3SOIngateGiven   :1;
    unsigned  B3SOIgamma1Given   :1;
    unsigned  B3SOIgamma2Given   :1;
    unsigned  B3SOIvbxGiven   :1;
    unsigned  B3SOIvbmGiven   :1;
    unsigned  B3SOIxtGiven   :1;
    unsigned  B3SOIk1Given   :1;
    unsigned  B3SOIkt1Given   :1;
    unsigned  B3SOIkt1lGiven   :1;
    unsigned  B3SOIkt2Given   :1;
    unsigned  B3SOIk2Given   :1;
    unsigned  B3SOIk3Given   :1;
    unsigned  B3SOIk3bGiven   :1;
    unsigned  B3SOIw0Given   :1;
    unsigned  B3SOInlxGiven   :1;
    unsigned  B3SOIdvt0Given   :1;   
    unsigned  B3SOIdvt1Given   :1;     
    unsigned  B3SOIdvt2Given   :1;     
    unsigned  B3SOIdvt0wGiven   :1;   
    unsigned  B3SOIdvt1wGiven   :1;     
    unsigned  B3SOIdvt2wGiven   :1;     
    unsigned  B3SOIdroutGiven   :1;     
    unsigned  B3SOIdsubGiven   :1;     
    unsigned  B3SOIvth0Given   :1;
    unsigned  B3SOIuaGiven   :1;
    unsigned  B3SOIua1Given   :1;
    unsigned  B3SOIubGiven   :1;
    unsigned  B3SOIub1Given   :1;
    unsigned  B3SOIucGiven   :1;
    unsigned  B3SOIuc1Given   :1;
    unsigned  B3SOIu0Given   :1;
    unsigned  B3SOIuteGiven   :1;
    unsigned  B3SOIvoffGiven   :1;
    unsigned  B3SOIrdswGiven   :1;      
    unsigned  B3SOIprwgGiven   :1;      
    unsigned  B3SOIprwbGiven   :1;      
    unsigned  B3SOIprtGiven   :1;      
    unsigned  B3SOIeta0Given   :1;    
    unsigned  B3SOIetabGiven   :1;    
    unsigned  B3SOIpclmGiven   :1;   
    unsigned  B3SOIpdibl1Given   :1;   
    unsigned  B3SOIpdibl2Given   :1;  
    unsigned  B3SOIpdiblbGiven   :1;  
    unsigned  B3SOIpvagGiven   :1;    
    unsigned  B3SOIdeltaGiven  :1;     
    unsigned  B3SOIwrGiven   :1;
    unsigned  B3SOIdwgGiven   :1;
    unsigned  B3SOIdwbGiven   :1;
    unsigned  B3SOIb0Given   :1;
    unsigned  B3SOIb1Given   :1;
    unsigned  B3SOIalpha0Given   :1;


/* v2.2 release */
    unsigned  B3SOIwth0Given   :1;
    unsigned  B3SOIrhaloGiven  :1;
    unsigned  B3SOIntoxGiven   :1;
    unsigned  B3SOItoxrefGiven   :1;
    unsigned  B3SOIebgGiven     :1;
    unsigned  B3SOIvevbGiven   :1;
    unsigned  B3SOIalphaGB1Given :1;
    unsigned  B3SOIbetaGB1Given  :1;
    unsigned  B3SOIvgb1Given        :1;
    unsigned  B3SOIvecbGiven        :1;
    unsigned  B3SOIalphaGB2Given    :1;
    unsigned  B3SOIbetaGB2Given     :1;
    unsigned  B3SOIvgb2Given        :1;
    unsigned  B3SOItoxqmGiven  :1;
    unsigned  B3SOIigbModGiven  :1; /* v3.0 */
    unsigned  B3SOIvoxhGiven   :1;
    unsigned  B3SOIdeltavoxGiven :1;
    unsigned  B3SOIigcModGiven  :1; /* v3.0 */
/* v3.0 */
    unsigned  B3SOIaigcGiven   :1;
    unsigned  B3SOIbigcGiven   :1;
    unsigned  B3SOIcigcGiven   :1;
    unsigned  B3SOIaigsdGiven   :1;
    unsigned  B3SOIbigsdGiven   :1;
    unsigned  B3SOIcigsdGiven   :1;   
    unsigned  B3SOInigcGiven   :1;
    unsigned  B3SOIpigcdGiven   :1;
    unsigned  B3SOIpoxedgeGiven   :1;
    unsigned  B3SOIdlcigGiven   :1;


/* v2.0 release */
    unsigned  B3SOIk1w1Given   :1;   
    unsigned  B3SOIk1w2Given   :1;
    unsigned  B3SOIketasGiven  :1;
    unsigned  B3SOIdwbcGiven  :1;
    unsigned  B3SOIbeta0Given  :1;
    unsigned  B3SOIbeta1Given  :1;
    unsigned  B3SOIbeta2Given  :1;
    unsigned  B3SOIvdsatii0Given  :1;
    unsigned  B3SOItiiGiven  :1;
    unsigned  B3SOIliiGiven  :1;
    unsigned  B3SOIsii0Given  :1;
    unsigned  B3SOIsii1Given  :1;
    unsigned  B3SOIsii2Given  :1;
    unsigned  B3SOIsiidGiven  :1;
    unsigned  B3SOIfbjtiiGiven :1;
    unsigned  B3SOIesatiiGiven :1;
    unsigned  B3SOIntunGiven  :1;
    unsigned  B3SOInrecf0Given  :1;
    unsigned  B3SOInrecr0Given  :1;
    unsigned  B3SOIisbjtGiven  :1;
    unsigned  B3SOIisdifGiven  :1;
    unsigned  B3SOIisrecGiven  :1;
    unsigned  B3SOIistunGiven  :1;
    unsigned  B3SOIlnGiven  :1;
    unsigned  B3SOIvrec0Given  :1;
    unsigned  B3SOIvtun0Given  :1;
    unsigned  B3SOInbjtGiven  :1;
    unsigned  B3SOIlbjt0Given  :1;
    unsigned  B3SOIldif0Given  :1;
    unsigned  B3SOIvabjtGiven  :1;
    unsigned  B3SOIaelyGiven  :1;
    unsigned  B3SOIahliGiven  :1;
    unsigned  B3SOIrbodyGiven :1;
    unsigned  B3SOIrbshGiven  :1;
    unsigned  B3SOIndifGiven  :1;
    unsigned  B3SOIntrecfGiven  :1;
    unsigned  B3SOIntrecrGiven  :1;
    unsigned  B3SOIdlcbGiven    :1;
    unsigned  B3SOIfbodyGiven   :1;
    unsigned  B3SOItcjswgGiven  :1;
    unsigned  B3SOItpbswgGiven  :1;
    unsigned  B3SOIacdeGiven  :1;
    unsigned  B3SOImoinGiven  :1;
    unsigned  B3SOIdelvtGiven  :1;
    unsigned  B3SOIdlbgGiven  :1;

    
    /* CV model */
    unsigned  B3SOIcgslGiven   :1;
    unsigned  B3SOIcgdlGiven   :1;
    unsigned  B3SOIckappaGiven   :1;
    unsigned  B3SOIcfGiven   :1;
    unsigned  B3SOIclcGiven   :1;
    unsigned  B3SOIcleGiven   :1;
    unsigned  B3SOIdwcGiven   :1;
    unsigned  B3SOIdlcGiven   :1;

/* Added for binning - START2 */
    /* Length Dependence */
/* v3.0 */
    unsigned  B3SOIlaigcGiven   :1;
    unsigned  B3SOIlbigcGiven   :1;
    unsigned  B3SOIlcigcGiven   :1;
    unsigned  B3SOIlaigsdGiven   :1;
    unsigned  B3SOIlbigsdGiven   :1;
    unsigned  B3SOIlcigsdGiven   :1;   
    unsigned  B3SOIlnigcGiven   :1;
    unsigned  B3SOIlpigcdGiven   :1;
    unsigned  B3SOIlpoxedgeGiven   :1;

    unsigned  B3SOIlnpeakGiven   :1;        
    unsigned  B3SOIlnsubGiven   :1;
    unsigned  B3SOIlngateGiven   :1;        
    unsigned  B3SOIlvth0Given   :1;
    unsigned  B3SOIlk1Given   :1;
    unsigned  B3SOIlk1w1Given   :1;
    unsigned  B3SOIlk1w2Given   :1;
    unsigned  B3SOIlk2Given   :1;
    unsigned  B3SOIlk3Given   :1;
    unsigned  B3SOIlk3bGiven   :1;
    unsigned  B3SOIlkb1Given   :1;
    unsigned  B3SOIlw0Given   :1;
    unsigned  B3SOIlnlxGiven   :1;
    unsigned  B3SOIldvt0Given   :1;      
    unsigned  B3SOIldvt1Given   :1;      
    unsigned  B3SOIldvt2Given   :1;      
    unsigned  B3SOIldvt0wGiven   :1;      
    unsigned  B3SOIldvt1wGiven   :1;      
    unsigned  B3SOIldvt2wGiven   :1;      
    unsigned  B3SOIlu0Given   :1;
    unsigned  B3SOIluaGiven   :1;
    unsigned  B3SOIlubGiven   :1;
    unsigned  B3SOIlucGiven   :1;
    unsigned  B3SOIlvsatGiven   :1;         
    unsigned  B3SOIla0Given   :1;   
    unsigned  B3SOIlagsGiven   :1;      
    unsigned  B3SOIlb0Given   :1;
    unsigned  B3SOIlb1Given   :1;
    unsigned  B3SOIlketaGiven   :1;
    unsigned  B3SOIlketasGiven   :1;
    unsigned  B3SOIla1Given   :1;         
    unsigned  B3SOIla2Given   :1;         
    unsigned  B3SOIlrdswGiven   :1;       
    unsigned  B3SOIlprwbGiven   :1;
    unsigned  B3SOIlprwgGiven   :1;
    unsigned  B3SOIlwrGiven   :1;
    unsigned  B3SOIlnfactorGiven   :1;      
    unsigned  B3SOIldwgGiven   :1;
    unsigned  B3SOIldwbGiven   :1;
    unsigned  B3SOIlvoffGiven   :1;
    unsigned  B3SOIleta0Given   :1;         
    unsigned  B3SOIletabGiven   :1;         
    unsigned  B3SOIldsubGiven   :1;      
    unsigned  B3SOIlcitGiven   :1;           
    unsigned  B3SOIlcdscGiven   :1;           
    unsigned  B3SOIlcdscbGiven   :1; 
    unsigned  B3SOIlcdscdGiven   :1;          
    unsigned  B3SOIlpclmGiven   :1;      
    unsigned  B3SOIlpdibl1Given   :1;      
    unsigned  B3SOIlpdibl2Given   :1;      
    unsigned  B3SOIlpdiblbGiven   :1;      
    unsigned  B3SOIldroutGiven   :1;      
    unsigned  B3SOIlpvagGiven   :1;       
    unsigned  B3SOIldeltaGiven   :1;
    unsigned  B3SOIlalpha0Given   :1;
    unsigned  B3SOIlfbjtiiGiven   :1;
    unsigned  B3SOIlbeta0Given   :1;
    unsigned  B3SOIlbeta1Given   :1;
    unsigned  B3SOIlbeta2Given   :1;         
    unsigned  B3SOIlvdsatii0Given   :1;     
    unsigned  B3SOIlliiGiven   :1;      
    unsigned  B3SOIlesatiiGiven   :1;     
    unsigned  B3SOIlsii0Given   :1;      
    unsigned  B3SOIlsii1Given   :1;       
    unsigned  B3SOIlsii2Given   :1;       
    unsigned  B3SOIlsiidGiven   :1;
    unsigned  B3SOIlagidlGiven   :1;
    unsigned  B3SOIlbgidlGiven   :1;
    unsigned  B3SOIlngidlGiven   :1;
    unsigned  B3SOIlntunGiven   :1;
    unsigned  B3SOIlndiodeGiven   :1;
    unsigned  B3SOIlnrecf0Given   :1;
    unsigned  B3SOIlnrecr0Given   :1;
    unsigned  B3SOIlisbjtGiven   :1;       
    unsigned  B3SOIlisdifGiven   :1;
    unsigned  B3SOIlisrecGiven   :1;       
    unsigned  B3SOIlistunGiven   :1;       
    unsigned  B3SOIlvrec0Given   :1;
    unsigned  B3SOIlvtun0Given   :1;
    unsigned  B3SOIlnbjtGiven   :1;
    unsigned  B3SOIllbjt0Given   :1;
    unsigned  B3SOIlvabjtGiven   :1;
    unsigned  B3SOIlaelyGiven   :1;
    unsigned  B3SOIlahliGiven   :1;
    /* CV model */
    unsigned  B3SOIlvsdfbGiven   :1;
    unsigned  B3SOIlvsdthGiven   :1;
    unsigned  B3SOIldelvtGiven   :1;
    unsigned  B3SOIlacdeGiven   :1;
    unsigned  B3SOIlmoinGiven   :1;

    /* Width Dependence */
/* v3.0 */
    unsigned  B3SOIwaigcGiven   :1;
    unsigned  B3SOIwbigcGiven   :1;
    unsigned  B3SOIwcigcGiven   :1;
    unsigned  B3SOIwaigsdGiven   :1;
    unsigned  B3SOIwbigsdGiven   :1;
    unsigned  B3SOIwcigsdGiven   :1;   
    unsigned  B3SOIwnigcGiven   :1;
    unsigned  B3SOIwpigcdGiven   :1;
    unsigned  B3SOIwpoxedgeGiven   :1;

    unsigned  B3SOIwnpeakGiven   :1;        
    unsigned  B3SOIwnsubGiven   :1;
    unsigned  B3SOIwngateGiven   :1;        
    unsigned  B3SOIwvth0Given   :1;
    unsigned  B3SOIwk1Given   :1;
    unsigned  B3SOIwk1w1Given   :1;
    unsigned  B3SOIwk1w2Given   :1;
    unsigned  B3SOIwk2Given   :1;
    unsigned  B3SOIwk3Given   :1;
    unsigned  B3SOIwk3bGiven   :1;
    unsigned  B3SOIwkb1Given   :1;
    unsigned  B3SOIww0Given   :1;
    unsigned  B3SOIwnlxGiven   :1;
    unsigned  B3SOIwdvt0Given   :1;      
    unsigned  B3SOIwdvt1Given   :1;      
    unsigned  B3SOIwdvt2Given   :1;      
    unsigned  B3SOIwdvt0wGiven   :1;      
    unsigned  B3SOIwdvt1wGiven   :1;      
    unsigned  B3SOIwdvt2wGiven   :1;      
    unsigned  B3SOIwu0Given   :1;
    unsigned  B3SOIwuaGiven   :1;
    unsigned  B3SOIwubGiven   :1;
    unsigned  B3SOIwucGiven   :1;
    unsigned  B3SOIwvsatGiven   :1;         
    unsigned  B3SOIwa0Given   :1;   
    unsigned  B3SOIwagsGiven   :1;      
    unsigned  B3SOIwb0Given   :1;
    unsigned  B3SOIwb1Given   :1;
    unsigned  B3SOIwketaGiven   :1;
    unsigned  B3SOIwketasGiven   :1;
    unsigned  B3SOIwa1Given   :1;         
    unsigned  B3SOIwa2Given   :1;         
    unsigned  B3SOIwrdswGiven   :1;       
    unsigned  B3SOIwprwbGiven   :1;
    unsigned  B3SOIwprwgGiven   :1;
    unsigned  B3SOIwwrGiven   :1;
    unsigned  B3SOIwnfactorGiven   :1;      
    unsigned  B3SOIwdwgGiven   :1;
    unsigned  B3SOIwdwbGiven   :1;
    unsigned  B3SOIwvoffGiven   :1;
    unsigned  B3SOIweta0Given   :1;         
    unsigned  B3SOIwetabGiven   :1;         
    unsigned  B3SOIwdsubGiven   :1;      
    unsigned  B3SOIwcitGiven   :1;           
    unsigned  B3SOIwcdscGiven   :1;           
    unsigned  B3SOIwcdscbGiven   :1; 
    unsigned  B3SOIwcdscdGiven   :1;          
    unsigned  B3SOIwpclmGiven   :1;      
    unsigned  B3SOIwpdibl1Given   :1;      
    unsigned  B3SOIwpdibl2Given   :1;      
    unsigned  B3SOIwpdiblbGiven   :1;      
    unsigned  B3SOIwdroutGiven   :1;      
    unsigned  B3SOIwpvagGiven   :1;       
    unsigned  B3SOIwdeltaGiven   :1;
    unsigned  B3SOIwalpha0Given   :1;
    unsigned  B3SOIwfbjtiiGiven   :1;
    unsigned  B3SOIwbeta0Given   :1;
    unsigned  B3SOIwbeta1Given   :1;
    unsigned  B3SOIwbeta2Given   :1;         
    unsigned  B3SOIwvdsatii0Given   :1;     
    unsigned  B3SOIwliiGiven   :1;      
    unsigned  B3SOIwesatiiGiven   :1;     
    unsigned  B3SOIwsii0Given   :1;      
    unsigned  B3SOIwsii1Given   :1;       
    unsigned  B3SOIwsii2Given   :1;       
    unsigned  B3SOIwsiidGiven   :1;
    unsigned  B3SOIwagidlGiven   :1;
    unsigned  B3SOIwbgidlGiven   :1;
    unsigned  B3SOIwngidlGiven   :1;
    unsigned  B3SOIwntunGiven   :1;
    unsigned  B3SOIwndiodeGiven   :1;
    unsigned  B3SOIwnrecf0Given   :1;
    unsigned  B3SOIwnrecr0Given   :1;
    unsigned  B3SOIwisbjtGiven   :1;       
    unsigned  B3SOIwisdifGiven   :1;
    unsigned  B3SOIwisrecGiven   :1;       
    unsigned  B3SOIwistunGiven   :1;       
    unsigned  B3SOIwvrec0Given   :1;
    unsigned  B3SOIwvtun0Given   :1;
    unsigned  B3SOIwnbjtGiven   :1;
    unsigned  B3SOIwlbjt0Given   :1;
    unsigned  B3SOIwvabjtGiven   :1;
    unsigned  B3SOIwaelyGiven   :1;
    unsigned  B3SOIwahliGiven   :1;
    /* CV model */
    unsigned  B3SOIwvsdfbGiven   :1;
    unsigned  B3SOIwvsdthGiven   :1;
    unsigned  B3SOIwdelvtGiven   :1;
    unsigned  B3SOIwacdeGiven   :1;
    unsigned  B3SOIwmoinGiven   :1;

    /* Cross-term Dependence */
/* v3.0 */
    unsigned  B3SOIpaigcGiven   :1;
    unsigned  B3SOIpbigcGiven   :1;
    unsigned  B3SOIpcigcGiven   :1;
    unsigned  B3SOIpaigsdGiven   :1;
    unsigned  B3SOIpbigsdGiven   :1;
    unsigned  B3SOIpcigsdGiven   :1;   
    unsigned  B3SOIpnigcGiven   :1;
    unsigned  B3SOIppigcdGiven   :1;
    unsigned  B3SOIppoxedgeGiven   :1;

    unsigned  B3SOIpnpeakGiven   :1;        
    unsigned  B3SOIpnsubGiven   :1;
    unsigned  B3SOIpngateGiven   :1;        
    unsigned  B3SOIpvth0Given   :1;
    unsigned  B3SOIpk1Given   :1;
    unsigned  B3SOIpk1w1Given   :1;
    unsigned  B3SOIpk1w2Given   :1;
    unsigned  B3SOIpk2Given   :1;
    unsigned  B3SOIpk3Given   :1;
    unsigned  B3SOIpk3bGiven   :1;
    unsigned  B3SOIpkb1Given   :1;
    unsigned  B3SOIpw0Given   :1;
    unsigned  B3SOIpnlxGiven   :1;
    unsigned  B3SOIpdvt0Given   :1;      
    unsigned  B3SOIpdvt1Given   :1;      
    unsigned  B3SOIpdvt2Given   :1;      
    unsigned  B3SOIpdvt0wGiven   :1;      
    unsigned  B3SOIpdvt1wGiven   :1;      
    unsigned  B3SOIpdvt2wGiven   :1;      
    unsigned  B3SOIpu0Given   :1;
    unsigned  B3SOIpuaGiven   :1;
    unsigned  B3SOIpubGiven   :1;
    unsigned  B3SOIpucGiven   :1;
    unsigned  B3SOIpvsatGiven   :1;         
    unsigned  B3SOIpa0Given   :1;   
    unsigned  B3SOIpagsGiven   :1;      
    unsigned  B3SOIpb0Given   :1;
    unsigned  B3SOIpb1Given   :1;
    unsigned  B3SOIpketaGiven   :1;
    unsigned  B3SOIpketasGiven   :1;
    unsigned  B3SOIpa1Given   :1;         
    unsigned  B3SOIpa2Given   :1;         
    unsigned  B3SOIprdswGiven   :1;       
    unsigned  B3SOIpprwbGiven   :1;
    unsigned  B3SOIpprwgGiven   :1;
    unsigned  B3SOIpwrGiven   :1;
    unsigned  B3SOIpnfactorGiven   :1;      
    unsigned  B3SOIpdwgGiven   :1;
    unsigned  B3SOIpdwbGiven   :1;
    unsigned  B3SOIpvoffGiven   :1;
    unsigned  B3SOIpeta0Given   :1;         
    unsigned  B3SOIpetabGiven   :1;         
    unsigned  B3SOIpdsubGiven   :1;      
    unsigned  B3SOIpcitGiven   :1;           
    unsigned  B3SOIpcdscGiven   :1;           
    unsigned  B3SOIpcdscbGiven   :1; 
    unsigned  B3SOIpcdscdGiven   :1;          
    unsigned  B3SOIppclmGiven   :1;      
    unsigned  B3SOIppdibl1Given   :1;      
    unsigned  B3SOIppdibl2Given   :1;      
    unsigned  B3SOIppdiblbGiven   :1;      
    unsigned  B3SOIpdroutGiven   :1;      
    unsigned  B3SOIppvagGiven   :1;       
    unsigned  B3SOIpdeltaGiven   :1;
    unsigned  B3SOIpalpha0Given   :1;
    unsigned  B3SOIpfbjtiiGiven   :1;
    unsigned  B3SOIpbeta0Given   :1;
    unsigned  B3SOIpbeta1Given   :1;
    unsigned  B3SOIpbeta2Given   :1;         
    unsigned  B3SOIpvdsatii0Given   :1;     
    unsigned  B3SOIpliiGiven   :1;      
    unsigned  B3SOIpesatiiGiven   :1;     
    unsigned  B3SOIpsii0Given   :1;      
    unsigned  B3SOIpsii1Given   :1;       
    unsigned  B3SOIpsii2Given   :1;       
    unsigned  B3SOIpsiidGiven   :1;
    unsigned  B3SOIpagidlGiven   :1;
    unsigned  B3SOIpbgidlGiven   :1;
    unsigned  B3SOIpngidlGiven   :1;
    unsigned  B3SOIpntunGiven   :1;
    unsigned  B3SOIpndiodeGiven   :1;
    unsigned  B3SOIpnrecf0Given   :1;
    unsigned  B3SOIpnrecr0Given   :1;
    unsigned  B3SOIpisbjtGiven   :1;       
    unsigned  B3SOIpisdifGiven   :1;
    unsigned  B3SOIpisrecGiven   :1;       
    unsigned  B3SOIpistunGiven   :1;       
    unsigned  B3SOIpvrec0Given   :1;
    unsigned  B3SOIpvtun0Given   :1;
    unsigned  B3SOIpnbjtGiven   :1;
    unsigned  B3SOIplbjt0Given   :1;
    unsigned  B3SOIpvabjtGiven   :1;
    unsigned  B3SOIpaelyGiven   :1;
    unsigned  B3SOIpahliGiven   :1;
    /* CV model */
    unsigned  B3SOIpvsdfbGiven   :1;
    unsigned  B3SOIpvsdthGiven   :1;
    unsigned  B3SOIpdelvtGiven   :1;
    unsigned  B3SOIpacdeGiven   :1;
    unsigned  B3SOIpmoinGiven   :1;
/* Added for binning - END2 */

    unsigned  B3SOIuseFringeGiven   :1;

    unsigned  B3SOItnomGiven   :1;
    unsigned  B3SOIcgsoGiven   :1;
    unsigned  B3SOIcgdoGiven   :1;
    unsigned  B3SOIcgeoGiven   :1;
    unsigned  B3SOIxpartGiven   :1;
    unsigned  B3SOIsheetResistanceGiven   :1;
    unsigned  B3SOIGatesidewallJctPotentialGiven   :1;
    unsigned  B3SOIbodyJctGateSideGradingCoeffGiven   :1;
    unsigned  B3SOIunitLengthGateSidewallJctCapGiven   :1;
    unsigned  B3SOIcsdeswGiven :1;

    unsigned  B3SOIoxideTrapDensityAGiven  :1;         
    unsigned  B3SOIoxideTrapDensityBGiven  :1;        
    unsigned  B3SOIoxideTrapDensityCGiven  :1;     
    unsigned  B3SOIemGiven  :1;     
    unsigned  B3SOIefGiven  :1;     
    unsigned  B3SOIafGiven  :1;     
    unsigned  B3SOIkfGiven  :1;     
    unsigned  B3SOInoifGiven  :1;     

    unsigned  B3SOILintGiven   :1;
    unsigned  B3SOILlGiven   :1;
    unsigned  B3SOILlcGiven   :1; /* v2.2.3 */
    unsigned  B3SOILlnGiven   :1;
    unsigned  B3SOILwGiven   :1;
    unsigned  B3SOILwcGiven   :1; /* v2.2.3 */
    unsigned  B3SOILwnGiven   :1;
    unsigned  B3SOILwlGiven   :1;
    unsigned  B3SOILwlcGiven   :1; /* v2.2.3 */
    unsigned  B3SOILminGiven   :1;
    unsigned  B3SOILmaxGiven   :1;

    unsigned  B3SOIWintGiven   :1;
    unsigned  B3SOIWlGiven   :1;
    unsigned  B3SOIWlcGiven   :1; /* v2.2.3 */
    unsigned  B3SOIWlnGiven   :1;
    unsigned  B3SOIWwGiven   :1;
    unsigned  B3SOIWwcGiven   :1; /* v2.2.3 */
    unsigned  B3SOIWwnGiven   :1;
    unsigned  B3SOIWwlGiven   :1;
    unsigned  B3SOIWwlcGiven   :1;  /* v2.2.3 */
    unsigned  B3SOIWminGiven   :1;
    unsigned  B3SOIWmaxGiven   :1;

} B3SOImodel;


#ifndef NMOS
#define NMOS 1
#define PMOS -1
#endif /*NMOS*/


/* device parameters */
#define B3SOI_W 1
#define B3SOI_L 2
#define B3SOI_M 31
#define B3SOI_AS 3
#define B3SOI_AD 4
#define B3SOI_PS 5
#define B3SOI_PD 6
#define B3SOI_NRS 7
#define B3SOI_NRD 8
#define B3SOI_OFF 9
#define B3SOI_IC_VBS 10
#define B3SOI_IC_VDS 11
#define B3SOI_IC_VGS 12
#define B3SOI_IC_VES 13
#define B3SOI_IC_VPS 14
#define B3SOI_BJTOFF 15
#define B3SOI_RTH0 16
#define B3SOI_CTH0 17
#define B3SOI_NRB 18
#define B3SOI_IC 19
#define B3SOI_NQSMOD 20
#define B3SOI_DEBUG  21

/* v2.0 release */
#define B3SOI_NBC    22  
#define B3SOI_NSEG   23
#define B3SOI_PDBCP  24
#define B3SOI_PSBCP  25
#define B3SOI_AGBCP  26
#define B3SOI_AEBCP  27
#define B3SOI_VBSUSR 28
#define B3SOI_TNODEOUT 29

/* v2.2.2 */
#define B3SOI_FRBODY 30


/* model parameters */
#define B3SOI_MOD_CAPMOD          101
#define B3SOI_MOD_NQSMOD          102
#define B3SOI_MOD_MOBMOD          103    
#define B3SOI_MOD_NOIMOD          104    
#define B3SOI_MOD_SHMOD           105
#define B3SOI_MOD_DDMOD           106

#define B3SOI_MOD_TOX             107

#define B3SOI_MOD_CDSC            108
#define B3SOI_MOD_CDSCB           109
#define B3SOI_MOD_CIT             110
#define B3SOI_MOD_NFACTOR         111
#define B3SOI_MOD_XJ              112
#define B3SOI_MOD_VSAT            113
#define B3SOI_MOD_AT              114
#define B3SOI_MOD_A0              115
#define B3SOI_MOD_A1              116
#define B3SOI_MOD_A2              117
#define B3SOI_MOD_KETA            118   
#define B3SOI_MOD_NSUB            119
#define B3SOI_MOD_NPEAK           120
#define B3SOI_MOD_NGATE           121
#define B3SOI_MOD_GAMMA1          122
#define B3SOI_MOD_GAMMA2          123
#define B3SOI_MOD_VBX             124
#define B3SOI_MOD_BINUNIT         125    

#define B3SOI_MOD_VBM             126

#define B3SOI_MOD_XT              127
#define B3SOI_MOD_K1              129
#define B3SOI_MOD_KT1             130
#define B3SOI_MOD_KT1L            131
#define B3SOI_MOD_K2              132
#define B3SOI_MOD_KT2             133
#define B3SOI_MOD_K3              134
#define B3SOI_MOD_K3B             135
#define B3SOI_MOD_W0              136
#define B3SOI_MOD_NLX             137

#define B3SOI_MOD_DVT0            138
#define B3SOI_MOD_DVT1            139
#define B3SOI_MOD_DVT2            140

#define B3SOI_MOD_DVT0W           141
#define B3SOI_MOD_DVT1W           142
#define B3SOI_MOD_DVT2W           143

#define B3SOI_MOD_DROUT           144
#define B3SOI_MOD_DSUB            145
#define B3SOI_MOD_VTH0            146
#define B3SOI_MOD_UA              147
#define B3SOI_MOD_UA1             148
#define B3SOI_MOD_UB              149
#define B3SOI_MOD_UB1             150
#define B3SOI_MOD_UC              151
#define B3SOI_MOD_UC1             152
#define B3SOI_MOD_U0              153
#define B3SOI_MOD_UTE             154
#define B3SOI_MOD_VOFF            155
#define B3SOI_MOD_DELTA           156
#define B3SOI_MOD_RDSW            157
#define B3SOI_MOD_PRT             158
#define B3SOI_MOD_LDD             159
#define B3SOI_MOD_ETA             160
#define B3SOI_MOD_ETA0            161
#define B3SOI_MOD_ETAB            162
#define B3SOI_MOD_PCLM            163
#define B3SOI_MOD_PDIBL1          164
#define B3SOI_MOD_PDIBL2          165
#define B3SOI_MOD_PSCBE1          166
#define B3SOI_MOD_PSCBE2          167
#define B3SOI_MOD_PVAG            168
#define B3SOI_MOD_WR              169
#define B3SOI_MOD_DWG             170
#define B3SOI_MOD_DWB             171
#define B3SOI_MOD_B0              172
#define B3SOI_MOD_B1              173
#define B3SOI_MOD_ALPHA0          174
#define B3SOI_MOD_PDIBLB          178

#define B3SOI_MOD_PRWG            179
#define B3SOI_MOD_PRWB            180

#define B3SOI_MOD_CDSCD           181
#define B3SOI_MOD_AGS             182

#define B3SOI_MOD_FRINGE          184
#define B3SOI_MOD_CGSL            186
#define B3SOI_MOD_CGDL            187
#define B3SOI_MOD_CKAPPA          188
#define B3SOI_MOD_CF              189
#define B3SOI_MOD_CLC             190
#define B3SOI_MOD_CLE             191
#define B3SOI_MOD_PARAMCHK        192
#define B3SOI_MOD_VERSION         193

#define B3SOI_MOD_TBOX            195
#define B3SOI_MOD_TSI             196
#define B3SOI_MOD_KB1             197
#define B3SOI_MOD_KB3             198
#define B3SOI_MOD_DELP            201
#define B3SOI_MOD_RBODY           204
#define B3SOI_MOD_ADICE0          205
#define B3SOI_MOD_ABP             206
#define B3SOI_MOD_MXC             207
#define B3SOI_MOD_RTH0            208
#define B3SOI_MOD_CTH0            209
#define B3SOI_MOD_ALPHA1	  214
#define B3SOI_MOD_NGIDL           215
#define B3SOI_MOD_AGIDL           216
#define B3SOI_MOD_BGIDL           217
#define B3SOI_MOD_NDIODE          218
#define B3SOI_MOD_LDIOF           219
#define B3SOI_MOD_LDIOR           220
#define B3SOI_MOD_NTUN            221
#define B3SOI_MOD_ISBJT           222
#define B3SOI_MOD_ISDIF           223
#define B3SOI_MOD_ISREC           224
#define B3SOI_MOD_ISTUN           225
#define B3SOI_MOD_XBJT            226
#define B3SOI_MOD_XDIF            227
#define B3SOI_MOD_XREC            228
#define B3SOI_MOD_XTUN            229
#define B3SOI_MOD_TT              232
#define B3SOI_MOD_VSDTH           233
#define B3SOI_MOD_VSDFB           234
#define B3SOI_MOD_ASD             235
#define B3SOI_MOD_CSDMIN          236
#define B3SOI_MOD_RBSH            237
#define B3SOI_MOD_ESATII          238


/* v2.0 release */
#define B3SOI_MOD_K1W1            239     
#define B3SOI_MOD_K1W2            240
#define B3SOI_MOD_KETAS           241
#define B3SOI_MOD_DWBC            242
#define B3SOI_MOD_BETA0           243
#define B3SOI_MOD_BETA1           244
#define B3SOI_MOD_BETA2           245
#define B3SOI_MOD_VDSATII0        246
#define B3SOI_MOD_TII             247
#define B3SOI_MOD_LII             248
#define B3SOI_MOD_SII0            249
#define B3SOI_MOD_SII1            250
#define B3SOI_MOD_SII2            251
#define B3SOI_MOD_SIID            252
#define B3SOI_MOD_FBJTII          253
#define B3SOI_MOD_NRECF0          255
#define B3SOI_MOD_NRECR0          256
#define B3SOI_MOD_LN              257
#define B3SOI_MOD_VREC0           258
#define B3SOI_MOD_VTUN0           259
#define B3SOI_MOD_NBJT            260
#define B3SOI_MOD_LBJT0           261
#define B3SOI_MOD_VABJT           262
#define B3SOI_MOD_AELY            263
#define B3SOI_MOD_AHLI            264
#define B3SOI_MOD_NTRECF          265
#define B3SOI_MOD_NTRECR          266
#define B3SOI_MOD_DLCB            267
#define B3SOI_MOD_FBODY           268
#define B3SOI_MOD_NDIF            269
#define B3SOI_MOD_TCJSWG          270
#define B3SOI_MOD_TPBSWG          271
#define B3SOI_MOD_ACDE            272
#define B3SOI_MOD_MOIN            273
#define B3SOI_MOD_DELVT           274
#define B3SOI_MOD_DLBG            275
#define B3SOI_MOD_LDIF0           276


/* v2.2 release */
#define B3SOI_MOD_WTH0            277
#define B3SOI_MOD_RHALO           278
#define B3SOI_MOD_NTOX            279
#define B3SOI_MOD_TOXREF          280
#define B3SOI_MOD_EBG             281
#define B3SOI_MOD_VEVB            282
#define B3SOI_MOD_ALPHAGB1        283
#define B3SOI_MOD_BETAGB1         284
#define B3SOI_MOD_VGB1            285
#define B3SOI_MOD_VECB            286
#define B3SOI_MOD_ALPHAGB2        287
#define B3SOI_MOD_BETAGB2         288
#define B3SOI_MOD_VGB2            289
#define B3SOI_MOD_TOXQM           290
#define B3SOI_MOD_IGBMOD          291 /* v3.0 */
#define B3SOI_MOD_VOXH            292
#define B3SOI_MOD_DELTAVOX        293
#define B3SOI_MOD_IGCMOD          294 /* v3.0 */


/* Added for binning - START3 */
/* Length dependence */
#define B3SOI_MOD_LNPEAK           301
#define B3SOI_MOD_LNSUB            302
#define B3SOI_MOD_LNGATE           303
#define B3SOI_MOD_LVTH0            304
#define B3SOI_MOD_LK1              305     
#define B3SOI_MOD_LK1W1            306
#define B3SOI_MOD_LK1W2            307
#define B3SOI_MOD_LK2              308
#define B3SOI_MOD_LK3              309
#define B3SOI_MOD_LK3B             310
#define B3SOI_MOD_LKB1             311
#define B3SOI_MOD_LW0              312
#define B3SOI_MOD_LNLX             313
#define B3SOI_MOD_LDVT0            314
#define B3SOI_MOD_LDVT1            315
#define B3SOI_MOD_LDVT2            316
#define B3SOI_MOD_LDVT0W           317
#define B3SOI_MOD_LDVT1W           318
#define B3SOI_MOD_LDVT2W           319
#define B3SOI_MOD_LU0              320
#define B3SOI_MOD_LUA              321
#define B3SOI_MOD_LUB              322
#define B3SOI_MOD_LUC              323
#define B3SOI_MOD_LVSAT            324
#define B3SOI_MOD_LA0              325
#define B3SOI_MOD_LAGS             326
#define B3SOI_MOD_LB0              327
#define B3SOI_MOD_LB1              328
#define B3SOI_MOD_LKETA            329   
#define B3SOI_MOD_LKETAS           330   
#define B3SOI_MOD_LA1              331
#define B3SOI_MOD_LA2              332
#define B3SOI_MOD_LRDSW            333
#define B3SOI_MOD_LPRWB            334
#define B3SOI_MOD_LPRWG            335
#define B3SOI_MOD_LWR              336
#define B3SOI_MOD_LNFACTOR         337
#define B3SOI_MOD_LDWG             338
#define B3SOI_MOD_LDWB             339
#define B3SOI_MOD_LVOFF            340
#define B3SOI_MOD_LETA0            341
#define B3SOI_MOD_LETAB            342
#define B3SOI_MOD_LDSUB            343
#define B3SOI_MOD_LCIT             344
#define B3SOI_MOD_LCDSC            345
#define B3SOI_MOD_LCDSCB           346
#define B3SOI_MOD_LCDSCD           347
#define B3SOI_MOD_LPCLM            348
#define B3SOI_MOD_LPDIBL1          349
#define B3SOI_MOD_LPDIBL2          350
#define B3SOI_MOD_LPDIBLB          351
#define B3SOI_MOD_LDROUT           352
#define B3SOI_MOD_LPVAG            353
#define B3SOI_MOD_LDELTA           354
#define B3SOI_MOD_LALPHA0          355
#define B3SOI_MOD_LFBJTII          356
#define B3SOI_MOD_LBETA0           357
#define B3SOI_MOD_LBETA1           358
#define B3SOI_MOD_LBETA2           359
#define B3SOI_MOD_LVDSATII0        360
#define B3SOI_MOD_LLII             361
#define B3SOI_MOD_LESATII          362
#define B3SOI_MOD_LSII0            363
#define B3SOI_MOD_LSII1            364
#define B3SOI_MOD_LSII2            365
#define B3SOI_MOD_LSIID            366
#define B3SOI_MOD_LAGIDL           367
#define B3SOI_MOD_LBGIDL           368
#define B3SOI_MOD_LNGIDL           369
#define B3SOI_MOD_LNTUN            370
#define B3SOI_MOD_LNDIODE          371
#define B3SOI_MOD_LNRECF0          372
#define B3SOI_MOD_LNRECR0          373
#define B3SOI_MOD_LISBJT           374
#define B3SOI_MOD_LISDIF           375
#define B3SOI_MOD_LISREC           376
#define B3SOI_MOD_LISTUN           377
#define B3SOI_MOD_LVREC0           378
#define B3SOI_MOD_LVTUN0           379
#define B3SOI_MOD_LNBJT            380
#define B3SOI_MOD_LLBJT0           381
#define B3SOI_MOD_LVABJT           382
#define B3SOI_MOD_LAELY            383
#define B3SOI_MOD_LAHLI            384
#define B3SOI_MOD_LVSDFB           385
#define B3SOI_MOD_LVSDTH           386
#define B3SOI_MOD_LDELVT           387
#define B3SOI_MOD_LACDE            388
#define B3SOI_MOD_LMOIN            389

/* Width dependence */
#define B3SOI_MOD_WNPEAK           401
#define B3SOI_MOD_WNSUB            402
#define B3SOI_MOD_WNGATE           403
#define B3SOI_MOD_WVTH0            404
#define B3SOI_MOD_WK1              405     
#define B3SOI_MOD_WK1W1            406
#define B3SOI_MOD_WK1W2            407
#define B3SOI_MOD_WK2              408
#define B3SOI_MOD_WK3              409
#define B3SOI_MOD_WK3B             410
#define B3SOI_MOD_WKB1             411
#define B3SOI_MOD_WW0              412
#define B3SOI_MOD_WNLX             413
#define B3SOI_MOD_WDVT0            414
#define B3SOI_MOD_WDVT1            415
#define B3SOI_MOD_WDVT2            416
#define B3SOI_MOD_WDVT0W           417
#define B3SOI_MOD_WDVT1W           418
#define B3SOI_MOD_WDVT2W           419
#define B3SOI_MOD_WU0              420
#define B3SOI_MOD_WUA              421
#define B3SOI_MOD_WUB              422
#define B3SOI_MOD_WUC              423
#define B3SOI_MOD_WVSAT            424
#define B3SOI_MOD_WA0              425
#define B3SOI_MOD_WAGS             426
#define B3SOI_MOD_WB0              427
#define B3SOI_MOD_WB1              428
#define B3SOI_MOD_WKETA            429   
#define B3SOI_MOD_WKETAS           430   
#define B3SOI_MOD_WA1              431
#define B3SOI_MOD_WA2              432
#define B3SOI_MOD_WRDSW            433
#define B3SOI_MOD_WPRWB            434
#define B3SOI_MOD_WPRWG            435
#define B3SOI_MOD_WWR              436
#define B3SOI_MOD_WNFACTOR         437
#define B3SOI_MOD_WDWG             438
#define B3SOI_MOD_WDWB             439
#define B3SOI_MOD_WVOFF            440
#define B3SOI_MOD_WETA0            441
#define B3SOI_MOD_WETAB            442
#define B3SOI_MOD_WDSUB            443
#define B3SOI_MOD_WCIT             444
#define B3SOI_MOD_WCDSC            445
#define B3SOI_MOD_WCDSCB           446
#define B3SOI_MOD_WCDSCD           447
#define B3SOI_MOD_WPCLM            448
#define B3SOI_MOD_WPDIBL1          449
#define B3SOI_MOD_WPDIBL2          450
#define B3SOI_MOD_WPDIBLB          451
#define B3SOI_MOD_WDROUT           452
#define B3SOI_MOD_WPVAG            453
#define B3SOI_MOD_WDELTA           454
#define B3SOI_MOD_WALPHA0          455
#define B3SOI_MOD_WFBJTII          456
#define B3SOI_MOD_WBETA0           457
#define B3SOI_MOD_WBETA1           458
#define B3SOI_MOD_WBETA2           459
#define B3SOI_MOD_WVDSATII0        460
#define B3SOI_MOD_WLII             461
#define B3SOI_MOD_WESATII          462
#define B3SOI_MOD_WSII0            463
#define B3SOI_MOD_WSII1            464
#define B3SOI_MOD_WSII2            465
#define B3SOI_MOD_WSIID            466
#define B3SOI_MOD_WAGIDL           467
#define B3SOI_MOD_WBGIDL           468
#define B3SOI_MOD_WNGIDL           469
#define B3SOI_MOD_WNTUN            470
#define B3SOI_MOD_WNDIODE          471
#define B3SOI_MOD_WNRECF0          472
#define B3SOI_MOD_WNRECR0          473
#define B3SOI_MOD_WISBJT           474
#define B3SOI_MOD_WISDIF           475
#define B3SOI_MOD_WISREC           476
#define B3SOI_MOD_WISTUN           477
#define B3SOI_MOD_WVREC0           478
#define B3SOI_MOD_WVTUN0           479
#define B3SOI_MOD_WNBJT            480
#define B3SOI_MOD_WLBJT0           481
#define B3SOI_MOD_WVABJT           482
#define B3SOI_MOD_WAELY            483
#define B3SOI_MOD_WAHLI            484
#define B3SOI_MOD_WVSDFB           485
#define B3SOI_MOD_WVSDTH           486
#define B3SOI_MOD_WDELVT           487
#define B3SOI_MOD_WACDE            488
#define B3SOI_MOD_WMOIN            489

/* Cross-term dependence */
#define B3SOI_MOD_PNPEAK           501
#define B3SOI_MOD_PNSUB            502
#define B3SOI_MOD_PNGATE           503
#define B3SOI_MOD_PVTH0            504
#define B3SOI_MOD_PK1              505     
#define B3SOI_MOD_PK1W1            506
#define B3SOI_MOD_PK1W2            507
#define B3SOI_MOD_PK2              508
#define B3SOI_MOD_PK3              509
#define B3SOI_MOD_PK3B             510
#define B3SOI_MOD_PKB1             511
#define B3SOI_MOD_PW0              512
#define B3SOI_MOD_PNLX             513
#define B3SOI_MOD_PDVT0            514
#define B3SOI_MOD_PDVT1            515
#define B3SOI_MOD_PDVT2            516
#define B3SOI_MOD_PDVT0W           517
#define B3SOI_MOD_PDVT1W           518
#define B3SOI_MOD_PDVT2W           519
#define B3SOI_MOD_PU0              520
#define B3SOI_MOD_PUA              521
#define B3SOI_MOD_PUB              522
#define B3SOI_MOD_PUC              523
#define B3SOI_MOD_PVSAT            524
#define B3SOI_MOD_PA0              525
#define B3SOI_MOD_PAGS             526
#define B3SOI_MOD_PB0              527
#define B3SOI_MOD_PB1              528
#define B3SOI_MOD_PKETA            529   
#define B3SOI_MOD_PKETAS           530   
#define B3SOI_MOD_PA1              531
#define B3SOI_MOD_PA2              532
#define B3SOI_MOD_PRDSW            533
#define B3SOI_MOD_PPRWB            534
#define B3SOI_MOD_PPRWG            535
#define B3SOI_MOD_PWR              536
#define B3SOI_MOD_PNFACTOR         537
#define B3SOI_MOD_PDWG             538
#define B3SOI_MOD_PDWB             539
#define B3SOI_MOD_PVOFF            540
#define B3SOI_MOD_PETA0            541
#define B3SOI_MOD_PETAB            542
#define B3SOI_MOD_PDSUB            543
#define B3SOI_MOD_PCIT             544
#define B3SOI_MOD_PCDSC            545
#define B3SOI_MOD_PCDSCB           546
#define B3SOI_MOD_PCDSCD           547
#define B3SOI_MOD_PPCLM            548
#define B3SOI_MOD_PPDIBL1          549
#define B3SOI_MOD_PPDIBL2          550
#define B3SOI_MOD_PPDIBLB          551
#define B3SOI_MOD_PDROUT           552
#define B3SOI_MOD_PPVAG            553
#define B3SOI_MOD_PDELTA           554
#define B3SOI_MOD_PALPHA0          555
#define B3SOI_MOD_PFBJTII          556
#define B3SOI_MOD_PBETA0           557
#define B3SOI_MOD_PBETA1           558
#define B3SOI_MOD_PBETA2           559
#define B3SOI_MOD_PVDSATII0        560
#define B3SOI_MOD_PLII             561
#define B3SOI_MOD_PESATII          562
#define B3SOI_MOD_PSII0            563
#define B3SOI_MOD_PSII1            564
#define B3SOI_MOD_PSII2            565
#define B3SOI_MOD_PSIID            566
#define B3SOI_MOD_PAGIDL           567
#define B3SOI_MOD_PBGIDL           568
#define B3SOI_MOD_PNGIDL           569
#define B3SOI_MOD_PNTUN            570
#define B3SOI_MOD_PNDIODE          571
#define B3SOI_MOD_PNRECF0          572
#define B3SOI_MOD_PNRECR0          573
#define B3SOI_MOD_PISBJT           574
#define B3SOI_MOD_PISDIF           575
#define B3SOI_MOD_PISREC           576
#define B3SOI_MOD_PISTUN           577
#define B3SOI_MOD_PVREC0           578
#define B3SOI_MOD_PVTUN0           579
#define B3SOI_MOD_PNBJT            580
#define B3SOI_MOD_PLBJT0           581
#define B3SOI_MOD_PVABJT           582
#define B3SOI_MOD_PAELY            583
#define B3SOI_MOD_PAHLI            584
#define B3SOI_MOD_PVSDFB           585
#define B3SOI_MOD_PVSDTH           586
#define B3SOI_MOD_PDELVT           587
#define B3SOI_MOD_PACDE            588
#define B3SOI_MOD_PMOIN            589
/* Added for binning - END3 */

#define B3SOI_MOD_TNOM             701
#define B3SOI_MOD_CGSO             702
#define B3SOI_MOD_CGDO             703
#define B3SOI_MOD_CGEO             704
#define B3SOI_MOD_XPART            705

#define B3SOI_MOD_RSH              706
#define B3SOI_MOD_NMOS             814
#define B3SOI_MOD_PMOS             815

#define B3SOI_MOD_NOIA             816
#define B3SOI_MOD_NOIB             817
#define B3SOI_MOD_NOIC             818

#define B3SOI_MOD_LINT             819
#define B3SOI_MOD_LL               820
#define B3SOI_MOD_LLN              821
#define B3SOI_MOD_LW               822
#define B3SOI_MOD_LWN              823
#define B3SOI_MOD_LWL              824

#define B3SOI_MOD_WINT             827
#define B3SOI_MOD_WL               828
#define B3SOI_MOD_WLN              829
#define B3SOI_MOD_WW               830
#define B3SOI_MOD_WWN              831
#define B3SOI_MOD_WWL              832

/* v2.2.3 */
#define B3SOI_MOD_LWLC             847
#define B3SOI_MOD_LLC              848
#define B3SOI_MOD_LWC              849
#define B3SOI_MOD_WWLC             850  
#define B3SOI_MOD_WLC              851
#define B3SOI_MOD_WWC              852
#define B3SOI_MOD_DTOXCV           853 

#define B3SOI_MOD_DWC              835
#define B3SOI_MOD_DLC              836

#define B3SOI_MOD_EM               837
#define B3SOI_MOD_EF               838
#define B3SOI_MOD_AF               839
#define B3SOI_MOD_KF               840
#define B3SOI_MOD_NOIF             841


#define B3SOI_MOD_PBSWG            843
#define B3SOI_MOD_MJSWG            844
#define B3SOI_MOD_CJSWG            845
#define B3SOI_MOD_CSDESW           846

/* device questions */
#define B3SOI_DNODE                901
#define B3SOI_GNODE                902
#define B3SOI_SNODE                903
#define B3SOI_BNODE                904
#define B3SOI_ENODE                905
#define B3SOI_DNODEPRIME           906
#define B3SOI_SNODEPRIME           907
#define B3SOI_VBD                  908
#define B3SOI_VBS                  909
#define B3SOI_VGS                  910
#define B3SOI_VES                  911
#define B3SOI_VDS                  912
#define B3SOI_CD                   913
#define B3SOI_CBS                  914
#define B3SOI_CBD                  915
#define B3SOI_GM                   916
#define B3SOI_GDS                  917
#define B3SOI_GMBS                 918
#define B3SOI_GBD                  919
#define B3SOI_GBS                  920
#define B3SOI_QB                   921
#define B3SOI_CQB                  922
#define B3SOI_QG                   923
#define B3SOI_CQG                  924
#define B3SOI_QD                   925
#define B3SOI_CQD                  926
#define B3SOI_CGG                  927
#define B3SOI_CGD                  928
#define B3SOI_CGS                  929
#define B3SOI_CBG                  930
#define B3SOI_CAPBD                931
#define B3SOI_CQBD                 932
#define B3SOI_CAPBS                933
#define B3SOI_CQBS                 934
#define B3SOI_CDG                  935
#define B3SOI_CDD                  936
#define B3SOI_CDS                  937
#define B3SOI_VON                  938
#define B3SOI_VDSAT                939
#define B3SOI_QBS                  940
#define B3SOI_QBD                  941
#define B3SOI_SOURCECONDUCT        942
#define B3SOI_DRAINCONDUCT         943
#define B3SOI_CBDB                 944
#define B3SOI_CBSB                 945
#define B3SOI_GMID                 946

/* v3.0 */
#define B3SOI_MOD_SOIMOD           1001
#define B3SOI_MOD_VBSA             1002
#define B3SOI_MOD_NOFFFD           1003
#define B3SOI_MOD_VOFFFD           1004
#define B3SOI_MOD_K1B              1005
#define B3SOI_MOD_K2B              1006
#define B3SOI_MOD_DK2B             1007
#define B3SOI_MOD_DVBD0            1008
#define B3SOI_MOD_DVBD1            1009
#define B3SOI_MOD_MOINFD           1010

/* v3.0 */
#define B3SOI_MOD_AIGC             1021
#define B3SOI_MOD_BIGC             1022 
#define B3SOI_MOD_CIGC             1023 
#define B3SOI_MOD_AIGSD            1024
#define B3SOI_MOD_BIGSD            1025
#define B3SOI_MOD_CIGSD            1026
#define B3SOI_MOD_NIGC             1027
#define B3SOI_MOD_PIGCD            1028
#define B3SOI_MOD_POXEDGE          1029
#define B3SOI_MOD_DLCIG            1030

#define B3SOI_MOD_LAIGC            1031
#define B3SOI_MOD_LBIGC            1032
#define B3SOI_MOD_LCIGC            1033
#define B3SOI_MOD_LAIGSD           1034
#define B3SOI_MOD_LBIGSD           1035
#define B3SOI_MOD_LCIGSD           1036
#define B3SOI_MOD_LNIGC            1037
#define B3SOI_MOD_LPIGCD           1038
#define B3SOI_MOD_LPOXEDGE         1039

#define B3SOI_MOD_WAIGC            1041
#define B3SOI_MOD_WBIGC            1042
#define B3SOI_MOD_WCIGC            1043
#define B3SOI_MOD_WAIGSD           1044
#define B3SOI_MOD_WBIGSD           1045
#define B3SOI_MOD_WCIGSD           1046
#define B3SOI_MOD_WNIGC            1047
#define B3SOI_MOD_WPIGCD           1048
#define B3SOI_MOD_WPOXEDGE         1049

#define B3SOI_MOD_PAIGC            1051
#define B3SOI_MOD_PBIGC            1052
#define B3SOI_MOD_PCIGC            1053
#define B3SOI_MOD_PAIGSD           1054
#define B3SOI_MOD_PBIGSD           1055
#define B3SOI_MOD_PCIGSD           1056
#define B3SOI_MOD_PNIGC            1057
#define B3SOI_MOD_PPIGCD           1058
#define B3SOI_MOD_PPOXEDGE         1059



#include "b3soiext.h"

extern void B3SOIevaluate(double,double,double,B3SOIinstance*,B3SOImodel*,
        double*,double*,double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, CKTcircuit*);
extern int B3SOIdebug(B3SOImodel*, B3SOIinstance*, CKTcircuit*, int);
extern int B3SOIcheckModel(B3SOImodel*, B3SOIinstance*, CKTcircuit*);

#endif /*B3SOI*/

