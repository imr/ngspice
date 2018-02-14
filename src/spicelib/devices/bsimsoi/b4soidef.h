/**********
Copyright 2010 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung
authors:1999-2004 Pin Su, Hui Wan b3soidef.h
Authors:2005-  Hui Wan, Jane Xi
Authors: 2009- Wenwei Yang, Chung-Hsun Lin, Ali Niknejad, Chenming Hu
Authors: 2009- Tanvir Morshed, Ali Niknejad, Chenming Hu
Authors: 2010- Tanvir Morshed, Ali Niknejad, Chenming Hu
File: b4soidef.h
**********/

#ifndef B4SOI
#define B4SOI

#define SOICODE

/* Uncomment the following line to activate debugging variable output */
/* debug1, debug2, debug3, ... */
/* #define B4SOI_DEBUG_OUT */

#include "ngspice/ifsim.h"
#include "ngspice/gendefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/noisedef.h"

typedef struct sB4SOIinstance
{

    struct GENinstance gen;

#define B4SOImodPtr(inst) ((struct sB4SOImodel *)((inst)->gen.GENmodPtr))
#define B4SOInextInstance(inst) ((struct sB4SOIinstance *)((inst)->gen.GENnextInstance))
#define B4SOIname gen.GENname
#define B4SOIstates gen.GENstate

    const int B4SOIdNode;
    const int B4SOIgNodeExt; /* v3.1 changed gNode to gNodeExt */
    const int B4SOIsNode;
    const int B4SOIeNode;
    const int B4SOIpNodeExt;
    const int B4SOIbNodeExt;
    const int B4SOItempNodeExt;
    int B4SOIpNode;
    int B4SOIbNode;
    int B4SOItempNode;
    int B4SOIdNodePrime;
    int B4SOIsNodePrime;
    int B4SOIgNode;    /* v3.1 */
    int B4SOIgNodeMid; /* v3.1 */
    int B4SOIdbNode;   /* v4.0 */
    int B4SOIsbNode;   /* v4.0 */

    int B4SOIvbsNode;
    /* for Debug */
    int B4SOIidsNode;
    int B4SOIicNode;
    int B4SOIibsNode;
    int B4SOIibdNode;
    int B4SOIiiiNode;
    int B4SOIigNode;
    int B4SOIgiggNode;
    int B4SOIgigdNode;
    int B4SOIgigbNode;
    int B4SOIigidlNode;
    int B4SOIitunNode;
    int B4SOIibpNode;
    int B4SOIcbbNode;
    int B4SOIcbdNode;
    int B4SOIcbgNode;

    int B4SOIqbfNode;
    int B4SOIqjsNode;
    int B4SOIqjdNode;

    double B4SOIphi;
    double B4SOIvtm;
    double B4SOIni;
    double B4SOIueff;
    double B4SOIthetavth;
    double B4SOIvon;
    double B4SOIvdsat;
    double B4SOIcgdo;
    double B4SOIcgso;
    double B4SOIcgeo;

    double B4SOIids;
    double B4SOIic;
    double B4SOIibs;
    double B4SOIibd;
    double B4SOIiii;
    double B4SOIig;
    double B4SOIgigg;
    double B4SOIgigd;
    double B4SOIgigb;
    double B4SOIgige; /* v3.0 */
/* v3.1 added for RF */
    double B4SOIgcrg;
    double B4SOIgcrgd;
    double B4SOIgcrgg;
    double B4SOIgcrgs;
    double B4SOIgcrgb;
/* v3.1 added for end */

    double B4SOIInv_ODeff;      /* v4.0 */
    double B4SOIkvsat;          /* v4.0 */
    double B4SOIgrbsb;          /* v4.0 */
    double B4SOIgrbdb;          /* v4.0 */
    double B4SOIrbsb;           /* v4.0 */
    double B4SOIrbdb;           /* v4.0 */
    double B4SOIGGjdb;          /* v4.0 */
    double B4SOIGGjsb;          /* v4.0 */
    double B4SOIgbgiigbpb;      /* v4.0 */
    double B4SOIgiigidlb;       /* v4.0 */
    double B4SOIgiigidld;       /* v4.0 */
    double B4SOIdelvto;         /* v4.0 */
    double B4SOIgstot;          /* v4.0 for rdsmod */
    double B4SOIgstotd;         /* v4.0 for rdsmod */
    double B4SOIgstotg;         /* v4.0 for rdsmod */
    double B4SOIgstots;         /* v4.0 for rdsmod */
    double B4SOIgstotb;         /* v4.0 for rdsmod */

    double B4SOIgdtot;          /* v4.0 for rdsmod */
    double B4SOIgdtotd;         /* v4.0 for rdsmod */
    double B4SOIgdtotg;         /* v4.0 for rdsmod */
    double B4SOIgdtots;         /* v4.0 for rdsmod */
    double B4SOIgdtotb;         /* v4.0 for rdsmod */

    double B4SOIigp;            /* v4.1 for Igb in the AGBCP2 region */
    double B4SOIgigpg;          /* v4.1 for Igb in the AGBCP2 region */
    double B4SOIgigpp;          /* v4.1 for Igb in the AGBCP2 region */
    double B4SOITempSH;         /* v4.2 for portability of SH temp */

    double B4SOIigidl;
        double B4SOIigisl;
    double B4SOIitun;
    double B4SOIibp;
  /*  double B4SOIabeff; */     /* v4.2 never used in the code */
    double B4SOIvbseff;
    double B4SOIcbg;
    double B4SOIcbb;
    double B4SOIcbd;
    double B4SOIqb;
    double B4SOIqbf;
    double B4SOIqjs;
    double B4SOIqjd;
    int    B4SOIfloat;

/* v3.1 for RF */
    double B4SOIgrgeltd;
/* v3.1 for RF end */

    double B4SOIl;
    double B4SOIw;
    double B4SOIm;
    double B4SOIdrainArea;
    double B4SOIsourceArea;
    double B4SOIdrainSquares;
    double B4SOIsourceSquares;
    double B4SOIdrainPerimeter;
    double B4SOIsourcePerimeter;
    double B4SOIsourceConductance;
    double B4SOIdrainConductance;

/* v4.0 stress effect instance param */
    double B4SOIsa;
    double B4SOIsb;
    double B4SOIsd;
    double B4SOIu0temp;
    double B4SOIvsattemp;
    double B4SOIvth0;
    double B4SOIvfb;
    double B4SOIvfbzb;
    double B4SOIk2;
    double B4SOIk2ox;
    double B4SOIeta0;
/* end of v4.0 stress effect instance param */
    /*4.1 mobmod=4*/
        double B4SOIvtfbphi1;
        double B4SOIvgsteffvth;





    double B4SOIicVBS;
    double B4SOIicVDS;
    double B4SOIicVGS;
    double B4SOIicVES;
    double B4SOIicVPS;

    int B4SOIbjtoff;
    int B4SOIbodyMod;
    int B4SOIdebugMod;
    double B4SOIrth0;
    double B4SOIcth0;
    double B4SOIbodySquares;
    double B4SOIrbodyext;
    double B4SOIfrbody;

/* v2.0 release */
    double B4SOInbc;
    double B4SOInseg;
    double B4SOIpdbcp;
    double B4SOIpsbcp;
    double B4SOIagbcp;
    double B4SOIagbcp2; /* v4.1 improvement on BC */
    double B4SOIagbcpd; /* v4.0 */
    double B4SOIaebcp;
    double B4SOIvbsusr;
    int B4SOItnodeout;

/* Deleted from pParam and moved to here */
    double B4SOIcsesw;
    double B4SOIcdesw;
    double B4SOIcsbox;
    double B4SOIcdbox;
    double B4SOIcsmin;
    double B4SOIcdmin;
    double B4SOIst4;
    double B4SOIdt4;

    int B4SOIoff;
    int B4SOImode;

/* v4.0 added */
    double B4SOInf;
/* v4.0 added end */

/* v3.2 added */
    double B4SOInstar;
    double B4SOIAbulk;
/* v3.2 added end */
    int B4SOIsoiMod; /* v3.2 */

/* v3.1 for RF */
    int B4SOIrgateMod;
/* v3.1 for RF end */
/* v4.0 */
    int B4SOIrbodyMod;
/* v4.0 end */

    /* OP point */
    double B4SOIqinv;
    double B4SOIcd;
    double B4SOIcjs;
    double B4SOIcjd;
    double B4SOIcbody;
    double B4SOIcbs;
    double B4SOIIgb;


/* v2.2 release */
    double B4SOIcgate;
    double B4SOIgigs;
    double B4SOIgigT;

    double B4SOIcbodcon;
    double B4SOIcth;
  /*  double B4SOIcsubstrate; */      /* v4.2 never used in the code */

    double B4SOIgm;
    double B4SOIgme;   /* v3.0 */
    double B4SOIcb;
    double B4SOIcdrain;
    double B4SOIgds;
    double B4SOIgmbs;
    double B4SOIgmT;

#ifdef B4SOI_DEBUG_OUT
    double B4SOIdebug1;
    double B4SOIdebug2;
    double B4SOIdebug3;
#endif

    double B4SOIgbbs;
    double B4SOIgbgs;
    double B4SOIgbds;
    double B4SOIgbes; /* v3.0 */
    double B4SOIgbps;
    double B4SOIgbT;

/* v3.0 */
    double B4SOIIgcs;
    double B4SOIgIgcsg;
    double B4SOIgIgcsd;
    double B4SOIgIgcss;
    double B4SOIgIgcsb;
    double B4SOIgIgcse;
    double B4SOIIgcd;
    double B4SOIgIgcdg;
    double B4SOIgIgcdd;
    double B4SOIgIgcds;
    double B4SOIgIgcdb;
    double B4SOIgIgcde;

    double B4SOIIgs;
    double B4SOIgIgsg;
    double B4SOIgIgss;
    double B4SOIIgd;
    double B4SOIgIgdg;
    double B4SOIgIgdd;


    double B4SOIgjsd;
    double B4SOIgjsb;
    double B4SOIgjsg;
    double B4SOIgjsT;

    double B4SOIgjdb;
    double B4SOIgjdd;
    double B4SOIgjdg;
    double B4SOIgjde; /* v3.0 */
    double B4SOIgjdT;

    double B4SOIgbpbs;
    double B4SOIgbpps;
    double B4SOIgbpT;

    double B4SOIgtempb;
    double B4SOIgtempg;
    double B4SOIgtempd;
    double B4SOIgtempe; /* v3.0 */
    double B4SOIgtempT;

    double B4SOIqgate;
    double B4SOIqdrn;
    double B4SOIqsrc;
    double B4SOIqbulk;
    double B4SOIcapbd;
    double B4SOIcapbs;


    double B4SOIcggb;
    double B4SOIcgdb;
    double B4SOIcgsb;
    double B4SOIcgeb;
    double B4SOIcgT;

    double B4SOIcbgb;
    double B4SOIcbdb;
    double B4SOIcbsb;
    double B4SOIcbeb;
    double B4SOIcbT;

    double B4SOIcdgb;
    double B4SOIcddb;
    double B4SOIcdsb;
    double B4SOIcdeb;
    double B4SOIcdT;

    double B4SOIceeb;
    double B4SOIceT;

    double B4SOIqse;
    double B4SOIgcse;
    double B4SOIqde;
    double B4SOIgcde;
    double B4SOIrds; /* v2.2.3 */
    double B4SOIVgsteff; /* v2.2.3 */
    double B4SOIVdseff;  /* v2.2.3 */
    double B4SOIAbovVgst2Vtm; /* v2.2.3 */

/* 4.0 */
    double B4SOIidovVds;
    double B4SOIcjsb;
    double B4SOIcjdb;

/* 4.0 end */

    struct b4soiSizeDependParam  *pParam;

    unsigned B4SOIlGiven :1;
    unsigned B4SOIwGiven :1;

    unsigned B4SOImGiven :1;
    unsigned B4SOIdrainAreaGiven :1;
    unsigned B4SOIsourceAreaGiven    :1;
    unsigned B4SOIdrainSquaresGiven  :1;
    unsigned B4SOIsourceSquaresGiven :1;
    unsigned B4SOIdrainPerimeterGiven    :1;
    unsigned B4SOIsourcePerimeterGiven   :1;
    unsigned B4SOIdNodePrimeSet  :1;
    unsigned B4SOIsNodePrimeSet  :1;
    unsigned B4SOIsaGiven :1; /* v4.0 for stress */
    unsigned B4SOIsbGiven :1; /* v4.0 for stress */
    unsigned B4SOIsdGiven :1; /* v4.0 for stress */
    unsigned B4SOIrbdbGiven :1; /* v4.0 for rbody network */
    unsigned B4SOIrbsbGiven :1; /* v4.0 for rbody network */

    unsigned B4SOIicVBSGiven :1;
    unsigned B4SOIicVDSGiven :1;
    unsigned B4SOIicVGSGiven :1;
    unsigned B4SOIicVESGiven :1;
    unsigned B4SOIicVPSGiven :1;
    unsigned B4SOIbjtoffGiven :1;
    unsigned B4SOIdebugModGiven :1;
    unsigned B4SOIrth0Given :1;
    unsigned B4SOIcth0Given :1;
    unsigned B4SOIbodySquaresGiven :1;
    unsigned B4SOIfrbodyGiven:  1;

/* v2.0 release */
    unsigned B4SOInbcGiven :1;
    unsigned B4SOInsegGiven :1;
    unsigned B4SOIpdbcpGiven :1;
    unsigned B4SOIpsbcpGiven :1;
    unsigned B4SOIagbcpGiven :1;
        unsigned B4SOIagbcp2Given :1;       /* v4.1 improvement on BC */
    unsigned B4SOIagbcpdGiven :1;       /* v4.0 */
    unsigned B4SOIaebcpGiven :1;
    unsigned B4SOIvbsusrGiven :1;
    unsigned B4SOItnodeoutGiven :1;
    unsigned B4SOIoffGiven :1;

    unsigned B4SOIrgateModGiven :1; /* v3.1 */
    unsigned B4SOIsoiModGiven :1;   /* v3.2 */

/* v4.0 added */
    unsigned B4SOInfGiven :1;
    unsigned B4SOIrbodyModGiven :1;
    unsigned B4SOIdelvtoGiven :1;
/* v4.0 added end */

/* v4.0 */
    double *B4SOIDBdpPtr;
    double *B4SOIDBdbPtr;
    double *B4SOIDBbPtr;
    double *B4SOIDPdbPtr;

    double *B4SOISBspPtr;
    double *B4SOISBbPtr;
    double *B4SOISBsbPtr;
    double *B4SOISPsbPtr;

    double *B4SOIBsbPtr;
    double *B4SOIBdbPtr;

    double *B4SOIDgPtr;         /* v4.0 for rdsMod =1 */
    double *B4SOIDspPtr;        /* v4.0 for rdsMod =1 */
    double *B4SOIDbPtr;         /* v4.0 for rdsMod =1 */
    double *B4SOISdpPtr;        /* v4.0 for rdsMod =1 */
    double *B4SOISgPtr;         /* v4.0 for rdsMod =1 */
    double *B4SOISbPtr;         /* v4.0 for rdsMod =1 */
/* v4.0 end */

    double *B4SOIPgPtr;         /* v4.1 for Ig_agbcp2 */
    double *B4SOIGpPtr;         /* v4.1 for Ig_agbcp2 */

/* v3.1  added for RF */
    double *B4SOIGgmPtr;
    double *B4SOIGgePtr;

    double *B4SOIGMdpPtr;
    double *B4SOIGMgPtr;
    double *B4SOIGMspPtr;
    double *B4SOIGMgmPtr;
    double *B4SOIGMgePtr;
    double *B4SOIGMePtr;
    double *B4SOIGMbPtr;

    double *B4SOISPgmPtr;
    double *B4SOIDPgmPtr;
    double *B4SOIEgmPtr;

    double *B4SOIGEdpPtr;
    double *B4SOIGEgPtr;
    double *B4SOIGEgmPtr;
    double *B4SOIGEgePtr;
    double *B4SOIGEspPtr;
    double *B4SOIGEbPtr;
/* v3.1 added for RF end */

    double *B4SOIGePtr;
    double *B4SOIDPePtr;
    double *B4SOISPePtr;

    double *B4SOIEePtr;
    double *B4SOIEbPtr;
    double *B4SOIBePtr;
    double *B4SOIEgPtr;
    double *B4SOIEdpPtr;
    double *B4SOIEspPtr;
    double *B4SOITemptempPtr;
    double *B4SOITempdpPtr;
    double *B4SOITempspPtr;
    double *B4SOITempgPtr;
    double *B4SOITempbPtr;
    double *B4SOITempePtr;   /* v3.0 */
    double *B4SOIGtempPtr;
    double *B4SOIDPtempPtr;
    double *B4SOISPtempPtr;
    double *B4SOIEtempPtr;
    double *B4SOIBtempPtr;
    double *B4SOIPtempPtr;
    double *B4SOIBpPtr;
    double *B4SOIPbPtr;
    double *B4SOIPpPtr;
    double *B4SOIDdPtr;
    double *B4SOIGgPtr;
    double *B4SOISsPtr;
    double *B4SOIBbPtr;
    double *B4SOIDPdpPtr;
    double *B4SOISPspPtr;
    double *B4SOIDdpPtr;
    double *B4SOIGbPtr;
    double *B4SOIGdpPtr;
    double *B4SOIGspPtr;
    double *B4SOISspPtr;
    double *B4SOIBdpPtr;
    double *B4SOIBspPtr;
    double *B4SOIDPspPtr;
    double *B4SOIDPdPtr;
    double *B4SOIBgPtr;
    double *B4SOIDPgPtr;
    double *B4SOISPgPtr;
    double *B4SOISPsPtr;
    double *B4SOIDPbPtr;
    double *B4SOISPbPtr;
    double *B4SOISPdpPtr;

    double *B4SOIVbsPtr;
    /* Debug */
    double *B4SOIIdsPtr;
    double *B4SOIIcPtr;
    double *B4SOIIbsPtr;
    double *B4SOIIbdPtr;
    double *B4SOIIiiPtr;
    double *B4SOIIgPtr;
    double *B4SOIGiggPtr;
    double *B4SOIGigdPtr;
    double *B4SOIGigbPtr;
    double *B4SOIIgidlPtr;
    double *B4SOIItunPtr;
    double *B4SOIIbpPtr;
    double *B4SOICbbPtr;
    double *B4SOICbdPtr;
    double *B4SOICbgPtr;
    double *B4SOIqbPtr;
    double *B4SOIQbfPtr;
    double *B4SOIQjsPtr;
    double *B4SOIQjdPtr;

#ifdef USE_OMP
    /* per instance storage of results, to update matrix at a later stge */
    int B4SOINode_sh;
    double B4SOINode_1;
    double B4SOINode_2;
    double B4SOINode_3;
    double B4SOINode_4;
    double B4SOINode_5;
    double B4SOINode_6;
    double B4SOINode_7;
    double B4SOINode_8;
    double B4SOINode_9;
    double B4SOINode_10;
    double B4SOINode_11;
    double B4SOINode_12;
    double B4SOINode_13;
    double B4SOINode_14;
    double B4SOINode_15;

    double B4SOI_1;
    double B4SOI_2;
    double B4SOI_3;
    double B4SOI_4;
    double B4SOI_5;
    double B4SOI_6;
    double B4SOI_7;
    double B4SOI_8;
    double B4SOI_9;
    double B4SOI_10;
    double B4SOI_11;
    double B4SOI_12;
    double B4SOI_13;
    double B4SOI_14;
    double B4SOI_15;
    double B4SOI_16;
    double B4SOI_17;
    double B4SOI_18;
    double B4SOI_19;
    double B4SOI_20;
    double B4SOI_21;
    double B4SOI_22;
    double B4SOI_23;
    double B4SOI_24;
    double B4SOI_25;
    double B4SOI_26;
    double B4SOI_27;
    double B4SOI_28;
    double B4SOI_29;
    double B4SOI_30;
    double B4SOI_31;
    double B4SOI_32;
    double B4SOI_33;
    double B4SOI_34;
    double B4SOI_35;
    double B4SOI_36;
    double B4SOI_37;
    double B4SOI_38;
    double B4SOI_39;
    double B4SOI_40;
    double B4SOI_41;
    double B4SOI_42;
    double B4SOI_43;
    double B4SOI_44;
    double B4SOI_45;
    double B4SOI_46;
    double B4SOI_47;
    double B4SOI_48;
    double B4SOI_49;
    double B4SOI_50;
    double B4SOI_51;
    double B4SOI_52;
    double B4SOI_53;
    double B4SOI_54;
    double B4SOI_55;
    double B4SOI_56;
    double B4SOI_57;
    double B4SOI_58;
    double B4SOI_59;
    double B4SOI_60;
    double B4SOI_61;
    double B4SOI_62;
    double B4SOI_63;
    double B4SOI_64;
    double B4SOI_65;
    double B4SOI_66;
    double B4SOI_67;
    double B4SOI_68;
    double B4SOI_69;
    double B4SOI_70;
    double B4SOI_71;
    double B4SOI_72;
    double B4SOI_73;
    double B4SOI_74;
    double B4SOI_75;
    double B4SOI_76;
    double B4SOI_77;
    double B4SOI_78;
    double B4SOI_79;
    double B4SOI_80;
    double B4SOI_81;
    double B4SOI_82;
    double B4SOI_83;
    double B4SOI_84;
    double B4SOI_85;
    double B4SOI_86;
    double B4SOI_87;
    double B4SOI_88;
    double B4SOI_89;
    double B4SOI_90;
    double B4SOI_91;
    double B4SOI_92;
    double B4SOI_93;
    double B4SOI_94;
    double B4SOI_95;
    double B4SOI_96;
    double B4SOI_97;
    double B4SOI_98;
    double B4SOI_99;
    double B4SOI_100;
    double B4SOI_101;
    double B4SOI_102;
#endif

#define B4SOIvbd B4SOIstates+ 0
#define B4SOIvbs B4SOIstates+ 1
#define B4SOIvgs B4SOIstates+ 2
#define B4SOIvds B4SOIstates+ 3
#define B4SOIves B4SOIstates+ 4
#define B4SOIvps B4SOIstates+ 5

#define B4SOIvg B4SOIstates+ 6
#define B4SOIvd B4SOIstates+ 7
#define B4SOIvs B4SOIstates+ 8
#define B4SOIvp B4SOIstates+ 9
#define B4SOIve B4SOIstates+ 10
#define B4SOIdeltemp B4SOIstates+ 11

#define B4SOIqb B4SOIstates+ 12
#define B4SOIcqb B4SOIstates+ 13
#define B4SOIqg B4SOIstates+ 14
#define B4SOIcqg B4SOIstates+ 15
#define B4SOIqd B4SOIstates+ 16
#define B4SOIcqd B4SOIstates+ 17
#define B4SOIqe B4SOIstates+ 18
#define B4SOIcqe B4SOIstates+ 19

#define B4SOIqbs  B4SOIstates+ 20
#define B4SOIcqbs  B4SOIstates+ 21 /* v4.0 */
#define B4SOIqbd  B4SOIstates+ 22
#define B4SOIcqbd  B4SOIstates+ 23 /* v4.0 */
#define B4SOIqbe  B4SOIstates+ 24

#define B4SOIqth B4SOIstates+ 25
#define B4SOIcqth B4SOIstates+ 26

/* v3.1 added or changed for RF */
#define B4SOIvges B4SOIstates+ 27
#define B4SOIvgms B4SOIstates+ 28
#define B4SOIvgge B4SOIstates+ 29
#define B4SOIvggm B4SOIstates+ 30
/* #define B4SOIcqgmid B4SOIstates+ 31 */ /* Bug # 29 */
/* #define B4SOIqgmid B4SOIstates + 32 */ /* Bug # 29 */
#define B4SOIqgmid B4SOIstates+ 31              /* Bug Fix # 29 Jul09*/
#define B4SOIcqgmid B4SOIstates + 32    /* Bug Fix # 29 Jul09*/
/* v3.1 added or changed for RF end */

/* v4.0 */
#define B4SOIvdbs B4SOIstates + 33
#define B4SOIvdbd B4SOIstates + 34
#define B4SOIvsbs B4SOIstates + 35
#define B4SOIvses B4SOIstates + 36
#define B4SOIvdes B4SOIstates + 37
/* v4.0 end */

#define B4SOInumStates 38

/* indices to the array of B4SOI NOISE SOURCES */

#define B4SOIRDNOIZ       0
#define B4SOIRSNOIZ       1
#define B4SOIRGNOIZ       2
#define B4SOIIDNOIZ       3
#define B4SOIFLNOIZ       4
#define B4SOIFB_IBSNOIZ   5     /* v4.0 */
#define B4SOIFB_IBDNOIZ   6     /* v4.0 */
#define B4SOIIGSNOIZ      7
#define B4SOIIGDNOIZ      8
#define B4SOIIGBNOIZ      9
#define B4SOIRBSBNOIZ     10    /* v4.0 */
#define B4SOIRBDBNOIZ     11    /* v4.0 */
#define B4SOIRBODYNOIZ    12    /* v4.0 */
#define B4SOITOTNOIZ      13    /* v4.0 */

#define B4SOINSRCS        14     /* Number of MOSFET(3) noise sources v3.2 */

#ifndef NONOISE
    double B4SOInVar[NSTATVARS][B4SOINSRCS];
#else /* NONOISE */
        double **B4SOInVar;
#endif /* NONOISE */

} B4SOIinstance ;

struct b4soiSizeDependParam
{
    double Width;
    double Length;
    double Rth0;
    double Cth0;

    double B4SOIcdsc;
    double B4SOIcdscb;
    double B4SOIcdscd;
    double B4SOIcit;
    double B4SOInfactor;
    double B4SOIvsat;
    double B4SOIat;
    double B4SOIa0;
    double B4SOIags;
    double B4SOIa1;
    double B4SOIa2;
    double B4SOIketa;
    double B4SOInpeak;
    double B4SOInsub;
    double B4SOIngate;
    double B4SOInsd;
    double B4SOIgamma1;
    double B4SOIgamma2;
    double B4SOIvbx;
    double B4SOIvbi;
    double B4SOIvbm;
    double B4SOIvbsc;
    double B4SOIxt;
    double B4SOIphi;
    double B4SOIlitl;
    double B4SOIk1;
    double B4SOIkt1;
    double B4SOIkt1l;
    double B4SOIkt2;
    double B4SOIk2;
    double B4SOIk3;
    double B4SOIk3b;
    double B4SOIw0;
    double B4SOIlpe0;
    double B4SOIdvt0;
    double B4SOIdvt1;
    double B4SOIdvt2;
    double B4SOIdvt0w;
    double B4SOIdvt1w;
    double B4SOIdvt2w;
    double B4SOIdrout;
    double B4SOIdsub;
    double B4SOIvth0;
    double B4SOIua;
    double B4SOIua1;
    double B4SOIub;
    double B4SOIub1;
    double B4SOIuc;
    double B4SOIuc1;
    double B4SOIu0;
    double B4SOIute;
                 /*4.1 high k mobility*/
        double B4SOIud;
    double B4SOIud1;
    double B4SOIeu;
        double B4SOIucs;
    double B4SOIucste;



    double B4SOIvoff;
    double B4SOIvfb;
    double B4SOIuatemp;
    double B4SOIubtemp;
    double B4SOIuctemp;
    double B4SOIrbody;
    double B4SOIrth;
    double B4SOIcth;
    double B4SOIrds0denom;
    double B4SOIvfbb;
    double B4SOIjbjts;  /* v4.0 */
    double B4SOIjbjtd;  /* v4.0 */
    double B4SOIjdifs;  /* v4.0 */
    double B4SOIjdifd;  /* v4.0 */
    double B4SOIjrecs;  /* v4.0 */
    double B4SOIjrecd;  /* v4.0 */
    double B4SOIjtuns;  /* v4.0 */
    double B4SOIjtund;  /* v4.0 */
    double B4SOIrdw;    /* v4.0 for rdsMod = 1 */
    double B4SOIrsw;    /* v4.0 for rdsMod = 1 */
    double B4SOIrdwmin; /* v4.0 for rdsMod = 1 */
    double B4SOIrswmin; /* v4.0 for rdsMod = 1 */
    double B4SOIrd0;    /* v4.0 for rdsMod = 1 */
    double B4SOIrs0;    /* v4.0 for rdsMod = 1 */
    double B4SOIsdt1;
    double B4SOIst2;
    double B4SOIst3;
    double B4SOIdt2;
    double B4SOIdt3;
    double B4SOIdelta;
    double B4SOIrdsw;
    double B4SOIrds0;
    double B4SOIprwg;
    double B4SOIprwb;
    double B4SOIprt;
    double B4SOIeta0;
    double B4SOIetab;
    double B4SOIpclm;
    double B4SOIpdibl1;
    double B4SOIpdibl2;
    double B4SOIpdiblb;
    double B4SOIpvag;
    double B4SOIwr;
    double B4SOIdwg;
    double B4SOIdwb;
    double B4SOIb0;
    double B4SOIb1;
    double B4SOIalpha0;
    double B4SOIbeta0;

/* v3.0 */
    double B4SOIaigc;
    double B4SOIbigc;
    double B4SOIcigc;
    double B4SOIaigsd;
    double B4SOIbigsd;
    double B4SOIcigsd;
    double B4SOInigc;
    double B4SOIpigcd;
    double B4SOIpoxedge;
    double B4SOIdlcig;

/* v3.1 added for RF */
    double B4SOIxrcrg1;
    double B4SOIxrcrg2;
/* v3.1 added for RF end */

/* v4.0 */
    /* added for stress effect */
    double B4SOIku0;
    double B4SOIkvth0;
    double B4SOIku0temp;
    double B4SOIrho_ref;
    double B4SOIinv_od_ref;
    /* added for stress effect end */
    double NFinger;
/* v4.0 end */


    /* CV model */
    double B4SOIcgsl;
    double B4SOIcgdl;
    double B4SOIckappa;
    double B4SOIcf;
    double B4SOIclc;
    double B4SOIcle;


/* Added for binning - START0 */
/* v3.1 */
    double B4SOIxj;
    double B4SOIalphaGB1;
    double B4SOIbetaGB1;
    double B4SOIalphaGB2;
    double B4SOIbetaGB2;
    double B4SOIaigbcp2;  /* 4.1 */
    double B4SOIbigbcp2;  /* 4.1 */
    double B4SOIcigbcp2;  /* 4.1 */
    double B4SOIndif;
    double B4SOIntrecf;
    double B4SOIntrecr;
    double B4SOIxbjt;
    double B4SOIxdif;
    double B4SOIxrec;
    double B4SOIxtun;
    double B4SOIxdifd;
    double B4SOIxrecd;
    double B4SOIxtund;

    double B4SOIkb1;
    double B4SOIk1w1;
    double B4SOIk1w2;
    double B4SOIketas;
    double B4SOIfbjtii;
        /*4.1 Iii model*/
    double B4SOIebjtii;
        double B4SOIcbjtii;
        double B4SOIvbci;
        double B4SOIabjtii;
        double B4SOImbjtii;

    double B4SOIbeta1;
    double B4SOIbeta2;
    double B4SOIvdsatii0;
    double B4SOIlii;
    double B4SOIesatii;
    double B4SOIsii0;
    double B4SOIsii1;
    double B4SOIsii2;
    double B4SOIsiid;
    double B4SOIagidl;
    double B4SOIbgidl;
    double B4SOIcgidl; /* v4.0 */
    double B4SOIegidl;
        double B4SOIrgidl;
        double B4SOIkgidl;
        double B4SOIfgidl;
        double B4SOIagisl;
    double B4SOIbgisl;
    double B4SOIcgisl; /* v4.0 */
    double B4SOIegisl;
        double B4SOIrgisl;
        double B4SOIkgisl;
        double B4SOIfgisl;


    double B4SOIntun;   /* v4.0 */
    double B4SOIntund;  /* v4.0 */
    double B4SOIndiode; /* v4.0 */
    double B4SOIndioded; /* v4.0 */
    double B4SOInrecf0; /* v4.0 */
    double B4SOInrecf0d; /* v4.0 */
    double B4SOInrecr0; /* v4.0 */
    double B4SOInrecr0d; /* v4.0 */
    double B4SOIisbjt;
    double B4SOIidbjt;  /* v4.0 */
    double B4SOIisdif;
    double B4SOIiddif;  /* v4.0 */
    double B4SOIisrec;
    double B4SOIidrec;  /* v4.0 */
    double B4SOIistun;
    double B4SOIidtun;  /* v4.0 */
    double B4SOIvrec0; /* v4.0 */
    double B4SOIvrec0d; /* v4.0 */
    double B4SOIvtun0;  /* v4.0 */
    double B4SOIvtun0d; /* v4.0 */
    double B4SOInbjt;
    double B4SOIlbjt0;
    double B4SOIvabjt;
    double B4SOIaely;
    double B4SOIvsdfb;
    double B4SOIvsdth;
    double B4SOIdelvt;
/* Added by binning - END0 */

/* Pre-calculated constants */

    double B4SOIdw;
    double B4SOIdl;
    double B4SOIleff;
    double B4SOIweff;

    double B4SOIdwc;
    double B4SOIdlc;
    double B4SOIleffCV;
    double B4SOIweffCV;

    double B4SOIabulkCVfactor;
    double B4SOIcgso;
    double B4SOIcgdo;
    double B4SOIcgeo;

    double B4SOIu0temp;
    double B4SOIvsattemp;
    double B4SOIsqrtPhi;
    double B4SOIphis3;
    double B4SOIXdep0;
    double B4SOIsqrtXdep0;
    double B4SOItheta0vb0;
    double B4SOIthetaRout;


/* v3.2 */
    double B4SOIqsi;

/* v2.2 release */
    double B4SOIoxideRatio;


/* v2.0 release */
    double B4SOIk1eff;
    double B4SOIwdios;
    double B4SOIwdiod;
    double B4SOIwdiodCV;
    double B4SOIwdiosCV;
    double B4SOIarfabjt;
    double B4SOIlratio;
    double B4SOIlratiodif;
    double B4SOIvearly;
    double B4SOIahli;   /* v4.0 */
    double B4SOIahlid;  /* v4.0 */
    double B4SOIahli0s; /* v4.0 */
    double B4SOIahli0d; /* v4.0 */
    double B4SOIvfbzb;
    double B4SOIldeb;
    double B4SOIacde;
    double B4SOImoin;
    double B4SOInoff; /* v3.2 */
    double B4SOIleffCVb;
    double B4SOIleffCVbg;

 /*   double B4SOIcof1;
    double B4SOIcof2;
    double B4SOIcof3;
    double B4SOIcof4; */ /* v4.2 nover used in the code */
    double B4SOIcdep0;
/* v3.0 */
    double B4SOIToxRatio;
    double B4SOIAechvb;
    double B4SOIBechvb;
    double B4SOIToxRatioEdge;
    double B4SOIAechvbEdges;   /* LFW */
    double B4SOIAechvbEdged;   /* LFW */
    double B4SOIBechvbEdge;
    double B4SOIvfbsd;
/* v4.0 */
    double B4SOIk1ox;   /* v4.0 for Vth */
    double B4SOIk2ox;   /* v4.0 for Vth */
    double B4SOIlpeb;   /* v4.0 for Vth */
    double B4SOIdvtp0;  /* v4.0 for Vth */
    double B4SOIdvtp1;  /* v4.0 for Vth */
    double B4SOIdvtp2;  /* v4.1 for Vth */
    double B4SOIdvtp3;  /* v4.1 for Vth */
    double B4SOIdvtp4;  /* v4.1 for Vth */
    double B4SOIminv;   /* v4.0 for Vgsteff */
    double B4SOImstar;  /* v4.0 for Vgsteff */
    double B4SOIfprout; /* v4.0 for DITS in Id */
    double B4SOIpdits;  /* v4.0 for DITS in Id */
    double B4SOIpditsd;  /* v4.0 for DITS in Id */

    /*4.1*/
    double B4SOImstarcv;
    double B4SOIminvcv;
    double B4SOIvoffcv;
    double B4SOIdvtp2factor;

    struct b4soiSizeDependParam  *pNext;
};


typedef struct sB4SOImodel
{

    struct GENmodel gen;

#define B4SOImodType gen.GENmodType
#define B4SOInextModel(inst) ((struct sB4SOImodel *)((inst)->gen.GENnextModel))
#define B4SOIinstances(inst) ((B4SOIinstance *)((inst)->gen.GENinstances))
#define B4SOImodName gen.GENmodName

    int B4SOItype;

    int    B4SOImobMod;
    int    B4SOIcapMod;
    int    B4SOIfnoiMod;        /* v3.2 */
    int    B4SOItnoiMod;        /* v3.2 */
    int    B4SOIshMod;
    int    B4SOIbinUnit;
    int    B4SOIparamChk;
        int    B4SOImtrlMod;  /*4.1*/
        int    B4SOIvgstcvMod;
        int    B4SOIgidlMod;
        int    B4SOIiiiMod;

    double B4SOIversion;

        double B4SOIeot;  /*4.1*/
        double B4SOIepsrox;
        double B4SOIepsrsub;
        double B4SOItoxp;
        double B4SOIleffeot;
        double B4SOIweffeot;
        double B4SOIvddeot;
        double B4SOItempeot;
        double B4SOIados;
        double B4SOIbdos;
        double B4SOIepsrgate;
        double B4SOIni0sub;
        double B4SOIbg0sub;
        double B4SOItbgasub;
        double B4SOItbgbsub;
        double B4SOIphig;
        double B4SOIeasub;
    double B4SOItvbci;

    double B4SOItox;
    double B4SOItoxm; /* v3.2 */
    double B4SOIdtoxcv; /* v2.2.3 */
    double B4SOIcdsc;
    double B4SOIcdscb;
    double B4SOIcdscd;
    double B4SOIcit;
    double B4SOInfactor;
    double B4SOIvsat;
    double B4SOIat;
    double B4SOIa0;
    double B4SOIags;
    double B4SOIa1;
    double B4SOIa2;
    double B4SOIketa;
    double B4SOInsub;
    double B4SOInpeak;
    double B4SOIngate;
    double B4SOInsd;
    double B4SOIlnsd;
    double B4SOIwnsd;
    double B4SOIpnsd;
    double B4SOIgamma1;
    double B4SOIgamma2;
    double B4SOIvbx;
    double B4SOIvbm;
    double B4SOIxt;
    double B4SOIk1;
    double B4SOIkt1;
    double B4SOIkt1l;
    double B4SOIkt2;
    double B4SOIk2;
    double B4SOIk3;
    double B4SOIk3b;
    double B4SOIw0;
    double B4SOIlpe0;
    double B4SOIdvt0;
    double B4SOIdvt1;
    double B4SOIdvt2;
    double B4SOIdvt0w;
    double B4SOIdvt1w;
    double B4SOIdvt2w;
    double B4SOIdrout;
    double B4SOIdsub;
    double B4SOIvth0;
    double B4SOIua;
    double B4SOIua1;
    double B4SOIub;
    double B4SOIub1;
    double B4SOIuc;
    double B4SOIuc1;
    double B4SOIu0;
    double B4SOIute;
                /*4.1 high k mobility*/
    double B4SOIud;
        double B4SOIlud;
        double B4SOIwud;
        double B4SOIpud;

        double B4SOIud1;
        double B4SOIlud1;
        double B4SOIwud1;
        double B4SOIpud1;

    double B4SOIeu;
        double B4SOIleu;
        double B4SOIweu;
        double B4SOIpeu;

        double B4SOIucs;
        double B4SOIlucs;
        double B4SOIwucs;
        double B4SOIpucs;

        double B4SOIucste;
        double B4SOIlucste;
        double B4SOIwucste;
        double B4SOIpucste;

    double B4SOIvoff;
    double B4SOIdelta;
    double B4SOIrdsw;
    double B4SOIrdw;    /* v4.0 for rdsMod = 1 */
    double B4SOIrsw;    /* v4.0 for rdsMod = 1 */
    double B4SOIrdwmin; /* v4.0 for rdsMod = 1 */
    double B4SOIrswmin; /* v4.0 for rdsMod = 1 */
    double B4SOIprwg;
    double B4SOIprwb;
    double B4SOIprt;
    double B4SOIeta0;
    double B4SOIetab;
    double B4SOIpclm;
    double B4SOIpdibl1;
    double B4SOIpdibl2;
    double B4SOIpdiblb;
    double B4SOIpvag;
    double B4SOIwr;
    double B4SOIdwg;
    double B4SOIdwb;
    double B4SOIb0;
    double B4SOIb1;
    double B4SOIalpha0;
    double B4SOItbox;
    double B4SOItsi;
        double B4SOIetsi;
    double B4SOIxj;
    double B4SOIkb1;
    double B4SOIrth0;
    double B4SOIcth0;
    double B4SOIegidl;
    double B4SOIcfrcoeff; /* v4.4 */ 
    double B4SOIagidl;
    double B4SOIbgidl;
    double B4SOIcgidl;   /* v4.0 */
        double B4SOIrgidl;
        double B4SOIkgidl;
        double B4SOIfgidl;
        double B4SOIegisl;
    double B4SOIagisl;
    double B4SOIbgisl;
    double B4SOIcgisl;
        double B4SOIrgisl;
        double B4SOIkgisl;
        double B4SOIfgisl;
    double B4SOIndiode; /* v4.0 */
    double B4SOIndioded; /* v4.0 */
    double B4SOIistun;
    double B4SOIidtun;   /* v4.0 */
    double B4SOIxbjt;
    double B4SOIxdif;
    double B4SOIxrec;
    double B4SOIxtun;
    double B4SOIxdifd;
    double B4SOIxrecd;
    double B4SOIxtund;


/* v3.0 */
    int B4SOIsoiMod; /* v3.2 bug fix */
    double B4SOIvbs0pd; /* v3.2 */
    double B4SOIvbs0fd; /* v3.2 */
    double B4SOIvbsa;
    double B4SOInofffd;
    double B4SOIvofffd;
    double B4SOIk1b;
    double B4SOIk2b;
    double B4SOIdk2b;
    double B4SOIdvbd0;
    double B4SOIdvbd1;
    double B4SOImoinFD;

/* v3.0 */
    int B4SOIigbMod;
    int B4SOIigcMod;
    double B4SOIaigc;
    double B4SOIbigc;
    double B4SOIcigc;
    double B4SOIaigsd;
    double B4SOIbigsd;
    double B4SOIcigsd;
    double B4SOInigc;
    double B4SOIpigcd;
    double B4SOIpoxedge;
    double B4SOIdlcig;

/* v3.1 added for RF */
    int    B4SOIrgateMod;
    double B4SOIxrcrg1;
    double B4SOIxrcrg2;
    double B4SOIrshg;
    double B4SOIngcon;
    double B4SOIxgw;
    double B4SOIxgl;
    /* v3.1 added for RF end */

    /* v3.2 for noise */
    double B4SOItnoia;
    double B4SOItnoib;
    double B4SOIrnoia;
    double B4SOIrnoib;
    double B4SOIntnoi;
    /* v3.2 for noise end */

    /* v4.0 */
    int B4SOIrbodyMod;
    double B4SOIrbdb;
    double B4SOIrbsb;
    double B4SOIgbmin;
    double B4SOIfrbody;

    int B4SOIrdsMod;    /* v4.0 */

    /* v4.0 end */
    /* v4.1 */
    int B4SOIfdMod;
    double B4SOIvfb;
    double B4SOIvsce;
    double B4SOIcdsbs;
    double B4SOIminvcv;
    double B4SOIlminvcv;
    double B4SOIwminvcv;
    double B4SOIpminvcv;
        double B4SOIvoffcv;
    double B4SOIlvoffcv;
    double B4SOIwvoffcv;
    double B4SOIpvoffcv;


    /* added end */

/* v2.2 release */
    double B4SOIwth0;
    double B4SOIrhalo;
    double B4SOIntox;
    double B4SOItoxref;
    double B4SOIebg;
    double B4SOIvevb;
    double B4SOIalphaGB1;
    double B4SOIbetaGB1;
    double B4SOIvgb1;
    double B4SOIvecb;
    double B4SOIalphaGB2;
    double B4SOIbetaGB2;
    double B4SOIvgb2;
    double B4SOIaigbcp2;  /* 4.1 */
    double B4SOIbigbcp2;  /* 4.1 */
    double B4SOIcigbcp2;  /* 4.1 */
    double B4SOItoxqm;
    double B4SOIvoxh;
    double B4SOIdeltavox;


/* v2.0 release */
    double B4SOIk1w1;
    double B4SOIk1w2;
    double B4SOIketas;
    double B4SOIdwbc;
    double B4SOIbeta0;
    double B4SOIbeta1;
    double B4SOIbeta2;
    double B4SOIvdsatii0;
    double B4SOItii;
    double B4SOIlii;
    double B4SOIsii0;
    double B4SOIsii1;
    double B4SOIsii2;
    double B4SOIsiid;
    double B4SOIfbjtii;
        /*4.1 Iii model*/
    double B4SOIebjtii;
        double B4SOIcbjtii;
        double B4SOIvbci;
        double B4SOIabjtii;
        double B4SOImbjtii;

    double B4SOIesatii;
    double B4SOIntun;           /* v4.0 */
    double B4SOIntund;          /* v4.0 */
    double B4SOInrecf0; /* v4.0 */
    double B4SOInrecf0d;        /* v4.0 */
    double B4SOInrecr0; /* v4.0 */
    double B4SOInrecr0d;        /* v4.0 */
    double B4SOIisbjt;
    double B4SOIidbjt;  /* v4.0 */
    double B4SOIisdif;
    double B4SOIiddif;  /* v4.0 */
    double B4SOIisrec;
    double B4SOIidrec;  /* v4.0 */
    double B4SOIln;
    double B4SOIvrec0;  /* v4.0 */
    double B4SOIvrec0d; /* v4.0 */
    double B4SOIvtun0;  /* v4.0 */
    double B4SOIvtun0d; /* v4.0 */
    double B4SOInbjt;
    double B4SOIlbjt0;
    double B4SOIldif0;
    double B4SOIvabjt;
    double B4SOIaely;
    double B4SOIahli;   /* v4.0 */
    double B4SOIahlid;  /* v4.0 */
    double B4SOIrbody;
    double B4SOIrbsh;
    double B4SOItt;
    double B4SOIndif;
    double B4SOIvsdfb;
    double B4SOIvsdth;
    double B4SOIcsdmin;
    double B4SOIasd;
    double B4SOIntrecf;
    double B4SOIntrecr;
    double B4SOIdlcb;
    double B4SOIfbody;
    double B4SOItcjswg;
    double B4SOItpbswg;
    double B4SOItcjswgd;
    double B4SOItpbswgd;
    double B4SOIacde;
    double B4SOImoin;
    double B4SOInoff; /* v3.2 */
    double B4SOIdelvt;
    double B4SOIdlbg;

    /* CV model */
    double B4SOIcgsl;
    double B4SOIcgdl;
    double B4SOIckappa;
    double B4SOIcf;
    double B4SOIclc;
    double B4SOIcle;
    double B4SOIdwc;
    double B4SOIdlc;


    double B4SOItnom;
    double B4SOIcgso;
    double B4SOIcgdo;
    double B4SOIcgeo;

    double B4SOIxpart;
  /*  double B4SOIcFringOut;
    double B4SOIcFringMax; */  /* v4.2 never used in the code */

    double B4SOIsheetResistance;
    double B4SOIbodyJctGateSideSGradingCoeff;   /* v4.0 */
    double B4SOIbodyJctGateSideDGradingCoeff;   /* v4.0 */
    double B4SOIGatesidewallJctSPotential;      /* v4.0 */
    double B4SOIGatesidewallJctDPotential;      /* v4.0 */
    double B4SOIunitLengthGateSidewallJctCapS;  /* v4.0 */
    double B4SOIunitLengthGateSidewallJctCapD;  /* v4.0 */
    double B4SOIcsdesw;

    double B4SOILint;
    double B4SOILl;
    double B4SOILlc; /* v2.2.3 */
    double B4SOILln;
    double B4SOILw;
    double B4SOILwc; /* v2.2.3 */
    double B4SOILwn;
    double B4SOILwl;
    double B4SOILwlc; /* v2.2.3 */
    double B4SOILmin;
    double B4SOILmax;

    double B4SOIWint;
    double B4SOIWl;
    double B4SOIWlc; /* v2.2.3 */
    double B4SOIWln;
    double B4SOIWw;
    double B4SOIWwc; /* v2.2.3 */
    double B4SOIWwn;
    double B4SOIWwl;
    double B4SOIWwlc; /* v2.2.3 */
    double B4SOIWmin;
    double B4SOIWmax;

/* Added for binning - START1 */
    /* Length Dependence */
/* v3.1 */
    double B4SOIlxj;
    double B4SOIlalphaGB1;
    double B4SOIlbetaGB1;
    double B4SOIlalphaGB2;
    double B4SOIlbetaGB2;
    double B4SOIlaigbcp2;  /* 4.1 */
    double B4SOIlbigbcp2;  /* 4.1 */
    double B4SOIlcigbcp2;  /* 4.1 */
    double B4SOIlndif;
    double B4SOIlntrecf;
    double B4SOIlntrecr;
    double B4SOIlxbjt;
    double B4SOIlxdif;
    double B4SOIlxrec;
    double B4SOIlxtun;
    double B4SOIlxdifd;
    double B4SOIlxrecd;
    double B4SOIlxtund;
    double B4SOIlcgsl;
    double B4SOIlcgdl;
    double B4SOIlckappa;
    double B4SOIlua1;
    double B4SOIlub1;
    double B4SOIluc1;
    double B4SOIlute;
    double B4SOIlkt1;
    double B4SOIlkt1l;
    double B4SOIlkt2;
    double B4SOIlat;
    double B4SOIlprt;

/* v3.0 */
    double B4SOIlaigc;
    double B4SOIlbigc;
    double B4SOIlcigc;
    double B4SOIlaigsd;
    double B4SOIlbigsd;
    double B4SOIlcigsd;
    double B4SOIlnigc;
    double B4SOIlpigcd;
    double B4SOIlpoxedge;

    double B4SOIlnpeak;
    double B4SOIlnsub;
    double B4SOIlngate;
    double B4SOIlvth0;
    double B4SOIlvfb;  /* v4.1 */
    double B4SOIlk1;
    double B4SOIlk1w1;
    double B4SOIlk1w2;
    double B4SOIlk2;
    double B4SOIlk3;
    double B4SOIlk3b;
    double B4SOIlkb1;
    double B4SOIlw0;
    double B4SOIllpe0;
    double B4SOIldvt0;
    double B4SOIldvt1;
    double B4SOIldvt2;
    double B4SOIldvt0w;
    double B4SOIldvt1w;
    double B4SOIldvt2w;
    double B4SOIlu0;
    double B4SOIlua;
    double B4SOIlub;
    double B4SOIluc;
    double B4SOIlvsat;
    double B4SOIla0;
    double B4SOIlags;
    double B4SOIlb0;
    double B4SOIlb1;
    double B4SOIlketa;
    double B4SOIlketas;
    double B4SOIla1;
    double B4SOIla2;
    double B4SOIlrdsw;
    double B4SOIlrdw;  /* v4.0 for rdsMod = 1 */
    double B4SOIlrsw;  /* v4.0 for rdsMod = 1 */
    double B4SOIlprwb;
    double B4SOIlprwg;
    double B4SOIlwr;
    double B4SOIlnfactor;
    double B4SOIldwg;
    double B4SOIldwb;
    double B4SOIlvoff;
    double B4SOIleta0;
    double B4SOIletab;
    double B4SOIldsub;
    double B4SOIlcit;
    double B4SOIlcdsc;
    double B4SOIlcdscb;
    double B4SOIlcdscd;
    double B4SOIlpclm;
    double B4SOIlpdibl1;
    double B4SOIlpdibl2;
    double B4SOIlpdiblb;
    double B4SOIldrout;
    double B4SOIlpvag;
    double B4SOIldelta;
    double B4SOIlalpha0;
    double B4SOIlfbjtii;
        /*4.1 Iii model*/
        double B4SOIlebjtii;
        double B4SOIlcbjtii;
        double B4SOIlvbci;
        double B4SOIlabjtii;
        double B4SOIlmbjtii;

    double B4SOIlbeta0;
    double B4SOIlbeta1;
    double B4SOIlbeta2;
    double B4SOIlvdsatii0;
    double B4SOIllii;
    double B4SOIlesatii;
    double B4SOIlsii0;
    double B4SOIlsii1;
    double B4SOIlsii2;
    double B4SOIlsiid;
    double B4SOIlagidl;
    double B4SOIlbgidl;
    double B4SOIlcgidl;
    double B4SOIlegidl;
        double B4SOIlrgidl;
        double B4SOIlkgidl;
        double B4SOIlfgidl;
            double B4SOIlagisl;
    double B4SOIlbgisl;
    double B4SOIlcgisl;
    double B4SOIlegisl;
        double B4SOIlrgisl;
        double B4SOIlkgisl;
        double B4SOIlfgisl;
    double B4SOIlntun;     /* v4.0 */
    double B4SOIlntund;    /* v4.0 */
    double B4SOIlndiode;  /* v4.0 */
    double B4SOIlndioded;  /* v4.0 */
    double B4SOIlnrecf0;        /* v4.0 */
    double B4SOIlnrecf0d;       /* v4.0 */
    double B4SOIlnrecr0;        /* v4.0 */
    double B4SOIlnrecr0d;       /* v4.0 */
    double B4SOIlisbjt;
    double B4SOIlidbjt; /* v4.0 */
    double B4SOIlisdif;
    double B4SOIliddif; /* v4.0 */
    double B4SOIlisrec;
    double B4SOIlidrec; /* v4.0 */
    double B4SOIlistun;
    double B4SOIlidtun; /* v4.0 */
    double B4SOIlvrec0; /* v4.0 */
    double B4SOIlvrec0d; /* v4.0 */
    double B4SOIlvtun0; /* v4.0 */
    double B4SOIlvtun0d; /* v4.0 */
    double B4SOIlnbjt;
    double B4SOIllbjt0;
    double B4SOIlvabjt;
    double B4SOIlaely;
    double B4SOIlahli;  /* v4.0 */
    double B4SOIlahlid; /* v4.0 */

/* v3.1 added for RF */
    double B4SOIlxrcrg1;
    double B4SOIlxrcrg2;
/* v3.1 added for RF end */

    /* CV model */
    double B4SOIlvsdfb;
    double B4SOIlvsdth;
    double B4SOIldelvt;
    double B4SOIlacde;
    double B4SOIlmoin;
    double B4SOIlnoff; /* v3.2 */

    /* Width Dependence */
/* v3.1 */
    double B4SOIwxj;
    double B4SOIwalphaGB1;
    double B4SOIwbetaGB1;
    double B4SOIwalphaGB2;
    double B4SOIwbetaGB2;
    double B4SOIwaigbcp2;  /* 4.1 */
    double B4SOIwbigbcp2;  /* 4.1 */
    double B4SOIwcigbcp2;  /* 4.1 */
    double B4SOIwndif;
    double B4SOIwntrecf;
    double B4SOIwntrecr;
    double B4SOIwxbjt;
    double B4SOIwxdif;
    double B4SOIwxrec;
    double B4SOIwxtun;
    double B4SOIwxdifd;
    double B4SOIwxrecd;
    double B4SOIwxtund;
    double B4SOIwcgsl;
    double B4SOIwcgdl;
    double B4SOIwckappa;
    double B4SOIwua1;
    double B4SOIwub1;
    double B4SOIwuc1;
    double B4SOIwute;
    double B4SOIwkt1;
    double B4SOIwkt1l;
    double B4SOIwkt2;
    double B4SOIwat;
    double B4SOIwprt;

/* v3.0 */
    double B4SOIwaigc;
    double B4SOIwbigc;
    double B4SOIwcigc;
    double B4SOIwaigsd;
    double B4SOIwbigsd;
    double B4SOIwcigsd;
    double B4SOIwnigc;
    double B4SOIwpigcd;
    double B4SOIwpoxedge;

    double B4SOIwnpeak;
    double B4SOIwnsub;
    double B4SOIwngate;
    double B4SOIwvth0;
        double B4SOIwvfb;   /* v4.1 */
    double B4SOIwk1;
    double B4SOIwk1w1;
    double B4SOIwk1w2;
    double B4SOIwk2;
    double B4SOIwk3;
    double B4SOIwk3b;
    double B4SOIwkb1;
    double B4SOIww0;
    double B4SOIwlpe0;
    double B4SOIwdvt0;
    double B4SOIwdvt1;
    double B4SOIwdvt2;
    double B4SOIwdvt0w;
    double B4SOIwdvt1w;
    double B4SOIwdvt2w;
    double B4SOIwu0;
    double B4SOIwua;
    double B4SOIwub;
    double B4SOIwuc;
    double B4SOIwvsat;
    double B4SOIwa0;
    double B4SOIwags;
    double B4SOIwb0;
    double B4SOIwb1;
    double B4SOIwketa;
    double B4SOIwketas;
    double B4SOIwa1;
    double B4SOIwa2;
    double B4SOIwrdsw;
    double B4SOIwrdw;  /* v4.0 for rdsMod = 1 */
    double B4SOIwrsw;  /* v4.0 for rdsMod = 1 */
    double B4SOIwprwb;
    double B4SOIwprwg;
    double B4SOIwwr;
    double B4SOIwnfactor;
    double B4SOIwdwg;
    double B4SOIwdwb;
    double B4SOIwvoff;
    double B4SOIweta0;
    double B4SOIwetab;
    double B4SOIwdsub;
    double B4SOIwcit;
    double B4SOIwcdsc;
    double B4SOIwcdscb;
    double B4SOIwcdscd;
    double B4SOIwpclm;
    double B4SOIwpdibl1;
    double B4SOIwpdibl2;
    double B4SOIwpdiblb;
    double B4SOIwdrout;
    double B4SOIwpvag;
    double B4SOIwdelta;
    double B4SOIwalpha0;
    double B4SOIwfbjtii;
        /*4.1 Iii model*/
        double B4SOIwebjtii;
        double B4SOIwcbjtii;
        double B4SOIwvbci;
        double B4SOIwabjtii;
        double B4SOIwmbjtii;

    double B4SOIwbeta0;
    double B4SOIwbeta1;
    double B4SOIwbeta2;
    double B4SOIwvdsatii0;
    double B4SOIwlii;
    double B4SOIwesatii;
    double B4SOIwsii0;
    double B4SOIwsii1;
    double B4SOIwsii2;
    double B4SOIwsiid;
    double B4SOIwagidl;
    double B4SOIwbgidl;
    double B4SOIwcgidl;
    double B4SOIwegidl;
        double B4SOIwrgidl;
        double B4SOIwkgidl;
        double B4SOIwfgidl;
            double B4SOIwagisl;
    double B4SOIwbgisl;
    double B4SOIwcgisl;
    double B4SOIwegisl;
        double B4SOIwrgisl;
        double B4SOIwkgisl;
        double B4SOIwfgisl;
    double B4SOIwntun;          /* v4.0 */
    double B4SOIwntund;         /* v4.0 */
    double B4SOIwndiode;        /* v4.0 */
    double B4SOIwndioded;       /* v4.0 */
    double B4SOIwnrecf0;        /* v4.0 */
    double B4SOIwnrecf0d;       /* v4.0 */
    double B4SOIwnrecr0;        /* v4.0 */
    double B4SOIwnrecr0d;       /* v4.0 */
    double B4SOIwisbjt;
    double B4SOIwidbjt;   /* v4.0 */
    double B4SOIwisdif;
    double B4SOIwiddif;   /* v4.0 */
    double B4SOIwisrec;
    double B4SOIwidrec;   /* v4.0 */
    double B4SOIwistun;
    double B4SOIwidtun;   /* v4.0 */
    double B4SOIwvrec0;  /* v4.0 */
    double B4SOIwvrec0d;  /* v4.0 */
    double B4SOIwvtun0;  /* v4.0 */
    double B4SOIwvtun0d;  /* v4.0 */
    double B4SOIwnbjt;
    double B4SOIwlbjt0;
    double B4SOIwvabjt;
    double B4SOIwaely;
    double B4SOIwahli;  /* v4.0 */
    double B4SOIwahlid; /* v4.0 */

/* v3.1 added for RF */
    double B4SOIwxrcrg1;
    double B4SOIwxrcrg2;
/* v3.1 added for RF end */

    /* CV model */
    double B4SOIwvsdfb;
    double B4SOIwvsdth;
    double B4SOIwdelvt;
    double B4SOIwacde;
    double B4SOIwmoin;
    double B4SOIwnoff; /* v3.2 */

    /* Cross-term Dependence */
/* v3.1 */
    double B4SOIpxj;
    double B4SOIpalphaGB1;
    double B4SOIpbetaGB1;
    double B4SOIpalphaGB2;
    double B4SOIpbetaGB2;
    double B4SOIpaigbcp2;  /* 4.1 */
    double B4SOIpbigbcp2;  /* 4.1 */
    double B4SOIpcigbcp2;  /* 4.1 */
    double B4SOIpndif;
    double B4SOIpntrecf;
    double B4SOIpntrecr;
    double B4SOIpxbjt;
    double B4SOIpxdif;
    double B4SOIpxrec;
    double B4SOIpxtun;
    double B4SOIpxdifd;
    double B4SOIpxrecd;
    double B4SOIpxtund;
    double B4SOIpcgsl;
    double B4SOIpcgdl;
    double B4SOIpckappa;
    double B4SOIpua1;
    double B4SOIpub1;
    double B4SOIpuc1;
    double B4SOIpute;
    double B4SOIpkt1;
    double B4SOIpkt1l;
    double B4SOIpkt2;
    double B4SOIpat;
    double B4SOIpprt;

/* v3.0 */
    double B4SOIpaigc;
    double B4SOIpbigc;
    double B4SOIpcigc;
    double B4SOIpaigsd;
    double B4SOIpbigsd;
    double B4SOIpcigsd;
    double B4SOIpnigc;
    double B4SOIppigcd;
    double B4SOIppoxedge;

    double B4SOIpnpeak;
    double B4SOIpnsub;
    double B4SOIpngate;
    double B4SOIpvth0;
    double B4SOIpvfb;  /* v4.1 */
    double B4SOIpk1;
    double B4SOIpk1w1;
    double B4SOIpk1w2;
    double B4SOIpk2;
    double B4SOIpk3;
    double B4SOIpk3b;
    double B4SOIpkb1;
    double B4SOIpw0;
    double B4SOIplpe0;
    double B4SOIpdvt0;
    double B4SOIpdvt1;
    double B4SOIpdvt2;
    double B4SOIpdvt0w;
    double B4SOIpdvt1w;
    double B4SOIpdvt2w;
    double B4SOIpu0;
    double B4SOIpua;
    double B4SOIpub;
    double B4SOIpuc;
    double B4SOIpvsat;
    double B4SOIpa0;
    double B4SOIpags;
    double B4SOIpb0;
    double B4SOIpb1;
    double B4SOIpketa;
    double B4SOIpketas;
    double B4SOIpa1;
    double B4SOIpa2;
    double B4SOIprdsw;
    double B4SOIprdw;  /* v4.0 for rdsMod = 1 */
    double B4SOIprsw;  /* v4.0 for rdsMod = 1 */
    double B4SOIpprwb;
    double B4SOIpprwg;
    double B4SOIpwr;
    double B4SOIpnfactor;
    double B4SOIpdwg;
    double B4SOIpdwb;
    double B4SOIpvoff;
    double B4SOIpeta0;
    double B4SOIpetab;
    double B4SOIpdsub;
    double B4SOIpcit;
    double B4SOIpcdsc;
    double B4SOIpcdscb;
    double B4SOIpcdscd;
    double B4SOIppclm;
    double B4SOIppdibl1;
    double B4SOIppdibl2;
    double B4SOIppdiblb;
    double B4SOIpdrout;
    double B4SOIppvag;
    double B4SOIpdelta;
    double B4SOIpalpha0;
    double B4SOIpfbjtii;
        /*4.1 Iii model*/
        double B4SOIpebjtii;
        double B4SOIpcbjtii;
        double B4SOIpvbci;
        double B4SOIpabjtii;
        double B4SOIpmbjtii;

    double B4SOIpbeta0;
    double B4SOIpbeta1;
    double B4SOIpbeta2;
    double B4SOIpvdsatii0;
    double B4SOIplii;
    double B4SOIpesatii;
    double B4SOIpsii0;
    double B4SOIpsii1;
    double B4SOIpsii2;
    double B4SOIpsiid;
    double B4SOIpagidl;
    double B4SOIpbgidl;
    double B4SOIpcgidl;
    double B4SOIpegidl;
        double B4SOIprgidl;
        double B4SOIpkgidl;
        double B4SOIpfgidl;
        double B4SOIpagisl;
    double B4SOIpbgisl;
    double B4SOIpcgisl;
    double B4SOIpegisl;
        double B4SOIprgisl;
        double B4SOIpkgisl;
        double B4SOIpfgisl;
    double B4SOIpntun;          /* v4.0 */
    double B4SOIpntund;         /* v4.0 */
    double B4SOIpndiode;        /* v4.0 */
    double B4SOIpndioded;       /* v4.0 */
    double B4SOIpnrecf0;        /* v4.0 */
    double B4SOIpnrecf0d;       /* v4.0 */
    double B4SOIpnrecr0;        /* v4.0 */
    double B4SOIpnrecr0d;       /* v4.0 */
    double B4SOIpisbjt;
    double B4SOIpidbjt;   /* v4.0 */
    double B4SOIpisdif;
    double B4SOIpiddif;   /* v4.0 */
    double B4SOIpisrec;
    double B4SOIpidrec;   /* v4.0 */
    double B4SOIpistun;
    double B4SOIpidtun;   /* v4.0 */
    double B4SOIpvrec0;  /* v4.0 */
    double B4SOIpvrec0d;  /* v4.0 */
    double B4SOIpvtun0;  /* v4.0 */
    double B4SOIpvtun0d;  /* v4.0 */
    double B4SOIpnbjt;
    double B4SOIplbjt0;
    double B4SOIpvabjt;
    double B4SOIpaely;
    double B4SOIpahli;  /* v4.0 */
    double B4SOIpahlid; /* v4.0 */
/* v3.1 added for RF */
    double B4SOIpxrcrg1;
    double B4SOIpxrcrg2;
/* v3.1 added for RF end */

    /* CV model */
    double B4SOIpvsdfb;
    double B4SOIpvsdth;
    double B4SOIpdelvt;
    double B4SOIpacde;
    double B4SOIpmoin;
    double B4SOIpnoff; /* v3.2 */
/* Added for binning - END1 */

/* Pre-calculated constants */
    double B4SOIcbox;
    double B4SOIcsi;
  /*  double B4SOIcsieff;
    double B4SOIcoxt;
    double B4SOInfb;
    double B4SOIadice  */     /* v4.2 never used in the code */
    double B4SOIeg0;
        double B4SOIeg; /* Jun 09*/

    /* v4.0 added for stress effect */
    double B4SOIsaref;
    double B4SOIsbref;
    double B4SOIwlod;
    double B4SOIku0;
    double B4SOIkvsat;
    double B4SOIkvth0;
    double B4SOItku0;
    double B4SOIllodku0;
    double B4SOIwlodku0;
    double B4SOIllodvth;
    double B4SOIwlodvth;
    double B4SOIlku0;
    double B4SOIwku0;
    double B4SOIpku0;
    double B4SOIlkvth0;
    double B4SOIwkvth0;
    double B4SOIpkvth0;
    double B4SOIstk2;
    double B4SOIlodk2;
    double B4SOIsteta0;
    double B4SOIlodeta0;
    /* v4.0 added for stress effect end */

    /* MCJ: move to size-dependent param. */
    double B4SOIvtm;
    double B4SOIcox;
  /*  double B4SOIcof1;
    double B4SOIcof2;
    double B4SOIcof3;
    double B4SOIcof4   */   /* v4.2 never used in the code */
    double B4SOIvcrit;
    double B4SOIfactor1;

    double B4SOIoxideTrapDensityA;
    double B4SOIoxideTrapDensityB;
    double B4SOIoxideTrapDensityC;
    double B4SOIem;
    double B4SOIef;
    double B4SOIaf;
    double B4SOIkf;
    double B4SOInoif;
    double B4SOIbf;     /* v4.0 for noise */
    double B4SOIw0flk;  /* v4.0 for noise */
    double B4SOIlpeb;   /* v4.0 for Vth */
    double B4SOIllpeb;  /* v4.0 for Vth */
    double B4SOIwlpeb;  /* v4.0 for Vth */
    double B4SOIplpeb;  /* v4.0 for Vth */
    double B4SOIdvtp0;  /* v4.0 for Vth */
    double B4SOIldvtp0; /* v4.0 for Vth */
    double B4SOIwdvtp0; /* v4.0 for Vth */
    double B4SOIpdvtp0; /* v4.0 for Vth */
    double B4SOIdvtp1;  /* v4.0 for Vth */
    double B4SOIldvtp1; /* v4.0 for Vth */
    double B4SOIwdvtp1; /* v4.0 for Vth */
    double B4SOIpdvtp1; /* v4.0 for Vth */
    double B4SOIdvtp2;  /* v4.1 for Vth */
    double B4SOIldvtp2; /* v4.1 for Vth */
    double B4SOIwdvtp2; /* v4.1 for Vth */
    double B4SOIpdvtp2; /* v4.1 for Vth */
    double B4SOIdvtp3;  /* v4.1 for Vth */
    double B4SOIldvtp3; /* v4.1 for Vth */
    double B4SOIwdvtp3; /* v4.1 for Vth */
    double B4SOIpdvtp3; /* v4.1 for Vth */
    double B4SOIdvtp4;  /* v4.1 for Vth */
    double B4SOIldvtp4; /* v4.1 for Vth */
    double B4SOIwdvtp4; /* v4.1 for Vth */
    double B4SOIpdvtp4; /* v4.1 for Vth */
    double B4SOIminv;   /* v4.0 for Vgsteff */
    double B4SOIlminv;  /* v4.0 for Vgsteff */
    double B4SOIwminv;  /* v4.0 for Vgsteff */
    double B4SOIpminv;  /* v4.0 for Vgsteff */
    double B4SOIfprout; /* v4.0 for DITS in Id */
    double B4SOIlfprout; /* v4.0 for DITS in Id */
    double B4SOIwfprout; /* v4.0 for DITS in Id */
    double B4SOIpfprout; /* v4.0 for DITS in Id */
    double B4SOIpdits;   /* v4.0 for DITS in Id */
    double B4SOIlpdits;  /* v4.0 for DITS in Id */
    double B4SOIwpdits;  /* v4.0 for DITS in Id */
    double B4SOIppdits;  /* v4.0 for DITS in Id */
    double B4SOIpditsd;  /* v4.0 for DITS in Id */
    double B4SOIlpditsd; /* v4.0 for DITS in Id */
    double B4SOIwpditsd; /* v4.0 for DITS in Id */
    double B4SOIppditsd; /* v4.0 for DITS in Id */
    double B4SOIpditsl;  /* v4.0 for DITS in Id */
   /* 4.0 backward compatibility  */
    double B4SOInlx;
    double B4SOIlnlx;
    double B4SOIwnlx;
    double B4SOIpnlx;
    unsigned  B4SOInlxGiven   :1;
    unsigned  B4SOIlnlxGiven   :1;
    unsigned  B4SOIwnlxGiven   :1;
    unsigned  B4SOIpnlxGiven   :1;
    double B4SOIngidl;
    double B4SOIlngidl;
    double B4SOIwngidl;
    double B4SOIpngidl;
    unsigned  B4SOIngidlGiven   :1;
    unsigned  B4SOIlngidlGiven   :1;
    unsigned  B4SOIwngidlGiven   :1;
    unsigned  B4SOIpngidlGiven   :1;

    double B4SOIvgsMax;
    double B4SOIvgdMax;
    double B4SOIvgbMax;
    double B4SOIvdsMax;
    double B4SOIvbsMax;
    double B4SOIvbdMax;
    double B4SOIvgsrMax;
    double B4SOIvgdrMax;
    double B4SOIvgbrMax;
    double B4SOIvbsrMax;
    double B4SOIvbdrMax;
    unsigned  B4SOIvgsMaxGiven  :1;
    unsigned  B4SOIvgdMaxGiven  :1;
    unsigned  B4SOIvgbMaxGiven  :1;
    unsigned  B4SOIvdsMaxGiven  :1;
    unsigned  B4SOIvbsMaxGiven  :1;
    unsigned  B4SOIvbdMaxGiven  :1;
    unsigned  B4SOIvgsrMaxGiven  :1;
    unsigned  B4SOIvgdrMaxGiven  :1;
    unsigned  B4SOIvgbrMaxGiven  :1;
    unsigned  B4SOIvbsrMaxGiven  :1;
    unsigned  B4SOIvbdrMaxGiven  :1;

    struct b4soiSizeDependParam *pSizeDependParamKnot;

#ifdef USE_OMP
    int B4SOIInstCount;
    struct sB4SOIinstance **B4SOIInstanceArray;
#endif

    /* Flags */
        unsigned B4SOIepsrgateGiven:1;
        unsigned B4SOIadosGiven    :1;
        unsigned B4SOIbdosGiven    :1;
        unsigned B4SOIleffeotGiven :1;
        unsigned B4SOIweffeotGiven :1;
        unsigned B4SOIvddeotGiven  :1;
        unsigned B4SOItempeotGiven :1;

        unsigned B4SOItoxpGiven    :1;
    unsigned B4SOImtrlModGiven :1; /*4.1*/
        unsigned B4SOIvgstcvModGiven :1;
        unsigned B4SOIgidlModGiven :1;
        unsigned B4SOIiiiModGiven  :1;
    unsigned B4SOIrdsModGiven :1; /* v4.0 */
    unsigned B4SOIrbodyModGiven :1; /* v4.0 */
    unsigned B4SOIrgateModGiven :1; /* v3.1 */

/* v3.0 */
    unsigned B4SOIsoiModGiven: 1;
    unsigned B4SOIvbs0pdGiven: 1; /* v3.2 */
    unsigned B4SOIvbs0fdGiven: 1; /* v3.2 */
    unsigned B4SOIvbsaGiven  : 1;
    unsigned B4SOInofffdGiven: 1;
    unsigned B4SOIvofffdGiven: 1;
    unsigned B4SOIk1bGiven:    1;
    unsigned B4SOIk2bGiven:    1;
    unsigned B4SOIdk2bGiven:   1;
    unsigned B4SOIdvbd0Given:  1;
    unsigned B4SOIdvbd1Given:  1;
    unsigned B4SOImoinFDGiven: 1;


    unsigned B4SOItboxGiven:1;
    unsigned B4SOItsiGiven :1;
        unsigned B4SOIetsiGiven :1;
    unsigned B4SOIxjGiven :1;
    unsigned B4SOIkb1Given :1;
    unsigned B4SOIrth0Given :1;
    unsigned B4SOIcth0Given :1;
    unsigned B4SOIcfrcoeffGiven :1;  /* v4.4 */
    unsigned B4SOIegidlGiven :1;
    unsigned B4SOIagidlGiven :1;
    unsigned B4SOIbgidlGiven :1;
    unsigned B4SOIcgidlGiven :1;
        unsigned B4SOIrgidlGiven :1;
        unsigned B4SOIkgidlGiven :1;
        unsigned B4SOIfgidlGiven :1;
        unsigned B4SOIegislGiven :1;
    unsigned B4SOIagislGiven :1;
    unsigned B4SOIbgislGiven :1;
    unsigned B4SOIcgislGiven :1;
        unsigned B4SOIrgislGiven :1;
        unsigned B4SOIkgislGiven :1;
        unsigned B4SOIfgislGiven :1;
    unsigned B4SOIndiodeGiven :1;  /* v4.0 */
    unsigned B4SOIndiodedGiven :1;  /* v4.0 */
    unsigned B4SOIxbjtGiven :1;
    unsigned B4SOIxdifGiven :1;
    unsigned B4SOIxrecGiven :1;
    unsigned B4SOIxtunGiven :1;
    unsigned B4SOIxdifdGiven :1;
    unsigned B4SOIxrecdGiven :1;
    unsigned B4SOIxtundGiven :1;
    unsigned B4SOIttGiven :1;
    unsigned B4SOIvsdfbGiven :1;
    unsigned B4SOIvsdthGiven :1;
    unsigned B4SOIasdGiven :1;
    unsigned B4SOIcsdminGiven :1;

    unsigned  B4SOImobModGiven :1;
    unsigned  B4SOIbinUnitGiven :1;
    unsigned  B4SOIcapModGiven :1;
    unsigned  B4SOIparamChkGiven :1;
/*    unsigned  B4SOInoiModGiven :1;  v3.2 */
    unsigned  B4SOIshModGiven :1;
    unsigned  B4SOItypeGiven   :1;
    unsigned  B4SOItoxGiven   :1;
    unsigned  B4SOItoxmGiven   :1; /* v3.2 */
    unsigned  B4SOIdtoxcvGiven   :1; /* v2.2.3 */
    unsigned  B4SOIversionGiven   :1;

    unsigned  B4SOIcdscGiven   :1;
    unsigned  B4SOIcdscbGiven   :1;
    unsigned  B4SOIcdscdGiven   :1;
    unsigned  B4SOIcitGiven   :1;
    unsigned  B4SOInfactorGiven   :1;
    unsigned  B4SOIvsatGiven   :1;
    unsigned  B4SOIatGiven   :1;
    unsigned  B4SOIa0Given   :1;
    unsigned  B4SOIagsGiven   :1;
    unsigned  B4SOIa1Given   :1;
    unsigned  B4SOIa2Given   :1;
    unsigned  B4SOIketaGiven   :1;
    unsigned  B4SOInsubGiven   :1;
    unsigned  B4SOInpeakGiven   :1;
    unsigned  B4SOIngateGiven   :1;
        unsigned  B4SOInsdGiven     :1;
    unsigned  B4SOIgamma1Given   :1;
    unsigned  B4SOIgamma2Given   :1;
    unsigned  B4SOIvbxGiven   :1;
    unsigned  B4SOIvbmGiven   :1;
    unsigned  B4SOIxtGiven   :1;
    unsigned  B4SOIk1Given   :1;
    unsigned  B4SOIkt1Given   :1;
    unsigned  B4SOIkt1lGiven   :1;
    unsigned  B4SOIkt2Given   :1;
    unsigned  B4SOIk2Given   :1;
    unsigned  B4SOIk3Given   :1;
    unsigned  B4SOIk3bGiven   :1;
    unsigned  B4SOIw0Given   :1;
    unsigned  B4SOIlpe0Given   :1;
    unsigned  B4SOIdvt0Given   :1;
    unsigned  B4SOIdvt1Given   :1;
    unsigned  B4SOIdvt2Given   :1;
    unsigned  B4SOIdvt0wGiven   :1;
    unsigned  B4SOIdvt1wGiven   :1;
    unsigned  B4SOIdvt2wGiven   :1;
    unsigned  B4SOIdroutGiven   :1;
    unsigned  B4SOIdsubGiven   :1;
    unsigned  B4SOIvth0Given   :1;
    unsigned  B4SOIuaGiven   :1;
    unsigned  B4SOIua1Given   :1;
    unsigned  B4SOIubGiven   :1;
    unsigned  B4SOIub1Given   :1;
    unsigned  B4SOIucGiven   :1;
    unsigned  B4SOIuc1Given   :1;
    unsigned  B4SOIu0Given   :1;
    unsigned  B4SOIuteGiven   :1;
        /*4.1 mobmod=4*/
        unsigned  B4SOIudGiven    :1;
        unsigned  B4SOIludGiven   :1;
        unsigned  B4SOIwudGiven   :1;
        unsigned  B4SOIpudGiven   :1;

        unsigned  B4SOIud1Given   :1;
        unsigned  B4SOIlud1Given  :1;
        unsigned  B4SOIwud1Given  :1;
        unsigned  B4SOIpud1Given  :1;

        unsigned  B4SOIeuGiven    :1;
        unsigned  B4SOIleuGiven   :1;
        unsigned  B4SOIweuGiven   :1;
        unsigned  B4SOIpeuGiven   :1;

        unsigned  B4SOIucsGiven   :1;
        unsigned  B4SOIlucsGiven  :1;
        unsigned  B4SOIwucsGiven  :1;
        unsigned  B4SOIpucsGiven  :1;

        unsigned  B4SOIucsteGiven :1;
        unsigned  B4SOIlucsteGiven:1;
        unsigned  B4SOIwucsteGiven:1;
        unsigned  B4SOIpucsteGiven:1;

    unsigned  B4SOIvoffGiven   :1;
    unsigned  B4SOIrdswGiven   :1;
    unsigned  B4SOIrdwGiven   :1;       /* v4.0 for rdsMod = 1 */
    unsigned  B4SOIrswGiven   :1;       /* v4.0 for rdsMod = 1 */
    unsigned  B4SOIrdwminGiven   :1;    /* v4.0 for rdsMod = 1 */
    unsigned  B4SOIrswminGiven   :1;    /* v4.0 for rdsMod = 1 */
    unsigned  B4SOIprwgGiven   :1;
    unsigned  B4SOIprwbGiven   :1;
    unsigned  B4SOIprtGiven   :1;
    unsigned  B4SOIeta0Given   :1;
    unsigned  B4SOIetabGiven   :1;
    unsigned  B4SOIpclmGiven   :1;
    unsigned  B4SOIpdibl1Given   :1;
    unsigned  B4SOIpdibl2Given   :1;
    unsigned  B4SOIpdiblbGiven   :1;
    unsigned  B4SOIpvagGiven   :1;
    unsigned  B4SOIdeltaGiven  :1;
    unsigned  B4SOIwrGiven   :1;
    unsigned  B4SOIdwgGiven   :1;
    unsigned  B4SOIdwbGiven   :1;
    unsigned  B4SOIb0Given   :1;
    unsigned  B4SOIb1Given   :1;
    unsigned  B4SOIalpha0Given   :1;


        /*4.1*/
        unsigned  B4SOIepsroxGiven :1;
    unsigned  B4SOIeotGiven    :1;
        unsigned  B4SOIepsrsubGiven   :1;
    unsigned  B4SOIni0subGiven :1;
        unsigned  B4SOIbg0subGiven :1;
        unsigned  B4SOItbgasubGiven:1;
        unsigned  B4SOItbgbsubGiven:1;
        unsigned  B4SOIphigGiven   :1;
        unsigned  B4SOIeasubGiven  :1;

    /* v3.1 added for RF */
    unsigned  B4SOIxrcrg1Given   :1;
    unsigned  B4SOIxrcrg2Given   :1;
    unsigned  B4SOIrshgGiven    :1;
    unsigned  B4SOIngconGiven   :1;
    unsigned  B4SOIxgwGiven     :1;
    unsigned  B4SOIxglGiven     :1;
    /* v3.1 added for RF end */

    /* v3.2 */
    unsigned  B4SOIfnoiModGiven :1;
    unsigned  B4SOItnoiModGiven :1;
    unsigned  B4SOItnoiaGiven   :1;
    unsigned  B4SOItnoibGiven   :1;
    unsigned  B4SOIrnoiaGiven   :1;
    unsigned  B4SOIrnoibGiven   :1;
    unsigned  B4SOIntnoiGiven   :1;
    /* v3.2 end */

    /* v4.0 */
    unsigned  B4SOIvfbGiven      :1;
    unsigned  B4SOIgbminGiven    :1;
    unsigned  B4SOIrbdbGiven     :1;
    unsigned  B4SOIrbsbGiven     :1;
    /* v4.0 end */
    /* v4.1 */
    unsigned  B4SOIfdModGiven        :1;
        unsigned  B4SOIvsceGiven         :1;
    unsigned  B4SOIcdsbsGiven        :1;
        unsigned  B4SOIminvcvGiven       :1;
    unsigned  B4SOIlminvcvGiven      :1;
    unsigned  B4SOIwminvcvGiven      :1;
    unsigned  B4SOIpminvcvGiven      :1;
    unsigned  B4SOIvoffcvGiven       :1;
    unsigned  B4SOIlvoffcvGiven      :1;
    unsigned  B4SOIwvoffcvGiven      :1;
    unsigned  B4SOIpvoffcvGiven      :1;
/* v2.2 release */
    unsigned  B4SOIwth0Given   :1;
    unsigned  B4SOIrhaloGiven  :1;
    unsigned  B4SOIntoxGiven   :1;
    unsigned  B4SOItoxrefGiven   :1;
    unsigned  B4SOIebgGiven     :1;
    unsigned  B4SOIvevbGiven   :1;
    unsigned  B4SOIalphaGB1Given :1;
    unsigned  B4SOIbetaGB1Given  :1;
    unsigned  B4SOIvgb1Given        :1;
    unsigned  B4SOIvecbGiven        :1;
    unsigned  B4SOIalphaGB2Given    :1;
    unsigned  B4SOIbetaGB2Given     :1;
    unsigned  B4SOIvgb2Given        :1;
    unsigned  B4SOIaigbcp2Given    :1;  /* 4.1 */
    unsigned  B4SOIbigbcp2Given    :1;  /* 4.1 */
    unsigned  B4SOIcigbcp2Given    :1;  /* 4.1 */
    unsigned  B4SOItoxqmGiven  :1;
    unsigned  B4SOIigbModGiven  :1; /* v3.0 */
    unsigned  B4SOIvoxhGiven   :1;
    unsigned  B4SOIdeltavoxGiven :1;
    unsigned  B4SOIigcModGiven  :1; /* v3.0 */
/* v3.0 */
    unsigned  B4SOIaigcGiven   :1;
    unsigned  B4SOIbigcGiven   :1;
    unsigned  B4SOIcigcGiven   :1;
    unsigned  B4SOIaigsdGiven   :1;
    unsigned  B4SOIbigsdGiven   :1;
    unsigned  B4SOIcigsdGiven   :1;
    unsigned  B4SOInigcGiven   :1;
    unsigned  B4SOIpigcdGiven   :1;
    unsigned  B4SOIpoxedgeGiven   :1;
    unsigned  B4SOIdlcigGiven   :1;


/* v2.0 release */
    unsigned  B4SOIk1w1Given   :1;
    unsigned  B4SOIk1w2Given   :1;
    unsigned  B4SOIketasGiven  :1;
    unsigned  B4SOIdwbcGiven  :1;
    unsigned  B4SOIbeta0Given  :1;
    unsigned  B4SOIbeta1Given  :1;
    unsigned  B4SOIbeta2Given  :1;
    unsigned  B4SOIvdsatii0Given  :1;
    unsigned  B4SOItiiGiven  :1;
        unsigned  B4SOItvbciGiven :1;
    unsigned  B4SOIliiGiven  :1;
    unsigned  B4SOIsii0Given  :1;
    unsigned  B4SOIsii1Given  :1;
    unsigned  B4SOIsii2Given  :1;
    unsigned  B4SOIsiidGiven  :1;
    unsigned  B4SOIfbjtiiGiven :1;
        /*4.1 Iii model*/
        unsigned  B4SOIebjtiiGiven :1;
        unsigned  B4SOIcbjtiiGiven :1;
        unsigned  B4SOIvbciGiven   :1;
        unsigned  B4SOIabjtiiGiven :1;
        unsigned  B4SOImbjtiiGiven :1;

    unsigned  B4SOIesatiiGiven :1;
    unsigned  B4SOIntunGiven  :1;      /* v4.0 */
    unsigned  B4SOIntundGiven  :1;      /* v4.0 */
    unsigned  B4SOInrecf0Given  :1;     /* v4.0 */
    unsigned  B4SOInrecf0dGiven  :1;    /* v4.0 */
    unsigned  B4SOInrecr0Given  :1;     /* v4.0 */
    unsigned  B4SOInrecr0dGiven  :1;    /* v4.0 */
    unsigned  B4SOIisbjtGiven  :1;
    unsigned  B4SOIidbjtGiven  :1;      /* v4.0 */
    unsigned  B4SOIisdifGiven  :1;
    unsigned  B4SOIiddifGiven  :1;      /* v4.0 */
    unsigned  B4SOIisrecGiven  :1;
    unsigned  B4SOIidrecGiven  :1;      /* v4.0 */
    unsigned  B4SOIistunGiven  :1;
    unsigned  B4SOIidtunGiven  :1;      /* v4.0 */
    unsigned  B4SOIlnGiven  :1;
    unsigned  B4SOIvrec0Given  :1;      /* v4.0 */
    unsigned  B4SOIvrec0dGiven  :1;     /* v4.0 */
    unsigned  B4SOIvtun0Given  :1;      /* v4.0 */
    unsigned  B4SOIvtun0dGiven  :1;     /* v4.0 */
    unsigned  B4SOInbjtGiven  :1;
    unsigned  B4SOIlbjt0Given  :1;
    unsigned  B4SOIldif0Given  :1;
    unsigned  B4SOIvabjtGiven  :1;
    unsigned  B4SOIaelyGiven  :1;
    unsigned  B4SOIahliGiven :1;        /* v4.0 */
    unsigned  B4SOIahlidGiven :1;       /* v4.0 */
    unsigned  B4SOIrbodyGiven :1;
    unsigned  B4SOIrbshGiven  :1;
    unsigned  B4SOIndifGiven  :1;
    unsigned  B4SOIntrecfGiven  :1;
    unsigned  B4SOIntrecrGiven  :1;
    unsigned  B4SOIdlcbGiven    :1;
    unsigned  B4SOIfbodyGiven   :1;
    unsigned  B4SOItcjswgGiven  :1;
    unsigned  B4SOItpbswgGiven  :1;
    unsigned  B4SOItcjswgdGiven  :1;
    unsigned  B4SOItpbswgdGiven  :1;
    unsigned  B4SOIacdeGiven  :1;
    unsigned  B4SOImoinGiven  :1;
    unsigned  B4SOInoffGiven:  1; /* v3.2 */
    unsigned  B4SOIdelvtGiven  :1;
    unsigned  B4SOIdlbgGiven  :1;


    /* CV model */
    unsigned  B4SOIcgslGiven   :1;
    unsigned  B4SOIcgdlGiven   :1;
    unsigned  B4SOIckappaGiven   :1;
    unsigned  B4SOIcfGiven   :1;
    unsigned  B4SOIclcGiven   :1;
    unsigned  B4SOIcleGiven   :1;
    unsigned  B4SOIdwcGiven   :1;
    unsigned  B4SOIdlcGiven   :1;

/* Added for binning - START2 */
    /* Length Dependence */
/* v3.1 */
    unsigned B4SOIlxjGiven :1;
    unsigned B4SOIlalphaGB1Given :1;
    unsigned B4SOIlbetaGB1Given :1;
    unsigned B4SOIlalphaGB2Given :1;
    unsigned B4SOIlbetaGB2Given :1;
    unsigned B4SOIlaigbcp2Given :1;  /* 4.1 */
    unsigned B4SOIlbigbcp2Given :1;  /* 4.1 */
    unsigned B4SOIlcigbcp2Given :1;  /* 4.1 */
    unsigned B4SOIlndifGiven :1;
    unsigned B4SOIlntrecfGiven  :1;
    unsigned B4SOIlntrecrGiven  :1;
    unsigned B4SOIlxbjtGiven :1;
    unsigned B4SOIlxdifGiven :1;
    unsigned B4SOIlxrecGiven :1;
    unsigned B4SOIlxtunGiven :1;
    unsigned B4SOIlxdifdGiven :1;
    unsigned B4SOIlxrecdGiven :1;
    unsigned B4SOIlxtundGiven :1;
    unsigned B4SOIlcgslGiven :1;
    unsigned B4SOIlcgdlGiven :1;
    unsigned B4SOIlckappaGiven :1;
    unsigned B4SOIlua1Given :1;
    unsigned B4SOIlub1Given :1;
    unsigned B4SOIluc1Given :1;
    unsigned B4SOIluteGiven :1;
    unsigned B4SOIlkt1Given :1;
    unsigned B4SOIlkt1lGiven :1;
    unsigned B4SOIlkt2Given :1;
    unsigned B4SOIlatGiven :1;
    unsigned B4SOIlprtGiven :1;

/* v3.0 */
    unsigned  B4SOIlaigcGiven   :1;
    unsigned  B4SOIlbigcGiven   :1;
    unsigned  B4SOIlcigcGiven   :1;
    unsigned  B4SOIlaigsdGiven   :1;
    unsigned  B4SOIlbigsdGiven   :1;
    unsigned  B4SOIlcigsdGiven   :1;
    unsigned  B4SOIlnigcGiven   :1;
    unsigned  B4SOIlpigcdGiven   :1;
    unsigned  B4SOIlpoxedgeGiven   :1;

    unsigned  B4SOIlnpeakGiven   :1;
    unsigned  B4SOIlnsubGiven   :1;
    unsigned  B4SOIlngateGiven   :1;
    unsigned  B4SOIlnsdGiven     :1;
    unsigned  B4SOIlvth0Given   :1;
    unsigned  B4SOIlvfbGiven   :1;   /* v4.1 */
    unsigned  B4SOIlk1Given   :1;
    unsigned  B4SOIlk1w1Given   :1;
    unsigned  B4SOIlk1w2Given   :1;
    unsigned  B4SOIlk2Given   :1;
    unsigned  B4SOIlk3Given   :1;
    unsigned  B4SOIlk3bGiven   :1;
    unsigned  B4SOIlkb1Given   :1;
    unsigned  B4SOIlw0Given   :1;
    unsigned  B4SOIllpe0Given   :1;
    unsigned  B4SOIldvt0Given   :1;
    unsigned  B4SOIldvt1Given   :1;
    unsigned  B4SOIldvt2Given   :1;
    unsigned  B4SOIldvt0wGiven   :1;
    unsigned  B4SOIldvt1wGiven   :1;
    unsigned  B4SOIldvt2wGiven   :1;
    unsigned  B4SOIlu0Given   :1;
    unsigned  B4SOIluaGiven   :1;
    unsigned  B4SOIlubGiven   :1;
    unsigned  B4SOIlucGiven   :1;
    unsigned  B4SOIlvsatGiven   :1;
    unsigned  B4SOIla0Given   :1;
    unsigned  B4SOIlagsGiven   :1;
    unsigned  B4SOIlb0Given   :1;
    unsigned  B4SOIlb1Given   :1;
    unsigned  B4SOIlketaGiven   :1;
    unsigned  B4SOIlketasGiven   :1;
    unsigned  B4SOIla1Given   :1;
    unsigned  B4SOIla2Given   :1;
    unsigned  B4SOIlrdswGiven   :1;
    unsigned  B4SOIlrdwGiven   :1;      /* v4.0 for rdsMod = 1 */
    unsigned  B4SOIlrswGiven   :1;      /* v4.0 for rdsMod = 1 */
    unsigned  B4SOIlprwbGiven   :1;
    unsigned  B4SOIlprwgGiven   :1;
    unsigned  B4SOIlwrGiven   :1;
    unsigned  B4SOIlnfactorGiven   :1;
    unsigned  B4SOIldwgGiven   :1;
    unsigned  B4SOIldwbGiven   :1;
    unsigned  B4SOIlvoffGiven   :1;
    unsigned  B4SOIleta0Given   :1;
    unsigned  B4SOIletabGiven   :1;
    unsigned  B4SOIldsubGiven   :1;
    unsigned  B4SOIlcitGiven   :1;
    unsigned  B4SOIlcdscGiven   :1;
    unsigned  B4SOIlcdscbGiven   :1;
    unsigned  B4SOIlcdscdGiven   :1;
    unsigned  B4SOIlpclmGiven   :1;
    unsigned  B4SOIlpdibl1Given   :1;
    unsigned  B4SOIlpdibl2Given   :1;
    unsigned  B4SOIlpdiblbGiven   :1;
    unsigned  B4SOIldroutGiven   :1;
    unsigned  B4SOIlpvagGiven   :1;
    unsigned  B4SOIldeltaGiven   :1;
    unsigned  B4SOIlalpha0Given   :1;
    unsigned  B4SOIlfbjtiiGiven   :1;
        /*4.1 Iii model*/
        unsigned  B4SOIlebjtiiGiven :1;
        unsigned  B4SOIlcbjtiiGiven :1;
        unsigned  B4SOIlvbciGiven   :1;
        unsigned  B4SOIlabjtiiGiven :1;
        unsigned  B4SOIlmbjtiiGiven :1;

    unsigned  B4SOIlbeta0Given   :1;
    unsigned  B4SOIlbeta1Given   :1;
    unsigned  B4SOIlbeta2Given   :1;
    unsigned  B4SOIlvdsatii0Given   :1;
    unsigned  B4SOIlliiGiven   :1;
    unsigned  B4SOIlesatiiGiven   :1;
    unsigned  B4SOIlsii0Given   :1;
    unsigned  B4SOIlsii1Given   :1;
    unsigned  B4SOIlsii2Given   :1;
    unsigned  B4SOIlsiidGiven   :1;
    unsigned  B4SOIlagidlGiven   :1;
    unsigned  B4SOIlbgidlGiven   :1;
    unsigned  B4SOIlcgidlGiven   :1;
    unsigned  B4SOIlegidlGiven   :1;
        unsigned  B4SOIlrgidlGiven   :1;
        unsigned  B4SOIlkgidlGiven   :1;
        unsigned  B4SOIlfgidlGiven   :1;
            unsigned  B4SOIlagislGiven   :1;
    unsigned  B4SOIlbgislGiven   :1;
    unsigned  B4SOIlcgislGiven   :1;
    unsigned  B4SOIlegislGiven   :1;
        unsigned  B4SOIlrgislGiven   :1;
        unsigned  B4SOIlkgislGiven   :1;
        unsigned  B4SOIlfgislGiven   :1;
    unsigned  B4SOIlntunGiven   :1;     /* v4.0 */
    unsigned  B4SOIlntundGiven   :1;    /* v4.0 */
    unsigned  B4SOIlndiodeGiven   :1;   /* v4.0 */
    unsigned  B4SOIlndiodedGiven   :1;  /* v4.0 */
    unsigned  B4SOIlnrecf0Given   :1;   /* v4.0 */
    unsigned  B4SOIlnrecf0dGiven   :1;  /* v4.0 */
    unsigned  B4SOIlnrecr0Given   :1;   /* v4.0 */
    unsigned  B4SOIlnrecr0dGiven   :1;  /* v4.0 */
    unsigned  B4SOIlisbjtGiven   :1;
    unsigned  B4SOIlidbjtGiven   :1;    /* v4.0 */
    unsigned  B4SOIlisdifGiven   :1;
    unsigned  B4SOIliddifGiven   :1;    /* v4.0 */
    unsigned  B4SOIlisrecGiven   :1;
    unsigned  B4SOIlidrecGiven   :1;    /* v4.0 */
    unsigned  B4SOIlistunGiven   :1;
    unsigned  B4SOIlidtunGiven   :1;    /* v4.0 */
    unsigned  B4SOIlvrec0Given   :1;    /* v4.0 */
    unsigned  B4SOIlvrec0dGiven   :1;   /* v4.0 */
    unsigned  B4SOIlvtun0Given   :1;    /* v4.0 */
    unsigned  B4SOIlvtun0dGiven   :1;   /* v4.0 */
    unsigned  B4SOIlnbjtGiven   :1;
    unsigned  B4SOIllbjt0Given   :1;
    unsigned  B4SOIlvabjtGiven   :1;
    unsigned  B4SOIlaelyGiven   :1;
    unsigned  B4SOIlahliGiven   :1;     /* v4.0 */
    unsigned  B4SOIlahlidGiven   :1;    /* v4.0 */
/* v3.1  added for RF */
    unsigned B4SOIlxrcrg1Given  :1;
    unsigned B4SOIlxrcrg2Given  :1;
/* v3.1  added for RF end */

    /* CV model */
    unsigned  B4SOIlvsdfbGiven   :1;
    unsigned  B4SOIlvsdthGiven   :1;
    unsigned  B4SOIldelvtGiven   :1;
    unsigned  B4SOIlacdeGiven   :1;
    unsigned  B4SOIlmoinGiven   :1;
    unsigned  B4SOIlnoffGiven   :1; /* v3.2 */

    /* Width Dependence */
/* v3.1 */
    unsigned B4SOIwxjGiven :1;
    unsigned B4SOIwalphaGB1Given :1;
    unsigned B4SOIwbetaGB1Given :1;
    unsigned B4SOIwalphaGB2Given :1;
    unsigned B4SOIwbetaGB2Given :1;
    unsigned B4SOIwaigbcp2Given :1;  /* 4.1 */
    unsigned B4SOIwbigbcp2Given :1;  /* 4.1 */
    unsigned B4SOIwcigbcp2Given :1;  /* 4.1 */
    unsigned B4SOIwndifGiven :1;
    unsigned B4SOIwntrecfGiven  :1;
    unsigned B4SOIwntrecrGiven  :1;
    unsigned B4SOIwxbjtGiven :1;
    unsigned B4SOIwxdifGiven :1;
    unsigned B4SOIwxrecGiven :1;
    unsigned B4SOIwxtunGiven :1;
    unsigned B4SOIwxdifdGiven :1;
    unsigned B4SOIwxrecdGiven :1;
    unsigned B4SOIwxtundGiven :1;
    unsigned B4SOIwcgslGiven :1;
    unsigned B4SOIwcgdlGiven :1;
    unsigned B4SOIwckappaGiven :1;
    unsigned B4SOIwua1Given :1;
    unsigned B4SOIwub1Given :1;
    unsigned B4SOIwuc1Given :1;
    unsigned B4SOIwuteGiven :1;
    unsigned B4SOIwkt1Given :1;
    unsigned B4SOIwkt1lGiven :1;
    unsigned B4SOIwkt2Given :1;
    unsigned B4SOIwatGiven :1;
    unsigned B4SOIwprtGiven :1;

/* v3.0 */
    unsigned  B4SOIwaigcGiven   :1;
    unsigned  B4SOIwbigcGiven   :1;
    unsigned  B4SOIwcigcGiven   :1;
    unsigned  B4SOIwaigsdGiven   :1;
    unsigned  B4SOIwbigsdGiven   :1;
    unsigned  B4SOIwcigsdGiven   :1;
    unsigned  B4SOIwnigcGiven   :1;
    unsigned  B4SOIwpigcdGiven   :1;
    unsigned  B4SOIwpoxedgeGiven   :1;

    unsigned  B4SOIwnpeakGiven   :1;
    unsigned  B4SOIwnsubGiven   :1;
    unsigned  B4SOIwngateGiven   :1;
    unsigned  B4SOIwnsdGiven     :1;
    unsigned  B4SOIwvth0Given   :1;
    unsigned  B4SOIwvfbGiven   :1;   /* v4.1 */
    unsigned  B4SOIwk1Given   :1;
    unsigned  B4SOIwk1w1Given   :1;
    unsigned  B4SOIwk1w2Given   :1;
    unsigned  B4SOIwk2Given   :1;
    unsigned  B4SOIwk3Given   :1;
    unsigned  B4SOIwk3bGiven   :1;
    unsigned  B4SOIwkb1Given   :1;
    unsigned  B4SOIww0Given   :1;
    unsigned  B4SOIwlpe0Given   :1;
    unsigned  B4SOIwdvt0Given   :1;
    unsigned  B4SOIwdvt1Given   :1;
    unsigned  B4SOIwdvt2Given   :1;
    unsigned  B4SOIwdvt0wGiven   :1;
    unsigned  B4SOIwdvt1wGiven   :1;
    unsigned  B4SOIwdvt2wGiven   :1;
    unsigned  B4SOIwu0Given   :1;
    unsigned  B4SOIwuaGiven   :1;
    unsigned  B4SOIwubGiven   :1;
    unsigned  B4SOIwucGiven   :1;
    unsigned  B4SOIwvsatGiven   :1;
    unsigned  B4SOIwa0Given   :1;
    unsigned  B4SOIwagsGiven   :1;
    unsigned  B4SOIwb0Given   :1;
    unsigned  B4SOIwb1Given   :1;
    unsigned  B4SOIwketaGiven   :1;
    unsigned  B4SOIwketasGiven   :1;
    unsigned  B4SOIwa1Given   :1;
    unsigned  B4SOIwa2Given   :1;
    unsigned  B4SOIwrdswGiven   :1;
    unsigned  B4SOIwrdwGiven   :1;      /* v4.0 for rdsMod = 1 */
    unsigned  B4SOIwrswGiven   :1;      /* v4.0 for rdsMod = 1 */
    unsigned  B4SOIwprwbGiven   :1;
    unsigned  B4SOIwprwgGiven   :1;
    unsigned  B4SOIwwrGiven   :1;
    unsigned  B4SOIwnfactorGiven   :1;
    unsigned  B4SOIwdwgGiven   :1;
    unsigned  B4SOIwdwbGiven   :1;
    unsigned  B4SOIwvoffGiven   :1;
    unsigned  B4SOIweta0Given   :1;
    unsigned  B4SOIwetabGiven   :1;
    unsigned  B4SOIwdsubGiven   :1;
    unsigned  B4SOIwcitGiven   :1;
    unsigned  B4SOIwcdscGiven   :1;
    unsigned  B4SOIwcdscbGiven   :1;
    unsigned  B4SOIwcdscdGiven   :1;
    unsigned  B4SOIwpclmGiven   :1;
    unsigned  B4SOIwpdibl1Given   :1;
    unsigned  B4SOIwpdibl2Given   :1;
    unsigned  B4SOIwpdiblbGiven   :1;
    unsigned  B4SOIwdroutGiven   :1;
    unsigned  B4SOIwpvagGiven   :1;
    unsigned  B4SOIwdeltaGiven   :1;
    unsigned  B4SOIwalpha0Given   :1;
    unsigned  B4SOIwfbjtiiGiven   :1;
        /*4.1 Iii model*/
        unsigned  B4SOIwebjtiiGiven :1;
        unsigned  B4SOIwcbjtiiGiven :1;
        unsigned  B4SOIwvbciGiven   :1;
        unsigned  B4SOIwabjtiiGiven :1;
        unsigned  B4SOIwmbjtiiGiven :1;

    unsigned  B4SOIwbeta0Given   :1;
    unsigned  B4SOIwbeta1Given   :1;
    unsigned  B4SOIwbeta2Given   :1;
    unsigned  B4SOIwvdsatii0Given   :1;
    unsigned  B4SOIwliiGiven   :1;
    unsigned  B4SOIwesatiiGiven   :1;
    unsigned  B4SOIwsii0Given   :1;
    unsigned  B4SOIwsii1Given   :1;
    unsigned  B4SOIwsii2Given   :1;
    unsigned  B4SOIwsiidGiven   :1;
    unsigned  B4SOIwagidlGiven   :1;
    unsigned  B4SOIwbgidlGiven   :1;
    unsigned  B4SOIwcgidlGiven   :1;
    unsigned  B4SOIwegidlGiven   :1;
        unsigned  B4SOIwrgidlGiven   :1;
        unsigned  B4SOIwkgidlGiven   :1;
        unsigned  B4SOIwfgidlGiven   :1;
            unsigned  B4SOIwagislGiven   :1;
    unsigned  B4SOIwbgislGiven   :1;
    unsigned  B4SOIwcgislGiven   :1;
    unsigned  B4SOIwegislGiven   :1;
        unsigned  B4SOIwrgislGiven   :1;
        unsigned  B4SOIwkgislGiven   :1;
        unsigned  B4SOIwfgislGiven   :1;
    unsigned  B4SOIwntunGiven   :1;     /* v4.0 */
    unsigned  B4SOIwntundGiven   :1;    /* v4.0 */
    unsigned  B4SOIwndiodeGiven   :1;   /* v4.0 */
    unsigned  B4SOIwndiodedGiven   :1;  /* v4.0 */
    unsigned  B4SOIwnrecf0Given   :1;   /* v4.0 */
    unsigned  B4SOIwnrecf0dGiven   :1;  /* v4.0 */
    unsigned  B4SOIwnrecr0Given   :1;   /* v4.0 */
    unsigned  B4SOIwnrecr0dGiven   :1;  /* v4.0 */
    unsigned  B4SOIwisbjtGiven   :1;
    unsigned  B4SOIwidbjtGiven   :1;    /* v4.0 */
    unsigned  B4SOIwisdifGiven   :1;
    unsigned  B4SOIwiddifGiven   :1;    /* v4.0 */
    unsigned  B4SOIwisrecGiven   :1;
    unsigned  B4SOIwidrecGiven   :1;    /* v4.0 */
    unsigned  B4SOIwistunGiven   :1;
    unsigned  B4SOIwidtunGiven   :1;    /* v4.0 */
    unsigned  B4SOIwvrec0Given   :1;    /* v4.0 */
    unsigned  B4SOIwvrec0dGiven   :1;   /* v4.0 */
    unsigned  B4SOIwvtun0Given   :1;    /* v4.0 */
    unsigned  B4SOIwvtun0dGiven   :1;   /* v4.0 */
    unsigned  B4SOIwnbjtGiven   :1;
    unsigned  B4SOIwlbjt0Given   :1;
    unsigned  B4SOIwvabjtGiven   :1;
    unsigned  B4SOIwaelyGiven   :1;
    unsigned  B4SOIwahliGiven  :1;      /* v4.0 */
    unsigned  B4SOIwahlidGiven  :1;     /* v4.0 */
/* v3.1  added for RF */
    unsigned B4SOIwxrcrg1Given  :1;
    unsigned B4SOIwxrcrg2Given  :1;
/* v3.1  added for RF end */

    /* CV model */
    unsigned  B4SOIwvsdfbGiven   :1;
    unsigned  B4SOIwvsdthGiven   :1;
    unsigned  B4SOIwdelvtGiven   :1;
    unsigned  B4SOIwacdeGiven   :1;
    unsigned  B4SOIwmoinGiven   :1;
    unsigned  B4SOIwnoffGiven   :1; /* v3.2 */

    /* Cross-term Dependence */
/* v3.1 */
    unsigned B4SOIpxjGiven :1;
    unsigned B4SOIpalphaGB1Given :1;
    unsigned B4SOIpbetaGB1Given :1;
    unsigned B4SOIpalphaGB2Given :1;
    unsigned B4SOIpbetaGB2Given :1;
    unsigned B4SOIpaigbcp2Given :1;  /* 4.1 */
    unsigned B4SOIpbigbcp2Given :1;  /* 4.1 */
    unsigned B4SOIpcigbcp2Given :1;  /* 4.1 */
    unsigned B4SOIpndifGiven :1;
    unsigned B4SOIpntrecfGiven  :1;
    unsigned B4SOIpntrecrGiven  :1;
    unsigned B4SOIpxbjtGiven :1;
    unsigned B4SOIpxdifGiven :1;
    unsigned B4SOIpxrecGiven :1;
    unsigned B4SOIpxtunGiven :1;
    unsigned B4SOIpxdifdGiven :1;
    unsigned B4SOIpxrecdGiven :1;
    unsigned B4SOIpxtundGiven :1;
    unsigned B4SOIpcgslGiven :1;
    unsigned B4SOIpcgdlGiven :1;
    unsigned B4SOIpckappaGiven :1;
    unsigned B4SOIpua1Given :1;
    unsigned B4SOIpub1Given :1;
    unsigned B4SOIpuc1Given :1;
    unsigned B4SOIputeGiven :1;
    unsigned B4SOIpkt1Given :1;
    unsigned B4SOIpkt1lGiven :1;
    unsigned B4SOIpkt2Given :1;
    unsigned B4SOIpatGiven :1;
    unsigned B4SOIpprtGiven :1;

/* v3.0 */
    unsigned  B4SOIpaigcGiven   :1;
    unsigned  B4SOIpbigcGiven   :1;
    unsigned  B4SOIpcigcGiven   :1;
    unsigned  B4SOIpaigsdGiven   :1;
    unsigned  B4SOIpbigsdGiven   :1;
    unsigned  B4SOIpcigsdGiven   :1;
    unsigned  B4SOIpnigcGiven   :1;
    unsigned  B4SOIppigcdGiven   :1;
    unsigned  B4SOIppoxedgeGiven   :1;

    unsigned  B4SOIpnpeakGiven   :1;
    unsigned  B4SOIpnsubGiven   :1;
    unsigned  B4SOIpngateGiven   :1;
    unsigned  B4SOIpnsdGiven     :1;
    unsigned  B4SOIpvth0Given   :1;
    unsigned  B4SOIpvfbGiven   :1;  /* v4.1 */
    unsigned  B4SOIpk1Given   :1;
    unsigned  B4SOIpk1w1Given   :1;
    unsigned  B4SOIpk1w2Given   :1;
    unsigned  B4SOIpk2Given   :1;
    unsigned  B4SOIpk3Given   :1;
    unsigned  B4SOIpk3bGiven   :1;
    unsigned  B4SOIpkb1Given   :1;
    unsigned  B4SOIpw0Given   :1;
    unsigned  B4SOIplpe0Given   :1;
    unsigned  B4SOIpdvt0Given   :1;
    unsigned  B4SOIpdvt1Given   :1;
    unsigned  B4SOIpdvt2Given   :1;
    unsigned  B4SOIpdvt0wGiven   :1;
    unsigned  B4SOIpdvt1wGiven   :1;
    unsigned  B4SOIpdvt2wGiven   :1;
    unsigned  B4SOIpu0Given   :1;
    unsigned  B4SOIpuaGiven   :1;
    unsigned  B4SOIpubGiven   :1;
    unsigned  B4SOIpucGiven   :1;
    unsigned  B4SOIpvsatGiven   :1;
    unsigned  B4SOIpa0Given   :1;
    unsigned  B4SOIpagsGiven   :1;
    unsigned  B4SOIpb0Given   :1;
    unsigned  B4SOIpb1Given   :1;
    unsigned  B4SOIpketaGiven   :1;
    unsigned  B4SOIpketasGiven   :1;
    unsigned  B4SOIpa1Given   :1;
    unsigned  B4SOIpa2Given   :1;
    unsigned  B4SOIprdswGiven   :1;
    unsigned  B4SOIprdwGiven   :1;      /* v4.0 for rdsMod = 1 */
    unsigned  B4SOIprswGiven   :1;      /* v4.0 for rdsMod = 1 */
    unsigned  B4SOIpprwbGiven   :1;
    unsigned  B4SOIpprwgGiven   :1;
    unsigned  B4SOIpwrGiven   :1;
    unsigned  B4SOIpnfactorGiven   :1;
    unsigned  B4SOIpdwgGiven   :1;
    unsigned  B4SOIpdwbGiven   :1;
    unsigned  B4SOIpvoffGiven   :1;
    unsigned  B4SOIpeta0Given   :1;
    unsigned  B4SOIpetabGiven   :1;
    unsigned  B4SOIpdsubGiven   :1;
    unsigned  B4SOIpcitGiven   :1;
    unsigned  B4SOIpcdscGiven   :1;
    unsigned  B4SOIpcdscbGiven   :1;
    unsigned  B4SOIpcdscdGiven   :1;
    unsigned  B4SOIppclmGiven   :1;
    unsigned  B4SOIppdibl1Given   :1;
    unsigned  B4SOIppdibl2Given   :1;
    unsigned  B4SOIppdiblbGiven   :1;
    unsigned  B4SOIpdroutGiven   :1;
    unsigned  B4SOIppvagGiven   :1;
    unsigned  B4SOIpdeltaGiven   :1;
    unsigned  B4SOIpalpha0Given   :1;
    unsigned  B4SOIpfbjtiiGiven   :1;
        /*4.1 Iii model*/
        unsigned  B4SOIpebjtiiGiven :1;
        unsigned  B4SOIpcbjtiiGiven :1;
        unsigned  B4SOIpvbciGiven   :1;
        unsigned  B4SOIpabjtiiGiven :1;
        unsigned  B4SOIpmbjtiiGiven :1;

    unsigned  B4SOIpbeta0Given   :1;
    unsigned  B4SOIpbeta1Given   :1;
    unsigned  B4SOIpbeta2Given   :1;
    unsigned  B4SOIpvdsatii0Given   :1;
    unsigned  B4SOIpliiGiven   :1;
    unsigned  B4SOIpesatiiGiven   :1;
    unsigned  B4SOIpsii0Given   :1;
    unsigned  B4SOIpsii1Given   :1;
    unsigned  B4SOIpsii2Given   :1;
    unsigned  B4SOIpsiidGiven   :1;
    unsigned  B4SOIpagidlGiven   :1;
    unsigned  B4SOIpbgidlGiven   :1;
    unsigned  B4SOIpcgidlGiven   :1;
    unsigned  B4SOIpegidlGiven   :1;
        unsigned  B4SOIprgidlGiven   :1;
        unsigned  B4SOIpkgidlGiven   :1;
        unsigned  B4SOIpfgidlGiven   :1;
        unsigned  B4SOIpagislGiven   :1;
    unsigned  B4SOIpbgislGiven   :1;
    unsigned  B4SOIpcgislGiven   :1;
    unsigned  B4SOIpegislGiven   :1;
        unsigned  B4SOIprgislGiven   :1;
        unsigned  B4SOIpkgislGiven   :1;
        unsigned  B4SOIpfgislGiven   :1;
    unsigned  B4SOIpntunGiven   :1;     /* v4.0 */
    unsigned  B4SOIpntundGiven   :1;    /* v4.0 */
    unsigned  B4SOIpndiodeGiven   :1;   /* v4.0 */
    unsigned  B4SOIpndiodedGiven   :1;  /* v4.0 */
    unsigned  B4SOIpnrecf0Given   :1;   /* v4.0 */
    unsigned  B4SOIpnrecf0dGiven   :1;  /* v4.0 */
    unsigned  B4SOIpnrecr0Given   :1;   /* v4.0 */
    unsigned  B4SOIpnrecr0dGiven   :1;  /* v4.0 */
    unsigned  B4SOIpisbjtGiven   :1;
    unsigned  B4SOIpidbjtGiven   :1;    /* v4.0 */
    unsigned  B4SOIpisdifGiven   :1;
    unsigned  B4SOIpiddifGiven   :1;    /* v4.0 */
    unsigned  B4SOIpisrecGiven   :1;
    unsigned  B4SOIpidrecGiven   :1;    /* v4.0 */
    unsigned  B4SOIpistunGiven   :1;
    unsigned  B4SOIpidtunGiven   :1;    /* v4.0 */
    unsigned  B4SOIpvrec0Given   :1;   /* v4.0 */
    unsigned  B4SOIpvrec0dGiven   :1;   /* v4.0 */
    unsigned  B4SOIpvtun0Given   :1;   /* v4.0 */
    unsigned  B4SOIpvtun0dGiven   :1;   /* v4.0 */
    unsigned  B4SOIpnbjtGiven   :1;
    unsigned  B4SOIplbjt0Given   :1;
    unsigned  B4SOIpvabjtGiven   :1;
    unsigned  B4SOIpaelyGiven   :1;
    unsigned  B4SOIpahliGiven  :1;      /* v4.0 */
    unsigned  B4SOIpahlidGiven  :1;     /* v4.0 */
/* v3.1 added for RF */
    unsigned B4SOIpxrcrg1Given  :1;
    unsigned B4SOIpxrcrg2Given  :1;
/* v3.1 added for RF end */

    /* CV model */
    unsigned  B4SOIpvsdfbGiven   :1;
    unsigned  B4SOIpvsdthGiven   :1;
    unsigned  B4SOIpdelvtGiven   :1;
    unsigned  B4SOIpacdeGiven   :1;
    unsigned  B4SOIpmoinGiven   :1;
    unsigned  B4SOIpnoffGiven   :1;/* v3.2 */

/* Added for binning - END2 */

    unsigned  B4SOIuseFringeGiven   :1;

    unsigned  B4SOItnomGiven   :1;
    unsigned  B4SOIcgsoGiven   :1;
    unsigned  B4SOIcgdoGiven   :1;
    unsigned  B4SOIcgeoGiven   :1;

    unsigned  B4SOIxpartGiven   :1;
    unsigned  B4SOIsheetResistanceGiven   :1;
    unsigned  B4SOIGatesidewallJctSPotentialGiven   :1;         /* v4.0 */
    unsigned  B4SOIGatesidewallJctDPotentialGiven   :1;         /* v4.0 */
    unsigned  B4SOIbodyJctGateSideSGradingCoeffGiven   :1;      /* v4.0 */
    unsigned  B4SOIbodyJctGateSideDGradingCoeffGiven   :1;      /* v4.0 */
    unsigned  B4SOIunitLengthGateSidewallJctCapSGiven  :1;      /* v4.0 */
    unsigned  B4SOIunitLengthGateSidewallJctCapDGiven  :1;      /* v4.0 */
    unsigned  B4SOIcsdeswGiven :1;

    unsigned  B4SOIoxideTrapDensityAGiven  :1;
    unsigned  B4SOIoxideTrapDensityBGiven  :1;
    unsigned  B4SOIoxideTrapDensityCGiven  :1;
    unsigned  B4SOIemGiven  :1;
    unsigned  B4SOIefGiven  :1;
    unsigned  B4SOIafGiven  :1;
    unsigned  B4SOIkfGiven  :1;
    unsigned  B4SOInoifGiven  :1;
    unsigned  B4SOIbfGiven  :1;       /* v4.0 */
    unsigned  B4SOIw0flkGiven  :1;    /* v4.0 */
    unsigned  B4SOIfrbodyGiven  :1;   /* v4.0 */

    unsigned  B4SOILintGiven   :1;
    unsigned  B4SOILlGiven   :1;
    unsigned  B4SOILlcGiven   :1; /* v2.2.3 */
    unsigned  B4SOILlnGiven   :1;
    unsigned  B4SOILwGiven   :1;
    unsigned  B4SOILwcGiven   :1; /* v2.2.3 */
    unsigned  B4SOILwnGiven   :1;
    unsigned  B4SOILwlGiven   :1;
    unsigned  B4SOILwlcGiven   :1; /* v2.2.3 */
    unsigned  B4SOILminGiven   :1;
    unsigned  B4SOILmaxGiven   :1;

    unsigned  B4SOIWintGiven   :1;
    unsigned  B4SOIWlGiven   :1;
    unsigned  B4SOIWlcGiven   :1; /* v2.2.3 */
    unsigned  B4SOIWlnGiven   :1;
    unsigned  B4SOIWwGiven   :1;
    unsigned  B4SOIWwcGiven   :1; /* v2.2.3 */
    unsigned  B4SOIWwnGiven   :1;
    unsigned  B4SOIWwlGiven   :1;
    unsigned  B4SOIWwlcGiven   :1;  /* v2.2.3 */
    unsigned  B4SOIWminGiven   :1;
    unsigned  B4SOIWmaxGiven   :1;

    /* added for stress effect */
    unsigned  B4SOIsarefGiven   :1;
    unsigned  B4SOIsbrefGiven   :1;
    unsigned  B4SOIwlodGiven  :1;
    unsigned  B4SOIku0Given   :1;
    unsigned  B4SOIkvsatGiven  :1;
    unsigned  B4SOIkvth0Given  :1;
    unsigned  B4SOItku0Given   :1;
    unsigned  B4SOIllodku0Given   :1;
    unsigned  B4SOIwlodku0Given   :1;
    unsigned  B4SOIllodvthGiven   :1;
    unsigned  B4SOIwlodvthGiven   :1;
    unsigned  B4SOIlku0Given   :1;
    unsigned  B4SOIwku0Given   :1;
    unsigned  B4SOIpku0Given   :1;
    unsigned  B4SOIlkvth0Given   :1;
    unsigned  B4SOIwkvth0Given   :1;
    unsigned  B4SOIpkvth0Given   :1;
    unsigned  B4SOIstk2Given   :1;
    unsigned  B4SOIlodk2Given  :1;
    unsigned  B4SOIsteta0Given :1;
    unsigned  B4SOIlodeta0Given :1;
    /* v4.0 added for stress effect end */
    unsigned  B4SOIlpebGiven    :1; /* v4.0 for vth */
    unsigned  B4SOIllpebGiven   :1; /* v4.0 for vth */
    unsigned  B4SOIwlpebGiven   :1; /* v4.0 for vth */
    unsigned  B4SOIplpebGiven   :1; /* v4.0 for vth */
    unsigned  B4SOIdvtp0Given   :1; /* v4.0 for vth */
    unsigned  B4SOIldvtp0Given  :1; /* v4.0 for vth */
    unsigned  B4SOIwdvtp0Given  :1; /* v4.0 for vth */
    unsigned  B4SOIpdvtp0Given  :1; /* v4.0 for vth */
    unsigned  B4SOIdvtp1Given   :1; /* v4.0 for vth */
    unsigned  B4SOIldvtp1Given  :1; /* v4.0 for vth */
    unsigned  B4SOIwdvtp1Given  :1; /* v4.0 for vth */
    unsigned  B4SOIpdvtp1Given  :1; /* v4.0 for vth */
    unsigned  B4SOIdvtp2Given   :1; /* v4.1 for vth */
    unsigned  B4SOIldvtp2Given  :1; /* v4.1 for vth */
    unsigned  B4SOIwdvtp2Given  :1; /* v4.1 for vth */
    unsigned  B4SOIpdvtp2Given  :1; /* v4.1 for vth */
    unsigned  B4SOIdvtp3Given   :1; /* v4.1 for vth */
    unsigned  B4SOIldvtp3Given  :1; /* v4.1 for vth */
    unsigned  B4SOIwdvtp3Given  :1; /* v4.1 for vth */
    unsigned  B4SOIpdvtp3Given  :1; /* v4.1 for vth */
    unsigned  B4SOIdvtp4Given   :1; /* v4.1 for vth */
    unsigned  B4SOIldvtp4Given  :1; /* v4.1 for vth */
    unsigned  B4SOIwdvtp4Given  :1; /* v4.1 for vth */
    unsigned  B4SOIpdvtp4Given  :1; /* v4.1 for vth */
    unsigned  B4SOIminvGiven    :1; /* v4.0 for Vgsteff */
    unsigned  B4SOIlminvGiven   :1; /* v4.0 for Vgsteff */
    unsigned  B4SOIwminvGiven   :1; /* v4.0 for Vgsteff */
    unsigned  B4SOIpminvGiven   :1; /* v4.0 for Vgsteff */
    unsigned  B4SOIfproutGiven  :1; /* v4.0 for DITS in ID */
    unsigned  B4SOIlfproutGiven  :1; /* v4.0 for DITS in ID */
    unsigned  B4SOIwfproutGiven  :1; /* v4.0 for DITS in ID */
    unsigned  B4SOIpfproutGiven  :1; /* v4.0 for DITS in ID */
    unsigned  B4SOIpditsGiven    :1; /* v4.0 for DITS in ID */
    unsigned  B4SOIlpditsGiven   :1; /* v4.0 for DITS in ID */
    unsigned  B4SOIwpditsGiven   :1; /* v4.0 for DITS in ID */
    unsigned  B4SOIppditsGiven   :1; /* v4.0 for DITS in ID */
    unsigned  B4SOIpditsdGiven   :1; /* v4.0 for DITS in ID */
    unsigned  B4SOIlpditsdGiven  :1; /* v4.0 for DITS in ID */
    unsigned  B4SOIwpditsdGiven  :1; /* v4.0 for DITS in ID */
    unsigned  B4SOIppditsdGiven  :1; /* v4.0 for DITS in ID */
    unsigned  B4SOIpditslGiven   :1; /* v4.0 for DITS in ID */

} B4SOImodel;


#ifndef NMOS
#define NMOS 1
#define PMOS -1
#endif /*NMOS*/


/* device parameters */
#define B4SOI_W 1
#define B4SOI_L 2
#define B4SOI_M 47
#define B4SOI_AS 3
#define B4SOI_AD 4
#define B4SOI_PS 5
#define B4SOI_PD 6
#define B4SOI_NRS 7
#define B4SOI_NRD 8
#define B4SOI_OFF 9
#define B4SOI_IC_VBS 10
#define B4SOI_IC_VDS 11
#define B4SOI_IC_VGS 12
#define B4SOI_IC_VES 13
#define B4SOI_IC_VPS 14
#define B4SOI_BJTOFF 15
#define B4SOI_RTH0 16
#define B4SOI_CTH0 17
#define B4SOI_NRB 18
#define B4SOI_IC 19
#define B4SOI_NQSMOD 20
#define B4SOI_DEBUG  21

/* v2.0 release */
#define B4SOI_NBC    22
#define B4SOI_NSEG   23
#define B4SOI_PDBCP  24
#define B4SOI_PSBCP  25
#define B4SOI_AGBCP  26
#define B4SOI_AEBCP  27
#define B4SOI_VBSUSR 28
#define B4SOI_TNODEOUT 29
#define B4SOI_FRBODY   30       /* v2.2.2 */
#define B4SOI_RGATEMOD 31       /* v3.1 */
#define B4SOI_SOIMOD   32       /* v3.2 */
#define B4SOI_NF       33       /* v4.0 */
#define B4SOI_RBODYMOD 34       /* v4.0 */
#define B4SOI_RBDB     35       /* v4.0 */
#define B4SOI_RBSB     36       /* v4.0 */
#define B4SOI_CJSB     37       /* v4.0 */
#define B4SOI_CJDB     38       /* v4.0 */
#define B4SOI_SA       41       /* v4.0 */
#define B4SOI_SB       42       /* v4.0 */
#define B4SOI_SD       43       /* v4.0 */
#define B4SOI_DELVTO   44       /* v4.0 */
#define B4SOI_AGBCPD   45       /* v4.0 */
#define B4SOI_AGBCP2   46       /* v4.1 improvement on BC */


/* model parameters */


#define B4SOI_MOD_PFGIDL           65
#define B4SOI_MOD_WFGIDL           66
#define B4SOI_MOD_LFGIDL           67
#define B4SOI_MOD_FGIDL            68
#define B4SOI_MOD_PKGIDL           69
#define B4SOI_MOD_WKGIDL           70
#define B4SOI_MOD_LKGIDL           71
#define B4SOI_MOD_KGIDL            72
#define B4SOI_MOD_PRGIDL           73
#define B4SOI_MOD_WRGIDL           74
#define B4SOI_MOD_LRGIDL           75
#define B4SOI_MOD_RGIDL            76
#define B4SOI_MOD_GIDLMOD          77
#define B4SOI_MOD_PNSD             78
#define B4SOI_MOD_WNSD             79
#define B4SOI_MOD_LNSD             80
#define B4SOI_MOD_NSD              81
#define B4SOI_MOD_PHIG             82
#define B4SOI_MOD_EASUB            83
#define B4SOI_MOD_TBGBSUB          84
#define B4SOI_MOD_TBGASUB          85
#define B4SOI_MOD_BG0SUB           86
#define B4SOI_MOD_NI0SUB           87
#define B4SOI_MOD_EPSRGATE         88
#define B4SOI_MOD_ADOS             89
#define B4SOI_MOD_BDOS             90
#define B4SOI_MOD_LEFFEOT          91
#define B4SOI_MOD_WEFFEOT          92
#define B4SOI_MOD_VDDEOT           93
#define B4SOI_MOD_TEMPEOT          94
#define B4SOI_MOD_TOXP             95
#define B4SOI_MOD_ETSI             96
#define B4SOI_MOD_EOT              97
#define B4SOI_MOD_EPSROX           98
#define B4SOI_MOD_EPSRSUB          99
#define B4SOI_MOD_MTRLMOD         100 /*4.1*/
#define B4SOI_MOD_CAPMOD          101
#define B4SOI_MOD_NQSMOD          102
#define B4SOI_MOD_MOBMOD          103
/*#define B4SOI_MOD_NOIMOD          104   v3.2 */
#define B4SOI_MOD_RDSMOD          104   /* v4.0 */
#define B4SOI_MOD_SHMOD           105
/*#define B4SOI_MOD_DDMOD           106   v4.2 ddmod is not used any more*/
#define B4SOI_MOD_TOX             107

#define B4SOI_MOD_CDSC            108
#define B4SOI_MOD_CDSCB           109
#define B4SOI_MOD_CIT             110
#define B4SOI_MOD_NFACTOR         111
#define B4SOI_MOD_XJ              112
#define B4SOI_MOD_VSAT            113
#define B4SOI_MOD_AT              114
#define B4SOI_MOD_A0              115
#define B4SOI_MOD_A1              116
#define B4SOI_MOD_A2              117
#define B4SOI_MOD_KETA            118
#define B4SOI_MOD_NSUB            119
#define B4SOI_MOD_NPEAK           120
#define B4SOI_MOD_NGATE           121
#define B4SOI_MOD_GAMMA1          122
#define B4SOI_MOD_GAMMA2          123
#define B4SOI_MOD_VBX             124
#define B4SOI_MOD_BINUNIT         125

#define B4SOI_MOD_VBM             126

#define B4SOI_MOD_XT              127
#define B4SOI_MOD_K1              129
#define B4SOI_MOD_KT1             130
#define B4SOI_MOD_KT1L            131
#define B4SOI_MOD_K2              132
#define B4SOI_MOD_KT2             133
#define B4SOI_MOD_K3              134
#define B4SOI_MOD_K3B             135
#define B4SOI_MOD_W0              136
#define B4SOI_MOD_LPE0            137

#define B4SOI_MOD_DVT0            138
#define B4SOI_MOD_DVT1            139
#define B4SOI_MOD_DVT2            140

#define B4SOI_MOD_DVT0W           141
#define B4SOI_MOD_DVT1W           142
#define B4SOI_MOD_DVT2W           143

#define B4SOI_MOD_DROUT           144
#define B4SOI_MOD_DSUB            145
#define B4SOI_MOD_VTH0            146
#define B4SOI_MOD_UA              147
#define B4SOI_MOD_UA1             148
#define B4SOI_MOD_UB              149
#define B4SOI_MOD_UB1             150
#define B4SOI_MOD_UC              151
#define B4SOI_MOD_UC1             152
#define B4SOI_MOD_U0              153
#define B4SOI_MOD_UTE             154
#define B4SOI_MOD_VOFF            155
#define B4SOI_MOD_DELTA           156
#define B4SOI_MOD_RDSW            157
#define B4SOI_MOD_PRT             158
#define B4SOI_MOD_LDD             159
#define B4SOI_MOD_ETA             160
#define B4SOI_MOD_ETA0            161
#define B4SOI_MOD_ETAB            162
#define B4SOI_MOD_PCLM            163
#define B4SOI_MOD_PDIBL1          164
#define B4SOI_MOD_PDIBL2          165
#define B4SOI_MOD_PSCBE1          166
#define B4SOI_MOD_PSCBE2          167
#define B4SOI_MOD_PVAG            168
#define B4SOI_MOD_WR              169
#define B4SOI_MOD_DWG             170
#define B4SOI_MOD_DWB             171
#define B4SOI_MOD_B0              172
#define B4SOI_MOD_B1              173
#define B4SOI_MOD_ALPHA0          174
#define B4SOI_MOD_PDIBLB          178

#define B4SOI_MOD_PRWG            179
#define B4SOI_MOD_PRWB            180

#define B4SOI_MOD_CDSCD           181
#define B4SOI_MOD_AGS             182

#define B4SOI_MOD_FRINGE          184
#define B4SOI_MOD_CGSL            186
#define B4SOI_MOD_CGDL            187
#define B4SOI_MOD_CKAPPA          188
#define B4SOI_MOD_CF              189
#define B4SOI_MOD_CLC             190
#define B4SOI_MOD_CLE             191
#define B4SOI_MOD_PARAMCHK        192
#define B4SOI_MOD_VERSION         193

#define B4SOI_MOD_TBOX            195
#define B4SOI_MOD_TSI             196
#define B4SOI_MOD_KB1             197
#define B4SOI_MOD_KB3             198
#define B4SOI_MOD_DELP            201
#define B4SOI_MOD_RBODY           204
#define B4SOI_MOD_ADICE0          205
#define B4SOI_MOD_ABP             206
#define B4SOI_MOD_MXC             207
#define B4SOI_MOD_RTH0            208
#define B4SOI_MOD_CTH0            209
#define B4SOI_MOD_ALPHA1          214
#define B4SOI_MOD_EGIDL           215
#define B4SOI_MOD_AGIDL           216
#define B4SOI_MOD_BGIDL           217
#define B4SOI_MOD_NDIODES         218   /* v4.0 */
#define B4SOI_MOD_LDIOF           219
#define B4SOI_MOD_LDIOR           220
#define B4SOI_MOD_NTUNS           221   /* v4.0 */
#define B4SOI_MOD_ISBJT           222
#define B4SOI_MOD_ISDIF           223
#define B4SOI_MOD_ISREC           224
#define B4SOI_MOD_ISTUN           225
#define B4SOI_MOD_XBJT            226
#define B4SOI_MOD_TT              232
#define B4SOI_MOD_VSDTH           233
#define B4SOI_MOD_VSDFB           234
#define B4SOI_MOD_ASD             235
#define B4SOI_MOD_CSDMIN          236
#define B4SOI_MOD_RBSH            237
#define B4SOI_MOD_ESATII          238


/* v2.0 release */
#define B4SOI_MOD_K1W1            239
#define B4SOI_MOD_K1W2            240
#define B4SOI_MOD_KETAS           241
#define B4SOI_MOD_DWBC            242
#define B4SOI_MOD_BETA0           243
#define B4SOI_MOD_BETA1           244
#define B4SOI_MOD_BETA2           245
#define B4SOI_MOD_VDSATII0        246
#define B4SOI_MOD_TII             247
#define B4SOI_MOD_LII             248
#define B4SOI_MOD_SII0            249
#define B4SOI_MOD_SII1            250
#define B4SOI_MOD_SII2            251
#define B4SOI_MOD_SIID            252
#define B4SOI_MOD_FBJTII          253
#define B4SOI_MOD_NRECF0S         255
#define B4SOI_MOD_NRECR0S         256
#define B4SOI_MOD_LN              257
#define B4SOI_MOD_VREC0S          258
#define B4SOI_MOD_VTUN0S          259
#define B4SOI_MOD_NBJT            260
#define B4SOI_MOD_LBJT0           261
#define B4SOI_MOD_VABJT           262
#define B4SOI_MOD_AELY            263
#define B4SOI_MOD_AHLIS           264   /* v4.0 */
#define B4SOI_MOD_NTRECF          265
#define B4SOI_MOD_NTRECR          266
#define B4SOI_MOD_DLCB            267
#define B4SOI_MOD_FBODY           268
#define B4SOI_MOD_NDIF            269
#define B4SOI_MOD_ACDE            272
#define B4SOI_MOD_MOIN            273
#define B4SOI_MOD_DELVT           274
#define B4SOI_MOD_DLBG            275
#define B4SOI_MOD_LDIF0           276


/* v2.2 release */
#define B4SOI_MOD_WTH0            277
#define B4SOI_MOD_RHALO           278
#define B4SOI_MOD_NTOX            279
#define B4SOI_MOD_TOXREF          280
#define B4SOI_MOD_EBG             281
#define B4SOI_MOD_VEVB            282
#define B4SOI_MOD_ALPHAGB1        283
#define B4SOI_MOD_BETAGB1         284
#define B4SOI_MOD_VGB1            285
#define B4SOI_MOD_VECB            286
#define B4SOI_MOD_ALPHAGB2        287
#define B4SOI_MOD_BETAGB2         288
#define B4SOI_MOD_VGB2            289
#define B4SOI_MOD_TOXQM           290
#define B4SOI_MOD_IGBMOD          291 /* v3.0 */
#define B4SOI_MOD_VOXH            292
#define B4SOI_MOD_DELTAVOX        293
#define B4SOI_MOD_IGCMOD          294 /* v3.0 */

/* v3.1 added for RF */
#define B4SOI_MOD_RGATEMOD        295
#define B4SOI_MOD_XRCRG1          296
#define B4SOI_MOD_XRCRG2          297
#define B4SOI_MOD_RSHG            298
#define B4SOI_MOD_NGCON           299
/* v3.1 added for RF end */

#define B4SOI_MOD_RBODYMOD        300 /* v4.0 */


/* Added for binning - START3 */
/* Length dependence */
#define B4SOI_MOD_LNPEAK           301
#define B4SOI_MOD_LNSUB            302
#define B4SOI_MOD_LNGATE           303
#define B4SOI_MOD_LVTH0            304
#define B4SOI_MOD_LK1              305
#define B4SOI_MOD_LK1W1            306
#define B4SOI_MOD_LK1W2            307
#define B4SOI_MOD_LK2              308
#define B4SOI_MOD_LK3              309
#define B4SOI_MOD_LK3B             310
#define B4SOI_MOD_LKB1             311
#define B4SOI_MOD_LW0              312
#define B4SOI_MOD_LLPE0            313
#define B4SOI_MOD_LDVT0            314
#define B4SOI_MOD_LDVT1            315
#define B4SOI_MOD_LDVT2            316
#define B4SOI_MOD_LDVT0W           317
#define B4SOI_MOD_LDVT1W           318
#define B4SOI_MOD_LDVT2W           319
#define B4SOI_MOD_LU0              320
#define B4SOI_MOD_LUA              321
#define B4SOI_MOD_LUB              322
#define B4SOI_MOD_LUC              323
#define B4SOI_MOD_LVSAT            324
#define B4SOI_MOD_LA0              325
#define B4SOI_MOD_LAGS             326
#define B4SOI_MOD_LB0              327
#define B4SOI_MOD_LB1              328
#define B4SOI_MOD_LKETA            329
#define B4SOI_MOD_LKETAS           330
#define B4SOI_MOD_LA1              331
#define B4SOI_MOD_LA2              332
#define B4SOI_MOD_LRDSW            333
#define B4SOI_MOD_LPRWB            334
#define B4SOI_MOD_LPRWG            335
#define B4SOI_MOD_LWR              336
#define B4SOI_MOD_LNFACTOR         337
#define B4SOI_MOD_LDWG             338
#define B4SOI_MOD_LDWB             339
#define B4SOI_MOD_LVOFF            340
#define B4SOI_MOD_LETA0            341
#define B4SOI_MOD_LETAB            342
#define B4SOI_MOD_LDSUB            343
#define B4SOI_MOD_LCIT             344
#define B4SOI_MOD_LCDSC            345
#define B4SOI_MOD_LCDSCB           346
#define B4SOI_MOD_LCDSCD           347
#define B4SOI_MOD_LPCLM            348
#define B4SOI_MOD_LPDIBL1          349
#define B4SOI_MOD_LPDIBL2          350
#define B4SOI_MOD_LPDIBLB          351
#define B4SOI_MOD_LDROUT           352
#define B4SOI_MOD_LPVAG            353
#define B4SOI_MOD_LDELTA           354
#define B4SOI_MOD_LALPHA0          355
#define B4SOI_MOD_LFBJTII          356
#define B4SOI_MOD_LBETA0           357
#define B4SOI_MOD_LBETA1           358
#define B4SOI_MOD_LBETA2           359
#define B4SOI_MOD_LVDSATII0        360
#define B4SOI_MOD_LLII             361
#define B4SOI_MOD_LESATII          362
#define B4SOI_MOD_LSII0            363
#define B4SOI_MOD_LSII1            364
#define B4SOI_MOD_LSII2            365
#define B4SOI_MOD_LSIID            366
#define B4SOI_MOD_LAGIDL           367
#define B4SOI_MOD_LBGIDL           368
#define B4SOI_MOD_LEGIDL           369
#define B4SOI_MOD_LNTUNS           370  /* v4.0 */
#define B4SOI_MOD_LNDIODES         371
#define B4SOI_MOD_LNRECF0S         372
#define B4SOI_MOD_LNRECR0S         373
#define B4SOI_MOD_LISBJT           374
#define B4SOI_MOD_LISDIF           375
#define B4SOI_MOD_LISREC           376
#define B4SOI_MOD_LISTUN           377
#define B4SOI_MOD_LVREC0S          378
#define B4SOI_MOD_LVTUN0S          379
#define B4SOI_MOD_LNBJT            380
#define B4SOI_MOD_LLBJT0           381
#define B4SOI_MOD_LVABJT           382
#define B4SOI_MOD_LAELY            383
#define B4SOI_MOD_LAHLIS           384  /* v4.0 */
#define B4SOI_MOD_LVSDFB           385
#define B4SOI_MOD_LVSDTH           386
#define B4SOI_MOD_LDELVT           387
#define B4SOI_MOD_LACDE            388
#define B4SOI_MOD_LMOIN            389

/* v3.1 added for RF */
#define B4SOI_MOD_LXRCRG1         390
#define B4SOI_MOD_LXRCRG2         391
#define B4SOI_MOD_XGW     392
#define B4SOI_MOD_XGL     393
/* v3.1 added for RF end */
#define B4SOI_MOD_CFRCOEFF        394   /* v4.4 */

/* Width dependence */
#define B4SOI_MOD_WNPEAK           401
#define B4SOI_MOD_WNSUB            402
#define B4SOI_MOD_WNGATE           403
#define B4SOI_MOD_WVTH0            404
#define B4SOI_MOD_WK1              405
#define B4SOI_MOD_WK1W1            406
#define B4SOI_MOD_WK1W2            407
#define B4SOI_MOD_WK2              408
#define B4SOI_MOD_WK3              409
#define B4SOI_MOD_WK3B             410
#define B4SOI_MOD_WKB1             411
#define B4SOI_MOD_WW0              412
#define B4SOI_MOD_WLPE0            413
#define B4SOI_MOD_WDVT0            414
#define B4SOI_MOD_WDVT1            415
#define B4SOI_MOD_WDVT2            416
#define B4SOI_MOD_WDVT0W           417
#define B4SOI_MOD_WDVT1W           418
#define B4SOI_MOD_WDVT2W           419
#define B4SOI_MOD_WU0              420
#define B4SOI_MOD_WUA              421
#define B4SOI_MOD_WUB              422
#define B4SOI_MOD_WUC              423
#define B4SOI_MOD_WVSAT            424
#define B4SOI_MOD_WA0              425
#define B4SOI_MOD_WAGS             426
#define B4SOI_MOD_WB0              427
#define B4SOI_MOD_WB1              428
#define B4SOI_MOD_WKETA            429
#define B4SOI_MOD_WKETAS           430
#define B4SOI_MOD_WA1              431
#define B4SOI_MOD_WA2              432
#define B4SOI_MOD_WRDSW            433
#define B4SOI_MOD_WPRWB            434
#define B4SOI_MOD_WPRWG            435
#define B4SOI_MOD_WWR              436
#define B4SOI_MOD_WNFACTOR         437
#define B4SOI_MOD_WDWG             438
#define B4SOI_MOD_WDWB             439
#define B4SOI_MOD_WVOFF            440
#define B4SOI_MOD_WETA0            441
#define B4SOI_MOD_WETAB            442
#define B4SOI_MOD_WDSUB            443
#define B4SOI_MOD_WCIT             444
#define B4SOI_MOD_WCDSC            445
#define B4SOI_MOD_WCDSCB           446
#define B4SOI_MOD_WCDSCD           447
#define B4SOI_MOD_WPCLM            448
#define B4SOI_MOD_WPDIBL1          449
#define B4SOI_MOD_WPDIBL2          450
#define B4SOI_MOD_WPDIBLB          451
#define B4SOI_MOD_WDROUT           452
#define B4SOI_MOD_WPVAG            453
#define B4SOI_MOD_WDELTA           454
#define B4SOI_MOD_WALPHA0          455
#define B4SOI_MOD_WFBJTII          456
#define B4SOI_MOD_WBETA0           457
#define B4SOI_MOD_WBETA1           458
#define B4SOI_MOD_WBETA2           459
#define B4SOI_MOD_WVDSATII0        460
#define B4SOI_MOD_WLII             461
#define B4SOI_MOD_WESATII          462
#define B4SOI_MOD_WSII0            463
#define B4SOI_MOD_WSII1            464
#define B4SOI_MOD_WSII2            465
#define B4SOI_MOD_WSIID            466
#define B4SOI_MOD_WAGIDL           467
#define B4SOI_MOD_WBGIDL           468
#define B4SOI_MOD_WEGIDL           469
#define B4SOI_MOD_WNTUNS           470  /* v4.0 */
#define B4SOI_MOD_WNDIODES         471
#define B4SOI_MOD_WNRECF0S         472
#define B4SOI_MOD_WNRECR0S         473
#define B4SOI_MOD_WISBJT           474
#define B4SOI_MOD_WISDIF           475
#define B4SOI_MOD_WISREC           476
#define B4SOI_MOD_WISTUN           477
#define B4SOI_MOD_WVREC0S          478
#define B4SOI_MOD_WVTUN0S          479
#define B4SOI_MOD_WNBJT            480
#define B4SOI_MOD_WLBJT0           481
#define B4SOI_MOD_WVABJT           482
#define B4SOI_MOD_WAELY            483
#define B4SOI_MOD_WAHLIS           484  /* v4.0 */
#define B4SOI_MOD_WVSDFB           485
#define B4SOI_MOD_WVSDTH           486
#define B4SOI_MOD_WDELVT           487
#define B4SOI_MOD_WACDE            488
#define B4SOI_MOD_WMOIN            489

/* v3.1 added for RF */
#define B4SOI_MOD_WXRCRG1          490
#define B4SOI_MOD_WXRCRG2          491
/* v3.1 added for RF end */

/* Cross-term dependence */

#define B4SOI_MOD_PNPEAK           501
#define B4SOI_MOD_PNSUB            502
#define B4SOI_MOD_PNGATE           503
#define B4SOI_MOD_PVTH0            504
#define B4SOI_MOD_PK1              505
#define B4SOI_MOD_PK1W1            506
#define B4SOI_MOD_PK1W2            507
#define B4SOI_MOD_PK2              508
#define B4SOI_MOD_PK3              509
#define B4SOI_MOD_PK3B             510
#define B4SOI_MOD_PKB1             511
#define B4SOI_MOD_PW0              512
#define B4SOI_MOD_PLPE0            513
#define B4SOI_MOD_PDVT0            514
#define B4SOI_MOD_PDVT1            515
#define B4SOI_MOD_PDVT2            516
#define B4SOI_MOD_PDVT0W           517
#define B4SOI_MOD_PDVT1W           518
#define B4SOI_MOD_PDVT2W           519
#define B4SOI_MOD_PU0              520
#define B4SOI_MOD_PUA              521
#define B4SOI_MOD_PUB              522
#define B4SOI_MOD_PUC              523
#define B4SOI_MOD_PVSAT            524
#define B4SOI_MOD_PA0              525
#define B4SOI_MOD_PAGS             526
#define B4SOI_MOD_PB0              527
#define B4SOI_MOD_PB1              528
#define B4SOI_MOD_PKETA            529
#define B4SOI_MOD_PKETAS           530
#define B4SOI_MOD_PA1              531
#define B4SOI_MOD_PA2              532
#define B4SOI_MOD_PRDSW            533
#define B4SOI_MOD_PPRWB            534
#define B4SOI_MOD_PPRWG            535
#define B4SOI_MOD_PWR              536
#define B4SOI_MOD_PNFACTOR         537
#define B4SOI_MOD_PDWG             538
#define B4SOI_MOD_PDWB             539
#define B4SOI_MOD_PVOFF            540
#define B4SOI_MOD_PETA0            541
#define B4SOI_MOD_PETAB            542
#define B4SOI_MOD_PDSUB            543
#define B4SOI_MOD_PCIT             544
#define B4SOI_MOD_PCDSC            545
#define B4SOI_MOD_PCDSCB           546
#define B4SOI_MOD_PCDSCD           547
#define B4SOI_MOD_PPCLM            548
#define B4SOI_MOD_PPDIBL1          549
#define B4SOI_MOD_PPDIBL2          550
#define B4SOI_MOD_PPDIBLB          551
#define B4SOI_MOD_PDROUT           552
#define B4SOI_MOD_PPVAG            553
#define B4SOI_MOD_PDELTA           554
#define B4SOI_MOD_PALPHA0          555
#define B4SOI_MOD_PFBJTII          556
#define B4SOI_MOD_PBETA0           557
#define B4SOI_MOD_PBETA1           558
#define B4SOI_MOD_PBETA2           559
#define B4SOI_MOD_PVDSATII0        560
#define B4SOI_MOD_PLII             561
#define B4SOI_MOD_PESATII          562
#define B4SOI_MOD_PSII0            563
#define B4SOI_MOD_PSII1            564
#define B4SOI_MOD_PSII2            565
#define B4SOI_MOD_PSIID            566
#define B4SOI_MOD_PAGIDL           567
#define B4SOI_MOD_PBGIDL           568
#define B4SOI_MOD_PEGIDL           569
#define B4SOI_MOD_PNTUNS           570  /* v4.0 */
#define B4SOI_MOD_PNDIODES         571
#define B4SOI_MOD_PNRECF0S         572
#define B4SOI_MOD_PNRECR0S         573
#define B4SOI_MOD_PISBJT           574
#define B4SOI_MOD_PISDIF           575
#define B4SOI_MOD_PISREC           576
#define B4SOI_MOD_PISTUN           577
#define B4SOI_MOD_PVREC0S          578
#define B4SOI_MOD_PVTUN0S          579
#define B4SOI_MOD_PNBJT            580
#define B4SOI_MOD_PLBJT0           581
#define B4SOI_MOD_PVABJT           582
#define B4SOI_MOD_PAELY            583
#define B4SOI_MOD_PAHLIS           584  /* v4.0 */
#define B4SOI_MOD_PVSDFB           585
#define B4SOI_MOD_PVSDTH           586
#define B4SOI_MOD_PDELVT           587
#define B4SOI_MOD_PACDE            588
#define B4SOI_MOD_PMOIN            589
#define B4SOI_MOD_PXRCRG1          590  /* v3.1 for RF */
#define B4SOI_MOD_PXRCRG2          591  /* v3.1 for RF */
#define B4SOI_MOD_EM               592  /* v3.2 for noise */
#define B4SOI_MOD_EF               593  /* v3.2 for noise */
#define B4SOI_MOD_AF               594  /* v3.2 for noise */
#define B4SOI_MOD_KF               595  /* v3.2 for noise */
#define B4SOI_MOD_NOIF             596  /* v3.2 for noise */
#define B4SOI_MOD_BF               597  /* v4.0 for noise */
#define B4SOI_MOD_W0FLK            598  /* v4.0 for noise */
#define B4SOI_MOD_FRBODY           599  /* v4.0 for Rbody */
#define B4SOI_MOD_CGIDL            600  /* v4.0 for gidl */
#define B4SOI_MOD_LCGIDL           601  /* v4.0 for gidl */
#define B4SOI_MOD_WCGIDL           602  /* v4.0 for gidl */
#define B4SOI_MOD_PCGIDL           603  /* v4.0 for gidl */
#define B4SOI_MOD_LPEB             604  /* v4.0 for Vth */
#define B4SOI_MOD_LLPEB            605  /* v4.0 for Vth */
#define B4SOI_MOD_WLPEB            606  /* v4.0 for Vth */
#define B4SOI_MOD_PLPEB            607  /* v4.0 for Vth */
#define B4SOI_MOD_DVTP0            608  /* v4.0 for Vth */
#define B4SOI_MOD_LDVTP0           609  /* v4.0 for Vth */
#define B4SOI_MOD_WDVTP0           610  /* v4.0 for Vth */
#define B4SOI_MOD_PDVTP0           611  /* v4.0 for Vth */
#define B4SOI_MOD_DVTP1            612  /* v4.0 for Vth */
#define B4SOI_MOD_LDVTP1           613  /* v4.0 for Vth */
#define B4SOI_MOD_WDVTP1           614  /* v4.0 for Vth */
#define B4SOI_MOD_PDVTP1           615  /* v4.0 for Vth */
#define B4SOI_MOD_MINV             616  /* v4.0 for Vgsteff */
#define B4SOI_MOD_LMINV            617  /* v4.0 for Vgsteff */
#define B4SOI_MOD_WMINV            618  /* v4.0 for Vgsteff */
#define B4SOI_MOD_PMINV            619  /* v4.0 for Vgsteff */
#define B4SOI_MOD_FPROUT           620  /* v4.0 for DITS in Id */
#define B4SOI_MOD_LFPROUT          621  /* v4.0 for DITS in Id */
#define B4SOI_MOD_WFPROUT          622  /* v4.0 for DITS in Id */
#define B4SOI_MOD_PFPROUT          623  /* v4.0 for DITS in Id */
#define B4SOI_MOD_PDITS            624  /* v4.0 for DITS in Id */
#define B4SOI_MOD_LPDITS           625  /* v4.0 for DITS in Id */
#define B4SOI_MOD_WPDITS           626  /* v4.0 for DITS in Id */
#define B4SOI_MOD_PPDITS           627  /* v4.0 for DITS in Id */
#define B4SOI_MOD_PDITSD           628  /* v4.0 for DITS in Id */
#define B4SOI_MOD_LPDITSD          629  /* v4.0 for DITS in Id */
#define B4SOI_MOD_WPDITSD          630  /* v4.0 for DITS in Id */
#define B4SOI_MOD_PPDITSD          631  /* v4.0 for DITS in Id */
#define B4SOI_MOD_PDITSL           632  /* v4.0 for DITS in Id */
#define B4SOI_MOD_NDIODED          633  /* v4.0 */
#define B4SOI_MOD_LNDIODED         634  /* v4.0 */
#define B4SOI_MOD_WNDIODED         635  /* v4.0 */
#define B4SOI_MOD_PNDIODED         636  /* v4.0 */
#define B4SOI_MOD_IDDIF            637  /* v4.0 */
#define B4SOI_MOD_LIDDIF           638  /* v4.0 */
#define B4SOI_MOD_WIDDIF           639  /* v4.0 */
#define B4SOI_MOD_PIDDIF           640  /* v4.0 */
#define B4SOI_MOD_IDBJT            641  /* v4.0 */
#define B4SOI_MOD_LIDBJT           642  /* v4.0 */
#define B4SOI_MOD_WIDBJT           643  /* v4.0 */
#define B4SOI_MOD_PIDBJT           644  /* v4.0 */
#define B4SOI_MOD_IDREC            645  /* v4.0 */
#define B4SOI_MOD_LIDREC           646  /* v4.0 */
#define B4SOI_MOD_WIDREC           647  /* v4.0 */
#define B4SOI_MOD_PIDREC           648  /* v4.0 */
#define B4SOI_MOD_IDTUN            649  /* v4.0 */
#define B4SOI_MOD_LIDTUN           650  /* v4.0 */
#define B4SOI_MOD_WIDTUN           651  /* v4.0 */
#define B4SOI_MOD_PIDTUN           652  /* v4.0 */
#define B4SOI_MOD_NRECF0D          653  /* v4.0 */
#define B4SOI_MOD_LNRECF0D         654  /* v4.0 */
#define B4SOI_MOD_WNRECF0D         655  /* v4.0 */
#define B4SOI_MOD_PNRECF0D         656  /* v4.0 */
#define B4SOI_MOD_NRECR0D          657  /* v4.0 */
#define B4SOI_MOD_LNRECR0D         658  /* v4.0 */
#define B4SOI_MOD_WNRECR0D         659  /* v4.0 */
#define B4SOI_MOD_PNRECR0D         660  /* v4.0 */
#define B4SOI_MOD_VREC0D           661  /* v4.0 */
#define B4SOI_MOD_LVREC0D          662  /* v4.0 */
#define B4SOI_MOD_WVREC0D          663  /* v4.0 */
#define B4SOI_MOD_PVREC0D          664  /* v4.0 */
#define B4SOI_MOD_VTUN0D           665  /* v4.0 */
#define B4SOI_MOD_LVTUN0D          666  /* v4.0 */
#define B4SOI_MOD_WVTUN0D          667  /* v4.0 */
#define B4SOI_MOD_PVTUN0D          668  /* v4.0 */
#define B4SOI_MOD_NTUND            669  /* v4.0 */
#define B4SOI_MOD_LNTUND           670  /* v4.0 */
#define B4SOI_MOD_WNTUND           671  /* v4.0 */
#define B4SOI_MOD_PNTUND           672  /* v4.0 */
#define B4SOI_MOD_RDW              673  /* v4.0 for rdsMod =1 */
#define B4SOI_MOD_LRDW             674  /* v4.0 for rdsMod =1 */
#define B4SOI_MOD_WRDW             675  /* v4.0 for rdsMod =1 */
#define B4SOI_MOD_PRDW             676  /* v4.0 for rdsMod =1 */
#define B4SOI_MOD_RSW              677  /* v4.0 for rdsMod =1 */
#define B4SOI_MOD_LRSW             678  /* v4.0 for rdsMod =1 */
#define B4SOI_MOD_WRSW             679  /* v4.0 for rdsMod =1 */
#define B4SOI_MOD_PRSW             680  /* v4.0 for rdsMod =1 */
#define B4SOI_MOD_RDWMIN           681  /* v4.0 for rdsMod =1 */
#define B4SOI_MOD_RSWMIN           682  /* v4.0 for rdsMod =1 */
#define B4SOI_MOD_AHLID            683  /* v4.0 */
#define B4SOI_MOD_LAHLID           684  /* v4.0 */
#define B4SOI_MOD_WAHLID           685  /* v4.0 */
#define B4SOI_MOD_PAHLID           686  /* v4.0 */
#define B4SOI_MOD_DVTP2            687  /* v4.1 for Vth */
#define B4SOI_MOD_LDVTP2           688  /* v4.1 for Vth */
#define B4SOI_MOD_WDVTP2           689  /* v4.1 for Vth */
#define B4SOI_MOD_PDVTP2           690  /* v4.1 for Vth */
#define B4SOI_MOD_DVTP3            691  /* v4.1 for Vth */
#define B4SOI_MOD_LDVTP3           692  /* v4.1 for Vth */
#define B4SOI_MOD_WDVTP3           693  /* v4.1 for Vth */
#define B4SOI_MOD_PDVTP3           694  /* v4.1 for Vth */
#define B4SOI_MOD_DVTP4            695  /* v4.1 for Vth */
#define B4SOI_MOD_LDVTP4           696  /* v4.1 for Vth */
#define B4SOI_MOD_WDVTP4           697  /* v4.1 for Vth */
#define B4SOI_MOD_PDVTP4           698  /* v4.1 for Vth */
#define B4SOI_MOD_AIGBCP2        10001  /* v4.1 for Ig in AGBCP2 Region */
#define B4SOI_MOD_LAIGBCP2       10002  /* v4.1 for Ig in AGBCP2 Region */
#define B4SOI_MOD_WAIGBCP2       10003  /* v4.1 for Ig in AGBCP2 Region */
#define B4SOI_MOD_PAIGBCP2       10004  /* v4.1 for Ig in AGBCP2 Region */
#define B4SOI_MOD_BIGBCP2        10005  /* v4.1 for Ig in AGBCP2 Region */
#define B4SOI_MOD_LBIGBCP2       10006  /* v4.1 for Ig in AGBCP2 Region */
#define B4SOI_MOD_WBIGBCP2       10007  /* v4.1 for Ig in AGBCP2 Region */
#define B4SOI_MOD_PBIGBCP2       10008  /* v4.1 for Ig in AGBCP2 Region */
#define B4SOI_MOD_CIGBCP2        10009  /* v4.1 for Ig in AGBCP2 Region */
#define B4SOI_MOD_LCIGBCP2       10010  /* v4.1 for Ig in AGBCP2 Region */
#define B4SOI_MOD_WCIGBCP2       10011  /* v4.1 for Ig in AGBCP2 Region */
#define B4SOI_MOD_PCIGBCP2       10012  /* v4.1 for Ig in AGBCP2 Region */


/* Added for binning - END3 */

#define B4SOI_MOD_TNOM             701
#define B4SOI_MOD_CGSO             702
#define B4SOI_MOD_CGDO             703
#define B4SOI_MOD_CGEO             704
#define B4SOI_MOD_XPART            705
#define B4SOI_MOD_RSH              706


#define B4SOI_MOD_NMOS             814
#define B4SOI_MOD_PMOS             815

#define B4SOI_MOD_NOIA             816
#define B4SOI_MOD_NOIB             817
#define B4SOI_MOD_NOIC             818

#define B4SOI_MOD_LINT             819
#define B4SOI_MOD_LL               820
#define B4SOI_MOD_LLN              821
#define B4SOI_MOD_LW               822
#define B4SOI_MOD_LWN              823
#define B4SOI_MOD_LWL              824

#define B4SOI_MOD_WINT             827
#define B4SOI_MOD_WL               828
#define B4SOI_MOD_WLN              829
#define B4SOI_MOD_WW               830
#define B4SOI_MOD_WWN              831
#define B4SOI_MOD_WWL              832

/* v2.2.3 */
#define B4SOI_MOD_LWLC             841
#define B4SOI_MOD_LLC              842
#define B4SOI_MOD_LWC              843
#define B4SOI_MOD_WWLC             844
#define B4SOI_MOD_WLC              845
#define B4SOI_MOD_WWC              846
#define B4SOI_MOD_DTOXCV           847

#define B4SOI_MOD_DWC              848
#define B4SOI_MOD_DLC              849



#define B4SOI_MOD_PBSWGS           860  /* v4.0 */
#define B4SOI_MOD_MJSWGS           861  /* v4.0 */
#define B4SOI_MOD_CJSWGS           862  /* v4.0 */
#define B4SOI_MOD_CSDESW           863

#define B4SOI_MOD_XDIFS                         870
#define B4SOI_MOD_XRECS                 871
#define B4SOI_MOD_XTUNS                 872
#define B4SOI_MOD_XDIFD                         873
#define B4SOI_MOD_XRECD                 874
#define B4SOI_MOD_XTUND                 875
#define B4SOI_MOD_LXDIFS                        876
#define B4SOI_MOD_LXRECS                        877
#define B4SOI_MOD_LXTUNS                        878
#define B4SOI_MOD_LXDIFD                        879
#define B4SOI_MOD_LXRECD                        880
#define B4SOI_MOD_LXTUND                        881
#define B4SOI_MOD_WXDIFS                        882
#define B4SOI_MOD_WXRECS                        883
#define B4SOI_MOD_WXTUNS                        884
#define B4SOI_MOD_WXDIFD                        885
#define B4SOI_MOD_WXRECD                        886
#define B4SOI_MOD_WXTUND                        887
#define B4SOI_MOD_PXDIFS                        888
#define B4SOI_MOD_PXRECS                        889
#define B4SOI_MOD_PXTUNS                        890
#define B4SOI_MOD_PXDIFD                        891
#define B4SOI_MOD_PXRECD                        892
#define B4SOI_MOD_PXTUND                        893
#define B4SOI_MOD_TCJSWGS          894
#define B4SOI_MOD_TPBSWGS          895
#define B4SOI_MOD_TCJSWGD          896
#define B4SOI_MOD_TPBSWGD          897

/* device questions */
#define B4SOI_DNODE                901
#define B4SOI_GNODE                902
#define B4SOI_SNODE                903
#define B4SOI_BNODE                904
#define B4SOI_ENODE                905
#define B4SOI_DNODEPRIME           906
#define B4SOI_SNODEPRIME           907
#define B4SOI_VBD                  908
#define B4SOI_VBS                  909
#define B4SOI_VGS                  910
#define B4SOI_VES                  911
#define B4SOI_VDS                  912
#define B4SOI_CD                   913
#define B4SOI_CBS                  914
#define B4SOI_CBD                  915
#define B4SOI_GM                   916
#define B4SOI_GDS                  917
#define B4SOI_GMBS                 918
#define B4SOI_GBD                  919
#define B4SOI_GBS                  920
#define B4SOI_QB                   921
#define B4SOI_CQB                  922
#define B4SOI_QG                   923
#define B4SOI_CQG                  924
#define B4SOI_QD                   925
#define B4SOI_CQD                  926
#define B4SOI_CGG                  927
#define B4SOI_CGD                  928
#define B4SOI_CGS                  929
#define B4SOI_CBG                  930
#define B4SOI_CAPBD                931
#define B4SOI_CQBD                 932
#define B4SOI_CAPBS                933
#define B4SOI_CQBS                 934
#define B4SOI_CDG                  935
#define B4SOI_CDD                  936
#define B4SOI_CDS                  937
#define B4SOI_VON                  938
#define B4SOI_VDSAT                939
#define B4SOI_QBS                  940
#define B4SOI_QBD                  941
#define B4SOI_SOURCECONDUCT        942
#define B4SOI_DRAINCONDUCT         943
#define B4SOI_CBDB                 944
#define B4SOI_CBSB                 945
#define B4SOI_GMID                 946
#define B4SOI_QS                   955

/* For debugging only */
#define B4SOI_DEBUG1               956
#define B4SOI_DEBUG2               957
#define B4SOI_DEBUG3               958
/* End debugging */

/* v3.1 added for RF */
#define B4SOI_GNODEEXT             947
#define B4SOI_GNODEMID             948
/* v3.1 added for RF end */

/* v4.0 */
#define B4SOI_DBNODE               949
#define B4SOI_SBNODE               950
/* v4.0 end */

/*4.1 mobmod=4*/
/*#define B4SOI_VGSTEFFVTH          3300*/
/*#define B4SOI_VTFBPHI1            3301*/
/*#define B4SOI_EG                  3350*/

#define B4SOI_MOD_UD              3400
#define B4SOI_MOD_LUD             3401
#define B4SOI_MOD_WUD             3402
#define B4SOI_MOD_PUD             3403
#define B4SOI_MOD_UD1             3404
#define B4SOI_MOD_LUD1            3405
#define B4SOI_MOD_WUD1            3406
#define B4SOI_MOD_PUD1            3407
#define B4SOI_MOD_EU              3500
#define B4SOI_MOD_LEU             3501
#define B4SOI_MOD_WEU             3502
#define B4SOI_MOD_PEU             3503
#define B4SOI_MOD_UCS             3504
#define B4SOI_MOD_LUCS            3505
#define B4SOI_MOD_WUCS            3506
#define B4SOI_MOD_PUCS            3507
#define B4SOI_MOD_UCSTE           3508
#define B4SOI_MOD_LUCSTE          3509
#define B4SOI_MOD_WUCSTE          3510
#define B4SOI_MOD_PUCSTE          3511




/*4.1 Iii model*/
#define B4SOI_MOD_IIIMOD          4000
#define B4SOI_MOD_TVBCI           4001
#define B4SOI_MOD_EBJTII          4002
#define B4SOI_MOD_CBJTII          4003
#define B4SOI_MOD_VBCI            4004
#define B4SOI_MOD_ABJTII          4005
#define B4SOI_MOD_MBJTII          4006
#define B4SOI_MOD_LEBJTII         4007
#define B4SOI_MOD_LCBJTII         4008
#define B4SOI_MOD_LVBCI           4009
#define B4SOI_MOD_LABJTII         4010
#define B4SOI_MOD_LMBJTII         4011
#define B4SOI_MOD_WEBJTII         4012
#define B4SOI_MOD_WCBJTII         4013
#define B4SOI_MOD_WVBCI           4014
#define B4SOI_MOD_WABJTII         4015
#define B4SOI_MOD_WMBJTII         4016
#define B4SOI_MOD_PEBJTII         4017
#define B4SOI_MOD_PCBJTII         4018
#define B4SOI_MOD_PVBCI           4019
#define B4SOI_MOD_PABJTII         4020
#define B4SOI_MOD_PMBJTII         4021


#define B4SOI_MOD_EGISL           2500
#define B4SOI_MOD_AGISL           2501
#define B4SOI_MOD_BGISL           2502
#define B4SOI_MOD_CGISL           2503
#define B4SOI_MOD_RGISL           2504
#define B4SOI_MOD_KGISL           2505
#define B4SOI_MOD_FGISL           2506
#define B4SOI_MOD_LEGISL          2507
#define B4SOI_MOD_WEGISL          2508
#define B4SOI_MOD_PEGISL          2509
#define B4SOI_MOD_LAGISL          2510
#define B4SOI_MOD_WAGISL          2511
#define B4SOI_MOD_PAGISL          2512
#define B4SOI_MOD_LBGISL          2513
#define B4SOI_MOD_WBGISL          2514
#define B4SOI_MOD_PBGISL          2515
#define B4SOI_MOD_LCGISL          2516
#define B4SOI_MOD_WCGISL          2517
#define B4SOI_MOD_PCGISL          2518
#define B4SOI_MOD_LRGISL          2519
#define B4SOI_MOD_WRGISL          2520
#define B4SOI_MOD_PRGISL          2521
#define B4SOI_MOD_LKGISL          2522
#define B4SOI_MOD_WKGISL          2523
#define B4SOI_MOD_PKGISL          2524
#define B4SOI_MOD_LFGISL          2525
#define B4SOI_MOD_WFGISL          2526
#define B4SOI_MOD_PFGISL          2527

#define B4SOI_IGISL                3001

#define B4SOI_IBS                  3002
#define B4SOI_IBD                  3003
#define B4SOI_ISUB                 3004
#define B4SOI_IGIDL                3005
#define B4SOI_IGS                  3006
#define B4SOI_IGD                  3007
#define B4SOI_IGB                  3008
#define B4SOI_IGCS                 3009
#define B4SOI_IGCD                 3010

/* v3.2 */
#define B4SOI_MOD_TNOIA            951
#define B4SOI_MOD_TNOIB            952
#define B4SOI_MOD_RNOIA            953
#define B4SOI_MOD_RNOIB            954
#define B4SOI_MOD_NTNOI            955
#define B4SOI_MOD_FNOIMOD          956
#define B4SOI_MOD_TNOIMOD          957
#define B4SOI_MOD_NOFF             958
#define B4SOI_MOD_LNOFF            959
#define B4SOI_MOD_WNOFF            960
#define B4SOI_MOD_PNOFF            961
#define B4SOI_MOD_TOXM             962
#define B4SOI_MOD_VBS0PD           963
#define B4SOI_MOD_VBS0FD           964
/* v3.2 */

/* v4.0 added for stress */
#define B4SOI_MOD_SAREF            965
#define B4SOI_MOD_SBREF            966
#define B4SOI_MOD_KU0              967
#define B4SOI_MOD_KVSAT            968
#define B4SOI_MOD_TKU0             969
#define B4SOI_MOD_LLODKU0          970
#define B4SOI_MOD_WLODKU0          971
#define B4SOI_MOD_LLODVTH          972
#define B4SOI_MOD_WLODVTH          973
#define B4SOI_MOD_LKU0             974
#define B4SOI_MOD_WKU0             975
#define B4SOI_MOD_PKU0             976
#define B4SOI_MOD_KVTH0            977
#define B4SOI_MOD_LKVTH0           978
#define B4SOI_MOD_WKVTH0           979
#define B4SOI_MOD_PKVTH0           980
#define B4SOI_MOD_WLOD             981
#define B4SOI_MOD_STK2             982
#define B4SOI_MOD_LODK2            983
#define B4SOI_MOD_STETA0           984
#define B4SOI_MOD_LODETA0          985
/* v4.0 added for stress end */
#define B4SOI_MOD_GBMIN            986 /* v4.0 */
#define B4SOI_MOD_RBDB             987 /* v4.0 */
#define B4SOI_MOD_RBSB             988 /* v4.0 */
#define B4SOI_MOD_MJSWGD           989 /* v4.0 */
#define B4SOI_MOD_CJSWGD           990  /* v4.0 */
#define B4SOI_MOD_PBSWGD           991  /* v4.0 */
/*4.1*/

#define B4SOI_MOD_VFB              1201  /* v4.1 */
#define B4SOI_MOD_LVFB             1202  /* v4.1 */
#define B4SOI_MOD_WVFB             1203  /* v4.1 */
#define B4SOI_MOD_PVFB             1204  /* v4.1 */

#define B4SOI_MOD_FDMOD            1221
#define B4SOI_MOD_VSCE             1222
#define B4SOI_MOD_CDSBS            1223
#define B4SOI_MOD_VGSTCVMOD        1224
#define B4SOI_MOD_MINVCV           1225
#define B4SOI_MOD_LMINVCV          1226
#define B4SOI_MOD_WMINVCV          1227
#define B4SOI_MOD_PMINVCV          1228
#define B4SOI_MOD_VOFFCV           1229
#define B4SOI_MOD_LVOFFCV          1230
#define B4SOI_MOD_WVOFFCV          1231
#define B4SOI_MOD_PVOFFCV          1232
/* v3.0 */
#define B4SOI_MOD_SOIMOD           1001
#define B4SOI_MOD_VBSA             1002
#define B4SOI_MOD_NOFFFD           1003
#define B4SOI_MOD_VOFFFD           1004
#define B4SOI_MOD_K1B              1005
#define B4SOI_MOD_K2B              1006
#define B4SOI_MOD_DK2B             1007
#define B4SOI_MOD_DVBD0            1008
#define B4SOI_MOD_DVBD1            1009
#define B4SOI_MOD_MOINFD           1010

/* v3.0 */
#define B4SOI_MOD_AIGC             1021
#define B4SOI_MOD_BIGC             1022
#define B4SOI_MOD_CIGC             1023
#define B4SOI_MOD_AIGSD            1024
#define B4SOI_MOD_BIGSD            1025
#define B4SOI_MOD_CIGSD            1026
#define B4SOI_MOD_NIGC             1027
#define B4SOI_MOD_PIGCD            1028
#define B4SOI_MOD_POXEDGE          1029
#define B4SOI_MOD_DLCIG            1030

#define B4SOI_MOD_LAIGC            1031
#define B4SOI_MOD_LBIGC            1032
#define B4SOI_MOD_LCIGC            1033
#define B4SOI_MOD_LAIGSD           1034
#define B4SOI_MOD_LBIGSD           1035
#define B4SOI_MOD_LCIGSD           1036
#define B4SOI_MOD_LNIGC            1037
#define B4SOI_MOD_LPIGCD           1038
#define B4SOI_MOD_LPOXEDGE         1039

#define B4SOI_MOD_WAIGC            1041
#define B4SOI_MOD_WBIGC            1042
#define B4SOI_MOD_WCIGC            1043
#define B4SOI_MOD_WAIGSD           1044
#define B4SOI_MOD_WBIGSD           1045
#define B4SOI_MOD_WCIGSD           1046
#define B4SOI_MOD_WNIGC            1047
#define B4SOI_MOD_WPIGCD           1048
#define B4SOI_MOD_WPOXEDGE         1049

#define B4SOI_MOD_PAIGC            1051
#define B4SOI_MOD_PBIGC            1052
#define B4SOI_MOD_PCIGC            1053
#define B4SOI_MOD_PAIGSD           1054
#define B4SOI_MOD_PBIGSD           1055
#define B4SOI_MOD_PCIGSD           1056
#define B4SOI_MOD_PNIGC            1057
#define B4SOI_MOD_PPIGCD           1058
#define B4SOI_MOD_PPOXEDGE         1059

/* v3.1 */
#define B4SOI_MOD_LXJ              1061
#define B4SOI_MOD_LALPHAGB1        1062
#define B4SOI_MOD_LALPHAGB2        1063
#define B4SOI_MOD_LBETAGB1         1064
#define B4SOI_MOD_LBETAGB2         1065
#define B4SOI_MOD_LNDIF            1066
#define B4SOI_MOD_LNTRECF          1067
#define B4SOI_MOD_LNTRECR          1068
#define B4SOI_MOD_LXBJT            1069
#define B4SOI_MOD_LCGDL            1073
#define B4SOI_MOD_LCGSL            1074
#define B4SOI_MOD_LCKAPPA          1075
#define B4SOI_MOD_LUTE             1078
#define B4SOI_MOD_LKT1             1079
#define B4SOI_MOD_LKT2             1080
#define B4SOI_MOD_LKT1L            1081
#define B4SOI_MOD_LUA1             1082
#define B4SOI_MOD_LUB1             1083
#define B4SOI_MOD_LUC1             1084
#define B4SOI_MOD_LAT              1085
#define B4SOI_MOD_LPRT             1086

#define B4SOI_MOD_WXJ              1091
#define B4SOI_MOD_WALPHAGB1        1092
#define B4SOI_MOD_WALPHAGB2        1093
#define B4SOI_MOD_WBETAGB1         1094
#define B4SOI_MOD_WBETAGB2         1095
#define B4SOI_MOD_WNDIF            1096
#define B4SOI_MOD_WNTRECF          1097
#define B4SOI_MOD_WNTRECR          1098
#define B4SOI_MOD_WXBJT            1099
#define B4SOI_MOD_WCGDL            2003
#define B4SOI_MOD_WCGSL            2004
#define B4SOI_MOD_WCKAPPA          2005
#define B4SOI_MOD_WUTE             2008
#define B4SOI_MOD_WKT1             2009
#define B4SOI_MOD_WKT2             2010
#define B4SOI_MOD_WKT1L            2011
#define B4SOI_MOD_WUA1             2012
#define B4SOI_MOD_WUB1             2013
#define B4SOI_MOD_WUC1             2014
#define B4SOI_MOD_WAT              2015
#define B4SOI_MOD_WPRT             2016

#define B4SOI_MOD_PXJ              2021
#define B4SOI_MOD_PALPHAGB1        2022
#define B4SOI_MOD_PALPHAGB2        2023
#define B4SOI_MOD_PBETAGB1         2024
#define B4SOI_MOD_PBETAGB2         2025
#define B4SOI_MOD_PNDIF            2026
#define B4SOI_MOD_PNTRECF          2027
#define B4SOI_MOD_PNTRECR          2028
#define B4SOI_MOD_PXBJT            2029
#define B4SOI_MOD_PCGDL            2033
#define B4SOI_MOD_PCGSL            2034
#define B4SOI_MOD_PCKAPPA          2035
#define B4SOI_MOD_PUTE             2038
#define B4SOI_MOD_PKT1             2039
#define B4SOI_MOD_PKT2             2040
#define B4SOI_MOD_PKT1L            2041
#define B4SOI_MOD_PUA1             2042
#define B4SOI_MOD_PUB1             2043
#define B4SOI_MOD_PUC1             2044
#define B4SOI_MOD_PAT              2045
#define B4SOI_MOD_PPRT             2046

/* 4.0 backward compatibility  */
#define B4SOI_MOD_NGIDL            2100
#define B4SOI_MOD_LNGIDL            2101
#define B4SOI_MOD_WNGIDL            2102
#define B4SOI_MOD_PNGIDL            2103
#define B4SOI_MOD_NLX            2104
#define B4SOI_MOD_LNLX            2105
#define B4SOI_MOD_WNLX            2106
#define B4SOI_MOD_PNLX            2107

#define B4SOI_MOD_VGS_MAX          2201
#define B4SOI_MOD_VGD_MAX          2202
#define B4SOI_MOD_VGB_MAX          2203
#define B4SOI_MOD_VDS_MAX          2204
#define B4SOI_MOD_VBS_MAX          2205
#define B4SOI_MOD_VBD_MAX          2206
#define B4SOI_MOD_VGSR_MAX         2207
#define B4SOI_MOD_VGDR_MAX         2208
#define B4SOI_MOD_VGBR_MAX         2209
#define B4SOI_MOD_VBSR_MAX         2210
#define B4SOI_MOD_VBDR_MAX         2211

#include "b4soiext.h"

extern void B4SOIevaluate(double,double,double,B4SOIinstance*,B4SOImodel*,
        double*,double*,double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, CKTcircuit*);
extern int B4SOIdebug(B4SOImodel*, B4SOIinstance*, CKTcircuit*, int);
extern int B4SOIcheckModel(B4SOImodel*, B4SOIinstance*, CKTcircuit*);

#endif /*B4SOI*/

