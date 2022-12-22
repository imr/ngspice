/**** BSIM4.6.2 Released by Wenwei Yang 07/31/2008 ****/
/**** OpenMP support for ngspice by Holger Vogt 06/28/2010 ****/
/**********
Copyright 2006 Regents of the University of California.  All rights reserved.
File: bsim4v6def.h
Author: 2000 Weidong Liu.
Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
Authors: 2006- Mohan Dunga, Ali Niknejad, Chenming Hu
Authors: 2007- Mohan Dunga, Wenwei Yang, Ali Niknejad, Chenming Hu
Authors: 2008- Wenwei Yang, Ali Niknejad, Chenming Hu 
Modified by Xuemei Xi, 11/15/2002.
Modified by Xuemei Xi, 05/09/2003.
Modified by Xuemei Xi, 03/04/2004.
Modified by Xuemei Xi, Mohan Dunga, 09/24/2004.
Modified by Xuemei Xi, 07/29/2005.
Modified by Mohan Dunga, 12/13/2006
Modified by Mohan Dunga, Wenwei Yang, 05/18/2007.
Modified by Wenwei Yang, 07/31/2008.
**********/

#ifndef BSIM4v6
#define BSIM4v6

#include "ngspice/ifsim.h"
#include "ngspice/gendefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/noisedef.h"         

typedef struct sBSIM4v6instance
{

    struct GENinstance gen;

#define BSIM4v6modPtr(inst) ((struct sBSIM4v6model *)((inst)->gen.GENmodPtr))
#define BSIM4v6nextInstance(inst) ((struct sBSIM4v6instance *)((inst)->gen.GENnextInstance))
#define BSIM4v6name gen.GENname
#define BSIM4v6states gen.GENstate

    const int BSIM4v6dNode;
    const int BSIM4v6gNodeExt;
    const int BSIM4v6sNode;
    const int BSIM4v6bNode;
    int BSIM4v6dNodePrime;
    int BSIM4v6gNodePrime;
    int BSIM4v6gNodeMid;
    int BSIM4v6sNodePrime;
    int BSIM4v6bNodePrime;
    int BSIM4v6dbNode;
    int BSIM4v6sbNode;
    int BSIM4v6qNode;

    double BSIM4v6ueff;
    double BSIM4v6thetavth; 
    double BSIM4v6von;
    double BSIM4v6vdsat;
    double BSIM4v6cgdo;
    double BSIM4v6qgdo;
    double BSIM4v6cgso;
    double BSIM4v6qgso;
    double BSIM4v6grbsb;
    double BSIM4v6grbdb;
    double BSIM4v6grbpb;
    double BSIM4v6grbps;
    double BSIM4v6grbpd;

    double BSIM4v6vjsmFwd;
    double BSIM4v6vjsmRev;
    double BSIM4v6vjdmFwd;
    double BSIM4v6vjdmRev;
    double BSIM4v6XExpBVS;
    double BSIM4v6XExpBVD;
    double BSIM4v6SslpFwd;
    double BSIM4v6SslpRev;
    double BSIM4v6DslpFwd;
    double BSIM4v6DslpRev;
    double BSIM4v6IVjsmFwd;
    double BSIM4v6IVjsmRev;
    double BSIM4v6IVjdmFwd;
    double BSIM4v6IVjdmRev;

    double BSIM4v6grgeltd;
    double BSIM4v6Pseff;
    double BSIM4v6Pdeff;
    double BSIM4v6Aseff;
    double BSIM4v6Adeff;

    double BSIM4v6l;
    double BSIM4v6w;
    double BSIM4v6drainArea;
    double BSIM4v6sourceArea;
    double BSIM4v6drainSquares;
    double BSIM4v6sourceSquares;
    double BSIM4v6drainPerimeter;
    double BSIM4v6sourcePerimeter;
    double BSIM4v6sourceConductance;
    double BSIM4v6drainConductance;
     /* stress effect instance param */
    double BSIM4v6sa;
    double BSIM4v6sb;
    double BSIM4v6sd;
    double BSIM4v6sca;
    double BSIM4v6scb;
    double BSIM4v6scc;
    double BSIM4v6sc;

    double BSIM4v6rbdb;
    double BSIM4v6rbsb;
    double BSIM4v6rbpb;
    double BSIM4v6rbps;
    double BSIM4v6rbpd;
    
    double BSIM4v6delvto;
    double BSIM4v6mulu0;
    double BSIM4v6xgw;
    double BSIM4v6ngcon;

     /* added here to account stress effect instance dependence */
    double BSIM4v6u0temp;
    double BSIM4v6vsattemp;
    double BSIM4v6vth0;
    double BSIM4v6vfb;
    double BSIM4v6vfbzb;
    double BSIM4v6vtfbphi1;
    double BSIM4v6vtfbphi2;
    double BSIM4v6k2;
    double BSIM4v6vbsc;
    double BSIM4v6k2ox;
    double BSIM4v6eta0;

    double BSIM4v6icVDS;
    double BSIM4v6icVGS;
    double BSIM4v6icVBS;
    double BSIM4v6m;
    double BSIM4v6nf;
    int BSIM4v6off;
    int BSIM4v6mode;
    int BSIM4v6trnqsMod;
    int BSIM4v6acnqsMod;
    int BSIM4v6rbodyMod;
    int BSIM4v6rgateMod;
    int BSIM4v6geoMod;
    int BSIM4v6rgeoMod;
    int BSIM4v6min;


    /* OP point */
    double BSIM4v6Vgsteff;
    double BSIM4v6vgs_eff;
    double BSIM4v6vgd_eff;
    double BSIM4v6dvgs_eff_dvg;
    double BSIM4v6dvgd_eff_dvg;
    double BSIM4v6Vdseff;
    double BSIM4v6nstar;
    double BSIM4v6Abulk;
    double BSIM4v6EsatL;
    double BSIM4v6AbovVgst2Vtm;
    double BSIM4v6qinv;
    double BSIM4v6cd;
    double BSIM4v6cbs;
    double BSIM4v6cbd;
    double BSIM4v6csub;
    double BSIM4v6Igidl;
    double BSIM4v6Igisl;
    double BSIM4v6gm;
    double BSIM4v6gds;
    double BSIM4v6gmbs;
    double BSIM4v6gbd;
    double BSIM4v6gbs;

    double BSIM4v6gbbs;
    double BSIM4v6gbgs;
    double BSIM4v6gbds;
    double BSIM4v6ggidld;
    double BSIM4v6ggidlg;
    double BSIM4v6ggidls;
    double BSIM4v6ggidlb;
    double BSIM4v6ggisld;
    double BSIM4v6ggislg;
    double BSIM4v6ggisls;
    double BSIM4v6ggislb;

    double BSIM4v6Igcs;
    double BSIM4v6gIgcsg;
    double BSIM4v6gIgcsd;
    double BSIM4v6gIgcss;
    double BSIM4v6gIgcsb;
    double BSIM4v6Igcd;
    double BSIM4v6gIgcdg;
    double BSIM4v6gIgcdd;
    double BSIM4v6gIgcds;
    double BSIM4v6gIgcdb;

    double BSIM4v6Igs;
    double BSIM4v6gIgsg;
    double BSIM4v6gIgss;
    double BSIM4v6Igd;
    double BSIM4v6gIgdg;
    double BSIM4v6gIgdd;

    double BSIM4v6Igb;
    double BSIM4v6gIgbg;
    double BSIM4v6gIgbd;
    double BSIM4v6gIgbs;
    double BSIM4v6gIgbb;

    double BSIM4v6grdsw;
    double BSIM4v6IdovVds;
    double BSIM4v6gcrg;
    double BSIM4v6gcrgd;
    double BSIM4v6gcrgg;
    double BSIM4v6gcrgs;
    double BSIM4v6gcrgb;

    double BSIM4v6gstot;
    double BSIM4v6gstotd;
    double BSIM4v6gstotg;
    double BSIM4v6gstots;
    double BSIM4v6gstotb;

    double BSIM4v6gdtot;
    double BSIM4v6gdtotd;
    double BSIM4v6gdtotg;
    double BSIM4v6gdtots;
    double BSIM4v6gdtotb;

    double BSIM4v6cggb;
    double BSIM4v6cgdb;
    double BSIM4v6cgsb;
    double BSIM4v6cbgb;
    double BSIM4v6cbdb;
    double BSIM4v6cbsb;
    double BSIM4v6cdgb;
    double BSIM4v6cddb;
    double BSIM4v6cdsb;
    double BSIM4v6csgb;
    double BSIM4v6csdb;
    double BSIM4v6cssb;
    double BSIM4v6cgbb;
    double BSIM4v6cdbb;
    double BSIM4v6csbb;
    double BSIM4v6cbbb;
    double BSIM4v6capbd;
    double BSIM4v6capbs;

    double BSIM4v6cqgb;
    double BSIM4v6cqdb;
    double BSIM4v6cqsb;
    double BSIM4v6cqbb;

    double BSIM4v6qgate;
    double BSIM4v6qbulk;
    double BSIM4v6qdrn;
    double BSIM4v6qsrc;
    double BSIM4v6qdef;

    double BSIM4v6qchqs;
    double BSIM4v6taunet;
    double BSIM4v6gtau;
    double BSIM4v6gtg;
    double BSIM4v6gtd;
    double BSIM4v6gts;
    double BSIM4v6gtb;
    double BSIM4v6SjctTempRevSatCur;
    double BSIM4v6DjctTempRevSatCur;
    double BSIM4v6SswTempRevSatCur;
    double BSIM4v6DswTempRevSatCur;
    double BSIM4v6SswgTempRevSatCur;
    double BSIM4v6DswgTempRevSatCur;

    struct bsim4v6SizeDependParam  *pParam;

    unsigned BSIM4v6lGiven :1;
    unsigned BSIM4v6wGiven :1;
    unsigned BSIM4v6mGiven :1;
    unsigned BSIM4v6nfGiven :1;
    unsigned BSIM4v6minGiven :1;
    unsigned BSIM4v6drainAreaGiven :1;
    unsigned BSIM4v6sourceAreaGiven    :1;
    unsigned BSIM4v6drainSquaresGiven  :1;
    unsigned BSIM4v6sourceSquaresGiven :1;
    unsigned BSIM4v6drainPerimeterGiven    :1;
    unsigned BSIM4v6sourcePerimeterGiven   :1;
    unsigned BSIM4v6saGiven :1;
    unsigned BSIM4v6sbGiven :1;
    unsigned BSIM4v6sdGiven :1;
    unsigned BSIM4v6scaGiven :1;
    unsigned BSIM4v6scbGiven :1;
    unsigned BSIM4v6sccGiven :1;
    unsigned BSIM4v6scGiven :1;
    unsigned BSIM4v6rbdbGiven   :1;
    unsigned BSIM4v6rbsbGiven   :1;
    unsigned BSIM4v6rbpbGiven   :1;
    unsigned BSIM4v6rbpdGiven   :1;
    unsigned BSIM4v6rbpsGiven   :1;
    unsigned BSIM4v6delvtoGiven   :1;
    unsigned BSIM4v6mulu0Given   :1;
    unsigned BSIM4v6xgwGiven   :1;
    unsigned BSIM4v6ngconGiven   :1;
    unsigned BSIM4v6icVDSGiven :1;
    unsigned BSIM4v6icVGSGiven :1;
    unsigned BSIM4v6icVBSGiven :1;
    unsigned BSIM4v6trnqsModGiven :1;
    unsigned BSIM4v6acnqsModGiven :1;
    unsigned BSIM4v6rbodyModGiven :1;
    unsigned BSIM4v6rgateModGiven :1;
    unsigned BSIM4v6geoModGiven :1;
    unsigned BSIM4v6rgeoModGiven :1;


    double *BSIM4v6DPdPtr;
    double *BSIM4v6DPdpPtr;
    double *BSIM4v6DPgpPtr;
    double *BSIM4v6DPgmPtr;
    double *BSIM4v6DPspPtr;
    double *BSIM4v6DPbpPtr;
    double *BSIM4v6DPdbPtr;

    double *BSIM4v6DdPtr;
    double *BSIM4v6DdpPtr;

    double *BSIM4v6GPdpPtr;
    double *BSIM4v6GPgpPtr;
    double *BSIM4v6GPgmPtr;
    double *BSIM4v6GPgePtr;
    double *BSIM4v6GPspPtr;
    double *BSIM4v6GPbpPtr;

    double *BSIM4v6GMdpPtr;
    double *BSIM4v6GMgpPtr;
    double *BSIM4v6GMgmPtr;
    double *BSIM4v6GMgePtr;
    double *BSIM4v6GMspPtr;
    double *BSIM4v6GMbpPtr;

    double *BSIM4v6GEdpPtr;
    double *BSIM4v6GEgpPtr;
    double *BSIM4v6GEgmPtr;
    double *BSIM4v6GEgePtr;
    double *BSIM4v6GEspPtr;
    double *BSIM4v6GEbpPtr;

    double *BSIM4v6SPdpPtr;
    double *BSIM4v6SPgpPtr;
    double *BSIM4v6SPgmPtr;
    double *BSIM4v6SPsPtr;
    double *BSIM4v6SPspPtr;
    double *BSIM4v6SPbpPtr;
    double *BSIM4v6SPsbPtr;

    double *BSIM4v6SspPtr;
    double *BSIM4v6SsPtr;

    double *BSIM4v6BPdpPtr;
    double *BSIM4v6BPgpPtr;
    double *BSIM4v6BPgmPtr;
    double *BSIM4v6BPspPtr;
    double *BSIM4v6BPdbPtr;
    double *BSIM4v6BPbPtr;
    double *BSIM4v6BPsbPtr;
    double *BSIM4v6BPbpPtr;

    double *BSIM4v6DBdpPtr;
    double *BSIM4v6DBdbPtr;
    double *BSIM4v6DBbpPtr;
    double *BSIM4v6DBbPtr;

    double *BSIM4v6SBspPtr;
    double *BSIM4v6SBbpPtr;
    double *BSIM4v6SBbPtr;
    double *BSIM4v6SBsbPtr;

    double *BSIM4v6BdbPtr;
    double *BSIM4v6BbpPtr;
    double *BSIM4v6BsbPtr;
    double *BSIM4v6BbPtr;

    double *BSIM4v6DgpPtr;
    double *BSIM4v6DspPtr;
    double *BSIM4v6DbpPtr;
    double *BSIM4v6SdpPtr;
    double *BSIM4v6SgpPtr;
    double *BSIM4v6SbpPtr;

    double *BSIM4v6QdpPtr;
    double *BSIM4v6QgpPtr;
    double *BSIM4v6QspPtr;
    double *BSIM4v6QbpPtr;
    double *BSIM4v6QqPtr;
    double *BSIM4v6DPqPtr;
    double *BSIM4v6GPqPtr;
    double *BSIM4v6SPqPtr;

#ifdef USE_OMP
    /* per instance storage of results, to update matrix at a later stge */
    double BSIM4v6rhsdPrime;
    double BSIM4v6rhsgPrime;
    double BSIM4v6rhsgExt;
    double BSIM4v6grhsMid;
    double BSIM4v6rhsbPrime;
    double BSIM4v6rhssPrime;
    double BSIM4v6rhsdb;
    double BSIM4v6rhssb;
    double BSIM4v6rhsd;
    double BSIM4v6rhss;
    double BSIM4v6rhsq;

    double BSIM4v6_1;
    double BSIM4v6_2;
    double BSIM4v6_3;
    double BSIM4v6_4;
    double BSIM4v6_5;
    double BSIM4v6_6;
    double BSIM4v6_7;
    double BSIM4v6_8;
    double BSIM4v6_9;
    double BSIM4v6_10;
    double BSIM4v6_11;
    double BSIM4v6_12;
    double BSIM4v6_13;
    double BSIM4v6_14;
    double BSIM4v6_15;
    double BSIM4v6_16;
    double BSIM4v6_17;
    double BSIM4v6_18;
    double BSIM4v6_19;
    double BSIM4v6_20;
    double BSIM4v6_21;
    double BSIM4v6_22;
    double BSIM4v6_23;
    double BSIM4v6_24;
    double BSIM4v6_25;
    double BSIM4v6_26;
    double BSIM4v6_27;
    double BSIM4v6_28;
    double BSIM4v6_29;
    double BSIM4v6_30;
    double BSIM4v6_31;
    double BSIM4v6_32;
    double BSIM4v6_33;
    double BSIM4v6_34;
    double BSIM4v6_35;
    double BSIM4v6_36;
    double BSIM4v6_37;
    double BSIM4v6_38;
    double BSIM4v6_39;
    double BSIM4v6_40;
    double BSIM4v6_41;
    double BSIM4v6_42;
    double BSIM4v6_43;
    double BSIM4v6_44;
    double BSIM4v6_45;
    double BSIM4v6_46;
    double BSIM4v6_47;
    double BSIM4v6_48;
    double BSIM4v6_49;
    double BSIM4v6_50;
    double BSIM4v6_51;
    double BSIM4v6_52;
    double BSIM4v6_53;
    double BSIM4v6_54;
    double BSIM4v6_55;
    double BSIM4v6_56;
    double BSIM4v6_57;
    double BSIM4v6_58;
    double BSIM4v6_59;
    double BSIM4v6_60;
    double BSIM4v6_61;
    double BSIM4v6_62;
    double BSIM4v6_63;
    double BSIM4v6_64;
    double BSIM4v6_65;
    double BSIM4v6_66;
    double BSIM4v6_67;
    double BSIM4v6_68;
    double BSIM4v6_69;
    double BSIM4v6_70;
    double BSIM4v6_71;
    double BSIM4v6_72;
    double BSIM4v6_73;
    double BSIM4v6_74;
    double BSIM4v6_75;
    double BSIM4v6_76;
    double BSIM4v6_77;
    double BSIM4v6_78;
    double BSIM4v6_79;
    double BSIM4v6_80;
    double BSIM4v6_81;
    double BSIM4v6_82;
    double BSIM4v6_83;
    double BSIM4v6_84;
    double BSIM4v6_85;
    double BSIM4v6_86;
    double BSIM4v6_87;
    double BSIM4v6_88;
    double BSIM4v6_89;
    double BSIM4v6_90;
    double BSIM4v6_91;
    double BSIM4v6_92;
    double BSIM4v6_93;
    double BSIM4v6_94;
    double BSIM4v6_95;
    double BSIM4v6_96;
    double BSIM4v6_97;
    double BSIM4v6_98;
    double BSIM4v6_99;
    double BSIM4v6_100;
    double BSIM4v6_101;
    double BSIM4v6_102;
    double BSIM4v6_103;

#endif

#define BSIM4v6vbd BSIM4v6states+ 0
#define BSIM4v6vbs BSIM4v6states+ 1
#define BSIM4v6vgs BSIM4v6states+ 2
#define BSIM4v6vds BSIM4v6states+ 3
#define BSIM4v6vdbs BSIM4v6states+ 4
#define BSIM4v6vdbd BSIM4v6states+ 5
#define BSIM4v6vsbs BSIM4v6states+ 6
#define BSIM4v6vges BSIM4v6states+ 7
#define BSIM4v6vgms BSIM4v6states+ 8
#define BSIM4v6vses BSIM4v6states+ 9
#define BSIM4v6vdes BSIM4v6states+ 10

#define BSIM4v6qb BSIM4v6states+ 11
#define BSIM4v6cqb BSIM4v6states+ 12
#define BSIM4v6qg BSIM4v6states+ 13
#define BSIM4v6cqg BSIM4v6states+ 14
#define BSIM4v6qd BSIM4v6states+ 15
#define BSIM4v6cqd BSIM4v6states+ 16
#define BSIM4v6qgmid BSIM4v6states+ 17
#define BSIM4v6cqgmid BSIM4v6states+ 18

#define BSIM4v6qbs  BSIM4v6states+ 19
#define BSIM4v6cqbs  BSIM4v6states+ 20
#define BSIM4v6qbd  BSIM4v6states+ 21
#define BSIM4v6cqbd  BSIM4v6states+ 22

#define BSIM4v6qcheq BSIM4v6states+ 23
#define BSIM4v6cqcheq BSIM4v6states+ 24
#define BSIM4v6qcdump BSIM4v6states+ 25
#define BSIM4v6cqcdump BSIM4v6states+ 26
#define BSIM4v6qdef BSIM4v6states+ 27
#define BSIM4v6qs BSIM4v6states+ 28

#define BSIM4v6numStates 29


/* indices to the array of BSIM4v6 NOISE SOURCES */

#define BSIM4v6RDNOIZ       0
#define BSIM4v6RSNOIZ       1
#define BSIM4v6RGNOIZ       2
#define BSIM4v6RBPSNOIZ     3
#define BSIM4v6RBPDNOIZ     4
#define BSIM4v6RBPBNOIZ     5
#define BSIM4v6RBSBNOIZ     6
#define BSIM4v6RBDBNOIZ     7
#define BSIM4v6IDNOIZ       8
#define BSIM4v6FLNOIZ       9
#define BSIM4v6IGSNOIZ      10
#define BSIM4v6IGDNOIZ      11
#define BSIM4v6IGBNOIZ      12
#define BSIM4v6TOTNOIZ      13

#define BSIM4v6NSRCS        14  /* Number of BSIM4v6 noise sources */

#ifndef NONOISE
    double BSIM4v6nVar[NSTATVARS][BSIM4v6NSRCS];
#else /* NONOISE */
        double **BSIM4v6nVar;
#endif /* NONOISE */

} BSIM4v6instance ;

struct bsim4v6SizeDependParam
{
    double Width;
    double Length;
    double NFinger;

    double BSIM4v6cdsc;           
    double BSIM4v6cdscb;    
    double BSIM4v6cdscd;       
    double BSIM4v6cit;           
    double BSIM4v6nfactor;      
    double BSIM4v6xj;
    double BSIM4v6vsat;         
    double BSIM4v6at;         
    double BSIM4v6a0;   
    double BSIM4v6ags;      
    double BSIM4v6a1;         
    double BSIM4v6a2;         
    double BSIM4v6keta;     
    double BSIM4v6nsub;
    double BSIM4v6ndep;        
    double BSIM4v6nsd;
    double BSIM4v6phin;
    double BSIM4v6ngate;        
    double BSIM4v6gamma1;      
    double BSIM4v6gamma2;     
    double BSIM4v6vbx;      
    double BSIM4v6vbi;       
    double BSIM4v6vbm;       
    double BSIM4v6xt;       
    double BSIM4v6phi;
    double BSIM4v6litl;
    double BSIM4v6k1;
    double BSIM4v6kt1;
    double BSIM4v6kt1l;
    double BSIM4v6kt2;
    double BSIM4v6k2;
    double BSIM4v6k3;
    double BSIM4v6k3b;
    double BSIM4v6w0;
    double BSIM4v6dvtp0;
    double BSIM4v6dvtp1;
    double BSIM4v6lpe0;
    double BSIM4v6lpeb;
    double BSIM4v6dvt0;      
    double BSIM4v6dvt1;      
    double BSIM4v6dvt2;      
    double BSIM4v6dvt0w;      
    double BSIM4v6dvt1w;      
    double BSIM4v6dvt2w;      
    double BSIM4v6drout;      
    double BSIM4v6dsub;      
    double BSIM4v6vth0;
    double BSIM4v6ua;
    double BSIM4v6ua1;
    double BSIM4v6ub;
    double BSIM4v6ub1;
    double BSIM4v6uc;
    double BSIM4v6uc1;
    double BSIM4v6ud;
    double BSIM4v6ud1;
    double BSIM4v6up;
    double BSIM4v6lp;
    double BSIM4v6u0;
    double BSIM4v6eu;
	double BSIM4v6ucs;
    double BSIM4v6ute;
	double BSIM4v6ucste;
    double BSIM4v6voff;
    double BSIM4v6tvoff;
    double BSIM4v6minv;
    double BSIM4v6minvcv;
    double BSIM4v6vfb;
    double BSIM4v6delta;
    double BSIM4v6rdsw;       
    double BSIM4v6rds0;       
    double BSIM4v6rs0;
    double BSIM4v6rd0;
    double BSIM4v6rsw;
    double BSIM4v6rdw;
    double BSIM4v6prwg;       
    double BSIM4v6prwb;       
    double BSIM4v6prt;       
    double BSIM4v6eta0;         
    double BSIM4v6etab;         
    double BSIM4v6pclm;      
    double BSIM4v6pdibl1;      
    double BSIM4v6pdibl2;      
    double BSIM4v6pdiblb;      
    double BSIM4v6fprout;
    double BSIM4v6pdits;
    double BSIM4v6pditsd;
    double BSIM4v6pscbe1;       
    double BSIM4v6pscbe2;       
    double BSIM4v6pvag;       
    double BSIM4v6wr;
    double BSIM4v6dwg;
    double BSIM4v6dwb;
    double BSIM4v6b0;
    double BSIM4v6b1;
    double BSIM4v6alpha0;
    double BSIM4v6alpha1;
    double BSIM4v6beta0;
    double BSIM4v6agidl;
    double BSIM4v6bgidl;
    double BSIM4v6cgidl;
    double BSIM4v6egidl;
    double BSIM4v6agisl;
    double BSIM4v6bgisl;
    double BSIM4v6cgisl;
    double BSIM4v6egisl;
    double BSIM4v6aigc;
    double BSIM4v6bigc;
    double BSIM4v6cigc;
    double BSIM4v6aigs;
    double BSIM4v6bigs;
    double BSIM4v6cigs;
    double BSIM4v6aigd;
    double BSIM4v6bigd;
    double BSIM4v6cigd;
    double BSIM4v6aigbacc;
    double BSIM4v6bigbacc;
    double BSIM4v6cigbacc;
    double BSIM4v6aigbinv;
    double BSIM4v6bigbinv;
    double BSIM4v6cigbinv;
    double BSIM4v6nigc;
    double BSIM4v6nigbacc;
    double BSIM4v6nigbinv;
    double BSIM4v6ntox;
    double BSIM4v6eigbinv;
    double BSIM4v6pigcd;
    double BSIM4v6poxedge;
    double BSIM4v6xrcrg1;
    double BSIM4v6xrcrg2;
    double BSIM4v6lambda; /* overshoot */
    double BSIM4v6vtl; /* thermal velocity limit */
    double BSIM4v6xn; /* back scattering parameter */
    double BSIM4v6lc; /* back scattering parameter */
    double BSIM4v6tfactor;  /* ballistic transportation factor  */
    double BSIM4v6vfbsdoff;  /* S/D flatband offset voltage  */
    double BSIM4v6tvfbsdoff;  

/* added for stress effect */
    double BSIM4v6ku0;
    double BSIM4v6kvth0;
    double BSIM4v6ku0temp;
    double BSIM4v6rho_ref;
    double BSIM4v6inv_od_ref;
/* added for well proximity effect */
    double BSIM4v6kvth0we;
    double BSIM4v6k2we;
    double BSIM4v6ku0we;

    /* CV model */
    double BSIM4v6cgsl;
    double BSIM4v6cgdl;
    double BSIM4v6ckappas;
    double BSIM4v6ckappad;
    double BSIM4v6cf;
    double BSIM4v6clc;
    double BSIM4v6cle;
    double BSIM4v6vfbcv;
    double BSIM4v6noff;
    double BSIM4v6voffcv;
    double BSIM4v6acde;
    double BSIM4v6moin;

/* Pre-calculated constants */

    double BSIM4v6dw;
    double BSIM4v6dl;
    double BSIM4v6leff;
    double BSIM4v6weff;

    double BSIM4v6dwc;
    double BSIM4v6dlc;
    double BSIM4v6dwj;
    double BSIM4v6leffCV;
    double BSIM4v6weffCV;
    double BSIM4v6weffCJ;
    double BSIM4v6abulkCVfactor;
    double BSIM4v6cgso;
    double BSIM4v6cgdo;
    double BSIM4v6cgbo;

    double BSIM4v6u0temp;       
    double BSIM4v6vsattemp;   
    double BSIM4v6sqrtPhi;   
    double BSIM4v6phis3;   
    double BSIM4v6Xdep0;          
    double BSIM4v6sqrtXdep0;          
    double BSIM4v6theta0vb0;
    double BSIM4v6thetaRout; 
    double BSIM4v6mstar;
	double BSIM4v6VgsteffVth;
    double BSIM4v6mstarcv;
    double BSIM4v6voffcbn;
    double BSIM4v6voffcbncv;
    double BSIM4v6rdswmin;
    double BSIM4v6rdwmin;
    double BSIM4v6rswmin;
    double BSIM4v6vfbsd;

    double BSIM4v6cof1;
    double BSIM4v6cof2;
    double BSIM4v6cof3;
    double BSIM4v6cof4;
    double BSIM4v6cdep0;
    double BSIM4v6ToxRatio;
    double BSIM4v6Aechvb;
    double BSIM4v6Bechvb;
    double BSIM4v6ToxRatioEdge;
    double BSIM4v6AechvbEdgeS;
    double BSIM4v6AechvbEdgeD;
    double BSIM4v6BechvbEdge;
    double BSIM4v6ldeb;
    double BSIM4v6k1ox;
    double BSIM4v6k2ox;
    double BSIM4v6vfbzbfactor;

    struct bsim4v6SizeDependParam  *pNext;
};


typedef struct sBSIM4v6model 
{

    struct GENmodel gen;

#define BSIM4v6modType gen.GENmodType
#define BSIM4v6nextModel(inst) ((struct sBSIM4v6model *)((inst)->gen.GENnextModel))
#define BSIM4v6instances(inst) ((BSIM4v6instance *)((inst)->gen.GENinstances))
#define BSIM4v6modName gen.GENmodName

    int BSIM4v6type;

    int    BSIM4v6mobMod;
    int    BSIM4v6cvchargeMod;
    int    BSIM4v6capMod;
    int    BSIM4v6dioMod;
    int    BSIM4v6trnqsMod;
    int    BSIM4v6acnqsMod;
    int    BSIM4v6fnoiMod;
    int    BSIM4v6tnoiMod;
    int    BSIM4v6rdsMod;
    int    BSIM4v6rbodyMod;
    int    BSIM4v6rgateMod;
    int    BSIM4v6perMod;
    int    BSIM4v6geoMod;
    int    BSIM4v6rgeoMod;
    int    BSIM4v6mtrlMod;
    int    BSIM4v6igcMod;
    int    BSIM4v6igbMod;
    int    BSIM4v6tempMod;
    int    BSIM4v6binUnit;
    int    BSIM4v6paramChk;
    char   *BSIM4v6version;
    double BSIM4v6eot;
    double BSIM4v6vddeot;
	double BSIM4v6tempeot;
	double BSIM4v6leffeot;
	double BSIM4v6weffeot;
    double BSIM4v6ados;
    double BSIM4v6bdos;
    double BSIM4v6toxe;             
    double BSIM4v6toxp;
    double BSIM4v6toxm;
    double BSIM4v6dtox;
    double BSIM4v6epsrox;
    double BSIM4v6cdsc;           
    double BSIM4v6cdscb; 
    double BSIM4v6cdscd;          
    double BSIM4v6cit;           
    double BSIM4v6nfactor;      
    double BSIM4v6xj;
    double BSIM4v6vsat;         
    double BSIM4v6at;         
    double BSIM4v6a0;   
    double BSIM4v6ags;      
    double BSIM4v6a1;         
    double BSIM4v6a2;         
    double BSIM4v6keta;     
    double BSIM4v6nsub;
    double BSIM4v6phig;
    double BSIM4v6epsrgate;
    double BSIM4v6easub;
    double BSIM4v6epsrsub;
    double BSIM4v6ni0sub;
    double BSIM4v6bg0sub;
    double BSIM4v6tbgasub;
    double BSIM4v6tbgbsub;
    double BSIM4v6ndep;        
    double BSIM4v6nsd;
    double BSIM4v6phin;
    double BSIM4v6ngate;        
    double BSIM4v6gamma1;      
    double BSIM4v6gamma2;     
    double BSIM4v6vbx;      
    double BSIM4v6vbm;       
    double BSIM4v6xt;       
    double BSIM4v6k1;
    double BSIM4v6kt1;
    double BSIM4v6kt1l;
    double BSIM4v6kt2;
    double BSIM4v6k2;
    double BSIM4v6k3;
    double BSIM4v6k3b;
    double BSIM4v6w0;
    double BSIM4v6dvtp0;
    double BSIM4v6dvtp1;
    double BSIM4v6lpe0;
    double BSIM4v6lpeb;
    double BSIM4v6dvt0;      
    double BSIM4v6dvt1;      
    double BSIM4v6dvt2;      
    double BSIM4v6dvt0w;      
    double BSIM4v6dvt1w;      
    double BSIM4v6dvt2w;      
    double BSIM4v6drout;      
    double BSIM4v6dsub;      
    double BSIM4v6vth0;
    double BSIM4v6eu;
	double BSIM4v6ucs;
    double BSIM4v6ua;
    double BSIM4v6ua1;
    double BSIM4v6ub;
    double BSIM4v6ub1;
    double BSIM4v6uc;
    double BSIM4v6uc1;
    double BSIM4v6ud;
    double BSIM4v6ud1;
    double BSIM4v6up;
    double BSIM4v6lp;
    double BSIM4v6u0;
    double BSIM4v6ute;
	double BSIM4v6ucste;
    double BSIM4v6voff;
    double BSIM4v6tvoff;
    double BSIM4v6minv;
    double BSIM4v6minvcv;
    double BSIM4v6voffl;
    double BSIM4v6voffcvl;
    double BSIM4v6delta;
    double BSIM4v6rdsw;       
    double BSIM4v6rdswmin;
    double BSIM4v6rdwmin;
    double BSIM4v6rswmin;
    double BSIM4v6rsw;
    double BSIM4v6rdw;
    double BSIM4v6prwg;
    double BSIM4v6prwb;
    double BSIM4v6prt;       
    double BSIM4v6eta0;         
    double BSIM4v6etab;         
    double BSIM4v6pclm;      
    double BSIM4v6pdibl1;      
    double BSIM4v6pdibl2;      
    double BSIM4v6pdiblb;
    double BSIM4v6fprout;
    double BSIM4v6pdits;
    double BSIM4v6pditsd;
    double BSIM4v6pditsl;
    double BSIM4v6pscbe1;       
    double BSIM4v6pscbe2;       
    double BSIM4v6pvag;       
    double BSIM4v6wr;
    double BSIM4v6dwg;
    double BSIM4v6dwb;
    double BSIM4v6b0;
    double BSIM4v6b1;
    double BSIM4v6alpha0;
    double BSIM4v6alpha1;
    double BSIM4v6beta0;
    double BSIM4v6agidl;
    double BSIM4v6bgidl;
    double BSIM4v6cgidl;
    double BSIM4v6egidl;
    double BSIM4v6agisl;
    double BSIM4v6bgisl;
    double BSIM4v6cgisl;
    double BSIM4v6egisl;
    double BSIM4v6aigc;
    double BSIM4v6bigc;
    double BSIM4v6cigc;
    double BSIM4v6aigsd;
    double BSIM4v6bigsd;
    double BSIM4v6cigsd;
    double BSIM4v6aigs;
    double BSIM4v6bigs;
    double BSIM4v6cigs;
    double BSIM4v6aigd;
    double BSIM4v6bigd;
    double BSIM4v6cigd;
    double BSIM4v6aigbacc;
    double BSIM4v6bigbacc;
    double BSIM4v6cigbacc;
    double BSIM4v6aigbinv;
    double BSIM4v6bigbinv;
    double BSIM4v6cigbinv;
    double BSIM4v6nigc;
    double BSIM4v6nigbacc;
    double BSIM4v6nigbinv;
    double BSIM4v6ntox;
    double BSIM4v6eigbinv;
    double BSIM4v6pigcd;
    double BSIM4v6poxedge;
    double BSIM4v6toxref;
    double BSIM4v6ijthdfwd;
    double BSIM4v6ijthsfwd;
    double BSIM4v6ijthdrev;
    double BSIM4v6ijthsrev;
    double BSIM4v6xjbvd;
    double BSIM4v6xjbvs;
    double BSIM4v6bvd;
    double BSIM4v6bvs;

    double BSIM4v6jtss;
    double BSIM4v6jtsd;
    double BSIM4v6jtssws;
    double BSIM4v6jtsswd;
    double BSIM4v6jtsswgs;
    double BSIM4v6jtsswgd;
	double BSIM4v6jtweff;
    double BSIM4v6njts;
    double BSIM4v6njtssw;
    double BSIM4v6njtsswg;
    double BSIM4v6njtsd;
    double BSIM4v6njtsswd;
    double BSIM4v6njtsswgd;
    double BSIM4v6xtss;
    double BSIM4v6xtsd;
    double BSIM4v6xtssws;
    double BSIM4v6xtsswd;
    double BSIM4v6xtsswgs;
    double BSIM4v6xtsswgd;
    double BSIM4v6tnjts;
    double BSIM4v6tnjtssw;
    double BSIM4v6tnjtsswg;
    double BSIM4v6tnjtsd;
    double BSIM4v6tnjtsswd;
    double BSIM4v6tnjtsswgd;
    double BSIM4v6vtss;
    double BSIM4v6vtsd;
    double BSIM4v6vtssws;
    double BSIM4v6vtsswd;
    double BSIM4v6vtsswgs;
    double BSIM4v6vtsswgd;

    double BSIM4v6xrcrg1;
    double BSIM4v6xrcrg2;
    double BSIM4v6lambda;
    double BSIM4v6vtl; 
    double BSIM4v6lc; 
    double BSIM4v6xn; 
    double BSIM4v6vfbsdoff;  /* S/D flatband offset voltage  */
    double BSIM4v6lintnoi;  /* lint offset for noise calculation  */
    double BSIM4v6tvfbsdoff;  

    double BSIM4v6vfb;
    double BSIM4v6gbmin;
    double BSIM4v6rbdb;
    double BSIM4v6rbsb;
    double BSIM4v6rbpb;
    double BSIM4v6rbps;
    double BSIM4v6rbpd;

    double BSIM4v6rbps0;
    double BSIM4v6rbpsl;
    double BSIM4v6rbpsw;
    double BSIM4v6rbpsnf;

    double BSIM4v6rbpd0;
    double BSIM4v6rbpdl;
    double BSIM4v6rbpdw;
    double BSIM4v6rbpdnf;

    double BSIM4v6rbpbx0;
    double BSIM4v6rbpbxl;
    double BSIM4v6rbpbxw;
    double BSIM4v6rbpbxnf;
    double BSIM4v6rbpby0;
    double BSIM4v6rbpbyl;
    double BSIM4v6rbpbyw;
    double BSIM4v6rbpbynf;

    double BSIM4v6rbsbx0;
    double BSIM4v6rbsby0;
    double BSIM4v6rbdbx0;
    double BSIM4v6rbdby0;

    double BSIM4v6rbsdbxl;
    double BSIM4v6rbsdbxw;
    double BSIM4v6rbsdbxnf;
    double BSIM4v6rbsdbyl;
    double BSIM4v6rbsdbyw;
    double BSIM4v6rbsdbynf;

    double BSIM4v6tnoia;
    double BSIM4v6tnoib;
    double BSIM4v6rnoia;
    double BSIM4v6rnoib;
    double BSIM4v6ntnoi;

    /* CV model and Parasitics */
    double BSIM4v6cgsl;
    double BSIM4v6cgdl;
    double BSIM4v6ckappas;
    double BSIM4v6ckappad;
    double BSIM4v6cf;
    double BSIM4v6vfbcv;
    double BSIM4v6clc;
    double BSIM4v6cle;
    double BSIM4v6dwc;
    double BSIM4v6dlc;
    double BSIM4v6xw;
    double BSIM4v6xl;
    double BSIM4v6dlcig;
    double BSIM4v6dlcigd;
    double BSIM4v6dwj;
    double BSIM4v6noff;
    double BSIM4v6voffcv;
    double BSIM4v6acde;
    double BSIM4v6moin;
    double BSIM4v6tcj;
    double BSIM4v6tcjsw;
    double BSIM4v6tcjswg;
    double BSIM4v6tpb;
    double BSIM4v6tpbsw;
    double BSIM4v6tpbswg;
    double BSIM4v6dmcg;
    double BSIM4v6dmci;
    double BSIM4v6dmdg;
    double BSIM4v6dmcgt;
    double BSIM4v6xgw;
    double BSIM4v6xgl;
    double BSIM4v6rshg;
    double BSIM4v6ngcon;

    /* Length Dependence */
    double BSIM4v6lcdsc;           
    double BSIM4v6lcdscb; 
    double BSIM4v6lcdscd;          
    double BSIM4v6lcit;           
    double BSIM4v6lnfactor;      
    double BSIM4v6lxj;
    double BSIM4v6lvsat;         
    double BSIM4v6lat;         
    double BSIM4v6la0;   
    double BSIM4v6lags;      
    double BSIM4v6la1;         
    double BSIM4v6la2;         
    double BSIM4v6lketa;     
    double BSIM4v6lnsub;
    double BSIM4v6lndep;        
    double BSIM4v6lnsd;
    double BSIM4v6lphin;
    double BSIM4v6lngate;        
    double BSIM4v6lgamma1;      
    double BSIM4v6lgamma2;     
    double BSIM4v6lvbx;      
    double BSIM4v6lvbm;       
    double BSIM4v6lxt;       
    double BSIM4v6lk1;
    double BSIM4v6lkt1;
    double BSIM4v6lkt1l;
    double BSIM4v6lkt2;
    double BSIM4v6lk2;
    double BSIM4v6lk3;
    double BSIM4v6lk3b;
    double BSIM4v6lw0;
    double BSIM4v6ldvtp0;
    double BSIM4v6ldvtp1;
    double BSIM4v6llpe0;
    double BSIM4v6llpeb;
    double BSIM4v6ldvt0;      
    double BSIM4v6ldvt1;      
    double BSIM4v6ldvt2;      
    double BSIM4v6ldvt0w;      
    double BSIM4v6ldvt1w;      
    double BSIM4v6ldvt2w;      
    double BSIM4v6ldrout;      
    double BSIM4v6ldsub;      
    double BSIM4v6lvth0;
    double BSIM4v6lua;
    double BSIM4v6lua1;
    double BSIM4v6lub;
    double BSIM4v6lub1;
    double BSIM4v6luc;
    double BSIM4v6luc1;
    double BSIM4v6lud;
    double BSIM4v6lud1;
    double BSIM4v6lup;
    double BSIM4v6llp;
    double BSIM4v6lu0;
    double BSIM4v6leu;
	double BSIM4v6lucs;
    double BSIM4v6lute;
	double BSIM4v6lucste;
    double BSIM4v6lvoff;
    double BSIM4v6ltvoff;
    double BSIM4v6lminv;
    double BSIM4v6lminvcv;
    double BSIM4v6ldelta;
    double BSIM4v6lrdsw;       
    double BSIM4v6lrsw;
    double BSIM4v6lrdw;
    double BSIM4v6lprwg;
    double BSIM4v6lprwb;
    double BSIM4v6lprt;       
    double BSIM4v6leta0;         
    double BSIM4v6letab;         
    double BSIM4v6lpclm;      
    double BSIM4v6lpdibl1;      
    double BSIM4v6lpdibl2;      
    double BSIM4v6lpdiblb;
    double BSIM4v6lfprout;
    double BSIM4v6lpdits;
    double BSIM4v6lpditsd;
    double BSIM4v6lpscbe1;       
    double BSIM4v6lpscbe2;       
    double BSIM4v6lpvag;       
    double BSIM4v6lwr;
    double BSIM4v6ldwg;
    double BSIM4v6ldwb;
    double BSIM4v6lb0;
    double BSIM4v6lb1;
    double BSIM4v6lalpha0;
    double BSIM4v6lalpha1;
    double BSIM4v6lbeta0;
    double BSIM4v6lvfb;
    double BSIM4v6lagidl;
    double BSIM4v6lbgidl;
    double BSIM4v6lcgidl;
    double BSIM4v6legidl;
    double BSIM4v6lagisl;
    double BSIM4v6lbgisl;
    double BSIM4v6lcgisl;
    double BSIM4v6legisl;
    double BSIM4v6laigc;
    double BSIM4v6lbigc;
    double BSIM4v6lcigc;
    double BSIM4v6laigsd;
    double BSIM4v6lbigsd;
    double BSIM4v6lcigsd;
    double BSIM4v6laigs;
    double BSIM4v6lbigs;
    double BSIM4v6lcigs;
    double BSIM4v6laigd;
    double BSIM4v6lbigd;
    double BSIM4v6lcigd;
    double BSIM4v6laigbacc;
    double BSIM4v6lbigbacc;
    double BSIM4v6lcigbacc;
    double BSIM4v6laigbinv;
    double BSIM4v6lbigbinv;
    double BSIM4v6lcigbinv;
    double BSIM4v6lnigc;
    double BSIM4v6lnigbacc;
    double BSIM4v6lnigbinv;
    double BSIM4v6lntox;
    double BSIM4v6leigbinv;
    double BSIM4v6lpigcd;
    double BSIM4v6lpoxedge;
    double BSIM4v6lxrcrg1;
    double BSIM4v6lxrcrg2;
    double BSIM4v6llambda;
    double BSIM4v6lvtl; 
    double BSIM4v6lxn; 
    double BSIM4v6lvfbsdoff; 
    double BSIM4v6ltvfbsdoff; 

    /* CV model */
    double BSIM4v6lcgsl;
    double BSIM4v6lcgdl;
    double BSIM4v6lckappas;
    double BSIM4v6lckappad;
    double BSIM4v6lcf;
    double BSIM4v6lclc;
    double BSIM4v6lcle;
    double BSIM4v6lvfbcv;
    double BSIM4v6lnoff;
    double BSIM4v6lvoffcv;
    double BSIM4v6lacde;
    double BSIM4v6lmoin;

    /* Width Dependence */
    double BSIM4v6wcdsc;           
    double BSIM4v6wcdscb; 
    double BSIM4v6wcdscd;          
    double BSIM4v6wcit;           
    double BSIM4v6wnfactor;      
    double BSIM4v6wxj;
    double BSIM4v6wvsat;         
    double BSIM4v6wat;         
    double BSIM4v6wa0;   
    double BSIM4v6wags;      
    double BSIM4v6wa1;         
    double BSIM4v6wa2;         
    double BSIM4v6wketa;     
    double BSIM4v6wnsub;
    double BSIM4v6wndep;        
    double BSIM4v6wnsd;
    double BSIM4v6wphin;
    double BSIM4v6wngate;        
    double BSIM4v6wgamma1;      
    double BSIM4v6wgamma2;     
    double BSIM4v6wvbx;      
    double BSIM4v6wvbm;       
    double BSIM4v6wxt;       
    double BSIM4v6wk1;
    double BSIM4v6wkt1;
    double BSIM4v6wkt1l;
    double BSIM4v6wkt2;
    double BSIM4v6wk2;
    double BSIM4v6wk3;
    double BSIM4v6wk3b;
    double BSIM4v6ww0;
    double BSIM4v6wdvtp0;
    double BSIM4v6wdvtp1;
    double BSIM4v6wlpe0;
    double BSIM4v6wlpeb;
    double BSIM4v6wdvt0;      
    double BSIM4v6wdvt1;      
    double BSIM4v6wdvt2;      
    double BSIM4v6wdvt0w;      
    double BSIM4v6wdvt1w;      
    double BSIM4v6wdvt2w;      
    double BSIM4v6wdrout;      
    double BSIM4v6wdsub;      
    double BSIM4v6wvth0;
    double BSIM4v6wua;
    double BSIM4v6wua1;
    double BSIM4v6wub;
    double BSIM4v6wub1;
    double BSIM4v6wuc;
    double BSIM4v6wuc1;
    double BSIM4v6wud;
    double BSIM4v6wud1;
    double BSIM4v6wup;
    double BSIM4v6wlp;
    double BSIM4v6wu0;
    double BSIM4v6weu;
	double BSIM4v6wucs;
    double BSIM4v6wute;
	double BSIM4v6wucste;
    double BSIM4v6wvoff;
    double BSIM4v6wtvoff;
    double BSIM4v6wminv;
    double BSIM4v6wminvcv;
    double BSIM4v6wdelta;
    double BSIM4v6wrdsw;       
    double BSIM4v6wrsw;
    double BSIM4v6wrdw;
    double BSIM4v6wprwg;
    double BSIM4v6wprwb;
    double BSIM4v6wprt;       
    double BSIM4v6weta0;         
    double BSIM4v6wetab;         
    double BSIM4v6wpclm;      
    double BSIM4v6wpdibl1;      
    double BSIM4v6wpdibl2;      
    double BSIM4v6wpdiblb;
    double BSIM4v6wfprout;
    double BSIM4v6wpdits;
    double BSIM4v6wpditsd;
    double BSIM4v6wpscbe1;       
    double BSIM4v6wpscbe2;       
    double BSIM4v6wpvag;       
    double BSIM4v6wwr;
    double BSIM4v6wdwg;
    double BSIM4v6wdwb;
    double BSIM4v6wb0;
    double BSIM4v6wb1;
    double BSIM4v6walpha0;
    double BSIM4v6walpha1;
    double BSIM4v6wbeta0;
    double BSIM4v6wvfb;
    double BSIM4v6wagidl;
    double BSIM4v6wbgidl;
    double BSIM4v6wcgidl;
    double BSIM4v6wegidl;
    double BSIM4v6wagisl;
    double BSIM4v6wbgisl;
    double BSIM4v6wcgisl;
    double BSIM4v6wegisl;
    double BSIM4v6waigc;
    double BSIM4v6wbigc;
    double BSIM4v6wcigc;
    double BSIM4v6waigsd;
    double BSIM4v6wbigsd;
    double BSIM4v6wcigsd;
    double BSIM4v6waigs;
    double BSIM4v6wbigs;
    double BSIM4v6wcigs;
    double BSIM4v6waigd;
    double BSIM4v6wbigd;
    double BSIM4v6wcigd;
    double BSIM4v6waigbacc;
    double BSIM4v6wbigbacc;
    double BSIM4v6wcigbacc;
    double BSIM4v6waigbinv;
    double BSIM4v6wbigbinv;
    double BSIM4v6wcigbinv;
    double BSIM4v6wnigc;
    double BSIM4v6wnigbacc;
    double BSIM4v6wnigbinv;
    double BSIM4v6wntox;
    double BSIM4v6weigbinv;
    double BSIM4v6wpigcd;
    double BSIM4v6wpoxedge;
    double BSIM4v6wxrcrg1;
    double BSIM4v6wxrcrg2;
    double BSIM4v6wlambda;
    double BSIM4v6wvtl; 
    double BSIM4v6wxn; 
    double BSIM4v6wvfbsdoff;  
    double BSIM4v6wtvfbsdoff;  

    /* CV model */
    double BSIM4v6wcgsl;
    double BSIM4v6wcgdl;
    double BSIM4v6wckappas;
    double BSIM4v6wckappad;
    double BSIM4v6wcf;
    double BSIM4v6wclc;
    double BSIM4v6wcle;
    double BSIM4v6wvfbcv;
    double BSIM4v6wnoff;
    double BSIM4v6wvoffcv;
    double BSIM4v6wacde;
    double BSIM4v6wmoin;

    /* Cross-term Dependence */
    double BSIM4v6pcdsc;           
    double BSIM4v6pcdscb; 
    double BSIM4v6pcdscd;          
    double BSIM4v6pcit;           
    double BSIM4v6pnfactor;      
    double BSIM4v6pxj;
    double BSIM4v6pvsat;         
    double BSIM4v6pat;         
    double BSIM4v6pa0;   
    double BSIM4v6pags;      
    double BSIM4v6pa1;         
    double BSIM4v6pa2;         
    double BSIM4v6pketa;     
    double BSIM4v6pnsub;
    double BSIM4v6pndep;        
    double BSIM4v6pnsd;
    double BSIM4v6pphin;
    double BSIM4v6pngate;        
    double BSIM4v6pgamma1;      
    double BSIM4v6pgamma2;     
    double BSIM4v6pvbx;      
    double BSIM4v6pvbm;       
    double BSIM4v6pxt;       
    double BSIM4v6pk1;
    double BSIM4v6pkt1;
    double BSIM4v6pkt1l;
    double BSIM4v6pkt2;
    double BSIM4v6pk2;
    double BSIM4v6pk3;
    double BSIM4v6pk3b;
    double BSIM4v6pw0;
    double BSIM4v6pdvtp0;
    double BSIM4v6pdvtp1;
    double BSIM4v6plpe0;
    double BSIM4v6plpeb;
    double BSIM4v6pdvt0;      
    double BSIM4v6pdvt1;      
    double BSIM4v6pdvt2;      
    double BSIM4v6pdvt0w;      
    double BSIM4v6pdvt1w;      
    double BSIM4v6pdvt2w;      
    double BSIM4v6pdrout;      
    double BSIM4v6pdsub;      
    double BSIM4v6pvth0;
    double BSIM4v6pua;
    double BSIM4v6pua1;
    double BSIM4v6pub;
    double BSIM4v6pub1;
    double BSIM4v6puc;
    double BSIM4v6puc1;
    double BSIM4v6pud;
    double BSIM4v6pud1;
    double BSIM4v6pup;
    double BSIM4v6plp;
    double BSIM4v6pu0;
    double BSIM4v6peu;
	double BSIM4v6pucs;
    double BSIM4v6pute;
	double BSIM4v6pucste;
    double BSIM4v6pvoff;
    double BSIM4v6ptvoff;
    double BSIM4v6pminv;
    double BSIM4v6pminvcv;
    double BSIM4v6pdelta;
    double BSIM4v6prdsw;
    double BSIM4v6prsw;
    double BSIM4v6prdw;
    double BSIM4v6pprwg;
    double BSIM4v6pprwb;
    double BSIM4v6pprt;       
    double BSIM4v6peta0;         
    double BSIM4v6petab;         
    double BSIM4v6ppclm;      
    double BSIM4v6ppdibl1;      
    double BSIM4v6ppdibl2;      
    double BSIM4v6ppdiblb;
    double BSIM4v6pfprout;
    double BSIM4v6ppdits;
    double BSIM4v6ppditsd;
    double BSIM4v6ppscbe1;       
    double BSIM4v6ppscbe2;       
    double BSIM4v6ppvag;       
    double BSIM4v6pwr;
    double BSIM4v6pdwg;
    double BSIM4v6pdwb;
    double BSIM4v6pb0;
    double BSIM4v6pb1;
    double BSIM4v6palpha0;
    double BSIM4v6palpha1;
    double BSIM4v6pbeta0;
    double BSIM4v6pvfb;
    double BSIM4v6pagidl;
    double BSIM4v6pbgidl;
    double BSIM4v6pcgidl;
    double BSIM4v6pegidl;
    double BSIM4v6pagisl;
    double BSIM4v6pbgisl;
    double BSIM4v6pcgisl;
    double BSIM4v6pegisl;
    double BSIM4v6paigc;
    double BSIM4v6pbigc;
    double BSIM4v6pcigc;
    double BSIM4v6paigsd;
    double BSIM4v6pbigsd;
    double BSIM4v6pcigsd;
    double BSIM4v6paigs;
    double BSIM4v6pbigs;
    double BSIM4v6pcigs;
    double BSIM4v6paigd;
    double BSIM4v6pbigd;
    double BSIM4v6pcigd;
    double BSIM4v6paigbacc;
    double BSIM4v6pbigbacc;
    double BSIM4v6pcigbacc;
    double BSIM4v6paigbinv;
    double BSIM4v6pbigbinv;
    double BSIM4v6pcigbinv;
    double BSIM4v6pnigc;
    double BSIM4v6pnigbacc;
    double BSIM4v6pnigbinv;
    double BSIM4v6pntox;
    double BSIM4v6peigbinv;
    double BSIM4v6ppigcd;
    double BSIM4v6ppoxedge;
    double BSIM4v6pxrcrg1;
    double BSIM4v6pxrcrg2;
    double BSIM4v6plambda;
    double BSIM4v6pvtl;
    double BSIM4v6pxn; 
    double BSIM4v6pvfbsdoff;  
    double BSIM4v6ptvfbsdoff;  

    /* CV model */
    double BSIM4v6pcgsl;
    double BSIM4v6pcgdl;
    double BSIM4v6pckappas;
    double BSIM4v6pckappad;
    double BSIM4v6pcf;
    double BSIM4v6pclc;
    double BSIM4v6pcle;
    double BSIM4v6pvfbcv;
    double BSIM4v6pnoff;
    double BSIM4v6pvoffcv;
    double BSIM4v6pacde;
    double BSIM4v6pmoin;

    double BSIM4v6tnom;
    double BSIM4v6cgso;
    double BSIM4v6cgdo;
    double BSIM4v6cgbo;
    double BSIM4v6xpart;
    double BSIM4v6cFringOut;
    double BSIM4v6cFringMax;

    double BSIM4v6sheetResistance;
    double BSIM4v6SjctSatCurDensity;
    double BSIM4v6DjctSatCurDensity;
    double BSIM4v6SjctSidewallSatCurDensity;
    double BSIM4v6DjctSidewallSatCurDensity;
    double BSIM4v6SjctGateSidewallSatCurDensity;
    double BSIM4v6DjctGateSidewallSatCurDensity;
    double BSIM4v6SbulkJctPotential;
    double BSIM4v6DbulkJctPotential;
    double BSIM4v6SbulkJctBotGradingCoeff;
    double BSIM4v6DbulkJctBotGradingCoeff;
    double BSIM4v6SbulkJctSideGradingCoeff;
    double BSIM4v6DbulkJctSideGradingCoeff;
    double BSIM4v6SbulkJctGateSideGradingCoeff;
    double BSIM4v6DbulkJctGateSideGradingCoeff;
    double BSIM4v6SsidewallJctPotential;
    double BSIM4v6DsidewallJctPotential;
    double BSIM4v6SGatesidewallJctPotential;
    double BSIM4v6DGatesidewallJctPotential;
    double BSIM4v6SunitAreaJctCap;
    double BSIM4v6DunitAreaJctCap;
    double BSIM4v6SunitLengthSidewallJctCap;
    double BSIM4v6DunitLengthSidewallJctCap;
    double BSIM4v6SunitLengthGateSidewallJctCap;
    double BSIM4v6DunitLengthGateSidewallJctCap;
    double BSIM4v6SjctEmissionCoeff;
    double BSIM4v6DjctEmissionCoeff;
    double BSIM4v6SjctTempExponent;
    double BSIM4v6DjctTempExponent;
    double BSIM4v6njtsstemp;
    double BSIM4v6njtsswstemp;
    double BSIM4v6njtsswgstemp;
    double BSIM4v6njtsdtemp;
    double BSIM4v6njtsswdtemp;
    double BSIM4v6njtsswgdtemp;

    double BSIM4v6Lint;
    double BSIM4v6Ll;
    double BSIM4v6Llc;
    double BSIM4v6Lln;
    double BSIM4v6Lw;
    double BSIM4v6Lwc;
    double BSIM4v6Lwn;
    double BSIM4v6Lwl;
    double BSIM4v6Lwlc;
    double BSIM4v6Lmin;
    double BSIM4v6Lmax;

    double BSIM4v6Wint;
    double BSIM4v6Wl;
    double BSIM4v6Wlc;
    double BSIM4v6Wln;
    double BSIM4v6Ww;
    double BSIM4v6Wwc;
    double BSIM4v6Wwn;
    double BSIM4v6Wwl;
    double BSIM4v6Wwlc;
    double BSIM4v6Wmin;
    double BSIM4v6Wmax;

    /* added for stress effect */
    double BSIM4v6saref;
    double BSIM4v6sbref;
    double BSIM4v6wlod;
    double BSIM4v6ku0;
    double BSIM4v6kvsat;
    double BSIM4v6kvth0;
    double BSIM4v6tku0;
    double BSIM4v6llodku0;
    double BSIM4v6wlodku0;
    double BSIM4v6llodvth;
    double BSIM4v6wlodvth;
    double BSIM4v6lku0;
    double BSIM4v6wku0;
    double BSIM4v6pku0;
    double BSIM4v6lkvth0;
    double BSIM4v6wkvth0;
    double BSIM4v6pkvth0;
    double BSIM4v6stk2;
    double BSIM4v6lodk2;
    double BSIM4v6steta0;
    double BSIM4v6lodeta0;

    double BSIM4v6web; 
    double BSIM4v6wec;
    double BSIM4v6kvth0we; 
    double BSIM4v6k2we; 
    double BSIM4v6ku0we; 
    double BSIM4v6scref; 
    double BSIM4v6wpemod; 
    double BSIM4v6lkvth0we;
    double BSIM4v6lk2we;
    double BSIM4v6lku0we;
    double BSIM4v6wkvth0we;
    double BSIM4v6wk2we;
    double BSIM4v6wku0we;
    double BSIM4v6pkvth0we;
    double BSIM4v6pk2we;
    double BSIM4v6pku0we;

/* Pre-calculated constants
 * move to size-dependent param */
    double BSIM4v6Eg0;
    double BSIM4v6vtm;   
    double BSIM4v6vtm0;   
    double BSIM4v6coxe;
    double BSIM4v6coxp;
    double BSIM4v6cof1;
    double BSIM4v6cof2;
    double BSIM4v6cof3;
    double BSIM4v6cof4;
    double BSIM4v6vcrit;
    double BSIM4v6factor1;
    double BSIM4v6PhiBS;
    double BSIM4v6PhiBSWS;
    double BSIM4v6PhiBSWGS;
    double BSIM4v6SjctTempSatCurDensity;
    double BSIM4v6SjctSidewallTempSatCurDensity;
    double BSIM4v6SjctGateSidewallTempSatCurDensity;
    double BSIM4v6PhiBD;
    double BSIM4v6PhiBSWD;
    double BSIM4v6PhiBSWGD;
    double BSIM4v6DjctTempSatCurDensity;
    double BSIM4v6DjctSidewallTempSatCurDensity;
    double BSIM4v6DjctGateSidewallTempSatCurDensity;
    double BSIM4v6SunitAreaTempJctCap;
    double BSIM4v6DunitAreaTempJctCap;
    double BSIM4v6SunitLengthSidewallTempJctCap;
    double BSIM4v6DunitLengthSidewallTempJctCap;
    double BSIM4v6SunitLengthGateSidewallTempJctCap;
    double BSIM4v6DunitLengthGateSidewallTempJctCap;

    double BSIM4v6oxideTrapDensityA;      
    double BSIM4v6oxideTrapDensityB;     
    double BSIM4v6oxideTrapDensityC;  
    double BSIM4v6em;  
    double BSIM4v6ef;  
    double BSIM4v6af;  
    double BSIM4v6kf;  

    double BSIM4v6vgsMax;
    double BSIM4v6vgdMax;
    double BSIM4v6vgbMax;
    double BSIM4v6vdsMax;
    double BSIM4v6vbsMax;
    double BSIM4v6vbdMax;
    double BSIM4v6vgsrMax;
    double BSIM4v6vgdrMax;
    double BSIM4v6vgbrMax;
    double BSIM4v6vbsrMax;
    double BSIM4v6vbdrMax;

    struct bsim4v6SizeDependParam *pSizeDependParamKnot;

    
#ifdef USE_OMP
    int BSIM4v6InstCount;
    struct sBSIM4v6instance **BSIM4v6InstanceArray;
#endif

    /* Flags */
    unsigned  BSIM4v6mobModGiven :1;
    unsigned  BSIM4v6binUnitGiven :1;
    unsigned  BSIM4v6cvchargeModGiven :1;
    unsigned  BSIM4v6capModGiven :1;
    unsigned  BSIM4v6dioModGiven :1;
    unsigned  BSIM4v6rdsModGiven :1;
    unsigned  BSIM4v6rbodyModGiven :1;
    unsigned  BSIM4v6rgateModGiven :1;
    unsigned  BSIM4v6perModGiven :1;
    unsigned  BSIM4v6geoModGiven :1;
    unsigned  BSIM4v6rgeoModGiven :1;
    unsigned  BSIM4v6paramChkGiven :1;
    unsigned  BSIM4v6trnqsModGiven :1;
    unsigned  BSIM4v6acnqsModGiven :1;
    unsigned  BSIM4v6fnoiModGiven :1;
    unsigned  BSIM4v6tnoiModGiven :1;
    unsigned  BSIM4v6mtrlModGiven :1;
    unsigned  BSIM4v6igcModGiven :1;
    unsigned  BSIM4v6igbModGiven :1;
    unsigned  BSIM4v6tempModGiven :1;
    unsigned  BSIM4v6typeGiven   :1;
    unsigned  BSIM4v6toxrefGiven   :1;
    unsigned  BSIM4v6eotGiven   :1;
    unsigned  BSIM4v6vddeotGiven   :1;
	unsigned  BSIM4v6tempeotGiven  :1;
	unsigned  BSIM4v6leffeotGiven  :1;
	unsigned  BSIM4v6weffeotGiven  :1;
    unsigned  BSIM4v6adosGiven   :1;
    unsigned  BSIM4v6bdosGiven   :1;
    unsigned  BSIM4v6toxeGiven   :1;
    unsigned  BSIM4v6toxpGiven   :1;
    unsigned  BSIM4v6toxmGiven   :1;
    unsigned  BSIM4v6dtoxGiven   :1;
    unsigned  BSIM4v6epsroxGiven   :1;
    unsigned  BSIM4v6versionGiven   :1;
    unsigned  BSIM4v6cdscGiven   :1;
    unsigned  BSIM4v6cdscbGiven   :1;
    unsigned  BSIM4v6cdscdGiven   :1;
    unsigned  BSIM4v6citGiven   :1;
    unsigned  BSIM4v6nfactorGiven   :1;
    unsigned  BSIM4v6xjGiven   :1;
    unsigned  BSIM4v6vsatGiven   :1;
    unsigned  BSIM4v6atGiven   :1;
    unsigned  BSIM4v6a0Given   :1;
    unsigned  BSIM4v6agsGiven   :1;
    unsigned  BSIM4v6a1Given   :1;
    unsigned  BSIM4v6a2Given   :1;
    unsigned  BSIM4v6ketaGiven   :1;    
    unsigned  BSIM4v6nsubGiven   :1;
    unsigned  BSIM4v6phigGiven   :1;
    unsigned  BSIM4v6epsrgateGiven   :1;
    unsigned  BSIM4v6easubGiven   :1;
    unsigned  BSIM4v6epsrsubGiven   :1;
    unsigned  BSIM4v6ni0subGiven   :1;
    unsigned  BSIM4v6bg0subGiven   :1;
    unsigned  BSIM4v6tbgasubGiven   :1;
    unsigned  BSIM4v6tbgbsubGiven   :1;
    unsigned  BSIM4v6ndepGiven   :1;
    unsigned  BSIM4v6nsdGiven    :1;
    unsigned  BSIM4v6phinGiven   :1;
    unsigned  BSIM4v6ngateGiven   :1;
    unsigned  BSIM4v6gamma1Given   :1;
    unsigned  BSIM4v6gamma2Given   :1;
    unsigned  BSIM4v6vbxGiven   :1;
    unsigned  BSIM4v6vbmGiven   :1;
    unsigned  BSIM4v6xtGiven   :1;
    unsigned  BSIM4v6k1Given   :1;
    unsigned  BSIM4v6kt1Given   :1;
    unsigned  BSIM4v6kt1lGiven   :1;
    unsigned  BSIM4v6kt2Given   :1;
    unsigned  BSIM4v6k2Given   :1;
    unsigned  BSIM4v6k3Given   :1;
    unsigned  BSIM4v6k3bGiven   :1;
    unsigned  BSIM4v6w0Given   :1;
    unsigned  BSIM4v6dvtp0Given :1;
    unsigned  BSIM4v6dvtp1Given :1;
    unsigned  BSIM4v6lpe0Given   :1;
    unsigned  BSIM4v6lpebGiven   :1;
    unsigned  BSIM4v6dvt0Given   :1;   
    unsigned  BSIM4v6dvt1Given   :1;     
    unsigned  BSIM4v6dvt2Given   :1;     
    unsigned  BSIM4v6dvt0wGiven   :1;   
    unsigned  BSIM4v6dvt1wGiven   :1;     
    unsigned  BSIM4v6dvt2wGiven   :1;     
    unsigned  BSIM4v6droutGiven   :1;     
    unsigned  BSIM4v6dsubGiven   :1;    
    unsigned  BSIM4v6vth0Given   :1;
    unsigned  BSIM4v6euGiven   :1;
	unsigned  BSIM4v6ucsGiven  :1;
    unsigned  BSIM4v6uaGiven   :1;
    unsigned  BSIM4v6ua1Given   :1;
    unsigned  BSIM4v6ubGiven   :1;
    unsigned  BSIM4v6ub1Given   :1;
    unsigned  BSIM4v6ucGiven   :1;
    unsigned  BSIM4v6uc1Given   :1;
    unsigned  BSIM4v6udGiven     :1;
    unsigned  BSIM4v6ud1Given     :1;
    unsigned  BSIM4v6upGiven     :1;
    unsigned  BSIM4v6lpGiven     :1;
    unsigned  BSIM4v6u0Given   :1;
    unsigned  BSIM4v6uteGiven   :1;
	unsigned  BSIM4v6ucsteGiven :1;
    unsigned  BSIM4v6voffGiven   :1;
    unsigned  BSIM4v6tvoffGiven   :1;
    unsigned  BSIM4v6vofflGiven  :1;
    unsigned  BSIM4v6voffcvlGiven  :1;
    unsigned  BSIM4v6minvGiven   :1;
    unsigned  BSIM4v6minvcvGiven   :1;
    unsigned  BSIM4v6rdswGiven   :1;      
    unsigned  BSIM4v6rdswminGiven :1;
    unsigned  BSIM4v6rdwminGiven :1;
    unsigned  BSIM4v6rswminGiven :1;
    unsigned  BSIM4v6rswGiven   :1;
    unsigned  BSIM4v6rdwGiven   :1;
    unsigned  BSIM4v6prwgGiven   :1;      
    unsigned  BSIM4v6prwbGiven   :1;      
    unsigned  BSIM4v6prtGiven   :1;      
    unsigned  BSIM4v6eta0Given   :1;    
    unsigned  BSIM4v6etabGiven   :1;    
    unsigned  BSIM4v6pclmGiven   :1;   
    unsigned  BSIM4v6pdibl1Given   :1;   
    unsigned  BSIM4v6pdibl2Given   :1;  
    unsigned  BSIM4v6pdiblbGiven   :1;  
    unsigned  BSIM4v6fproutGiven   :1;
    unsigned  BSIM4v6pditsGiven    :1;
    unsigned  BSIM4v6pditsdGiven    :1;
    unsigned  BSIM4v6pditslGiven    :1;
    unsigned  BSIM4v6pscbe1Given   :1;    
    unsigned  BSIM4v6pscbe2Given   :1;    
    unsigned  BSIM4v6pvagGiven   :1;    
    unsigned  BSIM4v6deltaGiven  :1;     
    unsigned  BSIM4v6wrGiven   :1;
    unsigned  BSIM4v6dwgGiven   :1;
    unsigned  BSIM4v6dwbGiven   :1;
    unsigned  BSIM4v6b0Given   :1;
    unsigned  BSIM4v6b1Given   :1;
    unsigned  BSIM4v6alpha0Given   :1;
    unsigned  BSIM4v6alpha1Given   :1;
    unsigned  BSIM4v6beta0Given   :1;
    unsigned  BSIM4v6agidlGiven   :1;
    unsigned  BSIM4v6bgidlGiven   :1;
    unsigned  BSIM4v6cgidlGiven   :1;
    unsigned  BSIM4v6egidlGiven   :1;
    unsigned  BSIM4v6agislGiven   :1;
    unsigned  BSIM4v6bgislGiven   :1;
    unsigned  BSIM4v6cgislGiven   :1;
    unsigned  BSIM4v6egislGiven   :1;
    unsigned  BSIM4v6aigcGiven   :1;
    unsigned  BSIM4v6bigcGiven   :1;
    unsigned  BSIM4v6cigcGiven   :1;
    unsigned  BSIM4v6aigsdGiven   :1;
    unsigned  BSIM4v6bigsdGiven   :1;
    unsigned  BSIM4v6cigsdGiven   :1;
    unsigned  BSIM4v6aigsGiven   :1;
    unsigned  BSIM4v6bigsGiven   :1;
    unsigned  BSIM4v6cigsGiven   :1;
    unsigned  BSIM4v6aigdGiven   :1;
    unsigned  BSIM4v6bigdGiven   :1;
    unsigned  BSIM4v6cigdGiven   :1;
    unsigned  BSIM4v6aigbaccGiven   :1;
    unsigned  BSIM4v6bigbaccGiven   :1;
    unsigned  BSIM4v6cigbaccGiven   :1;
    unsigned  BSIM4v6aigbinvGiven   :1;
    unsigned  BSIM4v6bigbinvGiven   :1;
    unsigned  BSIM4v6cigbinvGiven   :1;
    unsigned  BSIM4v6nigcGiven   :1;
    unsigned  BSIM4v6nigbinvGiven   :1;
    unsigned  BSIM4v6nigbaccGiven   :1;
    unsigned  BSIM4v6ntoxGiven   :1;
    unsigned  BSIM4v6eigbinvGiven   :1;
    unsigned  BSIM4v6pigcdGiven   :1;
    unsigned  BSIM4v6poxedgeGiven   :1;
    unsigned  BSIM4v6ijthdfwdGiven  :1;
    unsigned  BSIM4v6ijthsfwdGiven  :1;
    unsigned  BSIM4v6ijthdrevGiven  :1;
    unsigned  BSIM4v6ijthsrevGiven  :1;
    unsigned  BSIM4v6xjbvdGiven   :1;
    unsigned  BSIM4v6xjbvsGiven   :1;
    unsigned  BSIM4v6bvdGiven   :1;
    unsigned  BSIM4v6bvsGiven   :1;

    unsigned  BSIM4v6jtssGiven   :1;
    unsigned  BSIM4v6jtsdGiven   :1;
    unsigned  BSIM4v6jtsswsGiven   :1;
    unsigned  BSIM4v6jtsswdGiven   :1;
    unsigned  BSIM4v6jtsswgsGiven   :1;
    unsigned  BSIM4v6jtsswgdGiven   :1;
	unsigned  BSIM4v6jtweffGiven    :1;
    unsigned  BSIM4v6njtsGiven   :1;
    unsigned  BSIM4v6njtsswGiven   :1;
    unsigned  BSIM4v6njtsswgGiven   :1;
    unsigned  BSIM4v6njtsdGiven   :1;
    unsigned  BSIM4v6njtsswdGiven   :1;
    unsigned  BSIM4v6njtsswgdGiven   :1;
    unsigned  BSIM4v6xtssGiven   :1;
    unsigned  BSIM4v6xtsdGiven   :1;
    unsigned  BSIM4v6xtsswsGiven   :1;
    unsigned  BSIM4v6xtsswdGiven   :1;
    unsigned  BSIM4v6xtsswgsGiven   :1;
    unsigned  BSIM4v6xtsswgdGiven   :1;
    unsigned  BSIM4v6tnjtsGiven   :1;
    unsigned  BSIM4v6tnjtsswGiven   :1;
    unsigned  BSIM4v6tnjtsswgGiven   :1;
    unsigned  BSIM4v6tnjtsdGiven   :1;
    unsigned  BSIM4v6tnjtsswdGiven   :1;
    unsigned  BSIM4v6tnjtsswgdGiven   :1;
    unsigned  BSIM4v6vtssGiven   :1;
    unsigned  BSIM4v6vtsdGiven   :1;
    unsigned  BSIM4v6vtsswsGiven   :1;
    unsigned  BSIM4v6vtsswdGiven   :1;
    unsigned  BSIM4v6vtsswgsGiven   :1;
    unsigned  BSIM4v6vtsswgdGiven   :1;

    unsigned  BSIM4v6vfbGiven   :1;
    unsigned  BSIM4v6gbminGiven :1;
    unsigned  BSIM4v6rbdbGiven :1;
    unsigned  BSIM4v6rbsbGiven :1;
    unsigned  BSIM4v6rbpsGiven :1;
    unsigned  BSIM4v6rbpdGiven :1;
    unsigned  BSIM4v6rbpbGiven :1;

    unsigned BSIM4v6rbps0Given :1;
    unsigned BSIM4v6rbpslGiven :1;
    unsigned BSIM4v6rbpswGiven :1;
    unsigned BSIM4v6rbpsnfGiven :1;

    unsigned BSIM4v6rbpd0Given :1;
    unsigned BSIM4v6rbpdlGiven :1;
    unsigned BSIM4v6rbpdwGiven :1;
    unsigned BSIM4v6rbpdnfGiven :1;

    unsigned BSIM4v6rbpbx0Given :1;
    unsigned BSIM4v6rbpbxlGiven :1;
    unsigned BSIM4v6rbpbxwGiven :1;
    unsigned BSIM4v6rbpbxnfGiven :1;
    unsigned BSIM4v6rbpby0Given :1;
    unsigned BSIM4v6rbpbylGiven :1;
    unsigned BSIM4v6rbpbywGiven :1;
    unsigned BSIM4v6rbpbynfGiven :1;

    unsigned BSIM4v6rbsbx0Given :1;
    unsigned BSIM4v6rbsby0Given :1;
    unsigned BSIM4v6rbdbx0Given :1;
    unsigned BSIM4v6rbdby0Given :1;

    unsigned BSIM4v6rbsdbxlGiven :1;
    unsigned BSIM4v6rbsdbxwGiven :1;
    unsigned BSIM4v6rbsdbxnfGiven :1;
    unsigned BSIM4v6rbsdbylGiven :1;
    unsigned BSIM4v6rbsdbywGiven :1;
    unsigned BSIM4v6rbsdbynfGiven :1;

    unsigned  BSIM4v6xrcrg1Given   :1;
    unsigned  BSIM4v6xrcrg2Given   :1;
    unsigned  BSIM4v6tnoiaGiven    :1;
    unsigned  BSIM4v6tnoibGiven    :1;
    unsigned  BSIM4v6rnoiaGiven    :1;
    unsigned  BSIM4v6rnoibGiven    :1;
    unsigned  BSIM4v6ntnoiGiven    :1;

    unsigned  BSIM4v6lambdaGiven    :1;
    unsigned  BSIM4v6vtlGiven    :1;
    unsigned  BSIM4v6lcGiven    :1;
    unsigned  BSIM4v6xnGiven    :1;
    unsigned  BSIM4v6vfbsdoffGiven    :1;
    unsigned  BSIM4v6lintnoiGiven    :1;
    unsigned  BSIM4v6tvfbsdoffGiven    :1;

    /* CV model and parasitics */
    unsigned  BSIM4v6cgslGiven   :1;
    unsigned  BSIM4v6cgdlGiven   :1;
    unsigned  BSIM4v6ckappasGiven   :1;
    unsigned  BSIM4v6ckappadGiven   :1;
    unsigned  BSIM4v6cfGiven   :1;
    unsigned  BSIM4v6vfbcvGiven   :1;
    unsigned  BSIM4v6clcGiven   :1;
    unsigned  BSIM4v6cleGiven   :1;
    unsigned  BSIM4v6dwcGiven   :1;
    unsigned  BSIM4v6dlcGiven   :1;
    unsigned  BSIM4v6xwGiven    :1;
    unsigned  BSIM4v6xlGiven    :1;
    unsigned  BSIM4v6dlcigGiven   :1;
    unsigned  BSIM4v6dlcigdGiven   :1;
    unsigned  BSIM4v6dwjGiven   :1;
    unsigned  BSIM4v6noffGiven  :1;
    unsigned  BSIM4v6voffcvGiven :1;
    unsigned  BSIM4v6acdeGiven  :1;
    unsigned  BSIM4v6moinGiven  :1;
    unsigned  BSIM4v6tcjGiven   :1;
    unsigned  BSIM4v6tcjswGiven :1;
    unsigned  BSIM4v6tcjswgGiven :1;
    unsigned  BSIM4v6tpbGiven    :1;
    unsigned  BSIM4v6tpbswGiven  :1;
    unsigned  BSIM4v6tpbswgGiven :1;
    unsigned  BSIM4v6dmcgGiven :1;
    unsigned  BSIM4v6dmciGiven :1;
    unsigned  BSIM4v6dmdgGiven :1;
    unsigned  BSIM4v6dmcgtGiven :1;
    unsigned  BSIM4v6xgwGiven :1;
    unsigned  BSIM4v6xglGiven :1;
    unsigned  BSIM4v6rshgGiven :1;
    unsigned  BSIM4v6ngconGiven :1;


    /* Length dependence */
    unsigned  BSIM4v6lcdscGiven   :1;
    unsigned  BSIM4v6lcdscbGiven   :1;
    unsigned  BSIM4v6lcdscdGiven   :1;
    unsigned  BSIM4v6lcitGiven   :1;
    unsigned  BSIM4v6lnfactorGiven   :1;
    unsigned  BSIM4v6lxjGiven   :1;
    unsigned  BSIM4v6lvsatGiven   :1;
    unsigned  BSIM4v6latGiven   :1;
    unsigned  BSIM4v6la0Given   :1;
    unsigned  BSIM4v6lagsGiven   :1;
    unsigned  BSIM4v6la1Given   :1;
    unsigned  BSIM4v6la2Given   :1;
    unsigned  BSIM4v6lketaGiven   :1;    
    unsigned  BSIM4v6lnsubGiven   :1;
    unsigned  BSIM4v6lndepGiven   :1;
    unsigned  BSIM4v6lnsdGiven    :1;
    unsigned  BSIM4v6lphinGiven   :1;
    unsigned  BSIM4v6lngateGiven   :1;
    unsigned  BSIM4v6lgamma1Given   :1;
    unsigned  BSIM4v6lgamma2Given   :1;
    unsigned  BSIM4v6lvbxGiven   :1;
    unsigned  BSIM4v6lvbmGiven   :1;
    unsigned  BSIM4v6lxtGiven   :1;
    unsigned  BSIM4v6lk1Given   :1;
    unsigned  BSIM4v6lkt1Given   :1;
    unsigned  BSIM4v6lkt1lGiven   :1;
    unsigned  BSIM4v6lkt2Given   :1;
    unsigned  BSIM4v6lk2Given   :1;
    unsigned  BSIM4v6lk3Given   :1;
    unsigned  BSIM4v6lk3bGiven   :1;
    unsigned  BSIM4v6lw0Given   :1;
    unsigned  BSIM4v6ldvtp0Given :1;
    unsigned  BSIM4v6ldvtp1Given :1;
    unsigned  BSIM4v6llpe0Given   :1;
    unsigned  BSIM4v6llpebGiven   :1;
    unsigned  BSIM4v6ldvt0Given   :1;   
    unsigned  BSIM4v6ldvt1Given   :1;     
    unsigned  BSIM4v6ldvt2Given   :1;     
    unsigned  BSIM4v6ldvt0wGiven   :1;   
    unsigned  BSIM4v6ldvt1wGiven   :1;     
    unsigned  BSIM4v6ldvt2wGiven   :1;     
    unsigned  BSIM4v6ldroutGiven   :1;     
    unsigned  BSIM4v6ldsubGiven   :1;     
    unsigned  BSIM4v6lvth0Given   :1;
    unsigned  BSIM4v6luaGiven   :1;
    unsigned  BSIM4v6lua1Given   :1;
    unsigned  BSIM4v6lubGiven   :1;
    unsigned  BSIM4v6lub1Given   :1;
    unsigned  BSIM4v6lucGiven   :1;
    unsigned  BSIM4v6luc1Given   :1;
    unsigned  BSIM4v6ludGiven     :1;
    unsigned  BSIM4v6lud1Given     :1;
    unsigned  BSIM4v6lupGiven     :1;
    unsigned  BSIM4v6llpGiven     :1;
    unsigned  BSIM4v6lu0Given   :1;
    unsigned  BSIM4v6leuGiven   :1;
	unsigned  BSIM4v6lucsGiven   :1;
    unsigned  BSIM4v6luteGiven   :1;
	unsigned  BSIM4v6lucsteGiven  :1;
    unsigned  BSIM4v6lvoffGiven   :1;
    unsigned  BSIM4v6ltvoffGiven   :1;
    unsigned  BSIM4v6lminvGiven   :1;
    unsigned  BSIM4v6lminvcvGiven   :1;
    unsigned  BSIM4v6lrdswGiven   :1;      
    unsigned  BSIM4v6lrswGiven   :1;
    unsigned  BSIM4v6lrdwGiven   :1;
    unsigned  BSIM4v6lprwgGiven   :1;      
    unsigned  BSIM4v6lprwbGiven   :1;      
    unsigned  BSIM4v6lprtGiven   :1;      
    unsigned  BSIM4v6leta0Given   :1;    
    unsigned  BSIM4v6letabGiven   :1;    
    unsigned  BSIM4v6lpclmGiven   :1;   
    unsigned  BSIM4v6lpdibl1Given   :1;   
    unsigned  BSIM4v6lpdibl2Given   :1;  
    unsigned  BSIM4v6lpdiblbGiven   :1;  
    unsigned  BSIM4v6lfproutGiven   :1;
    unsigned  BSIM4v6lpditsGiven    :1;
    unsigned  BSIM4v6lpditsdGiven    :1;
    unsigned  BSIM4v6lpscbe1Given   :1;    
    unsigned  BSIM4v6lpscbe2Given   :1;    
    unsigned  BSIM4v6lpvagGiven   :1;    
    unsigned  BSIM4v6ldeltaGiven  :1;     
    unsigned  BSIM4v6lwrGiven   :1;
    unsigned  BSIM4v6ldwgGiven   :1;
    unsigned  BSIM4v6ldwbGiven   :1;
    unsigned  BSIM4v6lb0Given   :1;
    unsigned  BSIM4v6lb1Given   :1;
    unsigned  BSIM4v6lalpha0Given   :1;
    unsigned  BSIM4v6lalpha1Given   :1;
    unsigned  BSIM4v6lbeta0Given   :1;
    unsigned  BSIM4v6lvfbGiven   :1;
    unsigned  BSIM4v6lagidlGiven   :1;
    unsigned  BSIM4v6lbgidlGiven   :1;
    unsigned  BSIM4v6lcgidlGiven   :1;
    unsigned  BSIM4v6legidlGiven   :1;
    unsigned  BSIM4v6lagislGiven   :1;
    unsigned  BSIM4v6lbgislGiven   :1;
    unsigned  BSIM4v6lcgislGiven   :1;
    unsigned  BSIM4v6legislGiven   :1;
    unsigned  BSIM4v6laigcGiven   :1;
    unsigned  BSIM4v6lbigcGiven   :1;
    unsigned  BSIM4v6lcigcGiven   :1;
    unsigned  BSIM4v6laigsdGiven   :1;
    unsigned  BSIM4v6lbigsdGiven   :1;
    unsigned  BSIM4v6lcigsdGiven   :1;
    unsigned  BSIM4v6laigsGiven   :1;
    unsigned  BSIM4v6lbigsGiven   :1;
    unsigned  BSIM4v6lcigsGiven   :1;
    unsigned  BSIM4v6laigdGiven   :1;
    unsigned  BSIM4v6lbigdGiven   :1;
    unsigned  BSIM4v6lcigdGiven   :1;
    unsigned  BSIM4v6laigbaccGiven   :1;
    unsigned  BSIM4v6lbigbaccGiven   :1;
    unsigned  BSIM4v6lcigbaccGiven   :1;
    unsigned  BSIM4v6laigbinvGiven   :1;
    unsigned  BSIM4v6lbigbinvGiven   :1;
    unsigned  BSIM4v6lcigbinvGiven   :1;
    unsigned  BSIM4v6lnigcGiven   :1;
    unsigned  BSIM4v6lnigbinvGiven   :1;
    unsigned  BSIM4v6lnigbaccGiven   :1;
    unsigned  BSIM4v6lntoxGiven   :1;
    unsigned  BSIM4v6leigbinvGiven   :1;
    unsigned  BSIM4v6lpigcdGiven   :1;
    unsigned  BSIM4v6lpoxedgeGiven   :1;
    unsigned  BSIM4v6lxrcrg1Given   :1;
    unsigned  BSIM4v6lxrcrg2Given   :1;
    unsigned  BSIM4v6llambdaGiven    :1;
    unsigned  BSIM4v6lvtlGiven    :1;
    unsigned  BSIM4v6lxnGiven    :1;
    unsigned  BSIM4v6lvfbsdoffGiven    :1;
    unsigned  BSIM4v6ltvfbsdoffGiven    :1;

    /* CV model */
    unsigned  BSIM4v6lcgslGiven   :1;
    unsigned  BSIM4v6lcgdlGiven   :1;
    unsigned  BSIM4v6lckappasGiven   :1;
    unsigned  BSIM4v6lckappadGiven   :1;
    unsigned  BSIM4v6lcfGiven   :1;
    unsigned  BSIM4v6lclcGiven   :1;
    unsigned  BSIM4v6lcleGiven   :1;
    unsigned  BSIM4v6lvfbcvGiven   :1;
    unsigned  BSIM4v6lnoffGiven   :1;
    unsigned  BSIM4v6lvoffcvGiven :1;
    unsigned  BSIM4v6lacdeGiven   :1;
    unsigned  BSIM4v6lmoinGiven   :1;

    /* Width dependence */
    unsigned  BSIM4v6wcdscGiven   :1;
    unsigned  BSIM4v6wcdscbGiven   :1;
    unsigned  BSIM4v6wcdscdGiven   :1;
    unsigned  BSIM4v6wcitGiven   :1;
    unsigned  BSIM4v6wnfactorGiven   :1;
    unsigned  BSIM4v6wxjGiven   :1;
    unsigned  BSIM4v6wvsatGiven   :1;
    unsigned  BSIM4v6watGiven   :1;
    unsigned  BSIM4v6wa0Given   :1;
    unsigned  BSIM4v6wagsGiven   :1;
    unsigned  BSIM4v6wa1Given   :1;
    unsigned  BSIM4v6wa2Given   :1;
    unsigned  BSIM4v6wketaGiven   :1;    
    unsigned  BSIM4v6wnsubGiven   :1;
    unsigned  BSIM4v6wndepGiven   :1;
    unsigned  BSIM4v6wnsdGiven    :1;
    unsigned  BSIM4v6wphinGiven   :1;
    unsigned  BSIM4v6wngateGiven   :1;
    unsigned  BSIM4v6wgamma1Given   :1;
    unsigned  BSIM4v6wgamma2Given   :1;
    unsigned  BSIM4v6wvbxGiven   :1;
    unsigned  BSIM4v6wvbmGiven   :1;
    unsigned  BSIM4v6wxtGiven   :1;
    unsigned  BSIM4v6wk1Given   :1;
    unsigned  BSIM4v6wkt1Given   :1;
    unsigned  BSIM4v6wkt1lGiven   :1;
    unsigned  BSIM4v6wkt2Given   :1;
    unsigned  BSIM4v6wk2Given   :1;
    unsigned  BSIM4v6wk3Given   :1;
    unsigned  BSIM4v6wk3bGiven   :1;
    unsigned  BSIM4v6ww0Given   :1;
    unsigned  BSIM4v6wdvtp0Given :1;
    unsigned  BSIM4v6wdvtp1Given :1;
    unsigned  BSIM4v6wlpe0Given   :1;
    unsigned  BSIM4v6wlpebGiven   :1;
    unsigned  BSIM4v6wdvt0Given   :1;   
    unsigned  BSIM4v6wdvt1Given   :1;     
    unsigned  BSIM4v6wdvt2Given   :1;     
    unsigned  BSIM4v6wdvt0wGiven   :1;   
    unsigned  BSIM4v6wdvt1wGiven   :1;     
    unsigned  BSIM4v6wdvt2wGiven   :1;     
    unsigned  BSIM4v6wdroutGiven   :1;     
    unsigned  BSIM4v6wdsubGiven   :1;     
    unsigned  BSIM4v6wvth0Given   :1;
    unsigned  BSIM4v6wuaGiven   :1;
    unsigned  BSIM4v6wua1Given   :1;
    unsigned  BSIM4v6wubGiven   :1;
    unsigned  BSIM4v6wub1Given   :1;
    unsigned  BSIM4v6wucGiven   :1;
    unsigned  BSIM4v6wuc1Given   :1;
    unsigned  BSIM4v6wudGiven     :1;
    unsigned  BSIM4v6wud1Given     :1;
    unsigned  BSIM4v6wupGiven     :1;
    unsigned  BSIM4v6wlpGiven     :1;
    unsigned  BSIM4v6wu0Given   :1;
    unsigned  BSIM4v6weuGiven   :1;
	unsigned  BSIM4v6wucsGiven  :1;
    unsigned  BSIM4v6wuteGiven   :1;
	unsigned  BSIM4v6wucsteGiven  :1;
    unsigned  BSIM4v6wvoffGiven   :1;
    unsigned  BSIM4v6wtvoffGiven   :1;
    unsigned  BSIM4v6wminvGiven   :1;
    unsigned  BSIM4v6wminvcvGiven   :1;
    unsigned  BSIM4v6wrdswGiven   :1;      
    unsigned  BSIM4v6wrswGiven   :1;
    unsigned  BSIM4v6wrdwGiven   :1;
    unsigned  BSIM4v6wprwgGiven   :1;      
    unsigned  BSIM4v6wprwbGiven   :1;      
    unsigned  BSIM4v6wprtGiven   :1;      
    unsigned  BSIM4v6weta0Given   :1;    
    unsigned  BSIM4v6wetabGiven   :1;    
    unsigned  BSIM4v6wpclmGiven   :1;   
    unsigned  BSIM4v6wpdibl1Given   :1;   
    unsigned  BSIM4v6wpdibl2Given   :1;  
    unsigned  BSIM4v6wpdiblbGiven   :1;  
    unsigned  BSIM4v6wfproutGiven   :1;
    unsigned  BSIM4v6wpditsGiven    :1;
    unsigned  BSIM4v6wpditsdGiven    :1;
    unsigned  BSIM4v6wpscbe1Given   :1;    
    unsigned  BSIM4v6wpscbe2Given   :1;    
    unsigned  BSIM4v6wpvagGiven   :1;    
    unsigned  BSIM4v6wdeltaGiven  :1;     
    unsigned  BSIM4v6wwrGiven   :1;
    unsigned  BSIM4v6wdwgGiven   :1;
    unsigned  BSIM4v6wdwbGiven   :1;
    unsigned  BSIM4v6wb0Given   :1;
    unsigned  BSIM4v6wb1Given   :1;
    unsigned  BSIM4v6walpha0Given   :1;
    unsigned  BSIM4v6walpha1Given   :1;
    unsigned  BSIM4v6wbeta0Given   :1;
    unsigned  BSIM4v6wvfbGiven   :1;
    unsigned  BSIM4v6wagidlGiven   :1;
    unsigned  BSIM4v6wbgidlGiven   :1;
    unsigned  BSIM4v6wcgidlGiven   :1;
    unsigned  BSIM4v6wegidlGiven   :1;
    unsigned  BSIM4v6wagislGiven   :1;
    unsigned  BSIM4v6wbgislGiven   :1;
    unsigned  BSIM4v6wcgislGiven   :1;
    unsigned  BSIM4v6wegislGiven   :1;
    unsigned  BSIM4v6waigcGiven   :1;
    unsigned  BSIM4v6wbigcGiven   :1;
    unsigned  BSIM4v6wcigcGiven   :1;
    unsigned  BSIM4v6waigsdGiven   :1;
    unsigned  BSIM4v6wbigsdGiven   :1;
    unsigned  BSIM4v6wcigsdGiven   :1;
    unsigned  BSIM4v6waigsGiven   :1;
    unsigned  BSIM4v6wbigsGiven   :1;
    unsigned  BSIM4v6wcigsGiven   :1;
    unsigned  BSIM4v6waigdGiven   :1;
    unsigned  BSIM4v6wbigdGiven   :1;
    unsigned  BSIM4v6wcigdGiven   :1;
    unsigned  BSIM4v6waigbaccGiven   :1;
    unsigned  BSIM4v6wbigbaccGiven   :1;
    unsigned  BSIM4v6wcigbaccGiven   :1;
    unsigned  BSIM4v6waigbinvGiven   :1;
    unsigned  BSIM4v6wbigbinvGiven   :1;
    unsigned  BSIM4v6wcigbinvGiven   :1;
    unsigned  BSIM4v6wnigcGiven   :1;
    unsigned  BSIM4v6wnigbinvGiven   :1;
    unsigned  BSIM4v6wnigbaccGiven   :1;
    unsigned  BSIM4v6wntoxGiven   :1;
    unsigned  BSIM4v6weigbinvGiven   :1;
    unsigned  BSIM4v6wpigcdGiven   :1;
    unsigned  BSIM4v6wpoxedgeGiven   :1;
    unsigned  BSIM4v6wxrcrg1Given   :1;
    unsigned  BSIM4v6wxrcrg2Given   :1;
    unsigned  BSIM4v6wlambdaGiven    :1;
    unsigned  BSIM4v6wvtlGiven    :1;
    unsigned  BSIM4v6wxnGiven    :1;
    unsigned  BSIM4v6wvfbsdoffGiven    :1;
    unsigned  BSIM4v6wtvfbsdoffGiven    :1;

    /* CV model */
    unsigned  BSIM4v6wcgslGiven   :1;
    unsigned  BSIM4v6wcgdlGiven   :1;
    unsigned  BSIM4v6wckappasGiven   :1;
    unsigned  BSIM4v6wckappadGiven   :1;
    unsigned  BSIM4v6wcfGiven   :1;
    unsigned  BSIM4v6wclcGiven   :1;
    unsigned  BSIM4v6wcleGiven   :1;
    unsigned  BSIM4v6wvfbcvGiven   :1;
    unsigned  BSIM4v6wnoffGiven   :1;
    unsigned  BSIM4v6wvoffcvGiven :1;
    unsigned  BSIM4v6wacdeGiven   :1;
    unsigned  BSIM4v6wmoinGiven   :1;

    /* Cross-term dependence */
    unsigned  BSIM4v6pcdscGiven   :1;
    unsigned  BSIM4v6pcdscbGiven   :1;
    unsigned  BSIM4v6pcdscdGiven   :1;
    unsigned  BSIM4v6pcitGiven   :1;
    unsigned  BSIM4v6pnfactorGiven   :1;
    unsigned  BSIM4v6pxjGiven   :1;
    unsigned  BSIM4v6pvsatGiven   :1;
    unsigned  BSIM4v6patGiven   :1;
    unsigned  BSIM4v6pa0Given   :1;
    unsigned  BSIM4v6pagsGiven   :1;
    unsigned  BSIM4v6pa1Given   :1;
    unsigned  BSIM4v6pa2Given   :1;
    unsigned  BSIM4v6pketaGiven   :1;    
    unsigned  BSIM4v6pnsubGiven   :1;
    unsigned  BSIM4v6pndepGiven   :1;
    unsigned  BSIM4v6pnsdGiven    :1;
    unsigned  BSIM4v6pphinGiven   :1;
    unsigned  BSIM4v6pngateGiven   :1;
    unsigned  BSIM4v6pgamma1Given   :1;
    unsigned  BSIM4v6pgamma2Given   :1;
    unsigned  BSIM4v6pvbxGiven   :1;
    unsigned  BSIM4v6pvbmGiven   :1;
    unsigned  BSIM4v6pxtGiven   :1;
    unsigned  BSIM4v6pk1Given   :1;
    unsigned  BSIM4v6pkt1Given   :1;
    unsigned  BSIM4v6pkt1lGiven   :1;
    unsigned  BSIM4v6pkt2Given   :1;
    unsigned  BSIM4v6pk2Given   :1;
    unsigned  BSIM4v6pk3Given   :1;
    unsigned  BSIM4v6pk3bGiven   :1;
    unsigned  BSIM4v6pw0Given   :1;
    unsigned  BSIM4v6pdvtp0Given :1;
    unsigned  BSIM4v6pdvtp1Given :1;
    unsigned  BSIM4v6plpe0Given   :1;
    unsigned  BSIM4v6plpebGiven   :1;
    unsigned  BSIM4v6pdvt0Given   :1;   
    unsigned  BSIM4v6pdvt1Given   :1;     
    unsigned  BSIM4v6pdvt2Given   :1;     
    unsigned  BSIM4v6pdvt0wGiven   :1;   
    unsigned  BSIM4v6pdvt1wGiven   :1;     
    unsigned  BSIM4v6pdvt2wGiven   :1;     
    unsigned  BSIM4v6pdroutGiven   :1;     
    unsigned  BSIM4v6pdsubGiven   :1;     
    unsigned  BSIM4v6pvth0Given   :1;
    unsigned  BSIM4v6puaGiven   :1;
    unsigned  BSIM4v6pua1Given   :1;
    unsigned  BSIM4v6pubGiven   :1;
    unsigned  BSIM4v6pub1Given   :1;
    unsigned  BSIM4v6pucGiven   :1;
    unsigned  BSIM4v6puc1Given   :1;
    unsigned  BSIM4v6pudGiven     :1;
    unsigned  BSIM4v6pud1Given     :1;
    unsigned  BSIM4v6pupGiven     :1;
    unsigned  BSIM4v6plpGiven     :1;
    unsigned  BSIM4v6pu0Given   :1;
    unsigned  BSIM4v6peuGiven   :1;
	unsigned  BSIM4v6pucsGiven   :1;
    unsigned  BSIM4v6puteGiven   :1;
	unsigned  BSIM4v6pucsteGiven  :1;
    unsigned  BSIM4v6pvoffGiven   :1;
    unsigned  BSIM4v6ptvoffGiven   :1;
    unsigned  BSIM4v6pminvGiven   :1;
    unsigned  BSIM4v6pminvcvGiven   :1;
    unsigned  BSIM4v6prdswGiven   :1;      
    unsigned  BSIM4v6prswGiven   :1;
    unsigned  BSIM4v6prdwGiven   :1;
    unsigned  BSIM4v6pprwgGiven   :1;      
    unsigned  BSIM4v6pprwbGiven   :1;      
    unsigned  BSIM4v6pprtGiven   :1;      
    unsigned  BSIM4v6peta0Given   :1;    
    unsigned  BSIM4v6petabGiven   :1;    
    unsigned  BSIM4v6ppclmGiven   :1;   
    unsigned  BSIM4v6ppdibl1Given   :1;   
    unsigned  BSIM4v6ppdibl2Given   :1;  
    unsigned  BSIM4v6ppdiblbGiven   :1;  
    unsigned  BSIM4v6pfproutGiven   :1;
    unsigned  BSIM4v6ppditsGiven    :1;
    unsigned  BSIM4v6ppditsdGiven    :1;
    unsigned  BSIM4v6ppscbe1Given   :1;    
    unsigned  BSIM4v6ppscbe2Given   :1;    
    unsigned  BSIM4v6ppvagGiven   :1;    
    unsigned  BSIM4v6pdeltaGiven  :1;     
    unsigned  BSIM4v6pwrGiven   :1;
    unsigned  BSIM4v6pdwgGiven   :1;
    unsigned  BSIM4v6pdwbGiven   :1;
    unsigned  BSIM4v6pb0Given   :1;
    unsigned  BSIM4v6pb1Given   :1;
    unsigned  BSIM4v6palpha0Given   :1;
    unsigned  BSIM4v6palpha1Given   :1;
    unsigned  BSIM4v6pbeta0Given   :1;
    unsigned  BSIM4v6pvfbGiven   :1;
    unsigned  BSIM4v6pagidlGiven   :1;
    unsigned  BSIM4v6pbgidlGiven   :1;
    unsigned  BSIM4v6pcgidlGiven   :1;
    unsigned  BSIM4v6pegidlGiven   :1;
    unsigned  BSIM4v6pagislGiven   :1;
    unsigned  BSIM4v6pbgislGiven   :1;
    unsigned  BSIM4v6pcgislGiven   :1;
    unsigned  BSIM4v6pegislGiven   :1;
    unsigned  BSIM4v6paigcGiven   :1;
    unsigned  BSIM4v6pbigcGiven   :1;
    unsigned  BSIM4v6pcigcGiven   :1;
    unsigned  BSIM4v6paigsdGiven   :1;
    unsigned  BSIM4v6pbigsdGiven   :1;
    unsigned  BSIM4v6pcigsdGiven   :1;
    unsigned  BSIM4v6paigsGiven   :1;
    unsigned  BSIM4v6pbigsGiven   :1;
    unsigned  BSIM4v6pcigsGiven   :1;
    unsigned  BSIM4v6paigdGiven   :1;
    unsigned  BSIM4v6pbigdGiven   :1;
    unsigned  BSIM4v6pcigdGiven   :1;
    unsigned  BSIM4v6paigbaccGiven   :1;
    unsigned  BSIM4v6pbigbaccGiven   :1;
    unsigned  BSIM4v6pcigbaccGiven   :1;
    unsigned  BSIM4v6paigbinvGiven   :1;
    unsigned  BSIM4v6pbigbinvGiven   :1;
    unsigned  BSIM4v6pcigbinvGiven   :1;
    unsigned  BSIM4v6pnigcGiven   :1;
    unsigned  BSIM4v6pnigbinvGiven   :1;
    unsigned  BSIM4v6pnigbaccGiven   :1;
    unsigned  BSIM4v6pntoxGiven   :1;
    unsigned  BSIM4v6peigbinvGiven   :1;
    unsigned  BSIM4v6ppigcdGiven   :1;
    unsigned  BSIM4v6ppoxedgeGiven   :1;
    unsigned  BSIM4v6pxrcrg1Given   :1;
    unsigned  BSIM4v6pxrcrg2Given   :1;
    unsigned  BSIM4v6plambdaGiven    :1;
    unsigned  BSIM4v6pvtlGiven    :1;
    unsigned  BSIM4v6pxnGiven    :1;
    unsigned  BSIM4v6pvfbsdoffGiven    :1;
    unsigned  BSIM4v6ptvfbsdoffGiven    :1;

    /* CV model */
    unsigned  BSIM4v6pcgslGiven   :1;
    unsigned  BSIM4v6pcgdlGiven   :1;
    unsigned  BSIM4v6pckappasGiven   :1;
    unsigned  BSIM4v6pckappadGiven   :1;
    unsigned  BSIM4v6pcfGiven   :1;
    unsigned  BSIM4v6pclcGiven   :1;
    unsigned  BSIM4v6pcleGiven   :1;
    unsigned  BSIM4v6pvfbcvGiven   :1;
    unsigned  BSIM4v6pnoffGiven   :1;
    unsigned  BSIM4v6pvoffcvGiven :1;
    unsigned  BSIM4v6pacdeGiven   :1;
    unsigned  BSIM4v6pmoinGiven   :1;

    unsigned  BSIM4v6useFringeGiven   :1;

    unsigned  BSIM4v6tnomGiven   :1;
    unsigned  BSIM4v6cgsoGiven   :1;
    unsigned  BSIM4v6cgdoGiven   :1;
    unsigned  BSIM4v6cgboGiven   :1;
    unsigned  BSIM4v6xpartGiven   :1;
    unsigned  BSIM4v6sheetResistanceGiven   :1;

    unsigned  BSIM4v6SjctSatCurDensityGiven   :1;
    unsigned  BSIM4v6SjctSidewallSatCurDensityGiven   :1;
    unsigned  BSIM4v6SjctGateSidewallSatCurDensityGiven   :1;
    unsigned  BSIM4v6SbulkJctPotentialGiven   :1;
    unsigned  BSIM4v6SbulkJctBotGradingCoeffGiven   :1;
    unsigned  BSIM4v6SsidewallJctPotentialGiven   :1;
    unsigned  BSIM4v6SGatesidewallJctPotentialGiven   :1;
    unsigned  BSIM4v6SbulkJctSideGradingCoeffGiven   :1;
    unsigned  BSIM4v6SunitAreaJctCapGiven   :1;
    unsigned  BSIM4v6SunitLengthSidewallJctCapGiven   :1;
    unsigned  BSIM4v6SbulkJctGateSideGradingCoeffGiven   :1;
    unsigned  BSIM4v6SunitLengthGateSidewallJctCapGiven   :1;
    unsigned  BSIM4v6SjctEmissionCoeffGiven :1;
    unsigned  BSIM4v6SjctTempExponentGiven	:1;

    unsigned  BSIM4v6DjctSatCurDensityGiven   :1;
    unsigned  BSIM4v6DjctSidewallSatCurDensityGiven   :1;
    unsigned  BSIM4v6DjctGateSidewallSatCurDensityGiven   :1;
    unsigned  BSIM4v6DbulkJctPotentialGiven   :1;
    unsigned  BSIM4v6DbulkJctBotGradingCoeffGiven   :1;
    unsigned  BSIM4v6DsidewallJctPotentialGiven   :1;
    unsigned  BSIM4v6DGatesidewallJctPotentialGiven   :1;
    unsigned  BSIM4v6DbulkJctSideGradingCoeffGiven   :1;
    unsigned  BSIM4v6DunitAreaJctCapGiven   :1;
    unsigned  BSIM4v6DunitLengthSidewallJctCapGiven   :1;
    unsigned  BSIM4v6DbulkJctGateSideGradingCoeffGiven   :1;
    unsigned  BSIM4v6DunitLengthGateSidewallJctCapGiven   :1;
    unsigned  BSIM4v6DjctEmissionCoeffGiven :1;
    unsigned  BSIM4v6DjctTempExponentGiven :1;

    unsigned  BSIM4v6oxideTrapDensityAGiven  :1;         
    unsigned  BSIM4v6oxideTrapDensityBGiven  :1;        
    unsigned  BSIM4v6oxideTrapDensityCGiven  :1;     
    unsigned  BSIM4v6emGiven  :1;     
    unsigned  BSIM4v6efGiven  :1;     
    unsigned  BSIM4v6afGiven  :1;     
    unsigned  BSIM4v6kfGiven  :1;     

    unsigned  BSIM4v6vgsMaxGiven  :1;
    unsigned  BSIM4v6vgdMaxGiven  :1;
    unsigned  BSIM4v6vgbMaxGiven  :1;
    unsigned  BSIM4v6vdsMaxGiven  :1;
    unsigned  BSIM4v6vbsMaxGiven  :1;
    unsigned  BSIM4v6vbdMaxGiven  :1;
    unsigned  BSIM4v6vgsrMaxGiven  :1;
    unsigned  BSIM4v6vgdrMaxGiven  :1;
    unsigned  BSIM4v6vgbrMaxGiven  :1;
    unsigned  BSIM4v6vbsrMaxGiven  :1;
    unsigned  BSIM4v6vbdrMaxGiven  :1;

    unsigned  BSIM4v6LintGiven   :1;
    unsigned  BSIM4v6LlGiven   :1;
    unsigned  BSIM4v6LlcGiven   :1;
    unsigned  BSIM4v6LlnGiven   :1;
    unsigned  BSIM4v6LwGiven   :1;
    unsigned  BSIM4v6LwcGiven   :1;
    unsigned  BSIM4v6LwnGiven   :1;
    unsigned  BSIM4v6LwlGiven   :1;
    unsigned  BSIM4v6LwlcGiven   :1;
    unsigned  BSIM4v6LminGiven   :1;
    unsigned  BSIM4v6LmaxGiven   :1;

    unsigned  BSIM4v6WintGiven   :1;
    unsigned  BSIM4v6WlGiven   :1;
    unsigned  BSIM4v6WlcGiven   :1;
    unsigned  BSIM4v6WlnGiven   :1;
    unsigned  BSIM4v6WwGiven   :1;
    unsigned  BSIM4v6WwcGiven   :1;
    unsigned  BSIM4v6WwnGiven   :1;
    unsigned  BSIM4v6WwlGiven   :1;
    unsigned  BSIM4v6WwlcGiven   :1;
    unsigned  BSIM4v6WminGiven   :1;
    unsigned  BSIM4v6WmaxGiven   :1;

    /* added for stress effect */
    unsigned  BSIM4v6sarefGiven   :1;
    unsigned  BSIM4v6sbrefGiven   :1;
    unsigned  BSIM4v6wlodGiven  :1;
    unsigned  BSIM4v6ku0Given   :1;
    unsigned  BSIM4v6kvsatGiven  :1;
    unsigned  BSIM4v6kvth0Given  :1;
    unsigned  BSIM4v6tku0Given   :1;
    unsigned  BSIM4v6llodku0Given   :1;
    unsigned  BSIM4v6wlodku0Given   :1;
    unsigned  BSIM4v6llodvthGiven   :1;
    unsigned  BSIM4v6wlodvthGiven   :1;
    unsigned  BSIM4v6lku0Given   :1;
    unsigned  BSIM4v6wku0Given   :1;
    unsigned  BSIM4v6pku0Given   :1;
    unsigned  BSIM4v6lkvth0Given   :1;
    unsigned  BSIM4v6wkvth0Given   :1;
    unsigned  BSIM4v6pkvth0Given   :1;
    unsigned  BSIM4v6stk2Given   :1;
    unsigned  BSIM4v6lodk2Given  :1;
    unsigned  BSIM4v6steta0Given :1;
    unsigned  BSIM4v6lodeta0Given :1;

    unsigned  BSIM4v6webGiven   :1;
    unsigned  BSIM4v6wecGiven   :1;
    unsigned  BSIM4v6kvth0weGiven   :1;
    unsigned  BSIM4v6k2weGiven   :1;
    unsigned  BSIM4v6ku0weGiven   :1;
    unsigned  BSIM4v6screfGiven   :1;
    unsigned  BSIM4v6wpemodGiven   :1;
    unsigned  BSIM4v6lkvth0weGiven   :1;
    unsigned  BSIM4v6lk2weGiven   :1;
    unsigned  BSIM4v6lku0weGiven   :1;
    unsigned  BSIM4v6wkvth0weGiven   :1;
    unsigned  BSIM4v6wk2weGiven   :1;
    unsigned  BSIM4v6wku0weGiven   :1;
    unsigned  BSIM4v6pkvth0weGiven   :1;
    unsigned  BSIM4v6pk2weGiven   :1;
    unsigned  BSIM4v6pku0weGiven   :1;


} BSIM4v6model;


#ifndef NMOS
#define NMOS 1
#define PMOS -1
#endif /*NMOS*/


/* Instance parameters */
#define BSIM4v6_W                   1
#define BSIM4v6_L                   2
#define BSIM4v6_AS                  3
#define BSIM4v6_AD                  4
#define BSIM4v6_PS                  5
#define BSIM4v6_PD                  6
#define BSIM4v6_NRS                 7
#define BSIM4v6_NRD                 8
#define BSIM4v6_OFF                 9
#define BSIM4v6_IC                  10
#define BSIM4v6_IC_VDS              11
#define BSIM4v6_IC_VGS              12
#define BSIM4v6_IC_VBS              13
#define BSIM4v6_TRNQSMOD            14
#define BSIM4v6_RBODYMOD            15
#define BSIM4v6_RGATEMOD            16
#define BSIM4v6_GEOMOD              17
#define BSIM4v6_RGEOMOD             18
#define BSIM4v6_NF                  19
#define BSIM4v6_MIN                 20 
#define BSIM4v6_ACNQSMOD            22
#define BSIM4v6_RBDB                23
#define BSIM4v6_RBSB                24
#define BSIM4v6_RBPB                25
#define BSIM4v6_RBPS                26
#define BSIM4v6_RBPD                27
#define BSIM4v6_SA                  28
#define BSIM4v6_SB                  29
#define BSIM4v6_SD                  30
#define BSIM4v6_DELVTO              31
#define BSIM4v6_XGW                 32
#define BSIM4v6_NGCON               33
#define BSIM4v6_SCA                 34
#define BSIM4v6_SCB                 35
#define BSIM4v6_SCC                 36 
#define BSIM4v6_SC                  37
#define BSIM4v6_M                   38
#define BSIM4v6_MULU0               39

/* Global parameters */
#define BSIM4v6_MOD_TEMPEOT         65
#define BSIM4v6_MOD_LEFFEOT         66
#define BSIM4v6_MOD_WEFFEOT         67
#define BSIM4v6_MOD_UCSTE           68
#define BSIM4v6_MOD_LUCSTE          69
#define BSIM4v6_MOD_WUCSTE          70
#define BSIM4v6_MOD_PUCSTE          71
#define BSIM4v6_MOD_UCS             72
#define BSIM4v6_MOD_LUCS            73
#define BSIM4v6_MOD_WUCS            74
#define BSIM4v6_MOD_PUCS            75
#define BSIM4v6_MOD_CVCHARGEMOD     76
#define BSIM4v6_MOD_ADOS            77
#define BSIM4v6_MOD_BDOS            78
#define BSIM4v6_MOD_TEMPMOD         79
#define BSIM4v6_MOD_MTRLMOD         80
#define BSIM4v6_MOD_IGCMOD          81
#define BSIM4v6_MOD_IGBMOD          82
#define BSIM4v6_MOD_ACNQSMOD        83
#define BSIM4v6_MOD_FNOIMOD         84
#define BSIM4v6_MOD_RDSMOD          85
#define BSIM4v6_MOD_DIOMOD          86
#define BSIM4v6_MOD_PERMOD          87
#define BSIM4v6_MOD_GEOMOD          88
#define BSIM4v6_MOD_RGEOMOD         89
#define BSIM4v6_MOD_RGATEMOD        90
#define BSIM4v6_MOD_RBODYMOD        91
#define BSIM4v6_MOD_CAPMOD          92
#define BSIM4v6_MOD_TRNQSMOD        93
#define BSIM4v6_MOD_MOBMOD          94    
#define BSIM4v6_MOD_TNOIMOD         95
#define BSIM4v6_MOD_EOT             96
#define BSIM4v6_MOD_VDDEOT          97    
#define BSIM4v6_MOD_TOXE            98
#define BSIM4v6_MOD_CDSC            99
#define BSIM4v6_MOD_CDSCB           100
#define BSIM4v6_MOD_CIT             101
#define BSIM4v6_MOD_NFACTOR         102
#define BSIM4v6_MOD_XJ              103
#define BSIM4v6_MOD_VSAT            104
#define BSIM4v6_MOD_AT              105
#define BSIM4v6_MOD_A0              106
#define BSIM4v6_MOD_A1              107
#define BSIM4v6_MOD_A2              108
#define BSIM4v6_MOD_KETA            109   
#define BSIM4v6_MOD_NSUB            110
#define BSIM4v6_MOD_PHIG            111
#define BSIM4v6_MOD_EPSRGATE        112
#define BSIM4v6_MOD_EASUB           113
#define BSIM4v6_MOD_EPSRSUB         114
#define BSIM4v6_MOD_NI0SUB          115
#define BSIM4v6_MOD_BG0SUB          116
#define BSIM4v6_MOD_TBGASUB         117
#define BSIM4v6_MOD_TBGBSUB         118
#define BSIM4v6_MOD_NDEP            119
#define BSIM4v6_MOD_NGATE           120
#define BSIM4v6_MOD_GAMMA1          121
#define BSIM4v6_MOD_GAMMA2          122
#define BSIM4v6_MOD_VBX             123
#define BSIM4v6_MOD_BINUNIT         124    
#define BSIM4v6_MOD_VBM             125
#define BSIM4v6_MOD_XT              126
#define BSIM4v6_MOD_K1              129
#define BSIM4v6_MOD_KT1             130
#define BSIM4v6_MOD_KT1L            131
#define BSIM4v6_MOD_K2              132
#define BSIM4v6_MOD_KT2             133
#define BSIM4v6_MOD_K3              134
#define BSIM4v6_MOD_K3B             135
#define BSIM4v6_MOD_W0              136
#define BSIM4v6_MOD_LPE0            137
#define BSIM4v6_MOD_DVT0            138
#define BSIM4v6_MOD_DVT1            139
#define BSIM4v6_MOD_DVT2            140
#define BSIM4v6_MOD_DVT0W           141
#define BSIM4v6_MOD_DVT1W           142
#define BSIM4v6_MOD_DVT2W           143
#define BSIM4v6_MOD_DROUT           144
#define BSIM4v6_MOD_DSUB            145
#define BSIM4v6_MOD_VTH0            146
#define BSIM4v6_MOD_UA              147
#define BSIM4v6_MOD_UA1             148
#define BSIM4v6_MOD_UB              149
#define BSIM4v6_MOD_UB1             150
#define BSIM4v6_MOD_UC              151
#define BSIM4v6_MOD_UC1             152
#define BSIM4v6_MOD_U0              153
#define BSIM4v6_MOD_UTE             154
#define BSIM4v6_MOD_VOFF            155
#define BSIM4v6_MOD_DELTA           156
#define BSIM4v6_MOD_RDSW            157
#define BSIM4v6_MOD_PRT             158
#define BSIM4v6_MOD_LDD             159
#define BSIM4v6_MOD_ETA             160
#define BSIM4v6_MOD_ETA0            161
#define BSIM4v6_MOD_ETAB            162
#define BSIM4v6_MOD_PCLM            163
#define BSIM4v6_MOD_PDIBL1          164
#define BSIM4v6_MOD_PDIBL2          165
#define BSIM4v6_MOD_PSCBE1          166
#define BSIM4v6_MOD_PSCBE2          167
#define BSIM4v6_MOD_PVAG            168
#define BSIM4v6_MOD_WR              169
#define BSIM4v6_MOD_DWG             170
#define BSIM4v6_MOD_DWB             171
#define BSIM4v6_MOD_B0              172
#define BSIM4v6_MOD_B1              173
#define BSIM4v6_MOD_ALPHA0          174
#define BSIM4v6_MOD_BETA0           175
#define BSIM4v6_MOD_PDIBLB          178
#define BSIM4v6_MOD_PRWG            179
#define BSIM4v6_MOD_PRWB            180
#define BSIM4v6_MOD_CDSCD           181
#define BSIM4v6_MOD_AGS             182
#define BSIM4v6_MOD_FRINGE          184
#define BSIM4v6_MOD_CGSL            186
#define BSIM4v6_MOD_CGDL            187
#define BSIM4v6_MOD_CKAPPAS         188
#define BSIM4v6_MOD_CF              189
#define BSIM4v6_MOD_CLC             190
#define BSIM4v6_MOD_CLE             191
#define BSIM4v6_MOD_PARAMCHK        192
#define BSIM4v6_MOD_VERSION         193
#define BSIM4v6_MOD_VFBCV           194
#define BSIM4v6_MOD_ACDE            195
#define BSIM4v6_MOD_MOIN            196
#define BSIM4v6_MOD_NOFF            197
#define BSIM4v6_MOD_IJTHDFWD        198
#define BSIM4v6_MOD_ALPHA1          199
#define BSIM4v6_MOD_VFB             200
#define BSIM4v6_MOD_TOXM            201
#define BSIM4v6_MOD_TCJ             202
#define BSIM4v6_MOD_TCJSW           203
#define BSIM4v6_MOD_TCJSWG          204
#define BSIM4v6_MOD_TPB             205
#define BSIM4v6_MOD_TPBSW           206
#define BSIM4v6_MOD_TPBSWG          207
#define BSIM4v6_MOD_VOFFCV          208
#define BSIM4v6_MOD_GBMIN           209
#define BSIM4v6_MOD_RBDB            210
#define BSIM4v6_MOD_RBSB            211
#define BSIM4v6_MOD_RBPB            212
#define BSIM4v6_MOD_RBPS            213
#define BSIM4v6_MOD_RBPD            214
#define BSIM4v6_MOD_DMCG            215
#define BSIM4v6_MOD_DMCI            216
#define BSIM4v6_MOD_DMDG            217
#define BSIM4v6_MOD_XGW             218
#define BSIM4v6_MOD_XGL             219
#define BSIM4v6_MOD_RSHG            220
#define BSIM4v6_MOD_NGCON           221
#define BSIM4v6_MOD_AGIDL           222
#define BSIM4v6_MOD_BGIDL           223
#define BSIM4v6_MOD_EGIDL           224
#define BSIM4v6_MOD_IJTHSFWD        225
#define BSIM4v6_MOD_XJBVD           226
#define BSIM4v6_MOD_XJBVS           227
#define BSIM4v6_MOD_BVD             228
#define BSIM4v6_MOD_BVS             229
#define BSIM4v6_MOD_TOXP            230
#define BSIM4v6_MOD_DTOX            231
#define BSIM4v6_MOD_XRCRG1          232
#define BSIM4v6_MOD_XRCRG2          233
#define BSIM4v6_MOD_EU              234
#define BSIM4v6_MOD_IJTHSREV        235
#define BSIM4v6_MOD_IJTHDREV        236
#define BSIM4v6_MOD_MINV            237
#define BSIM4v6_MOD_VOFFL           238
#define BSIM4v6_MOD_PDITS           239
#define BSIM4v6_MOD_PDITSD          240
#define BSIM4v6_MOD_PDITSL          241
#define BSIM4v6_MOD_TNOIA           242
#define BSIM4v6_MOD_TNOIB           243
#define BSIM4v6_MOD_NTNOI           244
#define BSIM4v6_MOD_FPROUT          245
#define BSIM4v6_MOD_LPEB            246
#define BSIM4v6_MOD_DVTP0           247
#define BSIM4v6_MOD_DVTP1           248
#define BSIM4v6_MOD_CGIDL           249
#define BSIM4v6_MOD_PHIN            250
#define BSIM4v6_MOD_RDSWMIN         251
#define BSIM4v6_MOD_RSW             252
#define BSIM4v6_MOD_RDW             253
#define BSIM4v6_MOD_RDWMIN          254
#define BSIM4v6_MOD_RSWMIN          255
#define BSIM4v6_MOD_NSD             256
#define BSIM4v6_MOD_CKAPPAD         257
#define BSIM4v6_MOD_DMCGT           258
#define BSIM4v6_MOD_AIGC            259
#define BSIM4v6_MOD_BIGC            260
#define BSIM4v6_MOD_CIGC            261
#define BSIM4v6_MOD_AIGBACC         262
#define BSIM4v6_MOD_BIGBACC         263
#define BSIM4v6_MOD_CIGBACC         264            
#define BSIM4v6_MOD_AIGBINV         265
#define BSIM4v6_MOD_BIGBINV         266
#define BSIM4v6_MOD_CIGBINV         267
#define BSIM4v6_MOD_NIGC            268
#define BSIM4v6_MOD_NIGBACC         269
#define BSIM4v6_MOD_NIGBINV         270
#define BSIM4v6_MOD_NTOX            271
#define BSIM4v6_MOD_TOXREF          272
#define BSIM4v6_MOD_EIGBINV         273
#define BSIM4v6_MOD_PIGCD           274
#define BSIM4v6_MOD_POXEDGE         275
#define BSIM4v6_MOD_EPSROX          276
#define BSIM4v6_MOD_AIGSD           277
#define BSIM4v6_MOD_BIGSD           278
#define BSIM4v6_MOD_CIGSD           279
#define BSIM4v6_MOD_JSWGS           280
#define BSIM4v6_MOD_JSWGD           281
#define BSIM4v6_MOD_LAMBDA          282
#define BSIM4v6_MOD_VTL             283
#define BSIM4v6_MOD_LC              284
#define BSIM4v6_MOD_XN              285
#define BSIM4v6_MOD_RNOIA           286
#define BSIM4v6_MOD_RNOIB           287
#define BSIM4v6_MOD_VFBSDOFF        288
#define BSIM4v6_MOD_LINTNOI         289
#define BSIM4v6_MOD_UD              290
#define BSIM4v6_MOD_UD1             291
#define BSIM4v6_MOD_UP              292
#define BSIM4v6_MOD_LP              293
#define BSIM4v6_MOD_TVOFF           294
#define BSIM4v6_MOD_TVFBSDOFF       295
#define BSIM4v6_MOD_MINVCV          296
#define BSIM4v6_MOD_VOFFCVL         297

/* Length dependence */
#define BSIM4v6_MOD_LCDSC            301
#define BSIM4v6_MOD_LCDSCB           302
#define BSIM4v6_MOD_LCIT             303
#define BSIM4v6_MOD_LNFACTOR         304
#define BSIM4v6_MOD_LXJ              305
#define BSIM4v6_MOD_LVSAT            306
#define BSIM4v6_MOD_LAT              307
#define BSIM4v6_MOD_LA0              308
#define BSIM4v6_MOD_LA1              309
#define BSIM4v6_MOD_LA2              310
#define BSIM4v6_MOD_LKETA            311
#define BSIM4v6_MOD_LNSUB            312
#define BSIM4v6_MOD_LNDEP            313
#define BSIM4v6_MOD_LNGATE           315
#define BSIM4v6_MOD_LGAMMA1          316
#define BSIM4v6_MOD_LGAMMA2          317
#define BSIM4v6_MOD_LVBX             318
#define BSIM4v6_MOD_LVBM             320
#define BSIM4v6_MOD_LXT              322
#define BSIM4v6_MOD_LK1              325
#define BSIM4v6_MOD_LKT1             326
#define BSIM4v6_MOD_LKT1L            327
#define BSIM4v6_MOD_LK2              328
#define BSIM4v6_MOD_LKT2             329
#define BSIM4v6_MOD_LK3              330
#define BSIM4v6_MOD_LK3B             331
#define BSIM4v6_MOD_LW0              332
#define BSIM4v6_MOD_LLPE0            333
#define BSIM4v6_MOD_LDVT0            334
#define BSIM4v6_MOD_LDVT1            335
#define BSIM4v6_MOD_LDVT2            336
#define BSIM4v6_MOD_LDVT0W           337
#define BSIM4v6_MOD_LDVT1W           338
#define BSIM4v6_MOD_LDVT2W           339
#define BSIM4v6_MOD_LDROUT           340
#define BSIM4v6_MOD_LDSUB            341
#define BSIM4v6_MOD_LVTH0            342
#define BSIM4v6_MOD_LUA              343
#define BSIM4v6_MOD_LUA1             344
#define BSIM4v6_MOD_LUB              345
#define BSIM4v6_MOD_LUB1             346
#define BSIM4v6_MOD_LUC              347
#define BSIM4v6_MOD_LUC1             348
#define BSIM4v6_MOD_LU0              349
#define BSIM4v6_MOD_LUTE             350
#define BSIM4v6_MOD_LVOFF            351
#define BSIM4v6_MOD_LDELTA           352
#define BSIM4v6_MOD_LRDSW            353
#define BSIM4v6_MOD_LPRT             354
#define BSIM4v6_MOD_LLDD             355
#define BSIM4v6_MOD_LETA             356
#define BSIM4v6_MOD_LETA0            357
#define BSIM4v6_MOD_LETAB            358
#define BSIM4v6_MOD_LPCLM            359
#define BSIM4v6_MOD_LPDIBL1          360
#define BSIM4v6_MOD_LPDIBL2          361
#define BSIM4v6_MOD_LPSCBE1          362
#define BSIM4v6_MOD_LPSCBE2          363
#define BSIM4v6_MOD_LPVAG            364
#define BSIM4v6_MOD_LWR              365
#define BSIM4v6_MOD_LDWG             366
#define BSIM4v6_MOD_LDWB             367
#define BSIM4v6_MOD_LB0              368
#define BSIM4v6_MOD_LB1              369
#define BSIM4v6_MOD_LALPHA0          370
#define BSIM4v6_MOD_LBETA0           371
#define BSIM4v6_MOD_LPDIBLB          374
#define BSIM4v6_MOD_LPRWG            375
#define BSIM4v6_MOD_LPRWB            376
#define BSIM4v6_MOD_LCDSCD           377
#define BSIM4v6_MOD_LAGS             378

#define BSIM4v6_MOD_LFRINGE          381
#define BSIM4v6_MOD_LCGSL            383
#define BSIM4v6_MOD_LCGDL            384
#define BSIM4v6_MOD_LCKAPPAS         385
#define BSIM4v6_MOD_LCF              386
#define BSIM4v6_MOD_LCLC             387
#define BSIM4v6_MOD_LCLE             388
#define BSIM4v6_MOD_LVFBCV           389
#define BSIM4v6_MOD_LACDE            390
#define BSIM4v6_MOD_LMOIN            391
#define BSIM4v6_MOD_LNOFF            392
#define BSIM4v6_MOD_LALPHA1          394
#define BSIM4v6_MOD_LVFB             395
#define BSIM4v6_MOD_LVOFFCV          396
#define BSIM4v6_MOD_LAGIDL           397
#define BSIM4v6_MOD_LBGIDL           398
#define BSIM4v6_MOD_LEGIDL           399
#define BSIM4v6_MOD_LXRCRG1          400
#define BSIM4v6_MOD_LXRCRG2          401
#define BSIM4v6_MOD_LEU              402
#define BSIM4v6_MOD_LMINV            403
#define BSIM4v6_MOD_LPDITS           404
#define BSIM4v6_MOD_LPDITSD          405
#define BSIM4v6_MOD_LFPROUT          406
#define BSIM4v6_MOD_LLPEB            407
#define BSIM4v6_MOD_LDVTP0           408
#define BSIM4v6_MOD_LDVTP1           409
#define BSIM4v6_MOD_LCGIDL           410
#define BSIM4v6_MOD_LPHIN            411
#define BSIM4v6_MOD_LRSW             412
#define BSIM4v6_MOD_LRDW             413
#define BSIM4v6_MOD_LNSD             414
#define BSIM4v6_MOD_LCKAPPAD         415
#define BSIM4v6_MOD_LAIGC            416
#define BSIM4v6_MOD_LBIGC            417
#define BSIM4v6_MOD_LCIGC            418
#define BSIM4v6_MOD_LAIGBACC         419            
#define BSIM4v6_MOD_LBIGBACC         420
#define BSIM4v6_MOD_LCIGBACC         421
#define BSIM4v6_MOD_LAIGBINV         422
#define BSIM4v6_MOD_LBIGBINV         423
#define BSIM4v6_MOD_LCIGBINV         424
#define BSIM4v6_MOD_LNIGC            425
#define BSIM4v6_MOD_LNIGBACC         426
#define BSIM4v6_MOD_LNIGBINV         427
#define BSIM4v6_MOD_LNTOX            428
#define BSIM4v6_MOD_LEIGBINV         429
#define BSIM4v6_MOD_LPIGCD           430
#define BSIM4v6_MOD_LPOXEDGE         431
#define BSIM4v6_MOD_LAIGSD           432
#define BSIM4v6_MOD_LBIGSD           433
#define BSIM4v6_MOD_LCIGSD           434

#define BSIM4v6_MOD_LLAMBDA          435
#define BSIM4v6_MOD_LVTL             436
#define BSIM4v6_MOD_LXN              437
#define BSIM4v6_MOD_LVFBSDOFF        438
#define BSIM4v6_MOD_LUD              439
#define BSIM4v6_MOD_LUD1             440
#define BSIM4v6_MOD_LUP              441
#define BSIM4v6_MOD_LLP              442
#define BSIM4v6_MOD_LMINVCV          443

/* Width dependence */
#define BSIM4v6_MOD_WCDSC            481
#define BSIM4v6_MOD_WCDSCB           482
#define BSIM4v6_MOD_WCIT             483
#define BSIM4v6_MOD_WNFACTOR         484
#define BSIM4v6_MOD_WXJ              485
#define BSIM4v6_MOD_WVSAT            486
#define BSIM4v6_MOD_WAT              487
#define BSIM4v6_MOD_WA0              488
#define BSIM4v6_MOD_WA1              489
#define BSIM4v6_MOD_WA2              490
#define BSIM4v6_MOD_WKETA            491
#define BSIM4v6_MOD_WNSUB            492
#define BSIM4v6_MOD_WNDEP            493
#define BSIM4v6_MOD_WNGATE           495
#define BSIM4v6_MOD_WGAMMA1          496
#define BSIM4v6_MOD_WGAMMA2          497
#define BSIM4v6_MOD_WVBX             498
#define BSIM4v6_MOD_WVBM             500
#define BSIM4v6_MOD_WXT              502
#define BSIM4v6_MOD_WK1              505
#define BSIM4v6_MOD_WKT1             506
#define BSIM4v6_MOD_WKT1L            507
#define BSIM4v6_MOD_WK2              508
#define BSIM4v6_MOD_WKT2             509
#define BSIM4v6_MOD_WK3              510
#define BSIM4v6_MOD_WK3B             511
#define BSIM4v6_MOD_WW0              512
#define BSIM4v6_MOD_WLPE0            513
#define BSIM4v6_MOD_WDVT0            514
#define BSIM4v6_MOD_WDVT1            515
#define BSIM4v6_MOD_WDVT2            516
#define BSIM4v6_MOD_WDVT0W           517
#define BSIM4v6_MOD_WDVT1W           518
#define BSIM4v6_MOD_WDVT2W           519
#define BSIM4v6_MOD_WDROUT           520
#define BSIM4v6_MOD_WDSUB            521
#define BSIM4v6_MOD_WVTH0            522
#define BSIM4v6_MOD_WUA              523
#define BSIM4v6_MOD_WUA1             524
#define BSIM4v6_MOD_WUB              525
#define BSIM4v6_MOD_WUB1             526
#define BSIM4v6_MOD_WUC              527
#define BSIM4v6_MOD_WUC1             528
#define BSIM4v6_MOD_WU0              529
#define BSIM4v6_MOD_WUTE             530
#define BSIM4v6_MOD_WVOFF            531
#define BSIM4v6_MOD_WDELTA           532
#define BSIM4v6_MOD_WRDSW            533
#define BSIM4v6_MOD_WPRT             534
#define BSIM4v6_MOD_WLDD             535
#define BSIM4v6_MOD_WETA             536
#define BSIM4v6_MOD_WETA0            537
#define BSIM4v6_MOD_WETAB            538
#define BSIM4v6_MOD_WPCLM            539
#define BSIM4v6_MOD_WPDIBL1          540
#define BSIM4v6_MOD_WPDIBL2          541
#define BSIM4v6_MOD_WPSCBE1          542
#define BSIM4v6_MOD_WPSCBE2          543
#define BSIM4v6_MOD_WPVAG            544
#define BSIM4v6_MOD_WWR              545
#define BSIM4v6_MOD_WDWG             546
#define BSIM4v6_MOD_WDWB             547
#define BSIM4v6_MOD_WB0              548
#define BSIM4v6_MOD_WB1              549
#define BSIM4v6_MOD_WALPHA0          550
#define BSIM4v6_MOD_WBETA0           551
#define BSIM4v6_MOD_WPDIBLB          554
#define BSIM4v6_MOD_WPRWG            555
#define BSIM4v6_MOD_WPRWB            556
#define BSIM4v6_MOD_WCDSCD           557
#define BSIM4v6_MOD_WAGS             558

#define BSIM4v6_MOD_WFRINGE          561
#define BSIM4v6_MOD_WCGSL            563
#define BSIM4v6_MOD_WCGDL            564
#define BSIM4v6_MOD_WCKAPPAS         565
#define BSIM4v6_MOD_WCF              566
#define BSIM4v6_MOD_WCLC             567
#define BSIM4v6_MOD_WCLE             568
#define BSIM4v6_MOD_WVFBCV           569
#define BSIM4v6_MOD_WACDE            570
#define BSIM4v6_MOD_WMOIN            571
#define BSIM4v6_MOD_WNOFF            572
#define BSIM4v6_MOD_WALPHA1          574
#define BSIM4v6_MOD_WVFB             575
#define BSIM4v6_MOD_WVOFFCV          576
#define BSIM4v6_MOD_WAGIDL           577
#define BSIM4v6_MOD_WBGIDL           578
#define BSIM4v6_MOD_WEGIDL           579
#define BSIM4v6_MOD_WXRCRG1          580
#define BSIM4v6_MOD_WXRCRG2          581
#define BSIM4v6_MOD_WEU              582
#define BSIM4v6_MOD_WMINV            583
#define BSIM4v6_MOD_WPDITS           584
#define BSIM4v6_MOD_WPDITSD          585
#define BSIM4v6_MOD_WFPROUT          586
#define BSIM4v6_MOD_WLPEB            587
#define BSIM4v6_MOD_WDVTP0           588
#define BSIM4v6_MOD_WDVTP1           589
#define BSIM4v6_MOD_WCGIDL           590
#define BSIM4v6_MOD_WPHIN            591
#define BSIM4v6_MOD_WRSW             592
#define BSIM4v6_MOD_WRDW             593
#define BSIM4v6_MOD_WNSD             594
#define BSIM4v6_MOD_WCKAPPAD         595
#define BSIM4v6_MOD_WAIGC            596
#define BSIM4v6_MOD_WBIGC            597
#define BSIM4v6_MOD_WCIGC            598
#define BSIM4v6_MOD_WAIGBACC         599            
#define BSIM4v6_MOD_WBIGBACC         600
#define BSIM4v6_MOD_WCIGBACC         601
#define BSIM4v6_MOD_WAIGBINV         602
#define BSIM4v6_MOD_WBIGBINV         603
#define BSIM4v6_MOD_WCIGBINV         604
#define BSIM4v6_MOD_WNIGC            605
#define BSIM4v6_MOD_WNIGBACC         606
#define BSIM4v6_MOD_WNIGBINV         607
#define BSIM4v6_MOD_WNTOX            608
#define BSIM4v6_MOD_WEIGBINV         609
#define BSIM4v6_MOD_WPIGCD           610
#define BSIM4v6_MOD_WPOXEDGE         611
#define BSIM4v6_MOD_WAIGSD           612
#define BSIM4v6_MOD_WBIGSD           613
#define BSIM4v6_MOD_WCIGSD           614
#define BSIM4v6_MOD_WLAMBDA          615
#define BSIM4v6_MOD_WVTL             616
#define BSIM4v6_MOD_WXN              617
#define BSIM4v6_MOD_WVFBSDOFF        618
#define BSIM4v6_MOD_WUD              619
#define BSIM4v6_MOD_WUD1             620
#define BSIM4v6_MOD_WUP              621
#define BSIM4v6_MOD_WLP              622
#define BSIM4v6_MOD_WMINVCV          623

/* Cross-term dependence */
#define BSIM4v6_MOD_PCDSC            661
#define BSIM4v6_MOD_PCDSCB           662
#define BSIM4v6_MOD_PCIT             663
#define BSIM4v6_MOD_PNFACTOR         664
#define BSIM4v6_MOD_PXJ              665
#define BSIM4v6_MOD_PVSAT            666
#define BSIM4v6_MOD_PAT              667
#define BSIM4v6_MOD_PA0              668
#define BSIM4v6_MOD_PA1              669
#define BSIM4v6_MOD_PA2              670
#define BSIM4v6_MOD_PKETA            671
#define BSIM4v6_MOD_PNSUB            672
#define BSIM4v6_MOD_PNDEP            673
#define BSIM4v6_MOD_PNGATE           675
#define BSIM4v6_MOD_PGAMMA1          676
#define BSIM4v6_MOD_PGAMMA2          677
#define BSIM4v6_MOD_PVBX             678

#define BSIM4v6_MOD_PVBM             680

#define BSIM4v6_MOD_PXT              682
#define BSIM4v6_MOD_PK1              685
#define BSIM4v6_MOD_PKT1             686
#define BSIM4v6_MOD_PKT1L            687
#define BSIM4v6_MOD_PK2              688
#define BSIM4v6_MOD_PKT2             689
#define BSIM4v6_MOD_PK3              690
#define BSIM4v6_MOD_PK3B             691
#define BSIM4v6_MOD_PW0              692
#define BSIM4v6_MOD_PLPE0            693

#define BSIM4v6_MOD_PDVT0            694
#define BSIM4v6_MOD_PDVT1            695
#define BSIM4v6_MOD_PDVT2            696

#define BSIM4v6_MOD_PDVT0W           697
#define BSIM4v6_MOD_PDVT1W           698
#define BSIM4v6_MOD_PDVT2W           699

#define BSIM4v6_MOD_PDROUT           700
#define BSIM4v6_MOD_PDSUB            701
#define BSIM4v6_MOD_PVTH0            702
#define BSIM4v6_MOD_PUA              703
#define BSIM4v6_MOD_PUA1             704
#define BSIM4v6_MOD_PUB              705
#define BSIM4v6_MOD_PUB1             706
#define BSIM4v6_MOD_PUC              707
#define BSIM4v6_MOD_PUC1             708
#define BSIM4v6_MOD_PU0              709
#define BSIM4v6_MOD_PUTE             710
#define BSIM4v6_MOD_PVOFF            711
#define BSIM4v6_MOD_PDELTA           712
#define BSIM4v6_MOD_PRDSW            713
#define BSIM4v6_MOD_PPRT             714
#define BSIM4v6_MOD_PLDD             715
#define BSIM4v6_MOD_PETA             716
#define BSIM4v6_MOD_PETA0            717
#define BSIM4v6_MOD_PETAB            718
#define BSIM4v6_MOD_PPCLM            719
#define BSIM4v6_MOD_PPDIBL1          720
#define BSIM4v6_MOD_PPDIBL2          721
#define BSIM4v6_MOD_PPSCBE1          722
#define BSIM4v6_MOD_PPSCBE2          723
#define BSIM4v6_MOD_PPVAG            724
#define BSIM4v6_MOD_PWR              725
#define BSIM4v6_MOD_PDWG             726
#define BSIM4v6_MOD_PDWB             727
#define BSIM4v6_MOD_PB0              728
#define BSIM4v6_MOD_PB1              729
#define BSIM4v6_MOD_PALPHA0          730
#define BSIM4v6_MOD_PBETA0           731
#define BSIM4v6_MOD_PPDIBLB          734

#define BSIM4v6_MOD_PPRWG            735
#define BSIM4v6_MOD_PPRWB            736

#define BSIM4v6_MOD_PCDSCD           737
#define BSIM4v6_MOD_PAGS             738

#define BSIM4v6_MOD_PFRINGE          741
#define BSIM4v6_MOD_PCGSL            743
#define BSIM4v6_MOD_PCGDL            744
#define BSIM4v6_MOD_PCKAPPAS         745
#define BSIM4v6_MOD_PCF              746
#define BSIM4v6_MOD_PCLC             747
#define BSIM4v6_MOD_PCLE             748
#define BSIM4v6_MOD_PVFBCV           749
#define BSIM4v6_MOD_PACDE            750
#define BSIM4v6_MOD_PMOIN            751
#define BSIM4v6_MOD_PNOFF            752
#define BSIM4v6_MOD_PALPHA1          754
#define BSIM4v6_MOD_PVFB             755
#define BSIM4v6_MOD_PVOFFCV          756
#define BSIM4v6_MOD_PAGIDL           757
#define BSIM4v6_MOD_PBGIDL           758
#define BSIM4v6_MOD_PEGIDL           759
#define BSIM4v6_MOD_PXRCRG1          760
#define BSIM4v6_MOD_PXRCRG2          761
#define BSIM4v6_MOD_PEU              762
#define BSIM4v6_MOD_PMINV            763
#define BSIM4v6_MOD_PPDITS           764
#define BSIM4v6_MOD_PPDITSD          765
#define BSIM4v6_MOD_PFPROUT          766
#define BSIM4v6_MOD_PLPEB            767
#define BSIM4v6_MOD_PDVTP0           768
#define BSIM4v6_MOD_PDVTP1           769
#define BSIM4v6_MOD_PCGIDL           770
#define BSIM4v6_MOD_PPHIN            771
#define BSIM4v6_MOD_PRSW             772
#define BSIM4v6_MOD_PRDW             773
#define BSIM4v6_MOD_PNSD             774
#define BSIM4v6_MOD_PCKAPPAD         775
#define BSIM4v6_MOD_PAIGC            776
#define BSIM4v6_MOD_PBIGC            777
#define BSIM4v6_MOD_PCIGC            778
#define BSIM4v6_MOD_PAIGBACC         779            
#define BSIM4v6_MOD_PBIGBACC         780
#define BSIM4v6_MOD_PCIGBACC         781
#define BSIM4v6_MOD_PAIGBINV         782
#define BSIM4v6_MOD_PBIGBINV         783
#define BSIM4v6_MOD_PCIGBINV         784
#define BSIM4v6_MOD_PNIGC            785
#define BSIM4v6_MOD_PNIGBACC         786
#define BSIM4v6_MOD_PNIGBINV         787
#define BSIM4v6_MOD_PNTOX            788
#define BSIM4v6_MOD_PEIGBINV         789
#define BSIM4v6_MOD_PPIGCD           790
#define BSIM4v6_MOD_PPOXEDGE         791
#define BSIM4v6_MOD_PAIGSD           792
#define BSIM4v6_MOD_PBIGSD           793
#define BSIM4v6_MOD_PCIGSD           794

#define BSIM4v6_MOD_SAREF            795
#define BSIM4v6_MOD_SBREF            796
#define BSIM4v6_MOD_KU0              797
#define BSIM4v6_MOD_KVSAT            798
#define BSIM4v6_MOD_TKU0             799
#define BSIM4v6_MOD_LLODKU0          800
#define BSIM4v6_MOD_WLODKU0          801
#define BSIM4v6_MOD_LLODVTH          802
#define BSIM4v6_MOD_WLODVTH          803
#define BSIM4v6_MOD_LKU0             804
#define BSIM4v6_MOD_WKU0             805
#define BSIM4v6_MOD_PKU0             806
#define BSIM4v6_MOD_KVTH0            807
#define BSIM4v6_MOD_LKVTH0           808
#define BSIM4v6_MOD_WKVTH0           809
#define BSIM4v6_MOD_PKVTH0           810
#define BSIM4v6_MOD_WLOD		   811
#define BSIM4v6_MOD_STK2		   812
#define BSIM4v6_MOD_LODK2		   813
#define BSIM4v6_MOD_STETA0	   814
#define BSIM4v6_MOD_LODETA0	   815

#define BSIM4v6_MOD_WEB              816
#define BSIM4v6_MOD_WEC              817
#define BSIM4v6_MOD_KVTH0WE          818
#define BSIM4v6_MOD_K2WE             819
#define BSIM4v6_MOD_KU0WE            820
#define BSIM4v6_MOD_SCREF            821
#define BSIM4v6_MOD_WPEMOD           822
#define BSIM4v6_MOD_PMINVCV          823

#define BSIM4v6_MOD_PLAMBDA          825
#define BSIM4v6_MOD_PVTL             826
#define BSIM4v6_MOD_PXN              827
#define BSIM4v6_MOD_PVFBSDOFF        828

#define BSIM4v6_MOD_TNOM             831
#define BSIM4v6_MOD_CGSO             832
#define BSIM4v6_MOD_CGDO             833
#define BSIM4v6_MOD_CGBO             834
#define BSIM4v6_MOD_XPART            835
#define BSIM4v6_MOD_RSH              836
#define BSIM4v6_MOD_JSS              837
#define BSIM4v6_MOD_PBS              838
#define BSIM4v6_MOD_MJS              839
#define BSIM4v6_MOD_PBSWS            840
#define BSIM4v6_MOD_MJSWS            841
#define BSIM4v6_MOD_CJS              842
#define BSIM4v6_MOD_CJSWS            843
#define BSIM4v6_MOD_NMOS             844
#define BSIM4v6_MOD_PMOS             845
#define BSIM4v6_MOD_NOIA             846
#define BSIM4v6_MOD_NOIB             847
#define BSIM4v6_MOD_NOIC             848
#define BSIM4v6_MOD_LINT             849
#define BSIM4v6_MOD_LL               850
#define BSIM4v6_MOD_LLN              851
#define BSIM4v6_MOD_LW               852
#define BSIM4v6_MOD_LWN              853
#define BSIM4v6_MOD_LWL              854
#define BSIM4v6_MOD_LMIN             855
#define BSIM4v6_MOD_LMAX             856
#define BSIM4v6_MOD_WINT             857
#define BSIM4v6_MOD_WL               858
#define BSIM4v6_MOD_WLN              859
#define BSIM4v6_MOD_WW               860
#define BSIM4v6_MOD_WWN              861
#define BSIM4v6_MOD_WWL              862
#define BSIM4v6_MOD_WMIN             863
#define BSIM4v6_MOD_WMAX             864
#define BSIM4v6_MOD_DWC              865
#define BSIM4v6_MOD_DLC              866
#define BSIM4v6_MOD_XL               867
#define BSIM4v6_MOD_XW               868
#define BSIM4v6_MOD_EM               869
#define BSIM4v6_MOD_EF               870
#define BSIM4v6_MOD_AF               871
#define BSIM4v6_MOD_KF               872
#define BSIM4v6_MOD_NJS              873
#define BSIM4v6_MOD_XTIS             874
#define BSIM4v6_MOD_PBSWGS           875
#define BSIM4v6_MOD_MJSWGS           876
#define BSIM4v6_MOD_CJSWGS           877
#define BSIM4v6_MOD_JSWS             878
#define BSIM4v6_MOD_LLC              879
#define BSIM4v6_MOD_LWC              880
#define BSIM4v6_MOD_LWLC             881
#define BSIM4v6_MOD_WLC              882
#define BSIM4v6_MOD_WWC              883
#define BSIM4v6_MOD_WWLC             884
#define BSIM4v6_MOD_DWJ              885
#define BSIM4v6_MOD_JSD              886
#define BSIM4v6_MOD_PBD              887
#define BSIM4v6_MOD_MJD              888
#define BSIM4v6_MOD_PBSWD            889
#define BSIM4v6_MOD_MJSWD            890
#define BSIM4v6_MOD_CJD              891
#define BSIM4v6_MOD_CJSWD            892
#define BSIM4v6_MOD_NJD              893
#define BSIM4v6_MOD_XTID             894
#define BSIM4v6_MOD_PBSWGD           895
#define BSIM4v6_MOD_MJSWGD           896
#define BSIM4v6_MOD_CJSWGD           897
#define BSIM4v6_MOD_JSWD             898
#define BSIM4v6_MOD_DLCIG            899

/* trap-assisted tunneling */

#define BSIM4v6_MOD_JTSS             900
#define BSIM4v6_MOD_JTSD		   901
#define BSIM4v6_MOD_JTSSWS	   902
#define BSIM4v6_MOD_JTSSWD	   903
#define BSIM4v6_MOD_JTSSWGS	   904
#define BSIM4v6_MOD_JTSSWGD	   905
#define BSIM4v6_MOD_NJTS	 	   906
#define BSIM4v6_MOD_NJTSSW	   907
#define BSIM4v6_MOD_NJTSSWG	   908
#define BSIM4v6_MOD_XTSS		   909
#define BSIM4v6_MOD_XTSD		   910
#define BSIM4v6_MOD_XTSSWS	   911
#define BSIM4v6_MOD_XTSSWD	   912
#define BSIM4v6_MOD_XTSSWGS	   913
#define BSIM4v6_MOD_XTSSWGD	   914
#define BSIM4v6_MOD_TNJTS		   915
#define BSIM4v6_MOD_TNJTSSW	   916
#define BSIM4v6_MOD_TNJTSSWG	   917
#define BSIM4v6_MOD_VTSS             918
#define BSIM4v6_MOD_VTSD		   919
#define BSIM4v6_MOD_VTSSWS	   920
#define BSIM4v6_MOD_VTSSWD	   921
#define BSIM4v6_MOD_VTSSWGS	   922
#define BSIM4v6_MOD_VTSSWGD	   923
#define BSIM4v6_MOD_PUD              924
#define BSIM4v6_MOD_PUD1             925
#define BSIM4v6_MOD_PUP              926
#define BSIM4v6_MOD_PLP              927
#define BSIM4v6_MOD_JTWEFF           928

/* device questions */
#define BSIM4v6_DNODE                945
#define BSIM4v6_GNODEEXT             946
#define BSIM4v6_SNODE                947
#define BSIM4v6_BNODE                948
#define BSIM4v6_DNODEPRIME           949
#define BSIM4v6_GNODEPRIME           950
#define BSIM4v6_GNODEMIDE            951
#define BSIM4v6_GNODEMID             952
#define BSIM4v6_SNODEPRIME           953
#define BSIM4v6_BNODEPRIME           954
#define BSIM4v6_DBNODE               955
#define BSIM4v6_SBNODE               956
#define BSIM4v6_VBD                  957
#define BSIM4v6_VBS                  958
#define BSIM4v6_VGS                  959
#define BSIM4v6_VDS                  960
#define BSIM4v6_CD                   961
#define BSIM4v6_CBS                  962
#define BSIM4v6_CBD                  963
#define BSIM4v6_GM                   964
#define BSIM4v6_GDS                  965
#define BSIM4v6_GMBS                 966
#define BSIM4v6_GBD                  967
#define BSIM4v6_GBS                  968
#define BSIM4v6_QB                   969
#define BSIM4v6_CQB                  970
#define BSIM4v6_QG                   971
#define BSIM4v6_CQG                  972
#define BSIM4v6_QD                   973
#define BSIM4v6_CQD                  974
#define BSIM4v6_CGGB                 975
#define BSIM4v6_CGDB                 976
#define BSIM4v6_CGSB                 977
#define BSIM4v6_CBGB                 978
#define BSIM4v6_CAPBD                979
#define BSIM4v6_CQBD                 980
#define BSIM4v6_CAPBS                981
#define BSIM4v6_CQBS                 982
#define BSIM4v6_CDGB                 983
#define BSIM4v6_CDDB                 984
#define BSIM4v6_CDSB                 985
#define BSIM4v6_VON                  986
#define BSIM4v6_VDSAT                987
#define BSIM4v6_QBS                  988
#define BSIM4v6_QBD                  989
#define BSIM4v6_SOURCECONDUCT        990
#define BSIM4v6_DRAINCONDUCT         991
#define BSIM4v6_CBDB                 992
#define BSIM4v6_CBSB                 993
#define BSIM4v6_CSUB		   994
#define BSIM4v6_QINV		   995
#define BSIM4v6_IGIDL		   996
#define BSIM4v6_CSGB                 997
#define BSIM4v6_CSDB                 998
#define BSIM4v6_CSSB                 999
#define BSIM4v6_CGBB                 1000
#define BSIM4v6_CDBB                 1001
#define BSIM4v6_CSBB                 1002
#define BSIM4v6_CBBB                 1003
#define BSIM4v6_QS                   1004
#define BSIM4v6_IGISL		   1005
#define BSIM4v6_IGS		   1006
#define BSIM4v6_IGD		   1007
#define BSIM4v6_IGB		   1008
#define BSIM4v6_IGCS		   1009
#define BSIM4v6_IGCD		   1010
#define BSIM4v6_QDEF		   1011
#define BSIM4v6_DELVT0		   1012
#define BSIM4v6_GCRG                 1013
#define BSIM4v6_GTAU                 1014

#define BSIM4v6_MOD_LTVOFF           1051
#define BSIM4v6_MOD_LTVFBSDOFF       1052
#define BSIM4v6_MOD_WTVOFF           1053
#define BSIM4v6_MOD_WTVFBSDOFF       1054
#define BSIM4v6_MOD_PTVOFF           1055
#define BSIM4v6_MOD_PTVFBSDOFF       1056

#define BSIM4v6_MOD_LKVTH0WE          1061
#define BSIM4v6_MOD_LK2WE             1062
#define BSIM4v6_MOD_LKU0WE		1063
#define BSIM4v6_MOD_WKVTH0WE          1064
#define BSIM4v6_MOD_WK2WE             1065
#define BSIM4v6_MOD_WKU0WE		1066
#define BSIM4v6_MOD_PKVTH0WE          1067
#define BSIM4v6_MOD_PK2WE             1068
#define BSIM4v6_MOD_PKU0WE		1069

#define BSIM4v6_MOD_RBPS0               1101
#define BSIM4v6_MOD_RBPSL               1102
#define BSIM4v6_MOD_RBPSW               1103
#define BSIM4v6_MOD_RBPSNF              1104
#define BSIM4v6_MOD_RBPD0               1105
#define BSIM4v6_MOD_RBPDL               1106
#define BSIM4v6_MOD_RBPDW               1107
#define BSIM4v6_MOD_RBPDNF              1108

#define BSIM4v6_MOD_RBPBX0              1109
#define BSIM4v6_MOD_RBPBXL              1110
#define BSIM4v6_MOD_RBPBXW              1111
#define BSIM4v6_MOD_RBPBXNF             1112
#define BSIM4v6_MOD_RBPBY0              1113
#define BSIM4v6_MOD_RBPBYL              1114
#define BSIM4v6_MOD_RBPBYW              1115
#define BSIM4v6_MOD_RBPBYNF             1116

#define BSIM4v6_MOD_RBSBX0              1117
#define BSIM4v6_MOD_RBSBY0              1118
#define BSIM4v6_MOD_RBDBX0              1119
#define BSIM4v6_MOD_RBDBY0              1120

#define BSIM4v6_MOD_RBSDBXL             1121
#define BSIM4v6_MOD_RBSDBXW             1122
#define BSIM4v6_MOD_RBSDBXNF            1123
#define BSIM4v6_MOD_RBSDBYL             1124
#define BSIM4v6_MOD_RBSDBYW             1125
#define BSIM4v6_MOD_RBSDBYNF            1126

#define BSIM4v6_MOD_AGISL               1200
#define BSIM4v6_MOD_BGISL               1201
#define BSIM4v6_MOD_EGISL               1202
#define BSIM4v6_MOD_CGISL               1203
#define BSIM4v6_MOD_LAGISL              1204
#define BSIM4v6_MOD_LBGISL              1205
#define BSIM4v6_MOD_LEGISL              1206
#define BSIM4v6_MOD_LCGISL              1207
#define BSIM4v6_MOD_WAGISL              1208
#define BSIM4v6_MOD_WBGISL              1209
#define BSIM4v6_MOD_WEGISL              1210
#define BSIM4v6_MOD_WCGISL              1211
#define BSIM4v6_MOD_PAGISL              1212
#define BSIM4v6_MOD_PBGISL              1213
#define BSIM4v6_MOD_PEGISL              1214
#define BSIM4v6_MOD_PCGISL              1215

#define BSIM4v6_MOD_AIGS                1220
#define BSIM4v6_MOD_BIGS                1221
#define BSIM4v6_MOD_CIGS                1222
#define BSIM4v6_MOD_LAIGS               1223
#define BSIM4v6_MOD_LBIGS               1224
#define BSIM4v6_MOD_LCIGS               1225
#define BSIM4v6_MOD_WAIGS               1226
#define BSIM4v6_MOD_WBIGS               1227
#define BSIM4v6_MOD_WCIGS               1228
#define BSIM4v6_MOD_PAIGS               1229
#define BSIM4v6_MOD_PBIGS               1230
#define BSIM4v6_MOD_PCIGS               1231
#define BSIM4v6_MOD_AIGD                1232
#define BSIM4v6_MOD_BIGD                1233
#define BSIM4v6_MOD_CIGD                1234
#define BSIM4v6_MOD_LAIGD               1235
#define BSIM4v6_MOD_LBIGD               1236
#define BSIM4v6_MOD_LCIGD               1237
#define BSIM4v6_MOD_WAIGD               1238
#define BSIM4v6_MOD_WBIGD               1239
#define BSIM4v6_MOD_WCIGD               1240
#define BSIM4v6_MOD_PAIGD               1241
#define BSIM4v6_MOD_PBIGD               1242
#define BSIM4v6_MOD_PCIGD               1243
#define BSIM4v6_MOD_DLCIGD              1244

#define BSIM4v6_MOD_NJTSD               1250
#define BSIM4v6_MOD_NJTSSWD             1251
#define BSIM4v6_MOD_NJTSSWGD            1252
#define BSIM4v6_MOD_TNJTSD              1253
#define BSIM4v6_MOD_TNJTSSWD            1254
#define BSIM4v6_MOD_TNJTSSWGD           1255

#define BSIM4v6_MOD_VGS_MAX            1301
#define BSIM4v6_MOD_VGD_MAX            1302
#define BSIM4v6_MOD_VGB_MAX            1303
#define BSIM4v6_MOD_VDS_MAX            1304
#define BSIM4v6_MOD_VBS_MAX            1305
#define BSIM4v6_MOD_VBD_MAX            1306
#define BSIM4v6_MOD_VGSR_MAX           1307
#define BSIM4v6_MOD_VGDR_MAX           1308
#define BSIM4v6_MOD_VGBR_MAX           1309
#define BSIM4v6_MOD_VBSR_MAX           1310
#define BSIM4v6_MOD_VBDR_MAX           1311

#include "bsim4v6ext.h"

extern void BSIM4v6evaluate(double,double,double,BSIM4v6instance*,BSIM4v6model*,
        double*,double*,double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, CKTcircuit*);
extern int BSIM4v6debug(BSIM4v6model*, BSIM4v6instance*, CKTcircuit*, int);
extern int BSIM4v6checkModel(BSIM4v6model*, BSIM4v6instance*, CKTcircuit*);
extern int BSIM4v6PAeffGeo(double, int, int, double, double, double, double, double *, double *, double *, double *);
extern int BSIM4v6RdseffGeo(double, int, int, int, double, double, double, double, double, int, double *);


#endif /*BSIM4v6*/
