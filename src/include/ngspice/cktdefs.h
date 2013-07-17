/*
 * Copyright (c) 1985 Thomas L. Quarles
 * Modified 1999 Paolo Nenzi - Removed non STDC definitions
 * Modified 2000 AlansFixes
 */
#ifndef ngspice_CKTDEFS_H
#define ngspice_CKTDEFS_H

#include "ngspice/typedefs.h"


/* gtri - evt - wbk - 5/20/91 - add event-driven and enhancements data */
#ifdef XSPICE
#include "ngspice/evt.h"
#include "ngspice/enh.h"
#endif
/* gtri - evt - wbk - 5/20/91 - add event-driven and enhancements data */


#define MAXNUMDEVS 64   /* Max number of possible devices PN:XXX may cause toubles*/
#define MAXNUMDEVNODES 4        /* Max No. of nodes per device */
                         /* Need to change for SOI devs ? */

#include "ngspice/smpdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/acdefs.h"
#include "ngspice/gendefs.h"
#include "ngspice/trcvdefs.h"
#include "ngspice/optdefs.h"
#include "ngspice/sen2defs.h"
#include "ngspice/pzdefs.h"
#include "ngspice/noisedef.h"
#include "ngspice/hash.h"



struct CKTnode {
    IFuid name;
    int type;

#define SP_VOLTAGE 3
#define SP_CURRENT 4
#define NODE_VOLTAGE SP_VOLTAGE
#define NODE_CURRENT SP_CURRENT

    int number;                 /* Number of the node */
    double ic;                  /* Value of the initial condition */
    double nodeset;             /* Value of the .nodeset option */
    double *ptr;                /* ??? */
    CKTnode *next;              /* pointer to the next node */
    unsigned int icGiven:1;     /* FLAG ic given */
    unsigned int nsGiven:1;     /* FLAG nodeset given */
};

/* defines for node parameters */
#define PARM_NS        1
#define PARM_IC        2
#define PARM_NODETYPE  3


struct CKTcircuit {

/* gtri - begin - wbk - change declaration to allow dynamic sizing */

/* An associated change is made in CKTinit.c to alloc the space */
/* required for the pointers.  No changes are needed to the source */
/* code at the 3C1 level, although the compiler will generate */
/* slightly different code for references to this data. */

/*  GENmodel *CKThead[MAXNUMDEVS]; The max number of loadable devices */
    GENmodel **CKThead;

/* gtri - end   - wbk - change declaration to allow dynamic sizing */

/*    GENmodel *CKThead[MAXNUMDEVS];  maschmann : deleted */


    STATistics *CKTstat;        /* The STATistics structure */
    double *(CKTstates[8]);     /* Used as memory of past steps ??? */

    /* Some shortcut for CKTstates */
#define CKTstate0 CKTstates[0]
#define CKTstate1 CKTstates[1]
#define CKTstate2 CKTstates[2]
#define CKTstate3 CKTstates[3]
#define CKTstate4 CKTstates[4]
#define CKTstate5 CKTstates[5]
#define CKTstate6 CKTstates[6]
#define CKTstate7 CKTstates[7]
    double CKTtime;             /* ??? */
    double CKTdelta;            /* ??? */
    double CKTdeltaOld[7];      /* Memory for ??? */
    double CKTtemp;             /* Actual temperature of CKT */
    double CKTnomTemp;          /* Reference temperature 27 C ? */
    double CKTvt;               /* Thernmal voltage at CKTtemp */
    double CKTag[7];            /* the gear variable coefficient matrix */
#ifdef PREDICTOR
    double CKTagp[7];           /* the gear predictor variable
                                   coefficient matrix */
#endif /*PREDICTOR*/
    int CKTorder;               /* the integration method order */
    int CKTmaxOrder;            /* maximum integration method order */
    int CKTintegrateMethod;     /* the integration method to be used */

/* known integration methods */
#define TRAPEZOIDAL 1
#define GEAR 2

    SMPmatrix *CKTmatrix;       /* pointer to sparse matrix */
    int CKTniState;             /* internal state */
    double *CKTrhs;             /* current rhs value - being loaded */
    double *CKTrhsOld;          /* previous rhs value for convergence
                                   testing */
    double *CKTrhsSpare;        /* spare rhs value for reordering */
    double *CKTirhs;            /* current rhs value - being loaded
                                   (imag) */
    double *CKTirhsOld;         /* previous rhs value (imaginary)*/
    double *CKTirhsSpare;       /* spare rhs value (imaginary)*/
#ifdef PREDICTOR
    double *CKTpred;            /* predicted solution vector */
    double *CKTsols[8];         /* previous 8 solutions */
#endif /* PREDICTOR */

    double *CKTrhsOp;           /* opearating point values */
    double *CKTsenRhs;          /* current sensitivity rhs values */
    double *CKTseniRhs;         /* current sensitivity rhs values
                                   (imag)*/


/*
 *  symbolic constants for CKTniState
 *      Note that they are bitwise disjoint
 *  What is their meaning ????
 */

#define NISHOULDREORDER       0x1
#define NIREORDERED           0x2
#define NIUNINITIALIZED       0x4
#define NIACSHOULDREORDER    0x10
#define NIACREORDERED        0x20
#define NIACUNINITIALIZED    0x40
#define NIDIDPREORDER       0x100
#define NIPZSHOULDREORDER   0x200

    int CKTmaxEqNum;            /* And this ? */
    int CKTcurrentAnalysis;     /* the analysis in progress (if any) */

/* defines for the value of  CKTcurrentAnalysis */
/* are in TSKdefs.h */

    CKTnode *CKTnodes;          /* ??? */
    CKTnode *CKTlastNode;       /* ??? */

    /* This define should be somewhere else ??? */
#define NODENAME(ckt,nodenum) CKTnodName(ckt,nodenum)
    int CKTnumStates;           /* Number of sates effectively valid
                                   ??? */
    long CKTmode;               /* Mode of operation of the circuit
                                   ??? */

/* defines for CKTmode */

/* old 'mode' parameters */
#define MODE               0x3
#define MODETRAN           0x1
#define MODEAC             0x2

/* old 'modedc' parameters */
#define MODEDC            0x70
#define MODEDCOP          0x10
#define MODETRANOP        0x20
#define MODEDCTRANCURVE   0x40

/* old 'initf' parameters */
#define INITF           0x3f00
#define MODEINITFLOAT    0x100
#define MODEINITJCT      0x200
#define MODEINITFIX      0x400
#define MODEINITSMSIG    0x800
#define MODEINITTRAN    0x1000
#define MODEINITPRED    0x2000

/* old 'nosolv' paramater */
#define MODEUIC 0x10000l

    int CKTbypass;              /* bypass option, how does it work ?  */
    int CKTdcMaxIter;           /* iteration limit for dc op.  (itl1) */
    int CKTdcTrcvMaxIter;       /* iteration limit for dc tran. curv
                                   (itl2) */
    int CKTtranMaxIter;         /* iteration limit for each timepoint
                                   for tran*/
    /* (itl4) */
    int CKTbreakSize;           /* ??? */
    int CKTbreak;               /* ??? */
    double CKTsaveDelta;        /* ??? */
    double CKTminBreak;         /* ??? */
    double *CKTbreaks;          /* List of breakpoints ??? */
    double CKTabstol;           /* --- */
    double CKTpivotAbsTol;      /* --- */
    double CKTpivotRelTol;      /* --- */
    double CKTreltol;           /* --- */
    double CKTchgtol;           /* --- */
    double CKTvoltTol;          /* --- */
    /* What is this define for  ? */
#ifdef NEWTRUNC
    double CKTlteReltol;
    double CKTlteAbstol;
#endif /* NEWTRUNC */
    double CKTgmin;             /* Parallel Conductance --- */
    double CKTgshunt;
    double CKTdelmin;           /* ??? */
    double CKTtrtol;            /* ??? */
    double CKTfinalTime;        /* ??? */
    double CKTstep;             /* ??? */
    double CKTmaxStep;          /* ??? */
    double CKTinitTime;         /* ??? */
    double CKTomega;            /* ??? */
    double CKTsrcFact;          /* ??? */
    double CKTdiagGmin;         /* ??? */
    int CKTnumSrcSteps;         /* ??? */
    int CKTnumGminSteps;        /* ??? */
    double CKTgminFactor;
    int CKTnoncon;              /* ??? */
    double CKTdefaultMosM;
    double CKTdefaultMosL;      /* Default Channel Lenght of MOS devices */
    double CKTdefaultMosW;      /* Default Channel Width of MOS devics */
    double CKTdefaultMosAD;     /* Default Drain Area of MOS */
    double CKTdefaultMosAS;     /* Default Source Area of MOS */
    unsigned int CKThadNodeset:1; /* ??? */
    unsigned int CKTfixLimit:1; /* flag to indicate that the limiting
                                   of MOSFETs should be done as in
                                   SPICE2 */
    unsigned int CKTnoOpIter:1; /* flag to indicate not to try the operating
                                   point brute force, but to use gmin stepping
                                   first */
    unsigned int CKTisSetup:1;  /* flag to indicate if CKTsetup done */
#ifdef XSPICE
    unsigned int CKTadevFlag:1; /* flag indicates 'A' devices in the circuit */
#endif
    JOB *CKTcurJob;             /* Next analysis to be performed ??? */

    SENstruct *CKTsenInfo;      /* the sensitivity information */
    double *CKTtimePoints;      /* list of all accepted timepoints in
                                   the current transient simulation */
    double *CKTdeltaList;       /* list of all timesteps in the
                                   current transient simulation */
    int CKTtimeListSize;        /* size of above lists */
    int CKTtimeIndex;           /* current position in above lists */
    int CKTsizeIncr;            /* amount to increment size of above
                                   arrays when you run out of space */
    unsigned int CKTtryToCompact:1; /* try to compact past history for LTRA
                                       lines */
    unsigned int CKTbadMos3:1;  /* Use old, unfixed MOS3 equations */
    unsigned int CKTkeepOpInfo:1; /* flag for small signal analyses */
    unsigned int CKTcopyNodesets:1; /* NodesetFIX */
    unsigned int CKTnodeDamping:1; /* flag for node damping fix */
    double CKTabsDv;            /* abs limit for iter-iter voltage change */
    double CKTrelDv;            /* rel limit for iter-iter voltage change */
    int CKTtroubleNode;         /* Non-convergent node number */
    GENinstance *CKTtroubleElt; /* Non-convergent device instance */
    int CKTvarHertz;            /* variable HERTZ in B source */
/* gtri - evt - wbk - 5/20/91 - add event-driven and enhancements data */
#ifdef XSPICE
    Evt_Ckt_Data_t *evt;        /* all data about event driven stuff */
    Enh_Ckt_Data_t *enh;        /* data used by general enhancements */
#endif
/* gtri - evt - wbk - 5/20/91 - add event-driven and enhancements data */

#ifdef WITH_PSS
/* SP: Periodic Steady State Analysis - 100609 */
    double CKTstabTime;		/* PSS stab time */
    double CKTguessedFreq;	/* PSS guessed frequency */
    int CKTharms;		/* PSS harmonics */
    long int CKTpsspoints;	/* PSS number of samples */
    char *CKToscNode;       	/* PSS oscnode */
    double CKTsteady_coeff;	/* PSS Steady Coefficient */
    int CKTsc_iter;        	/* PSS Maximum Number of Shooting Iterations */
/* SP: 100609 */
#endif

    unsigned int CKTisLinear:1; /* flag to indicate that the circuit
                                   contains only linear elements */
    unsigned int CKTnoopac:1; /* flag to indicate that OP will not be evaluated
                                 during AC simulation */

    NGHASHPTR DEVnameHash;
};


/* Now function prottypes */

extern int ACan(CKTcircuit *, int);
extern int ACaskQuest(CKTcircuit *, JOB *, int , IFvalue *);
extern int ACsetParm(CKTcircuit *, JOB *, int , IFvalue *);
extern int CKTacDump(CKTcircuit *, double , void *);
extern int CKTacLoad(CKTcircuit *);
extern int CKTaccept(CKTcircuit *);
extern int CKTacct(CKTcircuit *, JOB *, int , IFvalue *);
extern int CKTask(CKTcircuit *, GENinstance *, int , IFvalue *, IFvalue *);
extern int CKTaskAnalQ(CKTcircuit *, JOB *, int , IFvalue *, IFvalue *);
extern int CKTaskNodQst(CKTcircuit *, CKTnode *, int , IFvalue *, IFvalue *);
extern int CKTbindNode(CKTcircuit *, GENinstance *, int , CKTnode *);
extern void CKTbreakDump(CKTcircuit *);
extern int CKTclrBreak(CKTcircuit *);
extern int CKTconvTest(CKTcircuit *);
extern int CKTcrtElt(CKTcircuit *, GENmodel *, GENinstance **, IFuid);
extern int CKTdelTask(CKTcircuit *, TSKtask *);
extern int CKTdestroy(CKTcircuit *);
extern int CKTdltAnal(void *, void *, void *);
extern int CKTdltInst(CKTcircuit *, void *);
extern int CKTdltMod(CKTcircuit *, GENmodel *);
extern int CKTdltNNum(CKTcircuit *, int);
extern int CKTdltNod(CKTcircuit *, CKTnode *);
extern int CKTdoJob(CKTcircuit *, int , TSKtask *);
extern void CKTdump(CKTcircuit *, double, void *);
#ifdef CIDER
extern void NDEVacct(CKTcircuit *ckt, FILE *file);
#endif /* CIDER */
extern void CKTncDump(CKTcircuit *);
extern int CKTfndAnal(CKTcircuit *, int *, JOB **, IFuid , TSKtask *, IFuid);
extern int CKTfndBranch(CKTcircuit *, IFuid);
extern GENinstance *CKTfndDev(CKTcircuit *, IFuid);
extern int CKTfndMod(CKTcircuit *, int *, GENmodel **, IFuid);
extern int CKTfndNode(CKTcircuit *, CKTnode **, IFuid);
extern int CKTfndTask(CKTcircuit *, TSKtask **, IFuid );
extern int CKTground(CKTcircuit *, CKTnode **, IFuid);
extern int CKTic(CKTcircuit *);
extern int CKTinit(CKTcircuit **);
extern int CKTinst2Node(CKTcircuit *, void *, int , CKTnode **, IFuid *);
extern int CKTlinkEq(CKTcircuit *, CKTnode *);
extern int CKTload(CKTcircuit *);
extern int CKTmapNode(CKTcircuit *, CKTnode **, IFuid);
extern int CKTmkCur(CKTcircuit  *, CKTnode **, IFuid , char *);
extern int CKTmkNode(CKTcircuit *, CKTnode **);
extern int CKTmkVolt(CKTcircuit  *, CKTnode **, IFuid , char *);
extern int CKTmodAsk(CKTcircuit *, GENmodel *, int , IFvalue *, IFvalue *);
extern int CKTmodCrt(CKTcircuit *, int , GENmodel **, IFuid);
extern int CKTmodParam(CKTcircuit *, GENmodel *, int , IFvalue *, IFvalue *);
extern int CKTnames(CKTcircuit *, int *, IFuid **);
extern int CKTdnames(CKTcircuit *);
extern int CKTnewAnal(CKTcircuit *, int , IFuid , JOB **, TSKtask *);
extern int CKTnewEq(CKTcircuit *, CKTnode **, IFuid);
extern int CKTnewNode(CKTcircuit *, CKTnode **, IFuid);
extern int CKTnewTask(CKTcircuit *, TSKtask **, IFuid, TSKtask **);
extern int CKTnoise (CKTcircuit *ckt, int mode, int operation, Ndata *data);
extern IFuid CKTnodName(CKTcircuit *, int);
extern void CKTnodOut(CKTcircuit *);
extern CKTnode * CKTnum2nod(CKTcircuit *, int);
extern int CKTop(CKTcircuit *, long, long, int);
extern int CKTpModName(char *, IFvalue *, CKTcircuit *, int , IFuid , GENmodel **);
extern int CKTpName(char *, IFvalue *, CKTcircuit *, int , char *, GENinstance **);
extern int CKTparam(CKTcircuit *, GENinstance *, int , IFvalue *, IFvalue *);
extern int CKTpzFindZeros(CKTcircuit *, PZtrial **, int *);
extern int CKTpzLoad(CKTcircuit *, SPcomplex *);
extern int CKTpzSetup(CKTcircuit *, int);
extern int CKTsenAC(CKTcircuit *);
extern int CKTsenComp(CKTcircuit *);
extern int CKTsenDCtran(CKTcircuit *);
extern int CKTsenLoad(CKTcircuit *);
extern void CKTsenPrint(CKTcircuit *);
extern int CKTsenSetup(CKTcircuit *);
extern int CKTsenUpdate(CKTcircuit *);
extern int CKTsetAnalPm(CKTcircuit *, JOB *, int , IFvalue *, IFvalue *);
extern int CKTsetBreak(CKTcircuit *, double);
extern int CKTsetNodPm(CKTcircuit *, CKTnode *, int , IFvalue *, IFvalue *);
extern int CKTsetOpt(CKTcircuit *, JOB *, int , IFvalue *);
extern int CKTsetup(CKTcircuit *);
extern int CKTunsetup(CKTcircuit *);
extern int CKTtemp(CKTcircuit *);
extern char *CKTtrouble(CKTcircuit *, char *);
extern void CKTterr(int , CKTcircuit *, double *);
extern int CKTtrunc(CKTcircuit *, double *);
extern int CKTtypelook(char *);
extern int DCOaskQuest(CKTcircuit *, JOB *, int , IFvalue *);
extern int DCOsetParm(CKTcircuit  *, JOB *, int , IFvalue *);
extern int DCTaskQuest(CKTcircuit *, JOB *, int , IFvalue *);
extern int DCTsetParm(CKTcircuit  *, JOB *, int , IFvalue *);
extern int DCop(CKTcircuit *ckt, int notused); /* va: notused avoids "init from incompatible pointer type" */
extern int DCtrCurv(CKTcircuit *, int);
extern int DCtran(CKTcircuit *, int);
extern int DISTOan(CKTcircuit *, int);
extern int NOISEan(CKTcircuit *, int);
extern int PZan(CKTcircuit *, int);
extern int PZinit(CKTcircuit *);
extern int PZpost(CKTcircuit *);
extern int PZaskQuest(CKTcircuit *, JOB *, int , IFvalue *);
extern int PZsetParm(CKTcircuit *, JOB *, int , IFvalue *);

#ifdef WANT_SENSE2
extern int SENaskQuest(CKTcircuit *, JOB *, int , IFvalue *);
extern void SENdestroy(SENstruct *);
extern int SENsetParm(CKTcircuit *, JOB *, int , IFvalue *);
extern int SENstartup(CKTcircuit *, int);
#endif

extern int SPIinit(IFfrontEnd *, IFsimulator **);
extern int TFanal(CKTcircuit *, int);
extern int TFaskQuest(CKTcircuit *, JOB *, int , IFvalue *);
extern int TFsetParm(CKTcircuit *, JOB *, int , IFvalue *);
extern int TRANaskQuest(CKTcircuit *, JOB *, int , IFvalue *);
extern int TRANsetParm(CKTcircuit *, JOB *, int , IFvalue *);
extern int TRANinit(CKTcircuit *, JOB *);

#ifdef WITH_PSS
/* SP: Steady State Analysis */
extern int PSSaskQuest(CKTcircuit *, JOB *, int , IFvalue *);
extern int PSSsetParm(CKTcircuit *, JOB *, int , IFvalue *);
extern int PSSinit(CKTcircuit *, JOB *);
extern int DCpss(CKTcircuit *, int);
/* SP */
#endif

extern int NaskQuest(CKTcircuit *, JOB *, int, IFvalue *);
extern int NsetParm(CKTcircuit *, JOB *, int, IFvalue *);
extern int NIacIter(CKTcircuit *);
extern int NIcomCof(CKTcircuit *);
extern int NIconvTest(CKTcircuit *);
extern void NIdestroy(CKTcircuit *);
extern int NIinit(CKTcircuit  *);
extern int NIintegrate(CKTcircuit *, double *, double *, double , int);
extern int NIiter(CKTcircuit * , int);
extern int NIpzMuller(PZtrial **, PZtrial *);
extern int NIpzComplex(PZtrial **, PZtrial *);
extern int NIpzSym(PZtrial **, PZtrial *);
extern int NIpzSym2(PZtrial **, PZtrial *);
extern int NIreinit(CKTcircuit *);
extern int NIsenReinit(CKTcircuit *);
extern int NIdIter (CKTcircuit *);
extern void NInzIter(CKTcircuit *, int, int);

#ifdef PREDICTOR
extern int NIpred(CKTcircuit *ckt);
#endif

extern IFfrontEnd *SPfrontEnd;

#endif
