/*************
 * Structure definitions header file
 * 1999 E. Rouat
 ************/

/* 
 * This file will contain all extern structure definitions needed
 * by ngspice code. (in construction)
 */

#ifndef _STRUCT_H_
#define _STRUCT_H_


/* cpstd.h */

/* Doubly linked lists of words. */

struct wordlist {
    char *wl_word;
    struct wordlist *wl_next;
    struct wordlist *wl_prev;
} ;

typedef struct wordlist wordlist;

/* Complex numbers. */

struct _complex {   /* IBM portability... */
    double cx_real;
    double cx_imag;
} ;

typedef struct _complex complex;


/*
 *  The display device structure (ftedev.h)
 */

typedef struct {
    char *name;
    int minx, miny;
    int width, height;      /* in screen coordinate system */
    int numlinestyles, numcolors;   /* number supported */
    int (*Init)();
    int (*NewViewport)();
    int (*Close)();
    int (*Clear)();
    int (*DrawLine)();
    int (*Arc)();
    int (*Text)();
    int (*DefineColor)();
    int (*DefineLinestyle)();
    int (*SetLinestyle)();
    int (*SetColor)();
    int (*Update)();
/*  int (*NDCtoScreen)(); */
    int (*Track)();
    int (*MakeMenu)();
    int (*MakeDialog)();
    int (*Input)();
    int (*DatatoScreen)();
} DISPDEVICE;

extern DISPDEVICE *dispdev;



/* ckt */


typedef struct sCKTnode {
  IFuid name;
  int type;
  int number;                /* Number of the node */
  double ic;                 /* Value of the initial condition */
  double nodeset;            /* Value of the .nodeset option */
  double *ptr;               /* ??? */
  struct sCKTnode *next;     /* pointer to the next node */
  unsigned int icGiven:1;    /* FLAG ic given */
  unsigned int nsGiven:1;    /* FLAG nodeset given */ 
} CKTnode;


/* the following structure is REALLY MESSY!! */


typedef struct {
  GENmodel *CKThead[MAXNUMDEVS];  /* The max number of loadable devices */
  STATistics *CKTstat;            /* The STATistics structure */
  double *(CKTstates[8]);         /* Used as memory of past steps ??? */
  double CKTtime;                /* ??? */
  double CKTdelta;               /* ??? */
  double CKTdeltaOld[7];         /* Memory for ??? */
  double CKTtemp;                /* Actual temperature of CKT */
  double CKTnomTemp;             /* Reference temperature 27 C ? */
  double CKTvt;                  /* Thernmal voltage at CKTtemp */
  double CKTag[7];               /* the gear variable coefficient matrix */
#ifdef PREDICTOR
    double CKTagp[7];       /* the gear predictor variable coefficient matrix */
#endif /*PREDICTOR*/
  int CKTorder;           /* the integration method order */
  int CKTmaxOrder;        /* maximum integration method order */
  int CKTintegrateMethod; /* the integration method to be used */
  SMPmatrix *CKTmatrix;   /* pointer to sparse matrix */
  int CKTniState;         /* internal state */
  double *CKTrhs;         /* current rhs value - being loaded */
  double *CKTrhsOld;      /* previous rhs value for convergence testing */
  double *CKTrhsSpare;    /* spare rhs value for reordering */
  double *CKTirhs;        /* current rhs value - being loaded (imag) */
  double *CKTirhsOld;     /* previous rhs value (imaginary)*/
  double *CKTirhsSpare;   /* spare rhs value (imaginary)*/
#ifdef PREDICTOR
  double *CKTpred;        /* predicted solution vector */
  double *CKTsols[8];     /* previous 8 solutions */
#endif /* PREDICTOR */
  double *CKTrhsOp;      /* opearating point values */
  double *CKTsenRhs;      /* current sensitivity rhs  values */
  double *CKTseniRhs;      /* current sensitivity rhs  values (imag)*/
  int CKTmaxEqNum;        /* And this ? */
  int CKTcurrentAnalysis; /* the analysis in progress (if any) */

/* defines for the value of  CKTcurrentAnalysis */
/* are in TSKdefs.h */

  CKTnode *CKTnodes;          /* ??? */
  CKTnode *CKTlastNode;       /* ??? */
  int CKTnumStates;       /* Number of sates effectively valid ??? */ 
  long CKTmode;           /* Mode of operation of the circuit ??? */
  int CKTbypass;          /* bypass option, how does it work ? */
  int CKTdcMaxIter;       /* iteration limit for dc op.  (itl1) */
  int CKTdcTrcvMaxIter;   /* iteration limit for dc tran. curv (itl2) */
  int CKTtranMaxIter;     /* iteration limit for each timepoint for tran*/
                          /* (itl4) */
  int CKTbreakSize;       /* ??? */
  int CKTbreak;           /* ??? */
  double CKTsaveDelta;    /* ??? */
  double CKTminBreak;     /* ??? */
  double *CKTbreaks;      /* List of breakpoints ??? */
  double CKTabstol;       /* --- */
  double CKTpivotAbsTol;  /* --- */
  double CKTpivotRelTol;  /* --- */
  double CKTreltol;       /* --- */
  double CKTchgtol;       /* --- */
  double CKTvoltTol;      /* --- */
  /* What is this define for  ? */
#ifdef NEWTRUNC
  double CKTlteReltol;  
  double CKTlteAbstol;
#endif /* NEWTRUNC */
  double CKTgmin;         /* Parallel Conductance --- */
  double CKTdelmin;       /* ??? */
  double CKTtrtol;        /* ??? */
  double CKTfinalTime;    /* ??? */
  double CKTstep;         /* ??? */
  double CKTmaxStep;      /* ??? */
  double CKTinitTime;     /* ??? */
  double CKTomega;        /* ??? */
  double CKTsrcFact;      /* ??? */
  double CKTdiagGmin;     /* ??? */
  int CKTnumSrcSteps;     /* ??? */
  int CKTnumGminSteps;    /* ??? */
  int CKTnoncon;          /* ??? */
  double CKTdefaultMosL;  /* Default Channel Lenght of MOS devices */
  double CKTdefaultMosW;  /* Default Channel Width of MOS devics */
  double CKTdefaultMosAD; /* Default Drain Area of MOS */
  double CKTdefaultMosAS; /* Default Source Area of MOS */
  unsigned int CKThadNodeset:1; /* ??? */
  unsigned int CKTfixLimit:1;   /* flag to indicate that the limiting of 
                                 * MOSFETs should be done as in SPICE2 */
  unsigned int CKTnoOpIter:1;   /* flag to indicate not to try the operating
                                 * point brute force, but to use gmin stepping
                                 * first */
  unsigned int CKTisSetup:1;    /* flag to indicate if CKTsetup done */
  JOB *CKTcurJob;               /* Next analysis to be performed ??? */

  SENstruct *CKTsenInfo;	/* the sensitivity information */
  double *CKTtimePoints;        /* list of all accepted timepoints in the
				   current transient simulation */
  double *CKTdeltaList;         /* list of all timesteps in the current
				   transient simulation */
  int CKTtimeListSize;	        /* size of above lists */
  int CKTtimeIndex;		/* current position in above lists */
  int CKTsizeIncr;		/* amount to increment size of above arrays
				   when you run out of space */
  unsigned int CKTtryToCompact:1; /* try to compact past history for LTRA
				   lines */
  unsigned int CKTbadMos3:1; /* Use old, unfixed MOS3 equations */
  unsigned int CKTkeepOpInfo:1; /* flag for small signal analyses */
  int CKTtroubleNode;		/* Non-convergent node number */
  GENinstance *CKTtroubleElt;	/* Non-convergent device instance */

} CKTcircuit;




#endif /* _STRUCT_H_ */
