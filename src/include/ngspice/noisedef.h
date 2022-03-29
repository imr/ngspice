/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Gary W. Ng
**********/

#ifndef ngspice_NOISEDEF_H
#define ngspice_NOISEDEF_H

#include "ngspice/jobdefs.h"

    /* structure used to describe an noise analysis */

typedef struct {
    int JOBtype;
    JOB *JOBnextJob;          /* pointer to next thing to do */
    char *JOBname;            /* name of this job */
    CKTnode *output;        /* noise output summation node */
    CKTnode *outputRef;     /* noise output reference node */
    IFuid input;      /* name of the AC source used as input reference */
    double NstartFreq;
    double NstopFreq;
    double NfreqDelta;    /* multiplier for decade/octave stepping, */
                              /* step for linear steps. */
    double NsavFstp; /* frequency step at which we stopped last time */
    double NsavOnoise;   /* integrated output noise when we left off last time */
    double NsavInoise;   /* integrated input noise when we left off last time */
    int NstpType;        /* values described below */
    int NnumSteps;
    int NStpsSm; /* number of steps before we output a noise summary report */
} NOISEAN;



    /* structure used to carry information between subparts of the noise analysis code */

typedef struct {
    double freq;
    double lstFreq;
    double delFreq;
    double outNoiz;       /* integrated output noise as of the last frequency point */
    double inNoise;        /* integrated input noise as of the last frequency point */
    double GainSqInv;
    double lnGainInv;
    double lnFreq;
    double lnLastFreq;
    double delLnFreq;
    int outNumber;      /* keeps track of the current output variable */
    int numPlots;        /* keeps track of the number of active plots so we can close them in */
			   /* a do loop. */
    unsigned int prtSummary;
    double *outpVector;  /* pointer to our array of noise outputs */
    char *squared_value;
    runDesc *NplotPtr; /* the plot pointer */
    IFuid *namelist;       /* list of plot names */
    unsigned squared : 1;
} Ndata;


/* codes for saving and retrieving integrated noise data */

#define LNLSTDENS 0     /* array location that the log of the last noise density is stored */
#define OUTNOIZ       1     /* array location that integrated output noise is stored */
#define INNOIZ        2     /* array location that integrated input noise is stored */

#define NSTATVARS 3    /* number of "state" variables that must be stored for each noise */
			      /* generator.  in this case it is three: LNLSTDENS, OUTNOIZ */
			      /* and INNOIZ */


/* available step types: */

#define DECADE 1
#define OCTAVE 2
#define LINEAR 3


/* noise analysis parameters */

#define N_OUTPUT     1
#define N_OUTREF 2
#define N_INPUT      3
#define N_START      4
#define N_STOP       5
#define N_STEPS      6
#define N_PTSPERSUM  7
#define N_DEC        8
#define N_OCT        9
#define N_LIN        10


/* noise routine operations/modes */

#define N_DENS    1
#define INT_NOIZ 2
#define N_OPEN       1
#define N_CALC       2
#define N_CLOSE      3
#define SHOTNOISE       1
#define THERMNOISE    2
#define N_GAIN       3


/* tolerances and limits to make numerical analysis more robust */

#define N_MINLOG          1E-38       /* the smallest number we can take the log of */
#define N_MINGAIN         1E-20       /* the smallest input-output gain we can tolerate */
					   /*    (to calculate input-referred noise we divide */
					   /*     the output noise by the gain) */
#define N_INTFTHRESH   1E-10       /* the largest slope (of a log-log noise spectral */
					   /*    density vs. freq plot) at which the noise */
					   /*    spectum is still considered flat. (no need for */
					   /*    log curve fitting) */
#define N_INTUSELOG      1E-10       /* decides which expression to use for the integral of */
					   /*    x**k.  If k is -1, then we must use a 'ln' form. */
					   /*    Otherwise, we use a 'power' form.  This */
					   /*    parameter is the region around (k=) -1 for which */
					   /*    use the 'ln' form. */


/* misc constants */
#ifdef RFSPICE

#define NOISE_ADD_OUTVAR(ckt, data, fmt, aname, bname)                  \
    if (ckt->CKTcurrentAnalysis & DOING_SP) {                           \
          ckt->CKTnoiseSourceCount++;                                   \
    }                                                                   \
    else\
    do {                                                                \
        data->namelist = TREALLOC(IFuid, data->namelist, data->numPlots + 1); \
        if (!data->namelist)                                            \
            return E_NOMEM;                                             \
        char *name = tprintf(fmt, aname, bname);                        \
        if (!name)                                                      \
            return E_NOMEM;                                             \
        SPfrontEnd->IFnewUid(ckt, &(data->namelist[data->numPlots++]),  \
                             NULL, name, UID_OTHER, NULL);              \
        tfree(name);                                                    \
    } while(0)                                           
#else
#define NOISE_ADD_OUTVAR(ckt, data, fmt, aname, bname)                  \
    do {                                                                \
        data->namelist = TREALLOC(IFuid, data->namelist, data->numPlots + 1); \
        if (!data->namelist)                                            \
            return E_NOMEM;                                             \
        char *name = tprintf(fmt, aname, bname);                        \
        if (!name)                                                      \
            return E_NOMEM;                                             \
        SPfrontEnd->IFnewUid(ckt, &(data->namelist[data->numPlots++]),  \
                             NULL, name, UID_OTHER, NULL);              \
        tfree(name);                                                    \
    } while(0)
#endif

void NevalSrc (double *noise, double *lnNoise, CKTcircuit *ckt, int type, int node1, int node2, double param);
void NevalSrc2 (double *, double *, CKTcircuit *, int, int, int, double, int, int, double, double);
void NevalSrcInstanceTemp (double *noise, double *lnNoise, CKTcircuit *ckt, int type, int node1, int node2, double param, double param2);
double Nintegrate (double noizDens, double lnNdens, double lnNlstDens, Ndata *data);

#endif
