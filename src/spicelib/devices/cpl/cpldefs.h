/**********
Copyright 1992 Regents of the University of California.  All rights
reserved.
**********/

#ifndef CPL
#define CPL

#include "ifsim.h"
#include "cktdefs.h"
#include "gendefs.h"
#include "complex.h"
#include "noisedef.h"
#include "swec.h"

/* information used to describe a single instance */

typedef struct sCPLinstance {
    struct sCPLmodel *CPLmodPtr;    /* backpointer to model */
    struct sCPLinstance *CPLnextInstance;   /* pointer to next instance of 
                                             * current model*/

    IFuid CPLname;  /* pointer to character string naming this instance */
    int CPLowner;   /* number of owner process */
    int CPLstate;   /* not used */

	int *CPLposNodes;
	int *CPLnegNodes;
        int dimension;
	double CPLlength;
	int *CPLibr1;
	int *CPLibr2;
	CPLine *cplines;  /* pointer to SWEC cplines type */
	CPLine *cplines2;  /* temporary pointer to SWEC cplines type */
	
	char **in_node_names;
	char **out_node_names;

	double **CPLibr1Ibr1;
	double **CPLibr2Ibr2;
	double **CPLposIbr1;
	double **CPLnegIbr2;
	/* trial */
	double **CPLposPos;
	double **CPLnegNeg;
	double **CPLposNeg;
	double **CPLnegPos;

	double ***CPLibr1Pos;
	double ***CPLibr2Neg;
	double ***CPLibr1Neg;
	double ***CPLibr2Pos;
	double ***CPLibr1Ibr2;
	double ***CPLibr2Ibr1;
	
	unsigned CPLibr1Given   : 1;
	unsigned CPLibr2Given   : 1;
	unsigned CPLdcGiven     : 1;
	unsigned CPLlengthGiven : 1;

} CPLinstance ;


/* per model data */

typedef struct sCPLmodel {       /* model structure for a cpl */
    int CPLmodType; /* type index of this device type */
    struct sCPLmodel *CPLnextModel; /* pointer to next possible model in 
                                     * linked list */
    CPLinstance * CPLinstances; /* pointer to list of instances that have this
                                 * model */
    IFuid CPLmodName;       /* pointer to character string naming this model */

	double *Rm;
        int Rm_counter;
	double *Gm;
        int Gm_counter;
	double *Lm;
        int Lm_counter;
	double *Cm;
        int Cm_counter;
	double length;
    unsigned Rmgiven : 1;   
    unsigned Lmgiven : 1;  
    unsigned Gmgiven : 1;
    unsigned Cmgiven : 1;
    unsigned lengthgiven : 1;

} CPLmodel;

/* instance parameters */
#define CPL_POS_NODE 1
#define CPL_NEG_NODE 2
#define CPL_DIM 3
#define CPL_LENGTH 4

/* model parameters */
#define CPL_R 101
#define CPL_C 102
#define CPL_G 103
#define CPL_L 104
#define CPL_length 105
#define CPL_MOD_R 106

#include "cplext.h"
extern VI_list *pool_vi;

#endif /*CPL*/
