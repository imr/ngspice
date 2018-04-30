/**********
Copyright 1992 Regents of the University of California.  All rights
reserved.
**********/

#ifndef CPL
#define CPL

#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/gendefs.h"
#include "ngspice/complex.h"
#include "ngspice/noisedef.h"
#include "ngspice/swec.h"

/* information used to describe a single instance */

typedef struct sCPLinstance {

    struct GENinstance gen;

#define CPLmodPtr(inst) ((struct sCPLmodel *)((inst)->gen.GENmodPtr))
#define CPLnextInstance(inst) ((struct sCPLinstance *)((inst)->gen.GENnextInstance))
#define CPLname gen.GENname
#define CPLstate gen.GENstate

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

	double **CPLibr1Ibr1Ptr;
	double **CPLibr2Ibr2Ptr;
	double **CPLposIbr1Ptr;
	double **CPLnegIbr2Ptr;
	/* trial */
	double **CPLposPosPtr;
	double **CPLnegNegPtr;
	double **CPLposNegPtr;
	double **CPLnegPosPtr;

	double ***CPLibr1PosPtr;
	double ***CPLibr2NegPtr;
	double ***CPLibr1NegPtr;
	double ***CPLibr2PosPtr;
	double ***CPLibr1Ibr2Ptr;
	double ***CPLibr2Ibr1Ptr;
	
	unsigned CPLibr1Given   : 1;
	unsigned CPLibr2Given   : 1;
	unsigned CPLdcGiven     : 1;
	unsigned CPLlengthGiven : 1;

} CPLinstance ;


/* per model data */

typedef struct sCPLmodel {       /* model structure for a cpl */

    struct GENmodel gen;

#define CPLmodType gen.GENmodType
#define CPLnextModel(inst) ((struct sCPLmodel *)((inst)->gen.GENnextModel))
#define CPLinstances(inst) ((CPLinstance *)((inst)->gen.GENinstances))
#define CPLmodName gen.GENmodName

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
enum {
    CPL_POS_NODE = 1,
    CPL_NEG_NODE,
    CPL_DIM,
    CPL_LENGTH,
};

/* model parameters */
enum {
    CPL_R = 101,
    CPL_C,
    CPL_G,
    CPL_L,
    CPL_length,
    CPL_MOD_R,
};

#include "cplext.h"
extern VI_list *pool_vi;

#endif /*CPL*/
