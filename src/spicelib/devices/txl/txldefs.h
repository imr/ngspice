#ifndef TXL
#define TXL

#include "ifsim.h"
#include "cktdefs.h"
#include "gendefs.h"
#include "complex.h"
#include "noisedef.h"
#include "swec.h"

/* information used to describe a single instance */

typedef struct sTXLinstance {
    struct sTXLmodel *TXLmodPtr;    /* backpointer to model */
    struct sTXLinstance *TXLnextInstance;   /* pointer to next instance of 
                                             * current model*/

    IFuid TXLname;  /* pointer to character string naming this instance */
    int TXLowner;  /* number of owner process */
    
        int dimensions; /* may we not need this but ... */
	int TXLposNode;
	int TXLnegNode;
	double TXLlength;
	int TXLibr1;
	int TXLibr2;
	TXLine *txline;  /* pointer to SWEC txline type */
	TXLine *txline2;  /* pointer to SWEC txline type. temporary storage */
	char *in_node_name;
	char *out_node_name;
	int TXLbranch;       /* unused */
	
	double *TXLposPosptr;
	double *TXLposNegptr;
	double *TXLnegPosptr;
	double *TXLnegNegptr;
	double *TXLibr1Posptr;
	double *TXLibr2Negptr;
	double *TXLposIbr1ptr;
	double *TXLnegIbr2ptr;
	double *TXLibr1Negptr;
	double *TXLibr2Posptr;
	double *TXLibr1Ibr1ptr;
	double *TXLibr2Ibr2ptr;
	double *TXLibr1Ibr2ptr;
	double *TXLibr2Ibr1ptr;
	
	unsigned TXLibr1Given : 1;
	unsigned TXLibr2Given : 1;
	unsigned TXLdcGiven : 1;
        unsigned TXLlengthgiven : 1;   /* flag to indicate C was specified */

} TXLinstance ;


/* per model data */

typedef struct sTXLmodel {       /* model structure for a txl */
    int TXLmodType; /* type index of this device type */
    struct sTXLmodel *TXLnextModel; /* pointer to next possible model in 
                                     * linked list */
    TXLinstance * TXLinstances; /* pointer to list of instances that have this
                                 * model */
    IFuid TXLmodName;       /* pointer to character string naming this model */

	double R;
	double L;
	double G;
	double C;
	double length;
    unsigned Rgiven : 1;   /* flag to indicate R was specified */
    unsigned Lgiven : 1;   /* flag to indicate L was specified */
    unsigned Ggiven : 1;   /* flag to indicate G was specified */
    unsigned Cgiven : 1;   /* flag to indicate C was specified */
    unsigned lengthgiven : 1;   /* flag to indicate C was specified */

} TXLmodel;

/* instance parameters */
#define TXL_IN_NODE 1
#define TXL_OUT_NODE 2
#define TXL_LENGTH 3

/* model parameters */
#define TXL_R 101
#define TXL_C 102
#define TXL_G 103
#define TXL_L 104
#define TXL_length 105
#define TXL_MOD_R 106

#include "txlext.h"
extern VI_list_txl *pool_vi_txl;

#endif /*TXL*/
