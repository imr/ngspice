#ifndef TXL
#define TXL

#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/gendefs.h"
#include "ngspice/complex.h"
#include "ngspice/noisedef.h"
#include "ngspice/swec.h"

/* information used to describe a single instance */

typedef struct sTXLinstance {

    struct GENinstance gen;

#define TXLmodPtr(inst) ((struct sTXLmodel *)((inst)->gen.GENmodPtr))
#define TXLnextInstance(inst) ((struct sTXLinstance *)((inst)->gen.GENnextInstance))
#define TXLname gen.GENname
#define TXLstates gen.GENstate

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
	
	double *TXLposPosPtr;
	double *TXLposNegPtr;
	double *TXLnegPosPtr;
	double *TXLnegNegPtr;
	double *TXLibr1PosPtr;
	double *TXLibr2NegPtr;
	double *TXLposIbr1Ptr;
	double *TXLnegIbr2Ptr;
	double *TXLibr1NegPtr;
	double *TXLibr2PosPtr;
	double *TXLibr1Ibr1Ptr;
	double *TXLibr2Ibr2Ptr;
	double *TXLibr1Ibr2Ptr;
	double *TXLibr2Ibr1Ptr;
	
	unsigned TXLibr1Given : 1;
	unsigned TXLibr2Given : 1;
	unsigned TXLdcGiven : 1;
	unsigned TXLlengthgiven : 1;   /* flag to indicate that instance parameter len is specified */

#ifdef KLU
    BindElement *TXLposPosBinding ;
    BindElement *TXLposNegBinding ;
    BindElement *TXLnegPosBinding ;
    BindElement *TXLnegNegBinding ;
    BindElement *TXLibr1PosBinding ;
    BindElement *TXLibr2NegBinding ;
    BindElement *TXLnegIbr2Binding ;
    BindElement *TXLposIbr1Binding ;
    BindElement *TXLibr1Ibr1Binding ;
    BindElement *TXLibr2Ibr2Binding ;
    BindElement *TXLibr1NegBinding ;
    BindElement *TXLibr2PosBinding ;
    BindElement *TXLibr1Ibr2Binding ;
    BindElement *TXLibr2Ibr1Binding ;
#endif

} TXLinstance ;


/* per model data */

typedef struct sTXLmodel {       /* model structure for a txl */

    struct GENmodel gen;

#define TXLmodType gen.GENmodType
#define TXLnextModel(inst) ((struct sTXLmodel *)((inst)->gen.GENnextModel))
#define TXLinstances(inst) ((TXLinstance *)((inst)->gen.GENinstances))
#define TXLmodName gen.GENmodName

	double R;
	double L;
	double G;
	double C;
	double length;
    unsigned Rgiven : 1;   /* flag to indicate R was specified */
    unsigned Lgiven : 1;   /* flag to indicate L was specified */
    unsigned Ggiven : 1;   /* flag to indicate G was specified */
    unsigned Cgiven : 1;   /* flag to indicate C was specified */
    unsigned lengthgiven : 1;   /* flag to indicate length was specified */

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
