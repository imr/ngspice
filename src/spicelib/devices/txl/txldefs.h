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
enum {
    TXL_IN_NODE = 1,
    TXL_OUT_NODE,
    TXL_LENGTH,
};

/* model parameters */
enum {
    TXL_R = 101,
    TXL_C,
    TXL_G,
    TXL_L,
    TXL_length,
    TXL_MOD_R,
};

#include "txlext.h"
extern VI_list_txl *pool_vi_txl;

#endif /*TXL*/
