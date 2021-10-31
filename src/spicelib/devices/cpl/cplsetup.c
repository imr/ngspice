/**********
Copyright 1992 Regents of the University of California.  All rights
reserved.
Author: 1992 Charles Hough
Modified: 2004 Paolo Nenzi - (ng)spice integration
**********/


#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "cpldefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

#include "ngspice/multi_line.h"
#include "cplhash.h"



#define VECTOR_ALLOC(type, vec, n) {            \
        vec = TMALLOC(type *, n);               \
        memsaved(vec);                          \
}

#define MATRIX_ALLOC(type, mat, m, j) {         \
        int k;                                  \
        mat = TMALLOC(type **, m);              \
        memsaved(mat);                          \
        for (k = 0; k < m; k++) {               \
            VECTOR_ALLOC(type, mat[k], j);      \
        }                                       \
}

#define VECTOR_FREE(vec) {                      \
        memdeleted(vec);                        \
        tfree(vec);                             \
}


#define MATRIX_FREE(mat, m, j) {                \
        int k;                                  \
        for (k = 0; k < m; k++) {               \
            memdeleted(mat[k]);                 \
            tfree(mat[k]);                      \
        }                                       \
        memdeleted(mat);                        \
        tfree(mat);                             \
}


#define CPLTFREE(ptr) {                         \
        memdeleted(ptr);                        \
        tfree(ptr);                             \
}

#define MAX_DEG 8
#define epsilon 1.0e-88
#define MAX_STRING 128

static  double  ZY[MAX_DIM][MAX_DIM];
static  double  Sv[MAX_DIM][MAX_DIM];
static  double  D[MAX_DIM];
static  double  Y5[MAX_DIM][MAX_DIM];
static  double  Y5_1[MAX_DIM][MAX_DIM];
static  double  Sv_1[MAX_DIM][MAX_DIM];

static  double R_m[MAX_DIM][MAX_DIM];
static  double G_m[MAX_DIM][MAX_DIM];
static  double L_m[MAX_DIM][MAX_DIM];
static  double C_m[MAX_DIM][MAX_DIM];
static  double length;
static  double TAU[MAX_DIM];

static  double  A[MAX_DIM][2 * MAX_DIM];

static  double  frequency[MAX_DEG];

static  double Si[MAX_DIM][MAX_DIM];
static  double Si_1[MAX_DIM][MAX_DIM];

/*  MacLaurin Series  */
static  double* SiSv_1[MAX_DIM][MAX_DIM];
static  double* Sip[MAX_DIM][MAX_DIM];
static  double* Si_1p[MAX_DIM][MAX_DIM];
static  double* Sv_1p[MAX_DIM][MAX_DIM];
static  double* W[MAX_DIM];

static  Mult_Out IWI[MAX_DIM][MAX_DIM];
static  Mult_Out IWV[MAX_DIM][MAX_DIM];
static  Single_Out SIV[MAX_DIM][MAX_DIM];
static  double  At[4][4];
static  double  Scaling_F;
static  double  Scaling_F2;

/* misc.c match */
static void new_memory(int, int, int);
static double* vector(int, int);
static void free_vector(double*, int, int);
static void polint(double*, double*, int, double, double*, double*);
static int match(int, double*, double*, double*);
/* static int match_x(int, double*, double*, double*); */
static int Gaussian_Elimination2(int, int);
static void eval_Si_Si_1(int, double);
static void loop_ZY(int, double);
static void poly_matrix(double* A[MAX_DIM][MAX_DIM], int dim, int deg);
/* static int checkW(double*, double); */
static void poly_W(int, int);
static void eval_frequency(int, int);
static void store(int, int);
static void store_SiSv_1(int, int);
/*static int check(); quale è il prototipo ?*/
static int coupled(int);
static int generate_out(int, int);
static int ReadCpL(CPLinstance*, CKTcircuit*);
/* static int divC(double, double, double, double, double*, double*); */

/* mult */
static void mult_p(double*, double*, double*, int, int, int);
static void matrix_p_mult(double* A[MAX_DIM][MAX_DIM],
	double* D[MAX_DIM],
	double* B[MAX_DIM][MAX_DIM],
	int     dim, int deg, int deg_o,
	Mult_Out  X[MAX_DIM][MAX_DIM]);
static double approx_mode(double*, double*, double);
static double eval2(double, double, double, double);
static int get_c(double, double, double, double, double, double, double, double*, double*);
static int Pade_apx(double, double*, double*, double*, double*, double*, double*, double*);
static int Gaussian_Elimination(int);
static double root3(double, double, double, double);
static int div3(double, double, double, double, double*, double*);
static int find_roots(double, double, double, double*, double*, double*);

static NODE* insert_node(char*);
static NDnamePt insert_ND(char*, NDnamePt*);
static NODE* NEW_node(void);
static NDnamePt ndn_btree;
static NODE* node_tab;
#define epsi_mult 1e-28

/* diag */
static MAXE_PTR sort(MAXE_PTR, double, int, int, MAXE_PTR);
static void ordering(void);
static MAXE_PTR delete_1(MAXE_PTR*, int);
static void reordering(int, int);
static void diag(int);
static int rotate(int, int, int);

#define epsi 1.0e-16
static char* message = "tau of coupled lines is larger than max time step";

/* ARGSUSED */
int
CPLsetup(SMPmatrix* matrix, GENmodel* inModel, CKTcircuit* ckt, int* state)
{
	CPLmodel* model = (CPLmodel*)inModel;
	CPLinstance* here;
	CKTnode* tmp, * node;
	int error, m, p;
	char** branchname;
	int noL;

	NG_IGNORE(state);

	/* hash table for local gc */
	mem_init();

	/*  loop through all the models */
	for (; model != NULL; model = CPLnextModel(model)) {

		if (!model->Rmgiven) {
			SPfrontEnd->IFerrorf(ERR_FATAL,
				"model %s: lossy line series resistance not given", model->CPLmodName);
			return(E_BADPARM);
		}
		if (!model->Gmgiven) {
			SPfrontEnd->IFerrorf(ERR_FATAL,
				"model %s: lossy line parallel conductance not given", model->CPLmodName);
			return(E_BADPARM);
		}
		if (!model->Lmgiven) {
			SPfrontEnd->IFerrorf(ERR_FATAL,
				"model %s: lossy line series inductance not given", model->CPLmodName);
			return (E_BADPARM);
		}
		if (!model->Cmgiven) {
			SPfrontEnd->IFerrorf(ERR_FATAL,
				"model %s: lossy line parallel capacitance not given", model->CPLmodName);
			return (E_BADPARM);
		}
		if (!model->lengthgiven) {
			SPfrontEnd->IFerrorf(ERR_FATAL,
				"model %s: lossy line length must be given", model->CPLmodName);
			return (E_BADPARM);
		}

		/* loop through all the instances of the model */
		for (here = CPLinstances(model); here != NULL;
			here = CPLnextInstance(here)) {

			if (!here->CPLlengthGiven)
				here->CPLlength = 0.0;

			/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

			noL = here->dimension;

			here->CPLposNodes = TMALLOC(int, noL);
			memsaved(here->CPLposNodes);
			here->CPLnegNodes = TMALLOC(int, noL);
			memsaved(here->CPLnegNodes);
			here->CPLibr1 = TMALLOC(int, noL);
			memsaved(here->CPLibr1);
			here->CPLibr2 = TMALLOC(int, noL);
			memsaved(here->CPLibr2);

			VECTOR_ALLOC(double, here->CPLibr1Ibr1Ptr, noL);
			VECTOR_ALLOC(double, here->CPLibr2Ibr2Ptr, noL);
			VECTOR_ALLOC(double, here->CPLposIbr1Ptr, noL);
			VECTOR_ALLOC(double, here->CPLnegIbr2Ptr, noL);
			VECTOR_ALLOC(double, here->CPLposPosPtr, noL);
			VECTOR_ALLOC(double, here->CPLnegNegPtr, noL);
			VECTOR_ALLOC(double, here->CPLnegPosPtr, noL);
			VECTOR_ALLOC(double, here->CPLposNegPtr, noL);

			MATRIX_ALLOC(double, here->CPLibr1PosPtr, noL, noL);
			MATRIX_ALLOC(double, here->CPLibr2NegPtr, noL, noL);
			MATRIX_ALLOC(double, here->CPLibr1NegPtr, noL, noL);
			MATRIX_ALLOC(double, here->CPLibr2PosPtr, noL, noL);
			MATRIX_ALLOC(double, here->CPLibr1Ibr2Ptr, noL, noL);
			MATRIX_ALLOC(double, here->CPLibr2Ibr1Ptr, noL, noL);


			branchname = TMALLOC(char*, here->dimension);
			memsaved(branchname);
			if (!here->CPLibr1Given) {
				for (m = 0; m < here->dimension; m++) {
					branchname[m] = TMALLOC(char, MAX_STRING);
					memsaved(branchname[m]);
					sprintf(branchname[m], "branch1_%d", m);
					error =
						CKTmkCur(ckt, &tmp, here->CPLname, branchname[m]);
					if (error) return (error);
					here->CPLibr1[m] = tmp->number;
					CPLTFREE(branchname[m]);
				}
				here->CPLibr1Given = 1;
			}
			CPLTFREE(branchname);
			branchname = TMALLOC(char*, here->dimension);
			memsaved(branchname);
			if (!here->CPLibr2Given) {
				for (m = 0; m < here->dimension; m++) {
					branchname[m] = TMALLOC(char, MAX_STRING);
					memsaved(branchname[m]);
					sprintf(branchname[m], "branch2_%d", m);
					error =
						CKTmkCur(ckt, &tmp, here->CPLname, branchname[m]);
					if (error) return (error);
					here->CPLibr2[m] = tmp->number;
					CPLTFREE(branchname[m]);
				}
				here->CPLibr2Given = 1;
			}
			CPLTFREE(branchname);

			for (m = 0; m < here->dimension; m++) {
				for (node = ckt->CKTnodes; node; node = node->next) {
					if (strcmp(here->in_node_names[m],
						node->name) == 0) {
						here->CPLposNodes[m] = node->number;
					}
				}
			}
			for (m = 0; m < here->dimension; m++) {
				for (node = ckt->CKTnodes; node; node = node->next) {
					if (strcmp(here->out_node_names[m],
						node->name) == 0) {
						here->CPLnegNodes[m] = node->number;
					}
				}
			}

			for (m = 0; m < here->dimension; m++) {
				TSTALLOC(CPLibr1Ibr1Ptr[m], CPLibr1[m], CPLibr1[m]);
				TSTALLOC(CPLibr2Ibr2Ptr[m], CPLibr2[m], CPLibr2[m]);
				TSTALLOC(CPLposIbr1Ptr[m], CPLposNodes[m], CPLibr1[m]);
				TSTALLOC(CPLnegIbr2Ptr[m], CPLnegNodes[m], CPLibr2[m]);
				TSTALLOC(CPLposPosPtr[m], CPLposNodes[m], CPLposNodes[m]);
				TSTALLOC(CPLnegNegPtr[m], CPLnegNodes[m], CPLnegNodes[m]);
				TSTALLOC(CPLnegPosPtr[m], CPLnegNodes[m], CPLposNodes[m]);
				TSTALLOC(CPLposNegPtr[m], CPLposNodes[m], CPLnegNodes[m]);

				for (p = 0; p < here->dimension; p++) {

					TSTALLOC(CPLibr1PosPtr[m][p], CPLibr1[m], CPLposNodes[p]);
					TSTALLOC(CPLibr2NegPtr[m][p], CPLibr2[m], CPLnegNodes[p]);
					TSTALLOC(CPLibr1NegPtr[m][p], CPLibr1[m], CPLnegNodes[p]);
					TSTALLOC(CPLibr2PosPtr[m][p], CPLibr2[m], CPLposNodes[p]);
					TSTALLOC(CPLibr1Ibr2Ptr[m][p], CPLibr1[m], CPLibr2[p]);
					TSTALLOC(CPLibr2Ibr1Ptr[m][p], CPLibr2[m], CPLibr1[p]);

				}
			}

			ReadCpL(here, ckt);

		}
	}

	return(OK);
}



int
CPLunsetup(GENmodel* inModel, CKTcircuit* ckt)
{
	CPLmodel* model;
	CPLinstance* here;
	int m;
	int noL;

	for (model = (CPLmodel*)inModel; model != NULL;
		model = CPLnextModel(model)) {
		for (here = CPLinstances(model); here != NULL;
			here = CPLnextInstance(here)) {

			noL = here->dimension;

			VECTOR_FREE(here->CPLibr1Ibr1Ptr);
			VECTOR_FREE(here->CPLibr2Ibr2Ptr);
			VECTOR_FREE(here->CPLposIbr1Ptr);
			VECTOR_FREE(here->CPLnegIbr2Ptr);
			VECTOR_FREE(here->CPLposPosPtr);
			VECTOR_FREE(here->CPLnegNegPtr);
			VECTOR_FREE(here->CPLnegPosPtr);
			VECTOR_FREE(here->CPLposNegPtr);


			MATRIX_FREE(here->CPLibr1PosPtr, noL, noL);
			MATRIX_FREE(here->CPLibr2NegPtr, noL, noL);
			MATRIX_FREE(here->CPLibr1NegPtr, noL, noL);
			MATRIX_FREE(here->CPLibr2PosPtr, noL, noL);
			MATRIX_FREE(here->CPLibr1Ibr2Ptr, noL, noL);
			MATRIX_FREE(here->CPLibr2Ibr1Ptr, noL, noL);


			for (m = 0; m < noL; m++) {
				if (here->CPLibr2[m]) {
					CKTdltNNum(ckt, here->CPLibr2[m]);
					here->CPLibr2[m] = 0;
				}
			}

			for (m = 0; m < noL; m++) {
				if (here->CPLibr1[m]) {
					CKTdltNNum(ckt, here->CPLibr1[m]);
					here->CPLibr1[m] = 0;
				}
			}

			CPLTFREE(here->CPLposNodes);
			CPLTFREE(here->CPLnegNodes);
			CPLTFREE(here->CPLibr1);
			CPLTFREE(here->CPLibr2);

			/* reset switches */
			here->CPLdcGiven = 0;
			here->CPLibr1Given = 0;
			here->CPLibr2Given = 0;
		}
	}
	mem_delete();
	ndn_btree = NULL;
	return OK;
}



static int
ReadCpL(CPLinstance* here, CKTcircuit* ckt)
{
	int i, j, noL, counter;
	double f;
	char* name;
	CPLine* c, * c2;
	ECPLine* ec;
	NODE* nd;
	RLINE* lines[MAX_CP_TX_LINES];
	ERLINE* er;

	c = TMALLOC(CPLine, 1);
	memsaved(c);
	c2 = TMALLOC(CPLine, 1);
	memsaved(c2);
	c->vi_head = c->vi_tail = NULL;
	noL = c->noL = here->dimension;
	here->cplines = c;
	here->cplines2 = c2;

	for (i = 0; i < noL; i++) {
		ec = TMALLOC(ECPLine, 1);
		memsaved(ec);
		name = here->in_node_names[i];
		nd = insert_node(name);
		ec->link = nd->cplptr;
		nd->cplptr = ec;
		ec->line = c;
		c->in_node[i] = nd;
		c2->in_node[i] = nd;

		er = TMALLOC(ERLINE, 1);
		memsaved(er);
		er->link = nd->rlptr;
		nd->rlptr = er;
		er->rl = lines[i] = TMALLOC(RLINE, 1);
		memsaved(er->rl);
		er->rl->in_node = nd;

		c->dc1[i] = c->dc2[i] = 0.0;
	}

	for (i = 0; i < noL; i++) {
		ec = TMALLOC(ECPLine, 1);
		memsaved(ec);
		name = here->out_node_names[i];
		nd = insert_node(name);
		ec->link = nd->cplptr;
		nd->cplptr = ec;
		ec->line = c;
		c->out_node[i] = nd;
		c2->out_node[i] = nd;

		er = TMALLOC(ERLINE, 1);
		memsaved(er);
		er->link = nd->rlptr;
		nd->rlptr = er;
		er->rl = lines[i];
		er->rl->out_node = nd;
	}


	counter = 0;
	for (i = 0; i < noL; i++) {
		for (j = 0; j < noL; j++) {
			if (i > j) {
				R_m[i][j] = R_m[j][i];
				G_m[i][j] = G_m[j][i];
				C_m[i][j] = C_m[j][i];
				L_m[i][j] = L_m[j][i];
			}
			else {
				f = CPLmodPtr(here)->Rm[counter];
				R_m[i][j] = CPLmodPtr(here)->Rm[counter] = MAX(f, 1.0e-4);
				G_m[i][j] = CPLmodPtr(here)->Gm[counter];
				L_m[i][j] = CPLmodPtr(here)->Lm[counter];
				C_m[i][j] = CPLmodPtr(here)->Cm[counter];
				counter++;
			}
		}
	}
	if (here->CPLlengthGiven)
		length = here->CPLlength;
	else length = CPLmodPtr(here)->length;

	for (i = 0; i < noL; i++)
		lines[i]->g = 1.0 / (R_m[i][i] * length);

	coupled(noL);

	for (i = 0; i < noL; i++) {
		double d, t;
		int k;

		c->taul[i] = TAU[i] * 1.0e+12;
		for (j = 0; j < noL; j++) {
			if (SIV[i][j].C_0 == 0.0)
				c->h1t[i][j] = NULL;
			else {
				c->h1t[i][j] = TMALLOC(TMS, 1);
				memsaved(c->h1t[i][j]);
				d = c->h1t[i][j]->aten = SIV[i][j].C_0;
				c->h1t[i][j]->ifImg = (int)(SIV[i][j].Poly[6] - 1.0);
				/* since originally 2 for img 1 for noimg */
				c->h1t[i][j]->tm[0].c = SIV[i][j].Poly[0] * d;
				c->h1t[i][j]->tm[1].c = SIV[i][j].Poly[1] * d;
				c->h1t[i][j]->tm[2].c = SIV[i][j].Poly[2] * d;
				c->h1t[i][j]->tm[0].x = SIV[i][j].Poly[3];
				c->h1t[i][j]->tm[1].x = SIV[i][j].Poly[4];
				c->h1t[i][j]->tm[2].x = SIV[i][j].Poly[5];
				if (c->h1t[i][j]->ifImg)
					c->h1C[i][j] = c->h1t[i][j]->tm[0].c + 2.0 * c->h1t[i][j]->tm[1].c;
				else {
					t = 0.0;
					for (k = 0; k < 3; k++)
						t += c->h1t[i][j]->tm[k].c;
					c->h1C[i][j] = t;
				}
			}

			for (k = 0; k < noL; k++) {
				if (IWI[i][j].C_0[k] == 0.0)
					c->h2t[i][j][k] = NULL;
				else {
					c->h2t[i][j][k] = TMALLOC(TMS, 1);
					memsaved(c->h2t[i][j][k]);
					d = c->h2t[i][j][k]->aten = IWI[i][j].C_0[k];
					c->h2t[i][j][k]->ifImg = (int)(IWI[i][j].Poly[k][6] - 1.0);
					/* since originally 2 for img 1 for noimg */
					c->h2t[i][j][k]->tm[0].c = IWI[i][j].Poly[k][0] * d;
					c->h2t[i][j][k]->tm[1].c = IWI[i][j].Poly[k][1] * d;
					c->h2t[i][j][k]->tm[2].c = IWI[i][j].Poly[k][2] * d;
					c->h2t[i][j][k]->tm[0].x = IWI[i][j].Poly[k][3];
					c->h2t[i][j][k]->tm[1].x = IWI[i][j].Poly[k][4];
					c->h2t[i][j][k]->tm[2].x = IWI[i][j].Poly[k][5];
					if (c->h2t[i][j][k]->ifImg)
						c->h2C[i][j][k] = c->h2t[i][j][k]->tm[0].c + 2.0 *
						c->h2t[i][j][k]->tm[1].c;
					else
						c->h2C[i][j][k] = c->h2t[i][j][k]->tm[0].c +
						c->h2t[i][j][k]->tm[1].c +
						c->h2t[i][j][k]->tm[2].c;
				}
				if (IWV[i][j].C_0[k] == 0.0)
					c->h3t[i][j][k] = NULL;
				else {
					c->h3t[i][j][k] = TMALLOC(TMS, 1);
					memsaved(c->h3t[i][j][k]);
					d = c->h3t[i][j][k]->aten = IWV[i][j].C_0[k];
					c->h3t[i][j][k]->ifImg = (int)(IWV[i][j].Poly[k][6] - 1.0);
					/* since originally 2 for img 1 for noimg */
					c->h3t[i][j][k]->tm[0].c = IWV[i][j].Poly[k][0] * d;
					c->h3t[i][j][k]->tm[1].c = IWV[i][j].Poly[k][1] * d;
					c->h3t[i][j][k]->tm[2].c = IWV[i][j].Poly[k][2] * d;
					c->h3t[i][j][k]->tm[0].x = IWV[i][j].Poly[k][3];
					c->h3t[i][j][k]->tm[1].x = IWV[i][j].Poly[k][4];
					c->h3t[i][j][k]->tm[2].x = IWV[i][j].Poly[k][5];
					if (c->h3t[i][j][k]->ifImg)
						c->h3C[i][j][k] = c->h3t[i][j][k]->tm[0].c + 2.0 *
						c->h3t[i][j][k]->tm[1].c;
					else
						c->h3C[i][j][k] = c->h3t[i][j][k]->tm[0].c +
						c->h3t[i][j][k]->tm[1].c +
						c->h3t[i][j][k]->tm[2].c;
				}
			}
		}
	}

	for (i = 0; i < noL; i++) {
		if (c->taul[i] < ckt->CKTmaxStep) {
			errMsg = TMALLOC(char, strlen(message) + 1);
			memsaved(errMsg);
			strcpy(errMsg, message);
			return(-1);
		}
	}

	return(1);
}


/****************************************************************
	 misc.c      Miscellaneous procedures for simulation of
				 coupled transmission lines.
 ****************************************************************/


static void
new_memory(int dim, int deg, int deg_o)
{
	int i, j;

	NG_IGNORE(deg);

	for (i = 0; i < dim; i++)
		for (j = 0; j < dim; j++) {
			SiSv_1[i][j] = (double*)calloc((size_t)(deg_o + 1), sizeof(double));
			memsaved(SiSv_1[i][j]);
		}

	for (i = 0; i < dim; i++)
		for (j = 0; j < dim; j++) {
			Sip[i][j] = (double*)calloc((size_t)(deg_o + 1), sizeof(double));
			memsaved(Sip[i][j]);
		}

	for (i = 0; i < dim; i++)
		for (j = 0; j < dim; j++) {
			Si_1p[i][j] = (double*)calloc((size_t)(deg_o + 1), sizeof(double));
			memsaved(Si_1p[i][j]);
		}

	for (i = 0; i < dim; i++)
		for (j = 0; j < dim; j++) {
			Sv_1p[i][j] = (double*)calloc((size_t)(deg_o + 1), sizeof(double));
			memsaved(Sv_1p[i][j]);
		}

	for (i = 0; i < dim; i++) {
		W[i] = (double*)calloc(MAX_DEG, sizeof(double));
		memsaved(W[i]);
	}
}

/***
 ***/

 /****************************************************************
	  match     Create a polynomial matching given data points
  ****************************************************************/


static double*
vector(int nl, int nh)
{
	double* v = TMALLOC(double, nh - nl + 1);
	memsaved(v);

	if (!v) {
		fprintf(stderr, "Memory Allocation Error by tmalloc in vector().\n");
		fprintf(stderr, "...now exiting to system ...\n");
		controlled_exit(EXIT_FAILURE);
	}

	return v - nl;
}

static void
free_vector(double* v, int nl, int nh)
{
	NG_IGNORE(nh);
	double* freev = v + nl;
	CPLTFREE(freev);
}

static void
polint(double* xa, double* ya, int n, double x, double* y, double* dy)
/*
   Given arrays xa[1..n] and ya[1..n], and given a value x, this routine
   returns a value y, and an error estimate dy.  If P(x) is the
   polynomial of degree n-1 such that P(xa) = ya, then the returned
   value y = P(x)
 */
{
	int i, m, ns = 1;
	double den, dif, dift, ho, hp, w;
	double* c, * d;

	dif = ABS(x - xa[1]);
	c = vector(1, n);
	d = vector(1, n);
	for (i = 1; i <= n; i++) {
		if ((dift = ABS(x - xa[i])) < dif) {
			ns = i;
			dif = dift;
		}
		c[i] = ya[i];
		d[i] = ya[i];
	}
	*y = ya[ns--];
	for (m = 1; m < n; m++) {
		for (i = 1; i <= n - m; i++) {
			ho = xa[i] - x;
			hp = xa[i + m] - x;
			w = c[i + 1] - d[i];
			if ((den = ho - hp) == 0.0) {
				fprintf(stderr, "(Error) in routine POLINT\n");
				fprintf(stderr, "...now exiting to system ...\n");
				controlled_exit(EXIT_FAILURE);
			}
			den = w / den;
			d[i] = hp * den;
			c[i] = ho * den;
		}
		*y += (*dy = (2 * ns < (n - m) ? c[ns + 1] : d[ns--]));
	}
	free_vector(d, 1, n);
	free_vector(c, 1, n);
}

static int
match(int n, double* cof, double* xa, double* ya)
/*
   Given arrays xa[0..n] and ya[0..n] containing a tabulated function
   ya = f(xa), this routine returns an array of coefficients cof[0..n],
   such that ya[i] = sum_j {cof[j]*xa[i]**j}.
 */
{
	int k, j, i;
	double xmin, dy, * x, * y, * xx;

	x = vector(0, n);
	y = vector(0, n);
	xx = vector(0, n);
	for (j = 0; j <= n; j++) {
		x[j] = xa[j];
		xx[j] = y[j] = ya[j];
	}
	for (j = 0; j <= n; j++) {
		polint(x - 1, y - 1, n + 1 - j, 0.0, &cof[j], &dy);
		xmin = 1.0e38;
		k = -1;
		for (i = 0; i <= n - j; i++) {
			if (ABS(x[i]) < xmin) {
				xmin = ABS(x[i]);
				k = i;
			}
			if (x[i]) y[i] = (y[i] - cof[j]) / x[i];
		}
		for (i = k + 1; i <= n - j; i++) {
			y[i - 1] = y[i];
			x[i - 1] = x[i];
		}
	}
	free_vector(y, 0, n);
	free_vector(x, 0, n);
	free_vector(xx, 0, n);

	/****   check   ****/
	/*
	for (i = 0; i <= n; i++) {
	   xmin = xa[i];
	   dy = cof[0];
	   for (j = 1; j <= n; j++) {
		  dy += xmin * cof[j];
		  xmin *= xa[i];
	   }
	   printf("*** real x = %e y = %e\n", xa[i], xx[i]);
	   printf("*** calculated  y = %e\n", dy);
	   printf("*** error = %e \% \n", (dy-xx[i])/xx[i]);
	}
	*/
	return 0;
}

/***
 ***/
 /***
 static int
 match_x(int dim, double *Alfa, double *X, double *Y)
 {
	int i, j;
	double f;
	double scale;

	****   check   ****
	double xx[16];
	for (i = 0; i <= dim; i++)
	   xx[i] = Y[i];

	if (Y[1] == Y[0])
	   scale = 1.0;
	else
	   scale = X[1] / (Y[1] - Y[0]);
	for (i = 0; i < dim; i++) {
	   f = X[i+1];
	   for (j = dim-1; j >= 0; j--) {
		  A[i][j] = f;
		  f *= X[i+1];
	   }
	   A[i][dim] = (Y[i+1] - Y[0])*scale;
	}
	Gaussian_Elimination2(dim, 1);
	Alfa[0] = Y[0];
	for (i = 1; i <= dim; i++)
	   Alfa[i] = A[dim-i][dim] / scale;

	****   check   ****
	*
	for (i = 0; i <= dim; i++) {
	   f = X[i];
	   scale = Alfa[0];
	   for (j = 1; j <= dim; j++) {
		  scale += f * Alfa[j];
		  f *= X[i];
	   }
	   printf("*** real x = %e y = %e\n", X[i], xx[i]);
	   printf("*** calculated  y = %e\n", scale);
	   printf("*** error = %e \% \n", (scale-xx[i])/xx[i]);
	}
	*

	return(1);
 }
 ***/
 /***
  ***/

static int
Gaussian_Elimination2(int dims, int type)
/*  type = 1 : to solve a linear system
		  -1 : to inverse a matrix  */
{
	int i, j, k, dim;
	double f;
	double max;
	int imax;

	if (type == -1)
		dim = 2 * dims;
	else
		dim = dims;

	for (i = 0; i < dims; i++) {
		imax = i;
		max = ABS(A[i][i]);
		for (j = i + 1; j < dim; j++)
			if (ABS(A[j][i]) > max) {
				imax = j;
				max = ABS(A[j][i]);
			}
		if (max < epsilon) {
			fprintf(stderr, " can not choose a pivot (misc)\n");
			controlled_exit(EXIT_FAILURE);
		}
		if (imax != i)
			for (k = i; k <= dim; k++) {
				SWAP(double, A[i][k], A[imax][k]);
			}

		f = 1.0 / A[i][i];
		A[i][i] = 1.0;

		for (j = i + 1; j <= dim; j++)
			A[i][j] *= f;

		for (j = 0; j < dims; j++) {
			if (i == j)
				continue;
			f = A[j][i];
			A[j][i] = 0.0;
			for (k = i + 1; k <= dim; k++)
				A[j][k] -= f * A[i][k];
		}
	}

	return(1);
}

/***

static void
eval_Si_Si_1(int dim, double y)
{
   int i, j, k;

   for (i = 0; i < dim; i++)
	  for (j = 0; j < dim; j++) {
		 Si_1[i][j] = 0.0;
		 for (k = 0; k < dim; k++)
			if (k == j)
						Si_1[i][j] += Sv_1[i][k] *
								(y * R_m[k][j] + Scaling_F * L_m[k][j]);
			else
						Si_1[i][j] += Sv_1[i][k] * L_m[k][j] * Scaling_F;
						/
						Si_1[i][j] *= Scaling_F;
						/
	  }

   for (i = 0; i < dim; i++)
	  for (j = 0; j < dim; j++)
		 Si_1[i][j] /= sqrt((double) D[i]);

   for (i = 0; i < dim; i++) {
	  for (j = 0; j < dim; j++)
		 A[i][j] = Si_1[i][j];
	  for (j = dim; j < 2* dim; j++)
		 A[i][j] = 0.0;
	  A[i][i+dim] = 1.0;
   }
   Gaussian_Elimination2(dim, -1);

   for (i = 0; i < dim; i++)
	  for (j = 0; j < dim; j++)
		 Si[i][j] = A[i][j+dim];
}

***/

static void
eval_Si_Si_1(int dim, double y)
{
	int i, j, k;

	for (i = 0; i < dim; i++)
		for (j = 0; j < dim; j++) {
			Si_1[i][j] = 0.0;
			for (k = 0; k < dim; k++)
				Si_1[i][j] += Sv_1[i][k] * (y * R_m[k][j] + Scaling_F * L_m[k][j]);
			/*
			else
			Si_1[i][j] += Sv_1[i][k] * L_m[k][j] * Scaling_F;
			Si_1[i][j] *= Scaling_F;
			 */
		}

	for (i = 0; i < dim; i++)
		for (j = 0; j < dim; j++)
			Si_1[i][j] /= sqrt(D[i]);

	for (i = 0; i < dim; i++) {
		for (j = 0; j < dim; j++)
			A[i][j] = Si_1[i][j];
		for (j = dim; j < 2 * dim; j++)
			A[i][j] = 0.0;
		A[i][i + dim] = 1.0;
	}
	Gaussian_Elimination2(dim, -1);

	for (i = 0; i < dim; i++)
		for (j = 0; j < dim; j++)
			Si[i][j] = A[i][j + dim];
}

/***

static void
loop_ZY(int dim, double y)
{
   int i, j, k;
   double fmin, fmin1;

   for (i = 0; i < dim; i++)
	  for (j = 0; j < dim; j++)
		 if (i == j)
			ZY[i][j] = Scaling_F * C_m[i][i] + G_m[i] * y;
		 else
			ZY[i][j] = Scaling_F * C_m[i][j];
   diag(dim);
   fmin = D[0];
   for (i = 1; i < dim; i++)
	  if (D[i] < fmin)
		 fmin = D[i];
   if (fmin < 0) {
	  fprintf(stderr, "(Error) The capacitance matrix of the multiconductor system is not positive definite.\n");
	  exit(0);
   } else {
	  fmin = sqrt(fmin);
	  fmin1 = 1 / fmin;
   }
   for (i = 0; i < dim; i++)
	  D[i] = sqrt((double) D[i]);
   for (i = 0; i < dim; i++)
	  for (j = 0; j < dim; j++) {
		 Y5[i][j] = D[i] * Sv[j][i];
		 Y5_1[i][j] = Sv[j][i] / D[i];
	  }
   for (i = 0; i < dim; i++)
	  for (j = 0; j < dim; j++) {
		 Sv_1[i][j] = 0.0;
		 for (k = 0; k < dim; k++)
			Sv_1[i][j] += Sv[i][k] * Y5[k][j];
	  }
   for (i = 0; i < dim; i++)
	  for (j = 0; j < dim; j++)
		 Y5[i][j] = Sv_1[i][j];
   for (i = 0; i < dim; i++)
	  for (j = 0; j < dim; j++) {
		 Sv_1[i][j] = 0.0;
		 for (k = 0; k < dim; k++)
			Sv_1[i][j] += Sv[i][k] * Y5_1[k][j];
	  }
   for (i = 0; i < dim; i++)
	  for (j = 0; j < dim; j++)
		 Y5_1[i][j] = Sv_1[i][j];

   for (i = 0; i < dim; i++)
	  for (j = 0; j < dim; j++) {
		 ZY[i][j] = 0.0;
		 for (k = 0; k < dim; k++)
			if (k == i)
			   ZY[i][j] += (Scaling_F *  L_m[i][i] + R_m[i] * y) *
							   Y5[k][j];
			else
			   ZY[i][j] += L_m[i][k] * Y5[k][j] * Scaling_F;
	  }
   for (i = 0; i < dim; i++)
	  for (j = 0; j < dim; j++) {
		 Sv_1[i][j] = 0.0;
		 for (k = 0; k < dim; k++)
			Sv_1[i][j] += Y5[i][k] * ZY[k][j];
	  }
   for (i = 0; i < dim; i++)
	  for (j = 0; j < dim; j++)
		 ZY[i][j] = Sv_1[i][j];

   diag(dim);

   for (i = 0; i < dim; i++)
	  for (j = 0; j < dim; j++) {
		 Sv_1[i][j] = 0.0;
		 for (k = 0; k < dim; k++)
			Sv_1[i][j] += Sv[k][i] * Y5[k][j];
		 Sv_1[i][j] *= fmin1;
	  }
   for (i = 0; i < dim; i++)
	  for (j = 0; j < dim; j++) {
		 ZY[i][j] = 0.0;
		 for (k = 0; k < dim; k++)
			ZY[i][j] += Y5_1[i][k] * Sv[k][j];
		 ZY[i][j] *= fmin;
	  }
   for (i = 0; i < dim; i++)
	  for (j = 0; j < dim; j++)
		 Sv[i][j] = ZY[i][j];

}
***/

static void
loop_ZY(int dim, double y)
{
	int i, j, k;
	double fmin, fmin1 = 0.0;

	for (i = 0; i < dim; i++)
		for (j = 0; j < dim; j++)
			ZY[i][j] = Scaling_F * C_m[i][j] + G_m[i][j] * y;
	/*
	else
	ZY[i][j] = Scaling_F * C_m[i][j];
	 */
	diag(dim);
	fmin = D[0];
	for (i = 1; i < dim; i++)
		if (D[i] < fmin)
			fmin = D[i];
	if (fmin < 0) {
		fprintf(stderr, "(Error) The capacitance matrix of the multiconductor system is not positive definite.\n");
		controlled_exit(EXIT_FAILURE);
	}
	else {
		fmin = sqrt(fmin);
		fmin1 = 1 / fmin;
	}
	for (i = 0; i < dim; i++)
		D[i] = sqrt(D[i]);
	for (i = 0; i < dim; i++)
		for (j = 0; j < dim; j++) {
			Y5[i][j] = D[i] * Sv[j][i];
			Y5_1[i][j] = Sv[j][i] / D[i];
		}
	for (i = 0; i < dim; i++)
		for (j = 0; j < dim; j++) {
			Sv_1[i][j] = 0.0;
			for (k = 0; k < dim; k++)
				Sv_1[i][j] += Sv[i][k] * Y5[k][j];
		}
	for (i = 0; i < dim; i++)
		for (j = 0; j < dim; j++)
			Y5[i][j] = Sv_1[i][j];
	for (i = 0; i < dim; i++)
		for (j = 0; j < dim; j++) {
			Sv_1[i][j] = 0.0;
			for (k = 0; k < dim; k++)
				Sv_1[i][j] += Sv[i][k] * Y5_1[k][j];
		}
	for (i = 0; i < dim; i++)
		for (j = 0; j < dim; j++)
			Y5_1[i][j] = Sv_1[i][j];

	for (i = 0; i < dim; i++)
		for (j = 0; j < dim; j++) {
			ZY[i][j] = 0.0;
			for (k = 0; k < dim; k++)
				ZY[i][j] += (Scaling_F * L_m[i][k] + R_m[i][k] * y) * Y5[k][j];
			/*
			   else
				  ZY[i][j] += L_m[i][k] * Y5[k][j] * Scaling_F;
			 */
		}
	for (i = 0; i < dim; i++)
		for (j = 0; j < dim; j++) {
			Sv_1[i][j] = 0.0;
			for (k = 0; k < dim; k++)
				Sv_1[i][j] += Y5[i][k] * ZY[k][j];
		}
	for (i = 0; i < dim; i++)
		for (j = 0; j < dim; j++)
			ZY[i][j] = Sv_1[i][j];

	diag(dim);

	for (i = 0; i < dim; i++)
		for (j = 0; j < dim; j++) {
			Sv_1[i][j] = 0.0;
			for (k = 0; k < dim; k++)
				Sv_1[i][j] += Sv[k][i] * Y5[k][j];
			Sv_1[i][j] *= fmin1;
		}
	for (i = 0; i < dim; i++)
		for (j = 0; j < dim; j++) {
			ZY[i][j] = 0.0;
			for (k = 0; k < dim; k++)
				ZY[i][j] += Y5_1[i][k] * Sv[k][j];
			ZY[i][j] *= fmin;
		}
	for (i = 0; i < dim; i++)
		for (j = 0; j < dim; j++)
			Sv[i][j] = ZY[i][j];

}


/***
 ***/

static void
poly_matrix(
	double* A_in[MAX_DIM][MAX_DIM],
	int dim, int deg)
{
	int i, j;

	for (i = 0; i < dim; i++)
		for (j = 0; j < dim; j++)
			match(deg, A_in[i][j], frequency, A_in[i][j]);
}

/***
 ***/
 /***
 static int
 checkW(double *W, double d)
 {
   double f, y;
   float  y1;
   int k;

   printf("(W)y =");
   scanf("%f", &y1);

   f = W[0];
   y = y1;
   f += y * W[1];
   for (k = 2; k < 6; k++) {
	  y *= y1;
	  f += y * W[k];
   }
   printf("W[i]= %e\n ", f*exp((double)-d/y1));

   return(1);
 }
 ***/
 /***
  ***/

static void
poly_W(int dim, int deg)
{
	int i;

	for (i = 0; i < dim; i++) {
		match(deg, W[i], frequency, W[i]);
		TAU[i] = approx_mode(W[i], W[i], length);
		/*
		checkW(W[i], TAU[i]);
		*/
	}
}

/***
 ***/

static void
eval_frequency(int dim, int deg_o)
{
	int i;
	double min;

	min = D[0];

	for (i = 1; i < dim; i++)
		if (D[i] < min) {
			min = D[i];
		}

	if (min <= 0) {
		fprintf(stderr, "A mode frequency is not positive.  Abort!\n");
		controlled_exit(EXIT_FAILURE);
	}

	Scaling_F2 = 1.0 / min;
	Scaling_F = sqrt(Scaling_F2);
	min = length * 8.0;
	/*
	min *= 1.0e18;
	min = sqrt(min)*1.0e-9*length/8.0;
	 */

	frequency[0] = 0.0;

	for (i = 1; i <= deg_o; i++)
		frequency[i] = frequency[i - 1] + min;

	for (i = 0; i < dim; i++)
		D[i] *= Scaling_F2;
}

/***
 ***/

static void
store(int dim, int ind)
{
	int i, j;

	for (i = 0; i < dim; i++) {
		for (j = 0; j < dim; j++) {
			/*  store_Sip  */
			Sip[i][j][ind] = Si[i][j];
			/*  store_Si_1p  */
			Si_1p[i][j][ind] = Si_1[i][j];
			/*  store_Sv_1p  */
			Sv_1p[i][j][ind] = Sv_1[i][j];
		}
		/*  store_W  */
		W[i][ind] = D[i];
	}
}

/***
 ***/

static void
store_SiSv_1(int dim, int ind)
{
	int i, j, k;
	double temp;

	for (i = 0; i < dim; i++)
		for (j = 0; j < dim; j++) {
			temp = 0.0;
			for (k = 0; k < dim; k++)
				temp += Si[i][k] * Sv_1[k][j];
			SiSv_1[i][j][ind] = temp;
		}
}

/***
 ***/
 /***
 static int
 check(Sip, Si_1p, Sv_1p, SiSv_1p)
	double *Sip[MAX_DIM][MAX_DIM];
	double *Si_1p[MAX_DIM][MAX_DIM];
	double *Sv_1p[MAX_DIM][MAX_DIM];
	double *SiSv_1p[MAX_DIM][MAX_DIM];
 {
	double f, y;
	float  y1;
	int i, j, k;

	printf("y =");
	scanf("%f", &y1);

	printf("\n");
	printf("Si =\n");
	for (i = 0; i < 4; i++) {
	   for (j = 0; j < 4; j++) {
		  f = Sip[i][j][0];
		  y = y1;
		  f += y * Sip[i][j][1];
		  for (k = 2; k < 8; k++) {
			 y *= y1;
			 f += y * Sip[i][j][k];
		  }
		  printf("%e ", f);
	   }
	   printf("\n");
	}
	printf("\n");
	printf("Si_1 =\n");
	for (i = 0; i < 4; i++) {
	   for (j = 0; j < 4; j++) {
		  f = Si_1p[i][j][0];
		  y = y1;
		  f += y * Si_1p[i][j][1];
		  for (k = 2; k < 8; k++) {
			 y *= y1;
			 f += y * Si_1p[i][j][k];
		  }
		  printf("%e ", f);
	   }
	   printf("\n");
	}
	printf("\n");
	printf("Sv_1 =\n");
	for (i = 0; i < 4; i++) {
	   for (j = 0; j < 4; j++) {
		  f = Sv_1p[i][j][0];
		  y = y1;
		  f += y * Sv_1p[i][j][1];
		  for (k = 2; k < 8; k++) {
			 y *= y1;
			 f += y * Sv_1p[i][j][k];
		  }
		  printf("%e ", f);
	   }
	   printf("\n");
	}
	printf("\n");
	printf("SiSv_1 =\n");
	for (i = 0; i < 4; i++) {
	   for (j = 0; j < 4; j++) {
		  f = SiSv_1p[i][j][0];
		  y = y1;
		  f += y * SiSv_1p[i][j][1];
		  for (k = 2; k < 8; k++) {
			 y *= y1;
			 f += y * SiSv_1p[i][j][k];
		  }
		  printf("%e ", f);
	   }
	   printf("\n");
	}
	return(1);
 }
 ***/
 /***
  ***/

static int
coupled(int dim)
{
	int deg, deg_o;
	int i;

	deg = Right_deg;
	deg_o = Left_deg;
	new_memory(dim, deg, deg_o);

	Scaling_F = Scaling_F2 = 1.0;

	/***     y = 0 : ZY = LC    ***/
	loop_ZY(dim, 0.0);
	eval_frequency(dim, deg_o);
	eval_Si_Si_1(dim, 0.0);
	store_SiSv_1(dim, 0);
	store(dim, 0);

	/***     Step  1     ***/
	/***     Step  2     ***/
	for (i = 1; i <= deg_o; i++) {
		loop_ZY(dim, frequency[i]);
		eval_Si_Si_1(dim, frequency[i]);
		store_SiSv_1(dim, i);
		store(dim, i);
	}
	poly_matrix(Sip, dim, deg_o);
	poly_matrix(Si_1p, dim, deg_o);
	poly_matrix(Sv_1p, dim, deg_o);
	poly_W(dim, deg_o);
	matrix_p_mult(Sip, W, Si_1p, dim, deg_o, deg_o, IWI);
	matrix_p_mult(Sip, W, Sv_1p, dim, deg_o, deg_o, IWV);

	poly_matrix(SiSv_1, dim, deg_o);

	/***
	check(Sip, Si_1p, Sv_1p, SiSv_1);
	***/

	generate_out(dim, deg_o);

	return(1);
}

/***
 ***/

static int
generate_out(int dim, int deg_o)
{
	int i, j, k, rtv;
	double C;
	double* p;
	double c1, c2, c3, x1, x2, x3;

	for (i = 0; i < dim; i++)
		for (j = 0; j < dim; j++) {
			p = SiSv_1[i][j];
			SIV[i][j].C_0 = C = p[0];
			if (C == 0.0)
				continue;
			for (k = 0; k <= deg_o; k++)
				p[k] /= C;
			if (i == j) {
				rtv = Pade_apx(sqrt(G_m[i][i] / R_m[i][i]) / C,
					p, &c1, &c2, &c3, &x1, &x2, &x3);
				if (rtv == 0) {
					SIV[i][j].C_0 = 0.0;
					printf("SIV\n");
					continue;
				}
			}
			else {
				rtv = Pade_apx(0.0,
					p, &c1, &c2, &c3, &x1, &x2, &x3);
				if (rtv == 0) {
					SIV[i][j].C_0 = 0.0;
					printf("SIV\n");
					continue;
				}
			}
			p = SIV[i][j].Poly = (double*)calloc(7, sizeof(double));
			memsaved(p);
			p[0] = c1;
			p[1] = c2;
			p[2] = c3;
			p[3] = x1;
			p[4] = x2;
			p[5] = x3;
			p[6] = rtv;
		}
	for (i = 0; i < dim; i++)
		for (j = 0; j < dim; j++)
			for (k = 0; k < dim; k++) {
				p = IWI[i][j].Poly[k];
				C = IWI[i][j].C_0[k];
				if (C == 0.0)
					continue;
				if (i == j && k == i) {
					rtv = Pade_apx(
						exp(-sqrt(G_m[i][i] * R_m[i][i]) * length) / C,
						p, &c1, &c2, &c3, &x1, &x2, &x3);
					if (rtv == 0) {
						IWI[i][j].C_0[k] = 0.0;
						printf("IWI %d %d %d\n", i, j, k);
						continue;
					}
				}
				else {
					rtv = Pade_apx(0.0,
						p, &c1, &c2, &c3, &x1, &x2, &x3);
					if (rtv == 0) {
						IWI[i][j].C_0[k] = 0.0;
						printf("IWI %d %d %d\n", i, j, k);
						continue;
					}
				}
				p[0] = c1;
				p[1] = c2;
				p[2] = c3;
				p[3] = x1;
				p[4] = x2;
				p[5] = x3;
				p[6] = rtv;
			}

	for (i = 0; i < dim; i++)
		for (j = 0; j < dim; j++)
			for (k = 0; k < dim; k++) {
				p = IWV[i][j].Poly[k];
				C = IWV[i][j].C_0[k];
				if (C == 0.0)
					continue;
				if (i == j && k == i) {
					rtv = Pade_apx(sqrt(G_m[i][i] / R_m[i][i]) *
						exp(-sqrt(G_m[i][i] * R_m[i][i]) * length) / C,
						p, &c1, &c2, &c3, &x1, &x2, &x3);
					if (rtv == 0) {
						IWV[i][j].C_0[k] = 0.0;
						printf("IWV %d %d %d\n", i, j, k);
						continue;
					}
				}
				else {
					rtv = Pade_apx(0.0,
						p, &c1, &c2, &c3, &x1, &x2, &x3);
					if (rtv == 0) {
						IWV[i][j].C_0[k] = 0.0;
						printf("IWV %d %d %d\n", i, j, k);
						continue;
					}
				}
				p[0] = c1;
				p[1] = c2;
				p[2] = c3;
				p[3] = x1;
				p[4] = x2;
				p[5] = x3;
				p[6] = rtv;
			}
	return(1);
}

/****************************************************************
	 mult.c     Multiplication for Matrix of Polynomial
				   X(y) = A(y) D(y) B(y),
				   where D(y) is a diagonal matrix with each
				   diagonal entry of the form
						   e^{-a_i s}d(y), for which s = 1/y
												 and i = 1..N.
				   Each entry of X(y) will be of the form
					  \sum_{i=1}^N c_i e^{-a_i s} b_i(y), where
					  b_i(0) = 1; therefore, those
					  b_i(y)'s will be each entry's output.
 ****************************************************************/

static void
mult_p(double* p1, double* p2, double* p3, int d1, int d2, int d3)
/*   p3 = p1 * p2   */
{
	int i, j, k;

	for (i = 0; i <= d3; i++)
		p3[i] = 0.0;
	for (i = 0; i <= d1; i++)
		for (j = i, k = 0; k <= d2; j++, k++) {
			if (j > d3)
				break;
			p3[j] += p1[i] * p2[k];
		}
}


static void matrix_p_mult(
	double* A_in[MAX_DIM][MAX_DIM],
	double* D1[MAX_DIM],
	double* B[MAX_DIM][MAX_DIM],
	int dim, int deg, int deg_o,
	Mult_Out  X[MAX_DIM][MAX_DIM])
{
	int i, j, k, l;
	double* p;
	double* T[MAX_DIM][MAX_DIM];
	double t1;

	for (i = 0; i < dim; i++)
		for (j = 0; j < dim; j++) {
			p = T[i][j] = (double*)calloc((size_t)(deg_o + 1), sizeof(double));
			memsaved(p);
			mult_p(B[i][j], D1[i], p, deg, deg_o, deg_o);
		}
	for (i = 0; i < dim; i++)
		for (j = 0; j < dim; j++)
			for (k = 0; k < dim; k++) {
				p = X[i][j].Poly[k] =
					(double*)calloc((size_t)(deg_o + 1), sizeof(double));
				memsaved(p);
				mult_p(A_in[i][k], T[k][j], p, deg, deg_o, deg_o);
				t1 = X[i][j].C_0[k] = p[0];
				if (t1 != 0.0) {
					p[0] = 1.0;
					for (l = 1; l <= deg_o; l++)
						p[l] /= t1;
				}
			}
	for (i = 0; i < dim; i++)
		for (j = 0; j < dim; j++)
			CPLTFREE(T[i][j]);

	/**********
	for (i  = 0; i < dim; i++)
	for (j = 0; j < dim; j++) {
		  for (k = 0; k < dim; k++) {
		  fprintf(outFile, "(%5.3f)", X[i][j].C_0[k]);
		  p = X[i][j].Poly[k];
		  for (l = 0; l <= deg_o; l++)
			 fprintf(outFile, "%5.3f ", p[l]);
		  fprintf(outFile, "\n");
	   }
	   fprintf(outFile, "\n");
	}
	 ***********/
}

/****************************************************************
				  mode  approximation

 ****************************************************************/

 /***
  ***/

static double
approx_mode(double* X, double* b, double length_in)
{
	double w0, w1, w2, w3, w4, w5;
	double a[8];
	double delay, atten;
	double y1, y2, y3, y4, y5, y6;
	int    i, j;

	w0 = X[0];
	w1 = X[1] / w0;  /* a */
	w2 = X[2] / w0;  /* b */
	w3 = X[3] / w0;  /* c */
	w4 = X[4] / w0;  /* d */
	w5 = X[5] / w0;  /* e */

	y1 = 0.5 * w1;
	y2 = w2 - y1 * y1;
	y3 = 3 * w3 - 3.0 * y1 * y2;
	y4 = 12.0 * w4 - 3.0 * y2 * y2 - 4.0 * y1 * y3;
	y5 = 60.0 * w5 - 5.0 * y1 * y4 - 10.0 * y2 * y3;
	y6 = -10.0 * y3 * y3 - 15.0 * y2 * y4 - 6.0 * y1 * y5;

	delay = sqrt(w0) * length_in / Scaling_F;
	atten = exp(-delay * y1);

	a[1] = y2 / 2.0;
	a[2] = y3 / 6.0;
	a[3] = y4 / 24.0;
	a[4] = y5 / 120.0;
	a[5] = y6 / 720.0;

	a[1] *= -delay;
	a[2] *= -delay;
	a[3] *= -delay;
	a[4] *= -delay;
	a[5] *= -delay;

	b[0] = 1.0;
	b[1] = a[1];
	for (i = 2; i <= 5; i++) {
		b[i] = 0.0;
		for (j = 1; j <= i; j++)
			b[i] += j * a[j] * b[i - j];
		b[i] = b[i] / (double)i;
	}

	for (i = 0; i <= 5; i++)
		b[i] *= atten;

	return(delay);
}

/***
 ***/

static double
eval2(double a, double b, double c, double x)
{
	return(a * x * x + b * x + c);
}

/***
 ***/

static int
get_c(double q1, double q2, double q3, double p1, double p2, double a, double b,
	double* cr, double* ci)
{
	double d, n;

	d = (3.0 * (a * a - b * b) + 2.0 * p1 * a + p2) * (3.0 * (a * a - b * b) + 2.0 * p1 * a + p2);
	d += (6.0 * a * b + 2.0 * p1 * b) * (6.0 * a * b + 2.0 * p1 * b);
	n = -(q1 * (a * a - b * b) + q2 * a + q3) * (6.0 * a * b + 2.0 * p1 * b);
	n += (2.0 * q1 * a * b + q2 * b) * (3.0 * (a * a - b * b) + 2.0 * p1 * a + p2);
	*ci = n / d;
	n = (3.0 * (a * a - b * b) + 2.0 * p1 * a + p2) * (q1 * (a * a - b * b) + q2 * a + q3);
	n += (6.0 * a * b + 2.0 * p1 * b) * (2.0 * q1 * a * b + q2 * b);
	*cr = n / d;

	return(1);
}


static int
Pade_apx(double a_b, double* b, double* c1, double* c2, double* c3,
	double* x1, double* x2, double* x3)
	/*
			b[0] + b[1]*y + b[2]*y^2 + ... + b[5]*y^5 + ...
		  = (q3*y^3 + q2*y^2 + q1*y + 1) / (p3*y^3 + p2*y^2 + p1*y + 1)

			where b[0] is always equal to 1.0 and neglected,
			  and y = 1/s.

			(q3*y^3 + q2*y^2 + q1*y + 1) / (p3*y^3 + p2*y^2 + p1*y + 1)
		  = (s^3 + q1*s^2 + q2*s + q3) / (s^3 + p1*s^2 + p2*s + p3)
		  = c1 / (s - x1) + c2 / (s - x2) + c3 / (s - x3) + 1.0
	 */
{
	double p1, p2, p3, q1, q2, q3;

	At[0][0] = 1.0 - a_b;
	At[0][1] = b[1];
	At[0][2] = b[2];
	At[0][3] = -b[3];

	At[1][0] = b[1];
	At[1][1] = b[2];
	At[1][2] = b[3];
	At[1][3] = -b[4];

	At[2][0] = b[2];
	At[2][1] = b[3];
	At[2][2] = b[4];
	At[2][3] = -b[5];

	Gaussian_Elimination(3);

	p3 = At[0][3];
	p2 = At[1][3];
	p1 = At[2][3];
	/*
	if (p3 < 0.0 || p2 < 0.0 || p1 < 0.0 || p1*p2 <= p3)
	   return(0);
	 */
	q1 = p1 + b[1];
	q2 = b[1] * p1 + p2 + b[2];
	q3 = p3 * a_b;

	if (find_roots(p1, p2, p3, x1, x2, x3)) {
		/*
		printf("complex roots : %e %e %e \n", *x1, *x2, *x3);
		 */
		*c1 = eval2(q1 - p1, q2 - p2, q3 - p3, *x1) /
			eval2(3.0, 2.0 * p1, p2, *x1);
		get_c(q1 - p1, q2 - p2, q3 - p3, p1, p2, *x2, *x3, c2, c3);
		return(2);
	}
	else {
		/* new
			printf("roots are %e %e %e \n", *x1, *x2, *x3);
			*/
		*c1 = eval2(q1 - p1, q2 - p2, q3 - p3, *x1) /
			eval2(3.0, 2.0 * p1, p2, *x1);
		*c2 = eval2(q1 - p1, q2 - p2, q3 - p3, *x2) /
			eval2(3.0, 2.0 * p1, p2, *x2);
		*c3 = eval2(q1 - p1, q2 - p2, q3 - p3, *x3) /
			eval2(3.0, 2.0 * p1, p2, *x3);
		return(1);
	}
}

static int
Gaussian_Elimination(int dims)
{
	int i, j, k, dim;
	double f;
	double max;
	int imax;

	dim = dims;

	for (i = 0; i < dim; i++) {
		imax = i;
		max = ABS(At[i][i]);
		for (j = i + 1; j < dim; j++)
			if (ABS(At[j][i]) > max) {
				imax = j;
				max = ABS(At[j][i]);
			}
		if (max < epsi_mult) {
			fprintf(stderr, " can not choose a pivot (mult)\n");
			controlled_exit(EXIT_FAILURE);
		}
		if (imax != i)
			for (k = i; k <= dim; k++) {
				SWAP(double, At[i][k], At[imax][k]);
			}

		f = 1.0 / At[i][i];
		At[i][i] = 1.0;

		for (j = i + 1; j <= dim; j++)
			At[i][j] *= f;

		for (j = 0; j < dim; j++) {
			if (i == j)
				continue;
			f = At[j][i];
			At[j][i] = 0.0;
			for (k = i + 1; k <= dim; k++)
				At[j][k] -= f * At[i][k];
		}
	}
	return(1);
}

static double
root3(double a1, double a2, double a3, double x)
{
	double t1, t2;

	t1 = x * (x * (x + a1) + a2) + a3;
	t2 = x * (2.0 * a1 + 3.0 * x) + a2;

	return(x - t1 / t2);
}

static int
div3(double a1, double a2, double a3, double x, double* p1, double* p2)
{
	NG_IGNORE(a2);

	*p1 = a1 + x;

	/* *p2 = a2 + (a1 + x) * x; */

	*p2 = -a3 / x;

	return(1);
}


static int
find_roots(double a1, double a2, double a3, double* x1, double* x2, double* x3)
{
	double x, t;
	double p, q;

	/***********************************************
	double m,n;
	p = a1*a1/3.0 - a2;
	q = a1*a2/3.0 - a3 - 2.0*a1*a1*a1/27.0;
	p = p*p*p/27.0;
	t = q*q - 4.0*p;
	if (t < 0.0) {
	   if (q != 0.0) {
		  t = atan(sqrt((double)-t)/q);
		  if (t < 0.0)
			 t += M_PI;
		  t /= 3.0;
		  x = 2.0 * pow(p, 0.16666667) * cos(t) - a1 / 3.0;
	   } else {
		  t /= -4.0;
		  x = pow(t, 0.16666667) * 1.732 - a1 / 3.0;
	   }
	} else {
	   t = sqrt(t);
	   m = 0.5*(q - t);
	   n = 0.5*(q + t);
	   if (m < 0.0)
		  m = -pow((double) -m, (double) 0.3333333);
	   else
		  m = pow((double) m, (double) 0.3333333);
	   if (n < 0.0)
		  n = -pow((double) -n, (double) 0.3333333);
	   else
		  n = pow((double) n, (double) 0.3333333);
	   x = m + n - a1 / 3.0;
	}
	 ************************************************/
	q = (a1 * a1 - 3.0 * a2) / 9.0;
	p = (2.0 * a1 * a1 * a1 - 9.0 * a1 * a2 + 27.0 * a3) / 54.0;
	t = q * q * q - p * p;
	if (t >= 0.0) {
		t = acos(p / (q * sqrt(q)));
		x = -2.0 * sqrt(q) * cos(t / 3.0) - a1 / 3.0;
	}
	else {
		if (p > 0.0) {
			t = pow(sqrt(-t) + p, 1.0 / 3.0);
			x = -(t + q / t) - a1 / 3.0;
		}
		else if (p == 0.0) {
			x = -a1 / 3.0;
		}
		else {
			t = pow(sqrt(-t) - p, 1.0 / 3.0);
			x = (t + q / t) - a1 / 3.0;
		}
	}
	/*
	fprintf(stderr, "..1.. %e\n", x*x*x+a1*x*x+a2*x+a3);
	 */
	{
		double x_backup = x;
		int i = 0;
		for (t = root3(a1, a2, a3, x); ABS(t - x) > 5.0e-4;
			t = root3(a1, a2, a3, x))
			if (++i == 32) {
				x = x_backup;
				break;
			}
			else
				x = t;
	}
	/*
	fprintf(stderr, "..2.. %e\n", x*x*x+a1*x*x+a2*x+a3);
	 */


	*x1 = x;
	div3(a1, a2, a3, x, &a1, &a2);

	t = a1 * a1 - 4.0 * a2;
	if (t < 0) {
		/*
		fprintf(stderr, "***** Two Imaginary Roots.\n Update.\n");
		*x2 = -0.5 * a1;
		*x3 = a2 / *x2;
		 */
		*x3 = 0.5 * sqrt(-t);
		*x2 = -0.5 * a1;
		return(1);
	}
	else {
		t = sqrt(t);
		if (a1 >= 0.0)
			*x2 = t = -0.5 * (a1 + t);
		else
			*x2 = t = -0.5 * (a1 - t);
		*x3 = a2 / t;
		return(0);
	}
}


static NDnamePt
insert_ND(char* name, NDnamePt* ndn)
{
	int       cmp;
	NDnamePt  p;

	if (*ndn == NULL) {
		p = *ndn = TMALLOC(NDname, 1);
		memsaved(p);
		p->nd = NULL;
		p->right = p->left = NULL;
		strcpy(p->id, name);
		return(p);
	}
	cmp = strcmp((*ndn)->id, name);
	if (cmp == 0)
		return(*ndn);
	else {
		if (cmp < 0)
			return(insert_ND(name, &((*ndn)->left)));
		else
			return(insert_ND(name, &((*ndn)->right)));
	}
}

static NODE*
insert_node(char* name)
{
	NDnamePt n;
	NODE* p;

	n = insert_ND(name, &ndn_btree);
	if (n->nd == NULL) {
		p = NEW_node();
		p->name = n;
		n->nd = p;
		p->next = node_tab;
		node_tab = p;
		return(p);
	}
	else
		return(n->nd);
}
/***
static int divC(double ar, double ai, double br, double bi, double *cr, double *ci)
{
		double t;

		t = br*br + bi*bi;
		*cr = (ar*br + ai*bi) / t;
		*ci = (ai*br - ar*bi) / t;

		return(1);
}
***/

static NODE
* NEW_node(void)
{
	NODE* n;

	n = TMALLOC(NODE, 1);
	memsaved(n);
	n->mptr = NULL;
	n->gptr = NULL;
	n->cptr = NULL;
	n->rptr = NULL;
	n->tptr = NULL;
	n->cplptr = NULL;
	n->rlptr = NULL;
	n->ddptr = NULL;
	n->cvccsptr = NULL;
	n->vccsptr = NULL;
	n->CL = 0.001;
	n->V = n->dv = 0.0;
	n->gsum = n->cgsum = 0;
	n->is = 0;
	n->tag = 0;
	n->flag = 0;
	n->region = NULL;
	n->ofile = NULL;
	n->dvtag = 0;

	return(n);
}



/****************************************************************
	 diag.c      This file contains the main().
 ****************************************************************/

#define  epsi2    1.0e-8

static  int         dim;
static  MAXE_PTR    row;

static MAXE_PTR
sort(MAXE_PTR list, double val, int r, int c, MAXE_PTR e)
{
	if (list == NULL || list->value < val) {
		e->row = r;
		e->col = c;
		e->value = val;
		e->next = list;
		return(e);
	}
	else {
		list->next = sort(list->next, val, r, c, e);
		return(list);
	}
}


static void
ordering(void)
{
	MAXE_PTR e;
	int i, j, m;
	double mv;

	for (i = 0; i < dim - 1; i++) {
		m = i + 1;
		mv = ABS(ZY[i][m]);
		for (j = m + 1; j < dim; j++)
			if ((int)(ABS(ZY[i][j]) * 1e7) > (int) (1e7 * mv)) {

				mv = ABS(ZY[i][j]);
				m = j;
			}
		e = TMALLOC(MAXE, 1);
		memsaved(e);
		row = sort(row, mv, i, m, e);
	}
}


static MAXE_PTR
delete_1(MAXE_PTR* list, int rc)
{
	MAXE_PTR list1, e;

	list1 = *list;
	if ((*list)->row == rc) {
		*list = (*list)->next;
		return(list1);
	}
	for (e = list1->next; e->row != rc; e = e->next)
		list1 = e;
	list1->next = e->next;
	return(e);
}


static void
reordering(int p, int q)
{
	MAXE_PTR e;
	int j, m;
	double mv;

	m = p + 1;
	mv = ABS(ZY[p][m]);
	for (j = m + 1; j < dim; j++)
		if ((int)(ABS(ZY[p][j]) * 1e7) > (int) (1e7 * mv)) {
			mv = ABS(ZY[p][j]);
			m = j;
		}
	e = delete_1(&row, p);
	row = sort(row, mv, p, m, e);

	m = q + 1;
	if (m != dim) {
		mv = ABS(ZY[q][m]);
		for (j = m + 1; j < dim; j++)
			if ((int)(ABS(ZY[q][j]) * 1e7) > (int) (1e7 * mv)) {

				mv = ABS(ZY[q][j]);
				m = j;
			}
		e = delete_1(&row, q);
		row = sort(row, mv, q, m, e);
	}

}

static void
diag(int dims)
{
	int i, j, c;
	double fmin, fmax;

	dim = dims;
	row = NULL;

	fmin = fmax = ABS(ZY[0][0]);
	for (i = 0; i < dim; i++)
		for (j = i; j < dim; j++)
			if (ABS(ZY[i][j]) > fmax)
				fmax = ABS(ZY[i][j]);
			else if (ABS(ZY[i][j]) < fmin)
				fmin = ABS(ZY[i][j]);
	fmin = 2.0 / (fmin + fmax);
	for (i = 0; i < dim; i++)
		for (j = i; j < dim; j++)
			ZY[i][j] *= fmin;

	for (i = 0; i < dim; i++) {
		for (j = 0; j < dim; j++)
			if (i == j)
				Sv[i][i] = 1.0;
			else
				Sv[i][j] = 0.0;
	}

	ordering();

	if (row)
		for (c = 0; row->value > epsi2; c++) {
			int p, q;

			p = row->row;
			q = row->col;

			rotate(dim, p, q);
			reordering(p, q);
		}

	for (i = 0; i < dim; i++)
		D[i] = ZY[i][i] / fmin;

	while (row) {
		MAXE_PTR tmp_row = row->next;
		CPLTFREE(row);
		row = tmp_row;
	}
}

/****************************************************************
	 rotate()      rotation of the Jacobi's method
 ****************************************************************/

static int rotate(int dim_in, int p, int q)
{
	int j;
	double co, si;
	double ve, mu, ld;
	double T[MAX_DIM];
	double t;

	ld = -ZY[p][q];
	mu = 0.5 * (ZY[p][p] - ZY[q][q]);
	ve = sqrt(ld * ld + mu * mu);
	co = sqrt((ve + ABS(mu)) / (2.0 * ve));
	si = SGN(mu) * ld / (2.0 * ve * co);

	for (j = p + 1; j < dim_in; j++)
		T[j] = ZY[p][j];
	for (j = 0; j < p; j++)
		T[j] = ZY[j][p];

	for (j = p + 1; j < dim_in; j++) {
		if (j == q)
			continue;
		if (j > q)
			ZY[p][j] = T[j] * co - ZY[q][j] * si;
		else
			ZY[p][j] = T[j] * co - ZY[j][q] * si;
	}
	for (j = q + 1; j < dim_in; j++) {
		if (j == p)
			continue;
		ZY[q][j] = T[j] * si + ZY[q][j] * co;
	}
	for (j = 0; j < p; j++) {
		if (j == q)
			continue;
		ZY[j][p] = T[j] * co - ZY[j][q] * si;
	}
	for (j = 0; j < q; j++) {
		if (j == p)
			continue;
		ZY[j][q] = T[j] * si + ZY[j][q] * co;
	}

	t = ZY[p][p];
	ZY[p][p] = t * co * co + ZY[q][q] * si * si - 2.0 * ZY[p][q] * si * co;
	ZY[q][q] = t * si * si + ZY[q][q] * co * co + 2.0 * ZY[p][q] * si * co;

	ZY[p][q] = 0.0;

	{
		double R[MAX_DIM];

		for (j = 0; j < dim_in; j++) {
			T[j] = Sv[j][p];
			R[j] = Sv[j][q];
		}

		for (j = 0; j < dim_in; j++) {
			Sv[j][p] = T[j] * co - R[j] * si;
			Sv[j][q] = T[j] * si + R[j] * co;
		}
	}

	return(1);

}

