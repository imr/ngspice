/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/ifsim.h"
#include "ngspice/inpmacs.h"
#include "ngspice/fteext.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/inpptree.h"
#include "inpxx.h"
#include "ngspice/cktdefs.h"

/* Uncomment to allow tracing */
/*#define TRACE*/


extern double PTfudge_factor;

static int PTeval(INPparseNode * tree, double gmin, double *res,
		  double *vals);



int
IFeval(IFparseTree * tree, double gmin, double *result, double *vals,
       double *derivs)
{
    int i, err;
    INPparseTree *myTree = (INPparseTree *) tree;

    if (!myTree) {
        fprintf(stderr, "\nInternal error: No tree to evaluate.\n");
        controlled_exit(EXIT_BAD);
    }

#ifdef TRACE
    INPptPrint("calling PTeval, tree = ", tree);
    printf("values:");
    for (i = 0; i < myTree->p.numVars; i++)
	printf("\tvar%d = %lg\n", i, vals[i]);
#endif

    if ((err = PTeval(myTree->tree, gmin, result, vals)) != OK) {
        if (ft_ngdebug) {
            INPptPrint("calling PTeval, tree = ", tree);
            printf("values:");
            for (i = 0; i < myTree->p.numVars; i++)
	        printf("\tvar%d = %lg\n", i, vals[i]);
        }
        if (ft_stricterror)
            controlled_exit(EXIT_BAD);
        return err;
    }

    for (i = 0; i < myTree->p.numVars; i++)
        if ((err = PTeval(myTree->derivs[i], gmin, &derivs[i], vals)) != OK) {
            if (ft_ngdebug) {
                INPptPrint("calling PTeval, tree = ", tree);
                printf("results: function = %lg\n", *result);
                for (i = 0; i < myTree->p.numVars; i++)
	                printf("\td / d var%d = %lg\n", i, derivs[i]);
            }
            if (ft_stricterror)
                controlled_exit(EXIT_BAD);
            return err;
        }

#ifdef TRACE
    printf("results: function = %lg\n", *result);
    for (i = 0; i < myTree->p.numVars; i++)
	printf("\td / d var%d = %lg\n", i, derivs[i]);
#endif

    return (OK);
}

static int
PTeval(INPparseNode * tree, double gmin, double *res, double *vals)
{
    double r1, r2;
    int err;

    PTfudge_factor = gmin * 1.0e-20; /* defaults to 1e-32, should be small enough */
    switch (tree->type) {
    case PT_CONSTANT:
	*res = tree->constant;
	break;

    case PT_VAR:
	*res = vals[tree->valueIndex];
	break;

    case PT_FUNCTION:
        switch(tree->funcnum) {
       
        case PTF_POW:
        case PTF_PWR:
        case PTF_MIN:
        case PTF_MAX:
            err = PTeval(tree->left->left, gmin, &r1, vals);
            if (err != OK)
                return (err);
            err = PTeval(tree->left->right, gmin, &r2, vals);
            if (err != OK)
                return (err);
            *res = PTbinary(tree -> function) (r1, r2);
            if (*res == HUGE) {
                fprintf(stderr, "Error: %g, %g out of range for %s\n",
                r1, r2, tree->funcname);
                return (E_PARMVAL);
            }
        break;
        /* fcns with single argument */
        default:
            err = PTeval(tree->left, gmin, &r1, vals);
            if (err != OK)
                return (err);
            if(tree->data == NULL)
                *res = PTunary(tree -> function) (r1);
            else
                *res = PTunary_with_private(tree -> function) (r1, tree->data);
            if (*res == HUGE) {
                fprintf(stderr, "Error: %g out of range for %s\n",
                r1, tree->funcname);
                return (E_PARMVAL);
            }
        break;
        }
	break;

    case PT_TERN:
      {
        INPparseNode *arg1 = tree->left;
        INPparseNode *arg2 = tree->right->left;
        INPparseNode *arg3 = tree->right->right;

        err = PTeval(arg1, gmin, &r1, vals);
        if (err != OK)
          return (err);

        /*FIXME > 0.0, >= 0.5, != 0.0 or what ? */
        err = PTeval((r1 != 0.0) ? arg2 : arg3, gmin, &r2, vals);
        if (err != OK)
           return (err);

        *res = r2;
        break;
      }

    case PT_PLUS:
    case PT_MINUS:
    case PT_TIMES:
    case PT_DIVIDE:
    case PT_POWER:
	err = PTeval(tree->left, gmin, &r1, vals);
	if (err != OK)
	    return (err);
	err = PTeval(tree->right, gmin, &r2, vals);
	if (err != OK)
	    return (err);
	*res = PTbinary(tree -> function) (r1, r2);
	if (*res == HUGE) {
	    fprintf(stderr, "\nError: %g, %g out of range for %s\n",
		    r1, r2, tree->funcname);
	    return (E_PARMVAL);
	}
	break;

    case PT_TIME:
        *res = ((CKTcircuit*) tree->data) -> CKTtime;
	break;

    case PT_TEMPERATURE:
        *res = ((CKTcircuit*) tree->data) -> CKTtemp - CONSTCtoK;
	break;
	
    case PT_FREQUENCY:
        *res = (((CKTcircuit*) tree->data) -> CKTomega)/2./M_PI;
	break;
	
    default:
	fprintf(stderr, "Internal Error: bad node type %d\n", tree->type);
	return (E_PANIC);
    }

    return (OK);
}
