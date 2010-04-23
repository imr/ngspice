/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Wayne A. Christopher, U. C. Berkeley CAD Group 
**********/
/*#define TRACE*/

#include "ngspice.h"
#include "ifsim.h"
#include "iferrmsg.h"
#include "inpdefs.h"
#include "inpptree.h"
#include "inp.h"

static INPparseNode *mkcon(double value);
static INPparseNode *mkb(int type, INPparseNode * left,
			 INPparseNode * right);
static INPparseNode *mkf(int type, INPparseNode * arg);
static int PTcheck(INPparseNode * p);
static INPparseNode *mkbnode(const char *opstr, INPparseNode * arg1,
			     INPparseNode * arg2);
static INPparseNode *mkfnode(const char *fname, INPparseNode * arg);
static INPparseNode *mknnode(double number);
static INPparseNode *mksnode(const char *string, void *ckt);
static INPparseNode *PTdifferentiate(INPparseNode * p, int varnum);

#include "inpptree-parser.c"

static IFvalue *values = NULL;
static int *types;
static int numvalues;
static void *circuit;
static INPtables *tables;

#if defined (_MSC_VER)
# define __func__ __FUNCTION__ /* __func__ is C99, but MSC can't */
#endif

extern IFsimulator *ft_sim;	/* XXX */

/* Some tables that the parser uses. */

static struct op {
    int number;
    char *name;
    double (*funcptr) ();
} ops[] = {
    {
    PT_COMMA, ",", NULL}, {
    PT_PLUS, "+", PTplus}, {
    PT_MINUS, "-", PTminus}, {
    PT_TIMES, "*", PTtimes}, {
    PT_DIVIDE, "/", PTdivide}, {
    PT_POWER, "^", PTpower}
};

#define NUM_OPS (int)(sizeof (ops) / sizeof (struct op))

static struct func {
    char *name;
    int number;
    double (*funcptr) ();
} funcs[] = {
    { "abs",    PTF_ABS,    PTabs } ,
    { "acos",   PTF_ACOS,   PTacos } ,
    { "acosh",  PTF_ACOSH,  PTacosh } ,
    { "asin",   PTF_ASIN,   PTasin } ,
    { "asinh",  PTF_ASINH,  PTasinh } ,
    { "atan",   PTF_ATAN,   PTatan } ,
    { "atanh",  PTF_ATANH,  PTatanh } ,
    { "cos",    PTF_COS,    PTcos } ,
    { "cosh",   PTF_COSH,   PTcosh } ,
    { "exp",    PTF_EXP,    PTexp } ,
    { "ln",     PTF_LN,     PTln } ,
    { "log",    PTF_LOG,    PTlog } ,
    { "sgn",    PTF_SGN,    PTsgn } ,
    { "sin",    PTF_SIN,    PTsin } ,
    { "sinh",   PTF_SINH,   PTsinh } ,
    { "sqrt",   PTF_SQRT,   PTsqrt } ,
    { "tan",    PTF_TAN,    PTtan } ,
    { "tanh",   PTF_TANH,   PTtanh } ,
    { "u",   	PTF_USTEP,  PTustep } ,
    { "uramp",  PTF_URAMP,  PTuramp } ,
    { "-",      PTF_UMINUS, PTuminus },
    /* MW. cif function added */
    { "u2",	PTF_USTEP2, PTustep2},
    { "pwl",	PTF_PWL,    PTpwl},
    { "pwl_derivative",	PTF_PWL_DERIVATIVE, PTpwl_derivative},
    { "eq0",    PTF_EQ0,    PTeq0},
    { "ne0",    PTF_NE0,    PTne0},
    { "gt0",    PTF_GT0,    PTgt0},
    { "lt0",    PTF_LT0,    PTlt0},
    { "ge0",    PTF_GE0,    PTge0},
    { "le0",    PTF_LE0,    PTle0},
    { "pow",    PTF_POW,    PTpower},  
    { "min",    PTF_MIN,    PTmin},
    { "max",    PTF_MAX,    PTmax},
} ;

#define NUM_FUNCS (int)(sizeof (funcs) / sizeof (struct func))

/* These are all the constants any sane person needs. */

static struct constant {
    char *name;
    double value;
} constants[] = {
    {
    "e", M_E}, {
    "pi", M_PI}
};

#define NUM_CONSTANTS (int)(sizeof (constants) / sizeof (struct constant))

/* Parse the expression in *line as far as possible, and return the parse
 * tree obtained.  If there is an error, *pt will be set to NULL and an error
 * message will be printed.
 */

void
INPgetTree(char **line, INPparseTree ** pt, void *ckt, INPtables * tab)
{
    INPparseNode *p;
    int i, rv;

    values = NULL;
    types = NULL;
    numvalues = 0;

    circuit = ckt;
    tables = tab;

#ifdef TRACE
    fprintf(stderr,"%s, line = \"%s\"\n", __func__, *line);
#endif

    rv = PTparse(line, &p, ckt);

    if (rv || !PTcheck(p)) {
	*pt = NULL;
	return;
    }

    (*pt) = (INPparseTree *) MALLOC(sizeof(INPparseTree));

    (*pt)->p.numVars = numvalues;
    (*pt)->p.varTypes = types;
    (*pt)->p.vars = values;
    (*pt)->p.IFeval = IFeval;
    (*pt)->tree = p;

    (*pt)->derivs = (INPparseNode **)
	MALLOC(numvalues * sizeof(INPparseNode *));

    for (i = 0; i < numvalues; i++)
	(*pt)->derivs[i] = PTdifferentiate(p, i);

    return;
}

/* This routine takes the partial derivative of the parse tree with respect to
 * the i'th variable.  We try to do optimizations like getting rid of 0-valued
 * terms.
 *
 *** Note that in the interests of simplicity we share some subtrees between
 *** the function and its derivatives.  This means that you can't free the
 *** trees.
 */

static INPparseNode *PTdifferentiate(INPparseNode * p, int varnum)
{
    INPparseNode *arg1 = NULL, *arg2, *newp;

    switch (p->type) {
    case PT_TIME:
    case PT_TEMPERATURE:
    case PT_FREQUENCY:
    case PT_CONSTANT:
	newp = mkcon((double) 0);
	break;

    case PT_VAR:
	/* Is this the variable we're differentiating wrt? */
	if (p->valueIndex == varnum)
	    newp = mkcon((double) 1);
	else
	    newp = mkcon((double) 0);
	break;

    case PT_PLUS:
    case PT_MINUS:
	arg1 = PTdifferentiate(p->left, varnum);
	arg2 = PTdifferentiate(p->right, varnum);
	newp = mkb(p->type, arg1, arg2);
	break;

    case PT_TIMES:
	/* d(a * b) = d(a) * b + d(b) * a */
	arg1 = PTdifferentiate(p->left, varnum);
	arg2 = PTdifferentiate(p->right, varnum);

	newp = mkb(PT_PLUS, mkb(PT_TIMES, arg1, p->right),
		   mkb(PT_TIMES, p->left, arg2));
	break;

    case PT_DIVIDE:
	/* d(a / b) = (d(a) * b - d(b) * a) / b^2 */
	arg1 = PTdifferentiate(p->left, varnum);
	arg2 = PTdifferentiate(p->right, varnum);

	newp = mkb(PT_DIVIDE, mkb(PT_MINUS, mkb(PT_TIMES, arg1,
						p->right), mkb(PT_TIMES,
							       p->left,
							       arg2)),
		   mkb(PT_POWER, p->right, mkcon((double) 2)));
	break;

    case PT_POWER:
	/* Two cases... If the power is a constant then we're cool.
	 * Otherwise we have to be tricky.
	 */
	if (p->right->type == PT_CONSTANT) {
	    arg1 = PTdifferentiate(p->left, varnum);

	    newp = mkb(PT_TIMES, mkb(PT_TIMES,
				     mkcon(p->right->constant),
				     mkb(PT_POWER, p->left,
					 mkcon(p->right->constant - 1))),
		       arg1);
	} else {
	    /* This is complicated.  f(x) ^ g(x) ->
	     * exp(y(x) * ln(f(x)) ...
	     */
	    arg1 = PTdifferentiate(p->left, varnum);
	    arg2 = PTdifferentiate(p->right, varnum);
	    newp = mkb(PT_TIMES, mkf(PTF_EXP, mkb(PT_TIMES,
						  p->right, mkf(PTF_LN,
								p->left))),
		       mkb(PT_PLUS,
			   mkb(PT_TIMES, p->right,
			       mkb(PT_DIVIDE, arg1, p->left)),
			   mkb(PT_TIMES, arg2, mkf(PTF_LN, /*arg1*/p->left))));
		                                        /*changed by HT, '05/06/29*/

	}
	break;
 
    case PT_TERN: /* ternary_fcn(cond,exp1,exp2) */
      // naive:
      //   d/d ternary_fcn(cond,exp1,exp2) --> ternary_fcn(cond, d/d exp1, d/d exp2)
      {
	INPparseNode *arg1 = p->left;
	INPparseNode *arg2 = p->right->left;
	INPparseNode *arg3 = p->right->right;

//	extern void printTree(INPparseNode *);
//
//	printf("debug: %s, PT_TERN: ", __func__);
//	printTree(p);
//	printf("\n");

	newp = mkb(PT_TERN, arg1, mkb(PT_COMMA,
				      PTdifferentiate(arg2, varnum),
				      PTdifferentiate(arg3, varnum)));

//	printf("debug, %s, returns; ", __func__);
//	printTree(newp);
//	printf("\n");

	return (newp);
      }

    case PT_FUNCTION:
	/* Many cases.  Set arg1 to the derivative of the function,
	 * and arg2 to the derivative of the argument.
	 */
	switch (p->funcnum) {
	case PTF_ABS:		/* sgn(u) */
	    /* arg1 = mkf(PTF_SGN, p->left, 0); */
	    arg1 = mkf(PTF_SGN, p->left);
	    break;

	case PTF_SGN:
	    arg1 = mkcon((double) 0.0);
	    break;

	case PTF_ACOS:		/* - 1 / sqrt(1 - u^2) */
	    arg1 = mkb(PT_DIVIDE, mkcon((double) -1), mkf(PTF_SQRT,
							  mkb(PT_MINUS,
							      mkcon(
								    (double)
								    1),
							      mkb(PT_POWER,
								  p->left,
								  mkcon(
									(double)
									2)))));
	    break;

	case PTF_ACOSH:	/* 1 / sqrt(u^2 - 1) */
	    arg1 = mkb(PT_DIVIDE, mkcon((double) 1), mkf(PTF_SQRT,
							 mkb(PT_MINUS,
							     mkb(PT_POWER,
								 p->left,
								 mkcon(
								       (double)
								       2)),
							     mkcon((double)
								   1))));

	    break;

	case PTF_ASIN:		/* 1 / sqrt(1 - u^2) */
	    arg1 = mkb(PT_DIVIDE, mkcon((double) 1), mkf(PTF_SQRT,
							 mkb(PT_MINUS,
							     mkcon((double)
								   1),
							     mkb(PT_POWER,
								 p->left,
								 mkcon(
								       (double)
								       2)))));
	    break;

	case PTF_ASINH:	/* 1 / sqrt(u^2 + 1) */
	    arg1 = mkb(PT_DIVIDE, mkcon((double) 1), mkf(PTF_SQRT,
							 mkb(PT_PLUS,
							     mkb(PT_POWER,
								 p->left,
								 mkcon(
								       (double)
								       2)),
							     mkcon((double)
								   1))));
	    break;

	case PTF_ATAN:		/* 1 / (1 + u^2) */
	    arg1 = mkb(PT_DIVIDE, mkcon((double) 1), mkb(PT_PLUS,
							 mkb(PT_POWER,
							     p->left,
							     mkcon((double)
								   2)),
							 mkcon((double)
							       1)));
	    break;

	case PTF_ATANH:	/* 1 / (1 - u^2) */
	    arg1 = mkb(PT_DIVIDE, mkcon((double) 1), mkb(PT_MINUS,
							 mkcon((double) 1),
							 mkb(PT_POWER,
							     p->left,
							     mkcon((double)
								   2))));
	    break;

	case PTF_COS:		/* - sin(u) */
	    arg1 = mkf(PTF_UMINUS, mkf(PTF_SIN, p->left));
	    break;

	case PTF_COSH:		/* sinh(u) */
	    arg1 = mkf(PTF_SINH, p->left);
	    break;

	case PTF_EXP:		/* exp(u) */
	    /* arg1 = mkf(PTF_EXP, p->left, 0); */
	    arg1 = mkf(PTF_EXP, p->left);
	    break;

	case PTF_LN:		/* 1 / u */
	    arg1 = mkb(PT_DIVIDE, mkcon((double) 1), p->left);
	    break;

	case PTF_LOG:		/* log(e) / u */
	    arg1 = mkb(PT_DIVIDE, mkcon((double) M_LOG10E), p->left);
	    break;

	case PTF_SIN:		/* cos(u) */
	    arg1 = mkf(PTF_COS, p->left);
	    break;

	case PTF_SINH:		/* cosh(u) */
	    arg1 = mkf(PTF_COSH, p->left);
	    break;

	case PTF_SQRT:		/* 1 / (2 * sqrt(u)) */
	    arg1 = mkb(PT_DIVIDE, mkcon((double) 1), mkb(PT_TIMES,
							 mkcon((double) 2),
							 mkf(PTF_SQRT,
							     p->left)));
	    break;

	case PTF_TAN:		/* 1 / (cos(u) ^ 2) */
	    arg1 = mkb(PT_DIVIDE, mkcon((double) 1), mkb(PT_POWER,
							 mkf(PTF_COS,
							     p->left),
							 mkcon((double)
							       2)));
	    break;

	case PTF_TANH:		/* 1 / (cosh(u) ^ 2) */
	    arg1 = mkb(PT_DIVIDE, mkcon((double) 1), mkb(PT_POWER,
							 mkf(PTF_COSH,
							     p->left),
							 mkcon((double)
							       2)));
	    break;

	case PTF_USTEP:
	case PTF_EQ0:
	case PTF_NE0:
	case PTF_GT0:
	case PTF_LT0:
	case PTF_GE0:
	case PTF_LE0:
	    arg1 = mkcon((double) 0.0);
	    break;

	case PTF_URAMP:
	    arg1 = mkf(PTF_USTEP, p->left);
	    break;

	    /* MW. PTF_CIF for new cif function */
	case PTF_USTEP2: /* ustep2=uramp(x)-uramp(x-1) ustep2'=ustep(x)-ustep(x-1) */
            arg1 = mkb(PT_MINUS,
                       mkf(PTF_USTEP, p->left),
                       mkf(PTF_USTEP,
                           mkb(PT_MINUS,
                               p->left,
                               mkcon((double) 1.0))));
	    break;
	    
        case PTF_UMINUS:    /* - 1 ; like a constant (was 0 !) */
            arg1 = mkcon((double) - 1.0);
            break;

        case PTF_PWL: /* PWL(var, x1, y1, x2, y2, ... a const list) */
            arg1 = mkf(PTF_PWL_DERIVATIVE, p->left);
            arg1->data = p->data;
            break;

        case PTF_PWL_DERIVATIVE: /* d/dvar PWL(var, ...) */
            arg1 = mkcon((double) 0.0);
            break;

        case PTF_MIN:
        /*
        min(a,b)
        p->left: ','    p->left->left: a       p->left->right: b 
        */	        
            newp = mkcon((double) 0);
	        return (newp);

        case PTF_MAX:
            newp = mkcon((double) 0);
	        return (newp);

        case PTF_POW:
        {
        /*
        pow(a,b)
        p->left: ','    p->left->left: a       p->left->right: b 
        */

        /* Two cases...
           The power is constant 
        */
            if (p->left->right->type == PT_CONSTANT) {
                arg1 = PTdifferentiate(p->left->left, varnum);

                newp = mkb(PT_TIMES, mkb(PT_TIMES,
				     mkcon(p->left->right->constant),
				     mkb(PT_POWER, p->left->left,
					 mkcon(p->left->right->constant - 1))),
		             arg1);
            } else {
            /* This is complicated.  f(x) ^ g(x) ->
               exp(y(x) * ln(f(x)) ...
             */
             arg1 = PTdifferentiate(p->left->left, varnum);
             arg2 = PTdifferentiate(p->left->right, varnum);
             newp = mkb(PT_TIMES, mkf(PTF_EXP, mkb(PT_TIMES,
						p->left->right, mkf(PTF_LN,
						p->left->left))),
		                mkb(PT_PLUS,
			            mkb(PT_TIMES, p->left->right,
			            mkb(PT_DIVIDE, arg1, p->left->left)),
			            mkb(PT_TIMES, arg2, mkf(PTF_LN, /*arg1*/p->left->left))));
		                                        /*changed by HT, '05/06/29*/

            }
            arg2 = PTdifferentiate(p->left->left, varnum);
            newp = mkb(PT_TIMES, arg1, arg2);
            return (newp);
        }

	default:
	    fprintf(stderr, "Internal Error: bad function # %d\n",
		    p->funcnum);
	    newp = NULL;
	    break;
	}

	arg2 = PTdifferentiate(p->left, varnum);

	newp = mkb(PT_TIMES, arg1, arg2);

	break;

    default:
	fprintf(stderr, "Internal error: bad node type %d\n", p->type);
	newp = NULL;
	break;
    }

    return (newp);
}

static INPparseNode *mkcon(double value)
{
    INPparseNode *p = (INPparseNode *) MALLOC(sizeof(INPparseNode));

    p->type = PT_CONSTANT;
    p->constant = value;

    return (p);
}

static INPparseNode *mkb(int type, INPparseNode * left,
			 INPparseNode * right)
{
    INPparseNode *p = (INPparseNode *) MALLOC(sizeof(INPparseNode));
    int i;

    if ((right->type == PT_CONSTANT) && (left->type == PT_CONSTANT)) {
	switch (type) {
	case PT_TIMES:
	    return (mkcon(left->constant * right->constant));

	case PT_DIVIDE:
	    return (mkcon(left->constant / right->constant));

	case PT_PLUS:
	    return (mkcon(left->constant + right->constant));

	case PT_MINUS:
	    return (mkcon(left->constant - right->constant));

	case PT_POWER:
	    return (mkcon(pow(left->constant, right->constant)));
	}
    }
    switch (type) {
    case PT_TIMES:
	if ((left->type == PT_CONSTANT) && (left->constant == 0))
	    return (left);
	else if ((right->type == PT_CONSTANT) && (right->constant == 0))
	    return (right);
	else if ((left->type == PT_CONSTANT) && (left->constant == 1))
	    return (right);
	else if ((right->type == PT_CONSTANT) && (right->constant == 1))
	    return (left);
	break;

    case PT_DIVIDE:
	if ((left->type == PT_CONSTANT) && (left->constant == 0))
	    return (left);
	else if ((right->type == PT_CONSTANT) && (right->constant == 1))
	    return (left);
	break;

    case PT_PLUS:
	if ((left->type == PT_CONSTANT) && (left->constant == 0))
	    return (right);
	else if ((right->type == PT_CONSTANT) && (right->constant == 0))
	    return (left);
	break;

    case PT_MINUS:
	if ((right->type == PT_CONSTANT) && (right->constant == 0))
	    return (left);
	else if ((left->type == PT_CONSTANT) && (left->constant == 0))
	    return (mkf(PTF_UMINUS, right));
	break;

    case PT_POWER:
	if (right->type == PT_CONSTANT) {
	    if (right->constant == 0)
		return (mkcon(1.0));
	    else if (right->constant == 1)
		return (left);
	}
	break;

    case PT_TERN:
	if (left->type == PT_CONSTANT)
	    /*FIXME > 0.0, >= 0.5, != 0.0 or what ? */
	    return ((left->constant != 0.0) ? right->left : right->right);
	if((right->left->type == PT_CONSTANT) &&
	   (right->right->type == PT_CONSTANT) &&
	   (right->left->constant == right->right->constant))
	    return (right->left);
	break;
     }
 
     p->type = type;
     p->left = left;
     p->right = right;
 
    if(type == PT_TERN) {
	p->function = NULL;
	p->funcname = NULL;
	return (p);
    }


    for (i = 0; i < NUM_OPS; i++)
	if (ops[i].number == type)
	    break;
    if (i == NUM_OPS) {
	fprintf(stderr, "Internal Error: bad type %d\n", type);
	return (NULL);
    }
    p->function = ops[i].funcptr;
    p->funcname = ops[i].name;

    return (p);
}

static INPparseNode *mkf(int type, INPparseNode * arg)
{
    INPparseNode *p = (INPparseNode *) MALLOC(sizeof(INPparseNode));
    int i;
    double constval;

    for (i = 0; i < NUM_FUNCS; i++)
	if (funcs[i].number == type)
	    break;
    if (i == NUM_FUNCS) {
	fprintf(stderr, "Internal Error: bad type %d\n", type);
	return (NULL);
    }

    if (arg->type == PT_CONSTANT) {
	constval = ((*funcs[i].funcptr) (arg->constant));
	return (mkcon(constval));
    }

    p->type = PT_FUNCTION;
    p->left = arg;

    p->funcnum = i;
    p->function = funcs[i].funcptr;
    p->funcname = funcs[i].name;

    p->data = NULL;

    return (p);
}

/* Check for remaining PT_PLACEHOLDERs in the parse tree.  Returns 1 if ok. */

static int PTcheck(INPparseNode * p)
{
    switch (p->type) {
    case PT_PLACEHOLDER:
	return (0);

    case PT_TIME:
    case PT_TEMPERATURE:
    case PT_FREQUENCY:
    case PT_CONSTANT:
    case PT_VAR:
	return (1);

    case PT_FUNCTION:
	return (PTcheck(p->left));

    case PT_PLUS:
    case PT_MINUS:
    case PT_TIMES:
    case PT_DIVIDE:
    case PT_POWER:
    case PT_COMMA:
	return (PTcheck(p->left) && PTcheck(p->right));
    case PT_TERN:
	return (PTcheck(p->left) && PTcheck(p->right->left) && PTcheck(p->right->right));

    default:
	fprintf(stderr, "Internal error: bad node type %d\n", p->type);
	return (0);
    }
}

/* Binop node. */

static INPparseNode *mkbnode(const char *opstr, INPparseNode * arg1,
			     INPparseNode * arg2)
{
    INPparseNode *p;
    int i;

    for (i = 0; i < NUM_OPS; i++)
	if (!strcmp(ops[i].name, opstr))
	    break;

    if (i == NUM_OPS) {
	fprintf(stderr, "Internal Error: no such op num %s\n", opstr);
	return (NULL);
    }
    p = (INPparseNode *) MALLOC(sizeof(INPparseNode));

    p->type = ops[i].number;
    p->funcname = ops[i].name;
    p->function = ops[i].funcptr;
    p->left = arg1;
    p->right = arg2;

    return (p);
}

/*
 * prepare_PTF_PWL()
 *   for the PWL(expr, points...) function
 *     collect the given points, which are expected to be given
 *       literal constant
 *   strip them from the INPparseNode
 *     and pass them as an opaque struct alongside the
 *     INPparseNode for the PWL(expr) function call
 *
 * Note:
 *  the functionINPgetTree() is missing a recursive decending simplifier
 *  as a consequence we can meet a PTF_UMINUS->PTF_CONSTANT
 *    instead of a plain PTF_CONSTANT here
 *  of course this can get arbitrarily more complex
 *    for example PTF_TIMES -> PTF_CONSTANT, PTF_CONSTANT
 *      etc.
 *  currently we support only PFT_CONST and PTF_UMINUS->PTF_CONST
 */

#define Breakpoint do { __asm__ __volatile__ ("int $03"); } while(0)

static INPparseNode *prepare_PTF_PWL(INPparseNode *p)
{
    INPparseNode *w;
    struct pwldata { int n; double *vals; } *data;
    int i;

    if (p->funcnum != PTF_PWL) {
        fprintf(stderr, "PWL-INFO: %s, very unexpected\n", __func__);
        exit(1);
    }

#ifdef TRACE
    fprintf(stderr, "PWL-INFO: %s  building a PTF_PWL\n", __func__);
#endif
    i = 0;
    for(w = p->left; w->type == PT_COMMA; w = w->left)
        i++;

    if (i<2 || (i%1)) {
        fprintf(stderr, "Error: PWL(expr, points...) needs an even and >=2 number of constant args\n");
        return (NULL);
    }

    data = (struct pwldata *) MALLOC(sizeof(struct pwldata));
    data->vals = (double*) MALLOC(i*sizeof(double));

    data->n = i;

    for (w = p->left ; --i >= 0 ; w = w->left)
        if (w->right->type == PT_CONSTANT) {
            data->vals[i] = w->right->constant;
        } else if (w->right->type == PT_FUNCTION &&
                   w->right->funcnum == PTF_UMINUS &&
                   w->right->left->type == PT_CONSTANT) {
            data->vals[i] = - w->right->left->constant;
        } else {
            fprintf(stderr, "PWL-ERROR: %s, not a constant\n", __func__);
            fprintf(stderr, "   type = %d\n", w->right->type);
            //Breakpoint;
            fprintf(stderr, "Error: PWL(expr, points...) only *literal* points are supported\n");
            return (NULL);
        }

#ifdef TRACE
    for (i = 0 ; i < data->n ; i += 2)
        fprintf(stderr, "  (%lf %lf)\n", data->vals[i], data->vals[i+1]);
#endif

    for (i = 2 ; i < data->n ; i += 2)
        if(data->vals[i-2] >= data->vals[i]) {
            fprintf(stderr, "Error: PWL(expr, points...) the abscissa of points must be ascending\n");
            return (NULL);
        }

    /* strip all but the first arg,
     *   and attach the rest as opaque data to the INPparseNode
     */

    p->left = w;
    p->data = (void *) data;

    return (p);
}


static INPparseNode *mkfnode(const char *fname, INPparseNode * arg)
{
    int i;
    INPparseNode *p;
    char buf[128], *name, *s;
    IFnode temp;

    /* Make sure the case is ok. */
    (void) strcpy(buf, fname);
    for (s = buf; *s; s++)
	if (isupper(*s))
	    *s = tolower(*s);

    p = (INPparseNode *) MALLOC(sizeof(INPparseNode));

    if (!strcmp(buf, "v")) {
	name = MALLOC(128);
	if (arg->type == PT_PLACEHOLDER) {
	    strcpy(name, arg->funcname);
	} else if (arg->type == PT_CONSTANT) {
	    (void) sprintf(name, "%d", (int) arg->constant);
	} else if (arg->type != PT_COMMA) {
	    fprintf(stderr, "Error: badly formed node voltage\n");
	    return (NULL);
	}

	if (arg->type == PT_COMMA) {
	    /* Change v(a,b) into v(a) - v(b) */
	    p = mkb(PT_MINUS, mkfnode(fname, arg->left),
		    mkfnode(fname, arg->right));
	} else {
	    INPtermInsert(circuit, &name, tables, &temp);
	    for (i = 0; i < numvalues; i++)
	 	if ((types[i] == IF_NODE) && (values[i].nValue == temp))
		     break;
	    if (i == numvalues) {
		if (numvalues) {
		    values = (IFvalue *)
			REALLOC((char *) values,
				(numvalues + 1) * sizeof(IFvalue));
		    types = (int *)
			REALLOC((char *) types,
				(numvalues + 1) * sizeof(int));
		} else {
		    values = (IFvalue *) MALLOC(sizeof(IFvalue));
		    types = (int *) MALLOC(sizeof(int));
		}
		values[i].nValue = temp;
		types[i] = IF_NODE;
		numvalues++;
	    }
	    p->valueIndex = i;
	    p->type = PT_VAR;
	}
    } else if (!strcmp(buf, "i")) {
	name = MALLOC(128);
	if (arg->type == PT_PLACEHOLDER)
	    strcpy(name, arg->funcname);
	else if (arg->type == PT_CONSTANT)
	    (void) sprintf(name, "%d", (int) arg->constant);
	else {
	    fprintf(stderr, "Error: badly formed branch current\n");
	    return (NULL);
	}
	INPinsert(&name, tables);
	for (i = 0; i < numvalues; i++)
	    if ((types[i] == IF_INSTANCE) && (values[i].uValue == name))
		break;
	if (i == numvalues) {
	    if (numvalues) {
		values = (IFvalue *)
		    REALLOC((char *) values,
			    (numvalues + 1) * sizeof(IFvalue));
		types = (int *)
		    REALLOC((char *) types, (numvalues + 1) * sizeof(int));
	    } else {
		values = (IFvalue *) MALLOC(sizeof(IFvalue));
		types = (int *) MALLOC(sizeof(int));
	    }
	    values[i].uValue = (IFuid) name;
	    types[i] = IF_INSTANCE;
	    numvalues++;
	}
	p->valueIndex = i;
	p->type = PT_VAR;

    } else if(!strcmp("ternary_fcn", buf)) {

//	extern void printTree(INPparseNode *);
//
//	printf("debug: %s ternary_fcn: ", __func__);
//	printTree(arg);
//	printf("\n");

	if(arg->type != PT_COMMA || arg->left->type != PT_COMMA) {
	    fprintf(stderr, "Error: bogus ternary_fcn form\n");
	    return (NULL);
	} else {
	    INPparseNode *arg1 = arg->left->left;
	    INPparseNode *arg2 = arg->left->right;
	    INPparseNode *arg3 = arg->right;

	    p->type = PT_TERN;
	    p->left = arg1;
	    p->right = mkb(PT_COMMA, arg2, arg3);
	}


    } else {
	for (i = 0; i < NUM_FUNCS; i++)
	    if (!strcmp(funcs[i].name, buf))
		break;

	if (i == NUM_FUNCS) {
	    fprintf(stderr, "Error: no such function '%s'\n", buf);
	    return (NULL);
	}

	p->type = PT_FUNCTION;
	p->left = arg;
	p->funcname = funcs[i].name;
	p->funcnum = funcs[i].number;
	p->function = funcs[i].funcptr;
        p->data = NULL;

        if(p->funcnum == PTF_PWL)
            p = prepare_PTF_PWL(p);
    }

    return (p);
}

/* Number node. */

static INPparseNode *mknnode(double number)
{
    struct INPparseNode *p;

    p = (INPparseNode *) MALLOC(sizeof(INPparseNode));

    p->type = PT_CONSTANT;
    p->constant = number;

    return (p);
}

/* String node. */

static INPparseNode *mksnode(const char *string, void *ckt)
{
    int i, j;
    char buf[128], *s;
    INPparseNode *p;

    /* Make sure the case is ok. */
    (void) strcpy(buf, string);
    for (s = buf; *s; s++)
	if (isupper(*s))
	    *s = tolower(*s);

    p = (INPparseNode *) MALLOC(sizeof(INPparseNode));

    if(!strcmp("time", buf)) {
        p->type = PT_TIME;
        p->data = ckt;
        return p;
    }

    if(!strcmp("temper", buf)) {
        p->type = PT_TEMPERATURE;
        p->data = ckt;
        return p;
    }

    if(!strcmp("hertz", buf)) {
        p->type = PT_FREQUENCY;
        p->data = ckt;
        return p;
    }

    /* First see if it's something special. */
    for (i = 0; i < ft_sim->numSpecSigs; i++)
	if (!strcmp(ft_sim->specSigs[i], buf))
	    break;
    if (i < ft_sim->numSpecSigs) {
	for (j = 0; j < numvalues; j++)
	    if ((types[j] == IF_STRING) && !strcmp(buf, values[i].sValue))
		break;
	if (j == numvalues) {
	    if (numvalues) {
		values = (IFvalue *)
		    REALLOC((char *) values,
			    (numvalues + 1) * sizeof(IFvalue));
		types = (int *)
		    REALLOC((char *) types, (numvalues + 1) * sizeof(int));
	    } else {
		values = (IFvalue *) MALLOC(sizeof(IFvalue));
		types = (int *) MALLOC(sizeof(int));
	    }
	    values[i].sValue = MALLOC(strlen(buf) + 1);
	    strcpy(values[i].sValue, buf);
	    types[i] = IF_STRING;
	    numvalues++;
	}
	p->valueIndex = i;
	p->type = PT_VAR;
	return (p);
    }

    for (i = 0; i < NUM_CONSTANTS; i++)
	if (!strcmp(constants[i].name, buf))
	    break;

    if (i == NUM_CONSTANTS) {
	/* We'd better save this in case it's part of i(something). */
	p->type = PT_PLACEHOLDER;
	p->funcname = (/*nonconst*/ char *) string;
    } else {
	p->type = PT_CONSTANT;
	p->constant = constants[i].value;
    }

    return (p);
}

/* The lexical analysis routine. */

int PTlex (YYSTYPE *lvalp, char **line)
{
    double td;
    int err;
    static char *specials = " \t()^+-*/,";
    char *sbuf, *s;
    int token;

    sbuf = *line;
#ifdef TRACE
//    printf("entering lexer, sbuf = '%s', lastoken = %d, lasttype = %d\n", 
//        sbuf, lasttoken, lasttype);
#endif
    while ((*sbuf == ' ') || (*sbuf == '\t'))
	sbuf++;

    switch (*sbuf) {
    case '\0':
	token = 0;
	break;

    case '?':
    case ':':
    case ',':
    case '-':
    case '+':
    case '/':
    case '^':
    case '(':
    case ')':
	token = *sbuf++;
	break;

    case '*':
      if(sbuf[1] == '*') {
        sbuf += 2;
        token = '^';            /* `**' is exponentiation */
        break;
      } else {
        token = *sbuf++;
        break;
      }

    case '&':
      if(sbuf[1] == '&') {
        sbuf += 2;
        token = TOK_AND;
        break;
      } else {
        token = *sbuf++;
        break;
      }

    case '|':
      if(sbuf[1] == '|') {
        sbuf += 2;
        token = TOK_OR;
        break;
      } else {
        token = *sbuf++;
        break;
      }

    case '=':
      if(sbuf[1] == '=') {
        sbuf += 2;
        token = TOK_EQ;
        break;
      } else {
        token = *sbuf++;
        break;
      }

    case '!':
      if(sbuf[1] == '=') {
        sbuf += 2;
        token = TOK_NE;
        break;
      } else {
        token = *sbuf++;
        break;
      }

    case '>':
      if(sbuf[1] == '=') {
        sbuf += 2;
        token = TOK_GE;
        break;
      } else {
        sbuf += 1;
        token = TOK_GT;
        break;
      }

    case '<':
      if(sbuf[1] == '>') {
        sbuf += 2;
        token = TOK_NE;
        break;
      }
      else if(sbuf[1] == '=') {
        sbuf += 2;
        token = TOK_LE;
        break;
      } else {
        sbuf += 1;
        token = TOK_LT;
        break;
      }

    default:
	td = INPevaluate(&sbuf, &err, 1);
	if (err == OK) {
	    token = TOK_NUM;
	    lvalp->num = td;
	} else {
        char *tmp;
	    token = TOK_STR;
	    for (s = sbuf; *s; s++)
		if (index(specials, *s))
		    break;
	    tmp = MALLOC(s - sbuf + 1);
	    strncpy(tmp, sbuf, s - sbuf);
	    tmp[s - sbuf] = '\0';
	    lvalp->str = tmp;
	    sbuf = s;
	}
    }

    *line = sbuf;

#ifdef TRACE
//    printf("PTlexer: token = %d, type = %d, left = '%s'\n", 
//        el.token, el.type, sbuf); */
#endif
    return (token);
}

#ifdef TRACE

/* Debugging stuff. */

void printTree(INPparseNode *);

void INPptPrint(char *str, IFparseTree * ptree)
{
    int i;

    printf("%s\n\t", str);
    printTree(((INPparseTree *) ptree)->tree);
    printf("\n");
    for (i = 0; i < ptree->numVars; i++) {
	printf("d / d v%d : ", i);
	printTree(((INPparseTree *) ptree)->derivs[i]);
	printf("\n");
    }
    return;
}

void printTree(INPparseNode * pt)
{
    switch (pt->type) {
    case PT_TIME:
        printf("time(ckt = %p)", pt->data);
	break;

    case PT_TEMPERATURE:
        printf("temperature(ckt = %p)", pt->data);
	break;

    case PT_FREQUENCY:
        printf("frequency(ckt = %p)", pt->data);
	break;

    case PT_CONSTANT:
	printf("%g", pt->constant);
	break;

    case PT_VAR:
	printf("v%d", pt->valueIndex);
	break;

    case PT_PLUS:
	printf("(");
	printTree(pt->left);
	printf(") + (");
	printTree(pt->right);
	printf(")");
	break;

    case PT_MINUS:
	printf("(");
	printTree(pt->left);
	printf(") - (");
	printTree(pt->right);
	printf(")");
	break;

    case PT_TIMES:
	printf("(");
	printTree(pt->left);
	printf(") * (");
	printTree(pt->right);
	printf(")");
	break;

    case PT_DIVIDE:
	printf("(");
	printTree(pt->left);
	printf(") / (");
	printTree(pt->right);
	printf(")");
	break;

    case PT_POWER:
	printf("(");
	printTree(pt->left);
	printf(") ^ (");
	printTree(pt->right);
	printf(")");
	break;

 
    case PT_COMMA:
	printf("(");
	printTree(pt->left);
	printf(") , (");
	printTree(pt->right);
	printf(")");
	break;

    case PT_FUNCTION:
	printf("%s (", pt->funcname);
	printTree(pt->left);
	printf(")");
	break;
 
    case PT_TERN:
	printf("ternary_fcn (");
	printTree(pt->left);
	printf(") , (");
	printTree(pt->right);
	printf(")");
	break;

    default:
	printf("oops");
	break;
    }
    return;
}

#endif
