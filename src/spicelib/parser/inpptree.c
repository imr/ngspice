/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Wayne A. Christopher, U. C. Berkeley CAD Group 
**********/

#include "ngspice.h"
#include <stdio.h>
#include <ctype.h>
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
static INPparseNode *PTparse(char **line);
static INPparseNode *makepnode(PTelement * elem);
static INPparseNode *mkbnode(int opnum, INPparseNode * arg1,
			     INPparseNode * arg2);
static INPparseNode *mkfnode(char *fname, INPparseNode * arg);
static INPparseNode *mknnode(double number);
static INPparseNode *mksnode(char *string);
static INPparseNode *PTdifferentiate(INPparseNode * p, int varnum);
static PTelement *PTlexer(char **line);

static IFvalue *values = NULL;
static int *types;
static int numvalues;
static void *circuit;
static INPtables *tables;



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

#define NUM_OPS (sizeof (ops) / sizeof (struct op))

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
    { "u2",	PTF_USTEP2, PTustep2}
} ;

#define NUM_FUNCS (sizeof (funcs) / sizeof (struct func))

/* These are all the constants any sane person needs. */

static struct constant {
    char *name;
    double value;
} constants[] = {
    {
    "e", M_E}, {
    "pi", M_PI}
};

#define NUM_CONSTANTS (sizeof (constants) / sizeof (struct constant))

/* Parse the expression in *line as far as possible, and return the parse
 * tree obtained.  If there is an error, *pt will be set to NULL and an error
 * message will be printed.
 */

void
INPgetTree(char **line, INPparseTree ** pt, void *ckt, INPtables * tab)
{
    INPparseNode *p;
    int i;

    values = NULL;
    types = NULL;
    numvalues = 0;

    circuit = ckt;
    tables = tab;

    p = PTparse(line);

    if (!p || !PTcheck(p)) {
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

/* printf("differentiating: "); printTree(p); printf(" wrt var %d\n", varnum);*/

    switch (p->type) {
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
	    arg1 = mkcon((double) 0.0);
	    break;

	case PTF_URAMP:
	    arg1 = mkf(PTF_USTEP, p->left);
	    break;

	    /* MW. PTF_CIF for new cif function */
	case PTF_USTEP2:
	    arg1 = mkcon((double) 0.0);
	    break;
	    
        case PTF_UMINUS:    /* - 1 ; like a constant (was 0 !) */
            arg1 = mkcon((double) - 1.0);
            break;

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

/* printf("result is: "); printTree(newp); printf("\n"); */
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
    }

    p->type = type;
    p->left = left;
    p->right = right;

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

    return (p);
}

/* Check for remaining PT_PLACEHOLDERs in the parse tree.  Returns 1 if ok. */

static int PTcheck(INPparseNode * p)
{
    switch (p->type) {
    case PT_PLACEHOLDER:
	return (0);

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
	return (PTcheck(p->left) && PTcheck(p->right));

    default:
	fprintf(stderr, "Internal error: bad node type %d\n", p->type);
	return (0);
    }
}

/* The operator-precedence table for the parser. */

#define G 1			/* Greater than. */
#define L 2			/* Less than. */
#define E 3			/* Equal. */
#define R 4			/* Error. */

static char prectable[11][11] = {
    /* $  +  -  *  /  ^  u- (  )  v  , */
/* $ */ {R, L, L, L, L, L, L, L, R, L, R},
/* + */ {G, G, G, L, L, L, L, L, G, L, G},
/* - */ {G, G, G, L, L, L, L, L, G, L, G},
/* * */ {G, G, G, G, G, L, L, L, G, L, G},
/* / */ {G, G, G, G, G, L, L, L, G, L, G},
/* ^ */ {G, G, G, G, G, L, L, L, G, L, G},
/* u-*/ {G, G, G, G, G, G, G, L, G, L, R},
/* ( */ {R, L, L, L, L, L, L, L, E, L, L},
/* ) */ {G, G, G, G, G, G, G, R, G, R, G},
/* v */ {G, G, G, G, G, G, G, G, G, R, G},
/* , */ {G, L, L, L, L, L, L, L, G, L, G}

};

/* Return an expr. */

static INPparseNode *PTparse(char **line)
{
    PTelement stack[PT_STACKSIZE];
    int sp = 0, st, i;
    PTelement *top, *next;
    INPparseNode *pn, *lpn, *rpn;

    stack[0].token = TOK_END;
    next = PTlexer(line);

    while ((sp > 1) || (next->token != TOK_END)) {
	/* Find the top-most terminal. */
	i = sp;
	do {
	    top = &stack[i--];
	} while (top->token == TOK_VALUE);


	switch (prectable[top->token][next->token]) {
	case L:
	case E:
	    /* Push the token read. */
	    if (sp == (PT_STACKSIZE - 1)) {
		fprintf(stderr, "Error: stack overflow\n");
		return (NULL);
	    }
	    bcopy((char *) next, (char *) &stack[++sp], sizeof(PTelement));
	    next = PTlexer(line);
	    continue;

	case R:
	    fprintf(stderr, "Syntax error.\n");
	    return (NULL);

	case G:
	    /* Reduce. Make st and sp point to the elts on the
	     * stack at the end and beginning of the junk to
	     * reduce, then try and do some stuff. When scanning
	     * back for a <, ignore VALUES.
	     */
	    st = sp;
	    if (stack[sp].token == TOK_VALUE)
		sp--;
	    while (sp > 0) {
		if (stack[sp - 1].token == TOK_VALUE)
		    i = 2;	/* No 2 pnodes together... */
		else
		    i = 1;
		if (prectable[stack[sp - i].token]
		    [stack[sp].token] == L)
		    break;
		else
		    sp = sp - i;
	    }
	    if (stack[sp - 1].token == TOK_VALUE)
		sp--;
	    /* Now try and see what we can make of this.
	     * The possibilities are: - node
	     *            node op node
	     *            ( node )
	     *            func ( node )
	     *            func ( node, node, node, ... )        <- new
	     *            node
	     */
	    if (st == sp) {
		pn = makepnode(&stack[st]);
		if (pn == NULL)
		    goto err;
	    } else if ((stack[sp].token == TOK_UMINUS) && (st == sp + 1)) {
		lpn = makepnode(&stack[st]);
		if (lpn == NULL)
		    goto err;
		pn = mkfnode("-", lpn);
	    } else if ((stack[sp].token == TOK_LPAREN) &&
		       (stack[st].token == TOK_RPAREN)) {
		pn = makepnode(&stack[sp + 1]);
		if (pn == NULL)
		    goto err;
	    } else if ((stack[sp + 1].token == TOK_LPAREN) &&
		       (stack[st].token == TOK_RPAREN)) {
		lpn = makepnode(&stack[sp + 2]);
		if ((lpn == NULL) || (stack[sp].type != TYP_STRING))
		    goto err;
		if (!(pn = mkfnode(stack[sp].value.string, lpn)))
		    return (NULL);
	    } else {		/* node op node */
		lpn = makepnode(&stack[sp]);
		rpn = makepnode(&stack[st]);
		if ((lpn == NULL) || (rpn == NULL))
		    goto err;
		pn = mkbnode(stack[sp + 1].token, lpn, rpn);
	    }
	    stack[sp].token = TOK_VALUE;
	    stack[sp].type = TYP_PNODE;
	    stack[sp].value.pnode = pn;
	    continue;
	}
    }
    pn = makepnode(&stack[1]);
    if (pn)
	return (pn);
  err:
    fprintf(stderr, "Syntax error.\n");
    return (NULL);
}

/* Given a pointer to an element, make a pnode out of it (if it already
 * is one, return a pointer to it). If it isn't of type VALUE, then return
 * NULL.
 */

static INPparseNode *makepnode(PTelement * elem)
{
    if (elem->token != TOK_VALUE)
	return (NULL);

    switch (elem->type) {
    case TYP_STRING:
	return (mksnode(elem->value.string));

    case TYP_NUM:
	return (mknnode(elem->value.real));

    case TYP_PNODE:
	return (elem->value.pnode);

    default:
	fprintf(stderr, "Internal Error: bad token type\n");
	return (NULL);
    }
}

/* Binop node. */

static INPparseNode *mkbnode(int opnum, INPparseNode * arg1,
			     INPparseNode * arg2)
{
    INPparseNode *p;
    int i;

    for (i = 0; i < NUM_OPS; i++)
	if (ops[i].number == opnum)
	    break;

    if (i == NUM_OPS) {
	fprintf(stderr, "Internal Error: no such op num %d\n", opnum);
	return (NULL);
    }
    p = (INPparseNode *) MALLOC(sizeof(INPparseNode));

    p->type = opnum;
    p->funcname = ops[i].name;
    p->function = ops[i].funcptr;
    p->left = arg1;
    p->right = arg2;

    return (p);
}

static INPparseNode *mkfnode(char *fname, INPparseNode * arg)
{
    int i;
    INPparseNode *p;
    char buf[128], *name, *s;
    IFvalue temp;

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
	    /* printf("getting a node called '%s'\n", name); */
	    INPtermInsert(circuit, &name, tables, &(temp.nValue));
	    for (i = 0; i < numvalues; i++)
		if ((types[i] == IF_NODE) && (values[i].nValue ==
					      temp.nValue)) break;
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
		values[i] = temp;
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
/* printf("getting a device called '%s'\n", name); */
	INPinsert(&name, tables);
	for (i = 0; i < numvalues; i++)
	    if ((types[i] == IF_INSTANCE) && (values[i].uValue ==
					      temp.uValue)) break;
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

static INPparseNode *mksnode(char *string)
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
	p->funcname = string;
    } else {
	p->type = PT_CONSTANT;
	p->constant = constants[i].value;
    }

    return (p);
}

/* The lexical analysis routine. */

static PTelement *PTlexer(char **line)
{
    double td;
    int err;
    static PTelement el;
    static char *specials = " \t()^+-*/,";
    static int lasttoken = TOK_END, lasttype;
    char *sbuf, *s;

    sbuf = *line;
#ifdef notdef
    printf("entering lexer, sbuf = '%s', lastoken = %d, lasttype = %d\n", 
        sbuf, lasttoken, lasttype);
#endif
    while ((*sbuf == ' ') || (*sbuf == '\t') || (*sbuf == '='))
	sbuf++;

    switch (*sbuf) {
    case '\0':
	el.token = TOK_END;
	break;

    case ',':
	el.token = TOK_COMMA;
	sbuf++;
	break;

    case '-':
	if ((lasttoken == TOK_VALUE) || (lasttoken == TOK_RPAREN))
	    el.token = TOK_MINUS;
	else
	    el.token = TOK_UMINUS;
	sbuf++;
	break;

    case '+':
	el.token = TOK_PLUS;
	sbuf++;
	break;

    case '*':
	el.token = TOK_TIMES;
	sbuf++;
	break;

    case '/':
	el.token = TOK_DIVIDE;
	sbuf++;
	break;

    case '^':
	el.token = TOK_POWER;
	sbuf++;
	break;

    case '(':
	if (((lasttoken == TOK_VALUE) && ((lasttype == TYP_NUM))) ||
	    (lasttoken == TOK_RPAREN)) {
	    el.token = TOK_END;
	} else {
	    el.token = TOK_LPAREN;
	    sbuf++;
	}
	break;

    case ')':
	el.token = TOK_RPAREN;
	sbuf++;
	break;

    default:
	if ((lasttoken == TOK_VALUE) || (lasttoken == TOK_RPAREN)) {
	    el.token = TOK_END;
	    break;
	}

	td = INPevaluate(&sbuf, &err, 1);
	if (err == OK) {
	    el.token = TOK_VALUE;
	    el.type = TYP_NUM;
	    el.value.real = td;
	} else {
	    el.token = TOK_VALUE;
	    el.type = TYP_STRING;
	    for (s = sbuf; *s; s++)
		if (index(specials, *s))
		    break;
	    el.value.string = MALLOC(s - sbuf + 1);
	    strncpy(el.value.string, sbuf, s - sbuf);
	    el.value.string[s - sbuf] = '\0';
	    sbuf = s;
	}
    }

    lasttoken = el.token;
    lasttype = el.type;

    *line = sbuf;

/* printf("PTlexer: token = %d, type = %d, left = '%s'\n", 
        el.token, el.type, sbuf); */

    return (&el);
}

#ifdef notdef

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

    case PT_FUNCTION:
	printf("%s (", pt->funcname);
	printTree(pt->left);
	printf(")");
	break;

    default:
	printf("oops");
	break;
    }
    return;
}

#endif
