/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Wayne A. Christopher, U. C. Berkeley CAD Group 
**********/

/*
 *   faustus@cad.berkeley.edu, ucbvax!faustus
 *
 * These definitions specify the format of the parse tree parameter type.
 * The first four are the elements of IFparseTree, defined in IFsim.h.
 */

#include "ifsim.h"

#ifndef INP_PARSE
#define INP_PARSE


/* This is the parameter value passed to the device routines.  To get the
 * value of the function, where tree is a pointer to the INPparseTree,
 * result is a pointer to where you want the result, derivs is a pointer to
 * an vector of numVars doubles, and vals is a pointer to the selected
 * elements from the RHS, do
 *  tree->p.IFeval(&tree, result, vals, derivs)
 * This routine will return either OK, E_PARMVAL, or E_PANIC.  If an error
 * is reported the eval function will have printed something to standard
 * out before returning.
 */

typedef struct INPparseTree {
    IFparseTree p;
    struct INPparseNode *tree;  /* The real stuff. */
    struct INPparseNode **derivs;   /* The derivative parse trees. */
} INPparseTree;

/* This is what is passed as the actual parameter value.  The fields will all
 * be filled in as needed.
 *
 * Values with names like v(something) and i(something) are treated specially.
 * They are considered voltages at nodes and branch currents through
 * voltage sources, respectively.  The corresponding parameters will be of
 * type IF_NODE and IF_INSTANCE, respectively.
 */

typedef struct INPparseNode {
    int type;           /* One of PT_*, below. */
    struct INPparseNode *left;  /* Left operand, or single operand. */
    struct INPparseNode *right; /* Right operand, if there is one. */
    double constant;        /* If INP_CONSTANT. */
    int valueIndex;         /* If INP_VAR, index into vars array. */
    char *funcname;         /* If INP_FUNCTION, name of function, */
    int funcnum;            /* ... one of PTF_*, */
    double (*function)();       /* ... and pointer to the function. */
} INPparseNode;

/* These are the possible types of nodes we can have in the parse tree.  The
 * numbers for the ops 1 - 5 have to be the same as the token numbers,
 * below.
 */

#define PT_PLACEHOLDER  0       /* For i(something) ... */
#define PT_PLUS     1
#define PT_MINUS    2
#define PT_TIMES    3
#define PT_DIVIDE   4
#define PT_POWER    5
#define PT_FUNCTION 6
#define PT_CONSTANT 7
#define PT_VAR      8
#define PT_COMMA    10

/* These are the functions that we support. */

#define PTF_ACOS    0
#define PTF_ACOSH   1
#define PTF_ASIN    2
#define PTF_ASINH   3
#define PTF_ATAN    4
#define PTF_ATANH   5
#define PTF_COS     6
#define PTF_COSH    7
#define PTF_EXP     8
#define PTF_LN      9
#define PTF_LOG     10
#define PTF_SIN     11
#define PTF_SINH    12
#define PTF_SQRT    13
#define PTF_TAN     14
#define PTF_TANH    15
#define PTF_UMINUS  16
#define PTF_ABS		17
#define PTF_SGN		18
#define PTF_USTEP	19
#define PTF_URAMP	20

/* The following things are used by the parser -- these are the token types the
 * lexer returns.
 */

#define TOK_END         0
#define TOK_PLUS        1
#define TOK_MINUS       2
#define TOK_TIMES       3
#define TOK_DIVIDE      4
#define TOK_POWER       5
#define TOK_UMINUS      6
#define TOK_LPAREN      7
#define TOK_RPAREN      8
#define TOK_VALUE       9
#define TOK_COMMA       10

/* And the types for value tokens... */

#define TYP_NUM         0
#define TYP_STRING      1
#define TYP_PNODE       2

/* A parser stack element. */

typedef struct PTelement {
    int token;
    int type;
    union {
        char *string;
        double real;
        INPparseNode *pnode;
    } value;
} PTelement ;

#define PT_STACKSIZE 200

/* These are things defined in PTfunctions.c */

extern double PTplus();
extern double PTminus();
extern double PTtimes();
extern double PTdivide();
extern double PTpower();
extern double PTacos();
extern double PTabs();
extern double PTacosh();
extern double PTasin();
extern double PTasinh();
extern double PTatan();
extern double PTatanh();
extern double PTcos();
extern double PTcosh();
extern double PTexp();
extern double PTln();
extern double PTlog();
extern double PTsgn();
extern double PTsin();
extern double PTsinh();
extern double PTsqrt();
extern double PTtan();
extern double PTtanh();
extern double PTustep();
extern double PTuramp();
extern double PTuminus();

/* And in IFeval.c */

#ifdef __STDC__
extern int IFeval(IFparseTree *, double, double*, double*, double*);
#else /* stdc */
extern int IFeval();
#endif /* stdc */

#endif

