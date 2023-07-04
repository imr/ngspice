/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/
//#define TRACE

/* Tree generator for B-Source parser */

#include "ngspice/ngspice.h"
#include "ngspice/compatmode.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/inpdefs.h"
#include "ngspice/inpptree.h"
#include "inpxx.h"

#include "inpptree-parser.h"
#include "inpptree-parser-y.h"

extern bool ft_stricterror;

#ifdef OLD_BISON
int PTparse(char **line, INPparseNode **p, CKTcircuit *ckt);
#endif

static INPparseNode *mkcon(double value);
static INPparseNode *mkb(int type, INPparseNode * left,
                         INPparseNode * right);
static INPparseNode *mkf(int type, INPparseNode * arg);
static int PTcheck(INPparseNode * p, char* tline);
static INPparseNode *mkvnode(char *name);
static INPparseNode *mkinode(char *name);

static INPparseNode *PTdifferentiate(INPparseNode * p, int varnum);

static void free_tree(INPparseNode *);
static void printTree(INPparseNode *);


/*
 * LAW for INPparseNode* generator and consumer functions:
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *   Newly allocated structs shall be initialized with `usecnt' = 0
 *   When filling INPparseNode * slots of newly initialized structs
 *     their `usecnt' shall be incremented
 *   Generators pass the responsibility `to free' return values
 *     on to their invokers.
 *   Functions generally process args with exactly one of:
 *     - inc_usage(arg) if they insert an argument into a struct
 *     - release_tree(arg) if they don't make any use of it
 *     - pass it on to another function()
 *   Functions use the the result of a function invocations with one of:
 *     - inc_usage(result) if they insert the result into a struct
 *     - release_tree(result) if they don't make any use of it
 *     - pass it on to another function()
 *     - simply return the result
 *
 * mkfirst(first, second)
 *   is used to safely release its second argument,
 *   and return its first
 *
 */


static inline INPparseNode *
inc_usage(INPparseNode *p)
{
    if(p)
        p->usecnt ++;
    return p;
}


static void
dec_usage(INPparseNode *p)
{
    if(p && --p->usecnt <= 0)
        free_tree(p);
}


static void
release_tree(INPparseNode *p)
{
    if(p && p->usecnt <= 0)
        free_tree(p);
}


static INPparseNode *
mkfirst(INPparseNode *fst, INPparseNode *snd)
{
    if(fst) {
        fst->usecnt ++;
        release_tree(snd);
        fst->usecnt --;
    } else {
        release_tree(snd);
    }

    return fst;
}


static IFvalue *values = NULL;
static int *types;
static int numvalues;
static CKTcircuit *circuit;
static INPtables *tables;

extern IFsimulator *ft_sim;        /* XXX */

/* Some tables that the parser uses. */

static struct op {
    int number;
    char *name;
    void (*funcptr)(void);
} ops[] = {
    {
    PT_COMMA,  ",", NULL}, {
    PT_PLUS,   "+", (void(*)(void)) PTplus}, {
    PT_MINUS,  "-", (void(*)(void)) PTminus}, {
    PT_TIMES,  "*", (void(*)(void)) PTtimes}, {
    PT_DIVIDE, "/", (void(*)(void)) PTdivide}, {
    PT_POWER,  "^", (void(*)(void)) PTpowerH}
};

#define NUM_OPS (int)NUMELEMS(ops)

static struct func {
    char *name;
    int number;
    void (*funcptr)(void);
} funcs[] = {
    { "abs",    PTF_ABS,    (void(*)(void)) PTabs } ,
    { "acos",   PTF_ACOS,   (void(*)(void)) PTacos } ,
    { "acosh",  PTF_ACOSH,  (void(*)(void)) PTacosh } ,
    { "asin",   PTF_ASIN,   (void(*)(void)) PTasin } ,
    { "asinh",  PTF_ASINH,  (void(*)(void)) PTasinh } ,
    { "atan",   PTF_ATAN,   (void(*)(void)) PTatan } ,
    { "atanh",  PTF_ATANH,  (void(*)(void)) PTatanh } ,
    { "cos",    PTF_COS,    (void(*)(void)) PTcos } ,
    { "cosh",   PTF_COSH,   (void(*)(void)) PTcosh } ,
    { "exp",    PTF_EXP,    (void(*)(void)) PTexp } ,
    { "ln",     PTF_LOG,    (void(*)(void)) PTlog } ,
    { "log",    PTF_LOG,    (void(*)(void)) PTlog } ,
    { "log10",  PTF_LOG10,  (void(*)(void)) PTlog10 } ,
    { "sgn",    PTF_SGN,    (void(*)(void)) PTsgn } ,
    { "sin",    PTF_SIN,    (void(*)(void)) PTsin } ,
    { "sinh",   PTF_SINH,   (void(*)(void)) PTsinh } ,
    { "sqrt",   PTF_SQRT,   (void(*)(void)) PTsqrt } ,
    { "tan",    PTF_TAN,    (void(*)(void)) PTtan } ,
    { "tanh",   PTF_TANH,   (void(*)(void)) PTtanh } ,
    { "u",      PTF_USTEP,  (void(*)(void)) PTustep } ,
    { "uramp",  PTF_URAMP,  (void(*)(void)) PTuramp } ,
    { "ceil",   PTF_CEIL,   (void(*)(void)) PTceil } ,
    { "floor",  PTF_FLOOR,  (void(*)(void)) PTfloor } ,
    { "nint",   PTF_NINT,   (void(*)(void)) PTnint } ,
    { "-",      PTF_UMINUS, (void(*)(void)) PTuminus },
    { "u2",     PTF_USTEP2, (void(*)(void)) PTustep2},
    { "pwl",    PTF_PWL,    (void(*)(void)) PTpwl},
    { "pwl_derivative", PTF_PWL_DERIVATIVE, (void(*)(void)) PTpwl_derivative},
    { "eq0",    PTF_EQ0,    (void(*)(void)) PTeq0},
    { "ne0",    PTF_NE0,    (void(*)(void)) PTne0},
    { "gt0",    PTF_GT0,    (void(*)(void)) PTgt0},
    { "lt0",    PTF_LT0,    (void(*)(void)) PTlt0},
    { "ge0",    PTF_GE0,    (void(*)(void)) PTge0},
    { "le0",    PTF_LE0,    (void(*)(void)) PTle0},
    { "pow",    PTF_POW,    (void(*)(void)) PTpower},
    { "pwr",    PTF_PWR,    (void(*)(void)) PTpwr},
    { "min",    PTF_MIN,    (void(*)(void)) PTmin},
    { "max",    PTF_MAX,    (void(*)(void)) PTmax},
    { "ddt",    PTF_DDT,    (void(*)(void)) PTddt},
} ;

#define NUM_FUNCS (int)NUMELEMS(funcs)

/* These are all the constants any sane person needs. */

static struct constant {
    char *name;
    double value;
} constants[] = {
    {
    "e", M_E}, {
    "pi", M_PI}
};

#define NUM_CONSTANTS (int)NUMELEMS(constants)

/* Parse the expression in *line as far as possible, and return the parse
 * tree obtained.  If there is an error, *pt will be set to NULL and an error
 * message will be printed.
 */

void
INPgetTree(char **line, INPparseTree ** pt, CKTcircuit *ckt, INPtables * tab)
{
    INPparseNode *p = NULL;
    int i, rv;
    char* treeline = *line;

    values = NULL;
    types = NULL;
    numvalues = 0;

    circuit = ckt;
    tables = tab;

#ifdef TRACE
    fprintf(stderr,"%s, line = \"%s\"\n", __func__, *line);
#endif

    rv = PTparse(line, &p, ckt);

    if (rv || !p || !PTcheck(p, treeline)) {

        *pt = NULL;
        release_tree(p);

    } else {

        (*pt) = TMALLOC(INPparseTree, 1);

        (*pt)->p.numVars = numvalues;
        (*pt)->p.varTypes = types;
        (*pt)->p.vars = values;
        (*pt)->p.IFeval = IFeval;
        (*pt)->tree = inc_usage(p);

        (*pt)->derivs = TMALLOC(INPparseNode *, numvalues);

        for (i = 0; i < numvalues; i++)
            (*pt)->derivs[i] = inc_usage(PTdifferentiate(p, i));

    }

    values = NULL;
    types = NULL;
    numvalues = 0;

    circuit = NULL;
    tables = NULL;
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
    INPparseNode *arg1 = NULL, *arg2 = NULL, *newp = NULL;

    switch (p->type) {
    case PT_TIME:
    case PT_TEMPERATURE:
    case PT_FREQUENCY:
    case PT_CONSTANT:
        newp = mkcon(0.0);
        break;

    case PT_VAR:
        /* Is this the variable we're differentiating wrt? */
        if (p->valueIndex == varnum)
            newp = mkcon(1.0);
        else
            newp = mkcon(0.0);
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
                   mkb(PT_POWER, p->right, mkcon(2.0)));
        break;

    case PT_POWER:
        /*
         * ^ : a^b -> |a| math^ b
         *
         * D(pow(a,b))
         *   = D(exp(b*log(abs(a))))
         *   = exp(b*log(abs(a))) * D(b*log(abs(a)))
         *   = pow(a,b) * (D(b)*log(abs(a)) + b*D(abs(a))/abs(a))
         *   = pow(a,b) * (D(b)*log(abs(a)) + b*sgn(a)*D(a)/abs(a))
         *   = pow(a,b) * (D(b)*log(abs(a)) + b*D(a)/a)
         *
         * when D(b) == 0, then
         *
         * D(pow(a,b))
         *    = pow(a,b) * (D(b)*log(abs(a)) + b*D(a)/a)
         *    = pow(a,b) * b * D(a)/a
         *    = pow(a,b) * b * D(a)/(signum(a) * abs(a))
         *    = pow(a, b-1) * b * D(a) / signum(a)
         *    = pwr(a, b-1) * b * D(a)
         *
         * when D(a) == 0, then
         *
         * D(pow(a,b))
         *    = pow(a,b) * (D(b)*log(abs(a)) + b*D(a)/a)
         *    = pow(a,b) * D(b)*log(abs(a))
         */
#define a  p->left
#define b  p->right
        if (b->type == PT_CONSTANT) {
            arg1 = PTdifferentiate(a, varnum);
            if (newcompat.hs || newcompat.lt) {
                newp = mkb(PT_TIMES,
                    mkb(PT_TIMES,
                        mkcon(b->constant),
                        mkf(PTF_POW,
                            mkb(PT_COMMA, a, mkcon(b->constant - 1.0)))),
                    arg1);
            }
            else {
                newp = mkb(PT_TIMES,
                    mkb(PT_TIMES,
                        mkcon(b->constant),
                        mkf(PTF_PWR,
                            mkb(PT_COMMA, a, mkcon(b->constant - 1.0)))),
                    arg1);
            }
        }
        else if (a->type == PT_CONSTANT){
            arg2 = PTdifferentiate(b, varnum);
            newp = mkb(PT_TIMES,
                       mkf(PTF_POW, mkb(PT_COMMA, a, b)),
                           mkb(PT_TIMES, arg2, mkf(PTF_LOG, mkf(PTF_ABS, a))));
        }
        else {
            arg1 = PTdifferentiate(a, varnum);
            arg2 = PTdifferentiate(b, varnum);
            newp = mkb(PT_TIMES,
                       mkf(PTF_POW, mkb(PT_COMMA, a, b)),
                       mkb(PT_PLUS,
                           mkb(PT_TIMES, b,
                               mkb(PT_DIVIDE, arg1, a)),
                           mkb(PT_TIMES, arg2, mkf(PTF_LOG, mkf(PTF_ABS, a)))));
        }
#undef b
#undef a
        break;

    case PT_TERN: /* ternary_fcn(cond,exp1,exp2) */
      // naive:
      //   d/d ternary_fcn(cond,exp1,exp2) --> ternary_fcn(cond, d/d exp1, d/d exp2)
      {
//        extern void printTree(INPparseNode *);
//
//        printf("debug: %s, PT_TERN: ", __func__);
//        printTree(p);
//        printf("\n");

        newp = mkb(PT_TERN, p->left, mkb(PT_COMMA,
                                         PTdifferentiate(p->right->left, varnum),
                                         PTdifferentiate(p->right->right, varnum)));

//        printf("debug, %s, returns; ", __func__);
//        printTree(newp);
//        printf("\n");

        return mkfirst(newp, p);
      }

    case PT_FUNCTION:
        /* Many cases.  Set arg1 to the derivative of the function,
         * and arg2 to the derivative of the argument.
         */
        switch (p->funcnum) {
        case PTF_ABS:                /* sgn(u) */
            arg1 = mkf(PTF_SGN, p->left);
            break;

        case PTF_SGN:
            arg1 = mkcon(0.0);
            break;

        case PTF_ACOS:                /* - 1 / sqrt(1 - u^2) */
            arg1 = mkb(PT_DIVIDE, mkcon(-1.0), mkf(PTF_SQRT,
                                                          mkb(PT_MINUS,
                                                              mkcon(1.0),
                                                              mkb(PT_POWER,
                                                                  p->left,
                                                                  mkcon(2.0)))));
            break;

        case PTF_ACOSH:        /* 1 / sqrt(u^2 - 1) */
            arg1 = mkb(PT_DIVIDE, mkcon(1.0), mkf(PTF_SQRT,
                                                         mkb(PT_MINUS,
                                                             mkb(PT_POWER,
                                                                 p->left,
                                                                 mkcon(2.0)),
                                                             mkcon(1.0))));

            break;

        case PTF_ASIN:                /* 1 / sqrt(1 - u^2) */
            arg1 = mkb(PT_DIVIDE, mkcon(1.0), mkf(PTF_SQRT,
                                                         mkb(PT_MINUS,
                                                             mkcon(1.0),
                                                             mkb(PT_POWER,
                                                                 p->left,
                                                                 mkcon(2.0)))));
            break;

        case PTF_ASINH:        /* 1 / sqrt(u^2 + 1) */
            arg1 = mkb(PT_DIVIDE, mkcon(1.0), mkf(PTF_SQRT,
                                                         mkb(PT_PLUS,
                                                             mkb(PT_POWER,
                                                                 p->left,
                                                                 mkcon(2.0)),
                                                             mkcon(1.0))));
            break;

        case PTF_ATAN:                /* 1 / (1 + u^2) */
            arg1 = mkb(PT_DIVIDE, mkcon(1.0), mkb(PT_PLUS,
                                                         mkb(PT_POWER,
                                                             p->left,
                                                             mkcon(2.0)),
                                                         mkcon(1.0)));
            break;

        case PTF_ATANH:        /* 1 / (1 - u^2) */
            arg1 = mkb(PT_DIVIDE, mkcon(1.0), mkb(PT_MINUS,
                                                         mkcon(1.0),
                                                         mkb(PT_POWER,
                                                             p->left,
                                                             mkcon(2.0))));
            break;

        case PTF_COS:                /* - sin(u) */
            arg1 = mkf(PTF_UMINUS, mkf(PTF_SIN, p->left));
            break;

        case PTF_COSH:                /* sinh(u) */
            arg1 = mkf(PTF_SINH, p->left);
            break;

        case PTF_EXP:                /* u > EXPARGMAX -> EXPMAX, that is exp(EXPARGMAX), else exp(u) */
            if (newcompat.ps) {
                arg1 = mkb(PT_TERN,
                    mkf(PTF_GT0, mkb(PT_MINUS, p->left, mkcon(EXPARGMAX))),
                    mkb(PT_COMMA,
                        mkcon(EXPMAX),
                        mkf(PTF_EXP, p->left)));
            }
            else {                   /* exp(u) */
                arg1 = mkf(PTF_EXP, p->left);
            }

#ifdef TRACE1
            printf("debug exp, %s, returns; ", __func__);
            printTree(arg1);
            printf("\n");
#endif
            break;

        case PTF_LOG:               /* 1 / u */
            arg1 = mkb(PT_DIVIDE, mkcon(1.0), p->left);
            break;

        case PTF_LOG10:              /* log(e) / u */
            arg1 = mkb(PT_DIVIDE, mkcon(M_LOG10E), p->left);
            break;

        case PTF_SIN:                /* cos(u) */
            arg1 = mkf(PTF_COS, p->left);
            break;

        case PTF_SINH:                /* cosh(u) */
            arg1 = mkf(PTF_COSH, p->left);
            break;

        case PTF_SQRT:                /* 1 / (2 * sqrt(u)) */
            arg1 = mkb(PT_DIVIDE, mkcon(1.0), mkb(PT_TIMES,
                                                         mkcon(2.0),
                                                         mkf(PTF_SQRT,
                                                             p->left)));
            break;

        case PTF_TAN:                /* 1 + (tan(u) ^ 2) */
            arg1 = mkb(PT_PLUS, mkcon(1.0), mkb(PT_POWER,
                                                         mkf(PTF_TAN,
                                                             p->left),
                                                         mkcon(2.0)));
            break;

        case PTF_TANH:                /* 1 - (tanh(u) ^ 2) */
            arg1 = mkb(PT_MINUS, mkcon(1.0), mkb(PT_POWER,
                                                         mkf(PTF_TANH,
                                                             p->left),
                                                         mkcon(2.0)));
            break;

        case PTF_USTEP:
        case PTF_EQ0:
        case PTF_NE0:
        case PTF_GT0:
        case PTF_LT0:
        case PTF_GE0:
        case PTF_LE0:
            arg1 = mkcon(0.0);
            break;

        case PTF_URAMP:
            arg1 = mkf(PTF_USTEP, p->left);
            break;

        case PTF_FLOOR:                /* naive: D(floor(u)) = 0 */
            arg1 = mkcon(0.0);
            break;

        case PTF_CEIL:                /* naive: D(ceil(u)) = 0 */
            arg1 = mkcon(0.0);
            break;

        case PTF_NINT:                /* naive: D(nint(u)) = 0 */
            arg1 = mkcon(0.0);
            break;

        case PTF_USTEP2: /* ustep2=uramp(x)-uramp(x-1) ustep2'=ustep(x)-ustep(x-1) */
            arg1 = mkb(PT_MINUS,
                       mkf(PTF_USTEP, p->left),
                       mkf(PTF_USTEP,
                           mkb(PT_MINUS,
                               p->left,
                               mkcon(1.0))));
            break;

        case PTF_UMINUS:    /* - 1 ; like a constant (was 0 !) */
            arg1 = mkcon(-1.0);
            break;

        case PTF_PWL: /* PWL(var, x1, y1, x2, y2, ... a const list) */
            arg1 = mkf(PTF_PWL_DERIVATIVE, p->left);
            arg1->data = p->data;
            break;

        case PTF_PWL_DERIVATIVE: /* d/dvar PWL(var, ...) */
            arg1 = mkcon(0.0);
            break;

        case PTF_DDT:
            arg1 = mkcon(0.0);
            arg1->data = p->data;
            break;

        case PTF_MIN:
        case PTF_MAX:
        /* min(a,b) -->   (a<b)       ? a : b
        *           -->   ((a-b) < 0) ? a : b
        */
        {
            INPparseNode *a = p->left->left;
            INPparseNode *b = p->left->right;
            int comparison = (p->funcnum == PTF_MIN) ? PTF_LT0 : PTF_GT0;
#ifdef TRACE1
            printf("debug: %s, PTF_MIN: ", __func__);
            printTree(p);
            printf("\n");
            printf("debug: %s, PTF_MIN, a: ", __func__);
            printTree(a);
            printf("\n");
            printf("debug: %s, PTF_MIN, b: ", __func__);
            printTree(b);
            printf("\n");
#endif
            newp = mkb(PT_TERN,
                       mkf(comparison, mkb(PT_MINUS, a, b)),
                       mkb(PT_COMMA,
                           PTdifferentiate(a, varnum),
                           PTdifferentiate(b, varnum)));
#ifdef TRACE1
            printf("debug, %s, returns; ", __func__);
            printTree(newp);
            printf("\n");
#endif
            return mkfirst(newp, p);
        }

        break;

        case PTF_POW:
            /*
             * pow : pow(a,b) -> |a| math^ b
             *
             * D(pow(a,b))
             *   = D(exp(b*log(abs(a))))
             *   = exp(b*log(abs(a))) * D(b*log(abs(a)))
             *   = pow(a,b) * (D(b)*log(abs(a)) + b*D(abs(a))/abs(a))
             *   = pow(a,b) * (D(b)*log(abs(a)) + b*sgn(a)*D(a)/abs(a))
             *   = pow(a,b) * (D(b)*log(abs(a)) + b*D(a)/a)
             *
             * when D(b) == 0, then
             *
             * D(pow(a,b))
             *    = pow(a,b) * (D(b)*log(abs(a)) + b*D(a)/a)
             *    = pow(a,b) * b * D(a)/a
             *    = pow(a,b) * b * D(a)/(signum(a) * abs(a))
             *    = pow(a, b-1) * b * D(a) / signum(a)
             *    = pwr(a, b-1) * b * D(a)
             *
             * when D(a) == 0, then
             *
             * D(pow(a,b))
             *    = pow(a,b) * (D(b)*log(abs(a)) + b*D(a)/a)
             *    = pow(a,b) * D(b)*log(abs(a))
             */
        {
        /*
        pow(a,b)
        p->left: ','    p->left->left: a       p->left->right: b
        */
#define a  p->left->left
#define b  p->left->right

            if (b->type == PT_CONSTANT) {
                arg1 = PTdifferentiate(a, varnum);
                newp = mkb(PT_TIMES,
                           mkb(PT_TIMES,
                               mkcon(b->constant),
                               mkf(PTF_PWR,
                                   mkb(PT_COMMA, a, mkcon(b->constant - 1)))),
                           arg1);
            } else if (a->type == PT_CONSTANT) {
                arg2 = PTdifferentiate(b, varnum);
                newp = mkb(PT_TIMES,
                    mkf(PTF_POW, mkb(PT_COMMA, a, b)),
                    mkb(PT_TIMES, arg2, mkf(PTF_LOG, mkf(PTF_ABS, a))));

            } else {
                arg1 = PTdifferentiate(a, varnum);
                arg2 = PTdifferentiate(b, varnum);
                newp = mkb(PT_TIMES,
                           mkf(PTF_POW, mkb(PT_COMMA, a, b)),
                           mkb(PT_PLUS,
                               mkb(PT_TIMES,
                                   b,
                                   mkb(PT_DIVIDE, arg1, a)),
                               mkb(PT_TIMES,
                                   arg2,
                                   mkf(PTF_LOG, mkf(PTF_ABS, a)))));
            }
#ifdef TRACE
            printf("debug pow, %s, returns; ", __func__);
            printTree(newp);
            printf("\n");
#endif
            return mkfirst(newp, p);
#undef b
#undef a
        }

        break;

        case PTF_PWR:
            /*
             * pwr : pwr(a,b) -> signum(a) * (|a| math^ b)
             *                -> signum(a) * pow(a, b)
             *
             * Note:
             *   D(pow(a,b)) = pow(a,b) * (D(b)*log(abs(a)) + b*D(a)/a)
             *
             * D(pwr(a,b))
             *   = D(signum(a) * pow(a,b))
             *   = D(signum(a)) * pow(a,b) + signum(a) * D(pow(a,b))
             *   = 0 + signum(a) * pow(a,b) * (D(b)*log(abs(a)) + b*D(a)/a)
             *   = pwr(a,b) * (D(b)*log(abs(a)) + b*D(a)/a)
             *
             * with D(b) == 0
             *
             * D(pwr(a,b))
             *   = pwr(a,b) * b * D(a)/a
             *   = signum(a) * pow(a,b) * b * D(a)/(signum(a) * abs(a))
             *   = pow(a, b-1) * b * D(a)
             */
        {
        /*
        pwr(a,b)
        p->left: ','    p->left->left: a       p->left->right: b
        */
#define a  p->left->left
#define b  p->left->right
            if (b->type == PT_CONSTANT) {
                arg1 = PTdifferentiate(a, varnum);

                newp = mkb(PT_TIMES,
                           mkb(PT_TIMES,
                               mkcon(b->constant),
                               mkf(PTF_POW,
                                   mkb(PT_COMMA, a, mkcon(b->constant - 1.0)))),
                           arg1);

            } else {
                arg1 = PTdifferentiate(a, varnum);
                arg2 = PTdifferentiate(b, varnum);
                newp = mkb(PT_TIMES,
                           mkf(PTF_PWR, mkb(PT_COMMA, a, b)),
                           mkb(PT_PLUS,
                               mkb(PT_TIMES, b, mkb(PT_DIVIDE, arg1, a)),
                               mkb(PT_TIMES, arg2, mkf(PTF_LOG, mkf(PTF_ABS, a)))));
            }
#ifdef TRACE
                printf("debug pwr, %s, returns; ", __func__);
                printTree(newp);
                printf("\n");
#endif
            return mkfirst(newp, p);
#undef b
#undef a
        }

        default:
            fprintf(stderr, "Internal Error: bad function # %d\n",
                    p->funcnum);
            return mkfirst(NULL, p);
        }

        arg2 = PTdifferentiate(p->left, varnum);

        newp = mkb(PT_TIMES, arg1, arg2);

        break;

    default:
        fprintf(stderr, "Internal error: bad node type %d\n", p->type);
        newp = NULL;
        break;
    }

    return mkfirst(newp, p);
}

static INPparseNode *mkcon(double value)
{
    INPparseNode *p = TMALLOC(INPparseNode, 1);

    p->type = PT_CONSTANT;
    p->constant = value;
    p->usecnt = 0;

    return (p);
}

static INPparseNode *mkb(int type, INPparseNode * left,
                         INPparseNode * right)
{
    INPparseNode *p;
    int i;

    if ((right->type == PT_CONSTANT) && (left->type == PT_CONSTANT)) {
        double value;
        switch (type) {
        case PT_TIMES:
            value = left->constant * right->constant;
            return mkfirst(mkcon(value), mkfirst(left, right));

        case PT_DIVIDE:
            value = left->constant / right->constant;
            return mkfirst(mkcon(value), mkfirst(left, right));

        case PT_PLUS:
            value = left->constant + right->constant;
            return mkfirst(mkcon(value), mkfirst(left, right));

        case PT_MINUS:
            value = left->constant - right->constant;
            return mkfirst(mkcon(value), mkfirst(left, right));

        case PT_POWER:
            value = pow(left->constant, right->constant);
            return mkfirst(mkcon(value), mkfirst(left, right));
        }
    }
    switch (type) {
    case PT_TIMES:
        if ((left->type == PT_CONSTANT) && (left->constant == 0))
            return mkfirst(left, right);
        else if ((right->type == PT_CONSTANT) && (right->constant == 0))
            return mkfirst(right, left);
        else if ((left->type == PT_CONSTANT) && (left->constant == 1))
            return mkfirst(right, left);
        else if ((right->type == PT_CONSTANT) && (right->constant == 1))
            return mkfirst(left, right);
        break;

    case PT_DIVIDE:
        if ((left->type == PT_CONSTANT) && (left->constant == 0))
            return mkfirst(left, right);
        else if ((right->type == PT_CONSTANT) && (right->constant == 1))
            return mkfirst(left, right);
        break;

    case PT_PLUS:
        if ((left->type == PT_CONSTANT) && (left->constant == 0))
            return mkfirst(right, left);
        else if ((right->type == PT_CONSTANT) && (right->constant == 0))
            return mkfirst(left, right);
        break;

    case PT_MINUS:
        if ((right->type == PT_CONSTANT) && (right->constant == 0))
            return mkfirst(left, right);
        else if ((left->type == PT_CONSTANT) && (left->constant == 0))
            return mkfirst(mkf(PTF_UMINUS, right), left);
        break;

    case PT_POWER:
        if (right->type == PT_CONSTANT) {
            if (right->constant == 0)
                return mkfirst(mkcon(1.0), mkfirst(left, right));
            else if (right->constant == 1)
                return mkfirst(left, right);
        }
        break;

    case PT_TERN:
        if (left->type == PT_CONSTANT) {
            /*FIXME > 0.0, >= 0.5, != 0.0 or what ? */
            p = (left->constant != 0.0) ? right->left : right->right;
            return mkfirst(p, mkfirst(right, left));
        }
        if((right->left->type == PT_CONSTANT) &&
           (right->right->type == PT_CONSTANT) &&
           (right->left->constant == right->right->constant))
            return mkfirst(right->left, mkfirst(right, left));
        break;
     }

     p = TMALLOC(INPparseNode, 1);

     p->type = type;
     p->usecnt = 0;

     p->left = inc_usage(left);
     p->right = inc_usage(right);

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
    INPparseNode *p;
    int i;

    for (i = 0; i < NUM_FUNCS; i++)
        if (funcs[i].number == type)
            break;
    if (i == NUM_FUNCS) {
        fprintf(stderr, "Internal Error: bad type %d\n", type);
        return (NULL);
    }

    if (arg->type == PT_CONSTANT) {
        double constval = PTunary(funcs[i].funcptr) (arg->constant);
        return mkfirst(mkcon(constval), arg);
    }

    p = TMALLOC(INPparseNode, 1);

    p->type = PT_FUNCTION;
    p->usecnt = 0;

    p->left = inc_usage(arg);

    p->funcnum = funcs[i].number;
    p->function = funcs[i].funcptr;
    p->funcname = funcs[i].name;

    p->data = NULL;

    return (p);
}

/* Check for remaining PT_PLACEHOLDERs in the parse tree.  Returns 1 if ok. 
   Returns 0 and error message containing expression to parsed, if not ok. */

static int PTcheck(INPparseNode * p, char *tline)
{
    int ret;
    static bool msgsent = FALSE;
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
        ret = (PTcheck(p->left, tline));
        if (ret == 0 && !msgsent) {
            fprintf(stderr, "\nError: The internal check of parse tree \n%s\nfailed\n", tline);
            msgsent = TRUE;
        }
        return ret;

    case PT_PLUS:
    case PT_MINUS:
    case PT_TIMES:
    case PT_DIVIDE:
    case PT_POWER:
    case PT_COMMA:
        ret = (PTcheck(p->left, tline) && PTcheck(p->right, tline));
        if (ret == 0 && !msgsent) {
            fprintf(stderr, "\nError: The internal check of parse tree \n%s\nfailed\n", tline);
            msgsent = TRUE;
        }
        return ret;
    case PT_TERN:
        ret = (PTcheck(p->left, tline) && PTcheck(p->right->left, tline) && PTcheck(p->right->right, tline));
        if (ret == 0 && !msgsent) {
            fprintf(stderr, "\nError: The internal check of parse tree \n%s\nfailed\n", tline);
            msgsent = TRUE;
        }
        return ret;

    default:
        fprintf(stderr, "Internal error: bad node type %d\n", p->type);
        return (0);
    }
}

/* Binop node. */

INPparseNode *PT_mkbnode(const char *opstr, INPparseNode * arg1,
                             INPparseNode * arg2)
{
    INPparseNode *p;
    int i;

    for (i = 0; i < NUM_OPS; i++)
        if (!strcmp(ops[i].name, opstr))
            break;

    if (i == NUM_OPS) {
        fprintf(stderr, "Internal Error: no such op num %s\n", opstr);
        return mkfirst(NULL, mkfirst(arg1, arg2));
    }

    p = TMALLOC(INPparseNode, 1);

    p->type = ops[i].number;
    p->usecnt = 0;

    p->funcname = ops[i].name;
    p->function = ops[i].funcptr;
    p->left = inc_usage(arg1);
    p->right = inc_usage(arg2);

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
        controlled_exit(1);
    }

#ifdef TRACE
    fprintf(stderr, "PWL-INFO: %s  building a PTF_PWL\n", __func__);
#endif
    i = 0;
    for(w = p->left; w->type == PT_COMMA; w = w->left)
        i++;

    if (i<2 || (i%1)) {
        fprintf(stderr, "Error: PWL(expr, points...) needs an even and >=2 number of constant args\n");
        return mkfirst(NULL, p);
    }

    data = TMALLOC(struct pwldata, 1);
    data->vals = TMALLOC(double, i);

    data->n = i;

    p->data = (void *) data;

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
            return mkfirst(NULL, p);
        }

#ifdef TRACE
    for (i = 0 ; i < data->n ; i += 2)
        fprintf(stderr, "  (%lf %lf)\n", data->vals[i], data->vals[i+1]);
#endif

    for (i = 2 ; i < data->n ; i += 2)
        if(data->vals[i-2] >= data->vals[i]) {
            fprintf(stderr, "Error: PWL(expr, points...) the abscissa of points must be ascending\n");
            return mkfirst(NULL, p);
        }

    /* strip all but the first arg,
     *   and attach the rest as opaque data to the INPparseNode
     */

    w = inc_usage(w);
    dec_usage(p->left);
    p->left = w;

    return (p);
}

static INPparseNode* prepare_PTF_DDT(INPparseNode* p)
{
    struct ddtdata { int n; double* vals; } *data;
    int i, ii;
    /* store 3 recent times and 3 recent values in pairs t0, v0, t1, v1, t2, v2  */
    i = 0;
    data = TMALLOC(struct ddtdata, 1);
    data->vals = TMALLOC(double, 7);
    for (ii = 0; ii < 7; ii++) {
        data->vals[ii] = 0;
    }
    p->data = (void*)data;
    return (p);
}

INPparseNode *PT_mkfnode(const char *fname, INPparseNode * arg)
{
    int i;
    INPparseNode *p;
    char buf[128];

    if (!fname) {
        fprintf(stderr, "Error: bogus function name \n");
        return mkfirst(NULL, arg);
    }

    if (!arg) {
        fprintf(stderr, "Error: bad function arguments \n");
        return mkfirst(NULL, arg);
    }

    /* Make sure the case is ok. */
    (void)strncpy(buf, fname, 127);
    buf[127] = 0;
    strtolower(buf);

    if(!strcmp("ternary_fcn", buf)) {

        if(arg->type == PT_COMMA && arg->left->type == PT_COMMA) {

            INPparseNode *arg1 = arg->left->left;
            INPparseNode *arg2 = arg->left->right;
            INPparseNode *arg3 = arg->right;

            p = TMALLOC(INPparseNode, 1);

            p->type = PT_TERN;
            p->usecnt = 0;

            p->left = inc_usage(arg1);
            p->right = inc_usage(mkb(PT_COMMA, arg2, arg3));

            return mkfirst(p, arg);
        }

        fprintf(stderr, "Error: bogus ternary_fcn form\n");
        return mkfirst(NULL, arg);
    }

    for (i = 0; i < NUM_FUNCS; i++)
        if (!strcmp(funcs[i].name, buf))
            break;

    if (i == NUM_FUNCS) {
        fprintf(stderr, "Error: no such function '%s'\n", buf);
        if (ft_stricterror)
            controlled_exit(EXIT_BAD);
        return mkfirst(NULL, arg);
    }

    p = TMALLOC(INPparseNode, 1);

    p->type = PT_FUNCTION;
    p->usecnt = 0;

    p->left = inc_usage(arg);
    p->funcname = funcs[i].name;
    p->funcnum = funcs[i].number;
    p->function = funcs[i].funcptr;
    p->data = NULL;

    if (p->funcnum == PTF_PWL) {
        p = prepare_PTF_PWL(p);
        if (p == NULL) {
            fprintf(stderr, "Error while parsing function '%s'\n", buf);
            if (ft_stricterror)
                controlled_exit(EXIT_BAD);
            return mkfirst(NULL, arg);
        }
    }

    if (p->funcnum == PTF_DDT)
        p = prepare_PTF_DDT(p);


    return (p);
}

static INPparseNode *mkvnode(char *name)
{
    INPparseNode *p = TMALLOC(INPparseNode, 1);

    int i;
    CKTnode *temp;

    INPtermInsert(circuit, &name, tables, &temp);
    for (i = 0; i < numvalues; i++)
        if ((types[i] == IF_NODE) && (values[i].nValue == temp))
            break;
    if (i == numvalues) {
        if (numvalues) {
            values = TREALLOC(IFvalue, values, numvalues + 1);
            types = TREALLOC(int, types, numvalues + 1);
        } else {
            values = TMALLOC(IFvalue, 1);
            types = TMALLOC(int, 1);
        }
        values[i].nValue = temp;
        types[i] = IF_NODE;
        numvalues++;
    }
    p->valueIndex = i;
    p->type = PT_VAR;
    p->usecnt = 0;

    return (p);
}

static INPparseNode *mkinode(char *name)
{
    INPparseNode *p = TMALLOC(INPparseNode, 1);

    int i;

    INPinsert(&name, tables);
    for (i = 0; i < numvalues; i++)
        if ((types[i] == IF_INSTANCE) && (values[i].uValue == name))
            break;
    if (i == numvalues) {
        if (numvalues) {
            values = TREALLOC(IFvalue, values, numvalues + 1);
            types = TREALLOC(int, types, numvalues + 1);
        } else {
            values = TMALLOC(IFvalue, 1);
            types = TMALLOC(int, 1);
        }
        values[i].uValue = name;
        types[i] = IF_INSTANCE;
        numvalues++;
    }
    p->valueIndex = i;
    p->type = PT_VAR;
    p->usecnt = 0;

    return (p);
}

/* Number node. */

INPparseNode *PT_mknnode(double number)
{
    struct INPparseNode *p;

    p = TMALLOC(INPparseNode, 1);

    p->type = PT_CONSTANT;
    p->usecnt = 0;

    p->constant = number;

    return (p);
}

/* String node. */

INPparseNode *PT_mksnode(const char *string, void *ckt)
{
    int i, j;
    char buf[128];
    INPparseNode *p;

    /* Make sure the case is ok. */
    (void) strncpy(buf, string, 127);
    buf[127] = 0;
    strtolower(buf);

    p = TMALLOC(INPparseNode, 1);

    p->usecnt = 0;

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
                values = TREALLOC(IFvalue, values, numvalues + 1);
                types = TREALLOC(int, types, numvalues + 1);
            } else {
                values = TMALLOC(IFvalue, 1);
                types = TMALLOC(int, 1);
            }
            values[i].sValue = TMALLOC(char, strlen(buf) + 1);
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
        p->funcname = copy(string);
    } else {
        p->type = PT_CONSTANT;
        p->constant = constants[i].value;
    }

    return (p);
}

/* The lexical analysis routine. */

int PTlex (YYSTYPE *lvalp, struct PTltype *llocp, char **line)
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

    llocp->start = sbuf;

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
    /* Don't parse the B source instance parameters, thus prevent memory leak.
       As soon as we meet such parameter, token=0 is returned. */
    case 't':
        if (ciprefix("tc1=", sbuf) || ciprefix("tc2=", sbuf) || ciprefix("temp=", sbuf)) {
            token = 0;
            break;
        }
        /* FALLTHROUGH */
    case 'd':
        if (ciprefix("dtemp=", sbuf)) {
            token = 0;
            break;
        }
        /* FALLTHROUGH */
    case 'r':
        if (ciprefix("reciproctc=", sbuf)) {
            token = 0;
            break;
        }
        /* FALLTHROUGH */
    default:
        {
            int n1 = -1;
            int n2 = -1;
            int n3 = -1;
            int n4 = -1;
            int n  = -1;

            sscanf(sbuf, "%*1[vV] ( %n%*[^ \t,()]%n , %n%*[^ \t,()]%n )%n",
                   &n1, &n2, &n3, &n4, &n);
            if(n != -1) {
                token = TOK_pnode;
                lvalp->pnode = mkb(PT_MINUS,
                                   mkvnode(copy_substring(sbuf+n1, sbuf+n2)),
                                   mkvnode(copy_substring(sbuf+n3, sbuf+n4)));
                sbuf += n;
                break;
            }
        }

        {
            int n1 = -1;
            int n2 = -1;
            int n  = -1;

            sscanf(sbuf, "%*1[vV] ( %n%*[^ \t,()]%n )%n", &n1, &n2, &n);
            if(n != -1) {
                token = TOK_pnode;
                lvalp->pnode = mkvnode(copy_substring(sbuf+n1, sbuf+n2));
                sbuf += n;
                break;
            }
        }

        {
            int n1 = -1;
            int n2 = -1;
            int n  = -1;

            sscanf(sbuf, "%*1[iI] ( %n%*[^ \t,()]%n )%n", &n1, &n2, &n);
            if(n != -1) {
                token = TOK_pnode;
                lvalp->pnode = mkinode(copy_substring(sbuf+n1, sbuf+n2));
                sbuf += n;
                break;
            }
        }

        td = INPevaluate(&sbuf, &err, 1);
        if (err == OK) {
            token = TOK_NUM;
            lvalp->num = td;
        } else {
        char *tmp;
            token = TOK_STR;
            for (s = sbuf; *s; s++)
                if (strchr(specials, *s))
                    break;
            tmp = TMALLOC(char, s - sbuf + 1);
            strncpy(tmp, sbuf, (size_t) (s - sbuf));
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
    llocp->stop = sbuf;
    return (token);
}


void INPfreeTree(IFparseTree *ptree)
{
    INPparseTree *pt = (INPparseTree *) ptree;

    int i;

    if (!pt)
        return;

    for (i = 0; i < pt->p.numVars; i++)
        dec_usage(pt->derivs[i]);

    dec_usage(pt->tree);

    txfree(pt->derivs);
    txfree(pt->p.varTypes);
    txfree(pt->p.vars);
    txfree(pt);
}


void free_tree(INPparseNode *pt)
{
    if(!pt)
        return;

    if(pt->usecnt) {
        fprintf(stderr, "ERROR: fatal internal error, %s\n", __func__);
        controlled_exit(1);
    }

    /* FALLTHROUGH added to suppress GCC warning due to
     * -Wimplicit-fallthrough flag */
    switch (pt->type) {
    case PT_TIME:
    case PT_TEMPERATURE:
    case PT_FREQUENCY:
    case PT_CONSTANT:
    case PT_VAR:
        break;

    case PT_PLUS:
    case PT_MINUS:
    case PT_TIMES:
    case PT_DIVIDE:
    case PT_POWER:
    case PT_COMMA:
    case PT_TERN:
        dec_usage(pt->right);
        /* FALLTHROUGH */
    case PT_FUNCTION:
        dec_usage(pt->left);
        break;

    default:
        printf("oops ");
        break;
    }

    if(pt->type == PT_FUNCTION && pt->funcnum == PTF_PWL) {
        struct pwldata { int n; double *vals; } *data = (struct pwldata*)(pt->data);
        if(data) {
            txfree(data->vals);
            txfree(data);
        }
    }

    if (pt->type == PT_FUNCTION && (pt->funcnum == PTF_DDT)) {
        struct ddtdata { int n; double* vals; } *data = (struct ddtdata*)(pt->data);
        if (data) {
            txfree(data->vals);
            txfree(data);
        }
    }

    txfree(pt);
}


/* Debugging stuff. */

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
        printf("oops ");
        break;
    }
    return;
}

