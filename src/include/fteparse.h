/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 *
 * Stuff for parsing -- used by the parser and in ft_evaluate().
 */

#ifndef FTEPARSE
#define FTEPARSE


#include "ftedata.h"

struct pnode {
    char *pn_name;		/* If non-NULL, the name. */
    struct dvec *pn_value;	/* Non-NULL in a terminal node. */
    struct func *pn_func;	/* Non-NULL is a function. */
    struct op *pn_op;		/* Operation if the above two NULL. */
    struct pnode *pn_left;	/* Left branch or function argument. */
    struct pnode *pn_right;	/* Right branch. */
    struct pnode *pn_next;	/* For expression lists. */
} ;

/* Operations. These should really be considered functions. */

struct op {
    int op_num;			/* From parser #defines. */
    char *op_name;		/* Printing name. */
    char op_arity;		/* One or two. */
    struct dvec *(*op_func)();  /* The function to do the work. */
} ;

/* The functions that are available. */

struct func {
    char *fu_name;		/* The print name of the function. */
    void *(*fu_func)();		/* The function. */
} ;

/* User-definable functions. The idea of ud_name is that the args are
 * kept in strings after the name, all seperated by '\0's. There
 * will be ud_arity of them.
 */

struct udfunc {
    char *ud_name;		/* The name. */
    int ud_arity;		/* The arity of the function. */
    struct pnode *ud_text;	/* The definition. */
    struct udfunc *ud_next;	/* Link pointer. */
} ;

#define MAXARITY    32

/* Parser elements. */

struct element {
    int e_token;		/* One of the below. */
    int e_type;			/* If the token is VALUE. */
    union  {
        char *un_string;
        double un_double;
        struct pnode *un_pnode;
    } e_un;
#define e_string    e_un.un_string
#define e_double    e_un.un_double
#define e_indices   e_un.un_indices
#define e_pnode     e_un.un_pnode
};

/* See the table in parse.c */

#define END 0
#define PLUS    1
#define MINUS   2
#define TIMES   3
#define MOD 4
#define DIVIDE  5
#define POWER   6
#define UMINUS  7
#define LPAREN  8
#define RPAREN  9
#define COMMA   10
#define VALUE   11
#define EQ   12
#define GT   13
#define LT   14
#define GE   15
#define LE   16
#define NE   17
#define AND   18
#define OR   19
#define NOT   20
#define INDX 21
#define RANGE   22

#define NUM 1
#define STRING  2
#define PNODE   3

#endif /* FTEPARSE */
