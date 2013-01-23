/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 *
 * Stuff for parsing -- used by the parser and in ft_evaluate().
 */

#ifndef ngspice_FTEPARSE_H
#define ngspice_FTEPARSE_H


#include "ngspice/cpstd.h"
#include "ngspice/dvec.h"
#include "ngspice/plot.h"

/* FIXME: Split this file and adjust all callers. */
#if 0
#warning "Please use a more specific header than fteparse.h"
#endif
#include "ngspice/pnode.h"

/* Operations. These should really be considered functions. */

struct op {
    int op_num;			/* From parser #defines. */
    char *op_name;		/* Printing name. */
    char op_arity;		/* One or two. */
    union {
        void (*anonymous)(void);
        struct dvec *(*unary)(struct pnode *);
        struct dvec *(*binary)(struct pnode *, struct pnode *);
    } op_func;  /* The function to do the work. */
} ;

/* The functions that are available. */

struct func {
    /* The print name of the function. */
    char *fu_name;

    /* The function. */
    void *(*fu_func)(void *data, short int type, int length,
		     int *newlength, short int *newtype);
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

#define PT_OP_END      0
#define PT_OP_PLUS     1
#define PT_OP_MINUS    2
#define PT_OP_TIMES    3
#define PT_OP_MOD      4
#define PT_OP_DIVIDE   5
#define PT_OP_POWER    6
#define PT_OP_UMINUS   7
#define PT_OP_LPAREN   8
#define PT_OP_RPAREN   9
#define PT_OP_COMMA    10
#define PT_OP_VALUE    11
#define PT_OP_EQ       12
#define PT_OP_GT       13
#define PT_OP_LT       14
#define PT_OP_GE       15
#define PT_OP_LE       16
#define PT_OP_NE       17
#define PT_OP_AND      18
#define PT_OP_OR       19
#define PT_OP_NOT      20
#define PT_OP_INDX     21
#define PT_OP_RANGE    22
#define PT_OP_TERNARY  23


#endif
