/*************
 * Header file for evaluate.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_EVALUATE_H
#define ngspice_EVALUATE_H

#include "ngspice/dvec.h"
#include "ngspice/pnode.h"

struct dvec *op_plus(struct pnode *arg1, struct pnode *arg2);
struct dvec *op_minus(struct pnode *arg1, struct pnode *arg2);
struct dvec *op_comma(struct pnode *arg1, struct pnode *arg2);
struct dvec *op_times(struct pnode *arg1, struct pnode *arg2);
struct dvec *op_mod(struct pnode *arg1, struct pnode *arg2);
struct dvec *op_divide(struct pnode *arg1, struct pnode *arg2);
struct dvec *op_power(struct pnode *arg1, struct pnode *arg2);
struct dvec *op_eq(struct pnode *arg1, struct pnode *arg2);
struct dvec *op_gt(struct pnode *arg1, struct pnode *arg2);
struct dvec *op_lt(struct pnode *arg1, struct pnode *arg2);
struct dvec *op_ge(struct pnode *arg1, struct pnode *arg2);
struct dvec *op_le(struct pnode *arg1, struct pnode *arg2);
struct dvec *op_ne(struct pnode *arg1, struct pnode *arg2);
struct dvec *op_and(struct pnode *arg1, struct pnode *arg2);
struct dvec *op_or(struct pnode *arg1, struct pnode *arg2);
struct dvec *op_range(struct pnode *arg1, struct pnode *arg2);
struct dvec *op_ind(struct pnode *arg1, struct pnode *arg2);
struct dvec *op_uminus(struct pnode *arg);
struct dvec *op_not(struct pnode *arg);

#endif
