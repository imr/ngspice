/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/* Routines for dealing with the circuit database.  This is currently
 * unimplemented.  */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"
#include "circuits.h"


struct circ *ft_curckt = NULL;  /* The default active circuit. */
struct circ *ft_circuits = NULL;


/* Now stuff to deal with circuits */

/* Add a circuit to the circuit list */

void
ft_newcirc(struct circ *ci)
{
    ci->ci_next = ft_circuits;
    ft_circuits = ci;
}
