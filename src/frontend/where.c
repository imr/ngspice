/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group 
**********/

#include "ngspice.h"
#include "cpdefs.h"
#include "ftedefs.h"
#include "dvec.h"
#include "ftehelp.h"
#include "hlpdefs.h"

#include "circuits.h"
#include "where.h"

void
com_where(void)
{
	char	*msg;

	if (ft_curckt) {
	    msg = (*ft_sim->nonconvErr)((void *) (ft_curckt->ci_ckt), 0);
	    fprintf(cp_out, "%s", msg);
	} else {
	    fprintf(cp_err, "Error: no circuit loaded.\n");
	}
}
