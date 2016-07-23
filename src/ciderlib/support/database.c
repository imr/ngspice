/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1992 David A. Gates, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/fteext.h"
/* #include "ftedata.h" */
#include "ngspice/cidersupt.h"

struct plot *
DBread( char *fileName )
{
  struct plot *plot;

  plot = raw_read( fileName );

  return(plot);
}

double *
DBgetData(struct plot *plot, char *name, int lengthWanted)
{
  struct dvec *v;
  double *data;
  int i;

  v = vec_fromplot(name,plot);

  if (!v) {
    fprintf( stderr, "Error: cannot locate variable '%s'\n", name );
    return(NULL);
  }
  if (v->v_length != lengthWanted ) {
    fprintf( stderr, "Error: vector '%s' has incorrect length\n", name );
    return(NULL);
  }

  data = TMALLOC(double, v->v_length);
  if (isreal(v)) {
    memcpy(data, v->v_realdata, sizeof (double) * (size_t) v->v_length);
  } else {
    for (i=0; i < v->v_length; i++) {
      data[i] = realpart(v->v_compdata[i]);
    }
  }
  return(data);
}

void
DBfree(struct plot *plot)
{
  struct dvec *v, *nextv;
  struct plot *pl, *nextpl;

  for (pl = plot; pl; pl = nextpl) {
    nextpl = pl->pl_next;
    tfree( pl->pl_title );
    tfree( pl->pl_date );
    tfree( pl->pl_name );
    tfree( pl->pl_typename );
    for (v = pl->pl_dvecs; v; v = nextv) {
      nextv = v->v_next;
      vec_free( v );
    }
    wl_free( pl->pl_commands );
    /* XXX Environment variables (pl->pl_env) will leak. */
  }
}
