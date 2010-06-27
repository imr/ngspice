/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1992 David A. Gates, U. C. Berkeley CAD Group
**********/

#include "ngspice.h"
#include "fteext.h"
/* #include "ftedata.h" */

struct plot *
DBread( fileName )
{
  struct plot *plot;

  plot = raw_read( fileName );

  return(plot);
}

double *
DBgetData( plot, name, lengthWanted )
struct plot *plot;
char *name;
int lengthWanted;
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

  data = (double *) tmalloc(sizeof (double) * v->v_length);
  if (isreal(v)) {
    bcopy(v->v_realdata, data, sizeof (double) * v->v_length);
  } else {
    for (i=0; i < v->v_length; i++) {
      data[i] = realpart(&v->v_compdata[i]);
    }
  }
  return(data);
}

void
DBfree( plot )
struct plot *plot;
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
