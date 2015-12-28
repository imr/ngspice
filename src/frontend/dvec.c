#include "ngspice/ngspice.h"
#include "ngspice/dvec.h"


struct dvec *
dvec_alloc(void)
{
    struct dvec *rv = TMALLOC(struct dvec, 1);

    if (!rv)
        return NULL;

    ZERO(rv, struct dvec);

    rv->v_plot = NULL;
    rv->v_scale = NULL;
    rv->v_numdims = 0;

    return rv;
}
