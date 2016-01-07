#include "ngspice/ngspice.h"
#include "ngspice/dvec.h"


struct dvec *
dvec_alloc(char *name, int type, short flags, int length, void *storage)
{
    struct dvec *rv = TMALLOC(struct dvec, 1);

    if (!rv)
        return NULL;

    ZERO(rv, struct dvec);

    rv->v_name = name;
    rv->v_type = type;
    rv->v_flags = flags;
    rv->v_length = length;
    rv->v_alloc_length = length;

    if (!length) {
        rv->v_realdata = NULL;
        rv->v_compdata = NULL;
    } else if (flags & VF_REAL) {
        rv->v_realdata = storage
            ? (double *) storage
            : TMALLOC(double, length);
        rv->v_compdata = NULL;
    } else if (flags & VF_COMPLEX) {
        rv->v_realdata = NULL;
        rv->v_compdata = storage
            ? (ngcomplex_t *) storage
            : TMALLOC(ngcomplex_t, length);
    }

    rv->v_plot = NULL;
    rv->v_scale = NULL;
    rv->v_numdims = 0;

    return rv;
}


void
dvec_realloc(struct dvec *v, int length, void *storage)
{
    if (isreal(v)) {
        if (storage) {
            tfree(v->v_realdata);
            v->v_realdata = (double *) storage;
        } else {
            v->v_realdata = TREALLOC(double, v->v_realdata, length);
        }
    } else {
        if (storage) {
            tfree(v->v_compdata);
            v->v_compdata = (ngcomplex_t *) storage;
        } else {
            v->v_compdata = TREALLOC(ngcomplex_t, v->v_compdata, length);
        }
    }

    v->v_length = length;
    v->v_alloc_length = length;
}


void
dvec_extend(struct dvec *v, int length)
{
    if (isreal(v))
        v->v_realdata = TREALLOC(double, v->v_realdata, length);
    else
        v->v_compdata = TREALLOC(ngcomplex_t, v->v_compdata, length);

    v->v_alloc_length = length;
}


void
dvec_trunc(struct dvec *v, int length)
{
    v->v_length = length;
}


void
dvec_free(struct dvec *v)
{
    if (v->v_name)
        tfree(v->v_name);
    if (v->v_realdata)
        tfree(v->v_realdata);
    if (v->v_compdata)
        tfree(v->v_compdata);
    tfree(v);
}
