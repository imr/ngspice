#include "ngspice/ngspice.h"
#include "ngspice/dvec.h"


struct dvec *dvec_alloc(/* NOT const -- assigned to char */ char *name,
        int type, short flags, int length, void *storage)
{
    struct dvec * const rv = TMALLOC(struct dvec, 1);

    /* If the allocation failed, return NULL as a failure flag.
     * As of 2019-03, TMALLOC will not return on failure, so this check is
     * redundant, but it may be useful if it is decided to allow the
     * allocation functions to return NULL on failure and handle recovery
     * by the calling functions */
    if (!rv) {
        return NULL;
    }

    /* Set all fields to 0 */
    ZERO(rv, struct dvec);

    /* Set information on the vector from parameters. Note that storage for
     * the name string belongs to the dvec when this function returns. */
    rv->v_name = name;
    rv->v_type = type;
    rv->v_flags = flags;
    rv->v_length = length;
    rv->v_alloc_length = length;
    rv->v_numdims = 1; /* Assume 1 D */
    rv->v_dims[0] = length;

    if (length == 0) { /* Redundant due to ZERO() call above */
        rv->v_realdata = NULL;
        rv->v_compdata = NULL;
    }
    else if (flags & VF_REAL) {
        /* Vector consists of real data. Use the supplied storage if given
         * or allocate if not */
        rv->v_realdata = storage
            ? (double *) storage
            : TMALLOC(double, length);
        rv->v_compdata = NULL;
    }
    else if (flags & VF_COMPLEX) {
        /* Vector holds complex data. Perform actions as for real data */
        rv->v_realdata = NULL;
        rv->v_compdata = storage
            ? (ngcomplex_t *) storage
            : TMALLOC(ngcomplex_t, length);
    }

    /* Set remaining fields to none/unknown. Again not required due to
     * the ZERO() call */
    rv->v_plot = NULL;
    rv->v_scale = NULL;

    return rv;
} /* end of function dvec_alloc */


/* Resize dvec to length if storage is NULL orr replace
 * its existing allocation with storage if not
 */
void dvec_realloc(struct dvec *v, int length, void *storage)
{
    if (isreal(v)) {
        if (storage) {
            tfree(v->v_realdata);
            v->v_realdata = (double *) storage;
        }
        else {
            v->v_realdata = TREALLOC(double, v->v_realdata, length);
        }
    }
    else {
        if (storage) {
            tfree(v->v_compdata);
            v->v_compdata = (ngcomplex_t *) storage;
        }
        else {
            v->v_compdata = TREALLOC(ngcomplex_t, v->v_compdata, length);
        }
    }

    v->v_length = length;
    v->v_alloc_length = length;
} /* end of function dvec_realloc */


void dvec_extend(struct dvec *v, int length)
{
    if (isreal(v)) {
        v->v_realdata = TREALLOC(double, v->v_realdata, length);
    }
    else {
        v->v_compdata = TREALLOC(ngcomplex_t, v->v_compdata, length);
    }

    v->v_alloc_length = length;
} /* end of function dvec_extend */



void dvec_trunc(struct dvec *v, int length)
{
    /* Ensure valid */
    if (v->v_alloc_length <= length) {
        v->v_length = length;
    }
} /* end of function dvec_trunc */



void dvec_free(struct dvec *v)
{
    /* Check for freed vector */
    if (v == (struct dvec *) NULL) {
        return;
    }

    /* Free the various allocations */
    if (v->v_name) {
        txfree(v->v_name);
    }
    if (v->v_realdata) {
        txfree(v->v_realdata);
    }
    else if (v->v_compdata) { /* if data real, not complex */
        txfree(v->v_compdata);
    }
    txfree(v);
} /* end of function dvec_free */



