/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
**********/

/*
 * Some new post-processor commands having to do with vectors.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/fteparse.h"
#include "ngspice/dvec.h"
#include "ngspice/stringskip.h"

#include "newcoms.h"


/*
 *      reshape v(1) vxx#branch [10]
 *      reshape v(1) vxx#branch [10,4]
 *      reshape v(1) [,4]
 */

void
com_reshape(wordlist *wl)
{
    wordlist    *w, *w2, *wlast, *wsave;
    char        *p;
    struct dvec *dv, *d;
    int         numdims;
    int         *dims;
    int         local_dims[MAXDIMS];
    int         state;
    int         empty;
    int         err;
    int         missing, nprod, prod;
    char        *vname;
    int         i;

    do {
        if (!wl)
            return;

        /* find the first '[' */

        p = NULL;
        for (w = wl; w; w = w->wl_next) {
            if ((p = strchr(w->wl_word, '[')) != NULL)
                break;
        }

        if (p && *p) {
            if (p != w->wl_word)
                w = w->wl_next;
            wlast = w;
            *p++ = '\0';
        } else {
            wlast = NULL;
        }

        /* get the dimensions */
        dims = local_dims;
        numdims = 0;
        state = 0;
        empty = -1;
        err = 0;
        wsave = NULL;
        do {

            if (!p || !*p) {
                if (!wlast)
                    break;
                p = wlast->wl_word;
                if (state == 2)
                    wsave = wlast;
                else
                    wsave = NULL;
                wlast = wlast->wl_next;
            }

            p = skip_ws(p);

            switch (state) {
            case 0: /* p just at or before a number */

                if (numdims >= MAXDIMS) {
                    if (numdims == MAXDIMS)
                        printf("Maximum of %d dimensions possible\n", MAXDIMS);
                    numdims += 1;
                } else if (!isdigit_c(*p)) {
                    if (empty > -1) {
                        printf("dimensions underspecified at dimension %d\n",
                               numdims++);
                        err = 1;
                    } else {
                        empty = numdims;
                        dims[numdims++] = 1;
                    }
                } else {
                    dims[numdims++] = atoi(p);
                    while (isdigit_c(*p))
                        p++;
                }
                state = 1;
                break;

            case 1: /* p after a number, looking for ',' or ']' */
                if (*p == ']') {
                    p++;
                    state = 2;
                } else if (*p == ',') {
                    p++;
                    state = 0;
                } else if (isdigit_c(*p)) {
                    state = 0;
                    break;
                } else if (!isspace_c(*p)) {
                    /* error */
                    state = 4;
                }
                break;

            case 2: /* p after a ']', either at the end or looking for '[' */
                if (*p == '[') {
                    p++;
                    state = 0;
                } else {
                    state = 3;
                }
            }

            p = skip_ws(p);

        } while (state < 3);

        if (state == 2) {
            wlast = wsave;
        } else if ((state == 4 || state < 2) && ((state != 0 || p) && *p)) {
            printf("syntax error specifying dimensions\n");
            return;
        }

        if (numdims > MAXDIMS)
            continue;
        if (err)
            continue;

        /* Copy dimensions from the first item if none are explicitly given */
        if (!numdims) {
            /* Copy from the first */
            vname = cp_unquote(wl->wl_word);
            dv = vec_get(vname);
            if (!dv) {
                printf("'%s' dimensions vector not found\n", vname);
                return;
            }
            numdims = dv->v_numdims;
            dims = dv->v_dims;
            wl = wl->wl_next;
            empty = -1; /* just in case */
        }

        prod = 1;
        for (i = 0; i < numdims; i++)
            prod *= dims[i];

        /* resize each vector */
        for (w2 = wl; w2 && w2 != w; w2 = w2->wl_next) {
            vname = cp_unquote(w2->wl_word);

            dv = vec_get(vname);
            if (!dv) {
                printf("'%s' vector not found\n", vname);
                continue;
            }

            /* The name may expand to several vectors */
            for (d = dv; d; d = d->v_link2) {
                nprod = 1;
                for (i = 0; i < d->v_numdims; i++)
                    nprod *= d->v_dims[i];
                if (nprod != d->v_length) {
                    printf("dimensions of \"%s\" were inconsistent\n",
                           d->v_name);
                    nprod = d->v_length;
                }

                missing = nprod / prod;
                if (missing * prod != nprod) {
                    printf("dimensions don't fit \"%s\" (total size = %d)\n",
                           d->v_name, nprod);
                    continue;
                }

                if (missing > 1 && empty < 0) {
                    /* last dimension unspecified */
                    d->v_numdims = numdims + 1;
                    d->v_dims[numdims] = missing;
                } else {
                    d->v_numdims = numdims;
                }

                /* fill in dimensions */
                for (i = 0; i < numdims; i++) {
                    if (i == empty)
                        d->v_dims[i] = missing;
                    else
                        d->v_dims[i] = dims[i];
                }
            }

            if (vname)
                tfree(vname);
        }
    } while ((wl = wlast) != NULL);
}
