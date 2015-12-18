#include <stddef.h>

#include "ngspice/dvec.h"
#include "ngspice/ngspice.h"
#include "ngspice/fteext.h"
#include "ngspice/cpextern.h"
#include "ngspice/sim.h"
#include "ngspice/stringskip.h"

#include "com_let.h"
#include "com_display.h"
#include "completion.h"


void
com_let(wordlist *wl)
{
    char *p, *q, *s;
    int indices[MAXDIMS];
    int numdims;
    wordlist fake_wl;
    int need_open;
    int offset, length;
    struct pnode *names;
    struct dvec *n, *t;
    int i, cube;
    int j, depth;
    int newvec;
    char *rhs;

    fake_wl.wl_next = NULL;

    if (!wl) {
        com_display(NULL);
        return;
    }

    p = wl_flatten(wl);

    /* extract indices */
    numdims = 0;
    if ((rhs = strchr(p, '=')) != NULL) {
        *rhs++ = '\0';
    } else {
        fprintf(cp_err, "Error: bad let syntax\n");
        tfree(p);
        return;
    }

    if ((s = strchr(p, '[')) != NULL) {
        need_open = 0;
        *s++ = '\0';
        while (!need_open || *s == '[') {
            depth = 0;
            if (need_open)
                s++;
            for (q = s; *q && (*q != ']' && (*q != ',' || depth > 0)); q++) {
                switch (*q) {
                case '[':
                    depth += 1;
                    break;
                case ']':
                    depth -= 1;
                    break;
                }
            }

            if (depth != 0 || !*q) {
                printf("syntax error specifying index\n");
                tfree(p);
                return;
            }

            if (*q == ']')
                need_open = 1;
            else
                need_open = 0;

            if (*q)
                *q++ = '\0';

            /* evaluate expression between s and q */
            /* va, indexing */
            fake_wl.wl_word = s;
            names = ft_getpnames(&fake_wl, TRUE);
            if (!names) {
                /* XXX error message */
                tfree(p);
                return;
            }
            t = ft_evaluate(names);
            if (!t) {
                fprintf(cp_err, "Error: Can't evaluate %s\n", s);
                free_pnode(names);
                tfree(p);
                return;
            }
            if (!isreal(t) || t->v_link2 || t->v_length != 1 || !t->v_realdata) {
                fprintf(cp_err, "Error: index is not a scalar.\n");
                goto quit;
            }
            j = (int)floor(t->v_realdata[0]+0.5); /* ignore sanity checks for now, va, which checks? */

            if (j < 0) {
                printf("negative index (%d) is not allowed\n", j);
                goto quit;
            }

            indices[numdims++] = j;

            /* va: garbage collection for t, if pnode `names' is no simple value */
            if (names && !names->pn_value && t)
                vec_free(t);
            free_pnode(names); /* frees also t, if pnode `names' is simple value */

            s = skip_ws(q);
        }
    }
    /* vector name at p */

    for (q = p + strlen(p) - 1; *q <= ' ' && p <= q; q--)
        ;

    *++q = '\0';

    /* sanity check */
    if (eq(p, "all") || strchr(p, '@') || isdigit_c(*p)) {
        fprintf(cp_err, "Error: bad variable name %s\n", p);
        tfree(p);
        return;
    }

    /* evaluate rhs */

    /* first we check for
    let xyz = []   or
    let xyz = [3 -5.7 0.6]
    If found, xyz is created or overwritten by a new vector, either empty
    with length 0, or with length and values given by [3 -5.7 0.6] */
    char *br = skip_ws(rhs);
    if (*br == '[') {
        /* we may have [...] */
        char *cr, *tok;
        int ii = 0, error;
        double *doublevec = NULL;
        br++;
        if ((cr = strchr(rhs, ']')) != NULL) {
            *cr = '\0';
            tok = gettok(&br);
            if (tok == NULL) {
                /* We have [] and generate an empty vector */
                /* remove the old vector 'p' */
                vec_remove(p);
                /* create and assign the new vector 'p' */
                n = dvec_alloc(copy(p),
                               SV_NOTYPE,
                               VF_REAL | VF_PERMANENT,
                               0, NULL);
                vec_new(n);
                tfree(p);
                return;
            }
            else {
                while (tok != NULL) {
                    ii++;
                    doublevec = TREALLOC(double, doublevec, ii);
                    doublevec[ii - 1] = INPevaluate(&tok, &error, 1);
                    if (error) {
                        fprintf(cp_err, "Cannot evaluate token %s]\n", rhs);
                        tfree(p);
                        return;
                    }
                    tok = gettok(&br);
                }
                /* We have [ 2 -4 6.7 ] and generate corresponding vector */
                /* remove the old vector 'p' */
                vec_remove(p);
                /* create and assign the new vector 'p' */
                n = dvec_alloc(copy(p),
                               SV_NOTYPE,
                               VF_REAL | VF_PERMANENT,
                               ii, doublevec);
                vec_new(n);
                tfree(p);
                return;
            }
        }
    }
    fake_wl.wl_word = rhs;
    names = ft_getpnames(&fake_wl, TRUE);
    if (names == NULL) {
        /* error message from ft_getpnames() */
        tfree(p);
        return;
    }
    t = ft_evaluate(names);
    if (!t) {
        fprintf(cp_err, "Error: Can't evaluate %s\n", rhs);
        free_pnode(names);
        tfree(p);
        return;
    }

    if (t->v_link2)
        fprintf(cp_err, "Warning: extra wildcard values ignored\n");

    n = vec_get(p);

    if (n) {
        /* re-allocate? */
        /* vec_free(n); */
        newvec = 0;
    } else {
        if (numdims) {
            fprintf(cp_err, "Can't assign into a subindex of a new vector\n");
            goto quit;
        }

        /* create and assign a new vector */
        n = dvec_alloc(copy(p),
                       t->v_type,
                       t->v_flags | VF_PERMANENT,
                       t->v_length, NULL);

        if ((t->v_numdims) <= 1) { // changed from "!t->v_numdims" by Friedrich Schmidt
            n->v_numdims = 1;
            n->v_dims[0] = n->v_length;
        } else {
            n->v_numdims = t->v_numdims;
            for (i = 0; i < t->v_numdims; i++)
                n->v_dims[i] = t->v_dims[i];
        }

        newvec = 1;
        vec_new(n);
    }

    /* fix-up dimensions; va, also for v_dims */
    if (n->v_numdims < 1 || n->v_dims[0] == 0 ) {
        n->v_numdims = 1;
        n->v_dims[0] = n->v_length;
    }

    /* Compare dimensions */
    offset = 0;
    length = n->v_length;

    cube = 1;
    for (i = n->v_numdims - 1; i >= numdims; i--)
        cube *= n->v_dims[i];

    for (i = numdims - 1; i >= 0; i--) {
        offset += cube * indices[i];
        if (i < n->v_numdims) {
            cube *= n->v_dims[i];
            length /= n->v_dims[i];
        }
    }

    /* length is the size of the unit refered to */
    /* cube ends up being the length */

    if (length > t->v_length) {
        fprintf(cp_err, "left-hand expression is too small (need %d)\n",
                length * cube);
        if (newvec)
            n->v_flags &= ~VF_PERMANENT;
        goto quit;
    }
    if (isreal(t) != isreal(n)) {
        fprintf(cp_err,
                "Types of vectors are not the same (real vs. complex)\n");
        if (newvec)
            n->v_flags &= ~VF_PERMANENT;
        goto quit;
    } else if (isreal(t)) {
        memcpy(n->v_realdata + offset, t->v_realdata,
              (size_t) length * sizeof(double));
    } else {
        memcpy(n->v_compdata + offset, t->v_compdata,
              (size_t) length * sizeof(ngcomplex_t));
    }

    n->v_minsignal = 0.0; /* How do these get reset ??? */
    n->v_maxsignal = 0.0;

    n->v_scale = t->v_scale;

    if (newvec)
        cp_addkword(CT_VECTOR, n->v_name);

quit:
    /* va: garbage collection for t, if pnode `names' is no simple value */
    if (names && !names->pn_value && t)
        vec_free(t);
    free_pnode(names); /* frees also t, if pnode `names' is simple value */
    tfree(p);
}
