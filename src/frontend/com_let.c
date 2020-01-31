#include <stddef.h>
#include <limits.h>

#include "ngspice/bool.h"
#include "ngspice/cpextern.h"
#include "ngspice/dvec.h"
#include "ngspice/fteext.h"
#include "ngspice/ngspice.h"
#include "ngspice/stringskip.h"

#include "com_display.h"
#include "com_let.h"
#include "completion.h"

/* Range of index values, such as 2:3 */
typedef struct index_range {
    int low;
    int high;
} index_range_t;

static void copy_vector_data(struct dvec *vec_dst,
        const struct dvec *vec_src);
static void copy_vector_data_with_stride(struct dvec *vec_dst,
        const struct dvec *vec_src,
        int n_dst_index, const index_range_t *p_dst_index);
static int find_indices(char *s, index_range_t *p_index, int *p_n_index);
static int get_index_values(char *s, index_range_t *p_range);
int get_one_index_value(char *s, int *p_index);

/* let <vec_name> = <expr>
 * let <vec_name>[<bracket_expr>] = <expr>
 *      <bracket_expr> = <index_expr> <sep> <index_expr> <sep> ...
 *                              <index_expr>
 *      <index_expr> = <expr> | <expr> : <expr>
 *      <sep> = "," | "] <ws> ["
 *      <expr> = standard ngspice expression
 */
void com_let(wordlist *wl)
{
    char *p, *s;
    index_range_t p_dst_index[MAXDIMS];
    int n_dst_index;
    struct pnode *names = (struct pnode *) NULL;
    struct dvec *vec_src = (struct dvec *) NULL;
    char *rhs;


    /* let with no arguments is equivalent to display */
    if (!wl) {
        com_display(NULL);
        return;
    }

    p = wl_flatten(wl); /* Everything after let -> string */

    /* Separate vector name from RHS of assignment */
    n_dst_index = 0;
    if ((rhs = strchr(p, '=')) == (char *) NULL) {
        fprintf(cp_err, "Error: bad let syntax\n");
        txfree(p);
        return;
    }
    *rhs++ = '\0';

    /* Handle indexing. At start, p = LHS; rhs = RHS. If index is found
     * p = leftmost part of orig p up to first '['. So p always
     * becomes the vector name, possibly with some spaces at the end. */
    if ((s = strchr(p, '[')) != NULL) {
        *s = '\0';
        if (find_indices(s + 1, p_dst_index, &n_dst_index) != 0) {
            txfree(p);
            return;
        }
    } /* end of case that an indexing bracket '[' was found */


    /* "Remove" any spaces at the end of the vector name at p */
    {
        char *q;
        for (q = p + strlen(p) - 1; *q <= ' ' && p <= q; q--) {
            ;
        }
        *++q = '\0';
    }

    /* Sanity check */
    if (eq(p, "all") || strchr(p, '@') || *p == '\0' || isdigit_c(*p)) {
        fprintf(cp_err, "Error: bad variable name \"%s\"\n", p);
        goto quit;
    }

    /* Evaluate rhs */
    names = ft_getpnames_from_string(rhs, TRUE);
    if (names == (struct pnode *) NULL) {
        fprintf(cp_err, "Error: RHS \"%s\" invalid\n", rhs);
        goto quit;
    }
    vec_src = ft_evaluate(names);
    if (!vec_src) {
        fprintf(cp_err, "Error: Can't evaluate \"%s\"\n", rhs);
        goto quit;
    }

    if (vec_src->v_link2) {
        fprintf(cp_err, "Warning: extra wildcard values ignored\n");
    }

    /* Fix-up dimension count and limit. Sometimes these are
     * not set properly. If not set, make 1-d vector and ensure
     * the right length */
    if (vec_src->v_numdims < 1) {
        vec_src->v_numdims = 1;
    }
    if (vec_src->v_numdims == 1) {
        vec_src->v_dims[0] = vec_src->v_length;
    }

    /* Locate the vector being assigned values. If NULL, the vector
     * does not exist */
    struct dvec * vec_dst = vec_get(p);

    if (vec_dst == (struct dvec *) NULL) {
        /* p is not an existing vector. So make a new one equal to vec_src
         * in all ways, except enforce that it is a permanent vector. */
        if (n_dst_index > 0) {
            fprintf(cp_err,
                    "When creating a new vector, it cannot be indexed.\n");
            goto quit;
        }

        /* Create and assign a new vector */
        vec_dst = dvec_alloc(copy(p),
                vec_src->v_type,
                vec_src->v_flags | VF_PERMANENT,
                vec_src->v_length, NULL);

        copy_vector_data(vec_dst, vec_src);
        vec_new(vec_dst); /* Add tp current plot */
        cp_addkword(CT_VECTOR, vec_dst->v_name);
    } /* end of case of new vector */
    else {
        /* Existing vector.*/
        /* Fix-up dimension count and limit. Sometimes these are
         * not set properly. If not set, make 1-d vector and ensure
         * the right length */
        if (vec_dst->v_numdims < 1) {
            vec_dst->v_numdims = 1;
        }
        if (vec_dst->v_numdims == 1) {
            vec_dst->v_dims[0] = vec_dst->v_length;
        }

        if (n_dst_index == 0) {
            /* Not indexed, so make equal to source vector as if it
             * was a new vector, except reuse the allocation if it
             * is the same type (real/complex) and the allocation size
             * is sufficient but not too large (>2X) . */
            if (isreal(vec_dst) == isreal(vec_src) &&
                    vec_dst->v_alloc_length >= vec_src->v_length &&
                    vec_dst->v_alloc_length <= 2 * vec_src->v_length) {
                vec_dst->v_length = vec_src->v_length;
                copy_vector_data(vec_dst, vec_src);
            }
            else { /* Something not OK, so free and allocate again */
                int n_elem_alloc = vec_src->v_alloc_length;
                if (isreal(vec_dst)) {
                    tfree(vec_dst->v_realdata);
                }
                else { /* complex */
                    tfree(vec_dst->v_compdata);
                }
                if (isreal(vec_src)) {
                    vec_dst->v_realdata = TMALLOC(double, n_elem_alloc);
                }
                else { /* complex source */
                    vec_dst->v_compdata = TMALLOC(ngcomplex_t, n_elem_alloc);
                }

                /* Make the destination vector the right data  type. A few
                 * extra () added to keep some compilers from warning. */
                vec_dst->v_flags =
                        (vec_dst->v_flags & ~(VF_REAL | VF_COMPLEX)) |
                        (vec_src->v_flags & (VF_REAL | VF_COMPLEX));
                vec_dst->v_alloc_length = vec_src->v_alloc_length;
                vec_dst->v_length = vec_src->v_length;
                copy_vector_data(vec_dst, vec_src);
            }
        }
        /* Else indexed. In this case, the source data must fit the indexed
         * range */
        else {
            {
                int n_dst_elem = 1;
                int i;
                for (i = 0; i < n_dst_index; ++i) {
                    index_range_t *p_range_cur = p_dst_index + i;
                    n_dst_elem *= p_range_cur->high - p_range_cur->low + 1;
                }

                /* Check # elem required vs available */
                if (n_dst_elem != vec_src->v_length) {
                    const int v_length = vec_src->v_length;
                    const bool f_1 = v_length == 1;
                    (void) fprintf(cp_err, "Data for an index vector must "
                            "fit exactly. The indexed range required %d "
                            "element%s to fill it, but there %s %d "
                            "element%s supplied.\n",
                            n_dst_elem, n_dst_elem == 1 ? "" : "s",
                            f_1 ? "was" : "were", v_length, f_1 ? "" : "s");
                    goto quit;
                }
            }

            /* Real source data can be put into a complex destination,
             * but the other way around is not possible */
            if (isreal(vec_dst) && iscomplex(vec_src)) {
                (void) fprintf(cp_err, "Complex data cannot be used "
                        "to fill an array of real data.\n");
                goto quit;
            }

            /* Check dimension numbers */
            if (n_dst_index != vec_dst->v_numdims) {
                fprintf(cp_err, "Number of vector indices given (%d) "
                        "does not match the dimension of the vector (%d).\n",
                        n_dst_index, vec_dst->v_numdims);
                goto quit;
            }

            /* Check dimension ranges */
            {
                int i;
                int *vec_dst_dims = vec_dst->v_dims;
                for (i = 0; i < n_dst_index; ++i) {
                    const int n_dst_cur = vec_dst_dims[i];
                    if (p_dst_index[i].high >= n_dst_cur) {
                        fprintf(cp_err,
                                "Vector index %d out of range (%d).\n",
                                i + 1, n_dst_cur);
                        goto quit;
                    }
                } /* end of loop over dimensions */
            }

            /* OK to copy, so copy */
            copy_vector_data_with_stride(vec_dst, vec_src,
                    n_dst_index, p_dst_index);
        } /* end of indexed vector */
    } /* end of existing vector */

    vec_dst->v_minsignal = 0.0; /* How do these get reset ??? */
    vec_dst->v_maxsignal = 0.0;
    vec_dst->v_scale = vec_src->v_scale;

quit:
    /* va: garbage collection for vec_src, if ft_evaluate() created a
     * new vector while evaluating pnode `names' */
    if (names != (struct pnode *) NULL) {
        if (!names->pn_value && vec_src) {
            vec_free(vec_src);
        }
        /* frees also vec_src, if pnode `names' is simple value */
        free_pnode(names);
    }
    txfree(p);
} /* end of function com_let */



/* Process indexing portion of a let command. On entry, s is the address
 * of the first byte after the first opening index bracket */
static int find_indices(char *s, index_range_t *p_index, int *p_n_index)
{
    /* Can be either comma-separated or individual dimensions */
    if (strchr(s, ',') != 0) { /* has commas */
        char *p_end;
        int dim_cur = 0;
        const int dim_max = MAXDIMS - 1;
        while ((p_end = strchr(s, ',')) != (char *) NULL) {
            *p_end = '\0';
            if (dim_cur == dim_max) {
                (void) fprintf(cp_err, "Too many dimensions given.\n");
                return -1;
            }
            if (get_index_values(s, p_index + dim_cur) != 0) {
                (void) fprintf(cp_err, "Dimension ranges "
                        "for dimension %d could not be found.\n",
                        dim_cur + 1);
                return -1;
            }
            ++dim_cur;
            s = p_end + 1; /* after (former) comma */
        } /* end of loop over comma-separated indices */

        /* Must be one more index ending with a bracket */
        if ((p_end = strchr(s, ']')) == (char *) NULL) {
            (void) fprintf(cp_err,
                    "Final dimension was not found.\n");
            return -1;
        }

        *p_end = '\0';
        if (dim_cur == dim_max) {
            (void) fprintf(cp_err,
                    "Final dimension exceded maximum number.\n");
            return -1;
        }
        if (get_index_values(s, p_index + dim_cur) != 0) {
            (void) fprintf(cp_err, "Dimension ranges "
                    "for last dimension (%d) could not be found.\n",
                    dim_cur + 1);
            return -1;
        }
        ++dim_cur;
        s = p_end + 1;

        /* Only white space is allowed after closing brace */
        if (*(s = skip_ws(s)) != '\0') {
            (void) fprintf(cp_err, "Invalid text was found "
                    "after dimension data for vector: \"%s\".\n",
                    s);
            return -1;
        }

        *p_n_index = dim_cur;
        return 0;
    } /* end of case x[ , , ] */
    else { /* x[][][] */
        char *p_end;
        int dim_cur = 0;
        const int dim_max = MAXDIMS - 1;
        while ((p_end = strchr(s, ']')) != (char *) NULL) {
            *p_end = '\0';
            if (dim_cur == dim_max) {
                (void) fprintf(cp_err, "Too many dimensions given.\n");
                return -1;
            }
               if (get_index_values(s, p_index + dim_cur) != 0) {
                (void) fprintf(cp_err, "Dimension ranges "
                        "for dimension %d could not be found.\n",
                        dim_cur + 1);
                return -1;
            }
            ++dim_cur;
            s = p_end + 1; /* after (former) ']' */
            if (*(s = skip_ws(s)) == '\0') { /* reached end */
                *p_n_index = dim_cur;
                return 0;
            }

            /* Not end of expression, so must be '[' */
            if (*s != '[') {
                (void) fprintf(cp_err, "Dimension bracket '[' "
                        "for dimension %d could not be found.\n",
                        dim_cur + 1);
                return -1;
            }
            s++; /* past '[' */
        } /* end of loop over individual bracketed entries */

        /* Did not find a single ']' in the string */
        (void) fprintf(cp_err, "The ']' for dimension 1 "
                "could not be found.\n");
        return -1;
    } /* end of case x[][][][] */
} /* end of function find_indices */



/* Convert expresion expr -> low and high ranges equal or
 * expression expr1 : epr2 -> low = expr1 and high = expr2.
 * Values are tested to ensure they are positive and that the low
 * value does not exceed the high value. Since the extent of the index
 * is not known, that cannot be checked. */
static int get_index_values(char *s, index_range_t *p_range)
{
    char *p_colon;
    if ((p_colon = strchr(s, ':')) == (char *) NULL) { /* One expression */
        if (get_one_index_value(s, &p_range->low) != 0) {
            (void) fprintf(cp_err, "Error geting index.\n");
            return -1;
        }
        p_range->high = p_range->low;
    }
    else { /* l:h */
        *p_colon = '\0';
        if (get_one_index_value(s, &p_range->low) != 0) {
            (void) fprintf(cp_err, "Error geting low range.\n");
            return -1;
        }
        s = p_colon + 1; /* past (former) colon */
        if (get_one_index_value(s, &p_range->high) != 0) {
            (void) fprintf(cp_err, "Error geting high range.\n");
            return -1;
        }
        if (p_range->low > p_range->high) {
            (void) fprintf(cp_err, "Error low range (%d) is greater "
                    "than high range (%d).\n",
                    p_range->low, p_range->high);
            return -1;
        }
    }
    return 0;
} /* end of function get_index_values */



/* Get an index value */
int get_one_index_value(char *s, int *p_index)
{
    /* Parse the expression */
    struct pnode * const names = ft_getpnames_from_string(s, TRUE);
    if (names == (struct pnode *) NULL) {
        (void) fprintf(cp_err, "Unable to parse index expression.\n");
        return -1;
    }

    /* Evaluate the parsing */
    struct dvec * const t = ft_evaluate(names);
    if (t == (struct dvec *) NULL) {
        (void) fprintf(cp_err, "Unable to evaluate index expression.\n");
        free_pnode_x(names);
        return -1;
    }

    int xrc = 0;
    if (t->v_link2 || t->v_length != 1 || !t->v_realdata) {
        fprintf(cp_err, "Index expression is not a real scalar.\n");
        xrc = -1;
    }
    else {
        const int index = (int) floor(t->v_realdata[0] + 0.5);
        if (index < 0) {
            printf("Negative index (%d) is not allowed.\n", index);
            xrc = -1;
        }
        else { /* index found ok */
            *p_index = index;
        }
    }

    /* Free resources */
    if (names->pn_value != (struct dvec *) NULL) {
        /* allocated value given to t */
        vec_free_x(t);
    }
    free_pnode_x(names);

    return xrc;
    } /* end of function get_one_index_value */



/* Copy vector data and its metadata */
static void copy_vector_data(struct dvec *vec_dst,
        const struct dvec *vec_src)
{
    const size_t length = (size_t) vec_src->v_length;
    int n_dim = vec_dst->v_numdims = vec_src->v_numdims;
    (void) memcpy(vec_dst->v_dims, vec_src->v_dims,
            n_dim * sizeof(int));
    if (isreal(vec_src)) {
        (void) memcpy(vec_dst->v_realdata, vec_src->v_realdata,
              length * sizeof(double));
    }
    else {
        (void) memcpy(vec_dst->v_compdata, vec_src->v_compdata,
              length * sizeof(ngcomplex_t));
    }
} /* end of function copy_vector_data */



/* Copy vector data and its metadata using stride info */
static void copy_vector_data_with_stride(struct dvec *vec_dst,
        const struct dvec *vec_src,
        int n_dim, const index_range_t *p_range)
{
    /* Offsets and related expressions at different levels of indexing
     * given in elements
     *
     * Example
     * Dimensions:        4
     * Dimension extents: 10  X   8  X   100    X   5
     * Selected ranges:   2:5 X 3:4  X  20:30  X   3:4
     * Strides:          4000,  500,      5,        1
     * Min offsets:      8000, 1500,    100,        3 -- offset to 1st
     *                                                  element of range
     * Cur cum offsets:  8000, 9500,   9600,     9603 (initial)
     * Cur index:           2,    3,     20,        X (initial)
     *
     * Note that the strides are built from the highest dimension,
     * which always has stride 1, backwards.
     */
    int p_stride_level[MAXDIMS];
                            /* Stride changing index by 1 at each level */
    int p_offset_level_min[MAXDIMS]; /* Offset to 1st elem at level */

    /* Current cumulative offset at each level. A -1 index is created
     * to handle the case of a single dimension more uniformly */
    int p_offset_level_cum_full[MAXDIMS + 1];
    int *p_offset_level_cum = p_offset_level_cum_full + 1;

    int p_index_cur[MAXDIMS]; /* Current range value at each level */

    {
        const int index_max = n_dim - 1;
        p_stride_level[index_max] = 1;
        int *p_dim_ext = vec_dst->v_dims;
        int i;
        for (i = n_dim - 2; i >= 0; --i) {
            const int i1 = i + 1;
            p_stride_level[i] = p_stride_level[i1] * p_dim_ext[i1];
        }
    }

    /* Initialize the minimum offsets, cumulative current offsets, and
     * current index based on ranges and strides */
    {
        const int low_cur = p_index_cur[0] = p_range[0].low;
        p_offset_level_cum[0] = p_offset_level_min[0] =
                low_cur * p_stride_level[0];
    }

    {
        int i;
        for (i = 1; i < n_dim; ++i) {
            const int low_cur = p_index_cur[i] = p_range[i].low;
            p_offset_level_cum[i] = p_offset_level_cum[i - 1] +
                    (p_offset_level_min[i] = low_cur * p_stride_level[i]);
        }
    }

    /* There are three cases to consider:
     *  1) real dst <- real src
     *  2) complex dst <- complex src
     *  3) complex dst <- real src
     *
     * The first two can copy blocks at the highest dimesion and the can
     * be combined by generalizing to the data size (sizeof(double) or
     * sizeof(ngcomplex_t)) and offset of the data array. The third one
     * must be assigned element by element with 0's given to the imaginary
     * part of the data.
     */

    if (isreal(vec_src) && iscomplex(vec_dst)) {
        /* complex dst <- real src */
        int n_elem_topdim; /* # elements copied in top (stride 1) dimension */
        ngcomplex_t *p_vec_data_dst = vec_dst->v_compdata;
                                    /* Location of data in dvec struct */
        double *p_vec_data_src = vec_src->v_realdata;
                                    /* Location of data in dvec struct */

        {
            const int index_max = n_dim - 1;
            const index_range_t * const p_range_max = p_range + index_max;
            n_elem_topdim = p_range_max->high - p_range_max->low + 1;
        }

        /* Copy all data. Each loop iteration copies all of the elements
         * at the highest dimension (which are contiguous). On entry to
         * the loop, the arrays are initialized so that the first element
         * can be copied, and they are updated in each iteration to
         * process the next element. Note that if this function is called,
         * there will always be at least one element to copy, so it
         * is always safe to copy then check for the end of data. */
        {
            const int n_cpy = n_dim - 1; /* index where copying done */
            const double *p_vec_data_src_end = p_vec_data_src +
                    vec_src->v_length; /* end of copying */
            for ( ; ; ) {
                /* Copy the data currently being located by the cumulative
                 * offset and the source location */
                {
                    ngcomplex_t *p_dst_cur = p_vec_data_dst +
                            p_offset_level_cum[n_cpy];
                    ngcomplex_t *p_dst_end = p_dst_cur + n_elem_topdim;
                    for ( ; p_dst_cur < p_dst_end;
                            ++p_dst_cur, ++p_vec_data_src) {
                        p_dst_cur->cx_real = *p_vec_data_src;
                        p_dst_cur->cx_imag = 0.0;
                    }
                }

                /* Test for end of source data and exit if reached */
                if (p_vec_data_src == p_vec_data_src_end) {
                    break; /* Copy is complete */
                }

                /* Move to the next destination location. Since the loop
                 * was not exited yet, it must exist */
                {
                    int level_cur = n_cpy;

                    /* Move back to the first dimension that is not at its
                     * last element */
                    while (p_index_cur[level_cur] ==
                            p_range[level_cur].high) {
                        --level_cur;
                    }

                    /* Now at the first dimension level that is not full.
                     * Increment here and reset the highe ones to their
                     * minimum values to "count up." */
                    ++p_index_cur[level_cur];
                    p_offset_level_cum[level_cur] +=
                            p_stride_level[level_cur];
                    for (++level_cur; level_cur <= n_cpy; ++level_cur) {
                        p_index_cur[level_cur] = p_range[level_cur].low;
                        p_offset_level_cum[level_cur] =
                                p_offset_level_cum[level_cur - 1] +
                                p_offset_level_min[level_cur];
                    }
                } /* end of block updating destination */
            } /* end of loop copying from source to destination */
        } /* end of block */
    } /* end of case both real or complex */
    else { /* Both real or complex (complex src and real dst not allowed) */
        int n_byte_elem; /* Size of element */
        int n_elem_topdim; /* # elements copied in top (stride 1) dimension */
        int n_byte_topdim; /* contiguous bytes */
        void *p_vec_data_dst; /* Location of data in dvec struct */
        void *p_vec_data_src; /* Location of data in dvec struct */

        {
            const int index_max = n_dim - 1;
            const index_range_t * const p_range_max = p_range + index_max;
            n_elem_topdim = p_range_max->high - p_range_max->low + 1;
        }

        if (isreal(vec_src)) { /* Both real */
            n_byte_elem = (int) sizeof(double);
            n_byte_topdim = (int) n_elem_topdim * sizeof(double);
            p_vec_data_dst = vec_dst->v_realdata;
            p_vec_data_src = vec_src->v_realdata;
        }
        else {
            n_byte_elem = (int) sizeof(ngcomplex_t);
            n_byte_topdim = (int) n_elem_topdim * sizeof(ngcomplex_t);
            p_vec_data_dst = vec_dst->v_compdata;
            p_vec_data_src = vec_src->v_compdata;
        }

        /* Add the offset of the top dimension to all of the lower ones
         * since it will always be added when copying */
        {
            int i;
            const int n_max = n_dim - 1;
            int offset_top = p_range[n_max].low;
            p_offset_level_cum[-1] = offset_top;
            for (i = 0; i < n_max; ++i) {
                p_offset_level_cum[i] += offset_top;
            }
        }

        /* Because the copies are being done in terms of bytes rather
         * than complex data elements or real data elements, convert
         * the strides and offsets from elements to bytes */
        {
            p_offset_level_cum[-1] *= n_byte_elem;
            int i;
            const int n_max = n_dim - 1;
            for (i = 0; i < n_max; i++) {
                p_stride_level[i] *= n_byte_elem;
                p_offset_level_min[i] *= n_byte_elem;
                p_offset_level_cum[i] *= n_byte_elem;
            }
        }

        /* Copy all data. Each loop iteration copies all of the elements
         * at the highest dimension (which are contiguous). On entry to
         * the loop, the arrays are initialized so that the first element
         * can be copied, and they are updated in each iteration to
         * process the next element. Note that if this function is called,
         * there will always be at least one element to copy, so it
         * is always safe to copy then check for the end of data. */
        {
            const int n_cpy = n_dim - 2; /* index where copying done */
            const void *p_vec_data_src_end = (char *) p_vec_data_src +
                    (size_t) vec_src->v_length *
                    n_byte_elem; /* end of copying */
            for ( ; ; ) {
                /* Copy the data currently being located by the cumulative
                 * offset and the source location */
                (void) memcpy(
                        (char *) p_vec_data_dst + p_offset_level_cum[n_cpy],
                        p_vec_data_src,
                        n_byte_topdim);

                /* Move to the next source data and exit the loop if
                 * the end is reached.
                 * NOTE: EXITING BEFORE UPDATING THE DESTINATION WILL
                 * PREVENT OVERRUNNING BUFFERS */
                if ((p_vec_data_src = (char *) p_vec_data_src +
                        n_byte_topdim) == p_vec_data_src_end) {
                    break; /* Copy is complete */
                }

                /* Move to the next destination location. Since the loop
                 * was not exited yet, it must exist */
                {
                    int level_cur = n_cpy;

                    /* Move back to the first dimension that is not at its
                     * last element */
                    while (p_index_cur[level_cur] ==
                            p_range[level_cur].high) {
                        --level_cur;
                    }

                    /* Now at the first dimension level that is not full.
                     * Increment here and reset the highe ones to their
                     * minimum values to "count up." */
                    ++p_index_cur[level_cur];
                    p_offset_level_cum[level_cur] +=
                            p_stride_level[level_cur];
                    for (++level_cur; level_cur <= n_cpy; ++level_cur) {
                        p_index_cur[level_cur] = p_range[level_cur].low;
                        p_offset_level_cum[level_cur] =
                                p_offset_level_cum[level_cur - 1] +
                                p_offset_level_min[level_cur];
                    }
                } /* end of block updating destination */
            } /* end of loop copying from source to destination */
        } /* end of block */
    } /* end of case both real or complex */
} /* end of function copy_vector_data_with_stride */



