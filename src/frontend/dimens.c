/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author: 1992 David A. Gates, U. C. Berkeley CAD Group
**********/

/*
 * Read and write dimension/index arrays via strings.
 */

#include "ngspice/ngspice.h"
#include "ngspice/dvec.h"               /* For MAXDIMS */
#include "dimens.h"
#include "ngspice/stringskip.h"


static int atodims_bracketed(const char *p, int *data, int *p_n_dim);
static int atodims_unbracketed(const char *p, int *data, int *p_n_dim);
static int get_bracketed_dim(const char *p, int *p_val);
static int atodims_csv(const char *p, int *data, int *p_n_dim);
static int get_dim(const char *p, int *p_val);



/*
 * Create a string of the form "12,1,10".
 *
 * Parameters
 * dim_data: Array of sizes of dimensions, [12, 1, 10] for the example
 * n_dim: Number of elements in the array, 3 in the example
 * retstring: Address of buffer where the string is returned.

 * Remarks
 * It is assumed that the buffer in retstring is large enough, which for
 * MAXDIMS, would require MAXDIMS * 11 bytes in the worst case assuming
 * 32-bit ints. A looser but more general bound only assuming 8-bit bytes
 * would be MAXDIMS * (3 * sizeof(int) + 1).
 */
void
dimstring(const int *dim_data, int n_dim, char *retstring)
{
    /* Handle case of no dimensions */
    if (dim_data == (int *) NULL || n_dim < 1) {
        *retstring = '\0';
        return;
    }

    /* Append each dimension */
    retstring += sprintf(retstring, "%d", dim_data[0]); /* first */
    int  i;
    for (i = 1; i < n_dim; i++) { /* rest are prefixed by a comma */
        retstring += sprintf(retstring, ",%d", dim_data[i]);
    }
} /* end of function dimstring */



/*
 * Create a string of the form "[12][1][10]" in retstring.
 *
 * Parameters
 * dim_data: Array of sizes of dimensions, [12, 1, 10] for the example
 * n_dim: Number of elements in the array, 3 in the example
 * retstring: Address of buffer where the string is returned.

 * Remarks
 * It is assumed that the buffer in retstring is large enough, which for
 * MAXDIMS, would require MAXDIMS * 12 + 1 bytes in the worst case assuming
 * 32-bit ints. A looser but more general bound only assuming 8-bit bytes
 * would be MAXDIMS * (3 * sizeof(int) + 2) + 1.
 */
void
indexstring(const int *dim_data, int n_dim, char *retstring)
{
    /* Handle case of no dimensions */
    if (dim_data == (int *) NULL || n_dim < 1) {
        *retstring = '\0';
        return;
    }

    /* Append each dimension */
    int  i;
    for (i = 0; i < n_dim; i++) {
        retstring += sprintf(retstring, "[%d]", dim_data[i]);
    }
} /* end of function indexstring */



/*
 * Add one to anstrchr into an array with sizes in dims.
 * Return 1 when all counters overflow at once.
 */
int
incindex(int *counts, int numcounts, const int *dims, int numdims)
{
    int i, start;

    if (!counts || numcounts < 1 || !dims || numdims < 1)
        return 0;

    start = numcounts - 1;

    for (i = start; i >= 0; i--)
        if (++counts[i] < dims[i])
            break;      /* This counter is not maxed out. */
        else
            counts[i] = 0;

    if (i == 0)
        return (1);
    else
        return (0);
}


/*
 * Read a string of one of the following forms into a dimensions array:
 *  [12][1][10]
 *  [12,1,10]
 *  12,1,10
 *  12, 1, 10
 *  12 , 1 , 10
 *  Basically, we require that all brackets be matched, that all numbers
 *  be separated by commas or by "][", that all whitespace is ignored, and
 *  the beginning [ and end ] are ignored if they exist.  The only valid
 *  characters in the string are brackets, commas, spaces, and digits.
 *  If any dimension is blank, its entry in the array is set to 0.
 *
 *  Return 0 on success, 1 on failure.
 */
int atodims(const char *p, int *data, int *p_n_dim)
{
    /* Validate arguments partially */
    if (!data || !p_n_dim) {
        return 1;
    }

    /* NULL string = no dimensions */
    if (!p) {
        *p_n_dim = 0;
        return 0;
    }

    /* Move to first "real" character */
    p = skip_ws(p);

    /* Allowed first char is [ to start bracked string or a number */
    return *p == '[' ? atodims_bracketed(p, data, p_n_dim) :
            atodims_unbracketed(p, data, p_n_dim);
} /* end of function atodims */



/* This function processes a dimension string of the form
 * [1,2,3,4] or [1][2][3][4]. Whitespace is allowed anywhere except
 * at the beginning of the string or between the digits of a dimension.
 *
 * Return codes
 * 0: OK
 * +1: Error
 */
static int atodims_bracketed(const char *p, int *data, int *p_n_dim)
{
    /* Process the first element, which is special because it determines
     * if the string is of form [] or [1,2...] or [1][2]... */
    p = skip_ws(++p); /* Step to number */

    {
        int rc;

        /* Get the dimension value exiting with an error on failure */
        if ((rc = get_dim(p, data)) <= 0) { /* no number or overflow */
            if (rc < 0) { /* overflow */
                return +1;
            }
            /* Handle special case of [] */
            if (*p == ']') {
                *p_n_dim = 0;
                return 0;
            }
            return +1; /* else an error */
        }

        
        p = skip_ws(p + rc); /* at comma or ] (or error) */
        switch (*p) {
        case ',': /* form [1,2,... */
            *p_n_dim = 1;
            rc = atodims_csv(++p, data, p_n_dim);
            if (rc <= 1) { /* error or invalid termination */
                return +1; /* Return error */
            }

            /* Else scan ended with ']', but did it end the string,
             * whitespace excluded? */
            p = skip_ws(p + rc);

            return *p != '\0';
        case ']': /* form [1][2]... */
            ++p; /* step past ']' */
            break;
        default: /* invalid char */
            return +1;
        }
    }

    /* Continue parsing form [1][2]... */
    unsigned int n_dim = 1; /* already 1 dim from above */
    for ( ; ; ) {
        if (n_dim == MAXDIMS) { /* too many dimensions */
            return +1;
        }

        int rc = get_bracketed_dim(p, data + n_dim);
        if (rc <= 0) { /* error or normal exit */
            *p_n_dim = (int) n_dim;
            return !!rc;
        }
        p += rc; /* step after the dimension that was processed */
        ++n_dim; /* one more dimension */
    } /* end of loop getting dimensions */
} /* end of function atodims_bracketed */



/* This function processes a dimension string of the form
 *  1,2,3,4. Whiltespace is allowed anywhere except
 * at the beginning of the string or between the digits of a dimension.
 *
 * Return codes
 * 0: OK
 * +1: Error
 */
static int atodims_unbracketed(const char *p, int *data, int *p_n_dim)
{
    *p_n_dim = 0; /* either "" so 0 or init for atodims_csv */

    if (*p == '\0') { /* special case of "" */
        return 0;
    }

    /* Scan comma-separated dimensions. Must end with '\0' (rc=0) */
    return !!atodims_csv(p, data, p_n_dim);
} /* end of function atodims_unbracked */



/* This function processes dimension strings of the form
 *  1,2,3,4 and 1,2,3,4]. Whiltespace is allowed anywhere except
 * at the beginning of the string or between the digits of a dimension.
 * On entry, *p_n_dim is the number of dimensions already added to data
 * and p points to the first number to be processed.

 *
 * Return codes
 * -1: Error
 * 0: OK, scan ended by '\0'
 * >0: OK, scan ended by ']', returned value = # chars processed
 */
static int atodims_csv(const char *p, int *data, int *p_n_dim)
{
    const char *p0 = p;
    unsigned int n_dim = (unsigned int) *p_n_dim;
    for ( ; ; ) {
        int val;
        p = skip_ws(p);
        int rc = get_dim(p, &val);
        if (rc <= 0) { /* No number or overflow */
            return -1;
        }

        /* Dimension was read */
        if (n_dim >= MAXDIMS) { /* too many dimensions */
            return -1;
        }
        data[n_dim++] = val; /* Add data for this dimension */
        p = skip_ws(p + rc); /* step after the dimension that was processed */

        /* Should normally be at comma, but there are special cases for
         * end of regular list or bracketed list */
        switch (*p) {
        case ',': /* inter-dimension comma */
            ++p;
            break;
        case ']': /* ] ended scan */
            *p_n_dim = (int) n_dim;
            return (int) (p - p0) + 1;
        case '\0': /* end of string ended scan */
            *p_n_dim = (int) n_dim;
            return 0;
        default: /* invalid char */
            return -1;
        } /* end of switch over character ending scan */
    } /* end of loop getting dimensions */
} /* end of function atodims_csv */



/* This function gets the dimension value in a string of the form
 * [1] where spaces may appear anywhere except between the digits of the
 * number.
 *
 * Return codes
 * -1: Error
 * 0: String ended before '['
 * >0: Number of characters processed
 */
static int get_bracketed_dim(const char *p, int *p_val)
{
    const char *p0 = p; /* save start */
    p = skip_ws(p); /* move to opening bracket */

    const char char_cur = *p;
    if (char_cur == '\0') { /* end of string */
        return 0;
    }
    if (char_cur != '[') { /* no bracket */
        return -1;
    }
    p = skip_ws(++p); /* move to dimension */

    int rc = get_dim(p, p_val); /* read the dimension */
    if (rc <= 0) { /* error */
        return -1;
    }

    p = skip_ws(p + rc); /* move to closing backet */
    if (*p != ']') { /* no bracket */
        return -1;
    }

    return (int) (p - p0) + 1;
} /* end of function get_bracketed_dim */



/* This function reads the unsigned number at p as a dimension
 *
 * Return codes
 * -1: overflow
 * 0: *p is not a digit
 * >0: Number of characters processed
 */
static int get_dim(const char *p, int *p_val)
{
    unsigned int val = 0;
    const char *p0 = p;
    for ( ; ; ++p) {
        const char c_cur = *p;
        unsigned int digit_cur = (unsigned int) (c_cur - '0');
        unsigned int val_new;
        if (digit_cur > 9) { /* not a digit */
            if ((*p_val = (int) val) < 0) { /* overflow */
                return -1;
            }
            return (int) (p - p0);
        } /* end of case of not a digit */
        if ((val_new = 10 * val + digit_cur) < val) { /* overflow */
            return -1;
        }
        val = val_new; /* update number */
    } /* end of loop over digits */
} /* end of function get_dim */



#ifdef COMPILE_UNUSED_FUNCTIONS
/* #ifdef COMPILE_UNUSED_FUNCTIONS added 2019-03-31 */
/*
 * Count number of empty dimensions in an array.
 */
int
emptydims(int *data, int length)
{
    int i, numempty = 0;

    for (i = 0; i < length; i++)
        if (data[i] == 0)
            numempty++;

    return (numempty);
}



/*
 * Skip to the first character that cannot be part of a dimension string.
 */
char *
skipdims(char *p)
{
    if (!p)
        return NULL;

    while (*p && (*p == '[' || *p == ']' || *p == ',' ||
            isspace_c(*p) || isdigit_c(*p)))
        p++;

    return (p);
}
#endif /* COMPILE_UNUSED_FUNCTIONS */
