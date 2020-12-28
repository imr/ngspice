#include "eno2.h"
#include "eno3.h"

typedef struct {
    int   ix;   /* size of array in x */
    int   iy;   /* size of array in y */
    int   iz;   /* size of array in z */

    sf_eno3 newtable;   /* the table, code borrowed from madagascar project */

    /* Input values corresponding to each index. They define the value
     * in the domain at each index value */
    double *xcol;   /* array of doubles in x */
    double *ycol;   /* array of doubles in y */
    double *zcol;   /* array of doubles in z */

    double ***table; /* f(xi, yj, zk) */
} Table3_Data_t;

void free_local_data(Table3_Data_t *loc);


Table3_Data_t *init_local_data(const char *filename, int order);

/* Finds difference between column values */
static inline double get_local_diff(int n, double *col, int ind)
{
    if (ind >= n - 1) {
        return col[n - 1] - col[n - 2];
    }
    if (ind <= 0) {
        return col[1] - col[0];
    }
    return 0.5 * (col[ind + 1] - col[ind - 1]);
} /* end of function get_local_diff */



