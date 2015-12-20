#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*********************/
/* 3d geometry types */
/*********************/

typedef struct Point3Struct {   /* 3d point */
        double x, y, z;
        } Point3;
typedef Point3 Vector3;



/* Function to find the cross over point (the point before
   which elements are smaller than or equal to x and after
   which greater than x)
   http://www.geeksforgeeks.org/find-k-closest-elements-given-value/ */
int
findCrossOver(double arr[], int low, int high, double x)
{
    int mid;
    // Base cases
    if (arr[high] <= x) // x is greater than all
        return high;
    if (arr[low] > x)  // x is smaller than all
        return low;

    // Find the middle point
    mid = (low + high)/2;  /* low + (high - low)/2 */

    /* If x is same as middle element, then return mid */
    if (arr[mid] <= x && arr[mid+1] > x)
        return mid;

    /* If x is greater than arr[mid], then either arr[mid + 1]
      is ceiling of x or ceiling lies in arr[mid+1...high] */
    if (arr[mid] < x)
        return findCrossOver(arr, mid+1, high, x);

  return findCrossOver(arr, low, mid - 1, x);
}

/* https://helloacm.com/cc-function-to-compute-the-bilinear-interpolation/ */
double
BilinearInterpolation(double q11, double q12, double q21, double q22, double x1, double x2, double y1, double y2, double x, double y)
{
    double x2x1, y2y1, x2x, y2y, yy1, xx1;
    x2x1 = x2 - x1;
    y2y1 = y2 - y1;
    x2x = x2 - x;
    y2y = y2 - y;
    yy1 = y - y1;
    xx1 = x - x1;
    return 1.0 / (x2x1 * y2y1) * (
        q11 * x2x * y2y +
        q21 * xx1 * y2y +
        q12 * x2x * yy1 +
        q22 * xx1 * yy1
    );
}


/*
 * C code from the article
 * "Tri-linear Interpolation"
 * by Steve Hill, sah@ukc.ac.uk
 * in "Graphics Gems IV", Academic Press, 1994
 *
 */


double
trilinear(Point3 *p, double *d, int xsize, int ysize, int zsize, double def)
{
#   define DENS(X, Y, Z) d[(X)+xsize*((Y)+ysize*(Z))]

    int        x0, y0, z0,
               x1, y1, z1;
    double     *dp,
               fx, fy, fz,
               d000, d001, d010, d011,
               d100, d101, d110, d111,
               dx00, dx01, dx10, dx11,
               dxy0, dxy1, dxyz;

    x0 = floor(p->x);
    fx = p->x - x0;
    y0 = floor(p->y);
    fy = p->y - y0;
    z0 = floor(p->z);
    fz = p->z - z0;

    x1 = x0 + 1;
    y1 = y0 + 1;
    z1 = z0 + 1;

    if (x0 >= 0 && x1 < xsize &&
            y0 >= 0 && y1 < ysize &&
            z0 >= 0 && z1 < zsize) {
        dp = &DENS(x0, y0, z0);
        d000 = dp[0];
        d100 = dp[1];
        dp += xsize;
        d010 = dp[0];
        d110 = dp[1];
        dp += xsize*ysize;
        d011 = dp[0];
        d111 = dp[1];
        dp -= xsize;
        d001 = dp[0];
        d101 = dp[1];
    } else {
#       define INRANGE(X, Y, Z) \
                  ((X) >= 0 && (X) < xsize && \
                   (Y) >= 0 && (Y) < ysize && \
                   (Z) >= 0 && (Z) < zsize)

        d000 = INRANGE(x0, y0, z0) ? DENS(x0, y0, z0) : def;
        d001 = INRANGE(x0, y0, z1) ? DENS(x0, y0, z1) : def;
        d010 = INRANGE(x0, y1, z0) ? DENS(x0, y1, z0) : def;
        d011 = INRANGE(x0, y1, z1) ? DENS(x0, y1, z1) : def;

        d100 = INRANGE(x1, y0, z0) ? DENS(x1, y0, z0) : def;
        d101 = INRANGE(x1, y0, z1) ? DENS(x1, y0, z1) : def;
        d110 = INRANGE(x1, y1, z0) ? DENS(x1, y1, z0) : def;
        d111 = INRANGE(x1, y1, z1) ? DENS(x1, y1, z1) : def;
    }
/* linear interpolation from l (when a=0) to h (when a=1)*/
/* (equal to (a*h)+((1-a)*l) */
#define LERP(a,l,h)     ((l)+(((h)-(l))*(a)))

    dx00 = LERP(fx, d000, d100);
    dx01 = LERP(fx, d001, d101);
    dx10 = LERP(fx, d010, d110);
    dx11 = LERP(fx, d011, d111);

    dxy0 = LERP(fy, dx00, dx10);
    dxy1 = LERP(fy, dx01, dx11);

    dxyz = LERP(fz, dxy0, dxy1);

    return dxyz;
}

/* trilinear interpolation
Paul Bourke
July 1997
http://paulbourke.net/miscellaneous/interpolation/ */

double
TrilinearInterpolation(double x, double y, double z, int xind, int yind, int zind, double ***td)
{
    double V000, V100, V010, V001, V101, V011, V110, V111, Vxyz;

    V000 = td[zind][yind][xind];
    V100 = td[zind][yind][xind+1];
    V010 = td[zind][yind+1][xind];
    V001 = td[zind+1][yind][xind];
    V101 = td[zind+1][yind][xind+1];
    V011 = td[zind+1][yind+1][xind];
    V110 = td[zind][yind+1][xind+1];
    V111 = td[zind+1][yind+1][xind+1];

    Vxyz = V000 * (1 - x) * (1 - y) * (1 - z) +
        V100 * x * (1 - y) * (1 - z) +
        V010 * (1 - x) * y * (1 - z) +
        V001 * (1 - x) * (1 - y) * z +
        V101 * x * (1 - y) * z +
        V011 * (1 - x) * y * z +
        V110 * x * y * (1 - z) +
        V111 * x * y * z;
    return Vxyz;
}
