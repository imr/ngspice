#ifndef interp_h_included
#define interp_h_included
double BilinearInterpolation(double x, double y,
        int xind, int yind, double **td);
double TrilinearInterpolation(double x, double y, double z,
        int xind, int yind, int zind, double ***td);
int findCrossOver(double arr[], int n, double x);
#endif /* include guard */
