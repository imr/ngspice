/*************
 * Header file for interp.c
 * 1999 E. Rouat
 ************/

#ifndef INTERP_H_INCLUDED
#define INTERP_H_INCLUDED

bool ft_interpolate(double *data, double *ndata, double *oscale, int olen, double *nscale, 
		    int nlen, int degree);
bool ft_polyfit(double *xdata, double *ydata, double *result, int degree, double *scratch);
double ft_peval(double x, double *coeffs, int degree);
void lincopy(struct dvec *ov, double *newscale, int newlen, struct dvec *oldscale);
void ft_polyderiv(double *coeffs, int degree);


#endif
