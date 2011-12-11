/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1990 Jaijeet S. Roychowdhury
**********/

#include "ngspice/ngspice.h"
#include "ltradefs.h"
#include "ngspice/suffix.h"

/*
 * Miscellaneous functions to do with lossy lines
 */

/*
 * LTRAquadInterp - quadratic interpolation function t = timepoint where
 * value wanted t1, t2, t3 are three timepoints where the value is known c1,
 * c2, c3 are set to the proper coefficients by the function the interpolated
 * value is c1*v1 + c2*v2 + c3*v3; this should be done in the calling
 * program; (v1,v2,v3 are the known values at t1,t2,t3)
 */


int
LTRAquadInterp(double t, double t1, double t2, double t3, double *c1, double *c2, double *c3)
{
  double f1, f2, f3;

  if (t == t1) {
    *c1 = 1.0;
    *c2 = 0.0;
    *c3 = 0.0;
    return (0);
  }
  if (t == t2) {
    *c1 = 0.0;
    *c2 = 1.0;
    *c3 = 0.0;
    return (0);
  }
  if (t == t3) {
    *c1 = 0.0;
    *c2 = 0.0;
    *c3 = 1.0;
    return (0);
  }
  if ((t2 - t1) == 0 || (t3 - t2) == 0 || (t1 - t3) == 0)
    return (1);

  f1 = (t - t2) * (t - t3);
  f2 = (t - t1) * (t - t3);
  f3 = (t - t1) * (t - t2);
  if ((t2 - t1) == 0) {		/* should never happen, but don't want to
				 * divide by zero, EVER... */
    f1 = 0;
    f2 = 0;
  } else {
    f1 /= (t1 - t2);
    f2 /= (t2 - t1);
  }
  if ((t3 - t2) == 0) {		/* should never happen, but don't want to
				 * divide by zero, EVER... */
    f2 = 0;
    f3 = 0;
  } else {
    f2 /= (t2 - t3);
    f3 /= (t2 - t3);
  }
  if ((t3 - t1) == 0) {		/* should never happen, but don't want to
				 * divide by zero, EVER... */
    f1 = 0;
    f2 = 0;
  } else {
    f1 /= (t1 - t3);
    f3 /= (t1 - t3);
  }
  *c1 = f1;
  *c2 = f2;
  *c3 = f3;
  return (0);
}

/* linear interpolation */

int
LTRAlinInterp(double t, double t1, double t2, double *c1, double *c2)
{
  double temp;

  if (t1 == t2)
    return (1);

  if (t == t1) {
    *c1 = 1.0;
    *c2 = 0.0;
    return (0);
  }
  if (t == t2) {
    *c1 = 0.0;
    *c2 = 1.0;
    return (0);
  }
  temp = (t - t1) / (t2 - t1);
  *c2 = temp;
  *c1 = 1 - temp;
  return (0);
}

/*
 * intlinfunc returns \int_lolimit^hilimit h(\tau) d \tau, where h(\tau) is
 * assumed to be linear, with values lovalue and hivalue \tau = t1 and t2
 * respectively this is used only locally
 */

static double
intlinfunc(double lolimit, double hilimit, double lovalue, double hivalue, double t1, double t2)
{
  double width, m;


  width = t2 - t1;
  if (width == 0.0)
    return (0.0);
  m = (hivalue - lovalue) / width;

  return ((hilimit - lolimit) * lovalue + 0.5 * m * ((hilimit - t1) * (hilimit - t1)
	  - (lolimit - t1) * (lolimit - t1)));
}


/*
 * twiceintlinfunc returns \int_lolimit^hilimit \int_otherlolimit^\tau
 * h(\tau') d \tau' d \tau , where h(\tau') is assumed to be linear, with
 * values lovalue and hivalue \tau = t1 and t2 respectively this is used only
 * locally
 */

static double
twiceintlinfunc(double lolimit, double hilimit, double otherlolimit, double lovalue, double hivalue, double t1, double t2)
{
  double width, m, dummy;
  double temp1, temp2, temp3;


  width = t2 - t1;
  if (width == 0.0)
    return (0.0);
  m = (hivalue - lovalue) / width;

  temp1 = hilimit - t1;
  temp2 = lolimit - t1;
  temp3 = otherlolimit - t1;
  dummy = lovalue * ((hilimit - otherlolimit) * (hilimit - otherlolimit) -
      (lolimit - otherlolimit) * (lolimit - otherlolimit));
  dummy += m * ((temp1 * temp1 * temp1 - temp2 * temp2 * temp2) / 3.0 -
      temp3 * temp3 * (hilimit - lolimit));
  return (dummy * 0.5);
}

/*
 * thriceintlinfunc returns \int_lolimit^hilimit \int_secondlolimit^\tau
 * \int_thirdlolimit^\tau' h(\tau'') d \tau'' d \tau' d \tau , where
 * h(\tau'') is assumed to be linear, with values lovalue and hivalue \tau =
 * t1 and t2 respectively this is used only locally
 */

static double
thriceintlinfunc(double lolimit, double hilimit, double secondlolimit, double thirdlolimit, double lovalue, double hivalue, double t1, double t2)
{
  double width, m, dummy;
  double temp1, temp2, temp3, temp4;
  double temp5, temp6, temp7, temp8, temp9, temp10;


  width = t2 - t1;
  if (width == 0.0)
    return (0.0);
  m = (hivalue - lovalue) / width;

  temp1 = hilimit - t1;
  temp2 = lolimit - t1;
  temp3 = secondlolimit - t1;
  temp4 = thirdlolimit - t1;
  temp5 = hilimit - thirdlolimit;
  temp6 = lolimit - thirdlolimit;
  temp7 = secondlolimit - thirdlolimit;
  temp8 = hilimit - lolimit;
  temp9 = hilimit - secondlolimit;
  temp10 = lolimit - secondlolimit;
  dummy = lovalue * ((temp5 * temp5 * temp5 - temp6 * temp6 * temp6) / 3 -
      temp7 * temp5 * temp8);
  dummy += m * (((temp1 * temp1 * temp1 * temp1 - temp2 * temp2 * temp2 * temp2) * 0.25 -
	  temp3 * temp3 * temp3 * temp8) / 3 - temp4 * temp4 * 0.5 * (temp9 * temp9 -
	  temp10 * temp10));
  return (dummy * 0.5);
}


/*
 * These are from the book Numerical Recipes in C
 * 
 */

static double
bessI0(double x)
{
  double ax, ans;
  double y;

  if ((ax = fabs(x)) < 3.75) {
    y = x / 3.75;
    y *= y;
    ans = 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492
		+ y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2)))));
  } else {
    y = 3.75 / ax;
    ans = (exp(ax) / sqrt(ax)) * (0.39894228 + y * (0.1328592e-1
	    + y * (0.225319e-2 + y * (-0.157565e-2 + y * (0.916281e-2
			+ y * (-0.2057706e-1 + y * (0.2635537e-1 + y * (-0.1647633e-1
				    + y * 0.392377e-2))))))));
  }
  return (ans);
}

static double
bessI1(double x)
{
  double ax, ans;
  double y;

  if ((ax = fabs(x)) < 3.75) {
    y = x / 3.75;
    y *= y;
    ans = ax * (0.5 + y * (0.87890594 + y * (0.51498869 + y * (0.15084934
		    + y * (0.2658733e-1 + y * (0.301532e-2 + y * 0.32411e-3))))));
  } else {
    y = 3.75 / ax;
    ans = 0.2282967e-1 + y * (-0.2895312e-1 + y * (0.1787654e-1
	    - y * 0.420059e-2));
    ans = 0.39894228 + y * (-0.3988024e-1 + y * (-0.362018e-2
	    + y * (0.163801e-2 + y * (-0.1031555e-1 + y * ans))));
    ans *= (exp(ax) / sqrt(ax));
  }
  return (x < 0.0 ? -ans : ans);
}

static double
bessI1xOverX(double x)
{
  double ax, ans;
  double y;

  if ((ax = fabs(x)) < 3.75) {
    y = x / 3.75;
    y *= y;
    ans = 0.5 + y * (0.87890594 + y * (0.51498869 + y * (0.15084934
		+ y * (0.2658733e-1 + y * (0.301532e-2 + y * 0.32411e-3)))));
  } else {
    y = 3.75 / ax;
    ans = 0.2282967e-1 + y * (-0.2895312e-1 + y * (0.1787654e-1
	    - y * 0.420059e-2));
    ans = 0.39894228 + y * (-0.3988024e-1 + y * (-0.362018e-2
	    + y * (0.163801e-2 + y * (-0.1031555e-1 + y * ans))));
    ans *= (exp(ax) / (ax * sqrt(ax)));
  }
  return (ans);
}

/* LTRArlcH1dashFunc - the first impulse response function */

double
LTRArlcH1dashFunc(double time, double T, double alpha, double beta)
{
  double besselarg, exparg, returnval;
  /* T is not used in this function */
  NG_IGNORE(T);

  /*
   * result = alpha * e^{- beta*time} * {I_1(alpha*time) - I_0(alpha*time)}
   */

  if (alpha == 0.0)
    return (0.0);

  exparg = -beta * time;
  besselarg = alpha * time;

  returnval = (bessI1(besselarg) - bessI0(besselarg)) * alpha * exp(exparg);
  return (returnval);
}

double
LTRArlcH2Func(double time, double T, double alpha, double beta)
{
  double besselarg, exparg, returnval;

  /*
   * result = 0, time < T = (alpha*T*e^{-beta*time})/sqrt(t^2 - T^2) *
   * I_1(alpha*sqrt(t^2 - T^2)), time >= T
   */

  if (alpha == 0.0)
    return (0.0);
  if (time < T)
    return (0.0);

  if (time != T) {
    besselarg = alpha * sqrt(time * time - T * T);
  } else {
    besselarg = 0.0;
  }
  exparg = -beta * time;

  returnval = alpha * alpha * T * exp(exparg) * bessI1xOverX(besselarg);
  return (returnval);
}

double
LTRArlcH3dashFunc(double time, double T, double alpha, double beta)
{
  double exparg, besselarg, returnval;

  /*
   * result = 0, time < T = alpha*e^{-beta*time}*(t/sqrt(t^2-T^2)*
   * I_1(alpha*sqrt(t^2-T^2)) - I_0(alpha*sqrt(t^2-T^2)))
   */

  if (alpha == 0.0)
    return (0.0);
  if (time < T)
    return (0.0);

  exparg = -beta * time;
  if (time != T) {
    besselarg = alpha * sqrt(time * time - T * T);
  } else {
    besselarg = 0.0;
  }

  returnval = alpha * time * bessI1xOverX(besselarg) - bessI0(besselarg);
  returnval *= alpha * exp(exparg);
  return (returnval);
}

/*
 * LTRArlcH1dashTwiceIntFunc - twice repeated integral of h1dash for the
 * special case of G = 0
 */

double
LTRArlcH1dashTwiceIntFunc(double time, double beta)
{
  double arg, returnval;

  /*
   * result = time * e^{- beta*time} * {I_0(beta*time) + I_1(beta*time)} -
   * time
   */

  if (beta == 0.0)
    return (time);
  arg = beta * time;
  if (arg == 0.0)
    return (0.0);

  returnval = (bessI1(arg) + bessI0(arg)) * time * exp(-arg) - time;
  return (returnval);
}

/*
 * LTRArlcH3dashIntFunc - twice repeated integral of h1dash for the special
 * case of G = 0
 */

double
LTRArlcH3dashIntFunc(double time, double T, double beta)
{
  double exparg, besselarg;
  double returnval;

  if (time <= T)
    return (0.0);
  if (beta == 0.0)
    return (0.0);
  exparg = -beta * time;
  besselarg = beta * sqrt(time * time - T * T);
  returnval = exp(exparg) * bessI0(besselarg) - exp(-beta * T);
  return (returnval);
}


double
LTRArcH1dashTwiceIntFunc(double time, double cbyr)
{

  return (sqrt(4 * cbyr * time / M_PI));
}




double
LTRArcH2TwiceIntFunc(double time, double rclsqr)
{
  double temp;

  if (time != 0.0) {
    temp = rclsqr / (4 * time);
    return ((time + rclsqr * 0.5) * erfc(sqrt(temp)) - sqrt(time * rclsqr / M_PI) * exp(-temp));
  } else {
    return (0.0);
  }
}



double
LTRArcH3dashTwiceIntFunc(double time, double cbyr, double rclsqr)
{
  double temp;

  if (time != 0.0) {
    temp = rclsqr / (4 * time);
    temp = 2 * sqrt(time / M_PI) * exp(-temp) - sqrt(rclsqr) * erfc(sqrt(temp));
    return (sqrt(cbyr) * temp);
  } else {
    return (0.0);
  }
}


/*
 * LTRArcCoeffsSetup sets up the all coefficient lists for the special case
 * where L=G=0
 */

void
LTRArcCoeffsSetup(double *h1dashfirstcoeff, double *h2firstcoeff, double *h3dashfirstcoeff, double *h1dashcoeffs,
                  double *h2coeffs, double *h3dashcoeffs, int listsize, double cbyr,
                  double rclsqr, double curtime, double *timelist, int timeindex, double reltol)
{
  double delta1, delta2;
  double h1dummy1, h1dummy2;
  double h2dummy1, h2dummy2;
  double h3dummy1, h3dummy2;
  double lolimit1, lolimit2, hilimit1, hilimit2;
  double h1lovalue1, h1lovalue2, h1hivalue1, h1hivalue2;
  double h2lovalue1, h2lovalue2, h2hivalue1, h2hivalue2;
  double h3lovalue1, h3lovalue2, h3hivalue1, h3hivalue2;
  double temp, temp2, temp3, temp4, temp5;
  double h1relval, h2relval, h3relval;
  int doh1 = 1, doh2 = 1, doh3 = 1;
  int i, auxindex;

  NG_IGNORE(listsize);

  /* coefflists should already have been allocated to the necessary size */

#ifdef LTRAdebug
  if (listsize <= timeindex) {
    printf("LTRAcoeffSetup: not enough space in coefflist\n");
  }
#endif



  auxindex = timeindex;

  /* the first coefficients */

  delta1 = curtime - *(timelist + auxindex);
  lolimit1 = 0.0;
  hilimit1 = delta1;

  h1lovalue1 = 0.0;
  h1hivalue1 =			/* LTRArcH1dashTwiceIntFunc(hilimit1,cbyr); */
      sqrt(4 * cbyr * hilimit1 / M_PI);
  h1dummy1 = h1hivalue1 / delta1;
  *h1dashfirstcoeff = h1dummy1;
  h1relval = fabs(h1dummy1 * reltol);

  temp = rclsqr / (4 * hilimit1);
  temp2 = (temp >= 100.0 ? 0.0 : erfc(sqrt(temp)));
  temp3 = exp(-temp);
  temp4 = sqrt(rclsqr);
  temp5 = sqrt(cbyr);

  h2lovalue1 = 0.0;
  h2hivalue1 =			/* LTRArcH2TwiceIntFunc(hilimit1,rclsqr); */
      (hilimit1 != 0.0 ? (hilimit1 + rclsqr * 0.5) * temp2 - sqrt(hilimit1 * rclsqr / M_PI) * temp3 : 0.0);


  h2dummy1 = h2hivalue1 / delta1;
  *h2firstcoeff = h2dummy1;
  h2relval = fabs(h2dummy1 * reltol);

  h3lovalue1 = 0.0;
  h3hivalue1 =			/* LTRArcH3dashTwiceIntFunc(hilimit1,cbyr,rcls
				 * qr); */
      (hilimit1 != 0.0 ? temp = 2 * sqrt(hilimit1 / M_PI) * temp3 - temp4 * temp2, (temp5 * temp) : 0.0);


  h3dummy1 = h3hivalue1 / delta1;
  *h3dashfirstcoeff = h3dummy1;
  h3relval = fabs(h3dummy1 * reltol);

  /* the coefficients for the rest of the timepoints */

  for (i = auxindex; i > 0; i--) {

    delta2 = delta1;		/* previous delta1 */
    lolimit2 = lolimit1;	/* previous lolimit1 */
    hilimit2 = hilimit1;	/* previous hilimit1 */

    delta1 = *(timelist + i) - *(timelist + i - 1);
    lolimit1 = hilimit2;
    hilimit1 = curtime - *(timelist + i - 1);

    if (doh1) {
      h1lovalue2 = h1lovalue1;	/* previous lovalue1 */
      h1hivalue2 = h1hivalue1;	/* previous hivalue1 */
      h1dummy2 = h1dummy1;	/* previous dummy1 */

      h1lovalue1 = h1hivalue2;
      h1hivalue1 =		/* LTRArcH1dashTwiceIntFunc(hilimit1,cbyr); */
	  sqrt(4 * cbyr * hilimit1 / M_PI);
      h1dummy1 = (h1hivalue1 - h1lovalue1) / delta1;
      *(h1dashcoeffs + i) = h1dummy1 - h1dummy2;
      if (fabs(*(h1dashcoeffs + i)) < h1relval)
	doh1 = 0;
    } else
      *(h1dashcoeffs + i) = 0.0;

    if (doh2 || doh3) {
      temp = rclsqr / (4 * hilimit1);
      temp2 = (temp >= 100.0 ? 0.0 : erfc(sqrt(temp)));
      temp3 = exp(-temp);
    }
    if (doh2) {
      h2lovalue2 = h2lovalue1;	/* previous lovalue1 */
      h2hivalue2 = h2hivalue1;	/* previous hivalue1 */
      h2dummy2 = h2dummy1;	/* previous dummy1 */

      h2lovalue1 = h2hivalue2;
      h2hivalue1 =		/* LTRArcH2TwiceIntFunc(hilimit1,rclsqr); */
	  (hilimit1 != 0.0 ? (hilimit1 + rclsqr * 0.5) * temp2 - sqrt(hilimit1 * rclsqr / M_PI) * temp3 : 0.0);
      h2dummy1 = (h2hivalue1 - h2lovalue1) / delta1;
      *(h2coeffs + i) = h2dummy1 - h2dummy2;
      if (fabs(*(h2coeffs + i)) < h2relval)
	doh2 = 0;
    } else
      *(h2coeffs + i) = 0.0;

    if (doh3) {
      h3lovalue2 = h3lovalue1;	/* previous lovalue1 */
      h3hivalue2 = h3hivalue1;	/* previous hivalue1 */
      h3dummy2 = h3dummy1;	/* previous dummy1 */

      h3lovalue1 = h3hivalue2;
      h3hivalue1 =		/* LTRArcH3dashTwiceIntFunc(hilimit1,cbyr,rcls
				 * qr); */
	  (hilimit1 != 0.0 ? temp = 2 * sqrt(hilimit1 / M_PI) * temp3 - temp4 * temp2, (temp5 * temp) : 0.0);
      h3dummy1 = (h3hivalue1 - h3lovalue1) / delta1;
      *(h3dashcoeffs + i) = h3dummy1 - h3dummy2;
      if (fabs(*(h3dashcoeffs + i)) < h3relval)
	doh3 = 0;
    } else
      *(h3dashcoeffs + i) = 0.0;
  }
}

void
LTRArlcCoeffsSetup(double *h1dashfirstcoeff, double *h2firstcoeff, double *h3dashfirstcoeff, double *h1dashcoeffs, double *h2coeffs,
                   double *h3dashcoeffs, int listsize, double T, double alpha, double beta, double curtime, double *timelist, int timeindex, double reltol, int *auxindexptr)

{
  unsigned exact;
  double lolimit1, lolimit2 = 0.0, hilimit1, hilimit2 = 0.0;
  double delta1, delta2;

  double h1dummy1, h1dummy2;
  double h1lovalue1, h1lovalue2, h1hivalue1, h1hivalue2;

  double h2dummy1 = 0.0, h2dummy2;
  double h2lovalue1 = 0.0, h2lovalue2, h2hivalue1 = 0.0, h2hivalue2;

  double h3dummy1 = 0.0, h3dummy2;
  double h3lovalue1 = 0.0, h3lovalue2, h3hivalue1 = 0.0, h3hivalue2;

  double exparg, besselarg = 0.0, expterm, bessi1overxterm, bessi0term;
  double expbetaTterm = 0.0, alphasqTterm = 0.0;
  double h1relval, h2relval = 0.0, h3relval = 0.0;
  int doh1 = 1, doh2 = 1, doh3 = 1;

  int i, auxindex;

  NG_IGNORE(listsize);

  /* coefflists should already have been allocated to the necessary size */

#ifdef LTRAdebug
  if (listsize <= timeindex) {
    printf("LTRArlcCoeffsSetup: not enough space in coefflist\n");
  }
#endif


  /*
   * we assume a piecewise linear function, and we calculate the coefficients
   * using this assumption in the integration of the function
   */

  if (T == 0.0) {
    auxindex = timeindex;
  } else {

    if (curtime - T <= 0.0) {
      auxindex = 0;
    } else {
      exact = 0;
      for (i = timeindex; i >= 0; i--) {
	if (curtime - *(timelist + i) == T) {
	  exact = 1;
	  break;
	}
	if (curtime - *(timelist + i) > T)
	  break;
      }

#ifdef LTRADEBUG
      if ((i < 0) || ((i == 0) && (exact == 1)))
	printf("LTRAcoeffSetup: i <= 0: some mistake!\n");
#endif

      if (exact == 1) {
	auxindex = i - 1;
      } else {
	auxindex = i;
      }
    }
  }
  /* the first coefficient */

  if (auxindex != 0) {
    lolimit1 = T;
    hilimit1 = curtime - *(timelist + auxindex);
    delta1 = hilimit1 - lolimit1;

    h2lovalue1 = LTRArlcH2Func(T, T, alpha, beta);
    besselarg = (hilimit1 > T) ? alpha * sqrt(hilimit1 * hilimit1 - T * T) : 0.0;
    exparg = -beta * hilimit1;
    expterm = exp(exparg);
    bessi1overxterm = bessI1xOverX(besselarg);
    alphasqTterm = alpha * alpha * T;
    h2hivalue1 =		/* LTRArlcH2Func(hilimit1,T,alpha,beta); */
	((alpha == 0.0) || (hilimit1 < T)) ? 0.0 : alphasqTterm * expterm * bessi1overxterm;

    h2dummy1 = twiceintlinfunc(lolimit1, hilimit1, lolimit1, h2lovalue1,
	h2hivalue1, lolimit1, hilimit1) / delta1;
    *h2firstcoeff = h2dummy1;
    h2relval = fabs(reltol * h2dummy1);

    h3lovalue1 = 0.0;		/* E3dash should be consistent with this */
    bessi0term = bessI0(besselarg);
    expbetaTterm = exp(-beta * T);
    h3hivalue1 =		/* LTRArlcH3dashIntFunc(hilimit1,T,beta); */
	((hilimit1 <= T) || (beta == 0.0)) ? 0.0 : expterm * bessi0term - expbetaTterm;
    h3dummy1 = intlinfunc(lolimit1, hilimit1, h3lovalue1,
	h3hivalue1, lolimit1, hilimit1) / delta1;
    *h3dashfirstcoeff = h3dummy1;
    h3relval = fabs(h3dummy1 * reltol);
  } else {
    *h2firstcoeff = *h3dashfirstcoeff = 0.0;
  }

  lolimit1 = 0.0;
  hilimit1 = curtime - *(timelist + timeindex);
  delta1 = hilimit1 - lolimit1;
  exparg = -beta * hilimit1;
  expterm = exp(exparg);

  h1lovalue1 = 0.0;
  h1hivalue1 =			/* LTRArlcH1dashTwiceIntFunc(hilimit1,beta); */
      (beta == 0.0) ? hilimit1 : ((hilimit1 == 0.0) ? 0.0 : (bessI1(-exparg) + bessI0(-exparg)) * hilimit1 * expterm - hilimit1);
  h1dummy1 = h1hivalue1 / delta1;
  *h1dashfirstcoeff = h1dummy1;
  h1relval = fabs(h1dummy1 * reltol);


  /* the coefficients for the rest of the timepoints */

  for (i = timeindex; i > 0; i--) {

    if (doh1 || doh2 || doh3) {
      lolimit2 = lolimit1;	/* previous lolimit1 */
      hilimit2 = hilimit1;	/* previous hilimit1 */
      delta2 = delta1;		/* previous delta1 */

      lolimit1 = hilimit2;
      hilimit1 = curtime - *(timelist + i - 1);
      delta1 = *(timelist + i) - *(timelist + i - 1);

      exparg = -beta * hilimit1;
      expterm = exp(exparg);
    }
    if (doh1) {
      h1lovalue2 = h1lovalue1;	/* previous lovalue1 */
      h1hivalue2 = h1hivalue1;	/* previous hivalue1 */
      h1dummy2 = h1dummy1;	/* previous dummy1 */

      h1lovalue1 = h1hivalue2;
      h1hivalue1 =		/* LTRArlcH1dashTwiceIntFunc(hilimit1,beta); */
	  (beta == 0.0) ? hilimit1 : ((hilimit1 == 0.0) ? 0.0 : (bessI1(-exparg) + bessI0(-exparg)) * hilimit1 * expterm - hilimit1);
      h1dummy1 = (h1hivalue1 - h1lovalue1) / delta1;

      *(h1dashcoeffs + i) = h1dummy1 - h1dummy2;
      if (fabs(*(h1dashcoeffs + i)) <= h1relval)
	doh1 = 0;
    } else
      *(h1dashcoeffs + i) = 0.0;

    if (i <= auxindex) {

      /*
       * if (i == auxindex) { lolimit2 = T; delta2 = hilimit2 - lolimit2; }
       */

      if (doh2 || doh3)
	besselarg = (hilimit1 > T) ? alpha * sqrt(hilimit1 * hilimit1 - T * T) : 0.0;

      if (doh2) {
	h2lovalue2 = h2lovalue1;/* previous lovalue1 */
	h2hivalue2 = h2hivalue1;/* previous hivalue1 */
	h2dummy2 = h2dummy1;	/* previous dummy1 */

	h2lovalue1 = h2hivalue2;
	bessi1overxterm = bessI1xOverX(besselarg);
	h2hivalue1 =		/* LTRArlcH2Func(hilimit1,T,alpha,beta); */
	    ((alpha == 0.0) || (hilimit1 < T)) ? 0.0 : alphasqTterm * expterm * bessi1overxterm;
	h2dummy1 = twiceintlinfunc(lolimit1, hilimit1, lolimit1,
	    h2lovalue1, h2hivalue1, lolimit1, hilimit1) / delta1;

	*(h2coeffs + i) = h2dummy1 - h2dummy2 + intlinfunc(lolimit2, hilimit2,
	    h2lovalue2, h2hivalue2, lolimit2, hilimit2);
	if (fabs(*(h2coeffs + i)) <= h2relval)
	  doh2 = 0;
      } else
	*(h2coeffs + i) = 0.0;

      if (doh3) {
	h3lovalue2 = h3lovalue1;/* previous lovalue1 */
	h3hivalue2 = h3hivalue1;/* previous hivalue1 */
	h3dummy2 = h3dummy1;	/* previous dummy1 */

	h3lovalue1 = h3hivalue2;
	bessi0term = bessI0(besselarg);
	h3hivalue1 =		/* LTRArlcH3dashIntFunc(hilimit1,T,beta); */
	    ((hilimit1 <= T) || (beta == 0.0)) ? 0.0 : expterm * bessi0term - expbetaTterm;
	h3dummy1 = intlinfunc(lolimit1, hilimit1, h3lovalue1, h3hivalue1, lolimit1, hilimit1) / delta1;

	*(h3dashcoeffs + i) = h3dummy1 - h3dummy2;
	if (fabs(*(h3dashcoeffs + i)) <= h3relval)
	  doh3 = 0;
      } else
	*(h3dashcoeffs + i) = 0.0;
    }
  }
  *auxindexptr = auxindex;
}

/*
 * LTRAstraightLineCheck - takes the co-ordinates of three points, finds the
 * area of the triangle enclosed by these points and compares this area with
 * the area of the quadrilateral formed by the line between the first point
 * and the third point, the perpendiculars from the first and third points to
 * the x-axis, and the x-axis. If within reltol, then it returns 1, else 0.
 * The purpose of this function is to determine if three points lie
 * acceptably close to a straight line. This area criterion is used because
 * it is related to integrals and convolution
 */

int
LTRAstraightLineCheck(double x1, double y1, double x2, double y2, double x3, double y3, double reltol, double abstol)
{
  /*
   * double asqr, bsqr, csqr, c, c1sqr; double htsqr;
   */
  double TRarea, QUADarea1, QUADarea2, QUADarea3, area;

  /*
   * asqr = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1); bsqr = (x3-x2)*(x3-x2) +
   * (y3-y2)*(y3-y2); csqr = (x3-x1)*(x3-x1) + (y3-y1)*(y3-y1); c =
   * sqrt(csqr); c1sqr = (asqr - bsqr + csqr)/(2*c); c1sqr *= c1sqr; htsqr =
   * asqr - c1sqr; TRarea = c*sqrt(htsqr)*0.5;
   */
  /*
   * this should work if y1,y2,y3 all have the same sign and x1,x2,x3 are in
   * increasing order
   */

  QUADarea1 = (fabs(y2) + fabs(y1)) * 0.5 * fabs(x2 - x1);
  QUADarea2 = (fabs(y3) + fabs(y2)) * 0.5 * fabs(x3 - x2);
  QUADarea3 = (fabs(y3) + fabs(y1)) * 0.5 * fabs(x3 - x1);
  TRarea = fabs(QUADarea3 - QUADarea1 - QUADarea2);
  area = QUADarea1 + QUADarea2;
  if (area * reltol + abstol > TRarea)
    return (1);
  else
    return (0);
}


/*
 * i is thestrchr of the latest value, a,b,c values correspond to values at
 * t_{i-2}, t{i-1} and t_i
 */

#define SECONDDERIV(i,a,b,c) (oof = (i==ckt->CKTtimeIndex+1?curtime:\
*(ckt->CKTtimePoints+i)),\
(( c - b )/(oof-*(ckt->CKTtimePoints+i-1)) -\
( b - a )/(*(ckt->CKTtimePoints+i-1)-\
*(ckt->CKTtimePoints+i-2)))/(oof - \
*(ckt->CKTtimePoints+i-2)))

/*
 * LTRAlteCalculate - returns sum of the absolute values of the total local
 * truncation error of the 2 equations for the LTRAline
 */


double
LTRAlteCalculate(CKTcircuit *ckt, GENmodel *genmodel, GENinstance *geninstance, double curtime)
{
  LTRAmodel *model = (LTRAmodel *) genmodel;
  LTRAinstance *instance = (LTRAinstance *) geninstance;
  double h1dashTfirstCoeff;
  double h2TfirstCoeff = 0.0;
  double h3dashTfirstCoeff = 0.0;
  double dashdash;
  double oof;
  double hilimit1, lolimit1, hivalue1, lovalue1, f1i, g1i;
  double eq1LTE = 0.0, eq2LTE = 0.0;
  int auxindex = 0, tdover, i, exact;

  switch (model->LTRAspecialCase) {

  case LTRA_MOD_LC:
  case LTRA_MOD_RG:
    return (0.0);
    break;

  case LTRA_MOD_RLC:

    if (curtime > model->LTRAtd) {
      tdover = 1;

      exact = 0;
      for (i = ckt->CKTtimeIndex; i >= 0; i--) {
	if (curtime - *(ckt->CKTtimePoints + i)
	    == model->LTRAtd) {
	  exact = 1;
	  break;
	}
	if (curtime - *(ckt->CKTtimePoints + i)
	    > model->LTRAtd)
	  break;
      }

#ifdef LTRADEBUG
      if ((i < 0) || ((i == 0) && (exact == 1)))
	printf("LTRAlteCalculate: i <= 0: some mistake!\n");
#endif

      if (exact == 1) {
	auxindex = i - 1;
      } else {
	auxindex = i;
      }

    } else {
      tdover = 0;
    }

    hilimit1 = curtime - *(ckt->CKTtimePoints + ckt->CKTtimeIndex);
    lolimit1 = 0.0;
    hivalue1 = LTRArlcH1dashTwiceIntFunc(hilimit1, model->LTRAbeta);
    lovalue1 = 0.0;

    f1i = hivalue1;
    g1i = intlinfunc(lolimit1, hilimit1, lovalue1, hivalue1,
	lolimit1, hilimit1);
    h1dashTfirstCoeff = 0.5 * f1i *
	(curtime - *(ckt->CKTtimePoints + ckt->CKTtimeIndex)) - g1i;

    if (tdover) {
      hilimit1 = curtime - *(ckt->CKTtimePoints + auxindex);
      lolimit1 = *(ckt->CKTtimePoints + ckt->CKTtimeIndex) - *(ckt->CKTtimePoints + auxindex);
      lolimit1 = MAX(model->LTRAtd, lolimit1);

      /*
       * are the following really doing the operations in the write-up?
       */
      hivalue1 = LTRArlcH2Func(hilimit1, model->LTRAtd, model->LTRAalpha, model->LTRAbeta);
      lovalue1 = LTRArlcH2Func(lolimit1, model->LTRAtd, model->LTRAalpha, model->LTRAbeta);
      f1i = twiceintlinfunc(lolimit1, hilimit1, lolimit1, lovalue1, hivalue1, lolimit1,
	  hilimit1);
      g1i = thriceintlinfunc(lolimit1, hilimit1, lolimit1, lolimit1, lovalue1,
	  hivalue1, lolimit1, hilimit1);
      h2TfirstCoeff = 0.5 * f1i * (curtime - model->LTRAtd - *(ckt->CKTtimePoints + auxindex)) - g1i;

      hivalue1 = LTRArlcH3dashIntFunc(hilimit1, model->LTRAtd, model->LTRAbeta);
      lovalue1 = LTRArlcH3dashIntFunc(lolimit1, model->LTRAtd, model->LTRAbeta);
      f1i = intlinfunc(lolimit1, hilimit1, lovalue1, hivalue1, lolimit1,
	  hilimit1);
      g1i = twiceintlinfunc(lolimit1, hilimit1, lolimit1, lovalue1,
	  hivalue1, lolimit1, hilimit1);
      h3dashTfirstCoeff = 0.5 * f1i * (curtime - model->LTRAtd - *(ckt->CKTtimePoints + auxindex)) - g1i;
    }
    /* LTEs for convolution with v1 */
    /* get divided differences for v1 (2nd derivative estimates) */

    /*
     * no need to subtract operating point values because taking differences
     * anyway
     */

    dashdash = SECONDDERIV(ckt->CKTtimeIndex + 1,
	*(instance->LTRAv1 + ckt->CKTtimeIndex - 1),
	*(instance->LTRAv1 + ckt->CKTtimeIndex),
	*(ckt->CKTrhsOld + instance->LTRAposNode1) -
	*(ckt->CKTrhsOld + instance->LTRAnegNode1));
    eq1LTE += model->LTRAadmit * fabs(dashdash *
	h1dashTfirstCoeff);


    /*
     * not bothering to interpolate since everything is approximate anyway
     */

    if (tdover) {

      dashdash = SECONDDERIV(auxindex + 1,
	  *(instance->LTRAv1 + auxindex - 1),
	  *(instance->LTRAv1 + auxindex),
	  *(instance->LTRAv1 + auxindex + 1));

      eq2LTE += model->LTRAadmit * fabs(dashdash *
	  h3dashTfirstCoeff);

    }
    /* end LTEs for convolution with v1 */

    /* LTEs for convolution with v2 */
    /* get divided differences for v2 (2nd derivative estimates) */

    dashdash = SECONDDERIV(ckt->CKTtimeIndex + 1,
	*(instance->LTRAv2 + ckt->CKTtimeIndex - 1),
	*(instance->LTRAv2 + ckt->CKTtimeIndex),
	*(ckt->CKTrhsOld + instance->LTRAposNode2) -
	*(ckt->CKTrhsOld + instance->LTRAnegNode2));

    eq2LTE += model->LTRAadmit * fabs(dashdash *
	h1dashTfirstCoeff);


    if (tdover) {


      dashdash = SECONDDERIV(auxindex + 1,
	  *(instance->LTRAv2 + auxindex - 1),
	  *(instance->LTRAv2 + auxindex),
	  *(instance->LTRAv2 + auxindex + 1));

      eq1LTE += model->LTRAadmit * fabs(dashdash *
	  h3dashTfirstCoeff);

    }
    /* end LTEs for convolution with v2 */

    /* LTE for convolution with i1 */
    /* get divided differences for i1 (2nd derivative estimates) */

    if (tdover) {


      dashdash = SECONDDERIV(auxindex + 1,
	  *(instance->LTRAi1 + auxindex - 1),
	  *(instance->LTRAi1 + auxindex),
	  *(instance->LTRAi1 + auxindex + 1));

      eq2LTE += fabs(dashdash * h2TfirstCoeff);


    }
    /* end LTE for convolution with i1 */

    /* LTE for convolution with i2 */
    /* get divided differences for i2 (2nd derivative estimates) */

    if (tdover) {

      dashdash = SECONDDERIV(auxindex + 1,
	  *(instance->LTRAi2 + auxindex - 1),
	  *(instance->LTRAi2 + auxindex),
	  *(instance->LTRAi2 + auxindex + 1));

      eq1LTE += fabs(dashdash * h2TfirstCoeff);

    }
    /* end LTE for convolution with i1 */

    break;

  case LTRA_MOD_RC:

    hilimit1 = curtime - *(ckt->CKTtimePoints + ckt->CKTtimeIndex);
    lolimit1 = 0.0;

    hivalue1 = LTRArcH1dashTwiceIntFunc(hilimit1, model->LTRAcByR);
    lovalue1 = 0.0;

    f1i = hivalue1;
    g1i = intlinfunc(lolimit1, hilimit1, lovalue1, hivalue1, lolimit1, hilimit1);
    h1dashTfirstCoeff = 0.5 * f1i * (curtime - *(ckt->CKTtimePoints + ckt->CKTtimeIndex)) - g1i;




    hivalue1 = LTRArcH2TwiceIntFunc(hilimit1, model->LTRArclsqr);
    lovalue1 = 0.0;

    f1i = hivalue1;
    g1i = intlinfunc(lolimit1, hilimit1, lovalue1, hivalue1, lolimit1, hilimit1);
    h1dashTfirstCoeff = 0.5 * f1i * (curtime - *(ckt->CKTtimePoints + ckt->CKTtimeIndex)) - g1i;





    hivalue1 = LTRArcH2TwiceIntFunc(hilimit1, model->LTRArclsqr);
    lovalue1 = 0.0;

    f1i = hivalue1;
    g1i = intlinfunc(lolimit1, hilimit1, lovalue1,
	hivalue1, lolimit1, hilimit1);
    h1dashTfirstCoeff = 0.5 * f1i * (curtime -
	*(ckt->CKTtimePoints + ckt->CKTtimeIndex)) - g1i;



    /* LTEs for convolution with v1 */
    /* get divided differences for v1 (2nd derivative estimates) */

    /*
     * no need to subtract operating point values because taking differences
     * anyway
     */

    dashdash = SECONDDERIV(ckt->CKTtimeIndex + 1,
	*(instance->LTRAv1 + ckt->CKTtimeIndex - 1),
	*(instance->LTRAv1 + ckt->CKTtimeIndex),
	*(ckt->CKTrhsOld + instance->LTRAposNode1) -
	*(ckt->CKTrhsOld + instance->LTRAnegNode1));
    eq1LTE += fabs(dashdash * h1dashTfirstCoeff);
    eq2LTE += fabs(dashdash * h3dashTfirstCoeff);

    /* end LTEs for convolution with v1 */

    /* LTEs for convolution with v2 */
    /* get divided differences for v2 (2nd derivative estimates) */

    dashdash = SECONDDERIV(ckt->CKTtimeIndex + 1,
	*(instance->LTRAv2 + ckt->CKTtimeIndex - 1),
	*(instance->LTRAv2 + ckt->CKTtimeIndex),
	*(ckt->CKTrhsOld + instance->LTRAposNode2) -
	*(ckt->CKTrhsOld + instance->LTRAnegNode2));

    eq2LTE += fabs(dashdash * h1dashTfirstCoeff);
    eq1LTE += fabs(dashdash * h3dashTfirstCoeff);

    /* end LTEs for convolution with v2 */

    /* LTE for convolution with i1 */
    /* get divided differences for i1 (2nd derivative estimates) */

    dashdash = SECONDDERIV(ckt->CKTtimeIndex + 1,
	*(instance->LTRAi1 + ckt->CKTtimeIndex - 1),
	*(instance->LTRAi1 + ckt->CKTtimeIndex),
	*(ckt->CKTrhsOld + instance->LTRAbrEq1));

    eq2LTE += fabs(dashdash * h2TfirstCoeff);


    /* end LTE for convolution with i1 */

    /* LTE for convolution with i2 */
    /* get divided differences for i2 (2nd derivative estimates) */

    dashdash = SECONDDERIV(ckt->CKTtimeIndex + 1,
	*(instance->LTRAi2 + ckt->CKTtimeIndex - 1),
	*(instance->LTRAi2 + ckt->CKTtimeIndex),
	*(ckt->CKTrhsOld + instance->LTRAbrEq2));


    eq1LTE += fabs(dashdash * h2TfirstCoeff);


    /* end LTE for convolution with i1 */

    break;

  default:
    return (1 /* error */ );
  }


#ifdef LTRADEBUG
  fprintf(stdout, "%s: LTE/input for Eq1 at time %g is: %g\n",
      instance->LTRAname, curtime, eq1LTE / instance->LTRAinput1);
  fprintf(stdout, "%s: LTE/input for Eq2 at time %g is: %g\n",
      instance->LTRAname, curtime, eq2LTE / instance->LTRAinput1);
  fprintf(stdout, "\n");
#endif

  return (fabs(eq1LTE) + fabs(eq2LTE));
}

/*********************************************************************/
/****************** old stuff, retained for historical interest ******/
/*********************************************************************/



/*
 * LTRAcoeffSetup sets up the coefficient list for the convolution, returns
 * the coefficient at (current_timepoint-T)
 */

/*
 * double
 * LTRAcoeffSetup(coefflist,listsize,T,firstvalue,valuelist,curtime,timelist,t
 * imeindex,auxindexptr) double *coefflist, *timelist, *valuelist; int
 * listsize, timeindex; double T, firstvalue, curtime; int *auxindexptr;
 * 
 * { unsigned exact; double returnval, delta1, delta2; double dummy1, dummy2;
 * double lolimit1,lolimit2,hilimit1,hilimit2; double
 * lovalue1,lovalue2,hivalue1,hivalue2; int i,auxindex;
 */

/* coefflist should already have been allocated to the necessary size */

/*
 * #ifdef LTRAdebug if (listsize <= timeindex) { printf("LTRAcoeffSetup: not
 * enough space in coefflist\n"); } #endif
 * 
 */

/*
 * we assume a piecewise linear function, and we calculate the coefficients
 * using this assumption in the integration of the function
 */

/*
 * if (T == 0.0) { auxindex = timeindex; } else {
 * 
 * if (curtime - T <= 0.0) { for (i =0; i<= timeindex; i++) { (coefflist + i) =
 * 0.0; } auxindexptr = 0; return(0.0); } else { exact = 0; for (i =
 * timeindex; i>= 0; i--) { if (curtime - *(timelist + i) ==  T) { exact =1;
 * break; } if (curtime - *(timelist + i) > T) break; }
 * 
 * #ifdef LTRADEBUG if ((i < 0) || ((i==0) && (exact==1)))
 * printf("LTRAcoeffSetup: i <= 0: some mistake!\n"); #endif
 * 
 * if (exact == 1) { auxindex = i-1; } else { auxindex = i; } } }
 */
/* the first coefficient */

/*
 * delta1 = curtime -T - *(timelist + auxindex); lolimit1 = T; hilimit1 = T +
 * delta1; lovalue1 = firstvalue; hivalue1 = *(valuelist + auxindex); dummy1
 * = twiceintlinfunc(lolimit1,hilimit1,lolimit1,lovalue1,
 * hivalue1,lolimit1,hilimit1)/delta1; returnval = dummy1;
 * 
 */


/* the coefficients for the rest of the timepoints */

/*
 * for (i=auxindex; i>0; i--) {
 * 
 * delta2 = delta1;
 *//* previous delta1 */
/* lolimit2 = lolimit1; *//* previous lolimit1 */
/* hilimit2 = hilimit1; *//* previous hilimit1 */
/* lovalue2 = lovalue1; *//* previous lovalue1 */
/* hivalue2 = hivalue1; *//* previous hivalue1 */
/* dummy2 = dummy1; *//* previous dummy1 */
/*
 * delta1 = *(timelist + i) - *(timelist + i - 1); lolimit1 = hilimit2;
 * hilimit1 = curtime - *(timelist + i - 1); lovalue1 = hivalue2; hivalue1 =
 * *(valuelist + i - 1); dummy1 = twiceintlinfunc(lolimit1,hilimit1,lolimit1,
 * lovalue1,hivalue1,lolimit1,hilimit1)/delta1;
 * 
 * (coefflist + i) = dummy1 - dummy2 + intlinfunc(lolimit2,hilimit2,
 * lovalue2,hivalue2,lolimit2,hilimit2); } auxindexptr = auxindex;
 * return(returnval); }
 */

/*
 * LTRAtCoeffSetup sets up the coefficient list for the LTE calculation,
 * returns the coefficient at (current_timepoint-T)
 */

/*
 * double LTRAtCoeffSetup(coefflist,listsize,T,valuelist,
 * firstothervalue,othervaluelist,curtime,timelist,timeindex, auxindexptr,
 * ltecontype) double *coefflist, *timelist, *valuelist, *othervaluelist; int
 * listsize, timeindex; double T, firstothervalue, curtime; int *auxindexptr,
 * ltecontype;
 * 
 * { unsigned exact; double returnval, delta; double dummy; double f1i, f2i,
 * g1i, g2i; double lolimit1, hilimit1; double lovalue1, hivalue1; double
 * lolimit2, hilimit2; double lovalue2, hivalue2; double firstint1 = 0.0,
 * firstint2 = 0.0; double secondint1 = 0.0, secondint2 = 0.0; int
 * i,auxindex;
 * 
 */
/* coefflist should already have been allocated to the necessary size */

/*
 * #ifdef LTRAdebug if (listsize <= timeindex) { printf("LTRAtCoeffSetup: not
 * enough space in coefflist\n"); } #endif
 * 
 */

/*
 * we assume a piecewise linear function, and we calculate the coefficients
 * using this assumption in the integration of the function
 */

/*
 * if (T == 0.0) { auxindex = timeindex; } else {
 * 
 * if (curtime - T <= 0.0) { for (i =0; i<= timeindex; i++) { (coefflist + i) =
 * 0.0; } auxindexptr = 0; return(0.0); } else { exact = 0; for (i =
 * timeindex; i>= 0; i--) { if (curtime - *(timelist + i) ==  T) { exact =1;
 * break; } if (curtime - *(timelist + i) > T) break; }
 * 
 * #ifdef LTRADEBUG if ((i < 0) || ((i==0) && (exact==1)))
 * printf("LTRAcoeffSetup: i <= 0: some mistake!\n"); #endif
 * 
 * if (exact == 1) { auxindex = i-1; } else { auxindex = i; } } }
 */
/* the first coefficient */

/* i = n in the write-up */

/*
 * hilimit1 = curtime - *(timelist + auxindex); hivalue1 = *(valuelist +
 * auxindex); lolimit1 = *(timelist + timeindex) - *(timelist + auxindex);
 * lolimit1 = MAX(T,lolimit1); lovalue1 = firstothervalue; f1i =
 * twiceintlinfunc(lolimit1,hilimit1,lolimit1,lovalue1,hivalue1,lolimit1,
 * hilimit1); g1i =
 * thriceintlinfunc(lolimit1,hilimit1,lolimit1,lolimit1,lovalue1,
 * hivalue1,lolimit1,hilimit1); returnval = 0.5*f1i*(curtime-T-
 * *(timelist+auxindex)) - g1i;
 * 
 */
/* the coefficients for the rest of the timepoints */

/*
 * if (ltecontype != LTRA_MOD_HALFCONTROL) { for (i=auxindex; i>0; i--) {
 * 
 * lolimit2 = lolimit1;
 *//* previous lolimit1 */
/* hilimit2 = hilimit1; *//* previous hilimit1 */
/* lovalue2 = lovalue1; *//* previous lovalue1 */
/* hivalue2 = hivalue1; *//* previous hivalue1 */
/* f2i = f1i; *//* previous f1i */
/* g2i = g1i; *//* previous g1i */
/* firstint2 = firstint1; *//* previous firstint1 */
/* secondint2 = secondint1; *//* previous secondint1 */
/*
 * lolimit1 = *(timelist + timeindex) - *(timelist + i - 1); hilimit1 =
 * curtime - *(timelist + i - 1); lovalue1 = *(othervaluelist + i - 1);
 * hivalue1 = *(valuelist + i - 1); firstint1 += intlinfunc(lolimit2,
 * lolimit1, lovalue2, lovalue1, lolimit2, lolimit1); secondint1 +=
 * (lolimit1-lolimit2)*firstint2 + twiceintlinfunc(
 * lolimit2,lolimit1,lolimit2,lovalue2,lovalue1,lolimit2,lolimit1); f1i =
 * twiceintlinfunc(lolimit1,hilimit1,lolimit1,lovalue1,hivalue1,
 * lolimit1,hilimit1) + firstint1*(hilimit1-lolimit1); g1i =
 * thriceintlinfunc(lolimit1,hilimit1,lolimit1,lolimit1,
 * lovalue1,hivalue1,lolimit1,hilimit1) +
 * (hilimit1-lolimit1)*(hilimit1-lolimit1)*0.5*firstint1 +
 * (hilimit1-lolimit1)*secondint1;
 * 
 * (coefflist + i) = g2i - g1i + 0.5*(f1i + f2i)*(*(timelist+i) -
 * (timelist+i-1)); } } auxindexptr = auxindex; return(returnval); }
 */


/*
 * formulae taken from the Handbook of Mathematical Functions by Milton
 * Abramowitz and Irene A. Stegan, page 378, formulae 9.8.1 - 9.8.4
 */

/*
 * double bessi0(x) double x; { double t, tsq, oneovert, result, dummy; int i;
 * static double coeffs1[7], coeffs2[9];
 * 
 * coeffs1[0] = 1.0; coeffs1[1] = 3.5156229; coeffs1[2] = 3.0899424; coeffs1[3]
 * = 1.2067492; coeffs1[4] = 0.2659732; coeffs1[5] = 0.0360768; coeffs1[6] =
 * 0.0045813;
 * 
 * coeffs2[0] = 0.39894228; coeffs2[1] = 0.01328592; coeffs2[2] = 0.00225319;
 * coeffs2[3] = -0.00157565; coeffs2[4] = 0.00916281; coeffs2[5] =
 * -0.02057706; coeffs2[6] = 0.02635537; coeffs2[7] = -0.01647633; coeffs2[8]
 * = 0.00392377;
 * 
 * t = x/3.75; dummy = 1.0;
 * 
 * if (fabs(t) <= 1) { tsq = t*t;
 * 
 * result = 1.0; for (i=1;i<=6;i++) { dummy *= tsq; ; result += dummy *
 * coeffs1[i]; } } else { oneovert = 1/fabs(t);
 * 
 * result = coeffs2[0]; for (i=1;i<=8;i++) { dummy *= oneovert; result +=
 * coeffs2[2] * dummy; } result *= exp(x) * sqrt(1/fabs(x)); }
 * return(result); }
 * 
 * double bessi1(x) double x; { double t, tsq, oneovert, result, dummy; int i;
 * static double coeffs1[7], coeffs2[9];
 * 
 * coeffs1[0] = 0.5; coeffs1[1] = 0.87890594; coeffs1[2] = 0.51498869;
 * coeffs1[3] = 0.15084934; coeffs1[4] = 0.02658733; coeffs1[5] = 0.00301532;
 * coeffs1[6] = 0.00032411;
 * 
 * coeffs2[0] = 0.39894228; coeffs2[1] = -0.03988024; coeffs2[2] = -0.00362018;
 * coeffs2[3] = 0.00163801; coeffs2[4] = -0.01031555; coeffs2[5] =
 * 0.02282967; coeffs2[6] = -0.02895312; coeffs2[7] = 0.01787654; coeffs2[8]
 * = -0.00420059;
 * 
 * t = x/3.75; dummy = 1.0;
 * 
 * if (fabs(t) <= 1) { tsq = t*t;
 * 
 * result = 0.5; for (i=1;i<=6;i++) { dummy *= tsq; ; result += dummy *
 * coeffs1[i]; } result *= x; } else { oneovert = 1/fabs(t);
 * 
 * result = coeffs2[0]; for (i=1;i<=8;i++) { dummy *= oneovert; result +=
 * coeffs2[2] * dummy; } result *= exp(x) * sqrt(1/fabs(x)); if (x < 0)
 * result = -result; } return(result); } */

/*
 * LTRAdivDiffs returns divided differences after 2 iterations, an
 * approximation to the second derivatives. The algorithm is picked up
 * directly from Tom Quarles' CKTterr.c; no attempt has been made to figure
 * out why it does what it does.
 */

/*
 * double LTRAdivDiffs(difflist, valuelist, firstvalue, curtime, timelist,
 * timeindex) double *difflist, *valuelist, firstvalue,  *timelist, curtime;
 * int timeindex;
 * 
 * { double *dtime, *diffs, returnval; int i,j;
 * 
 * diffs = TMALLOC(double, timeindex + 2);
 * dtime = TMALLOC(double, timeindex + 2);
 */

/* now divided differences */
/*
 * for(i=timeindex+1;i>=0;i--) { (diffs+i) = (i == timeindex+1 ? firstvalue :
 * *(valuelist + i)); } for(i=timeindex+1 ; i > 0 ; i--) { (dtime+i) = (i ==
 * timeindex+1? curtime: *(timelist + i)) - (timelist + i - 1); } j = 2;
 *//* for the second derivative */
/*
 * while(1) { for(i=timeindex + 1;i > 0; i--) { (diffs+i) = (*(diffs+i) -
 * *(diffs+i-1))/ *(dtime+i); } j--; if (j <= 0) break; for(i=timeindex+1;i >
 * 0;i--) { (dtime+i) = *(dtime+i-1) + (i == timeindex+1? curtime: *(timelist
 * + i)) - *(timelist + i - 1); } }
 * 
 * for (i = timeindex; i>=0 ; i--) { (difflist+i) = *(diffs+i); }
 * 
 * returnval = *(diffs+timeindex+1); FREE(dtime); FREE(diffs);
 */

/* difflist[0] is going to be bad */
/*
 * return(returnval); } */


/*
 * LTRAlteCalculate - returns sum of the absolute values of the total local
 * truncation error of the 2 equations for the LTRAline
 */

/*
 * double LTRAlteCalculate(ckt,model,instance,curtime) CKTcircuit *ckt;
 * LTRAmodel *model; register LTRAinstance *instance; double
 * curtime;
 * 
 * { double *h1dashTcoeffs, h1dashTfirstCoeff; double *h2Tcoeffs, h2TfirstCoeff;
 * double *h3dashTcoeffs, h3dashTfirstCoeff; double *SecondDerivs,
 * FirstSecondDeriv; double t1, t2, t3, f1, f2, f3; double eq1LTE=0.0,
 * eq2LTE=0.0; int isaved, tdover, i;
 * 
 * if (curtime > model->LTRAtd) { tdover = 1; } else { tdover = 0; }
 * 
 * h1dashTcoeffs = TMALLOC(double, model->LTRAmodelListSize);
 * h2Tcoeffs = TMALLOC(double, model->LTRAmodelListSize);
 * h3dashTcoeffs = TMALLOC(double, model->LTRAmodelListSize);
 * SecondDerivs = TMALLOC(double, model->LTRAmodelListSize);
 * 
 */

/*
 * note that other OthVals have been set up in LTRAaccept, and Values in
 * LTRAload
 */


/*
 * h1dashTfirstCoeff = LTRAtCoeffSetup(h1dashTcoeffs,
 * model->LTRAmodelListSize, 0.0, model->LTRAh1dashValues,
 * model->LTRAh1dashFirstVal, model->LTRAh1dashOthVals, curtime,
 * ckt->CKTtimePoints,ckt->CKTtimeIndex, &(model->LTRAh1dashIndex),
 * model->LTRAlteConType);
 * 
 * if (tdover) {
 * 
 * h2TfirstCoeff = LTRAtCoeffSetup(h2Tcoeffs, model->LTRAmodelListSize,
 * model->LTRAtd, model->LTRAh2Values, model->LTRAh2FirstOthVal,
 * model->LTRAh2OthVals, curtime, ckt->CKTtimePoints, ckt->CKTtimeIndex,
 * &(model->LTRAh2Index), model->LTRAlteConType);
 * 
 * h3dashTfirstCoeff = LTRAtCoeffSetup(h3dashTcoeffs, model->LTRAmodelListSize,
 * model->LTRAtd, model->LTRAh3dashValues, model->LTRAh3dashFirstOthVal,
 * model->LTRAh3dashOthVals, curtime, ckt->CKTtimePoints,ckt->CKTtimeIndex,
 * &(model->LTRAh3dashIndex), model->LTRAlteConType);
 */

/* setting up the coefficients for interpolation */
/*
 * for (i = ckt->CKTtimeIndex; i>= 0; i--) { if (*(ckt->CKTtimePoints + i) <
 * curtime - model->LTRAtd) { break; } } #ifdef LTRAdebug if (i ==
 * ckt->CKTtimeIndex) || (i == -1) { printf("LTRAtrunc: mistake: cannot find
 * delayed timepoint\n"); } #endif t1 = *(ckt->CKTtimePoints + i - 1); t2 =
 * *(ckt->CKTtimePoints + i); t3 = *(ckt->CKTtimePoints + i + 1);
 * 
 * LTRAquadInterp(curtime - model->LTRAtd, t1,t2,t3,&f1,&f2,&f3);
 * 
 * isaved = i; } */

/* interpolation coefficients set-up */

/* LTEs for convolution with v1 */
/* get divided differences for v1 (2nd derivative estimates) */

/*
 * no need to subtract operating point values because taking differences
 * anyway
 */

/*
 * FirstSecondDeriv = LTRAdivDiffs(SecondDerivs,instance->LTRAv1,
 * (ckt->CKTrhsOld + instance->LTRAposNode1) - *(ckt->CKTrhsOld +
 * instance->LTRAnegNode1),curtime, ckt->CKTtimePoints,ckt->CKTtimeIndex);
 * 
 * eq1LTE += model->LTRAadmit*fabs(FirstSecondDeriv * h1dashTfirstCoeff);
 * 
 * if (model->LTRAlteConType != LTRA_MOD_HALFCONTROL) { for (i =
 * model->LTRAh1dashIndex; i > 0; i--) { if ((*(SecondDerivs+i) != 0.0) &&
 * (*(h1dashTcoeffs+i)!=0.0)) { eq1LTE +=
 * model->LTRAadmit*fabs(*(SecondDerivs+i) * (h1dashTcoeffs+i)); } } }
 * 
 */
/* interpolate */
/*
 * if (tdover) {
 * 
 * FirstSecondDeriv = *(SecondDerivs + isaved - 1) * f1 + *(SecondDerivs +
 * isaved) * f2 + *(SecondDerivs + isaved + 1) * f3;
 * 
 * eq2LTE += model->LTRAadmit*fabs(FirstSecondDeriv * h3dashTfirstCoeff);
 * 
 * if (model->LTRAlteConType != LTRA_MOD_HALFCONTROL) { for (i =
 * model->LTRAh3dashIndex; i > 0; i--) { if ((*(SecondDerivs+i) != 0.0) &&
 * (*(h3dashTcoeffs+i)!=0.0)) { eq2LTE +=
 * model->LTRAadmit*fabs(*(SecondDerivs+i) * (h3dashTcoeffs+i)); } } } }
 */
/* end LTEs for convolution with v1 */

/* LTEs for convolution with v2 */
/* get divided differences for v2 (2nd derivative estimates) */

/*
 * FirstSecondDeriv = LTRAdivDiffs(SecondDerivs,instance->LTRAv2,
 * (ckt->CKTrhsOld + instance->LTRAposNode2) - *(ckt->CKTrhsOld +
 * instance->LTRAnegNode2),curtime, ckt->CKTtimePoints,ckt->CKTtimeIndex);
 * 
 * eq2LTE += model->LTRAadmit*fabs(FirstSecondDeriv * h1dashTfirstCoeff);
 * 
 * if (model->LTRAlteConType != LTRA_MOD_HALFCONTROL) { for (i =
 * model->LTRAh1dashIndex; i > 0; i--) { if ((*(SecondDerivs+i) != 0.0) &&
 * (*(h1dashTcoeffs+i)!=0.0)) { eq2LTE +=
 * model->LTRAadmit*fabs(*(SecondDerivs+i) * (h1dashTcoeffs+i)); } } }
 * 
 * if (tdover) {
 */
/* interpolate */

/*
 * FirstSecondDeriv = *(SecondDerivs + isaved - 1) * f1 + *(SecondDerivs +
 * isaved) * f2 + *(SecondDerivs + isaved + 1) * f3;
 * 
 * eq1LTE += model->LTRAadmit*fabs(FirstSecondDeriv * h3dashTfirstCoeff);
 * 
 * if (model->LTRAlteConType != LTRA_MOD_HALFCONTROL) { for (i =
 * model->LTRAh3dashIndex; i > 0; i--) { if ((*(SecondDerivs+i) != 0.0) &&
 * (*(h3dashTcoeffs+i)!=0.0)) { eq1LTE +=
 * model->LTRAadmit*fabs(*(SecondDerivs+i) * (h3dashTcoeffs+i)); } } } }
 * 
 */
/* end LTEs for convolution with v2 */

/* LTE for convolution with i1 */
/* get divided differences for i1 (2nd derivative estimates) */

/*
 * if (tdover) { FirstSecondDeriv =
 * LTRAdivDiffs(SecondDerivs,instance->LTRAi1, (ckt->CKTrhsOld +
 * instance->LTRAbrEq1),curtime, ckt->CKTtimePoints,ckt->CKTtimeIndex);
 * 
 */
/* interpolate */

/*
 * FirstSecondDeriv = *(SecondDerivs + isaved - 1) * f1 + *(SecondDerivs +
 * isaved) * f2 + *(SecondDerivs + isaved + 1) * f3;
 * 
 * eq2LTE += fabs(FirstSecondDeriv * h2TfirstCoeff);
 * 
 * if (model->LTRAlteConType != LTRA_MOD_HALFCONTROL) { for (i =
 * model->LTRAh2Index; i > 0; i--) { if ((*(SecondDerivs+i) != 0.0) &&
 * (*(h2Tcoeffs+i)!=0.0)) { eq2LTE += model->LTRAadmit*fabs(*(SecondDerivs+i) *
 * (h2Tcoeffs+i)); } } }
 * 
 * } */
/* end LTE for convolution with i1 */

/* LTE for convolution with i2 */
/* get divided differences for i2 (2nd derivative estimates) */

/*
 * if (tdover) { FirstSecondDeriv =
 * LTRAdivDiffs(SecondDerivs,instance->LTRAi2, (ckt->CKTrhsOld +
 * instance->LTRAbrEq2),curtime, ckt->CKTtimePoints,ckt->CKTtimeIndex);
 * 
 */
/* interpolate */

/*
 * FirstSecondDeriv = *(SecondDerivs + isaved - 1) * f1 + *(SecondDerivs +
 * isaved) * f2 + *(SecondDerivs + isaved + 1) * f3;
 * 
 * eq1LTE += fabs(FirstSecondDeriv * h2TfirstCoeff);
 * 
 * if (model->LTRAlteConType != LTRA_MOD_HALFCONTROL) { for (i =
 * model->LTRAh2Index; i > 0; i--) { if ((*(SecondDerivs+i) != 0.0) &&
 * (*(h2Tcoeffs+i)!=0.0)) { eq1LTE += model->LTRAadmit*fabs(*(SecondDerivs+i) *
 * (h2Tcoeffs+i)); } } } }
 * 
 */
/* end LTE for convolution with i1 */

#ifdef LTRADEBUG
/*
 * fprintf(stdout,"%s: LTE/input for Eq1 at time %g is: %g\n",
 * instance->LTRAname, curtime, eq1LTE/instance->LTRAinput1);
 * fprintf(stdout,"%s: LTE/input for Eq2 at time %g is: %g\n",
 * instance->LTRAname, curtime, eq2LTE/instance->LTRAinput1);
 * fprintf(stdout,"\n");
 */
#endif

/*
 * FREE(SecondDerivs); FREE(h1dashTcoeffs); FREE(h2Tcoeffs);
 * FREE(h3dashTcoeffs);
 * 
 * return(fabs(eq1LTE) + fabs(eq2LTE)); }
 */

/*
 * LTRAh3dashCoeffSetup sets up the coefficient list for h3dash for the
 * special case where G=0, * returns the coefficient at (current_timepoint-T)
 */

/*
 * double
 * LTRAh3dashCoeffSetup(coefflist,listsize,T,beta,curtime,timelist,timeindex,a
 * uxindexptr) double *coefflist, *timelist; int listsize, timeindex; double
 * T, curtime, beta; int *auxindexptr;
 * 
 * { unsigned exact; double returnval, delta1, delta2; double dummy1, dummy2;
 * double lolimit1,lolimit2,hilimit1,hilimit2; double
 * lovalue1,lovalue2,hivalue1,hivalue2; int i,auxindex;
 * 
 */
/* coefflist should already have been allocated to the necessary size */

/*
 * #ifdef LTRAdebug if (listsize <= timeindex) { printf("LTRAcoeffSetup: not
 * enough space in coefflist\n"); } #endif
 * 
 * 
 */
/*
 * we assume a piecewise linear function, and we calculate the coefficients
 * using this assumption in the integration of the function
 */

/*
 * if (T == 0.0) { auxindex = timeindex; } else {
 * 
 * if (curtime - T <= 0.0) { for (i =0; i<= timeindex; i++) { (coefflist + i) =
 * 0.0; } auxindexptr = 0; return(0.0); } else { exact = 0; for (i =
 * timeindex; i>= 0; i--) { if (curtime - *(timelist + i) ==  T) { exact =1;
 * break; } if (curtime - *(timelist + i) > T) break; }
 * 
 * #ifdef LTRADEBUG if ((i < 0) || ((i==0) && (exact==1)))
 * printf("LTRAcoeffSetup: i <= 0: some mistake!\n"); #endif
 * 
 * if (exact == 1) { auxindex = i-1; } else { auxindex = i; } } }
 */
/* the first coefficient */

/*
 * delta1 = curtime -T - *(timelist + auxindex); lolimit1 = T; hilimit1 = T +
 * delta1; lovalue1 = 0.0;
 *//* E3dash should be consistent with this */
/*
 * hivalue1 = LTRArlcH3dashIntFunc(hilimit1,T,beta); dummy1 =
 * intlinfunc(lolimit1,hilimit1,lovalue1, hivalue1,lolimit1,hilimit1)/delta1;
 * returnval = dummy1;
 * 
 * 
 */
/* the coefficients for the rest of the timepoints */
/*
 * for (i=auxindex; i>0; i--) {
 * 
 * delta2 = delta1;
 *//* previous delta1 */
/* lolimit2 = lolimit1; *//* previous lolimit1 */
/* hilimit2 = hilimit1; *//* previous hilimit1 */
/* lovalue2 = lovalue1; *//* previous lovalue1 */
/* hivalue2 = hivalue1; *//* previous hivalue1 */
/* dummy2 = dummy1; *//* previous dummy1 */
/*
 * delta1 = *(timelist + i) - *(timelist + i - 1); lolimit1 = hilimit2;
 * hilimit1 = curtime - *(timelist + i - 1); lovalue1 = hivalue2; hivalue1 =
 * LTRArlcH3dashIntFunc(hilimit1,T,beta); dummy1 =
 * intlinfunc(lolimit1,hilimit1,lovalue1,hivalue1,lolimit1,hilimit1)/delta1;
 * 
 * (coefflist + i) = dummy1 - dummy2; } auxindexptr = auxindex;
 * return(returnval); }
 */

/*
 * LTRAh1dashCoeffSetup sets up the coefficient list for h1dash in the
 * special case where G=0 returns the coefficient at current_timepoint
 */
/*
 * double
 * LTRAh1dashCoeffSetup(coefflist,listsize,beta,curtime,timelist,timeindex,aux
 *strchrptr) double *coefflist, *timelist; int listsize, timeindex; double
 * beta, curtime; int *auxindexptr;
 * 
 * { double returnval, delta1, delta2; double dummy1, dummy2; double
 * lolimit1,lolimit2,hilimit1,hilimit2; double
 * lovalue1,lovalue2,hivalue1,hivalue2; int i,auxindex;
 * 
 */
/* coefflist should already have been allocated to the necessary size */

/*
 * #ifdef LTRAdebug if (listsize <= timeindex) {
 * printf("LTRAh1dashCoeffSetup: not enough space in coefflist\n"); } #endif
 * 
 * 
 * 
 * auxindex = timeindex;
 * 
 */
/* the first coefficient */
/*
 * delta1 = curtime - *(timelist + auxindex); lolimit1 = 0.0; hilimit1 =
 * delta1; lovalue1 = 0.0; hivalue1 =
 * LTRArlcH1dashTwiceIntFunc(hilimit1,beta); dummy1 = hivalue1/delta1;
 * returnval = dummy1;
 * 
 * 
 * 
 */
/* the coefficients for the rest of the timepoints */

/*
 * for (i=auxindex; i>0; i--) {
 * 
 * delta2 = delta1;
 *//* previous delta1 */
/* lolimit2 = lolimit1; *//* previous lolimit1 */
/* hilimit2 = hilimit1; *//* previous hilimit1 */
/* lovalue2 = lovalue1; *//* previous lovalue1 */
/* hivalue2 = hivalue1; *//* previous hivalue1 */
/* dummy2 = dummy1; *//* previous dummy1 */
/*
 * delta1 = *(timelist + i) - *(timelist + i - 1); lolimit1 = hilimit2;
 * hilimit1 = curtime - *(timelist + i - 1); lovalue1 = hivalue2; hivalue1 =
 * LTRArlcH1dashTwiceIntFunc(hilimit1,beta); dummy1 = (hivalue1 -
 * lovalue1)/delta1;
 * 
 * (coefflist + i) = dummy1 - dummy2; } auxindexptr = auxindex;
 * return(returnval); }
 */
