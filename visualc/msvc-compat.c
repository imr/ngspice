#include <math.h>

/*
 * some rather simple minded replacements
 *  for functions missing in most msvc incarnations
 */

double
x_trunc(double x)
{
    return (x < 0) ? ceil(x) : floor(x);
}


double
x_nearbyint(double x)
{
    /* thats grossly incorrect, anyway, don't worry, be crappy ... */
    return floor(x + 0.5);
}


double
x_asinh(double x)
{
    return (x > 0) ? log(x + sqrt(x * x + 1.0)) : -log(-x + sqrt(x * x + 1.0));
}

double
x_acosh(double x)
{
    /* domain check (HUGE_VAL like gnu libc) */
    if (x < 1.0)
        return HUGE_VAL;
    else
        return log(x + sqrt(x * x - 1.0));
}

double
x_atanh(double x)
{
    /* domain check (HUGE_VAL like gnu libc) */
    if (fabs(x) >= 1.0)
        return HUGE_VAL;
    else
        return log((1.0 + x) / (1.0 - x)) / 2.0;
}
