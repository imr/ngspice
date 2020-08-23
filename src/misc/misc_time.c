/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/

/*
 * Date and time utility functions
 */

#include "ngspice/ngspice.h"
#include <string.h>
#include "misc_time.h"


/* Return the date. Return value is static data. */

char *
datestring(void)
{

#ifdef HAVE_LOCALTIME
    static char tbuf[45];
    struct tm *tp;
    char *ap;
    size_t i;

    time_t tloc;
    time(&tloc);
    tp = localtime(&tloc);
    ap = asctime(tp);
    (void) sprintf(tbuf, "%.20s", ap);
    (void) strcat(tbuf, ap + 19);
    i = strlen(tbuf);
    tbuf[i - 1] = '\0';
    return (tbuf);

#else

    return ("today");

#endif
}

/* return time interval in seconds and milliseconds */

#ifdef HAVE_FTIME

void timediff(struct timeb* now, struct timeb* begin, int* sec, int* msec)
{

    *msec = (int)now->millitm - (int)begin->millitm;
    *sec = (int)now->time - (int)begin->time;
    if (*msec < 0) {
        *msec += 1000;
        (*sec)--;
    }
    return;

}

#endif


/* Initialize time */

#ifdef HAVE_GETTIMEOFDAY
struct timeval timezero;
void timebegin(void) {
    gettimeofday(&timezero, NULL);
}

#else
#ifdef HAVE_TIMES
clock_t timezero;
void timebegin(void) {
    struct tms ruse;
    timezero = times(&ruse);
}

#else
#ifdef HAVE_FTIME
struct timeb timezero;
void timebegin(void) {
    ftime(&timezero);
}
#endif /* FTIME */
#endif /* TIMES */
#endif /* GETTIMEOFDAY */


/*
 * How many seconds have elapsed in running time.
 * This is the routine called in IFseconds
 */

double
seconds(void)
{
#ifdef HAVE_GETTIMEOFDAY
    struct timeval timenow;
    int sec, msec, usec;

    gettimeofday(&timenow, NULL);

    sec = (int) timenow.tv_sec - (int) timezero.tv_sec;
    usec = (int) timenow.tv_usec - (int) timezero.tv_usec;
    msec = usec / 1000; // Get rid of extra accuracy
    return(sec + (double) msec / 1000.0);

#else
#ifdef HAVE_TIMES
    struct tms ruse;
    long long msec;

    clock_t timenow = times(&ruse);
    double hz = (double) sysconf(_SC_CLK_TCK);
    msec = (timenow - timezero) / hz * 1000.0; // Get rid of extra accuracy
    return((double) msec / 1000.0);

#else
#ifdef HAVE_FTIME
    struct timeb timenow;
    int sec, msec;

    ftime(&timenow);

    sec = (int) timenow.time - (int) timezero.time;
    msec = (int) timenow.millitm - (int) timezero.millitm;
    return(sec + (double) msec / 1000.0);

#else /* unknown */
    return(-1.0);	/* Obvious error condition */

#endif /* FTIME */
#endif /* TIMES */
#endif /* GETTIMEOFDAY */
}
