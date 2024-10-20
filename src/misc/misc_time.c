/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/

/*
 * Date and time utility functions
 */

#include "ngspice/ngspice.h"
#include <string.h>
#include "misc_time.h"

#ifdef HAVE_LOCALTIME
#include <time.h>
#endif


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

#if defined HAVE_CLOCK_GETTIME || defined HAVE_GETTIMEOFDAY

void timediff(PortableTime *now, PortableTime *begin, int *sec, int *msec)
{

    *msec = (int) now->milliseconds - (int) begin->milliseconds;
    *sec = (int) now->seconds - (int) begin->seconds;
    if (*msec < 0) {
      *msec += 1000;
      (*sec)--;
    }
    return;

}

#ifdef HAVE_CLOCK_GETTIME
void get_portable_time(PortableTime *pt) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    
    pt->seconds = ts.tv_sec;
    pt->milliseconds = ts.tv_nsec / 1000000; // Convert nanoseconds to milliseconds
}
#else
void get_portable_time(PortableTime *pt) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    pt->seconds = tv.tv_sec;
    pt->milliseconds = tv.tv_usec / 1000;
}
#endif

#endif

/*
 * How many seconds have elapsed in running time.
 * This is the routine called in IFseconds
 */

double
seconds(void)
{
#ifdef HAVE_GETRUSAGE
    int ret;
    struct rusage ruse;

    memset(&ruse, 0, sizeof(ruse));
    ret = getrusage(RUSAGE_SELF, &ruse);
    if(ret == -1) {
      perror("getrusage(): ");
      return 1;
    }
    return ((double)ruse.ru_utime.tv_sec + (double) ruse.ru_utime.tv_usec / 1000000.0);
#else
#ifdef HAVE_TIMES

    struct tms tmsbuf;

    times(&tmsbuf);
    return((double) tmsbuf.tms_utime / HZ);

#else
#if defined HAVE_CLOCK_GETTIME || defined HAVE_GETTIMEOFDAY
    PortableTime timenow;
    PortableTime timebegin;
    int sec, msec;

    get_portable_time(&timenow);
    timediff(&timenow, &timebegin, &sec, &msec);
    return(sec + (double) msec / 1000.0);

#else /* unknown */

    return(-1.0);   /* Obvious error condition */

#endif /* GETTIMEOFDAY || CLOCK_GETTIME */
#endif /* TIMES */
#endif /* GETRUSAGE */
}
