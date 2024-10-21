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

#if defined HAVE_CLOCK_GETTIME || defined HAVE_GETTIMEOFDAY || defined HAVE_FTIME

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
#ifdef HAVE_GETTIMEOFDAY
void get_portable_time(PortableTime *pt) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    pt->seconds = tv.tv_sec;
    pt->milliseconds = tv.tv_usec / 1000; // Convert microseconds to milliseconds
}
#else
#ifdef HAVE_FTIME
void get_portable_time(PortableTime *pt) {
    struct timeb timenow;
    ftime(&timenow);
    pt->seconds = timenow.time;
    pt->milliseconds = timenow.millitm;
}
#endif
#endif
#endif

#endif

/*
 * How many seconds have elapsed in running time.
 * This is the routine called in IFseconds where start / stop is handled
 * and we don't need calculate timediff here.
 */

double
seconds(void)
{
#if defined HAVE_CLOCK_GETTIME || defined HAVE_GETTIMEOFDAY || defined HAVE_FTIME
    PortableTime timenow;
    PortableTime timebegin;
    int sec, msec;

    get_portable_time(&timenow);
    return((double) timenow.seconds + (double) timenow.milliseconds / 1000.0);
#else
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
    return((double) tmsbuf.tms_utime / (clock_t) sysconf(_SC_CLK_TCK));

#else /* unknown */

    return(-1.0);   /* Obvious error condition */

#endif /* GETRUSAGE */
#endif /* TIMES */
#endif /* CLOCK_GETTIME || GETTIMEOFDAY || FTIME */
}

#ifdef HAVE_GETRUSAGE
void start_timer(GTimer *timer) {
    int ret;
    ret = getrusage(RUSAGE_SELF, &timer->start);
    if(ret == -1) {
      perror("getrusage(): ");
    }
}
void stop_timer(GTimer *timer) {
    int ret;
    ret = getrusage(RUSAGE_SELF, &timer->end);
    if(ret == -1) {
      perror("getrusage(): ");
    }
}
#endif /* GETRUSAGE */

#ifdef HAVE_TIMES
clock_t start_timer(TTimer *timer) {
    clock_t start_clock;
    start_clock = times(&timer->start);
    return start_clock;
}
clock_t stop_timer(TTimer *timer) {
    clock_t stop_clock;
    stop_clock = times(&timer->end);
    return stop_clock;
}
#endif /* TIMES */
