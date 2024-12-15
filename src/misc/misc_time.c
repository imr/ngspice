/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/

/*
 * Date and time utility functions
 */

#include "ngspice/ngspice.h"
#include <string.h>

#if defined(HAS_WINGUI) || defined(__MINGW32__) || defined(_MSC_VER)
#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#ifndef HAVE_GETTIMEOFDAY
#include <winsock2.h>
#include <stdint.h> // portable: uint64_t   MSVC: __int64

/*/ MSVC defines this in winsock2.h!?
typedef struct timeval {
    long tv_sec;
    long tv_usec;
} timeval;
*/
int gettimeofday(struct timeval * tp, void * unused)
{
    NG_IGNORE(unused);
    // Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
    // This magic number is the number of 100 nanosecond intervals since January 1, 1601 (UTC)
    // until 00:00:00 January 1, 1970
    static const uint64_t EPOCH = ((uint64_t) 116444736000000000ULL);

    SYSTEMTIME  system_time;
    FILETIME    file_time;
    uint64_t    time;

    GetSystemTime( &system_time );
    SystemTimeToFileTime( &system_time, &file_time );
    time =  ((uint64_t)file_time.dwLowDateTime )      ;
    time += ((uint64_t)file_time.dwHighDateTime) << 32;

    tp->tv_sec  = (long) ((time - EPOCH) / 10000000L);
    tp->tv_usec = (long) (system_time.wMilliseconds * 1000);
    return 0;
}
#endif
#endif

#include "misc_time.h"

#ifdef USE_OMP
#include <omp.h>
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

PerfTime timebegin;

void timediff(PerfTime *now, PerfTime *begin, int *sec, int *msec)
{

    *msec = (int) now->milliseconds - (int) begin->milliseconds;
    *sec = (int) now->seconds - (int) begin->seconds;
    if (*msec < 0) {
      *msec += 1000;
      (*sec)--;
    }
    return;

}

/*
 * How many seconds have elapsed in running time.
 * This is the routine called in IFseconds
 */

double
seconds(void)
{
#ifdef USE_OMP
    // Usage of OpenMP time function
    return(omp_get_wtime() - timebegin.secs);
#elif defined(HAVE_QUERYPERFORMANCECOUNTER)
    // Windows (MSC and mingw) specific implementation
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return ((double)counter.QuadPart / frequency.QuadPart - timebegin.secs);
#elif defined(HAVE_CLOCK_GETTIME)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec + ts.tv_nsec / 1e9 - timebegin.secs);
#elif defined(HAVE_GETTIMEOFDAY)
    // Usage of gettimeofday
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec + tv.tv_usec / 1e6 - timebegin.secs);
#elif defined(HAVE_FTIME)
    // Usage of ftime
    struct timeb tb;
    PerfTime timenow;
    int sec, msec;
    ftime(&tb);
    timenow.seconds = tb.time;
    timenow.milliseconds = tb.millitm;
    timediff(&timenow, &timebegin, &sec, &msec);
    return(sec + (double) msec / 1000.0);
#elif defined(HAVE_TIMES)
    // Usage of times
    struct tms tmsbuf;
    clock_t ticks = times(&tmsbuf);
    return((double) tmsbuf.tms_utime / HZ);
#elif defined(HAVE_GETRUSAGE)
    // Usage of getrusage
    struct rusage ruse;
    getrusage(RUSAGE_SELF, &ruse);
    return ((double)ruse.ru_utime.tv_sec + (double) ruse.ru_utime.tv_usec / 1000000.0);
#else
    #error "No timer function available."
#endif
}

void perf_timer_get_time(PerfTime *time)
{
    time->secs = seconds();
    time->seconds = (int)time->secs;
    time->milliseconds = (int)((time->secs - time->seconds) * 1000.0);

}
