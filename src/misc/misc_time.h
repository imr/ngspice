/*************
 * Header file for misc_time.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_MISC_TIME_H
#define ngspice_MISC_TIME_H

char * datestring(void);
double seconds(void);

#if defined HAVE_CLOCK_GETTIME || defined HAVE_GETTIMEOFDAY || defined HAVE_FTIME

typedef struct {
    long seconds;
    long milliseconds;
} PortableTime;

void get_portable_time(PortableTime *);
void timediff(PortableTime *, PortableTime *, int *, int *);

#endif

#ifdef HAVE_GETRUSAGE

typedef struct {
    struct rusage start;
    struct rusage end;
} GTimer;

void start_timer(GTimer *);
void stop_timer(GTimer *);

#endif

#ifdef HAVE_TIMES

typedef struct {
    struct tms start;
    struct tms end;
} TTimer;

clock_t start_timer(TTimer *);
clock_t stop_timer(TTimer *);

#endif

#endif
