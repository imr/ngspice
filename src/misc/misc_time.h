/*************
 * Header file for misc_time.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_MISC_TIME_H
#define ngspice_MISC_TIME_H

char * datestring(void);
double seconds(void);

#if defined HAVE_CLOCK_GETTIME || defined HAVE_GETTIMEOFDAY

typedef struct {
    long seconds;
    long milliseconds;
} PortableTime;

void get_portable_time(PortableTime *);
void timediff(PortableTime *, PortableTime *, int *, int *);

#endif

#endif
