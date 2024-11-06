/*************
 * Header file for misc_time.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_MISC_TIME_H
#define ngspice_MISC_TIME_H

char * datestring(void);
double seconds(void);

typedef struct {
    double secs;
    int seconds;
    int milliseconds;
} PerfTime;

void perf_timer_get_time(PerfTime *);

extern PerfTime timebegin;

void timediff(PerfTime *, PerfTime *, int *, int *);

#endif
