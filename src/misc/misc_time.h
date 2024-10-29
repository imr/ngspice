/*************
 * Header file for misc_time.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_MISC_TIME_H
#define ngspice_MISC_TIME_H

char * datestring(void);
double seconds(void);

typedef struct {
    double start;
    double end;
} PerfTimer;

typedef struct {
    int seconds;
    int milliseconds;
} PerfTime;

void perf_timer_start(PerfTimer *);
void perf_timer_stop(PerfTimer *);
void perf_timer_elapsed_sec_ms(const PerfTimer *, int *, int *);
void perf_timer_get_time(PerfTime *);

extern PerfTime timebegin;

void timediff(PerfTime *, PerfTime *, int *, int *);

#endif
