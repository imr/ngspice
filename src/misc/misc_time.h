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

void perf_timer_start(PerfTimer *);
void perf_timer_stop(PerfTimer *);
void perf_timer_elapsed_sec_ms(const PerfTimer *, int *, int *);

#ifdef HAVE_FTIME

extern struct timeb timebegin;

void timediff(struct timeb *, struct timeb *, int *, int *);

#endif

#endif
