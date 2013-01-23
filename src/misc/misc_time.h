/*************
 * Header file for misc_time.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_MISC_TIME_H
#define ngspice_MISC_TIME_H

char * datestring(void);
double seconds(void);

#ifndef HAVE_GETRUSAGE
#ifndef HAVE_TIMES
#ifdef HAVE_FTIME

extern struct timeb timebegin;

void timediff(struct timeb *, struct timeb *, int *, int *);

#endif
#endif
#endif

#endif
