/*************
 * Header file for misc_time.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_MISC_TIME_H
#define ngspice_MISC_TIME_H

char * datestring(void);
double seconds(void);
void timebegin(void);

// Needed for WIN32
#ifdef HAVE_FTIME
void timediff(struct timeb*, struct timeb*, int*, int*);
#endif


#endif
