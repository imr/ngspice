/*************
 * Header file for resources.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_RESOURCE_H
#define ngspice_RESOURCE_H

extern size_t getMemorySize(void);
extern size_t getPeakRSS(void);
extern size_t getCurrentRSS(void);
extern size_t getAvailableMemorySize(void);

void init_rlimits(void);
void init_time(void);
void com_rusage(wordlist *wl);


struct proc_mem {
   unsigned long long size;    /* Total ngspice program size */
   unsigned long long resident;/* Resident set size */
   unsigned long long shared;  /* Shared ngspice pages */
   unsigned long long trs;     /* Text (code) pages */
   unsigned long long drs;     /* Stack */
   unsigned long long lrs;     /* Library pages */
   unsigned long long dt;      /* Dirty pages (not used in kernel 2.6) */
};

struct sys_mem {
   unsigned long long size;    /* Total memory size */
   unsigned long long free;    /* Free memory */
   unsigned long long swap_t;  /* Swap total */
   unsigned long long swap_f;  /* Swap free */
};

#endif
