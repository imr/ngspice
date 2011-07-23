/*************
 * Header file for resources.c
 * 1999 E. Rouat
 ************/

#ifndef RESOURCES_H_INCLUDED
#define RESOURCES_H_INCLUDED

void init_rlimits(void);
void init_time(void);
void com_rusage(wordlist *wl);


struct proc_mem {
   size_t size;    /* Total ngspice program size */
   size_t resident;/* Resident set size */
   size_t shared;  /* Shared ngspice pages */
   size_t trs;     /* Text (code) pages */
   size_t drs;     /* Stack */
   size_t lrs;     /* Library pages */
   size_t dt;      /* Dirty pages (not used in kernel 2.6) */
};

struct sys_mem {
   size_t size;    /* Total memory size */
   size_t free;    /* Free memory */
   size_t swap_t;  /* Swap total */
   size_t swap_f;  /* Swap free */
};

#endif
