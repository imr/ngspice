#ifndef _ANALYSIS_H
#define _ANALYSIS_H

typedef struct {
    IFanalysis public;
    int size;
    int domain;
    int do_ic;
    int (*(setParm))(CKTcircuit *ckt, void *anal, int which, IFvalue *value);
    int (*(askQuest))(CKTcircuit *ckt, void *anal, int which, IFvalue *value);
    int (*an_init)(CKTcircuit *ckt, JOB *job);
    int (*an_func)(CKTcircuit *ckt, int restart);
} SPICEanalysis;


char *spice_analysis_get_name(int index);
char *spice_analysis_get_description(int index);
int spice_num_analysis(void);
SPICEanalysis **spice_analysis_ptr(void);

#endif
