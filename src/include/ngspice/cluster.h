#ifndef ngspice_CLUSTER_H
#define ngspice_CLUSTER_H
#include "ngspice/cktdefs.h"

/* Cluster definitions */
#define PORT 1234
#define TIME_PORT 1235
#define DOMAIN_NAME "cluster.multigig"
#define CLUSTER_WIDTH 4
#define TIME_HOST "time.cluster.multigig"
/* does all the setups */
extern int CLUsetup(CKTcircuit *ckt);

/* reads input pipes and sets voltages*/
/* call each time the present time is changed, ie just before NIinter*/
extern int CLUinput(CKTcircuit *ckt);

/* call after each accepted timestep, ie CKTdump */
extern int CLUoutput(CKTcircuit *ckt);


/* the time step control */
extern int CLUsync(double time,double *delta, int error);
#endif
