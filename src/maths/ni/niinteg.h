/*************
 * Header file for niinteg.c
 * 1999 E. Rouat
 ************/

#ifndef NIINTEG_H_INCLUDED
#define NIINTEG_H_INCLUDED


int NIintegrate(register CKTcircuit *ckt, double *geq, double *ceq, double cap, int qcap);


#endif
