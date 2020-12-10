/*************
 * Header file for subckt.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_SUBCKT_H
#define ngspice_SUBCKT_H

struct card *inp_subcktexpand(struct card *deck);
struct card *inp_deckcopy(struct card *deck);
struct card *inp_deckcopy_oc(struct card *deck);
struct card *inp_deckcopy_ln(struct card *deck);

#endif
