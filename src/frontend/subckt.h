/*************
 * Header file for subckt.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_SUBCKT_H
#define ngspice_SUBCKT_H

struct line * inp_subcktexpand(struct line *deck);
struct line * inp_deckcopy(struct line *deck);
struct line * inp_deckcopy_oc(struct line *deck);

#endif
