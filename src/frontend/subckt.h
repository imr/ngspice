/*************
 * Header file for subckt.c
 * 1999 E. Rouat
 ************/

#ifndef SUBCKT_H_INCLUDED
#define SUBCKT_H_INCLUDED

struct line * inp_subcktexpand(struct line *deck);
struct line * inp_deckcopy(struct line *deck);

#endif
