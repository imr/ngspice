/*************
 * Header file for inpcom.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_INPCOM_H
#define ngspice_INPCOM_H

struct card *insert_new_line(struct card *card, char *line,
                             int linenum, int linenum_orig);

#endif
