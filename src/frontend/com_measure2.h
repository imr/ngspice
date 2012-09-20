#ifndef _COM_MEASURE_H
#define _COM_MEASURE_H

#include "ngspice/config.h"

extern int measure_get_precision(void);
extern int get_measure2(wordlist *wl, double *result, char *out_line, bool auto_check);
extern int measure_extract_variables(char *line);

void com_dotmeasure(wordlist *wl);

#endif
