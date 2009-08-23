#ifndef _COM_MEASURE_H
#define _COM_MEASURE_H

#include <config.h>

extern int get_measure_precision(void) ;
/* void com_measure2(wordlist *wl); */
extern int get_measure2(wordlist *wl,double *result,char *out_line, bool auto_check) ;
extern int measure_extract_variables( char *line ) ;

#endif
