#ifndef _COM_MEASURE_H
#define _COM_MEASURE_H

#include <config.h>

  int get_measure_precision(void) ;
/*  void com_measure2(wordlist *wl); */
  int get_measure2(wordlist *wl,double *result,char *out_line, bool auto_check) ;

#endif
