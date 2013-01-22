#ifndef ngspice_COM_MEASURE2_H
#define ngspice_COM_MEASURE2_H


extern int measure_get_precision(void);
extern int get_measure2(wordlist *wl, double *result, char *out_line, bool auto_check);
extern int measure_extract_variables(char *line);

void com_dotmeasure(wordlist *wl);

#endif
