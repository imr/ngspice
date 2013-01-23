/*************
 * Header file for util.c
 * 2002 R. Oktas, <roktas@omu.edu.tr>
 ************/

#ifndef ngspice_UTIL_H
#define ngspice_UTIL_H

char *canonicalize_pathname(char *path);
char *absolute_pathname(char *string, char *dot_path);
char *ngdirname(const char *name);

#endif
