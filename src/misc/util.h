/*************
 * Header file for util.c
 * 2002 R. Oktas, <roktas@omu.edu.tr>
 ************/

#ifndef UTIL_H_INCLUDED
#define UTIL_H_INCLUDED

char *canonicalize_pathname(char *path);
char *absolute_pathname(char *string, char *dot_path);

#endif
