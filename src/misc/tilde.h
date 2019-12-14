/*************
 * Header file for tilde.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_TILDE_H
#define ngspice_TILDE_H

char * tildexpand(const char *string);

int get_local_home(size_t n_byte_buf, char **p_buf);


#ifdef HAVE_PWD_H
int get_usr_home(const char *usr, size_t n_byte_buf, char **p_buf);
#endif

#endif /* include guard */
