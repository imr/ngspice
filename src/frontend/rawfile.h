/*************
 * Header file for rawfile.c
 * 1999 E. Rouat
 ************/

#ifndef RAWFILE_H_INCLUDED
#define RAWFILE_H_INCLUDED

void raw_write(char *name, struct plot *pl, bool app, bool binary);
struct plot * raw_read(char *name);



#endif
