/*************
 * Header file for vectors.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_VECTORS_H
#define ngspice_VECTORS_H

#define vec_free(ptr)                           \
    do {                                        \
        vec_free_x(ptr);                        \
        ptr = NULL;                             \
    } while(0)


#endif
