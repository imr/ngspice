/*************
 * Header file for circuits.c
 * 1999 E. Rouat
 ************/

#ifndef CIRCUITS_H_INCLUDED
#define CIRCUITS_H_INCLUDED



struct subcirc {
    char *sc_name;  /* Whatever... */
} ;


extern struct circ *ft_curckt;  /* The default active circuit. */


void ft_newcirc(struct circ *ckt);


#endif
