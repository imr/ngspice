/*************
 * Header file for circuits.c
 * 1999 E. Rouat
 ************/

#ifndef CIRCUITS_H_INCLUDED
#define CIRCUITS_H_INCLUDED

/* The curcuits that are currently available to the user. */

struct circ {
    char *ci_name;    /* What the circuit can be called. */
    char *ci_ckt;      /* The CKTcircuit structure. */
    INPtables *ci_symtab;    /* The INP symbol table. */
    struct line *ci_deck;   /* The input deck. */
    struct line *ci_origdeck;/* The input deck, before subckt expansion. */
    struct line *ci_options;/* The .option cards from the deck... */
    struct variable *ci_vars; /* ... and the parsed versions. */
    bool ci_inprogress; /* We are in a break now. */
    bool ci_runonce;    /* So com_run can to a reset if necessary... */
    wordlist *ci_commands;  /* Things to do when this circuit is done. */
    struct circ *ci_next;   /* The next in the list. */
    char *ci_nodes;     /* ccom structs for the nodes... */
    char *ci_devices;   /* and devices in the circuit. */
    char *ci_filename;  /* Where this circuit came from. */
    char *ci_defTask;   /* the default task for this circuit */
    char *ci_specTask;  /* the special task for command line jobs */
    char *ci_curTask;   /* the most recent task for this circuit */
    char *ci_defOpt;    /* the default options anal. for this circuit */
    char *ci_specOpt;   /* the special options anal. for command line jobs */
    char *ci_curOpt;    /* the most recent options anal. for the circuit */
} ;

struct subcirc {
    char *sc_name;  /* Whatever... */
} ;


extern struct circ *ft_curckt;  /* The default active circuit. */


void ft_newcirc(struct circ *ckt);


#endif
