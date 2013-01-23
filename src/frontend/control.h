/**********
 * Header file for control.c
**********/

#ifndef ngspice_CONTROL_H
#define ngspice_CONTROL_H

#include "ngspice/bool.h"

/* Stuff to do control structures. We keep a history (seperate from
 * the cshpar history, for now at least) of commands and their event
 * numbers, with a block considered as a statement. In a goto, the
 * first word in co_text is where to go, likewise for label. For
 * conditional controls, we have to call ft_getpnames and ft_evaluate
 * each time, since the dvec pointers will change... Also we should do
 * variable and backquote substitution each time...  */
struct control {
    int co_type;            /* One of CO_* ... */
    wordlist *co_cond;      /* if, while, dowhile */
    char *co_foreachvar;        /* foreach */
    int co_numtimes;        /* repeat, break & continue levels */
    int co_timestodo;       /* the number of times left during a repeat loop */
    wordlist *co_text;      /* Ordinary text and foreach values. */
    struct control *co_parent;  /* If this is inside a block. */
    struct control *co_children;    /* The contents of this block. */
    struct control *co_elseblock;   /* For if-then-else. */
    struct control *co_next;
    struct control *co_prev;
};

enum co_command {
    CO_UNFILLED,
    CO_STATEMENT,
    CO_WHILE,
    CO_DOWHILE,
    CO_IF,
    CO_FOREACH,
    CO_BREAK,
    CO_CONTINUE,
    CO_LABEL,
    CO_GOTO,
    CO_REPEAT
};

#define CONTROLSTACKSIZE 256    /* Better be enough. */

extern struct control *control[CONTROLSTACKSIZE];
extern struct control *cend[CONTROLSTACKSIZE];
extern int stackp;

#endif
