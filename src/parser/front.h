/*************
 * Header file for front.c
 * 1999 E. Rouat
 ************/

#ifndef FRONT_H_INCLUDED
#define FRONT_H_INCLUDED

/* Stuff to do control structures. We keep a history (seperate from the
 * cshpar history, for now at least) of commands and their event numbers,
 * with a block considered as a statement. In a goto, the first word in
 * co_text is where to go, likewise for label. For conditional controls,
 * we have to call ft_getpnames and ft_evaluate each time, since the
 * dvec pointers will change... Also we should do variable and backquote
 * substitution each time...
 */

struct control {
    int co_type;            /* One of CO_* ... */
    wordlist *co_cond;      /* if, while, dowhile */
    char *co_foreachvar;        /* foreach */
    int co_numtimes;        /* repeat, break & continue levels */
    wordlist *co_text;      /* Ordinary text and foreach values. */
    struct control *co_parent;  /* If this is inside a block. */
    struct control *co_children;    /* The contents of this block. */
    struct control *co_elseblock;   /* For if-then-else. */
    struct control *co_next;
    struct control *co_prev;
} ;

int cp_evloop(char *string);
void cp_resetcontrol(void);
void cp_popcontrol(void);
void cp_pushcontrol(void);
void cp_toplevel(void);
void com_cdump(wordlist *wl);



#endif
