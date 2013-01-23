/*************
 * Header file for complete.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_COMPLETE_H
#define ngspice_COMPLETE_H

/* The data structure for the commands is as follows: every node has a pointer
 * to its leftmost child, where the children of a node are those of which
 * the node is a prefix. This means that for a word like "ducks", there
 * must be nodes "d", "du", "duc", etc (which are all marked "invalid",
 * of course).  This data structure is called a "trie".
 */


#define NARGS 4

struct ccom {
    char *cc_name;              /* Command or keyword name. */
    long cc_kwords[NARGS];      /* What this command takes. */
    char cc_invalid;            /* This node has been deleted. */
    struct ccom *cc_child;      /* Left-most child. */
    struct ccom *cc_sibling;    /* Right (alph. greater) sibling. */
    struct ccom *cc_ysibling;   /* Left (alph. less) sibling. */
    struct ccom *cc_parent;     /* Parent node. */
};


void throwaway(struct ccom *dbase);


#endif
