/*************
 * Header file for complete.c
 * 1999 E. Rouat
 ************/

#ifndef COMPLETE_H_INCLUDED
#define COMPLETE_H_INCLUDED

/* The data structure for the commands is as follows: every node has a pointer
 * to its leftmost child, where the children of a node are those of which
 * the node is a prefix. This means that for a word like "ducks", there
 * must be nodes "d", "du", "duc", etc (which are all marked "invalid",
 * of course).  This data structure is called a "trie".
 */



#define NARGS 4

struct ccom {
    char *cc_name;          /* Command or keyword name. */
    long cc_kwords[NARGS];  /* What this command takes. */
    char cc_invalid;    /* This node has been deleted. */
    struct ccom *cc_child;  /* Left-most child. */
    struct ccom *cc_sibling;/* Right (alph. greater) sibling. */
    struct ccom *cc_ysibling;/* Left (alph. less) sibling. */
    struct ccom *cc_parent; /* Parent node. */
} ;


void cp_ccom(wordlist *wlist, char *buf, bool esc);
wordlist * cp_cctowl(char *stuff);
void cp_ccon(bool on);
bool cp_comlook(char *word);
void cp_addcomm(char *word, long int bits0, long int bits1, long int bits2, 
		long int bits3);
void cp_remcomm(char *word);
void cp_addkword(int class, char *word);
void cp_remkword(int class, char *word);
char * cp_kwswitch(int class, char *tree);
void cp_ccrestart(bool kwords);
void throwaway(struct ccom *dbase);


#endif
