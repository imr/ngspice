/*************
 * Header file for vectors.c
 * 1999 E. Rouat
 * $Id$
 ************/

#ifndef VECTORS_H_INCLUDED
#define VECTORS_H_INCLUDED

void ft_loadfile(char *file);
void plot_add(struct plot *pl);
void vec_remove(char *name);
struct dvec * vec_fromplot(char *word, struct plot *plot);
void plot_docoms(wordlist *wl);
struct dvec * vec_copy(struct dvec *v);
struct plot * plot_alloc(char *name);
void vec_new(struct dvec *d);
void vec_gc(void);
#define vec_free(ptr)  vec_free_x(ptr); ptr=NULL
void vec_free_x(struct dvec *v);
bool vec_eq(struct dvec *v1, struct dvec *v2);
char * vec_basename(struct dvec *v);
void plot_setcur(char *name);
void plot_new(struct plot *pl);
void vec_transpose(struct dvec *v);
struct dvec * vec_mkfamily(struct dvec *v);
bool plot_prefix(char *pre, char *str);


#endif
