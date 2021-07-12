/**********
Copyright 2021 The ngspice team  All rights
reserved.
Author: 2021 Holger Vogt
3-clause BSD license
**********/

/* cplhash.c */
extern void mem_init(void);
extern void mem_delete(void);
extern int memsaved(void *ptr);
extern void memdeleted(const void *ptr);
