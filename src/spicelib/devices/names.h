/********************
    This file is part of the software library CADLIB written by Conrad Ziesler
    Copyright 2003, Conrad Ziesler, all rights reserved.

*************************
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

******************/
/* names.h, definitions for name storage
   Conrad Ziesler
*/

#ifndef __NAMES_H__
#define __NAMES_H__
#ifdef __NAMES_PRIVATE__

#define NAME_MAGIC 0x52a01250
typedef struct name_hash_st
{
  int magic;
  struct name_hash_st *cref,*cstr;
  void * refp;
  char str[1];
}namehash_t;

typedef struct names_st
{
  namehash_t **cref,**cstr;
  int avg_refl;
  int avg_strl;
  int qtybins;
  int qtynames;
  int namebytes;
  int bytesalloc;
}names_t;
#else

struct names_st;
typedef struct names_st names_t;
#endif

char      *names_stats(names_t *nt);
char      *names_lookup(names_t *nt, void * refp);
void *     names_check(names_t *nt, const char *name);
void       names_add(names_t *nt, void * refp, const char *name);
names_t   *names_new(void);
void       names_free(names_t *nt);
void names_rehash(names_t *nt, int newbins); /* private-ish */

#endif
