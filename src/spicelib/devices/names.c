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
/* names.c,  efficient string table support
 Conrad Ziesler
*/



#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>

#define __NAMES_PRIVATE__
#include "names.h"

static namehash_t *names_newhash(names_t *nt, void * refp, const char *str)
{
  namehash_t *b;
  int q=strlen(str);
  b=malloc(sizeof(namehash_t)+q);
  assert(b!=NULL);
  b->refp=refp;
  b->magic=NAME_MAGIC;
  strcpy(b->str,str);
  nt->bytesalloc+=sizeof(namehash_t)+q;
  nt->namebytes+=q;
  nt->qtynames++;
  
  return b;
}


static unsigned int names_ptrhash(names_t *nt, void * refp)
{
  unsigned q=nt->qtybins;
  unsigned int v,  r=0;

  v=(unsigned)refp;
  /*
  v>>=2;
  r= ((v%(7*q+(q/3)+5))^v) % q;
  */
  r= v ^ (v>>3) ^ ((v/3)>>15) ^ ((~v)>>11);
  return r%q;  
}

static unsigned int names_strhash(names_t *nt, const char *str)
{
  unsigned q=nt->qtybins;
  unsigned int r=0;
  
  while(*str!=0) 
    {
      r+= (str[0]^(r%q));
      r%=q;
      str++;
    }
  
  return r%q;
}


char *names_lookup(names_t *nt, void * refp)
{
  unsigned int id;
  namehash_t *nh;
  int i=0;

  id=names_ptrhash(nt,refp);
  
  for(nh=nt->cref[id];nh!=NULL;nh=nh->cref,i++)
    {
      assert(nh->magic==NAME_MAGIC);
      if(refp==nh->refp){ return nh->str; }
    }

  return NULL;
}

void * names_check(names_t *nt, const char *name)
{
  unsigned int id;
  namehash_t *nh;
  int i=0;

  id=names_strhash(nt,name);
  
  for(nh=nt->cstr[id];nh!=NULL;nh=nh->cstr,i++)
    {
      assert(nh->magic==NAME_MAGIC);
      if(strcmp(name,nh->str)==0){ return nh->refp; }
    }
  return NULL;
}



void  names_add(names_t *nt, void * refp, const char *name)
{
  unsigned int idn,idr,id;
  namehash_t *nh,*nhref=NULL,*nhstr=NULL;
  int i=0;

  if(nt->qtynames>((5*nt->qtybins)/4)) /* double table size each time we outgrow old table */
    names_rehash(nt, (2*nt->qtynames) );
  
  idn=names_strhash(nt,name);
  idr=names_ptrhash(nt,refp);

  for(nh=nt->cref[idr];nh!=NULL;nh=nh->cref,i++)
    {
      assert(nh->magic==NAME_MAGIC);
      if(refp==nh->refp){ nt->avg_refl=(nt->avg_refl+i)/2; nhref=nh; break; }
    }

  for(nh=nt->cstr[idn];nh!=NULL;nh=nh->cstr,i++)
    {
      assert(nh->magic==NAME_MAGIC);
      if(strcmp(name,nh->str)==0){ nt->avg_strl=(nt->avg_strl+i)/2; nhstr=nh; break; }
    }

  if(nhstr==NULL) /* adding new name entry */
    {
      if(nhref!=NULL) /* but refp was the same */
	{
	  fprintf(stderr,"**** DUPLICATE KEY NAME ****\n"); 
	  if(1)assert(0&&"duplicate key in names"); 
	}
      
      nh=names_newhash(nt, refp, name);
      nh->cstr=nt->cstr[idn];
      nt->cstr[idn]=nh;
      nh->cref=nt->cref[idr];
      nt->cref[idr]=nh;
    }
  else  /* replacing old name entry */
    {
      namehash_t *prev=NULL;
      assert(0 && "Replacing strings in names has a bug. do not use");
      /* lookup refp of old name and unlink */
      id=names_ptrhash(nt,nhstr->refp);
      for(nh=nt->cref[id];nh!=NULL;prev=nh,nh=nh->cref,i++)
	if(nhstr->refp==nh->refp)
	  {
	    if(prev==NULL)nt->cref[id]=nh->cref;
	    else prev->cref=nh->cref;
	    break;
	  }
      /* relink new name, refp */
      nhstr->refp=refp;
      nhstr->cref=nt->cref[idr];
      nt->cref[idr]=nhstr;
    }
}

char * names_stats(names_t *nt)
{
  static char buf[1024];
  int i,qs=0,qp=0,ms=0,mp=0,j,ks=0,kp=0;
  namehash_t *nh;
  
  for(i=0;i<nt->qtybins;i++)
    {
     
      for(j=0,nh=nt->cstr[i];nh!=NULL;nh=nh->cstr,j++)	      assert(nh->magic==NAME_MAGIC);
      if(j>0)ks++;
      if(ms<j)ms=j;
      qs+=j;
      for(j=0,nh=nt->cref[i];nh!=NULL;nh=nh->cref,j++) 	      assert(nh->magic==NAME_MAGIC);
      if(mp<j)mp=j;
      if(j>0)kp++;
      qp+=j;
    }

  qp/=kp;
  qs/=ks;
  
  sprintf(buf,"names: %i bins (%i totaling %i) , alloc %i, avg: %i %i max: %i %i",nt->qtybins,nt->qtynames,nt->namebytes,nt->bytesalloc,qp,qs,mp,ms);
	  
  return buf;
}



/* this should optimize our hash table for memory usage */
void names_rehash(names_t *nt, int newbins)
{
  int i;
  int oldqty;
  namehash_t *hp;

  oldqty=nt->qtybins;
  nt->qtybins=newbins;

  nt->bytesalloc+= ( (newbins-oldqty)*sizeof(void *)*2 );
  /* do cref first, using cstr */
  if(nt->cref!=NULL)
    free(nt->cref);
  nt->cref=malloc(sizeof(void *)*(nt->qtybins+1));
  assert(nt->cref!=NULL);  
  memset(nt->cref,0,sizeof(void*)*nt->qtybins);  

  /* iterate through list of string hashes, adding to ref hashes */
  for(i=0;i<oldqty;i++)
    for(hp=nt->cstr[i];hp!=NULL;hp=hp->cstr)
      {
	unsigned id;
	id=names_ptrhash(nt,hp->refp);
	hp->cref=nt->cref[id];
	nt->cref[id]=hp;
      }
  
  /* next do cstr, using new cref */
  if(nt->cstr!=NULL)
    free(nt->cstr);
  nt->cstr=malloc(sizeof(void *)*(nt->qtybins+1));
  assert(nt->cstr!=NULL);
  memset(nt->cstr,0,sizeof(void*)*nt->qtybins);
  
  for(i=0;i<nt->qtybins;i++)
    for(hp=nt->cref[i];hp!=NULL;hp=hp->cref)
      {
	unsigned id;
	id=names_strhash(nt,hp->str);
	hp->cstr=nt->cstr[id];
	nt->cstr[id]=hp;
      }
}



names_t   *names_new(void)
{
  names_t *p;
  p=malloc(sizeof(names_t));
  assert(p!=NULL);
  memset(p,0,sizeof(names_t));
  p->bytesalloc=sizeof(names_t);
  p->namebytes=0;
  p->qtynames=0;
  p->avg_strl=0;
  p->avg_refl=0;

  p->qtybins=0;
  p->cstr=NULL;
  p->cref=NULL;
  names_rehash(p,13); /* start small, grow bigger */
  return p;
}

void names_free(names_t *nt)
{
  int i;
  namehash_t *nh,*next;
  if(nt!=NULL)
    { 
      for(i=0;i<nt->qtybins;i++)
	{
	  for(nh=nt->cstr[i];nh!=NULL;nh=next)
	    {
	      assert(nh->magic==NAME_MAGIC);
	      next=nh->cstr;
	      free(nh);
	    }
	}
      free(nt->cstr);
      free(nt->cref);
      free(nt); 
    }
}
