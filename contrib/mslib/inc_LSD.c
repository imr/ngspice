/*
 * MW. Include for spice - LSData functions 
 */

#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <string.h>

#include "datadef.h"

char *Message = "Michael Widlok, all rights reserved \n"
"mslib - MW include for Spice models/subckts\n";

/*
 * Add or cretate new LS structure just after where pointer 
 */
struct LSData *
LSinsert(struct LSData *LS, struct LSData *where)
{

  if (!(LS))
	{
	  LS = (struct LSData *) malloc(sizeof(struct LSData));

	  if (!(LS))
		{
		  fprintf(stderr, "LSinsert: Can't allocate LSData srtucture.\n");
		  exit(FAILED);
		}
	  LS->filedes = NULL;
	}
/*
 * If where is given we must set nextLS and prevLS correctly 
 */
  if (where)
	{
	  LS->prevLS = where;
	  if (LS->nextLS = where->nextLS)
		where->nextLS->prevLS = LS;
	  where->nextLS = LS;
  } else
	LS->nextLS = LS->prevLS = NULL;
  return LS;
}

/*
 * Clear all LS list from end. This also closes opened files 
 */
struct LSData *
LSclear(struct LSData *LS)
{
  while (LS->nextLS)
	LS = LS->nextLS;
  return Backfree(LS);
}

/*
 * Used by LSclear 
 */
struct LSData *
Backfree(struct LSData *LS)
{
  if (LS->filedes)
	fclose(LS->filedes);
  return (LS->prevLS) ? Backfree(LS->prevLS) : free(LS), LS;
}

/*
 * Check if sub/mod name should by included 
 */
int
checkname(struct LSData *smp, char *name)
{
  do
	{
	  if (!(strcmp(smp->name, name)))
		{
		  if (smp->flag != FINDED)
			{
			  smp->flag = FINDED;
			  return NAMEVALID;
		  } else
			{
			  smp->flag = DUPLICATE;
			  return NAMEVALID;
			}
		}
	  smp = smp->nextLS;
	}
  while (smp);
  return NAMENOTV;
}
