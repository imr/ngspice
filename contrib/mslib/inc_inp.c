/*
 * MW. Include for spice 
 */
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <string.h>

#include "datadef.h"

static int whatdline(char *tbuf, char *firstname);
static int whatlline(char *tbuf, char *name);

extern int bsizer;
extern char buf[BSIZE];

/*
 * Read deck and search for *MOD, *LIB, *SUB commands 
 */
int
readdeck(FILE * tdeck, struct LSData *lib, \
		 struct LSData *sub, struct LSData *mod)
{
  char firstname[BSIZE];
  char *names, *smlp;
  struct LSData *which;

  while (fgets(buf, bsizer, tdeck))
	{

	  smlp = buf;
	/*
	 * Ignore control chars at the end of line 
	 */
	  do
		{
		  smlp++;
		}
	  while ((!(iscntrl(*smlp))) && (*smlp != '\0'));
	  *smlp = '\0';

	  switch (whatdline(buf, firstname))
		{

		case LIBDLINE:
		  lib->flag = IS_LIB;
		  which = lib;
		  break;

		case SUBDLINE:
		  sub->flag = IS_SUB;
		  which = sub;
		  break;

		case MODDLINE:
		  mod->flag = IS_MOD;
		  which = mod;
		  break;

		default:
		  which = NULL;
		}

/*
 * If we finded something 
 */
	  if (which)
		{
		  names = buf;
		  strsep(&names, " ");

		  while (smlp = strsep(&names, " "))
			{
			  if (*smlp)
				{
				  which = LSinsert(NULL, which);
				  strcpy(which->name, smlp);
				}
			}
		}
	}
  return ((lib->flag != IS_LIB) && (mod->flag != IS_MOD) \
		  &&(sub->flag != IS_SUB)) ? FAILED : SUCCESS;
}

/*
 * Read library and write specififed models/subckts to tmplib 
 */
int
readlib(struct LSData *lib, FILE * tlib, \
		struct LSData *sub, struct LSData *mod)
{

  char name[BSIZE];
  int numi, wflag, nextsub;

  numi = 0;
  wflag = NOWRITE;
  nextsub = 0;    

  while (fgets(buf, bsizer, lib->filedes))
	{
/*
 * Now we must check what line is it and if it should be written to tmplib 
 */
	  switch (whatlline(buf, name))
		{

		case (MODLLINE):
		  if (wflag == WRITESUB)
			fputs(buf, tlib);
		  else
			{

			  if (mod)
				{
				  if (checkname(mod, name))
					{
					  wflag = WRITEMOD;
					  numi++;
					  fprintf(tlib, "*  Model: %s, from: %s.\n", \
							  name, lib->name);
					  fputs(buf, tlib);
					}
				}
			}
		  break;

		case (SUBLLINE):
		  if (sub)
			{
			    if (wflag==WRITESUB) 
				{
				/* subckt inside subckt  
				    not so funny */
				    nextsub++;
				    fputs(buf, tlib);
				    break;
				}    
				
			    if (checkname(sub, name))
				{
				  wflag = WRITESUB;
				  numi++;
				  fprintf(tlib, "*  Subckt: %s, from: %s.\n", \
						  name, lib->name);
				  fputs(buf, tlib);
				}
			}
		  break;

		case (NORMLINE):
		  if (wflag == WRITEMOD)
			{
			  wflag = NOWRITE;
			  fputs("\n*  End Model.\n\n", tlib);
			}
		  if (wflag == WRITESUB)
			fputs(buf, tlib);
		  break;

		case (ENDSLLINE):
		    if (nextsub)
			{
			nextsub--;
			fputs(buf, tlib);
			break;
			} else {
		    
		    if (wflag == WRITESUB)
			{
			  fprintf(tlib, "%s\n*  End Subckt\n\n", buf);
			}
		  wflag = NOWRITE;
		  break;
		  }

		case (CONTLLINE):
		  if (wflag != NOWRITE)
			fputs(buf, tlib);
		}

	}
  return numi;
}

/*
 * Check what line in deck it is 
 */
int
whatdline(char *tbuf, char *firstname)
{
  if (sscanf(tbuf, "*LIB %s", firstname) == 1)
	return LIBDLINE;
  if (sscanf(tbuf, "*SUB %s", firstname) == 1)
	return SUBDLINE;
  if (sscanf(tbuf, "*MOD %s", firstname) == 1)
	return MODDLINE;
  return NORMLINE;
}

/*
 * Check what line it is. If we have model or subckt line we also read its name  
 */
int
whatlline(char *tbuf, char *name)
{
  if (sscanf(tbuf, ".SUBCKT %s %*s", name) == 1)
	return SUBLLINE;
  if (sscanf(tbuf, ".MODEL %s %*s", name) == 1)
	return MODLLINE;
  if (sscanf(tbuf, ".ENDS%c", name) == 1)
	return ENDSLLINE;
  if (sscanf(tbuf, "+%s", name) == 1)
	return CONTLLINE;
  return NORMLINE;
}
