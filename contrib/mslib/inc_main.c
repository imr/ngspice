/*
 * MW. Include - main functions 
 */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>

#include "datadef.h"

struct LSData *firstLIB;
struct LSData *firstSUB;
struct LSData *firstMOD;
struct LSData *deck;
struct LSData *tmplib;
int bsize, bsizer;
char buf[BSIZE];

int
main(int argc, char *argv[])
{

/*
 * Initialize everything 
 */
  struct LSData *subp, *libp, *modp;
  char tch[BSIZE];
  int mswritten;

  tmplib = LSinsert(NULL, NULL);
  deck = LSinsert(NULL, NULL);
  *tch = '\0';
  mswritten = 0;

  switch (argc)
	{

	case 3:
	  strcpy(tmplib->name, argv[--argc]);
	  strcpy(tch, tmplib->name);

	case 2:
	  strcpy(deck->name, argv[--argc]);
	  if (!(*tch))
		{
		  sprintf(tmplib->name, "%s%s", deck->name, TMPLIBNAME);
		  strcpy(tch, tmplib->name);
		}
	  break;

	case 1:
	  fprintf(stdout, "Usage: mslib deck [tmplib]\n");
	  return FAILED;

	default:
	  fprintf(stderr, "mslib: Incorrect parameters count.\n");
	  return FAILED;
	}

/*
 * Open deck 
 */
  if (!(deck->filedes = fopen(deck->name, "r")))
	{
	  sprintf(deck->name, "%s%s", DECKPATH, argv[1]);
	  sprintf(tmplib->name, "%s%s", DECKPATH, tch);

	  if (!(deck->filedes = fopen(deck->name, "r")))
		{
		  fprintf(stderr, "mslib: Can't open deck %s.\n", deck->name);
		  LSclear(deck);
		  LSclear(tmplib);
		  return FAILED;
		}
	}
  bsizer = BSIZE;
  bsize = bsizer--;

  deck->flag = DECK_OPEN;

/*
 * Create tmplib and write first line to it 
 */
  if (!(tmplib->filedes = fopen(tmplib->name, "w")))
	{
	  fprintf(stderr, "mslib: Can't creat tmplib %s.\n", tmplib->name);
	  LSclear(tmplib);
	  LSclear(deck);
	  return FAILED;
	}
  tmplib->flag = TLIB_OPEN;
  fprintf(tmplib->filedes, "%s\n*   Tmp library: %s,\n*   For deck: %s.\n\n", \
		  LIBMESSAGE, tmplib->name, deck->name);

  firstLIB = LSinsert(NULL, NULL);
  firstSUB = LSinsert(NULL, NULL);
  firstMOD = LSinsert(NULL, NULL);

/*
 * Find commands in deck 
 */
  readdeck(deck->filedes, firstLIB, firstSUB, firstMOD);

  if (firstLIB->flag = IS_LIB)
	{

	  libp = firstLIB->nextLS;
	  do
		{
		  if (!(libp->filedes = fopen(libp->name, "r")))
			{
			  strcpy(tch, libp->name);
			  sprintf(libp->name, "%s%s", LIBPATH, tch);

			  if (!(libp->filedes = fopen(libp->name, "r")))
				{
				  libp->flag = FAILED;
				}
			}
/*
 * Read libraries if everything is OK 
 */
		  if (libp->flag != FAILED)
			{
			  libp->flag = LIB_OPEN;

			  modp = (firstMOD->flag == IS_MOD) ? firstMOD->nextLS : NULL;
			  subp = (firstSUB->flag == IS_SUB) ? firstSUB->nextLS : NULL;

			  mswritten += readlib(libp, tmplib->filedes, subp, modp);
			}
		  libp = libp->nextLS;
		}
	  while (libp);
	}
  fprintf(stdout, "mslib: Written %d items to tmplib %s.\n", \
		  mswritten, tmplib->name);

  if (libp = firstLIB->nextLS)
	{
	  do
		{
		  if (libp->flag != LIB_OPEN)
			fprintf(stderr, "   Can't open lib %s.\n", libp->name);
		  libp = libp->nextLS;
		}
	  while (libp);
	}
/*
 * Check is models or subckts were find and 
 * * are not duplicated 
 */
  if (modp = firstMOD->nextLS)
	{
	  do
		{
		  switch (modp->flag)
			{
			case DUPLICATE:
			  fprintf(stderr, "   Model duplicated %s.\n", \
					  modp->name);
			  break;
			default:
			  fprintf(stderr, "   Can't find model %s.\n", \
					  modp->name);
			  break;

			case FINDED:
			}

		  modp = modp->nextLS;
		}
	  while (modp);
	}
  if (subp = firstSUB->nextLS)
	{
	  do
		{
		  switch (subp->flag)
			{
			case DUPLICATE:
			  fprintf(stderr, "   Subckt duplicated %s.\n", \
					  subp->name);
			  break;
			default:
			  fprintf(stderr, "   Can't find subckt %s.\n", \
					  subp->name);
			  break;

			case FINDED:
			}
		  subp = subp->nextLS;
		}
	  while (subp);
	}
/*
 * Clear all data and close files 
 */

  LSclear(tmplib);
  LSclear(deck);
  LSclear(firstLIB);
  LSclear(firstSUB);

  return SUCCESS;

}
