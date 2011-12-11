/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1992 David A. Gates, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cidersupt.h"

static char *LogFileName = "cider.log";
static int LogError = 0;

void
LOGmakeEntry(char *name, char *description)
{
  int procStamp;
  FILE *fpLog;

#ifdef HAS_GETPID
  procStamp = getpid();
#else
  procStamp = 0;
#endif

/* Want to make sure that multiple processes can access the log file
 * without stepping on each other.
 */
#ifdef ultrix
  if ((fpLog = fopen(LogFileName, "A")) == NULL) {
#else
  if ((fpLog = fopen(LogFileName, "a")) == NULL) {
#endif
    if (!LogError)
      perror(LogFileName);
    LogError = 1;
  } else {
    fprintf(fpLog, "<%05d> %s: %s\n", procStamp, name, description);
    fclose(fpLog);
    LogError = 0;
  }
}
