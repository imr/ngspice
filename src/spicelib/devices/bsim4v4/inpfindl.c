/**********
Copyright 2000 Regents of the University of California. All rights reserved.
Author: 2000 Weidong Liu
**********/

/* INPfindLev(line)
 * Find the 'level' parameter value on the model file
 * The BSIM4v4 model is level 14 in SPICE3.
 * Please note BSIM5 and BSIM6 will take level 15 and 16 in the future, respectively.
 */

#include "ngspice/ngspice.h"
#include "misc.h"
#include "strext.h"
#include "inpdefs.h"
#include "ngspice/suffix.h"

char *
INPfindLev(line,level)
    char *line;
    int *level;
{   char *where;
    char LevArray[3]; /* save individual level numerals */
    char *LevNumString; /* points to the level string */
    int i_array = 0;
    where = line;

    while(1)
    { where = index(where,'l');
      if(where == 0) /* no 'l' in the line => no 'level' => default */
      { *level = 14; /* the default model is BSIM4v4 */
        return((char *)NULL);
      }
      if(strncmp(where,"level",5)!=0)
      { /* this l isn't in the word 'level', try again */
        where++;    /* make sure we don't match same char again */
        continue;
      }
      /* The word level found, lets look at the rest of the line */
      where += 5;
      while((*where == ' ') || (*where == '\t') || (*where == '=') ||
           (*where == ',') || (*where == '(') || (*where == ')') ||
           (*where == '+'))
      { where++;
      }

      LevNumString = LevArray;
      while(!((*where == ' ') || (*where == '\t') || (*where == '=') ||
           (*where == ',') || (*where == '(') || (*where == ')') ||
           (*where == '+')))
      { LevArray[i_array] = *where;
        i_array++;
        where++;
      }
      LevArray[i_array] = '\0';

      if (strcmp(LevNumString, "1") == 0)
      {  *level=1;
         return((char *)NULL);
      }
      else if (strcmp(LevNumString, "2") == 0)
      {  *level=2;
         return((char *)NULL);
      }
      else if (strcmp(LevNumString, "3") == 0)
      {  *level=3;
         return((char *)NULL);
      }
      else if (strcmp(LevNumString, "4") == 0)
      {  *level=4;
         return((char *)NULL);
      }
      else if (strcmp(LevNumString, "5") == 0)
      {  *level=5;
         return((char *)NULL);
      }
      else if (strcmp(LevNumString, "6") == 0)
      {  *level=6;
         return((char *)NULL);
      }
      else if (strcmp(LevNumString, "7") == 0)
      {  *level=7;
         return((char *)NULL);
      }
      else if (strcmp(LevNumString, "8") == 0)
      {  *level=8;
         return((char *)NULL);
      }
      else if (strcmp(LevNumString, "9") == 0)
      {  *level=9;
         return((char *)NULL);
      }
      else if (strcmp(LevNumString, "10") == 0)
      {  *level=10;
         return((char *)NULL);
      }
      else if (strcmp(LevNumString, "11") == 0)
      {  *level=11;
         return((char *)NULL);
      }
      else if (strcmp(LevNumString, "12") == 0)
      {  *level=12;
         return((char *)NULL);
      }
      else if (strcmp(LevNumString, "13") == 0)
      {  *level=13;
         return((char *)NULL);
      }
      else if (strcmp(LevNumString, "14") == 0)
      {  *level=14;
         return((char *)NULL);
      }
      else
      {  *level=14;
         printf("illegal argument to 'level' - BSIM4v4 assumed\n");
         return(INPmkTemp("illegal argument to 'level' - BSIM4v4 assumed"));
      }
    }
}
