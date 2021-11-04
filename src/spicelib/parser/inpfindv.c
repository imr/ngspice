/* INPfindVer(line,version)
 *      find the 'version' parameter on the given line and return its
 *      return 'default' as version if not found
 *
 */

#include "ngspice/ngspice.h"
#include <stdio.h>
#include <string.h>
#include "ngspice/inpdefs.h"
#include "inpxx.h"

char *INPfindVer(char *line, char *version)
{
    char *where;

    where = strstr(line, "version");

    if (where != NULL) {    /* found a version keyword on the line */

        where += 7;     /* skip the version keyword */
        while ((*where == ' ') || (*where == '\t') || (*where == '=') ||
               (*where == ',') || (*where == '(') || (*where == ')') ||
               (*where == '+')) {   /* legal white space - ignore */
            where++;
        }

        /* now the magic string */
        if (sscanf(where, "%s", version) != 1) { /* We get the version number */
            sprintf( version, "default" );
            printf("Warning -- Version not specified correct on line \"%s\"\nSetting version to 'default'.\n", line);
        }
        return (NULL);
    }
    else {          /* no level on the line => default */
        sprintf( version, "default" );
        printf("Warning -- Version not specified on line \"%s\"\nSetting version to 'default'.\n", line);
        return (NULL);
    }
}
