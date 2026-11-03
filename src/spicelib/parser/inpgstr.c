/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/*
 * Get string input token from 'line', and return a pointer to it in 'token'
 */

#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/iferrmsg.h"
#include "ngspice/inpdefs.h"
#include "inpxx.h"

#define GARBAGE " \t=(),"

int INPgetStr(char **line, char **token, int gobble)
				/* eat non-whitespace trash AFTER token? */
{
    char *point;
    int escaped = 0;
    char separator = '\0';

    /* Scan along throwing away garbage characters. */

    point = *line;
    while (strchr(GARBAGE, *point))
        ++point;

    if (*point == '"') {
	separator = '"';
	point++;
    } else if (*point == '\'') {
	separator = '\'';
	point++;
    }
    /* mark beginning of token */
    *line = point;

    /* Now find all good characters. */

    if (separator) {
        /* Quoted string is everything up to a closing quote. */

        for (;;) {
            if (!*point || *point == separator)
                break;
            if (*point == '\\' && point[1] == separator) {
                escaped = 1;
                ++point;
            }
            ++point;
        }
    } else {
        /* Scan to next "garbage" character. */

        while (*point && !strchr(GARBAGE, *point))
            ++point;
    }

    /* Create token */

    *token = TMALLOC(char, 1 + point - *line);
    if (!*token)
	return (E_NOMEM);
    (void) strncpy(*token, *line, (size_t) (point - *line));
    *(*token + (point - *line)) = '\0';

    if (separator && *point == separator)
	point++;	/* Skip closing separator */
    *line = point;

    if (escaped) {
        char *wp;

        /* Remove escaped quotes. */

        for (wp = point = *token; *point; ) {
            if (*point == '\\' && point[1] == separator)
                ++point;
            *wp++ = *point++;
        }
        *wp = 0;
    }

    /* Gobble garbage to next token. */

    for (; **line != '\0'; (*line)++) {
	if (**line == ' ')
	    continue;
	if (**line == '\t')
	    continue;
	if ((**line == '=') && gobble)
	    continue;
	if ((**line == ',') && gobble)
	    continue;
	break;
    }
    return (OK);
}
