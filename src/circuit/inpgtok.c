/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

    /* get input token from 'line', 
     *  and return a pointer to it in 'token'
     */

/* INPgetTok: node names
   INPgetUTok: numbers and other elements in expressions
	   (called from INPevaluate)
 */

#include "ngspice.h"
#include <stdio.h>
#include "iferrmsg.h"
#include "inpdefs.h"
#include "inp.h"

int INPgetTok(char **line, char **token, int gobble)
		    /* eat non-whitespace trash AFTER token? */
{
    char *point;
    int signstate;

    /* scan along throwing away garbage characters */
    for (point = *line; *point != '\0'; point++) {
	if (*point == ' ')
	    continue;
	if (*point == '\t')
	    continue;
	if (*point == '=')
	    continue;
	if (*point == '(')
	    continue;
	if (*point == ')')
	    continue;
	if (*point == ',')
	    continue;
	break;
    }
    /* mark beginning of token */
    *line = point;
    /* now find all good characters */
    signstate = 0;
    for (point = *line; *point != '\0'; point++) {
	if (*point == ' ')
	    break;
	if (*point == '\t')
	    break;
	if (*point == '=')
	    break;
	if (*point == '(')
	    break;
	if (*point == ')')
	    break;
	if (*point == ',')
	    break;
	/* This is not complex enough to catch all errors, but it will get the "good" parses */
	if (*point == '+' && (signstate == 1 || signstate == 3))
	    break;
	if (*point == '-' && (signstate == 1 || signstate == 3))
	    break;
	if (*point == '*')
	    break;
	if (*point == '/')
	    break;
	if (*point == '^')
	    break;
	if (isdigit(*point) || *point == '.') {
	    if (signstate > 1)
		signstate = 3;
	    else
		signstate = 1;
	} else if (tolower(*point) == 'e' && signstate == 1)
	    signstate = 2;
	else
	    signstate = 3;

    }
    if (point == *line && *point)	/* Weird items, 1 char */
	point++;
    *token = (char *) MALLOC(1 + point - *line);
    if (!*token)
	return (E_NOMEM);
    (void) strncpy(*token, *line, point - *line);
    *(*token + (point - *line)) = '\0';
    *line = point;
    /* gobble garbage to next token */
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
    /*printf("found token (%s) and rest of line (%s)\n",*token,*line); */
    return (OK);
}

int INPgetUTok(char **line, char **token, int gobble)


		    /* eat non-whitespace trash AFTER token? */
{
    char *point, separator;
    int signstate;
    /* scan along throwing away garbage characters */
    for (point = *line; *point != '\0'; point++) {
	if (*point == ' ')
	    continue;
	if (*point == '\t')
	    continue;
	if (*point == '=')
	    continue;
	if (*point == '(')
	    continue;
	if (*point == ')')
	    continue;
	if (*point == ',')
	    continue;
	break;
    }
    if (*point == '"') {
	separator = '"';
	point++;
    } else if (*point == '\'') {
	separator = '\'';
	point++;
    } else
	separator = 0;

    /* mark beginning of token */
    *line = point;

    /* now find all good characters */
    signstate = 0;
    for (point = *line; *point != '\0'; point++) {
	if (separator) {
	    if (*point == separator)
		break;
	    else
		continue;
	}
	if (*point == ' ')
	    break;
	if (*point == '\t')
	    break;
	if (*point == '=')
	    break;
	if (*point == '(')
	    break;
	if (*point == ')')
	    break;
	if (*point == ',')
	    break;
	/* This is not complex enough to catch all errors, but it will
	   get the "good" parses */
	if (*point == '+' && (signstate == 1 || signstate == 3))
	    break;
	if (*point == '-') {
	    if (signstate == 1 || signstate == 3)
		break;
	    signstate += 1;
	    continue;
	}
	if (*point == '*')
	    break;
	if (*point == '/')
	    break;
	if (*point == '^')
	    break;
	if (isdigit(*point) || *point == '.') {
	    if (signstate > 1)
		signstate = 3;
	    else
		signstate = 1;
	} else if (tolower(*point) == 'e' && signstate == 1)
	    signstate = 2;
	else
	    signstate = 3;
    }
    if (separator && *point == separator)
	point--;
    if (point == *line && *point)	/* Weird items, 1 char */
	point++;
    *token = (char *) MALLOC(1 + point - *line);
    if (!*token)
	return (E_NOMEM);
    (void) strncpy(*token, *line, point - *line);
    *(*token + (point - *line)) = '\0';
    /* gobble garbage to next token */
    for (; *point != '\0'; point++) {
	if (*point == separator)
	    continue;
	if (*point == ' ')
	    continue;
	if (*point == '\t')
	    continue;
	if ((*point == '=') && gobble)
	    continue;
	if ((*point == ',') && gobble)
	    continue;
	break;
    }
    *line = point;
    /*  printf("found token (%s) and rest of line (%s)\n",*token,*line);  */
    return (OK);
}
