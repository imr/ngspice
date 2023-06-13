/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

/* get input token from 'line',
 *  and return a pointer to it in 'token'
 */

/* INPgetTok: node names
   INPgetUTok: numbers and other elements in expressions
   (called from INPevaluate)
*/

#include "ngspice/ngspice.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/inpdefs.h"
#include "inpxx.h"


/*-------------------------------------------------------------------
 * INPgetTok -- this fcn extracts a generic input token from
 * 'line' and returns a pointer to it in 'token'.
 *------------------------------------------------------------------*/

int
INPgetTok(char **line, char **token, int gobble)
/* gobble: eat non-whitespace trash AFTER token? */
{
    char *point;
    int signstate;

    if (!*line) {
        *token = NULL;
        return (E_PARMVAL);
    }

    /* scan along throwing away garbage characters until end of line
       or a separation char is found */
    for (point = *line; *point != '\0'; point++) {
        if (*point == ' ')
            continue;
        if (*point == '\t')
            continue;
        if (*point == '\r')
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

    /* now find all good characters up to next occurance of a
       separation character. */
    signstate = 0;
    for (point = *line; *point != '\0'; point++) {

        if (*point == ' ')
            break;
        if (*point == '\t')
            break;
        if (*point == '\r')
            break;
        if (*point == '=')
            break;
        if (*point == '(')
            break;
        if (*point == ')')
            break;
        if (*point == ',')
            break;
        /* This is not complex enough to catch all errors,
           but it will get the "good" parses */
        if ((*point == '+') || (*point == '-')) {
            /* Treat '+' signs same as '-' signs */
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

        if (isdigit_c(*point) || *point == '.') {
            if (signstate > 1)
                signstate = 3;
            else
                signstate = 1;
        } else if (tolower_c(*point) == 'e' && signstate == 1)
            signstate = 2;
        else
            signstate = 3;
    }

    if (point == *line && *point)       /* Weird items, 1 char */
        point++;

    *token = copy_substring(*line, point);
    if (!*token)
        return (E_NOMEM);

    *line = point;

    /* gobble garbage to next token */
    for (; **line != '\0'; (*line)++) {
        if (**line == ' ')
            continue;
        if (**line == '\t')
            continue;
        if (**line == '\r')
            continue;
        if ((**line == '=') && gobble)
            continue;
        if ((**line == ',') && gobble)
            continue;
        break;
    }

#ifdef TRACE
    /* SDB debug statement */
    /* printf("found generic token (%s) and rest of line (%s)\n", *token, *line); */
#endif

    return (OK);
}


/*-------------------------------------------------------------------
 * INPgetNetTok -- this fcn extracts an input netname token from
 * 'line' and returns a pointer to it in 'token'.
 * This fcn cloned from INPgetTok by SDB to enable
 * complex netnames (e.g. netnames like '+VCC' and 'IN-').
 * mailto:sdb@cloud9.net -- 4.7.2003
 *------------------------------------------------------------------*/

int
INPgetNetTok(char **line, char **token, int gobble)
/* gobble: eat non-whitespace trash AFTER token? */
{
    char *point;

    /* scan along throwing away garbage characters until end of line
       or a separation char is found */
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

    /* now find all good characters up to next occurance of a
       separation character. INPgetNetTok is very liberal about
       what it accepts.  */
    for (point = *line; *point != '\0'; point++) {
        if (*point == ' ')
            break;
        if (*point == '\t')
            break;
        if (*point == '\r')
            break;
        if (*point == '=')
            break;
        if (*point == ',')
            break;
        if (*point == ')')
            break;
    }

    if (point == *line && *point)       /* Weird items, 1 char */
        point++;

    *token = copy_substring(*line, point);
    if (!*token)
        return (E_NOMEM);

    *line = point;

    /* gobble garbage to next token */
    for (; **line != '\0'; (*line)++) {
        if (**line == ' ')
            continue;
        if (**line == '\t')
            continue;
        if (**line == '\r')
            continue;
        if ((**line == '=') && gobble)
            continue;
        if ((**line == ',') && gobble)
            continue;
        break;
    }

#ifdef TRACE
    /* SDB debug statement */
    /* printf("found netname token (%s) and rest of line (%s)\n", *token, *line); */
#endif

    return (OK);
}


/*-------------------------------------------------------------------
 * INPgetUTok -- this fcn extracts an input refdes token from
 * 'line' and returns a pointer to it in 'token'.
 *------------------------------------------------------------------*/

int
INPgetUTok(char **line, char **token, int gobble)
/* gobble: eat non-whitespace trash AFTER token? */
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
        separator = '\0';

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
        if (*point == '+' || *point == '-') {
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

        if (isdigit_c(*point) || *point == '.') {
            if (signstate > 1)
                signstate = 3;
            else
                signstate = 1;
        } else if (tolower_c(*point) == 'e' && signstate == 1)
            signstate = 2;
        else
            signstate = 3;
    }

    if (separator && *point == separator)
        point--;

    if (point == *line && *point)       /* Weird items, 1 char */
        point++;

    *token = copy_substring(*line, point);
    if (!*token)
        return (E_NOMEM);

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

#ifdef TRACE
    /* SDB debug statement */
    /* printf("found refdes token (%s) and rest of line (%s)\n",*token,*line); */
#endif

    return (OK);
}
