/*=== Static CNVgettok ROUTINE ================*/
/*
Get the next token from the input string.  The input string pointer
is advanced to the following token and the token from the input
string is copied to malloced storage and a pointer to that storage
is returned.  The original input string is undisturbed.
*/

#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "gettokens.h"



char *CNVgettok(char **s)
{
    char    *buf;       /* temporary storage to copy token into */
    /*char    *temp;*/      /* temporary storage to copy token into */

    int     i;

    /* allocate space big enough for the whole string */

    if ((buf = (char *) malloc(strlen(*s) + 1)) == (char *) NULL) {
        cm_message_printf("cannot allocate buffer to tokenize");
        return (char *) NULL;
    }

    /* skip over any white space */

    while (isspace(**s) || (**s == '=') ||
            (**s == '(') || (**s == ')') || (**s == ','))
        (*s)++;

    /* isolate the next token */

    switch (**s) {

    case '\0':           /* End of string found */
        if (buf) {
            free(buf);
        }
        return NULL;


    default:             /* Otherwise, we are dealing with a    */
        /* string representation of a number   */
        /* or a mess o' characters.            */
        i = 0;
        while ( (**s != '\0') &&
                (! ( isspace(**s) || (**s == '=') ||
                     (**s == '(') || (**s == ')') ||
                     (**s == ',')
                   ) )  ) {
            buf[i] = **s;
            i++;
            (*s)++;
        }
        buf[i] = '\0';
        break;
    }

    /* skip over white space up to next token */

    while (isspace(**s) || (**s == '=') ||
            (**s == '(') || (**s == ')') || (**s == ','))
        (*s)++;

    /* make a copy using only the space needed by the string length */


    {
        char * const ret_str = (char *) malloc(strlen(buf) + 1);
        if (ret_str == (char *) NULL) {
            return (char *) NULL;
        }
        (void) strcpy(ret_str, buf);
        free(buf);
        return ret_str;
    }
} /* end of function CNVgettok */



/*
Get the next token from the input string together with its type.
The input string pointer
is advanced to the following token and the token from the input
string is copied to malloced storage and a pointer to that storage
is returned.  The original input string is undisturbed.
*/
char * CNVget_token(char **s, Cnv_Token_Type_t *type)
{
    char    *ret_str;   /* storage for returned string */

    /* get the token from the input line */
    ret_str = CNVgettok(s);

    /* if no next token, return */
    if (ret_str == NULL) {
        *type = CNV_NO_TOK;
        return NULL;
    }

    /* else, determine and return token type */
    switch (*ret_str) {
    default:
        *type = CNV_STRING_TOK;
        break;
    }
    return ret_str;
} /* end of function CNVget_token */



/*
  Function takes as input a string token from a SPICE
  deck and returns a floating point equivalent value.
*/
int cnv_get_spice_value(char   *str,       /* IN - The value text e.g. 1.2K */
                    double *p_value)   /* OUT - The numerical value     */
{
    /* the following were "int4" devices - jpm */
    size_t  len;
    size_t  i;
    int     n_matched;

    /* A SPICE size line. <= 80 characters plus '\n\0' */
    typedef char line_t[82];
    line_t  val_str;

    char    c = ' ';
    char    c1;

    double  scale_factor;
    double  value;

    /* Scan the input string looking for an alpha character that is not  */
    /* 'e' or 'E'.  Such a character is assumed to be an engineering     */
    /* suffix as defined in the Spice 2G.6 user's manual.                */

    len = strlen(str);
    if (len > sizeof(val_str) - 1)
        len = sizeof(val_str) - 1;

    for (i = 0; i < len; i++) {
        c = str[i];
        if (isalpha(c) && (c != 'E') && (c != 'e'))
            break;
        else if (isspace(c))
            break;
        else
            val_str[i] = c;
    }
    val_str[i] = '\0';

    /* Determine the scale factor */

    if ((i >= len) || (! isalpha(c)))
        scale_factor = 1.0;
    else {
        c = (char) tolower(c);

        switch (c) {

        case 't':
            scale_factor = 1.0e12;
            break;

        case 'g':
            scale_factor = 1.0e9;
            break;

        case 'k':
            scale_factor = 1.0e3;
            break;

        case 'u':
            scale_factor = 1.0e-6;
            break;

        case 'n':
            scale_factor = 1.0e-9;
            break;

        case 'p':
            scale_factor = 1.0e-12;
            break;

        case 'f':
            scale_factor = 1.0e-15;
            break;

        case 'm':
            i++;
            if (i >= len) {
                scale_factor = 1.0e-3;
                break;
            }
            c1 = str[i];
            if (!isalpha(c1)) {
                scale_factor = 1.0e-3;
                break;
            }
            c1 = (char) toupper(c1);
            if (c1 == 'E')
                scale_factor = 1.0e6;
            else if (c1 == 'I')
                scale_factor = 25.4e-6;
            else
                scale_factor = 1.0e-3;
            break;

        default:
            scale_factor = 1.0;
        }
    }

    /* Convert the numeric portion to a float and multiply by the */
    /* scale factor.                                              */

    n_matched = sscanf(val_str, "%le", &value);

    if (n_matched < 1) {
        *p_value = 0.0;
        return -1;
    }

    *p_value = value * scale_factor;
    return 0;
} /* end of function cnv_get_spice_value */



