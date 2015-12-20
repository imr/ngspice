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

/*=== CONSTANTS ========================*/

#define OK 0
#define FAIL 1

/* Type definition for each possible token returned. */
typedef enum token_type_s { CNV_NO_TOK, CNV_STRING_TOK } Cnv_Token_Type_t;

extern char *CNVget_token(char **s, Cnv_Token_Type_t *type);

/*=== MACROS ===========================*/

#if defined(__MINGW32__) || defined(_MSC_VER)
#define DIR_PATHSEP    "\\"
#else
#define DIR_PATHSEP    "/"
#endif

#if defined(_MSC_VER)
#define strdup _strdup
#define snprintf _snprintf
#endif



char *
CNVgettok(char **s)
{
    char    *buf;       /* temporary storage to copy token into */
    /*char    *temp;*/      /* temporary storage to copy token into */
    char    *ret_str;   /* storage for returned string */

    int     i;

    /* allocate space big enough for the whole string */

    buf = (char *) malloc(strlen(*s) + 1);

    /* skip over any white space */

    while (isspace_c(**s) || (**s == '=') ||
            (**s == '(') || (**s == ')') || (**s == ','))
        (*s)++;

    /* isolate the next token */

    switch (**s) {

    case '\0':           /* End of string found */
        if (buf)
                    free(buf);
        return NULL;


    default:             /* Otherwise, we are dealing with a    */
        /* string representation of a number   */
        /* or a mess o' characters.            */
        i = 0;
        while ( (**s != '\0') &&
                (! ( isspace_c(**s) || (**s == '=') ||
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

    while (isspace_c(**s) || (**s == '=') ||
            (**s == '(') || (**s == ')') || (**s == ','))
        (*s)++;

    /* make a copy using only the space needed by the string length */


    ret_str = (char *) malloc(strlen(buf) + 1);
    ret_str = strcpy(ret_str,buf);

    if (buf) free(buf);

    return ret_str;
}



/*=== Static CNVget_token ROUTINE =============*/
/*
Get the next token from the input string together with its type.
The input string pointer
is advanced to the following token and the token from the input
string is copied to malloced storage and a pointer to that storage
is returned.  The original input string is undisturbed.
*/

char *
CNVget_token(char **s, Cnv_Token_Type_t *type)
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
}
