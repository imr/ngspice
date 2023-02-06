/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/

/*
 * String functions
 */
#include <ctype.h>
#include <stdarg.h>

#include "ngspice/ngspice.h"
#include "ngspice/stringutil.h"
#include "ngspice/stringskip.h"
#include "ngspice/dstring.h"


/* Instantiations of string functions */
extern inline char *copy(const char *str);
extern inline char *copy_substring(const char *str, const char *end);
extern inline int scannum(const char *str);
extern inline int substring(const char *sub, const char *str);



static size_t get_kr_msb_factor(size_t n);
static size_t kr_hash(size_t n, const char *p);
static inline const char *next_substr(
        size_t n_char_pattern, const char *p_pattern,
        const char **pp_string, const char * const p_last,
        const size_t msb_factor, const size_t h_pattern, size_t *p_h_string);
static bool can_overlap(size_t n_char_pattern, const char * const p_pattern);

static void findtok_np(char** p_str, char** p_token, char** p_token_end);


/* This function returns true if the string s begins with the
 * string p and false otherwise. */
int prefix(const char *p, const char *s)
{
    while (*p && (*p == *s))
        p++, s++;

    return *p == '\0';
} /* end of function prefix */



/* This function returns 1 if string begins with prefix and 0 otherwise.
 * Neither the prefix nor string needs a null termination. */
int prefix_n(size_t n_char_prefix, const char *prefix,
        size_t n_char_string, const char *string)
{
    /*Test that string is long enough */
    if (n_char_prefix > n_char_string) {
        return 0;
    }

    return memcmp(prefix, string, n_char_prefix) == 0;
} /* end of function prefix_n */




/* This function allocates a buffer and copies the specified number of
 * characters from the input string into the buffer followed by a
 * terminating null.
 *
 * Paramters
 * str: String to copy
 * n_char: Number of characters to copy
 *
 * Return values
 * NULL: Allocation failure
 * otherwise: The initialized string.
 */
char *dup_string(const char *str, size_t n_char)
{
    char *p = TMALLOC(char, n_char + 1);

    if (p != NULL) {
        (void) memcpy(p, str, n_char + 1);
        p[n_char] = '\0';
    }
    return p;
} /* end of function dup_string */



char *tvprintf(const char *fmt, va_list args)
{
    static char buf[1024];
    char *p = buf;
    int size = sizeof(buf);
    int nchars;

    for (;;) {

        va_list ap;

        va_copy(ap, args);
        nchars = vsnprintf(p, (size_t) size, fmt, ap);
        va_end(ap);

        /* This case was previously handled by doubling the size of
         * the buffer for "compatibility to old implementations."
         * However, vsnprintf is defined in both C99 and SUSv2 from 1997.
         * There is a slight difference which does not affect this
         * usage, but both return negative values (possibly -1) on an
         * encoding error, which would lead to an infinte loop (until
         * memory was exhausted) with the old behavior */

        if (nchars < 0) {
            fprintf(stderr, "Error: tvprintf failed\n");
            controlled_exit(-1);
        }

        if (nchars < size) { /* String formatted OK */
            break;
        }

        /* Output was truncated. Returned value is the number of chars
         * that would have been written if the buffer were large enough
         * excluding the terminiating null. */
        size = nchars + 1; /* min required allocation size */

        /* Allocate a larger buffer */
        if (p == buf) {
            p = TMALLOC(char, size);
        }
        else {
            p = TREALLOC(char, p, size);
        }
    }

    /* Return the formatted string, making a copy on the heap if the
     * stack's buffer (buf) contains the string */
    return (p == buf) ? dup_string(p, (size_t) nchars) : p;
} /* end of function tvprintf */



/* This function returns an allocation containing the string formatted
 * according to fmt and the variadic argument list provided. It is a wrapper
 * around tvprintf() which processes the argumens as a va_list. */
char *tprintf(const char *fmt, ...)
{
    char *rv;
    va_list ap;

    va_start(ap, fmt);
    rv = tvprintf(fmt, ap);
    va_end(ap);

    return rv;
} /* end of function tprintf */


/* Append one character to a string. Don't check for overflow. */
/* Almost like strcat( ) XXX */
void appendc(char *s, char c)
{
    while (*s) {
        s++;
    }
    *s++ = c;
    *s = '\0';
} /* end of function appendc */



/* Returns the unsigned number at *p_str or 0 if there is none. *p_str
 * points to the first character after the number that was read, so
 * it is possible to distingish between the value 0 and a missing number
 * by testing if the string has been advanced. */
int scannum_adv(char **p_str)
{
    const char *str = *p_str;
    int i = 0;

    while (isdigit_c(*str)) {
        i = i * 10 + *(str++) - '0';
    }

    *p_str = (char *) str; /* locate end of number */
    return i;
} /* end of function scannum_adv */



/* This function returns the integer at the current string location.
 * The string does not need to be null-terminated.
 *
 * Parameters
 * str: String containing the integer to return at the beginning
 * n: Number of characters in the string
 * p_value: Address where the integer is returned
 *
 * Return values
 * -1: No integer present
 * -2: Overflow
 * >0: Number of characters in the integer
 */
int get_int_n(const char *str, size_t n, int *p_value)
{
    if (n == 0) { /* no string */
        return -1;
    }

    unsigned int value = 0;
    const char *p_cur = str;
    const char * const p_end = str + n;
    bool f_neg;
    if (*p_cur == '-') { /* Check for leading negative sign */
        f_neg = 1;
        ++p_cur;
    }
    else {
        f_neg = 0;
    }
   
    /* Iterate over chars until end or char that is not numeric */ 
    for ( ; p_cur != p_end; ++p_cur) {
        char ch_cur = *p_cur;
        if (!isdigit(ch_cur)) { /* Test for exit due to non-numeric char */
            break;
        }

        /* Compute new value and check for overflow. */
        const unsigned int value_new =
                10 * value + (unsigned int) (ch_cur - '0');
        if (value_new < value) {
            return -2;
        }
        value = value_new;
    } /* end of loop over digits */

    /* Test for at least one digit */
    if (p_cur == str + f_neg) {
        return -1; /* no digit */
    }

    /* Test for overflow.
     * If negative, can be 1 greater (-2**n vs 2**n -1) */
    if (value - (unsigned int) f_neg > (unsigned int) INT_MAX) {
        return -2;
    }

    /* Take negative if negative sign present. (This operation works
     * correctly if value == INT_MIN since -INT_MIN == INT_MIN */
    *p_value = f_neg ? -(int) value : (int) value;

    return (int) (p_cur - str); /* number of chars in the number */
} /* end of function get_int_n */



/* Case insensitive str eq. */
/* Like strcasecmp( ) XXX */
int cieq(const char *p, const char *s)
{
    for (; *p; p++, s++) {
        if (tolower_c(*p) != tolower_c(*s)) {
            return FALSE;
        }
    }

    return *s == '\0';
} /* end of function cieq */



/* Case-insensitive string compare fore equialty with explicit length
 * given. Neither character array needs to be null terminated. By not
 * including the trailing null in the count, it can be used to check
 * for a prefix. This function is useful for avoiding string copies
 * to temporary buffers and the potential for buffer overruns that
 * can occur when using temporary buffers without checking lengths. */
int cieqn(const char *p, const char *s, size_t n)
{
    size_t i;
    for (i = 0; i < n; ++i) {
        if (tolower_c(p[i]) != tolower_c(s[i])) {
            return FALSE;
        }
    }
    return TRUE; /* all chars matched */
} /* end of function cineq */


/* Case insensitive prefix. */
int ciprefix(const char *p, const char *s)
{
    for (; *p; p++, s++)
        if (tolower_c(*p) != tolower_c(*s)) {
            return FALSE;
        }

    return TRUE;
} /* end of function ciprefix */



void strtolower(char *str)
{
    if (!str) {
        return;
    }

    for (; *str; str++) {
        *str = tolower_c(*str);
    }
} /* end of function strtolower */



void strtoupper(char *str)
{
    if (!str) {
        return;
    }

    for (; *str; str++) {
        *str = toupper_c(*str);
    }
} /* end of function strtoupper */


#ifdef CIDER

/*
 * Imported from cider file support/strmatch.c
 * Original copyright notice:
 * Author: 1991 David A. Gates, U. C. Berkeley CAD Group
 *
 */

/*
 * Case-insensitive test of whether p is a prefix of s and at least the
 * first n characters are the same
 */

int cinprefix(char *p, char *s, int n)
{
    if (!p || !s) {
        return 0;
    }

    for (; *p; p++, s++, n--) {
        if (tolower_c(*p) != tolower_c(*s)) {
            return 0;
        }
    }

    return n <= 0;
} /* end of function cinprefix */



/*
 * Case-insensitive match of prefix string p against string s
 * returns the number of matching characters
 *
 */

int
cimatch(char *p, char *s)
{
    int n = 0;

    if (!p || !s)
        return 0;

    for (; *p; p++, s++, n++)
        if (tolower_c(*p) != tolower_c(*s))
            return n;

    return n;
}

#endif /* CIDER */


/*-------------------------------------------------------------------------*
 * gettok skips over whitespace and returns the next token found.  This is
 * the original version.  It does not "do the right thing" when you have
 * parens or commas anywhere in the nodelist.  Note that I left this unmodified
 * since I didn't want to break any fcns which called it from elsewhere than
 * subckt.c.  -- SDB 12.3.2003.
 *-------------------------------------------------------------------------*/
char *
gettok(char **s)
{
    char c;
    int paren;
    const char *token, *token_e;

    if (!*s)
        return NULL;

    paren = 0;

    *s = skip_ws(*s);
    if (!**s)
        return NULL;

    token = *s;
    while ((c = **s) != '\0' && !isspace_c(c)) {
        if (c == '(')
            paren += 1;
        else if (c == ')')
            paren -= 1;
        else if (c == ',' && paren < 1)
            break;
        (*s)++;
    }
    token_e = *s;

    while (isspace_c(**s) || **s == ',')
        (*s)++;

    return copy_substring(token, token_e);
}


/*-------------------------------------------------------------------------*
 * nexttok skips over whitespaces and the next token in s
 *   returns NULL if there is nothing left to skip.
 * It replaces constructs like txfree(gettok(&actstring)) by
 * actstring = nexttok(actstring). This is derived from the original gettok version.
 * It does not "do the right thing" when
 * you have parens or commas anywhere in the nodelist.
 *-------------------------------------------------------------------------*/

char *
nexttok(const char *s)
{
    if (!s)
        return NULL;
    int paren = 0;

    s = skip_ws(s);
    if (!*s)
        return NULL;

    for (; *s && !isspace_c(*s); s++)
        if (*s == '(')
            paren += 1;
        else if (*s == ')')
            paren -= 1;
        else if (*s == ',' && paren < 1)
            break;

    while (isspace_c(*s) || *s == ',')
        s++;

    return (char *) s;
}

/*-------------------------------------------------------------------------*
 * nexttok skips over whitespaces and the next token in s
 *   returns NULL if there is nothing left to skip.
 * It replaces constructs like txfree(gettok(&actstring)) by
 * actstring = nexttok(actstring). This is derived from the gettok_np version.
 * It acts like gettok, except that it treats parens and commas like
 * whitespace.
 *-------------------------------------------------------------------------*/

char*
nexttok_noparens(const char* s)
{
    if (!s)
        return NULL;

    s = skip_ws(s);
    if (!*s)
        return NULL;

    for (; *s && !isspace_c(*s); s++)
        if (*s == '(')
            break;
        else if (*s == ')')
            break;
        else if (*s == ',')
            break;

    while (isspace_c(*s) || *s == ',' || *s == '(' || *s == ')')
        s++;

    return (char*)s;
}


/*-------------------------------------------------------------------------*
 * gettok skips over whitespaces or '=' and returns the next token found,
 * if the token is something like i(xxx), v(yyy), or v(xxx,yyy)
 *   -- h_vogt 10.07.2010.
 *-------------------------------------------------------------------------*/

char *
gettok_iv(char **s)
{
    char *p_src = *s; /* location in source string */
    char c; /* current char */

    /* Step past whitespace and '=' */
    while (isspace_c(c = *p_src) || (c == '=')) {
        p_src++;
    }

    /* Test for valid leading character */
    if (((c =*p_src) == '\0') ||
            ((c != 'v') && (c != 'i') && (c != 'V') && (c != 'I'))) {
        *s = p_src; /* update position in string */
        return (char *) NULL;
    }

    /* Allocate buffer for token being returned */
    char * const token = TMALLOC(char, strlen(p_src) + 1);
    char *p_dst = token; /* location in token */

    // add v or i to buf
    *p_dst++ = *p_src++;

    {
        int n_paren = 0;
        /* Skip any space between v/V/i/I and '(' */
        p_src = skip_ws(p_src);

        while ((c = *p_src) != '\0') {
            /* Keep track of nesting level */
            if (c == '(') {
                n_paren++;
            }
            else if (c == ')') {
                n_paren--;
            }

            if (isspace_c(c)) { /* Do not copy whitespace to output */
                p_src++;
            }
            else {
                *p_dst++ = *p_src++;
                if (n_paren == 0) {
                    break;
                }
            }
        }
    }

    /* Step past whitespace and ',' */
    while (isspace_c(c = *p_src) || (c == ',')) {
        p_src++;
    }

    *s = p_src; /* update position in string */
    return token;
} /* end of function gettok_iv */



/* findtok_noparen() does the string scanning for gettok_noparens() but
 * does not allocate a token. Hence it is useful when a copy of the token
 * is not required */
void findtok_noparen(char **p_str, char **p_token, char **p_token_end)
{
    char *str = *p_str;

    str = skip_ws(str);

    if (!*str) {
        *p_str = str;
        *p_token = (char *) NULL;
        return;
    }

    *p_token = str; /* Token starts after whitespace */
    {
        char c;
        while ((c = *str) != '\0' &&
               !isspace_c(c) &&
               (c != '(') &&
               (c != ')') &&
               (c != ',')
            ) {
            str++;
        }
    }
    *p_token_end = str;

    str = skip_ws(str);
    *p_str = str;
} /* end of function findtok_noparen */



/*-------------------------------------------------------------------------*
 * gettok_noparens was added by SDB on 4.21.2003.
 * It acts like gettok, except that it treats parens and commas like
 * whitespace while looking for the POLY token.  That is, it stops
 * parsing and returns when it finds one of those chars.  It is called from
 * 'translate' (subckt.c).
 *-------------------------------------------------------------------------*/
char *gettok_noparens(char **s)
{
    char *token, *token_e;

    if (!*s)
        return NULL;

    findtok_noparen(s, &token, &token_e);
    if (token == (char *) NULL) {
        return (char *) NULL; /* return NULL if we come to end of line */
    }

    return copy_substring(token, token_e);
} /* end of function gettok_noparens */


/* findtok_np() does the string scanning for gettok_np() but
 * does not allocate a token. It skips over all white spaces, ',',  '('and ')' */
static
void findtok_np(char** p_str, char** p_token, char** p_token_end)
{
    char* str = *p_str;

    while (isspace_c(*str) || *str == ',' || *str == '(' || *str == ')')
        str++;

    if (!*str) {
        *p_str = str;
        *p_token = (char*)NULL;
        return;
    }

    *p_token = str; /* Token starts after whitespace */
    {
        char c;
        while ((c = *str) != '\0' &&
            !isspace_c(c) &&
            (c != '(') &&
            (c != ')') &&
            (c != ',')
            ) {
            str++;
        }
    }
    *p_token_end = str;

    while (isspace_c(*str) || *str == ',' || *str == '(' || *str == ')')
        str++;

    *p_str = str;
} /* end of function findtok_noparen */



/*-------------------------------------------------------------------------*
 * gettok_np acts like gettok, except that it treats parens and commas like
 * whitespace. That is, it stops parsing and returns when it finds one of
 * those chars.  It then moves s beyond all white spaces, ',',  '('and ')'.
 *-------------------------------------------------------------------------*/
char* gettok_np(char** s)
{
    char* token, * token_e;

    if (!*s)
        return NULL;

    findtok_np(s, &token, &token_e);
    if (token == (char*)NULL) {
        return (char*)NULL; /* return NULL if we come to end of line */
    }

    return copy_substring(token, token_e);
} /* end of function gettok_noparens */

/*-------------------------------------------------------------------------*
* gettok_model acts like gettok_noparens, however when it encounters a '{', 
* it searches for the corresponding '}' and adds the string to the output
* token.
*-------------------------------------------------------------------------*/
char *
gettok_model(char **s)
{
    char c;
    const char *token, *token_e;

    if (!*s)
        return NULL;

    *s = skip_ws(*s);

    if (!**s)
        return NULL;  /* return NULL if we come to end of line */

    token = *s;
    while ((c = **s) != '\0' &&
        !isspace_c(c) &&
        (**s != '(') &&
        (**s != ')') &&
        (**s != ',')
        ) {
        (*s)++;
        if (**s == '{') {
            char *tmpstr = gettok_char(s, '}', FALSE, TRUE);
            tfree(tmpstr);
        }
    }
    token_e = *s;

    *s = skip_ws(*s);

    return copy_substring(token, token_e);
}



char *
gettok_instance(char **s)
{
    char c;
    const char *token, *token_e;

    if (!*s)
        return NULL;

    *s = skip_ws(*s);

    if (!**s)
        return NULL;  /* return NULL if we come to end of line */

    token = *s;
    while ((c = **s) != '\0' &&
           !isspace_c(c) &&
           (**s != '(') &&
           (**s != ')')
        ) {
        (*s)++;
    }
    token_e = *s;

    /* Now iterate up to next non-whitespace char */
    *s = skip_ws(*s);

    return copy_substring(token, token_e);
}


/* get the next token starting at next non white space, stopping
   at p. If inc_p is true, then including p, else excluding p.
   Return NULL if p is not found.
   If '}', ']'  or ')' and nested is true, find corresponding p.
*/

char *
gettok_char(char **s, char p, bool inc_p, bool nested)
{
    char c;
    const char *token, *token_e;

    if (!*s)
        return NULL;

    *s = skip_ws(*s);

    if (!**s)
        return NULL;  /* return NULL if we come to end of line */

    token = *s;
    if (nested && ((p == '}') || (p == ')') || (p == ']'))) {
        char q;
        int count = 0;
        /* find opening bracket */
        if (p == '}')
            q = '{';
        else if (p == ']')
            q = '[';
        else
            q = '(';
        /* add string in front of q, excluding q */
        while ((c = **s) != '\0' && (**s != q))
            (*s)++;
        /* return if nested bracket found, excluding its character */
        while ((c = **s) != '\0') {
            if (c == q)
                count++;
            else if (c == p)
                count--;
            if (count == 0)
                break;
            (*s)++;
        }
    }
    else
        /* just look for p and return string, excluding p */
        while ((c = **s) != '\0' && (**s != p))
            (*s)++;

    if (c == '\0')
        /* p not found */
        return NULL;

    if (inc_p)
        /* add p */
        (*s)++;

    token_e = *s;

    /* Now iterate up to next non-whitespace char */
    *s = skip_ws(*s);

    return copy_substring(token, token_e);
}


/*-------------------------------------------------------------------------*
 * gettok_node was added by SDB on 12.3.2003
 * It acts like gettok, except that it treats parens and commas like
 * whitespace (i.e. it ignores them).  Use it when parsing through netnames
 * (node names) since they may be grouped using ( , ).
 *-------------------------------------------------------------------------*/

char *
gettok_node(char **s)
{
    char c;
    const char *token, *token_e;

    if (*s == NULL)
        return NULL;

    while (isspace_c(**s) ||
           (**s == '(') ||
           (**s == ')') ||
           (**s == ',')
        )
        (*s)++;   /* iterate over whitespace and ( , ) */

    if (!**s)
        return NULL;  /* return NULL if we come to end of line */

    token = *s;
    while ((c = **s) != '\0' &&
           !isspace_c(c) &&
           (**s != '(') &&
           (**s != ')') &&
           (**s != ',')
        )            /* collect chars until whitespace or ( , ) */
        (*s)++;

    token_e = *s;

    /* Now iterate up to next non-whitespace char */
    while (isspace_c(**s) ||
           (**s == '(') ||
           (**s == ')') ||
           (**s == ',')
        )
        (*s)++;   /* iterate over whitespace and ( , ) */

    return copy_substring(token, token_e);
}


/*-------------------------------------------------------------------------*
 * get_l_paren iterates the pointer forward in a string until it hits
 * the position after the next left paren "(".  It returns 0 if it found a left
 * paren, 1 if no left paren is found, -1 if left paren is the last character.
 * It is called from 'translate' (subckt.c).
 *-------------------------------------------------------------------------*/

int
get_l_paren(char **s)
{
    while (**s && (**s != '('))
        (*s)++;

    if (!**s)
        return 1;

    (*s)++;

    if (**s == '\0')
        return -1;

    return 0;
}


/*-------------------------------------------------------------------------*
 * get_r_paren iterates the pointer forward in a string until it hits
 * the position after the next right paren ")".  It returns 0 if it found a right
 * paren, 1 if no right paren is found, and -1 if right paren is te last
 * character.  It is called from 'translate' (subckt.c).
 *-------------------------------------------------------------------------*/

int
get_r_paren(char **s)
{
    while (**s && (**s != ')'))
        (*s)++;

    if (!**s)
        return 1;

    (*s)++;

    if (**s == '\0')
        return -1;

    return 0;
}

/*-------------------------------------------------------------------------*
 * this function strips all white space inside parens
 * is needed in gettoks (dotcards.c) for correct processing of expressions
 * like "    .plot v(   5  , 4  ) v( 6 )" -> .plot v(5,4) v(6)"
 *-------------------------------------------------------------------------*/
char *
stripWhiteSpacesInsideParens(const char *str)
{
    str = skip_ws(str); /* Skip leading whitespace */
    const size_t n_char_str = strlen(str);

    /* Allocate buffer for string being built */
    char * const str_out = TMALLOC(char, n_char_str + 1);
    char *p_dst = str_out; /* location in str_out */
    char ch; /* current char */

    /* Process input string until its end */
    for ( ; ; ) {
        /* Add char. If at end of input string, return the string
         * that was built */
        if ((*p_dst++ = (ch = *str++)) == '\0') {
            return str_out;
        }

        /* If the char is a ')' add all non-whitespace until ')' or,
         * if the string is malformed, until '\0' */
        if (ch == '(') {
            for ( ; ; ) {
                /* If at end of input string, the closing ') was missing.
                 * The caller will need to resolve this issue. */
                if ((ch = *str++) == '\0') {
                    *p_dst = '\0';
                    return str_out;
                }

                if (isspace((int) ch)) { /* skip whitespace */
                    continue;
                }

                /* Not whitespace, so add next character */
                *p_dst++ = ch;

                /* If the char that was added was ')', done */
                if (ch == ')') {
                    break;
                }
            } /* end of loop processing () */
        } /* end of case of '(' found */
    } /* end of loop over chars in input string */
} /* end of function stripWhiteSpacesInsideParens */



bool
isquote(char ch)
{
    return ch == '\'' || ch == '"';
}


bool
is_arith_char(char c)
{
    return c != '\0' && strchr("+-*/()<>?:|&^!%\\", c);
}


bool
str_has_arith_char(char *s)
{
    for (; *s; s++)
        if (is_arith_char(*s))
            return TRUE;

    return FALSE;
}


int get_comma_separated_values(char *values[], char *str)
{
    int count = 0;
    char *comma_ptr;

    while ((comma_ptr = strchr(str, ',')) != NULL) {
        char *ptr = skip_back_ws(comma_ptr, str);
        values[count++] = copy_substring(str, ptr);
        str = skip_ws(comma_ptr + 1);
    }
    values[count++] = copy(str);
    return count;
}


/*
  check if the given token matches a model name
  either exact
  then return 1
  or
  modulo a trailing model binning extension '\.[0-9]+'
  then return 2
*/
int model_name_match(const char *token, const char *model_name)
{
    const char *p;
    size_t token_len = strlen(token);

    if (strncmp(token, model_name, token_len) != 0)
        return 0;

    p = model_name + token_len;

    // exact match
    if (*p == '\0')
        return 1;

    // check for .
    if (*p++ != '.')
        return 0;

    // minimum one trailing char
    if (*p == '\0')
        return 0;

    // all of them digits
    for (; *p; p++)
        if (!isdigit_c(*p))
            return 0;

    return 2;
} /* end of funtion model_name_match */



/* This function returns 1 if pattern is a substring anywhere in str and
 * 0 otherwise. A null pattern is considered a mismatch.
 *
 * Uses Karp-Rabin substring matching with base=256 and modulus=1009
 */
int substring_n(size_t n_char_pattern, const char *p_pattern,
        size_t n_char_string, const char *p_string)
{
    /* Test for a pattern to match */
    if (n_char_pattern == 0) {
        return 0;
    }

    /* Test for a string of sufficient length */
    if (n_char_pattern > n_char_string) {
        return 0;
    }

    /* Factor for rolling hash computation */
    const size_t msb_factor = get_kr_msb_factor(n_char_pattern);

    const size_t h_pattern = kr_hash(n_char_pattern, p_pattern);
    size_t h_string = kr_hash(n_char_pattern, p_string);

    /* Compare at beginning. If hashes match, do full compare */
    if (h_pattern == h_string &&
            memcmp(p_pattern, p_string, n_char_pattern) == 0) {
        return 1; /* match at start */
    }

    /* Compare at each possible starting point in the string */
    const char *p_last = p_string + (n_char_string - n_char_pattern - 1);

    return next_substr(n_char_pattern, p_pattern, &p_string, p_last,
            msb_factor, h_pattern, &h_string) == (char *) NULL ?
            0 : 1;
} /* end of function substring_n */



/* This function initializes a scan for substring matches */
void substring_match_init(size_t n_char_pattern, const char *p_pattern,
        size_t n_char_string, const char *p_string, bool f_overlap,
        struct substring_match_info *p_scan_state)
{
    /* Save input info into structure. Note that the strings are not
     * copied, so they must remain allocated and unaltered while the
     * search is in progress. */
    p_scan_state->n_char_pattern = n_char_pattern;
    p_scan_state->p_pattern = p_pattern;
    p_scan_state->n_char_string = n_char_string;
    p_scan_state->p_string = p_string;

    /*** Calculate intermediate data ***/

    /* Test for a pattern to match */
    if (n_char_pattern == 0) {
        p_scan_state->f_done = TRUE;
    }
    /* Test for a string of sufficient length */
    else if (n_char_pattern > n_char_string) {
        p_scan_state->f_done = TRUE;
    }
    else {
        p_scan_state->f_done = FALSE;

        /* Look for overlaps only if possible */
        p_scan_state->f_overlap= f_overlap ?
                !can_overlap(n_char_pattern, p_pattern) : FALSE;
        p_scan_state->n_char_pattern_1 = n_char_pattern - 1;
        p_scan_state->msb_factor = get_kr_msb_factor(n_char_pattern);
        p_scan_state->h_pattern = kr_hash(n_char_pattern, p_pattern);
        p_scan_state->h_string = kr_hash(n_char_pattern, p_string);
        p_scan_state->p_last =
                p_string + (n_char_string - n_char_pattern - 1);
    }

    return;
} /* end of function substring_match_init */



/* This function finds the next substring match
 *
 * Parameter
 * p_scan_state: Address of struct substring_match_info initialized by
 *      substring_match_init()
 *
 * Return value
 * NULL if there is no match or the address of the next match otherwise
 */
char *substring_match_next(struct substring_match_info *p_scan_state)
{
    /* First test if there are no more possible matches */
    if (p_scan_state->f_done) {
        return (char *) NULL;
    }

    /* Find next match, if any */
    const char * const p_match = next_substr(
            p_scan_state->n_char_pattern, p_scan_state->p_pattern,
            &p_scan_state->p_string, p_scan_state->p_last,
            p_scan_state->msb_factor,p_scan_state->h_pattern,
            &p_scan_state->h_string);

    /* Update done status if changed */
    if (p_match == (char *) NULL) {
        p_scan_state->f_done = TRUE;
    }
    else {
        if (!p_scan_state->f_overlap) {
            p_scan_state->p_string +=
                    p_scan_state->n_char_pattern_1; /* end of match */
            p_scan_state->h_string = p_scan_state->h_pattern;
        }
    }

    return (char *) p_match; /* Return result */
} /* end of function substring_match_next */



#ifdef COMPILE_UNUSED_FUNCTIONS
/* This funtion returns the locations of optionally non-overlapping substring
 * matches. For example, in the string aaaaa, aa is found in non-overlapping
 * locations at 0-based offsets 0 and 2 ahd with overlapping allowed atr
 * offsets 0, 1, 2, and 3 */
size_t get_substring_matches(size_t n_char_pattern, const char *p_pattern,
        size_t n_char_string, const char *p_string,
        size_t n_elem_buf, char *p_match_buf, bool f_overlap)
{
    /* Test for a pattern to match */
    if (n_char_pattern == 0) {
        return 0;
    }

    /* Test for a string of sufficient length */
    if (n_char_pattern > n_char_string) {
        return 0;
    }

    /* Handle 0-sized buffer */
    if (n_elem_buf == 0) {
        return 0;
    }

    /* Factor for rolling hash computation */
    const size_t msb_factor = get_kr_msb_factor(n_char_pattern);

    const size_t h_pattern = kr_hash(n_char_pattern, p_pattern);
    size_t h_string = kr_hash(n_char_pattern, p_string);

    /* Compare at beginning. If hashes match, do full compare */
    if (h_pattern == h_string &&
            memcmp(p_pattern, p_string, n_char_pattern) == 0) {
        return 1; /* match at start */
    }

    /* Compare at each possible starting point in the string */
    const char *p_last = p_string + (n_char_string - n_char_pattern - 1);
    const size_t n_char_pattern_1 = n_char_pattern - 1;
    char **pp_match_buf_cur = &p_match_buf;
    char * const * const pp_match_buf_end = pp_match_buf_cur + n_elem_buf;

    /* Look for overlaps only if possible */
    f_overlap = f_overlap ? !can_overlap(n_char_pattern, p_pattern) : FALSE;

    for ( ; pp_match_buf_cur < pp_match_buf_end; pp_match_buf_cur++) {
        const char *p_match = next_substr(n_char_pattern, p_pattern,
                &p_string, p_last, msb_factor, h_pattern, &h_string);
        if (p_match == (char *) NULL) { /* if no match, done */
            return (int) (pp_match_buf_cur - &p_match_buf);
        }

        /* Save result */
        *pp_match_buf_cur = (char *) p_match;

        /* If overlapping is not allowed, contniue search after the match.
         * Note that in this case, the string hash is the pattern hash. */
        if (!f_overlap) {
            p_string += n_char_pattern_1; /* end of match */
            h_string = h_pattern;
        }
    } /* end of loop over string */

    return n_elem_buf; /* full buffer */
} /* end of funtion get_substring_matches */
#endif /* COMPILE_UNUSED_FUNCTIONS */



/* This function determines if a pattern can allow overlapping matches.
 * For example, the pattern "starts" would have overlapped matches in the
 * string "startstarts".
 *
 * Remarks
 * While not directly related to this function, there is only a binary yes/no
 * interest regarding overlap rather than an offset into the the string where
 * such overlap may occur. That is because the hash value is being computed
 * incremetally, so the only time when there is substantial computational
 * savings in this approach is when the hash value is known, as it would be
 * at the end of a match (since the hash of the pattern is knonw.)
 */
static bool can_overlap(size_t n_char_pattern, const char * const p_pattern)
{
    if (n_char_pattern < 2) { /* does not matter */
        return TRUE;
    }

    /* Find the last occurrance of the first character */
    const char * const p_end = p_pattern + n_char_pattern;
    const char *p_cur = p_end - 1;
    const char ch_first = *p_pattern;
    for ( ; p_cur > p_pattern; --p_cur) {
        if (*p_cur == ch_first) {
            break;
        }
    } /* end of loop finding the first char */

    /* Test for no duplicate */
    if (p_cur == p_pattern) { /* not found */
        return FALSE; /* no duplicate so cannot overlap */
    }

    /* Now must match from this char onward to overlap */
    const char *p_src = p_pattern;
    for ( ; p_cur != p_end; ++p_cur, ++p_src) {
        if (*p_cur != *p_src) { /* comparing 'b' to 'd' in "abcad"
                                 * for example */
            return FALSE; /* Mismatch, so not an overlap */
        }
    } /* end of loop finding the first char */

    return TRUE; /* Matched to end of word */
} /* end of function can_overlap */



/* Prime number of Karp-Rabin hashing. Tradeoff between number of hash
 * collisions and number of times modulus must be taken. */
#define KR_MODULUS 1009
/* Compute (256^(n-1))%KR_MODULUS */
static size_t get_kr_msb_factor(size_t n)
{
    size_t i;
    size_t factor = 1;
    const size_t n_itr = n - 1;
    for (i = 0; i < n_itr; ++i) {
        size_t factor_new = (factor << 8);
        if (factor_new < factor) { /* overflow */
            factor %= KR_MODULUS; /* take modulus */
            factor <<= 8; /* and recompute */
        }
    } /* end of loop building factor */

    /* Return the factor after final modulus if necessary */
    if (factor >= KR_MODULUS) {
        factor %= KR_MODULUS;
    }
    return factor;
} /* end of function get_kr_msb_factor */



/* Compute KR hash assuming n >= 1 */
static size_t kr_hash(size_t n, const char *p)
{
    const char * const p_end = p + n;
    size_t hash = *(unsigned char *) p;
    for (p++; p < p_end; p++) {
        unsigned char ch = *(unsigned char *) p;
        size_t hash_new = (hash << 8) + ch;
        if (hash_new < hash) { /* overflow */
            hash %= KR_MODULUS; /* take modulus */
            hash = (hash << 8) + ch; /* and recompute */
        }
        else { /* no overflow, so no need for modulus yet */
            hash = hash_new;
        }
    } /* end of loop hasing chars */

    /* Do final modulus if necessary */
    if (hash >= KR_MODULUS) {
        hash %= KR_MODULUS;
    }

    return hash;
} /* end of function kr_hash */



/* This function locates the next substring match. It is intended to be called
 * as part of the scanning of a string for a substring
 *
 * Parameters
 * n_char_pattern: Length of pattern to find
 * p_pattern: Pattern to find. Need not be null-terminated
 * pp_string: Address containing the current location in the string. Updated
 *      if a match is found.
 * p_last: Address of last possible location of a match
 * msb_factor: Constant related to hash update
 * h_pattern: Computed hash of pattern
 * p_h_string: Address containing the current hash value of the location
 *      in the string being considered. It is updated in the function.
 *
 * Return value
 * NULL if no substring, or the address of the substring if one exists.
 */
static inline const char *next_substr(
        size_t n_char_pattern, const char *p_pattern,
        const char **pp_string, const char * const p_last,
        const size_t msb_factor, const size_t h_pattern, size_t *p_h_string)
{
    const char *p_string = *pp_string;
    size_t h_string = *p_h_string;

    for ( ; ; ) {
        /* Update hash for next starting point at p_string + 1 */
        if ((h_string = (((h_string - (unsigned char) p_string[0] *
                msb_factor) << 8) + (size_t) p_string[n_char_pattern]) %
                KR_MODULUS) > KR_MODULUS) { /* negative value when signed */
            h_string += KR_MODULUS;
        }
        ++p_string; /* step to next starting point */

        /* Compare at current starting point. If hashes match,
         * do full compare */
        if (h_pattern == h_string &&
                memcmp(p_pattern, p_string, n_char_pattern) == 0) {
            *pp_string = p_string; /* Update string location */
            *p_h_string = h_string; /* and hash for another call */
            return p_string; /* match here */
        }

        /* Exit with no match if at last starting point */
        if (p_string == p_last) {
            return (char *) NULL; /* no match found */
        }
    } /* end of loop over starting points in string */
} /* end of function next_substr */



/* This function returns TRUE if '\0' is among the n characters at p and
 * FALSE otherwise. */
static inline bool have_null(size_t n, const char *p)
{
    /* Scan backwards to make the common case of using a null termination
     * of a string for the null char be faster */
    const char *p_cur = p + n - 1;
    for ( ; p_cur >= p; --p_cur) { /* Locate '\0' among the chars */
        if (*p_cur == '\0') { /* found */
            return TRUE;
        }
    }
    return FALSE;
} /* end of function have_null */



/* This function "finds a needle in a haystack" aka the first occurrence of
 * any character of needle in haystack. NULL is returned if none is found.
 * haystack must be terminated with '\0'.
 *
 * Remarks
 * p_needle does not need to be null terminated. In fact, a null can be
 * included among the characters to be located so that this funtion will
 * locate the end of haystack if none of the other characters is found and
 * would guarantee that the returned value is not NULL.
 *
 * The case of a '\0' included among the chars to locate is treated as a
 * special case for improved efficiency.
 *
 * For a sufficiently large haystack, further gains in performance can be
 * achieved by analyzing the characteristics of the needle values and
 * developing comparisons based on bit values or range values. As a
 * trivial example, for the needle string "01234567", instead of 8
 * comparisons for the 8 values, 2 comparisons can be used by comparing
 * against >= 0 and against <= 7. Without a large enough haystack, the
 * computational time required for the analysis would not be recovered.
 */
char *find_first_of(const char *haystack,
        unsigned int n_needle, const char *p_needle)
{
    /* Hanldle case of nothing to find */
    if (n_needle == 0) {
        return (char *) NULL;
    }

    const char * const p_needle_end = p_needle + n_needle;
    if (have_null(n_needle, p_needle)) { /* searching for '\0' */
        for ( ; ; ++haystack) { /* iterate over straws in haystack */
            const char straw = *haystack;
            const char *p_needle_cur = p_needle;
            for ( ; p_needle_cur != p_needle_end; ++p_needle_cur) {
                const char needle = *p_needle_cur;
                if (straw == needle) { /* found needle */
                    return (char *) haystack;
                }
            } /* end of loop over needles */
        } /* end of loop over straws in haystack */
    } /* end of case that '\0' among items being located */

    /* Else '\0' is not among the items being located */
    for ( ; ; ++haystack) { /* iterate over straws in haystack */
        const char straw = *haystack;
        const char *p_needle_cur = p_needle;
        for ( ; p_needle_cur != p_needle_end; ++p_needle_cur) {
            const char needle = *p_needle_cur;
            if (straw == needle) { /* found needle */
                return (char *) haystack;
            }
        } /* end of loop over needles */
        if (straw == '\0') { /* entire haystack searched */
            return (char *) NULL;
        }
    } /* end of loop over straws in haystack */
} /* end of function find_first_of */



/* This function returns TRUE if the string has any of the characters
 * '"', '\'' or '\\' */
bool has_escape_or_quote(size_t n, const char *str)
{
    const char *str_end = str + n;
    for ( ; str != str_end; ++str) {
        const char ch_cur = *str;
        if (ch_cur == '"' || ch_cur == '\'' || ch_cur == '\\') {
            return TRUE;
        }
    } /* end of loop over chars in string */

    return FALSE;
} /* end of function may_have_eq */

/* Converts integer to string.
   Return the result string.
   Only 10 radix is supported */
char *itoa10(int n, char s[])
{
    int i, j, sign;
    char c;

    if ((sign = n) < 0)  /* record sign */
        n = -n;          /* make n positive */
    i = 0;
    do {       /* generate digits in reverse order */
        s[i++] = n % 10 + '0';   /* get next digit */
    } while ((n /= 10) > 0);     /* delete it */
    if (sign < 0)
        s[i++] = '-';
    s[i] = '\0';
    /* revert string */
    for (i = 0, j = (int)strlen(s) - 1; i < j; i++, j--) {
        c = s[i];
        s[i] = s[j];
        s[j] = c;
    }
    return s;
}

