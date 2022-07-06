/*************
 * Header file for string.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_STRINGUTIL_H
#define ngspice_STRINGUTIL_H

#include <stdarg.h>
#include <string.h>

#include "ngspice/config.h"
#include "ngspice/bool.h"


#ifdef __GNUC__
#define ATTR_TPRINTF    __attribute__ ((format (__printf__, 1, 2)))
#else
#define ATTR_TPRINTF
#endif


/* Structure for storing state to find substring matches in a string */
struct substring_match_info {
    /* Input data */
    size_t n_char_pattern; /* length of pattern being located */
    const char *p_pattern; /* pattern to find */
    size_t n_char_string; /* length of string to search */
    const char *p_string; /* String to search. Final null not required */
    bool f_overlap; /* flag that substring matches can overlap */

    /* Intermediate results */
    size_t n_char_pattern_1; /* length of pattern being located - 1 */
    size_t msb_factor; /* constant related to updating hash */
    size_t h_pattern; /* hash value of pattern */
    size_t h_string; /* current hash value of string */
    const char *p_last; /* last possible substring match location */
    bool f_done; /* flag that last match was found */
};

void appendc(char *s, char c);
int cieq(const char *p, const char *s);
int cieqn(const char *p, const char *s, size_t n);
int ciprefix(const char *p, const char *s);
char *dup_string(const char *str, size_t n_char);
char *find_first_of(const char *haystack,
        unsigned int n_needle, const char *p_needle);
int get_comma_separated_values(char *values[], char *str);
int get_int_n(const char *str, size_t n, int *p_value);
#ifdef COMPILE_UNUSED_FUNCTIONS
size_t get_substring_matches(size_t n_char_pattern, const char *p_pattern,
        size_t n_char_string, const char *p_string,
        size_t n_elem_buf, char *p_match_buf, bool f_overlap);
#endif
char *gettok(char **s);
char *gettok_char(char **s, char p, bool inc_p, bool nested);
char *gettok_instance(char **);
char *gettok_np(char **);
bool has_escape_or_quote(size_t n, const char *str);
bool is_arith_char(char c);
bool isquote(char ch);
int model_name_match(const char *token, const char *model_name);
int prefix(const char *p, const char *s);
int prefix_n(size_t n_char_prefix, const char *prefix,
        size_t n_char_string, const char *string);
int scannum_adv(char **p_str);
bool str_has_arith_char(char *s);
char *stripWhiteSpacesInsideParens(const char *str);
void strtolower(char *str);
void strtoupper(char *str);
void substring_match_init(size_t n_char_pattern, const char *p_pattern,
        size_t n_char_string, const char *p_string, bool f_overlap,
        struct substring_match_info *p_scan_state);
char *substring_match_next(struct substring_match_info *p_scan_state);
int substring_n(size_t n_char_pattern, const char *p_pattern,
        size_t n_char_str, const char *p_str);
char *tprintf(const char *fmt, ...) ATTR_TPRINTF;
char *tvprintf(const char *fmt, va_list args);
char* itoa10(int val, char* buf);


/* Allocate and create a copy of a string if the argument is not null or
 * returns null if it is. */
inline char *copy(const char *str)
{
    return str == (char *) NULL ?
            (char *) NULL : dup_string(str, strlen(str));
} /* end of function copy */



/* Allocate a buffer and copy a substring, from 'str' to 'end'
 *   including *str, excluding *end
 */
inline char *copy_substring(const char *str, const char *end)
{
    return dup_string(str, (size_t) (end - str));
} /* end of function copy_substring */



/* Try to identify an unsigned integer that begins a string. Stop when a
 * non- numeric character is reached. There is no way to distinguish
 * between a value of 0 and a string that does not contain a numeric
 * value. */
inline int scannum(const char *str)
{
    return scannum_adv((char **) &str);
} /* end of function scannum */



/* Determine whether sub is a substring of str. */
inline int substring(const char *sub, const char *str)
{
    return strstr(str, sub) != (char *) NULL;
} /* end of function substring */

#ifdef CIDER
/* cider integration */ 
int cinprefix(char *p, char *s, int n);
int cimatch(char *p, char *s); 
#endif


#endif /* include guard */
