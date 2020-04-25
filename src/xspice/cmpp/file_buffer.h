#ifndef file_buffer_h_included
#define file_buffer_h_included

#include <stdbool.h>


/* Null-terminated string prefixed by length excluding null */
typedef struct Filebuf_len_str {
    size_t n_char; /* length of string excluding null termination */
    char *sz; /* Start of string */
} FBSTRING;

/* Union for returned value */
typedef union Filebuf_obj {
    FBSTRING str_value;
    unsigned long ulong_value;
    long long_value;
    double dbl_value;
} FBOBJ;

/* Structure for getting file data */
typedef struct Filebuf {
    FILE *fp; /* handle to file */
    bool is_eof; /* flag that EOF reached */
    bool f_skip_to_eol;
                /* Flag that text until the next EOL character should be
                 * skipped before getting the next item from the buffer.
                 * This flag is set when a comment terminates an item,
                 * such as "abc# This is a comment." */
    size_t n_byte_buf_alloc; /* Allocated buffer size */
    char *p_buf; /* buffer to receive data from file */
    char *p_obj_start; /* start of object being returned */
    char *p_obj_end; /* byte past object being returned */
    char *p_data_cur; /* current position in buffer. Depending on
                       * circumstances, it points to either the character
                       * being processed or the next character to process */
    char *p_data_end; /* byte past end of data in buffer */
    char *p_buf_end; /* byte past end of allocated buffer size, equal to
                      * p_buf + n_byte_buf_alloc, so it is redundant, but
                      * convenient to have available */
} FILEBUF;

/* Types of data */
typedef enum FBtype {
    BUF_TYPE_STRING, /* value type string (always possible) */
    BUF_TYPE_ULONG, /* value type an unsigned int */
    BUF_TYPE_LONG, /* value type an int */
    BUF_TYPE_DOUBLE /* value type double */
} FBTYPE;


FILEBUF *fbopen(const char *filename, size_t n_byte_buf_init);
int fbget(FILEBUF *p_fb, unsigned int n_type_wanted, FBTYPE *p_type_wanted,
        FBTYPE *p_type_found, FBOBJ *p_fbobj);
int fbclose(FILEBUF *fbp);


#endif /* include guard */
