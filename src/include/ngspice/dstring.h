/*   dstring.h    */

#ifndef DSTRING_H
#define DSTRING_H

/* -----------------------------------------------------------------
 * This structure is modified from Tcl.   We do this to avoid a
 * conflict and later add a conditional compile to just use the Tcl
 * code if desired.
----------------------------------------------------------------- */
#define SPICE_DSTRING_STATIC_SIZE 200
typedef struct spice_dstring {
  char *string ;               /* Points to beginning of string:  either
			        * staticSpace below or a malloced array. */
  int length ;                 /* Number of non-NULL characters in the
				* string. */
  int spaceAvl ;               /* Total number of bytes available for the
			        * string and its terminating NULL char. */
  char staticSpace[SPICE_DSTRING_STATIC_SIZE] ;
			       /* Space to use in common case where string
				* is small. */
} SPICE_DSTRING, *SPICE_DSTRINGPTR ;

/* -----------------------------------------------------------------
 * spice_dstring_xxxx routines.  Used to manipulate dynamic strings.
----------------------------------------------------------------- */
extern void spice_dstring_init(SPICE_DSTRINGPTR dsPtr) ;
extern char *spice_dstring_append(SPICE_DSTRINGPTR dsPtr,char *string,int length) ;
extern char *spice_dstring_append_char(SPICE_DSTRINGPTR dsPtr,char c) ;
extern char *spice_dstring_print(SPICE_DSTRINGPTR dsPtr,char *format, ... ) ;
extern char *spice_dstring_setlength(SPICE_DSTRINGPTR dsPtr,int length) ;
extern char *_spice_dstring_setlength(SPICE_DSTRINGPTR dsPtr,int length) ;
extern void spice_dstring_free(SPICE_DSTRINGPTR dsPtr) ;
#define spice_dstring_reinit(x_xz) spice_dstring_setlength(x_xz,0) ;
#define spice_dstring_value(x_xz) ((x_xz)->string)
#define spice_dstring_space(x_xz) ((x_xz)->spaceAvl)
#define spice_dstring_length(x_xz) ((x_xz)->length)

#endif /* DSTRING_H */
