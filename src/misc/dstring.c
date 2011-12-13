/* -----------------------------------------------------------------
FILE:	    dstring.c
DESCRIPTION:This file contains the routines for manipulating dynamic strings.
----------------------------------------------------------------- */
#include "ngspice/ngspice.h"
#include <stdarg.h>
#include "ngspice/dstring.h"

/* definitions local to this file only */

/* ********************** TYPE DEFINITIONS ************************* */

/* ********************** STATIC DEFINITIONS ************************* */
/*
 *----------------------------------------------------------------------
 *
 * spice_dstring_init --
 *
 *	Initializes a dynamic string, discarding any previous contents
 *	of the string (spice_dstring_free should have been called already
 *	if the dynamic string was previously in use).
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	The dynamic string is initialized to be empty.
 *
 *----------------------------------------------------------------------
 */

void spice_dstring_init(SPICE_DSTRINGPTR dsPtr)
{
    dsPtr->string = dsPtr->staticSpace ;
    dsPtr->length = 0 ;
    dsPtr->spaceAvl = SPICE_DSTRING_STATIC_SIZE ;
    dsPtr->staticSpace[0] = '\0';
} /* end spice_dstring_init() */

/*
 *----------------------------------------------------------------------
 *
 * spice_dstring_append --
 *
 *	Append more characters to the current value of a dynamic string.
 *
 * Results:
 *	The return value is a pointer to the dynamic string's new value.
 *
 * Side effects:
 *	Length bytes from string (or all of string if length is less
 *	than zero) are added to the current value of the string. Memory
 *	gets reallocated if needed to accomodate the string's new size.
 *  
 * Notes: char *string;    String to append.  If length is -1 then
 *                         this must be null-terminated.
 *        INT length;	   Number of characters from string to append.  
 *                         If < 0, then append all of string, up to null at end.
 *
 *----------------------------------------------------------------------
 */
char *spice_dstring_append(SPICE_DSTRINGPTR dsPtr,char *string,int length)
{
    int newSize ;			/* needed size */
    char *newString ;			/* newly allocated string buffer */
    char *dst ;				/* destination */
    char *end ;				/* end of string */

    if( length < 0){
	length = (int) strlen(string) ;
    }
    newSize = length + dsPtr->length ;

    /* -----------------------------------------------------------------
     * Allocate a larger buffer for the string if the current one isn't
     * large enough. Allocate extra space in the new buffer so that there
     * will be room to grow before we have to allocate again.
     ----------------------------------------------------------------- */
    if (newSize >= dsPtr->spaceAvl) {
	dsPtr->spaceAvl = 2 * newSize ;
	newString = TMALLOC(char, dsPtr->spaceAvl) ;
	memcpy(newString, dsPtr->string, (size_t) dsPtr->length) ;
	if (dsPtr->string != dsPtr->staticSpace) {
	    txfree(dsPtr->string) ;
	}
	dsPtr->string = newString;
    }

    /* -----------------------------------------------------------------
     * Copy the new string into the buffer at the end of the old
     * one.
     ----------------------------------------------------------------- */
    for( dst = dsPtr->string + dsPtr->length, end = string+length;
	    string < end; string++, dst++) {
	*dst = *string ;
    }
    *dst = '\0' ;
    dsPtr->length += length ;

    return(dsPtr->string) ;

} /* end spice_dstring_append() */

/* -----------------------------------------------------------------
 * Function: add character c to dynamic string dstr_p.
 * ----------------------------------------------------------------- */
char *spice_dstring_append_char( SPICE_DSTRINGPTR dstr_p, char c)
{
  char tmp_str[2] ;			/* temporary string for append */
  char *val_p ;				/* return value */

  tmp_str[0] = c ;
  tmp_str[1] = '\0' ;
  val_p = spice_dstring_append( dstr_p, tmp_str, -1 ) ;
  return(val_p) ;
} /* end spice_dstring_append_char() */

static int spice_format_length( va_list args, char *fmt )
{
    int i ;					/* integer */
    int len ;					/* length of format */
    int size_format ;				/* width of field */
    int found_special ;				/* look for special characters */
    char *s ;					/* string */
    double d ;

    /* -----------------------------------------------------------------
     * First find length of buffer.
    ----------------------------------------------------------------- */
    len = 0 ;
    while(fmt && *fmt){
      if( *fmt == '%' ){
	fmt++ ;
	if( *fmt == '%' ){
	  len++ ;
	} else {
	  /* -----------------------------------------------------------------
	   * We have a real formatting character, loop until we get a special
	   * character.
	  ----------------------------------------------------------------- */
	  if( *fmt == '.' || *fmt == '-' ){
	    fmt++ ; /* skip over these characters */
	  }
	  size_format = atoi(fmt) ;
	  if( size_format > 0 ){
	    len += size_format ;
	  }
	  found_special = FALSE ;
	  for( ; fmt && *fmt ; fmt++ ){
	    switch( *fmt ){
	      case 's':
		s = va_arg(args, char *) ;
		if( s ){
		  len += (int) strlen(s) ;
		}
		found_special = TRUE ;
		break ;
	      case 'i':
	      case 'd':
	      case 'o':
	      case 'x':
	      case 'X':
	      case 'u':
		i = va_arg(args, int) ;
		len += 10 ;
		found_special = TRUE ;
		break ;
	      case 'c':
		i = va_arg(args, int) ;
		len++ ;
		found_special = TRUE ;
		break ;
	      case 'f':
	      case 'e':
	      case 'F':
	      case 'g':
	      case 'G':
		d = va_arg(args, double) ;
		len += 35 ;
		found_special = TRUE ;
		break ;
	      default:
		;
	    } /* end switch() */

	    if( found_special ){
	      break ;
	    }
	  }
	}
      } else {
	len++ ;
      }
      fmt++ ;
    } /* end while() */
    va_end(args) ;

    return(len) ;

} /* end Ymessage_format_length() */


char *spice_dstring_print( SPICE_DSTRINGPTR dsPtr,char *format, ... )
{
    va_list args ;
    int format_len ;				/* length of format */
    int length ;				/* new length */
    int orig_length ;				/* original length of buffer */
    char *buffer ;				/* proper length of buffer */

    /* -----------------------------------------------------------------
     * First get the length of the buffer needed.
    ----------------------------------------------------------------- */
    va_start( args, format ) ;
    format_len = spice_format_length(args,format) ;

    /* -----------------------------------------------------------------
     * Next allocate the proper buffer size.
    ----------------------------------------------------------------- */
    orig_length = dsPtr->length ;
    length = orig_length + format_len + 1 ;
    buffer = spice_dstring_setlength( dsPtr, length) ;

    /* -----------------------------------------------------------------
     * Convert the format.
    ----------------------------------------------------------------- */
    va_start( args, format ) ;
    if( format ){
      vsprintf( buffer + orig_length, format, args ) ;
      dsPtr->length = (int) strlen(buffer) ;
    } else {
      buffer = NULL ;
    }
    va_end(args) ;
    return( buffer ) ;

} /* end spice_dstring_print() */

/*
 *----------------------------------------------------------------------
 *
 * _spice_dstring_setlength --
 *
 *	Change the length of a dynamic string.  This can cause the
 *	string to either grow or shrink, depending on the value of
 *	length.
 *
 * Results:
 *	Returns the current string buffer.
 *
 * Side effects:
 *	The length of dsPtr is changed to length but a null byte is not
 *	stored at that position in the string.  Use spice_dstring_setlength
 *	for that function.   If length is larger
 *	than the space allocated for dsPtr, then a panic occurs.
 *
 *----------------------------------------------------------------------
 */

char *_spice_dstring_setlength(SPICE_DSTRINGPTR dsPtr,int length)
{
    char *newString ;

    if (length < 0) {
	length = 0 ;
    }
    if (length >= dsPtr->spaceAvl) {

	dsPtr->spaceAvl = length+1;
	newString = TMALLOC(char, dsPtr->spaceAvl) ;
	/* -----------------------------------------------------------------
	 * SPECIAL NOTE: must use memcpy, not strcpy, to copy the string
	 * to a larger buffer, since there may be embedded NULLs in the
	 * string in some cases.
	----------------------------------------------------------------- */
	memcpy(newString, dsPtr->string, (size_t) dsPtr->length) ;
	if( dsPtr->string != dsPtr->staticSpace ) {
	    txfree(dsPtr->string) ;
	}
	dsPtr->string = newString ;
    }
    dsPtr->length = length ;
    return(dsPtr->string) ;

} /* end _spice_dstring_setlength() */


/*
 *----------------------------------------------------------------------
 *
 * spice_dstring_setlength --
 *
 *	Change the length of a dynamic string.  This can cause the
 *	string to either grow or shrink, depending on the value of
 *	length.
 *
 * Results:
 *	Returns the current string buffer.
 *
 * Side effects:
 *	The length of dsPtr is changed to length and a null byte is
 *	stored at that position in the string.  If length is larger
 *	than the space allocated for dsPtr, then a panic occurs.
 *
 *----------------------------------------------------------------------
 */

char *spice_dstring_setlength(SPICE_DSTRINGPTR dsPtr,int length)
{
    char *str_p ;			/* newly create string */

    str_p = _spice_dstring_setlength( dsPtr,length) ;
    str_p[length] = '\0' ;
    return( str_p ) ;

} /* end spice_dstring_setlength() */

/*
 *----------------------------------------------------------------------
 *
 * spice_dstring_free --
 *
 *	Frees up any memory allocated for the dynamic string and
 *	reinitializes the string to an empty state.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	The previous contents of the dynamic string are lost, and
 *	the new value is an empty string.
 *
 *----------------------------------------------------------------------
 */

void spice_dstring_free(SPICE_DSTRINGPTR dsPtr)
{
    if (dsPtr->string != dsPtr->staticSpace) {
	txfree(dsPtr->string) ;
    }
    dsPtr->string = dsPtr->staticSpace ;
    dsPtr->length = 0 ;
    dsPtr->spaceAvl = SPICE_DSTRING_STATIC_SIZE;
    dsPtr->staticSpace[0] = '\0' ;

} /* end spice_dstring_free() */
