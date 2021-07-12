/* ----------------------------------------------------------------- 
FILE:	    nghash.h                                       
DESCRIPTION:Insert file for threaded hash routines.
            This code was donated from TimberWolf.
CONTENTS:   
DATE:	    Jul 17, 1988 - original coding
REVISIONS:  Aug 21, 2009 - adapted for ngspice
----------------------------------------------------------------- */
#ifndef ngspice_HASH_H
#define ngspice_HASH_H

#include "ngspice/bool.h"
#include <stdint.h>

#define _NGMALLOC(size_xz)            tmalloc((size_xz))
#define NGMALLOC(n, els)              TMALLOC(els, n)
#define NGREALLOC(ar,n,els)           TREALLOC(els, ar, n)
#define NGFREE(els)                   txfree(els)



/* ********************** TYPE DEFINITIONS ************************* */
typedef void (*ngdelete)(void *) ;



typedef struct ngtable_rec {
    void *key ;
    void *data ;
    struct ngtable_rec *next ;		/* collision list */
    struct ngtable_rec *thread_next ;    /* thread thru entire table */
    struct ngtable_rec *thread_prev ;    /* thread thru entire table */
} NGTABLEBOX, *NGTABLEPTR ;

struct nghashbox;

typedef unsigned int nghash_func_t(struct nghashbox *, void *) ;
typedef int nghash_compare_func_t(const void *, const void *) ;

struct nghashbox {
    NGTABLEPTR *hash_table ;
    NGTABLEPTR thread ;		/* thread of hash table */
    NGTABLEPTR last_entry ;	/* last entry into hash table */
    NGTABLEPTR enumeratePtr ;	/* used to enumerate hash table */
    NGTABLEPTR searchPtr ;	/* used for find again mechanism */
    nghash_compare_func_t *compare_func ;	/* the comparison function */
    nghash_func_t *hash_func ;	/* the hash function */
    double growth_factor ;	/* how much to grow table by */
    int size ;			/* the size of the table */
    int max_density ;		/* maximum number of entries before growth */
    int num_entries ;		/* current number of entries in table */
    int need_resize ;		/* amount before we need a resize */
    long access ;		/* used for statistics */
    long collision ; 		/* collision times */
    unsigned int  power_of_two : 8 ; /* build table as a power of two */
    unsigned int  call_from_free : 8 ;/* true if in a free calling sequence */
    unsigned int  unique : 16 ;	/* true if only one unique item in col. list */
};

typedef struct nghashbox NGHASHBOX, *NGHASHPTR;

/* -----------------------------------------------------------------
 * This enumerated type is used to control the base hash function types
 * as well as hash table attributes such as uniqueness and power of 2.
 * The negative definitions for the function are used so we catch a
 * segfault immediately if the hash table is used incorrectly.  Notice
 * that the default function string is zero.  Because there definitions
 * can be cast into functions, they are defined as longs so it will work
 * on all architecture types include 64bits.
 * ----------------------------------------------------------------- */
typedef enum {
  NGHASH_UNIQUE 	=  1 << 0,
  NGHASH_POWER_OF_TWO 	=  1 << 1,
  NGHASH_UNIQUE_TWO 	= NGHASH_UNIQUE | NGHASH_POWER_OF_TWO
} NGHASHFLAGS_T ;

enum nghash_default_func_t {
  NGHASH_FUNC_NUM	= -2,
  NGHASH_FUNC_PTR	= -1,
  NGHASH_FUNC_STR 	=  0
};


typedef struct nghash_iter_rec {
  struct ngtable_rec *position ;
} NGHASHITER, *NGHASHITERPTR ;

/* -----------------------------------------------------------------
 * macro definition for enumeration.  Strange looking but compiler
 * will optimize.  x_yz->position = 0 and x_yz will be returned.
#define NGHASH_FIRST(x_yz)		( ((x_yz)->position = NULL) ? (x_yz) : (x_yz) )
No longer use this trick and now explicitly write out both lines because
some compilers complain about unintentional assignment.  Unfortunately,
we want to intentionally assign it.  The compiler is warning unnecessarily.
----------------------------------------------------------------- */
#define NGHASH_FIRST(x_yz)		( (x_yz)->position = NULL ) ;
#define NGHASH_ITER_EQUAL(x_yz,y_yz)	( (x_yz)->position == (y_yz)->position )

#define NGHASH_DEF_HASH(x)	((nghash_func_t *) (intptr_t) (x))
#define NGHASH_DEF_CMP(x)	((nghash_compare_func_t *) (intptr_t) (x))

/* defines for unique flag */

/* misc definitions */
#define NGHASH_DEF_GROW_FACTOR	2.0
#define NGHASH_DEF_MAX_DENSITY	4
#define NGHASH_MIN_SIZE		4
#define nghash_tablesize( htable_xz )	( (htable_xz)->size )

/* -----------------------------------------------------------------
 * Here are three hash functions in order of increasingly better
 * behaviour. Remember hsum must be unsigned int.  The best one
 * is taken from tcl.
----------------------------------------------------------------- */

#define NGHASH_STR_TO_HASH( str, hsum, size ) \
    do { 	\
	char *name ; \
	int shift ;	\
	for( name=str,shift=1,hsum=0 ;*name; name++){  \
	    hsum = hsum + ((*name)<<shift);                \
	    shift = (shift + 1) & 0x0007;              \
	}  \
	hsum %= (size) ; \
    } while(0);

#undef NGHASH_STR_TO_HASH
#define NGHASH_STR_TO_HASH( str, hsum, size ) \
    do { 	\
      char *name ; \
      unsigned long g ; \
      for( name=str,hsum=0 ;*name; name++){  \
	hsum = (hsum << 4) + (*name); \
	g = hsum & 0xF0000000 ; \
	if (g) { \
	  hsum = hsum^(g >> 24); \
	  hsum = hsum^g; \
	} \
      } \
      hsum = hsum & (size-1); \
    } while(0);

#undef NGHASH_STR_TO_HASH
#define NGHASH_STR_TO_HASH( str, hsum, size ) \
    do { 	\
	int c ; \
	char *string ; \
	hsum = 0 ; \
	string = (char *) str ; \
	for (;;) { \
	  c = *string ; \
	  string++ ; \
	  if( c == 0) { \
	    break ; \
	  } \
	  hsum += (hsum<<3) + (unsigned int) c;      \
	}  \
	hsum %= (unsigned int) (size) ;              \
    } while(0);

#define NGHASH_NUM_TO_HASH( num, hsum, size ) \
    do { 	\
	int c, len ; \
	unsigned long temp; \
	char cptr[80] ; \
	sprintf( cptr, "%lx", (UNSIGNED_LONG) num) ; \
	len = strlen(cptr) ; \
	temp = (unsigned long) cptr[0] ; \
	for( c = 1 ; c < len ; c++ ){ \
	  temp += (temp<<3) + (unsigned long) cptr[c] ; \
	} \
	temp %= (size) ; \
	hsum = temp ; \
    } while(0);

#undef NGHASH_NUM_TO_HASH

/* -----------------------------------------------------------------
 * Replace the old num to hash with a new algorithm which is simple
 * and is 5 times faster.  Variance was even less on generated data.
 * To use this hash table power of 2 table size must be enforced.
----------------------------------------------------------------- */
#define NGHASH_NUM_TO_HASH( ptr, hsum, size ) \
    do { 	\
      unsigned int temp ; \
      intptr_t value = (intptr_t) ptr ; \
      temp = (unsigned int) value ;              \
      hsum = temp & (unsigned int) (size - 1) ;   \
    } while(0);


#define NGHASH_PTR_TO_HASH( ptr, hsum, size ) \
    do { 	\
	unsigned long temp; \
	temp = (unsigned long) (ptr); \
	temp %= (size) ; \
	hsum = temp ; \
    } while(0);

#undef NGHASH_PTR_TO_HASH
/* -----------------------------------------------------------------
 * We use strlen instead of looking at the return value of sprintf
 * because it is not standard. Actually no machine have 80 byte pointers
 * but just being cautious.
----------------------------------------------------------------- */

#define NGHASH_PTR_TO_HASH( ptr, hsum, size ) \
    do { 	\
	int c, len ; \
	unsigned long temp; \
	char cptr[80] ; \
	sprintf( cptr, "%lx", (UNSIGNED_LONG) ptr) ; \
	len = strlen(cptr) ; \
	temp = (UNSIGNED_LONG) cptr[0] ; \
	for( c = 1 ; c < len ; c++ ){ \
	  temp += (temp<<3) + (UNSIGNED_LONG) cptr[c] ; \
	} \
	temp %= (size) ; \
	hsum = temp ; \
    } while(0);

#undef NGHASH_PTR_TO_HASH
/* -----------------------------------------------------------------
 * Replace the old ptr to hash with a new algorithm which is simple
 * and is 5 times faster.  Variance was even less on generated data.
 * To use this hash table power of 2 table size must be enforced.
----------------------------------------------------------------- */
#define NGHASH_PTR_TO_HASH( ptr, hsum, size ) \
    do { 	\
      unsigned int temp ; \
      intptr_t value = (intptr_t) ptr ; \
      temp = (unsigned int) (value >> 4) ;       \
      hsum = temp & (unsigned int) (size - 1) ;      \
    } while(0);

#define NGHASH_PTR_COMPARE_FUNC( p1 , p2 ) ( (p1) != (p2) )


#define nghash_unique( htable_xz, flag_xz )	((htable_xz)->unique = flag_xz)

extern NGHASHPTR nghash_init( int numentries ) ;
/*
Function:
    Returns a hash table with the given number of entries.
    More that one hash table can coexist at the same time.
    The default key type is string. The default comparison function
    is strcmp.
*/

extern NGHASHPTR nghash_init_integer( int numentries ) ;
/*
Function:
    Returns a hash table with the given number of entries.
    More that one hash table can coexist at the same time.
    The default key type is an integer. The default comparison function
    is integer comparison.
*/

extern NGHASHPTR nghash_init_pointer( int numentries ) ;
/*
Function:
    Returns a hash table with the given number of entries.
    More that one hash table can coexist at the same time.
    The default key type is a pointer. The default comparison function
    is pointer comparison.
*/

extern NGHASHPTR nghash_init_with_parms( nghash_compare_func_t *comp_func, 
    nghash_func_t *hash_func, int numentries, int max_density, 
    double growth, NGHASHFLAGS_T flags ) ;
/*
Function:
    Returns a hash table with the given number of entries.
    More that one hash table can coexist at the same time.
    Tables may be given their own hash and compare functions.  If your keys
    are pointers, numbers or strings, it is recommended that you use the
    functions:
	* HASH_DEF_HASH_PTR and HASH_DEF_CMP_PTR for pointers,
	* HASH_DEF_HASH_NUM and HASH_DEF_CMP_NUM for numbers, and
	* HASH_DEF_HASH_STR and HASH_DEF_CMP_STR for strings.
    The hash package will recognize these and run faster as a result.

    You may use your own hash and compare functions provided they look like
    * INT hash(void * key) and
    * UNSIGNED_INT compare(void * key1, void * key2). 
    The hash function's return value should be in the interval [0, UINT_MAX].
    The compare should return zero if the two keys are equal and a non-zero
    value otherwise.

    Whenever
	number of entries in hash table >= size of table * max_density,
    the table is grown at a the specified by growth.  Unique if TRUE only allows
    one entry which matches comparison function.  Otherwise, multiple items
    which are not unique relative to the comparison function can exist in
    collision list.
*/

extern int nghash_max_density(NGHASHPTR hashtable,int max_density) ;
/*
Function:
    Changes the max_density limit in the hash table if max_density > 1.  
    This function returns the current value of max_density. 
*/


extern int nghash_table_get( NGHASHPTR  hashtable ) ;
/*
Function:
    Returns the current size of hash table set by nghash_table_create 
*/

extern int nghash_table_size( int num ) ;
/*
Function:
    Returns the closest prime number to given size.
*/

extern int nghash_table_size2( int num ) ;
/*
Function:
    Returns the table size to the closest power of 2.
*/

extern void *_nghash_find(NGHASHPTR hashtable, void * user_key,BOOL *status) ;
extern void *nghash_find(NGHASHPTR hashtable, void * user_key) ;
extern void *nghash_find_again(NGHASHPTR hashtable, void * user_key) ;
extern void *_nghash_find_again(NGHASHPTR hashtable, void * user_key,BOOL *status) ;
extern void *nghash_delete(NGHASHPTR hashtable, void * user_key) ;
extern void *nghash_delete_special(NGHASHPTR hashtable, void* user_key);
extern void *nghash_insert(NGHASHPTR hashtable, void * user_key, void * data) ;
/*
The four functions above replace the old nghash_search function.  The same
functionality but now four functions.
Function:
    Hash table search routine.  Given a hashtable and a key, perform
    the following operations:
	HASH_ENTER:if key is in table it, returns a pointer to the item.
	      if key is not in table, add it to the table. returns NULL.
	HASH_FIND:if key is in table, it returns data pointer.
	      if key is not in the table, it returns NULL.
	HASH_FIND_AGAIN:if additional keys are in table, returns data pointer.
	      if no more keys are in the table, it returns NULL.
	HASH_DELETE:if key is in table, it returns -1
	       if key is not in table, it return NULL.
	Memory is not freed in the delete case, but marked dirty.
    Data is a pointer to the information to be store with the given key.
    A new operation is available for using pointers are arguments.
    Just pass the pointer in as a key.  Make sure to call the
    nghash_init_pointer function first.
*/

/* -----------------------------------------------------------------
 * Convenience functions.
----------------------------------------------------------------- */
typedef void * (*nghash)(void *) ;

extern void nghash_free(NGHASHPTR htabl,void (*del_data)(void *),void (*del_key)(void *) );
/*
Function:
    Frees the memory associated with a hash table. The user make supply a
    function which deletes the memory associated with the data field. 
    In addition, the user may free the memory stored at the key.
    This function must have the data pointer supplied by the hash add routines
    as an argument,ie.
    nghash_table_delete( my_hash_table, my_free_func, my_key_free ) ;
    my_free_func( data )
    void * data ;
    {
    }
    my_key_free( key )
    void * key ;
    {
    }
    Note: for the default hash types: STR, PTR, and NUM, the free key
    operation is ignored.
*/

extern void nghash_free_string_func(char *str) ;
/*
Function:
    Just a wrapper for YFREE(string)
*/

extern void nghash_free_string_hashtable(NGHASHPTR htable) ;
/*
Function:
    Frees the memory associated with a hash table. This version is
    a convenience function as it is equivalent to the function:
    nghash_free( htable, (ngdelete) nghash_free_string_func, NULL ) ;
*/

extern void nghash_empty(NGHASHPTR htabl,void (*del_data)(void *),void (*del_key)(void *) ) ;
/*
Function:
    Similar to nghash_free except table structure is not delete. However,
    all entries have been deleted.
*/

extern NGHASHPTR nghash_merge( NGHASHPTR master_htable, NGHASHPTR merge_htable ) ;
/*
Function:
    Merge items in merge_htable into master_htable.  Create master_htable
    if master_htable is NULL.  Returns the merged hash table.
*/

extern int nghash_get_size( NGHASHPTR  hashtable ) ;
/*
Function:
    Since this is a threaded hash table we can find the size of
    all the valid entries in the table in O(n) where n is the
    number of valid entries. Returns the number of valid entries.
    One may enumerate the hash table by writing the following loop.
    TABLEPTR ptr ;
    for( ptr = mytable->thread; ptr ; ptr= ptr->threadNext ){
	...
    }
*/

extern void *nghash_enumeratek(NGHASHPTR hashtable,void **key_ret,BOOL flag) ;
/*
Function:
    Since this is a threaded hash table, we can enumerate the elements of
    the hash table in O(n) time where n is the number of valid entries. 
    Returns the data and key associated with each entry. The flag is similar
    to the rbtree enumerate function.  This eliminates the need for writing
    the following:
	TABLEPTR ptr ;
	for( ptr = mytable->thread; ptr ; ptr= ptr->threadNext ){
	    ...
	}
    Instead write:
    for( data = nghash_enumerate(hashtable,&key,TRUE) ; data ;
	data = nghash_enumerate(hashtable,&key,TRUE) ){
	    ...
    }
    The order returned is guaranteed to be the same as the order entered.
*/

extern void *nghash_enumeratekRE(NGHASHPTR hashtable,void **key_ret,NGHASHITERPTR iter_p) ;
/*
Function:
    Same as nghash_enumeratekRE but this is a reentrant version.   Use as
    void * key_ret ;
    for( i_p= nghash_enumeratekRE(htable_p,&key_ret,YHASH_FIRST(&iter) ) ; i_p ;
         i_p= nghash_enumeratekRE(htable_p,&key_ret,&iter) ) ){
      data_p = (data_cast) key_ret ;
   }
*/

extern void *nghash_enumerate(NGHASHPTR hashtable,BOOL flag) ;
/*
Function:
    Like above but we don't return the key.
*/

extern void *nghash_enumerateRE(NGHASHPTR hashtable,NGHASHITERPTR iter_p) ;
/*
Function:
    Like nghash_enumerate but this is a reentrant version.
    for( i_p= nghash_enumerateRE(htable_p,YHASH_FIRST(&iter) ) ; i_p ;
         i_p= nghash_enumerateRE(htable_p,&iter) ) ){
      data_p = (data_cast) key_ret ;
   }

*/

extern BOOL nghash_deleteItem( NGHASHPTR  hashtable, void *key, void *data ) ;
/*
Function:
    Delete a specific item in the hash table.  Returns true if the item was
    found and deleted.
*/

/*
 * This function has been removed because we now support reentrant versions.
extern VOID nghash_enumeratePush( P1(YHASHPTR hashtable));
Function:
    Push the current enumeration pointer onto a stack. This is useful
    for recursive enumeration.
*/

/*
 * This function has been removed because we now support reentrant versions.
extern VOID nghash_enumeratePop( P1(YHASHPTR hashtable));
Function:
    Pops the current enumeration pointer from the stack. This is useful
    for recursive enumeration.
*/

extern void nghash_dump( NGHASHPTR hashtable,void (*print_func)(void *) ) ;
/*
Function:
    Prints the contents of the hash table.
*/

extern void nghash_reset_stat( NGHASHPTR hashtable ) ;
extern void nghash_resize( NGHASHPTR hashtable, int num ) ;

extern void nghash_distribution( NGHASHPTR hashtable ) ;
/*
Function:
    Returns information about hash table distribution to message system.
*/


#endif
