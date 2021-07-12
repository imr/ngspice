/* -----------------------------------------------------------------
FILE:	    hash.c
DESCRIPTION:This file contains the routines for building and
        maintaining a hash table.
        Abstract : Contains routines for managing hash tables.  Tables
        may be given their own hash and compare functions.  If your keys
        are pointers, numbers or strings, it is recommended that you
        use the functions
        * HASH_DEF_HASH_PTR and HASH_DEF_CMP_PTR for pointers,
        * HASH_DEF_HASH_NUM and HASH_DEF_CMP_NUM for numbers, and
        * HASH_DEF_HASH_STR and HASH_DEF_CMP_STR for strings.
        The hash package will recognize these and run faster as
        a result.

        You may use your own hash and compare functions provided they
        look like
        * int hash(void * key) and
        * int compare(void * key1, void * key2).
        The hash function's return value should be in the interval
        [0, Uint_MAX].  The compare should return zero if the two
        keys are equal and a non-zero value otherwise.
CONTENTS:
DATE:	    Jul  7, 1988 - original coding.
            1988 - 2009 many updates...
REVISIONS:
----------------------------------------------------------------- */
#include "ngspice/ngspice.h"
#include "ngspice/hash.h"

/* definitions local to this file only */

/* ********************** TYPE DEFINITIONS ************************* */
#define  PRIMECOUNT   200
#define  MINPRIMESIZE 7

/* ********************** STATIC DEFINITIONS ************************* */
static NGTABLEPTR _nghash_find_item(NGHASHPTR hhtable,void *user_key,void *data) ;


NGHASHPTR nghash_init_with_parms(nghash_compare_func_t *comp_func, nghash_func_t *hash_func, int num,
                                 int max, double growth, NGHASHFLAGS_T flags)
{
    BOOL unique ;			/* entries are to be unique */
    BOOL power_of_two ;			/* want hash table power of 2 */
    NGHASHPTR  hashtable ;

    unique = flags & NGHASH_UNIQUE ;
    power_of_two = flags & NGHASH_POWER_OF_TWO ;

    hashtable = NGMALLOC( 1, NGHASHBOX ) ;
    if( power_of_two ) {
        hashtable->size = nghash_table_size2( num ) ;
    } else {
        /* prime size */
        hashtable->size = nghash_table_size( num ) ;
    }
    hashtable->compare_func = comp_func ;
    hashtable->hash_func = hash_func ;

    hashtable->hash_table = NGMALLOC( hashtable->size, NGTABLEPTR ) ;
    hashtable->max_density = max ;
    hashtable->need_resize = hashtable->size * hashtable->max_density ;
    hashtable->growth_factor = growth ;
    hashtable->unique = (unique ? 1 : 0);
    hashtable->power_of_two = (power_of_two ? 1 : 0);
    hashtable->thread = NULL ; /* initialize list */
    hashtable->last_entry = NULL ; /* end of list */
    hashtable->num_entries = 0 ;
    hashtable->call_from_free = FALSE ;
    hashtable->access = 0;
    hashtable->collision = 0;
    hashtable->enumeratePtr = NULL ;
    return(hashtable) ;
} /* end nghash_init_with_parms() */


void nghash_resize(NGHASHPTR hashtable, int num)
{
    NGTABLEPTR *oldtable, hptr, zapptr ;
    NGTABLEPTR new_hptr ;				/* new hash table entry */
    int i, oldsize ;

    oldsize = hashtable->size ;
    oldtable = hashtable->hash_table;

    if( hashtable->power_of_two ) {
        hashtable->size = nghash_table_size2( num - 1 ) ;
    } else {
        hashtable->size = nghash_table_size( num ) ;
    }
    hashtable->num_entries = 0 ;
    hashtable->thread = NULL ;
    hashtable->last_entry = NULL ; /* end of list */
    hashtable->need_resize = hashtable->size * hashtable->max_density ;

    hashtable->hash_table = NGMALLOC( hashtable->size, NGTABLEPTR);
    for( i = 0 ; i < oldsize ; i++ ) {
        for( hptr = oldtable[i]; hptr; ) {
            zapptr = hptr ;
            nghash_insert( hashtable, hptr->key, hptr->data ) ;
            if( hashtable->searchPtr && hashtable->searchPtr == hptr ) {
                new_hptr = _nghash_find_item(hashtable, hptr->key, hptr->data ) ;
                hashtable->searchPtr = new_hptr ;
            }
            if( hashtable->enumeratePtr && hashtable->enumeratePtr == hptr ) {
                new_hptr = _nghash_find_item(hashtable, hptr->key, hptr->data ) ;
                hashtable->enumeratePtr = new_hptr ;
            }
            /* Now safe to free */
            if( hashtable->hash_func == NGHASH_DEF_HASH(NGHASH_FUNC_STR) ) {
                NGFREE( hptr->key);
            }
            hptr = hptr->next ;
            NGFREE( zapptr ) ;
        }
    }

    NGFREE( oldtable );
} /* end nghash_resize() */


void nghash_reset_stat(NGHASHPTR hashtable)
{
    hashtable->collision = 0 ;
    hashtable->access = 0 ;
}

NGHASHPTR nghash_init(int num_entries)
{
    return( nghash_init_with_parms( NGHASH_DEF_CMP(NGHASH_FUNC_STR), NGHASH_DEF_HASH(NGHASH_FUNC_STR),
                                    num_entries, NGHASH_DEF_MAX_DENSITY,
                                    NGHASH_DEF_GROW_FACTOR, NGHASH_UNIQUE) ) ;
} /* end nghash_init() */

NGHASHPTR nghash_init_pointer(int num_entries)
{
    return( nghash_init_with_parms( NGHASH_DEF_CMP(NGHASH_FUNC_PTR), NGHASH_DEF_HASH(NGHASH_FUNC_PTR),
                                    num_entries, NGHASH_DEF_MAX_DENSITY,
                                    NGHASH_DEF_GROW_FACTOR,
                                    NGHASH_UNIQUE_TWO) ) ;
} /* end nghash_init_pointer() */

NGHASHPTR nghash_init_integer(int num_entries)
{
    return( nghash_init_with_parms( NGHASH_DEF_CMP(NGHASH_FUNC_NUM), NGHASH_DEF_HASH(NGHASH_FUNC_NUM),
                                    num_entries, NGHASH_DEF_MAX_DENSITY,
                                    NGHASH_DEF_GROW_FACTOR,
                                    NGHASH_UNIQUE_TWO) ) ;
} /* end nghash_init_integer() */

int nghash_table_get(NGHASHPTR hashtable)
{
    return(hashtable->size) ;
}

int nghash_max_density(NGHASHPTR hashtable,int max_density)
{
    if( max_density > 0 ) {
        hashtable->max_density = max_density ;
        hashtable->need_resize = hashtable->size * hashtable->max_density ;
    }
    return(hashtable->max_density) ;
}

void nghash_empty(NGHASHPTR hashtable, void (*delete_data) (void *),
                  void (*delete_key) (void *))
{
    NGTABLEPTR *table, hptr , zapptr ;

    nghash_reset_stat(hashtable);

    table = hashtable->hash_table ;
    if( table ) {
        for( hptr = hashtable->thread ; hptr ; ) {
            zapptr = hptr ;
            hptr = hptr->thread_next ;

            /* execute user define delete function if requested */
            if( delete_data ) {
                delete_data (zapptr->data);
            }
            if( hashtable->hash_func == NGHASH_DEF_HASH(NGHASH_FUNC_STR) ) {
                /* we allocated this ourselves we can delete it */
                NGFREE( zapptr->key ) ;
            } else if( delete_key ) {
                delete_key (zapptr->key);
            }
            NGFREE( zapptr ) ;
        }
        memset(table, 0, (size_t) hashtable->size*sizeof(NGTABLEPTR)) ;
    }
    /* free decks associated with tree if they exist */
    hashtable->thread = NULL ; /* initialize list */
    hashtable->last_entry = NULL ; /* initialize list */
    hashtable->num_entries = 0 ;
} /* end nghash_empty() */


void nghash_free(NGHASHPTR hashtable, void (*delete_data) (void *),
                 void (*delete_key) (void *))
{
    hashtable->call_from_free = TRUE;
    nghash_empty(hashtable, delete_data, delete_key ) ;
    hashtable->call_from_free = FALSE ;
    NGFREE( hashtable->hash_table ) ;
    NGFREE( hashtable ) ;
} /* end nghash_free() */

void nghash_free_string_func( char *str )
{
    NGFREE( str ) ;
} /* end nghash_free_string_func() */

void nghash_free_string_hashtable(NGHASHPTR hashtable)
{
    hashtable->call_from_free = TRUE;
    nghash_empty(hashtable, (ngdelete) nghash_free_string_func, NULL ) ;
    hashtable->call_from_free = FALSE ;
    NGFREE( hashtable->hash_table ) ;
    NGFREE( hashtable ) ;
} /* end nghash_free_string_hashtable() */

/* -----------------------------------------------------------------
 * Now nghash_search is broken into four routines: nghash_find,
 * nghash_find_again, nghash_delete, and nghash_insert.
----------------------------------------------------------------- */
void * _nghash_find(NGHASHPTR hashtable, void * user_key,BOOL *status)
{
    int  ret_code ;
    unsigned int hsum ;
    NGTABLEPTR curPtr ;
    NGTABLEPTR *table ;

    /* initialization */
    table = hashtable->hash_table ;
    DS(hashtable->access++;) ;

    /* -----------------------------------------------------------------
     * Process the hash function.
    ----------------------------------------------------------------- */
    switch( (intptr_t) hashtable->hash_func ) {
    case NGHASH_FUNC_STR:
        NGHASH_STR_TO_HASH( user_key, hsum, hashtable->size);
        break ;
    case NGHASH_FUNC_PTR:
        NGHASH_PTR_TO_HASH( user_key, hsum, hashtable->size);
        break ;
    case NGHASH_FUNC_NUM:
        NGHASH_NUM_TO_HASH( user_key, hsum, hashtable->size);
        break ;
    default:
        hsum = hashtable->hash_func (hashtable, user_key);
    }

    curPtr = table[hsum] ;
    if( curPtr ) {
        /* list started at this hash */
        for( ; curPtr ; curPtr=curPtr->next ) {
            if( hashtable->compare_func == NGHASH_DEF_CMP(NGHASH_FUNC_STR) ) {
                ret_code = strcmp((char *)curPtr->key, (char *) user_key ) ;
            } else if ( hashtable->compare_func == NGHASH_DEF_CMP(NGHASH_FUNC_PTR) ||
                        hashtable->compare_func == NGHASH_DEF_CMP(NGHASH_FUNC_NUM) ) {
                ret_code = NGHASH_PTR_COMPARE_FUNC( curPtr->key, user_key );
            } else {
                ret_code = hashtable->compare_func (curPtr->key, user_key);
            }
            if( ret_code == STRINGEQ ) {
                /* ----------------------------------------------------
                 * Operation find or enter with unique items, we
                 * return item.  On a nonunique enter, add item below.
                ------------------------------------------------------- */
                hashtable->searchPtr = curPtr ;
                if( status ) {
                    *status = TRUE ;
                }
                return( curPtr->data ) ;
            }
        }
    }
    /* cant find anything on a find operation */
    hashtable->searchPtr = NULL ;
    if( status ) {
        *status = FALSE ;
    }
    return( NULL ) ;

} /* end nghash_find() */

/* -----------------------------------------------------------------
 * This find does not understand 0 or NULL data items.  Use _nghash_find
 * instead.
 * ----------------------------------------------------------------- */
void * nghash_find(NGHASHPTR hashtable, void * user_key)
{
    return( _nghash_find( hashtable, user_key, NULL ) ) ;
} /* end nghash_find() */


void * _nghash_find_again(NGHASHPTR hashtable, void * user_key,BOOL *status)
{
    int  ret_code ;		/* comparison return code */
    NGTABLEPTR curPtr ;		/* current hashtable entry */

    /* initialization */
    DS(hashtable->access++;) ;

    if( hashtable->searchPtr ) {
        for(curPtr=hashtable->searchPtr->next; curPtr ; curPtr=curPtr->next ) {
            if( hashtable->compare_func == NGHASH_DEF_CMP(NGHASH_FUNC_STR) ) {
                ret_code = strcmp((char *)curPtr->key, (char *) user_key ) ;
            } else if ( hashtable->compare_func == NGHASH_DEF_CMP(NGHASH_FUNC_PTR) ||
                        hashtable->compare_func == NGHASH_DEF_CMP(NGHASH_FUNC_NUM) ) {
                ret_code = NGHASH_PTR_COMPARE_FUNC( curPtr->key, user_key );
            } else {
                ret_code = hashtable->compare_func (curPtr->key, user_key);
            }
            if( ret_code == STRINGEQ ) {
                hashtable->searchPtr = curPtr ;
                if( status ) {
                    *status = TRUE ;
                }
                return( curPtr->data ) ;
            }
        }
    }
    if( status ) {
        *status = FALSE ;
    }
    return(NULL) ;
} /* end _nghash_find_again() */

void * nghash_find_again(NGHASHPTR hashtable, void * user_key)
{
    return( _nghash_find_again( hashtable, user_key, NULL ) ) ;
} /* end nghash_find_again() */


void * nghash_delete(NGHASHPTR hashtable, void * user_key)
{
    int  ret_code ;
    unsigned int hsum ;
    void * user_data_p ;
    NGTABLEPTR curPtr, *prevPtr ;
    NGTABLEPTR *table ;

    /* initialization */
    table = hashtable->hash_table ;
    DS(hashtable->access++;) ;

    /* -----------------------------------------------------------------
     * Process the hash function.
    ----------------------------------------------------------------- */
    switch( (intptr_t) hashtable->hash_func ) {
    case NGHASH_FUNC_STR:
        NGHASH_STR_TO_HASH( user_key, hsum, hashtable->size);
        break ;
    case NGHASH_FUNC_PTR:
        NGHASH_PTR_TO_HASH( user_key, hsum, hashtable->size);
        break ;
    case NGHASH_FUNC_NUM:
        NGHASH_NUM_TO_HASH( user_key, hsum, hashtable->size);
        break ;
    default:
        hsum = hashtable->hash_func (hashtable, user_key);
    }

    /* insert into table only if distinct number */
    curPtr = table[hsum] ;
    if( curPtr ) {
        /* list started at this hash */
        prevPtr = table + hsum ;
        for( ; curPtr ; prevPtr = &(curPtr->next), curPtr=curPtr->next ) {
            if( hashtable->compare_func == NGHASH_DEF_CMP(NGHASH_FUNC_STR) ) {
                ret_code = strcmp((char *)curPtr->key, (char *) user_key ) ;
            } else if ( hashtable->compare_func == NGHASH_DEF_CMP(NGHASH_FUNC_PTR) ||
                        hashtable->compare_func == NGHASH_DEF_CMP(NGHASH_FUNC_NUM) ) {
                ret_code = NGHASH_PTR_COMPARE_FUNC( curPtr->key, user_key );
            } else {
                ret_code = hashtable->compare_func (curPtr->key, user_key);
            }
            if( ret_code == STRINGEQ ) {
                if( curPtr->thread_prev ) { /* no sentinel */
                    curPtr->thread_prev->thread_next = curPtr->thread_next;
                } else {
                    hashtable->thread = curPtr->thread_next ;
                }
                if( curPtr->thread_next ) { /* no sentinel */
                    curPtr->thread_next->thread_prev = curPtr->thread_prev ;
                } else {
                    hashtable->last_entry = curPtr->thread_prev ;
                }
                *prevPtr = curPtr->next ;
                if( hashtable->hash_func == NGHASH_DEF_HASH(NGHASH_FUNC_STR) ) {
                    /* we allocated this ourselves we can delete it */
                    NGFREE( curPtr->key ) ;
                }
                user_data_p = curPtr->data ;
                NGFREE(curPtr);
                hashtable->num_entries-- ;
                return( user_data_p ) ;
            }
        }
    }
    return( NULL ) ; /* didn't find anything to delete */

} /* end nghash_delete() */


/* different return value:
    NGHASH_FUNC_PTR always returns NULL.
    This function returns the pointer, if _not_ successful, otherwise NULL */
void * nghash_delete_special(NGHASHPTR hashtable, void* user_key)
{
    int  ret_code;
    unsigned int hsum;
    void* user_data_p;
    NGTABLEPTR curPtr, * prevPtr;
    NGTABLEPTR* table;

    /* initialization */
    table = hashtable->hash_table;
    DS(hashtable->access++;);

    /* -----------------------------------------------------------------
     * Process the hash function.
      ----------------------------------------------------------------- */
    switch ((intptr_t)hashtable->hash_func) {
    case NGHASH_FUNC_STR:
        NGHASH_STR_TO_HASH(user_key, hsum, hashtable->size);
        break;
    case NGHASH_FUNC_PTR:
        NGHASH_PTR_TO_HASH(user_key, hsum, hashtable->size);
        break;
    case NGHASH_FUNC_NUM:
        NGHASH_NUM_TO_HASH(user_key, hsum, hashtable->size);
        break;
    default:
        hsum = hashtable->hash_func(hashtable, user_key);
    }

    /* insert into table only if distinct number */
    curPtr = table[hsum];
    if (curPtr) {
        /* list started at this hash */
        prevPtr = table + hsum;
        for (; curPtr; prevPtr = &(curPtr->next), curPtr = curPtr->next) {
            if (hashtable->compare_func == NGHASH_DEF_CMP(NGHASH_FUNC_STR)) {
                ret_code = strcmp((char*)curPtr->key, (char*)user_key);
            }
            else if (hashtable->compare_func == NGHASH_DEF_CMP(NGHASH_FUNC_PTR) ||
                hashtable->compare_func == NGHASH_DEF_CMP(NGHASH_FUNC_NUM)) {
                ret_code = NGHASH_PTR_COMPARE_FUNC(curPtr->key, user_key);
            }
            else {
                ret_code = hashtable->compare_func(curPtr->key, user_key);
            }
            if (ret_code == STRINGEQ) {
                if (curPtr->thread_prev) { /* no sentinel */
                    curPtr->thread_prev->thread_next = curPtr->thread_next;
                }
                else {
                    hashtable->thread = curPtr->thread_next;
                }
                if (curPtr->thread_next) { /* no sentinel */
                    curPtr->thread_next->thread_prev = curPtr->thread_prev;
                }
                else {
                    hashtable->last_entry = curPtr->thread_prev;
                }
                *prevPtr = curPtr->next;
                if (hashtable->hash_func == NGHASH_DEF_HASH(NGHASH_FUNC_STR)) {
                    /* we allocated this ourselves we can delete it */
                    NGFREE(curPtr->key);
                }
                user_data_p = curPtr->data;
                NGFREE(curPtr);
                hashtable->num_entries--;
                return(user_data_p);
            }
        }
    }
    return(user_key); /* didn't find anything to delete */
}



void * nghash_insert(NGHASHPTR hashtable, void * user_key, void * data)
{
    int  ret_code ;
    unsigned int hsum ;
    NGTABLEPTR curPtr, temptr, curTable ;
    NGTABLEPTR *table ;

    /* initialization */
    table = hashtable->hash_table ;
    DS(hashtable->access++;)

    /* -----------------------------------------------------------------
     * Process the hash function.
    ----------------------------------------------------------------- */
    switch( (intptr_t) hashtable->hash_func ) {
    case NGHASH_FUNC_STR:
        NGHASH_STR_TO_HASH( user_key, hsum, hashtable->size);
        break ;
    case NGHASH_FUNC_PTR:
        NGHASH_PTR_TO_HASH( user_key, hsum, hashtable->size);
        break ;
    case NGHASH_FUNC_NUM:
        NGHASH_NUM_TO_HASH( user_key, hsum, hashtable->size);
        break ;
    default:
        hsum = hashtable->hash_func (hashtable, user_key);
    }

    /* insert into table only if distinct number */
    temptr = table[hsum] ;
    if( temptr ) {
        /* list started at this hash */
        for( curPtr = temptr ; curPtr ; curPtr=curPtr->next ) {
            DS(hashtable->collision++;) ;
            if( hashtable->compare_func == NGHASH_DEF_CMP(NGHASH_FUNC_STR) ) {
                ret_code = strcmp((char *)curPtr->key, (char *) user_key ) ;
            } else if ( hashtable->compare_func == NGHASH_DEF_CMP(NGHASH_FUNC_PTR) ||
                        hashtable->compare_func == NGHASH_DEF_CMP(NGHASH_FUNC_NUM) ) {
                ret_code = NGHASH_PTR_COMPARE_FUNC( curPtr->key, user_key);
            } else {
                ret_code = hashtable->compare_func (curPtr->key, user_key);
            }
            if( ret_code == STRINGEQ ) {
                if( hashtable->unique ) {
                    /* ----------------------------------------------------
                     * Operation enter with unique items, we
                     * return item.  On a nonunique enter, add item below.
                    ------------------------------------------------------- */
                    hashtable->searchPtr = curPtr ;
                    return( curPtr->data ) ;
                } else {
                    break ; /* avoid some work for nonunique hash_enter */
                }
            }
        }
    }

    /* now save data */
    hashtable->num_entries++ ;
    table[hsum] = curTable = NGMALLOC(1,NGTABLEBOX);
    curTable->data = data ;
    if( hashtable->hash_func == NGHASH_DEF_HASH(NGHASH_FUNC_STR) ) {
        curTable->key = copy((char *) user_key);
    } else {
        curTable->key = user_key ;
    }
    curTable->next = temptr ;
    /* now fix thread which goes through hash table */
    if( hashtable->last_entry ) {
        hashtable->last_entry->thread_next = curTable ;
        curTable->thread_prev = hashtable->last_entry ;
        hashtable->last_entry = curTable ;
    } else {
        hashtable->thread = hashtable->last_entry = curTable ;
        curTable->thread_prev = NULL ;
    }
    curTable->thread_next = NULL ;

    if( hashtable->num_entries >= hashtable->need_resize ) {
        int newsize ;		/* new size of table */
        newsize = (int)(hashtable->size * hashtable->growth_factor);
        nghash_resize(hashtable, newsize ) ;
    }

    return( NULL ) ; /* no conflict on a enter */

} /* end nghash_insert() */

/* returns the pointer with the item */
static NGTABLEPTR _nghash_find_item(NGHASHPTR htable,void * user_key,void * data)
{
    int  ret_code ;
    unsigned int hsum ;
    NGTABLEPTR curPtr, temptr ;
    NGTABLEPTR *table ;

    /* initialization */
    table = htable->hash_table ;

    /* -----------------------------------------------------------------
     * Process the hash function.
    ----------------------------------------------------------------- */
    switch( (intptr_t) htable->hash_func ) {
    case NGHASH_FUNC_STR:
        NGHASH_STR_TO_HASH( user_key, hsum, htable->size);
        break ;
    case NGHASH_FUNC_PTR:
        NGHASH_PTR_TO_HASH( user_key, hsum, htable->size);
        break ;
    case NGHASH_FUNC_NUM:
        NGHASH_NUM_TO_HASH( user_key, hsum, htable->size);
        break ;
    default:
        hsum = htable->hash_func (htable, user_key);
    }

    /* insert into table only if distinct number */
    if( (temptr = table[hsum]) != NULL ) {
        /* list started at this hash */
        for(curPtr=temptr ; curPtr ; curPtr=curPtr->next ) {
            if( htable->compare_func == NGHASH_DEF_CMP(NGHASH_FUNC_STR) ) {
                ret_code = strcmp((char *)curPtr->key, (char *) user_key ) ;
            } else if ( htable->compare_func == NGHASH_DEF_CMP(NGHASH_FUNC_PTR) ||
                        htable->compare_func == NGHASH_DEF_CMP(NGHASH_FUNC_NUM) ) {
                ret_code = NGHASH_PTR_COMPARE_FUNC( curPtr->key, user_key);
            } else {
                ret_code = htable->compare_func (curPtr->key, user_key);
            }
            if( ret_code == STRINGEQ ) {
                if( data ) {
                    if( curPtr->data == data ) {
                        return( curPtr ) ;
                    }
                } else {
                    return( curPtr ) ;
                }
            }
        }
    }
    return( NULL ) ; /* no matching item */

} /* end _nghash_find_item() */



void * nghash_enumeratek(NGHASHPTR htable, void * *key_return, BOOL start_flag)
{
    NGTABLEPTR current_spot ;	/* current place in threaded list */

    if( start_flag ) {
        if( (htable->enumeratePtr = htable->thread) != NULL ) {
            current_spot = htable->enumeratePtr ;
            *key_return = current_spot->key ;
            return( current_spot->data ) ;
        }
    } else {
        if( htable->enumeratePtr &&
                (htable->enumeratePtr = htable->enumeratePtr->thread_next) != NULL ) {
            current_spot = htable->enumeratePtr ;
            *key_return = current_spot->key ;
            return( current_spot->data ) ;
        }
    }
    *key_return = NULL ;
    return( NULL ) ;

} /* end nghash_enumeratek() */

void * nghash_enumerate(NGHASHPTR htable,BOOL start_flag)
{
    NGTABLEPTR current_spot ;	/* current place in threaded list */

    if( start_flag ) {
        if( (htable->enumeratePtr = htable->thread) != NULL ) {
            current_spot = htable->enumeratePtr ;
            return( current_spot->data ) ;
        }
    } else {
        if( htable->enumeratePtr &&
                (htable->enumeratePtr = htable->enumeratePtr->thread_next) != NULL ) {
            current_spot = htable->enumeratePtr ;
            return( current_spot->data ) ;
        }
    }
    return( NULL ) ;

} /* end nghash_enumerate() */

/* -----------------------------------------------------------------
 * This is the reentrant version which uses an iterator.
 * ----------------------------------------------------------------- */
void * nghash_enumeratekRE(NGHASHPTR htable, void * *key_return, NGHASHITERPTR iter_p)
{
    NGTABLEPTR current_spot ;	/* current place in threaded list */
    FUNC_NAME("nghash_enumeratekRE") ;

    if(!(iter_p)) {
        fprintf( stderr, "ERROR[%s]:Null iterator pointer.\n", routine ) ;
        return(NULL) ;
    }
    if(!(iter_p->position)) {
        if( (iter_p->position = htable->thread) != NULL ) {
            current_spot = iter_p->position ;
            *key_return = current_spot->key ;
            return( current_spot->data ) ;
        }
    } else {
        if( iter_p->position &&
                (iter_p->position = iter_p->position->thread_next) != NULL ) {
            current_spot = iter_p->position ;
            *key_return = current_spot->key ;
            return( current_spot->data ) ;
        }
    }
    *key_return = NULL ;
    return( NULL ) ;

} /* end nghash_enumeratekRE() */

/* -----------------------------------------------------------------
 * This is the reentrant version which uses an iterator.
 * ----------------------------------------------------------------- */
void * nghash_enumerateRE(NGHASHPTR htable, NGHASHITERPTR iter_p)
{
    NGTABLEPTR current_spot ;	/* current place in threaded list */
    FUNC_NAME("nghash_enumerateRE") ;

    if(!(iter_p)) {
        fprintf( stderr, "ERROR[%s]:Null iterator pointer.\n", routine ) ;
        return(NULL) ;
    }
    if(!(iter_p->position)) {
        if( (iter_p->position = htable->thread) != NULL ) {
            current_spot = iter_p->position ;
            return( current_spot->data ) ;
        }
    } else {
        if( iter_p->position &&
                (iter_p->position = iter_p->position->thread_next) != NULL ) {
            current_spot = iter_p->position ;
            return( current_spot->data ) ;
        }
    }
    return( NULL ) ;

} /* end nghash_enumerateRE() */

/* -----------------------------------------------------------------
 * Merge items in merge_htable into master_htable.  Create master_htable
 * if master_htable is NULL.  Returns the merged hash table.
 * ----------------------------------------------------------------- */
NGHASHPTR nghash_merge( NGHASHPTR master_htable, NGHASHPTR merge_htable )
{
    NGTABLEPTR ptr ;			/* traverse hash table */

    if(!(master_htable)) {
        master_htable = NGMALLOC( 1, NGHASHBOX ) ;
        *master_htable = *merge_htable ;
        master_htable->hash_table = NGMALLOC( master_htable->size, NGTABLEPTR ) ;
        master_htable->thread = NULL ;
        master_htable->last_entry = NULL ;
        master_htable->num_entries = 0 ;
        master_htable->enumeratePtr = NULL ;
        master_htable->searchPtr = NULL ;
        master_htable->access = 0 ;
        master_htable->collision = 0 ;
    }
    for( ptr = merge_htable->thread ; ptr ; ptr = ptr->thread_next ) {
        nghash_insert( master_htable, ptr->key, ptr->data ) ;
    }
    return( master_htable ) ;
} /* end nghash_merge() */


int nghash_get_size(NGHASHPTR hashtable)
{
    return( hashtable->num_entries ) ;
}

void nghash_dump(NGHASHPTR htable, void (*print_key) (void *))
{
    int i ;			/* counter */
    int count ;			/* counter */
    NGTABLEPTR *table ;		/* hash table */
    NGTABLEPTR hptr ;		/* hash table element */

    table = htable->hash_table ;
    fprintf( stderr, "Dump of hashtable containing %d entries...\n",
             htable->num_entries ) ;
    fprintf( stderr, "Table is %4.2f%% full\n",
             100.0 * (double) htable->num_entries / (double) htable->size ) ;
    for( i = 0 ; i < htable->size ; i++ ) {
        if( (hptr = table[i]) != NULL ) {
            fprintf( stderr, " [%5d]:", i ) ;
            count = 0 ;
            for( ; hptr ; hptr=hptr->next ) {
                if( ++count == 3 ) {
                    fprintf( stderr, "\n\t" ) ;
                    count = 0 ;
                }
                if( htable->hash_func == NGHASH_DEF_HASH(NGHASH_FUNC_STR) ) {
                    fprintf( stderr, " key:%s ", (char *) hptr->key ) ;
                } else {
                    fprintf( stderr, " key:%p ", hptr->key ) ;
                }
                if( print_key) {
                    print_key (hptr->data);
                } else {
                    fprintf( stderr, " data:%p ", hptr->data ) ;
                }
            }
            fprintf( stderr, "\n" ) ;
        }
    } /* end for( i = 0... */

} /* end nghash_enumerate() */

/* -----------------------------------------------------------------
 * nghash_deleteItem - deletes a specified item out of the hash table.
 * To be useful, unique flag should be off.  Otherwise just use nghash_delete.
 * Returns true if item was deleted.
----------------------------------------------------------------- */
BOOL nghash_deleteItem(NGHASHPTR hashtable, void * user_key, void * data)
{
    int  ret_code ;
    unsigned long hsum ;
    NGTABLEPTR curPtr, temptr, *prevPtr ;
    NGTABLEPTR *table ;

    /* initialization */
    table = hashtable->hash_table ;

    /* -----------------------------------------------------------------
     * Process the hash function.
    ----------------------------------------------------------------- */
    switch( (intptr_t) hashtable->hash_func ) {
    case NGHASH_FUNC_STR:
        NGHASH_STR_TO_HASH( user_key, hsum, hashtable->size);
        break ;
    case NGHASH_FUNC_PTR:
        NGHASH_PTR_TO_HASH( user_key, hsum, hashtable->size);
        break ;
    case NGHASH_FUNC_NUM:
        NGHASH_NUM_TO_HASH( user_key, hsum, hashtable->size);
        break ;
    default:
        hsum = hashtable->hash_func (hashtable, user_key);
    }

    /* insert into table only if distinct number */
    temptr = table[hsum] ;
    if( temptr ) {
        /* list started at this hash */
        prevPtr = table + hsum ;
        for(curPtr=temptr; curPtr; prevPtr = &(curPtr->next), curPtr=curPtr->next ) {
            /* -----------------------------------------------------------------
             * Look for match.
            ----------------------------------------------------------------- */
            if( hashtable->compare_func == NGHASH_DEF_CMP(NGHASH_FUNC_STR) ) {
                ret_code = strcmp((char *)curPtr->key, (char *) user_key ) ;
            } else if ( hashtable->compare_func == NGHASH_DEF_CMP(NGHASH_FUNC_PTR) ||
                        hashtable->compare_func == NGHASH_DEF_CMP(NGHASH_FUNC_NUM) ) {
                ret_code = NGHASH_PTR_COMPARE_FUNC( curPtr->key, user_key );
            } else {
                ret_code = hashtable->compare_func (curPtr->key, user_key);
            }
            if( ret_code == STRINGEQ ) {
                if( curPtr->data == data ) {
                    if( curPtr->thread_prev ) { /* no sentinel */
                        curPtr->thread_prev->thread_next = curPtr->thread_next;
                    } else {
                        hashtable->thread = curPtr->thread_next ;
                    }
                    if( curPtr->thread_next ) { /* no sentinel */
                        curPtr->thread_next->thread_prev = curPtr->thread_prev ;
                    } else {
                        hashtable->last_entry = curPtr->thread_prev ;
                    }
                    *prevPtr = curPtr->next;
                    if( hashtable->hash_func == NGHASH_DEF_HASH(NGHASH_FUNC_STR) ) {
                        /* we allocated this ourselves we can delete it */
                        NGFREE( curPtr->key ) ;
                    }
                    NGFREE(curPtr);
                    hashtable->num_entries-- ;
                    return( TRUE ) ;
                }
            }
        } /* end for(curPtr=temptr... */
    } /* end if( temptr... */

    return( FALSE ) ; /* could not find item */

} /* end nghash_deleteItem() */

/*---------------------------- hash_table_size -------------------------*/
int nghash_table_size(int minEntries)
{
    int   i;
    BOOL  isPrime;
    int   prime;
    int   testPrime;
    static int   primes[PRIMECOUNT] = {
        3,   5,   7,  11,  13,  17,  19,  23,  29,  31,
        37,  41,  43,  47,  53,  59,  61,  67,  71,  73,
        79,  83,  89,  97, 101, 103, 107, 109, 113, 127,
        131, 137, 139, 149, 151, 157, 163, 167, 173, 179,
        181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
        239, 241, 251, 257, 263, 269, 271, 277, 281, 283,
        293, 307, 311, 313, 317, 331, 337, 347, 349, 353,
        359, 367, 373, 379, 383, 389, 397, 401, 409, 419,
        421, 431, 433, 439, 443, 449, 457, 461, 463, 467,
        479, 487, 491, 499, 503, 509, 521, 523, 541, 547,
        557, 563, 569, 571, 577, 587, 593, 599, 601, 607,
        613, 617, 619, 631, 641, 643, 647, 653, 659, 661,
        673, 677, 683, 691, 701, 709, 719, 727, 733, 739,
        743, 751, 757, 761, 769, 773, 787, 797, 809, 811,
        821, 823, 827, 829, 839, 853, 857, 859, 863, 877,
        881, 883, 887, 907, 991, 919, 929, 937, 941, 947,
        953, 967, 971, 977, 983, 991, 997,1009,1013,1019,
        1021,1031,1033,1039,1049,1051,1061,1063,1069,1087,
        1091,1093,1097,1103,1109,1117,1123,1129,1151,1153,
        1163,1171,1181,1187,1193,1201,1213,1217,1223,1229
    };

    if (minEntries <= MINPRIMESIZE) {
        return(MINPRIMESIZE);
    } else {
        testPrime = minEntries;
        /* test to see if even */
        if ((testPrime % 2) == 0) {
            testPrime = testPrime + 1;
        }
        do {
            testPrime = testPrime + 2;
            isPrime = TRUE;
            for (i=0; i < PRIMECOUNT; i++) {
                prime = primes[i];
                if (testPrime < prime*prime) {
                    break;
                }
                if ((testPrime % prime) == 0) {
                    isPrime = FALSE;
                    break;
                }
            }
        } while (!(isPrime));

        return(testPrime);
    }

}  /*  FUNCTION nghash_table_size */

int nghash_table_size2(int minEntries)
{
    int power ;
    int table_size ;
    power = 0 ;
    while( minEntries > 0 ) {
        minEntries = minEntries >> 1 ;
        power++ ;
    }
    power = MIN( power, 32 ) ;
    table_size = 1 << power ;
    table_size = MAX( NGHASH_MIN_SIZE, table_size ) ;
    return( table_size ) ;

} /* end nghash_table_size2() */


void nghash_distribution(NGHASHPTR hashtable)
{
    int i ;				/* counter */
    long min ;				/* min count */
    long max ;				/* max count */
    long nzero_cnt ;			/* non zero count */
    long this_count ;			/* count items at this list */
    long tablesize ;			/* table size */
    double avg ;				/* average */
    double sum2 ;				/* squared sum */
    double diff ;				/* difference */
    double diff2 ;			/* difference squared */
    double nzavg ;			/* non zero average */
    double variance ;			/* variance of table */
    NGTABLEPTR *table ;			/* hash table */
    NGTABLEPTR hptr ;			/* hash table pointer */
    FUNC_NAME("nghash_distribution" ) ;

    tablesize = nghash_tablesize(hashtable) ;
    table = hashtable->hash_table ;
    avg = hashtable->num_entries / (double) tablesize ;
    sum2 = 0.0 ;
    min = max = 0 ;
    nzero_cnt = 0 ;
    for( i = 0 ; i < tablesize ; i++ ) {
        this_count = 0 ;
        for( hptr = table[i]; hptr; hptr = hptr->next ) {
            this_count++ ;
        }
        if( i == 0 ) {
            min = max = this_count ;
        } else {
            if( this_count < min ) {
                min = this_count ;
            }
            if( this_count > max ) {
                max = this_count ;
            }
        }
        if( this_count > 0 ) {
            nzero_cnt++ ;
        }
        diff = (double)this_count - avg ;
        diff2 = diff * diff ;
        sum2 += diff2 ;
    }
    variance = sum2 / hashtable->num_entries ;
    nzavg = hashtable->num_entries / (double) nzero_cnt ;
    fprintf( stderr, "[%s]:min:%ld max:%ld nonzero avg:%f\n", routine, min, max, nzavg ) ;
    fprintf( stderr, "  variance:%f std dev:%f target:%f nonzero entries:%ld / %ld\n",
             variance, sqrt(variance), avg, nzero_cnt, tablesize ) ;
} /* end nghash_distribution() */
