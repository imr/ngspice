/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE d_state/cfunc.mod

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503-405
               

AUTHORS                      

    6 June 1991     Jeffrey P. Murray


MODIFICATIONS   

    30 Sept 1991    Jeffrey P. Murray
                                   

SUMMARY

    This file contains the model-specific routines used to
    functionally describe the <model_name> code model.


INTERFACES       

    FILE                 ROUTINE CALLED     

    CMmacros.h           cm_message_send();                   
                             
    CMevt.c              void *cm_event_alloc()
                         void *cm_event_get_ptr()


REFERENCED FILES

    Inputs from and outputs to ARGS structure.
                     

NON-STANDARD FEATURES

    NONE

===============================================================================*/

/*=== INCLUDE FILES ====================*/

#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

/*=== CONSTANTS ========================*/

#define MAX_STRING_SIZE 200
                      
#define HEADER 1
#define CONTINUATION 2

#define OK 0
#define FAIL 1                                        

/*=== MACROS ===========================*/

/*=== LOCAL VARIABLES & TYPEDEFS =======*/                         

typedef struct {
    int current_state,  /* current state of the machine (similar to
                           current state of the union, current state
                           of the economy, etc. etc. 8-)                */
        index0,         /* actual word number (0 to depth-1) in which the
                           first line of the current_state definition
                           may be found.                                */
        indexN;         /* word number (0 to depth-1) of the final curren_state
                           definition line...if a state is defined in only one
                           line, index0 and indexN will be equal.       */
} Dynamic_State_t;

typedef struct {
    int     num_outputs,/* width of bits[] table...equal to size of out port.*/
            num_inputs, /* width of inputs[] table...equal to size of in port*/
            depth,      /* depth of table...equal to size of
                           current_state & next_state arrays,
                           and to the total number of vectors
                           retrieved from the state.in file.            */
           *state,      /* integer array holding the state
                           index values...note that each state will
                           have at least one and as many as 2^N
                           "words" assigned to it, where N = number
                           of input lines to the state machine.         */
           *next_state; /* integer array holding the next state
                           to jump to given the input values
                           held by the inputs[] array...note that each
                           state will have at least one and as many as 2^N
                           "words" assigned to it, where N = number
                           of input lines to the state machine.         */


    short  *bits,       /* the storage array for the
                           output bit representations...
                           this will have size equal to
                           (width * depth)/4, since one
                           short will hold four 12-state
                           bit descriptions.                */
           *inputs;     /* the storage array for the
                           input bit representations...
                           this will have size equal to
                           (width * depth)/8, since one
                           short will hold eight 3-state
                           bit descriptions.                */
} State_Table_t;

/* Type definition for each possible token returned. */

typedef enum token_type_s {CNV_NO_TOK,CNV_STRING_TOK} Cnv_Token_Type_t;
typedef char line_t[82]; /* A SPICE size line. <= 80 characters plus '\n\0' */


/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/


/*==============================================================================

FUNCTION *CNVgettok()

AUTHORS                      

    13 Jun 1991     Jeffrey P. Murray

MODIFICATIONS   

     8 Aug 1991     Jeffrey P. Murray
    30 Sep 1991     Jeffrey P. Murray

SUMMARY

    This function obtains the next token from an input stream.

INTERFACES       

    FILE                 ROUTINE CALLED

    N/A                  N/A

RETURNED VALUE
    
    Returns a string value representing the next token.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/


/*=== Static CNVgettok ROUTINE ================*/
/*
Get the next token from the input string.  The input string pointer
is advanced to the following token and the token from the input
string is copied to malloced storage and a pointer to that storage
is returned.  The original input string is undisturbed.
*/

static char  *CNVgettok(char **s)

{

    char    *buf;       /* temporary storage to copy token into */
    /*char    *temp;*/      /* temporary storage to copy token into */
    char    *ret_str;   /* storage for returned string */

    int     i;

    /* allocate space big enough for the whole string */

    buf = (char *) malloc(strlen(*s) + 1);

    /* skip over any white space */

    while(isspace_c(**s) || (**s == '=') ||
          (**s == '(') || (**s == ')') || (**s == ','))
          (*s)++;

    /* isolate the next token */

    switch(**s) {

    case '\0':           /* End of string found */
        if(buf) free(buf);
        return(NULL);


    default:             /* Otherwise, we are dealing with a    */
                         /* string representation of a number   */
                         /* or a mess o' characters.            */
        i = 0;
        while( (**s != '\0') &&
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

    while(isspace_c(**s) || (**s == '=') ||
          (**s == '(') || (**s == ')') || (**s == ','))
          (*s)++;

    /* make a copy using only the space needed by the string length */


    ret_str = (char *) malloc(strlen(buf) + 1);
    ret_str = strcpy(ret_str,buf);

    if(buf) free(buf);

    return(ret_str);
}


/*==============================================================================

FUNCTION *CNVget_token()

AUTHORS                      

    13 Jun 1991     Jeffrey P. Murray

MODIFICATIONS   

     8 Aug 1991     Jeffrey P. Murray
    30 Sep 1991     Jeffrey P. Murray

SUMMARY

    This function obtains the next token from an input stream.

INTERFACES       

    FILE                 ROUTINE CALLED     

    N/A                  N/A

RETURNED VALUE
    
    Returns a string value representing the next token. Uses 
    *CNVget_tok.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== Static CNVget_token ROUTINE =============*/
/*
Get the next token from the input string together with its type.
The input string pointer
is advanced to the following token and the token from the input
string is copied to malloced storage and a pointer to that storage
is returned.  The original input string is undisturbed.
*/

static char  *CNVget_token(char **s, Cnv_Token_Type_t *type)

{

    char    *ret_str;   /* storage for returned string */

    /* get the token from the input line */

    ret_str = CNVgettok(s);


    /* if no next token, return */

    if(ret_str == NULL) {
        *type = CNV_NO_TOK;
        return(NULL);
    }

    /* else, determine and return token type */

    switch(*ret_str) {

    default:
        *type = CNV_STRING_TOK;
        break;

    }

    return(ret_str);
}




/*==============================================================================

FUNCTION cnv_get_spice_value()

AUTHORS                      

    ???             Bill Kuhn

MODIFICATIONS   

    30 Sep 1991     Jeffrey P. Murray

SUMMARY

    This function takes as input a string token from a SPICE
    deck and returns a floating point equivalent value.

INTERFACES       

    FILE                 ROUTINE CALLED     

    N/A                  N/A

RETURNED VALUE
    
    Returns the floating point value in pointer *p_value. Also
    returns an integer representing successful completion.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== Static CNV_get_spice_value ROUTINE =============*/

/*
   Function takes as input a string token from a SPICE
deck and returns a floating point equivalent value.
*/


static int cnv_get_spice_value(
char    *str,        /* IN - The value text e.g. 1.2K */
double  *p_value )   /* OUT - The numerical value     */
{


    /* the following were "int4" devices - jpm */
    size_t len;
    size_t i;
    int    n_matched;

    line_t  val_str;

    /*char    *suffix;*/
    char    c = ' ';
    char    c1;

    double  scale_factor;
    double  value;


    /* Scan the input string looking for an alpha character that is not  */
    /* 'e' or 'E'.  Such a character is assumed to be an engineering     */
    /* suffix as defined in the Spice 2G.6 user's manual.                */

    len = strlen(str);
    if( len > (sizeof(val_str) - 1))
        len = sizeof(val_str) - 1;

    for(i = 0; i < len; i++) {
        c = str[i];
        if( isalpha_c(c) && (c != 'E') && (c != 'e') )
            break;
        else if( isspace_c(c) )
            break;
        else
            val_str[i] = c;
    }
    val_str[i] = '\0';


    /* Determine the scale factor */

    if( (i >= len) || (! isalpha_c(c)) )
        scale_factor = 1.0;
    else {

        if(islower_c(c))
            c = tolower_c(c);

        switch(c) {

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

        case 'a':
            scale_factor = 1.0e-18;
            break;

        case 'm':
            i++;
            if(i >= len) {
                scale_factor = 1.0e-3;
                break;
            }
            c1 = str[i];
            if(! isalpha_c(c1)) {
                scale_factor = 1.0e-3;
                break;
            }
            if(islower_c(c1))
                c1 = toupper_c(c1);
            if(c1 == 'E')
                scale_factor = 1.0e6;
            else if(c1 == 'I')
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

    n_matched = sscanf(val_str,"%le",&value);

    if(n_matched < 1) {
        *p_value = 0.0;
        return(FAIL);
    }

    *p_value = value * scale_factor;
    return(OK);
}


/*==============================================================================

FUNCTION cm_inputs_mask_and_retrieve()

AUTHORS                      

    23 Jul 1991     Jeffrey P. Murray

MODIFICATIONS   

    30 Sep 1991     Jeffrey P. Murray

SUMMARY

    Masks off and retrieves a two-bit value from a short
    integer word passed to the function, using an offset value. 
    This effectively handles retrieval of eight two-bit values 
    from a single short integer space in order to conserve memory.

INTERFACES       

    FILE                 ROUTINE CALLED     
 
    N/A                  N/A


RETURNED VALUE
    
    Returns a Digital_t value.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/
             
/*=== Static CM_INPUTS_MASK_AND_RETRIEVE ROUTINE ===*/

/**************************************************
*      The following routine masks and retrieves  *
*   the value passed to it by the out value       *
*   by masking the appropriate bits in the        *
*   base integer. The particular bit affected     *
*   is determined by the bit_offset value.        *
*                                                 *
*   Created 7/23/91               J.P.Murray      *
**************************************************/

static Digital_State_t cm_inputs_mask_and_retrieve(short base, int bit_offset)
{
                                                              

    int value;    /* the hexadecimal value of the masked bit  */


    value = 0x0003 & (base >> (bit_offset * 2));


    switch (value) {
    case 0:  return ZERO;
    case 1:  return ONE;
    default: return UNKNOWN;
    }

}


/*==============================================================================

FUNCTION cm_set_indices()

AUTHORS                      

    23 Jul 1991     Jeffrey P. Murray

MODIFICATIONS   

    30 Sep 1991     Jeffrey P. Murray

SUMMARY

    The following routine retrieves the current_state value 
    from the *states structure. It then searches the entire      
    state[] array to determine the index0 and indexN values 
    corresponding to the first and last entries in the 
    state[] array for the current state.

INTERFACES       

    FILE                 ROUTINE CALLED     
 
    N/A                  N/A


RETURNED VALUE
    
    Returns a status int value.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== Static CM_SET_INDICES ROUTINE ===*/

/************************************************
*      The following routine retrieves the      *
*   current_state value from the *states        *
*   structure. It then searches the entire      *
*   state[] array to determine the index0 and   *
*   indexN values corresponding to the first    *
*   and last entries in the state[] array for   *
*   the current state.                          *         
*                                               * 
*   Created 7/23/91               J.P. Murray   *
************************************************/


static int cm_set_indices(Dynamic_State_t *states, State_Table_t *table)
{             
    int i,           /* indexing variable */
        index0_set,  /* flag for index0   */
        indexN_set;  /* flag for index1   */


    states->index0 = 0;
    states->indexN = 0;
    index0_set = 0;
    indexN_set = 0;

    for (i = 0; i < table->depth; i++) {
        /* Loop through all states. */

        if (table->state[i] == states->current_state) {
            /* states match... */

            /* if not already set, set index0... */
            if ( 0 == index0_set ) {
                states->index0 = i;
                states->indexN = i;
                index0_set = 1;
            }
                              
            if ( 1 < (i - states->indexN) ) { 
                /* ERROR!! This means that a state match was
                   found in a nonn-contiguous portion of the
                   state.in file...this is not allowed! */
                return TRUE;
            }
            else {
                /* update indexN. */
                states->indexN = i; 
            }
        }
    } 
    return FALSE;
}                                    


/*==============================================================================

FUNCTION cm_compare_to_inputs()

AUTHORS                      

    23 Jul 1991     Jeffrey P. Murray

MODIFICATIONS   

    30 Sep 1991     Jeffrey P. Murray

SUMMARY

    The following routine retrieves the      
    the inputs[] bit value pointed to by        
    the word_num and bit_num integers, compares 
    this value with the "in" state value, and   
    returns a "0" if they match, and a "1" if   
    they do not match.                          

INTERFACES       

    FILE                 ROUTINE CALLED     
 
    N/A                  N/A


RETURNED VALUE
    
    Returns a status int value.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== Static CM_COMPARE_TO_INPUTS ROUTINE ===*/

/************************************************
*      The following routine retrieves the      *
*   the inputs[] bit value pointed to by        *
*   the word_num and bit_num integers, compares *
*   this value with the "in" state value, and   *
*   returns a "0" if they match, and a "1" if   *
*   they do not match.                          *
*                                               *
*   Created 7/23/91               J.P.Murray    *
************************************************/


static int cm_compare_to_inputs(State_Table_t *table, int index,
                                int bit_number, Digital_State_t in)
{
    int      int1,      /* temp storage variable    */
        bit_index,      /* bits base address at which word bits will
                           be found */
       bit_offset;      /* offset from ram base address at which bit[0]
                           of the required word can be found    */
        
    short    base;      /* variable to hold current base integer for
                           comparison purposes. */
                                        
    Digital_State_t     out;      /* output variable for state retrieved   */


    /* obtain offset value from index, word_width & bit_number */
    int1 = index * table->num_inputs + bit_number;

    bit_index = int1 >> 3;
    bit_offset = int1 & 7;

    /* retrieve entire base_address bits integer... */
    base = table->inputs[bit_index];
                         
    /* for each offset, mask off the bits and determine values */

    out = cm_inputs_mask_and_retrieve(base, bit_offset);

    if ( out == in ) {  /* bit matches */
        return 0;    
    }
    else 
    if ( out == UNKNOWN ) {     /* bit compared to is a "don't care" */
        return 0;    
    }
    else {                    /* no match */
        return 1;
    }
}                                    


/*==============================================================================

FUNCTION cm_inputs_mask_and_store()

AUTHORS                      

    22 Jul 1991     Jeffrey P. Murray

MODIFICATIONS   

    30 Sep 1991     Jeffrey P. Murray

SUMMARY

    The following routine masks and stores   
    the value passed to it by the out value     
    by masking the appropriate bits in the      
    base integer. The particular bit affected   
    is determined by the bit_offset value.      
    This routine stores to the inputs[] array.  


INTERFACES       

    FILE                 ROUTINE CALLED     
 
    N/A                  N/A


RETURNED VALUE
    
    Returns a status int value.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== Static CM_INPUTS_MASK_AND_STORE ROUTINE ===*/

/************************************************
*      The following routine masks and stores   *
*   the value passed to it by the out value     *
*   by masking the appropriate bits in the      *
*   base integer. The particular bit affected   *
*   is determined by the bit_offset value.      *
*   This routine stores to the inputs[] array.  *
*                                               *
*   Created 7/22/91                             *
*   Last Modified 7/22/91         J.P.Murray    *
************************************************/

static void cm_inputs_mask_and_store(short *base,int bit_offset,int bit_value)
{
    *base &= (short)  ~ (0x0003 << (bit_offset * 2));     
    *base |= (short) (bit_value << (bit_offset * 2));
}




/*==============================================================================

FUNCTION cm_store_inputs_value()

AUTHORS                      

    23 Jul 1991     Jeffrey P. Murray

MODIFICATIONS   

    30 Sep 1991     Jeffrey P. Murray

SUMMARY

    The following routine retrieves four-bit 
    data from short integer array "bits". The   
    integers are assumed to be at least two     
    bytes each, so each will hold four four-    
    bit values.                                 



INTERFACES       

    FILE                 ROUTINE CALLED     
 
    N/A                  N/A


RETURNED VALUE
    
    NONE

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/
             
/*=== Static CM_STORE_INPUTS_VALUE ROUTINE ===*/

/************************************************
*      The following routine retrieves four-bit *
*   data from short integer array "bits". The   *
*   integers are assumed to be at least two     *
*   bytes each, so each will hold four four-    *
*   bit values.                                 *
*                                               *
*   Created 7/23/91               J.P.Murray    *
************************************************/

static void cm_store_inputs_value(State_Table_t *table, int index,
                                  int bit_number, int in_val)

{
    int       /*err,*/      /* error index value    */
             int1,      /* temp storage variable    */
        bit_index,      /* bits base address at which word bits will
                           be found */
       bit_offset;      /* offset from ram base address at which bit[0]
                           of the required word can be found    */
        
    short    base;      /* variable to hold current base integer for
                           comparison purposes. */

    /* obtain offset value from word_number, word_width & bit_number */

    int1 = index * table->num_inputs + bit_number;

    bit_index = int1 >> 3;
    bit_offset = int1 & 7;

    /* retrieve entire base_address bits integer... */
    base = table->inputs[bit_index];
                         
    /* for each offset, mask off the bits and store values */
    cm_inputs_mask_and_store(&base,bit_offset,in_val);
                          
    /* store modified base value */
    table->inputs[bit_index] = base;
}


/*==============================================================================

FUNCTION cm_bits_mask_and_store()

AUTHORS                      

    22 Jul 1991     Jeffrey P. Murray

MODIFICATIONS   

    30 Sep 1991     Jeffrey P. Murray

SUMMARY

      The following routine masks and stores   
   the value passed to it by the out value     
   by masking the appropriate bits in the      
   base integer. The particular bit affected   
   is determined by the bit_offset value.      
   This routine stores to the bits[] array.    


INTERFACES       

    FILE                 ROUTINE CALLED     
 
    N/A                  N/A


RETURNED VALUE
    
    Returns a status integer.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/


/*=== Static CM_BITS_MASK_AND_STORE ROUTINE ===*/

/************************************************
*      The following routine masks and stores   *
*   the value passed to it by the out value     *
*   by masking the appropriate bits in the      *
*   base integer. The particular bit affected   *
*   is determined by the bit_offset value.      *
*   This routine stores to the bits[] array.    *
*                                               *
*   Created 7/22/91                             *
*   Last Modified 7/22/91         J.P.Murray    *
************************************************/

static void cm_bits_mask_and_store(short *base,int bit_offset,int bit_value)
{
    *base &= (short)  ~ (0x000f << (bit_offset * 4));
    *base |= (short) (bit_value << (bit_offset * 4));
}




/*==============================================================================

FUNCTION cm_bits_mask_and_retrieve()

AUTHORS                      

    22 Jul 1991     Jeffrey P. Murray

MODIFICATIONS   

    30 Sep 1991     Jeffrey P. Murray

SUMMARY

    The following routine masks and retrieves  
    the value passed to it by the out value       
    by masking the appropriate bits in the        
    base integer. The particular bit affected    
    is determined by the bit_offset value.        
 


INTERFACES       

    FILE                 ROUTINE CALLED     
 
    N/A                  N/A


RETURNED VALUE
    
    NONE

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== Static CM_BITS_MASK_AND_RETRIEVE ROUTINE ===*/

/**************************************************
*      The following routine masks and retrieves  *
*   the value passed to it by the out value       *
*   by masking the appropriate bits in the        *
*   base integer. The particular bit affected     *
*   is determined by the ram_offset value.        *
*                                                 *
*   Created 7/22/91                               *
*   Last Modified 7/22/91         J.P.Murray      *
**************************************************/

static void cm_bits_mask_and_retrieve(short base,int bit_offset,Digital_t *out)
{
                                                              

    int value;    /* the hexadecimal value of the masked bit  */

    value = 0x000f & (base >> (bit_offset * 4));


    switch (value) {

    case 0: out->state = ZERO;
            out->strength = STRONG;
            break;

    case 1: out->state = ONE;
            out->strength = STRONG;
            break;

    case 2: out->state = UNKNOWN;
            out->strength = STRONG;
            break;

    case 3: out->state = ZERO;
            out->strength = RESISTIVE;
            break;

    case 4: out->state = ONE;
            out->strength = RESISTIVE;
            break;

    case 5: out->state = UNKNOWN;
            out->strength = RESISTIVE;
            break;

    case 6: out->state = ZERO;
            out->strength = HI_IMPEDANCE;
            break;

    case 7: out->state = ONE;
            out->strength = HI_IMPEDANCE;
            break;

    case 8: out->state = UNKNOWN;
            out->strength = HI_IMPEDANCE;
            break;

    case 9: out->state = ZERO;
            out->strength = UNDETERMINED;
            break;

    case 10: out->state = ONE;
            out->strength = UNDETERMINED;
            break;

    case 11: out->state = UNKNOWN;
            out->strength = UNDETERMINED;
            break;
    }
}


/*==============================================================================

FUNCTION cm_get_bits_value()

AUTHORS                      

    22 Jul 1991     Jeffrey P. Murray

MODIFICATIONS   

    23 Jul 1991     Jeffrey P. Murray
    30 Sep 1991     Jeffrey P. Murray

SUMMARY

    The following routine retrieves four-bit 
    data from short integer array "bits". The   
    integers are assumed to be at least two     
    bytes each, so each will hold four four-    
    bit values.                                 
 


INTERFACES       

    FILE                 ROUTINE CALLED     
 
    N/A                  N/A


RETURNED VALUE
    
    NONE

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== Static CM_GET_BITS_VALUE ROUTINE ===*/

/************************************************
*      The following routine retrieves four-bit *
*   data from short integer array "bits". The   *
*   integers are assumed to be at least two     *
*   bytes each, so each will hold four four-    *
*   bit values.                                 *
*                                               *
*   Created 7/22/91                             *
*   Last Modified 7/23/91         J.P.Murray    *
************************************************/

static void cm_get_bits_value(State_Table_t *table, int index,
                              int bit_number, Digital_t *out)

{
    int       /*err,*/      /* error index value    */
             int1,      /* temp storage variable    */
        bit_index,      /* bits base address at which word bits will
                           be found */
       bit_offset;      /* offset from ram base address at which bit[0]
                           of the required word can be found    */
        
    short    base;      /* variable to hold current base integer for
                           comparison purposes. */
                                        


    /* obtain offset value from index, word_width & bit_number */
    int1 = index * table->num_outputs + bit_number;

    bit_index = int1 >> 2;
    bit_offset = int1 & 3;

    /* retrieve entire base_address bits integer... */
    base = table->bits[bit_index];
                         
    /* for each offset, mask off the bits and determine values */

    cm_bits_mask_and_retrieve(base,bit_offset,out);

}



/*==============================================================================

FUNCTION cm_store_bits_value()

AUTHORS                      

    23 Jul 1991     Jeffrey P. Murray

MODIFICATIONS   

    30 Sep 1991     Jeffrey P. Murray

SUMMARY

       The following routine retrieves four-bit 
    data from short integer array "bits". The   
    integers are assumed to be at least two     
    bytes each, so each will hold four four-    
    bit values.                                 
 


INTERFACES       

    FILE                 ROUTINE CALLED     
 
    N/A                  N/A


RETURNED VALUE
    
    NONE

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== Static CM_STORE_BITS_VALUE ROUTINE ===*/

/************************************************
*      The following routine retrieves four-bit *
*   data from short integer array "bits". The   *
*   integers are assumed to be at least two     *
*   bytes each, so each will hold four four-    *
*   bit values.                                 *
*                                               *
*   Created 7/23/91                             *
*   Last Modified 7/23/91         J.P.Murray    *
************************************************/

static void cm_store_bits_value(State_Table_t *table, int index,
                                int bit_number, int in_val)

{
    int       /*err,*/      /* error index value    */
             int1,      /* temp storage variable    */
        bit_index,      /* bits base address at which word bits will
                           be found */
       bit_offset;      /* offset from ram base address at which bit[0]
                           of the required word can be found    */
        
    short    base;      /* variable to hold current base integer for
                           comparison purposes. */
                                        

    /* obtain offset value from word_number, word_width & 
       bit_number */
    int1 = index * table->num_outputs + bit_number;

    bit_index = int1 >> 2;
    bit_offset = int1 & 3;

    /* retrieve entire base_address bits integer... */
    base = table->bits[bit_index];
                         
    /* for each offset, mask off the bits and store values */
    cm_bits_mask_and_store(&base,bit_offset,in_val);
                          
    /* store modified base value */
    table->bits[bit_index] = base;
}



/*==============================================================================

FUNCTION cm_read_state_file()

AUTHORS                      

    15 Jul 1991     Jeffrey P. Murray

MODIFICATIONS   
      
    23 Jul 1991     Jeffrey P. Murray
    30 Sep 1991     Jeffrey P. Murray

SUMMARY

       The following routine reads the file       
    *state_file, parses the file, and stores      
    the values found there into the *state,       
    *bits, *inputs and *next_state arrays.        
 


INTERFACES       

    FILE                 ROUTINE CALLED     
 
    N/A                  N/A


RETURNED VALUE
    
    Returns a status integer value.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== Static CM_READ_STATE_FILE ROUTINE ===*/
                                     

/**************************************************
*      The following routine reads the file       *
*   *state_file, parses the file, and stores      *
*   the values found there into the *state,       *
*   *bits, *inputs and *next_state arrays.        *
*                                                 *
*   Created 7/15/91                               *
*   Last Modified 7/23/91         J.P.Murray      *
**************************************************/

static int cm_read_state_file(FILE *state_file, State_Table_t *table)
{
    int         i,  /* indexing variable    */
                j,  /* indexing variable    */
       num_tokens,  /* number of tokens in a given string    */
        /*bit_index,*/  /* index to which bits[] integer we are accessing   */
       /*bit_offset,*/  /* index to which bit within the current bits[] 
                       integer we are accessing   */
      string_type;  /* integer holding value corresponding to the
                       type of input string obtained:
                           1 = HEADER => State Header string
                           2 = CONTINUATION => a continuation line...
                                               values of state and
                                               bits must be retreived from
                                               the previous string. */
             /*int1;*/  /* temporary holding variable   */


    Cnv_Token_Type_t type;  /* variable for testing token type returned.    */


    char   temp[MAX_STRING_SIZE],   /* holding string variable for testing
                                       input from state.in */
                              *s,   /* main string variable */
                   *base_address,   /* storage location for base address
                                       of string.   */    
                          *token;   /* a particular token from the string   */
   

    double                number;   /* holding variable for timepoint values */                                             
                                                 
    short              bit_value=0;   /* holding variable for value read from
                                       state.in file which needs to be stored */
                            /*base;*/   /* holding variable for existing 
                                       non-masked bits[] integer  */
	if (!state_file) {
		return 2;
	}                 

    i = 0;                                 
    s = temp;
    while ( fgets(s,MAX_STRING_SIZE,state_file) != NULL) {
        /* Test this string to see if it is whitespace... */

        base_address = s;
        while(isspace_c(*s) || (*s == '*')) 
              (s)++;
        if ( *s != '\0' ) {     /* This is not a blank line, so process... */
            s = base_address;

            if ( '*' != s[0] ) {
        
                /* Count up total number of tokens including \0... */
                j = 0;
                type = CNV_STRING_TOK;
                while ( type != CNV_NO_TOK ) {
                    token = CNVget_token(&s, &type);
		    if (token)
		       free(token);
                    j++;
                }
                num_tokens = (j-1);
    
                /* Test the type of entry this number of
                   tokens represents. Valid types are:
    
                        a. State Header with state, bits, inputs
                           and next_state entries...total tokens
                           equals num_inputs + num_outputs + 3
                           (e.g. "5 0s 0s 0s 0 0 -> 6").                             
    
                        b. State continuation line with inputs
                           and next_state only...total tokens
                           equals num_inputs + 2.
                           (e.g. "           0 1 -> 7").
                */
    
                if (3 + table->num_inputs + table->num_outputs == num_tokens) {
                    string_type = HEADER;
                }
                else {
                    if ( (2 + table->num_inputs) == num_tokens) {
                        string_type = CONTINUATION;
                    }
                    else { /* Number of tokens is incorrect */
                        return 1;
                    }
                }
    
                /* reset s to beginning... */                     
                s = base_address;
        
                /** Retrieve each token, analyze, and       **/
                /** store the state, bits, inputs &         **/
                /** next_state information.                 **/
    
                if ( HEADER == string_type ) { /**** header type loop ****/
    
                    for (j = 0;
                         j < table->num_inputs + table->num_outputs + 3;
                         j++) {
            
                        token = CNVget_token(&s, &type);

                        if(!token)
                            return 1;

                        if ( 0 == j ) { /* obtain state value... */
                            
                            /* convert to a floating point number... */
                            cnv_get_spice_value(token,&number);
            
                            table->state[i] = (int)number;

                             
                        }
                        else { /* obtain each bit value & set bits entry    */
            
                            if ( table->num_outputs >= j ) {
        
                                /* preset this bit location */
                                bit_value = 12;
                 
                                if (0 == strcmp(token,"0s")) bit_value = 0;
                                if (0 == strcmp(token,"1s")) bit_value = 1;
                                if (0 == strcmp(token,"Us")) bit_value = 2;
                                if (0 == strcmp(token,"0r")) bit_value = 3;
                                if (0 == strcmp(token,"1r")) bit_value = 4;
                                if (0 == strcmp(token,"Ur")) bit_value = 5;
                                if (0 == strcmp(token,"0z")) bit_value = 6;
                                if (0 == strcmp(token,"1z")) bit_value = 7;
                                if (0 == strcmp(token,"Uz")) bit_value = 8;
                                if (0 == strcmp(token,"0u")) bit_value = 9;
                                if (0 == strcmp(token,"1u")) bit_value = 10;
                                if (0 == strcmp(token,"Uu")) bit_value = 11;
                 
                                /* if this bit was not recognized, return with an error  */
                                if (12 == bit_value) {
                                    free(token);
                                    return 1;                            
                                }
                                else {
                                    /* need to store this value in the bits[] array */
                                    cm_store_bits_value(table, i, j - 1,
                                                        bit_value);
                                }
                            } 
                            else { /* obtain inputs info... */
                                int idx;

                                if (table->num_outputs + table->num_inputs >= j) {
                                    /* preset this bit location */
                                    bit_value = 3;
                     
                                    if (0 == strcmp(token,"0")) bit_value = 0;
                                    if (0 == strcmp(token,"1")) bit_value = 1;
                                    if (0 == strcmp(token,"x")) bit_value = 2;
                                    if (0 == strcmp(token,"X")) bit_value = 2;
                     
                                    /* if this bit was not recognized, return with an error  */
                                    if (3 == bit_value) {
                                        free(token);
                                        return 1;                            
                                    }
                                    else {
                                        /* need to store this value in the inputs[] array */
                                        idx = j - 1 - table->num_outputs;
                                        cm_store_inputs_value(table, i, idx,
                                                              bit_value);
                                    }
                                }
                                else { /* obtain next_state value */
                                    idx =  1 + table->num_outputs +
                                               table->num_inputs;
                                    if (idx == j) {
                                        /* skip the "->" token */
                                    }
                                    else {
                                        /* convert to a floating point number... */
                                        cnv_get_spice_value(token,&number);
            
                                        table->next_state[i] = (int) number;
                                    }
                                }
                            }
                        }
                        free(token);       
                    }
                }
                else { /**** continuation type loop ****/
                    /* set state value to previous state value */

                    table->state[i] = table->state[i - 1];
                                                         
                    /* set bits values to previous bits values */
                    for (j = 0; j < table->num_outputs; j++) {
        
                        /*** Store this bit value ***/
    
                        cm_store_bits_value(table, i, j, bit_value);
                    }
    
                    for (j=0; j<(2 + table->num_inputs); j++) {
            
                        token = CNVget_token(&s, &type);

                        if(!token)
                            return 1;

                        if ( j < table->num_inputs ) {
    
                            /* preset this bit location */
                            bit_value = 3;
             
                            if (0 == strcmp(token,"0")) bit_value = 0;
                            if (0 == strcmp(token,"1")) bit_value = 1;
                            if (0 == strcmp(token,"x")) bit_value = 2;
                            if (0 == strcmp(token,"X")) bit_value = 2;
             
                            /* if this bit was not recognized, return with an error  */
                            if (3 == bit_value) {
                                free(token);
                                return 1;                            
                            }
                            else {
                                /* need to store this value in the inputs[] array */
                                cm_store_inputs_value(table, i, j, bit_value);
                            } 
                        }
                        else { /* obtain next_state value */
                            if ( table->num_inputs == j ) {
                                /* skip the "->" token */
                            }
                            else {
                                /* convert to a floating point number... */

                                cnv_get_spice_value(token,&number);
                                table->next_state[i] = (int) number;
                            }
                        }
                        free(token);
                    } 

                }
                i++;
            }
            s = temp;  
        }
    }
    return 0;
}
            
/* Free allocated storage. */

static void callback(ARGS, Mif_Callback_Reason_t reason)
{
    if (reason == MIF_CB_DESTROY) {
        State_Table_t    *table;

        table = (State_Table_t *)STATIC_VAR(table);
        if (!table)
            return;
        if (table->state)
            free(table->state);
        if (table->bits)
            free(table->bits);
        if (table->inputs)
            free(table->inputs);
        if (table->next_state)
            free(table->next_state);
        free(table);
	STATIC_VAR(table) = NULL;
    }
}
                                        
/*==============================================================================

FUNCTION cm_d_state()

AUTHORS                      

    17 Jun 1991     Jeffrey P. Murray

MODIFICATIONS   

    20 Aug 1991     Jeffrey P. Murray
    30 Sep 1991     Jeffrey P. Murray

SUMMARY

    This function implements the d_state code model.

INTERFACES       

    FILE                 ROUTINE CALLED     

    CMmacros.h           cm_message_send();                   
                             
    CMevt.c              void *cm_event_alloc()
                         void *cm_event_get_ptr()

RETURNED VALUE
    
    Returns inputs and outputs via ARGS structure.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== CM_D_STATE ROUTINE ==============*/

/************************************************
*      The following is the model for the       *
*   digital state machine for the               *
*   ATESSE Version 2.0 system.                  *
*                                               *
*   Created 7/17/91                             *
*   Last Modified 8/20/91         J.P.Murray    *
************************************************/


void cm_d_state(ARGS) 
{
    int               i, j,       /* Generic loop counter indices. */
                      err=0,      /* Integer for storage of error status.  */
                      test;       /* Testing integer.  */
               
    State_Table_t    *table;      /* Pointer to base address structure
                                     for all state arrays. *table
                                     contains the pointers to
                                     state[], bits[], input[]
                                     and next_state[] arrays. These
                                     arrays are allocated using
                                     calloc, so they do not take up
                                     rotational storage space...only
                                     the *states structure does that. */
    Dynamic_State_t  *states,      /* Pointer to dynamic state structure. */
                     *states_old;                                        
    Digital_t         in, out;     /* Storage for each input and  bit. */

    Digital_State_t  *clk,         /* Storage for current and clock value.  */
                     *clk_old,     /* Previous clock value. */
                     *reset    ,   /* Current reset value.  */
                     *reset_old;   /* Previous reset value.  */

    static char      *index_error =
        "\nERROR\n  D_STATE: An error exists in the ordering of states "
        "values\n  in the states->state[] array. This is usually caused\n"
        " by non-contiguous state definitions in the state.in file.\n";

        /****** Setup required state variables ******/

    if (INIT) {  /* initial pass */
        FILE  *state_file; /* Pointer to the state.in input vector file. */
	char   temp[MAX_STRING_SIZE], /* Holding string variable for testing
                                         input from state.in. */
              *s;                     /* Main string variable. */

        static char *open_error =
	    "\nERROR\n  D_STATE: failed to open state file.\n";
        static char *loading_error =
	    "\nERROR\n  D_STATE: state file was not read successfully.\n  "
	    "The most common cause of this problem is a\n  "
	    "trailing blank line in the state.in file.\n";

	/* Allocate storage for the state transition table. */

        table = calloc(1, sizeof (State_Table_t));
        STATIC_VAR(table) = table;

        /*** open file and count the number of vectors in it ***/
        state_file = fopen_with_path( PARAM(state_file), "r");

        /* increment counter if not a comment until EOF reached... */
        i = 0;                                                
        s = temp;
        if (state_file != NULL)
            while (fgets(s, MAX_STRING_SIZE, state_file) != NULL) {
                if ( '*' != s[0] ) {
                    while (isspace_c(*s) || (*s == '*'))
                        (s)++;
                    if (*s != '\0')
                        i++;
                }
                s = temp;
            }

        /* Store depth value */
        table->depth = i;

        /* Retrieve widths for bits[] and inputs[] */
        table->num_inputs = PORT_SIZE(in);
        table->num_outputs = PORT_SIZE(out);
        
        /* Assign storage for arrays to pointers in state table. */
        table->state = (int *)calloc((size_t) (table->depth + 1), sizeof(int));
        table->bits =(short *)calloc(
                         (size_t)(table->num_outputs * table->depth / 4 + 1),
                         sizeof (short));
        table->inputs = (short *)calloc(
                            (size_t)(table->num_inputs * table->depth / 8 + 1),
                            sizeof (short));
        table->next_state = (int *)calloc((size_t)(table->depth + 1),
                                          sizeof (int));
        CALLBACK = callback;    // To free those allocations.

        /*** allocate storage for *states... ***/

        cm_event_alloc(0, sizeof (Dynamic_State_t));

        /*** allocate storage for *clk, *clk_old, *reset & *reset_old... ***/
        cm_event_alloc(1, sizeof (Digital_State_t));
        cm_event_alloc(2, sizeof (Digital_State_t));

        clk = clk_old = (Digital_State_t *)cm_event_get_ptr(1, 0);
        *clk = ZERO;
        reset = reset_old = (Digital_State_t *)cm_event_get_ptr(2, 0);
        *reset = ZERO;

        /*** Send file pointer and the four storage pointers      ***/
        /*** to "cm_read_state_file()". This will return after    ***/
        /*** reading the contents of state.in, and if no          ***/
        /*** errors have occurred, the "*state" and "*bits"       ***/
        /*** "*inputs", and "*next_state" vectors will be loaded  ***/
        /*** and the num_inputs, num_outputs and depth            ***/
        /*** values supplied.                                     ***/

        if (state_file) {
            rewind(state_file);
            err = cm_read_state_file(state_file, table);
        } else {
            err = 2;
        }

        if (err == 1) /* Problem occurred in load. */
            cm_message_send(loading_error);
        else if (err == 2)  /* Problem opening the state file. */
            cm_message_send(open_error);

        if (err > 0) {
            /* Reset arrays to zero */
            for (i = 0; i < table->depth; i++) {
                table->state[i] = 0;
                table->next_state[i] = 0;
            }        
            for (i=0; i<(test=(table->num_outputs * table->depth)/4); i++)
                table->bits[i] = 0;
            for (i=0; i<(test=(table->num_inputs * table->depth)/8); i++)
                table->inputs[i] = 0;
            return;
        }

        /* close state_file */
        if (state_file)
            fclose(state_file);

        /* declare load values */
       
        for (i = 0; i < table->num_inputs; i++) {
            LOAD(in[i]) = PARAM(input_load);
        }              

        LOAD(clk) = PARAM(clk_load);

        if ( !PORT_NULL(reset) ) {
            LOAD(reset) = PARAM(reset_load);
        }
    }
    else {  /**** Retrieve previous values ****/

        table = (State_Table_t *)STATIC_VAR(table);
        clk = (Digital_State_t *)cm_event_get_ptr(1, 0);
        clk_old = (Digital_State_t *)cm_event_get_ptr(1, 1);

        reset = (Digital_State_t *)cm_event_get_ptr(2, 0);
        reset_old = (Digital_State_t *)cm_event_get_ptr(2, 1);
    }

    states = (Dynamic_State_t *)cm_event_get_ptr(0, 0);
    states_old = (Dynamic_State_t *)cm_event_get_ptr(0, 1);

    /******* Determine analysis type and output appropriate values *******/

    if (0.0 == TIME) {   /****** DC analysis...output w/o delays ******/
        /* set current state to default */
        states->current_state = PARAM(reset_state);
        
        /* set indices for this state   */
        err = cm_set_indices(states, table);
        if (err == TRUE) {
            cm_message_send(index_error);
            return;
        }

        /* Output new values... */
        for (i=0; i < table->num_outputs; i++) {

            /* retrieve output value */
            cm_get_bits_value(table,states->index0,i,&out);

            OUTPUT_STATE(out[i]) = out.state;
            OUTPUT_STRENGTH(out[i]) = out.strength;
        }
    }

    else {      /****** Transient Analysis ******/

        /*** load current input values if reset 
             is not connected, set to zero... ***/
        *clk = INPUT_STATE(clk);

        if ( PORT_NULL(reset) ) {
            *reset = *reset_old = ZERO;
        }
        else {
            *reset = INPUT_STATE(reset);
        }

        /***** Find input that has changed... *****/

        /**** Test reset value for change ****/
        if ( *reset != *reset_old ) { /* either reset or reset release */
            switch ( *reset ) {

            case ONE:
                states->current_state = PARAM(reset_state);

                /* set indices for this state   */
                err = cm_set_indices(states, table);
                if ( err == TRUE ) {
                    cm_message_send(index_error);
                    return;
                }

                /* Output new values... */
                for (i=0; i<table->num_outputs; i++) {
        
                    /* retrieve output value */
                    cm_get_bits_value(table, states->index0, i, &out);
        
                    OUTPUT_STATE(out[i]) = out.state;
                    OUTPUT_STRENGTH(out[i]) = out.strength;
                    OUTPUT_DELAY(out[i]) = PARAM(reset_delay);
                }
                break;

            case ZERO:
            case UNKNOWN:
                /* output remains at current value */
                for (i=0; i<table->num_outputs; i++) {
                    OUTPUT_CHANGED(out[i]) = FALSE;
                }
                break;

            }
        }
        else {
          
            if ( ONE != *reset ) {
                /**** model is not held to reset value ****/  
                /**** Test clk value for change ****/
                if (*clk != *clk_old) { /* clock or clock release */
                    switch ( *clk ) {          
    
                    case ONE:
    
                        /** active edge...need to find out if a match   **/
                        /**   exists between the current inputs and     **/
                        /**   any of the inputs which drive the state   **/
                        /**   machine to the next state...if so, we     **/
                        /**   will need to retrieve that state value,   **/
                        /**   and set the new index boundaries...       **/
                                                                 
                        /* for each of the entries for the current_state... */
                        for (i=states->index0; i<=states->indexN; i++) {
    
                            /* first set err to zero... */
                            err = 0;
    
                            /* for each bit of the inputs value for this 
                               entry...*/
                            for (j = 0; j < table->num_inputs; j++) {
    
                                /* retrieve the current input bit... */
                                in.state = INPUT_STATE(in[j]);
    
                                /* ...& compare to the corresponding
                                   inputs[i] value. */
                                err = cm_compare_to_inputs(table, i, j,
                                                           in.state);
                                
                                /* break if comparison was no-good... */
                                if (0 != err)
                                    break;
                            }
                                                      
                            /* if the comparison of the entire input
                               word was good, break out of this loop too... */
                            if ( 0 == err ) break;
                        }
                                                                   
                        /* if err == 0, then a match was found...otherwise,
                           we will not change states... */
    
                        if ( 0 == err ) {
                            /* use "i" to retrieve the next state value:
                               store this into current_state... */

                            states->current_state = table->next_state[i];
    
                            /* set indices for this new state   */
                            err = cm_set_indices(states, table);
                            if ( err == TRUE ) {
                                cm_message_send(index_error);
                                return;
                            }
                    
                            /* Output new values... */
                            for (i=0; i<table->num_outputs; i++) {
                                /* retrieve output value */

                                cm_get_bits_value(table, states->index0,
                                                  i, &out);
                                OUTPUT_STATE(out[i]) = out.state;
                                OUTPUT_STRENGTH(out[i]) = out.strength;
                                OUTPUT_DELAY(out[i]) = PARAM(clk_delay);
                            }
    
                        }
                        else { /* no change in state or in output */
    
                            for (i=0; i<table->num_outputs; i++) {
                                OUTPUT_CHANGED(out[i]) = FALSE;
                            }
    
                        }
    
                        break;
    
                    case ZERO:
                    case UNKNOWN:
                        
                        /* no change in state or in output */
                        for (i=0; i<table->num_outputs; i++) {
                            OUTPUT_CHANGED(out[i]) = FALSE;
                        }
    
                        break;
                    }
                } 
                else {   /* input value must have changed...
                            return previous output value.    */

                    for (i=0; i<table->num_outputs; i++) {
                        OUTPUT_CHANGED(out[i]) = FALSE;
                    }
                }
            }
            else {   /* Reset is active...
                        return previous output values.    */

                for (i=0; i<table->num_outputs; i++) {
                    OUTPUT_CHANGED(out[i]) = FALSE;
                }
            }
        }        
    }
}
