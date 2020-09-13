/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE d_ram/cfunc.mod

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503-405
               

AUTHORS                      

    23 Aug 1991     Jeffrey P. Murray


MODIFICATIONS   

    30 Sep 1991    Jeffrey P. Murray
                                   

SUMMARY

    This file contains the model-specific routines used to
    functionally describe the d_ram code model.


INTERFACES       

    FILE                 ROUTINE CALLED     

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

                                      

/*=== CONSTANTS ========================*/




/*=== MACROS ===========================*/



  
/*=== LOCAL VARIABLES & TYPEDEFS =======*/                         


    
           
/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/




                   
/*==============================================================================

FUNCTION cm_address_to_decimal()

AUTHORS                      

    27 Jun 1991     Jeffrey P. Murray

MODIFICATIONS   

     8 Jul 1991     Jeffrey P. Murray
    30 Sep 1991     Jeffrey P. Murray

SUMMARY

    Calculates a decimal value from binary values passed.

INTERFACES       

    FILE                 ROUTINE CALLED     
 
    N/A                  N/A


RETURNED VALUE
    
    A pointer containing the total (*total).

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== CM_ADDRESS_TO_DECIMAL ROUTINE ===*/

/************************************************
*      The following routine calculates a       *
*   decimal value equivalent to the binary      *
*   value passed to it on address[i] bits.      *
*   The determined value is written to the      *
*   integer *total.                             *
*                                               *
*   Created 6/27/91               J.P.Murray    *
************************************************/


static int cm_address_to_decimal(Digital_State_t *address,int address_size,int *total)
{
    int     i,      /* indexing variable    */
   multiplier,      /* binary multiplier value   */
          err;      /* error value: 1 => output is unknown
                                    0 => output is valid    */

    err = 0;
    *total = 0;
    multiplier = 1;

    for (i=0; i<address_size; i++) {
        if ( UNKNOWN == address[i] ) {
            err = 1; 
            break;
        }
        else {
            if (address[i] == ONE) {
                *total += multiplier;  
            }
        }
        multiplier *= 2;
    }   

    return err;                                 
}



/*==============================================================================

FUNCTION cm_mask_and_store()

AUTHORS                      

     8 Jul 1991     Jeffrey P. Murray

MODIFICATIONS   

    30 Sep 1991     Jeffrey P. Murray

SUMMARY

    Masks and stores a two-bit value into a passed short pointer,
    using an offset value. This effectively handles storage of
    eight two-bit values into a single short integer space in
    order to conserve memory.

INTERFACES       

    FILE                 ROUTINE CALLED     
 
    N/A                  N/A


RETURNED VALUE
    
    Returns updated *base value.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== CM_MASK_AND_STORE ROUTINE ===*/

/************************************************
*      The following routine masks and stores   *
*   the value passed to it by the out value     *
*   by masking the appropriate bits in the      *
*   base integer. The particular bit affected   *
*   is determined by the ram_offset value.      *
*                                               *
*   Created 7/8/91                J.P.Murray    *
************************************************/

static void cm_mask_and_store(short *base,int ram_offset,Digital_State_t out)
{
    int val;

    if ( ZERO == out )
        val = 0;
    else if (ONE == out)
        val = 1;
    else
        val = 2;

    *base &= (short) ~ (3 << (ram_offset * 2));
    *base |= (short) (val << (ram_offset * 2));
}





/*==============================================================================

FUNCTION cm_mask_and_retrieve()

AUTHORS                      

     8 Jul 1991     Jeffrey P. Murray

MODIFICATIONS   

    30 Sep 1991     Jeffrey P. Murray

SUMMARY

    This is a companion function to cm_mask_and_store().
    Masks off and retrieves a two-bit value from a short
    integer word passed to the function, using an offset value. 
    This effectively handles retrieval of eight two-bit values 
    from a single short integer space in order to conserve memory.

INTERFACES       

    FILE                 ROUTINE CALLED     
 
    N/A                  N/A


RETURNED VALUE
    
    Returns updated *base value.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== CM_MASK_AND_RETRIEVE ROUTINE ===*/

/**************************************************
*      The following routine masks and retrieves  *
*   the value passed to it by the out value       *
*   by masking the appropriate bits in the        *
*   base integer. The particular bit affected     *
*   is determined by the ram_offset value.        *
*                                                 *
*   Created 7/8/91                 J.P.Murray     *
**************************************************/

static Digital_State_t cm_mask_and_retrieve(short base, int ram_offset)
{
    int value = 0x0003 & (base >> (ram_offset * 2));

    switch (value) {
    case 0:  return ZERO;
    case 1:  return ONE;
    default: return UNKNOWN;
    }
}



/*==============================================================================

FUNCTION cm_initialize_ram()

AUTHORS                      

     9 Jul 1991     Jeffrey P. Murray

MODIFICATIONS   

    30 Sep 1991     Jeffrey P. Murray

SUMMARY

    This function stores digital data into specific short
    integer array locations.

INTERFACES       

    FILE                 ROUTINE CALLED     
 
    N/A                  N/A


RETURNED VALUE
    
    Returns updated ram[] value.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== CM_INITIALIZE_RAM ROUTINE ===*/

/************************************************
*      The following routine stores two-bit     *
*   data into short integer array "ram". The    *
*   integers are assumed to be at least two     *
*   bytes each, so each will hold eight two-    *
*   bit values.                                 *
*                                               *
*   Created 7/9/91               J.P.Murray     *
************************************************/


static void cm_initialize_ram(Digital_State_t out,int word_width,int bit_number,
                        int word_number,short *ram)
{
    int       /*err,*/      /* error index value    */
             int1,      /* temp storage variable    */
             /*int2,*/      /* temp storage variable    */
        ram_index,      /* ram base address at which word bits will
                           be found */
       ram_offset;      /* offset from ram base address at which bit[0]
                           of the required word can be found    */
        
    short    base;      /* variable to hold current base integer for
                           comparison purposes. */


    /* obtain offset value from word_number, word_width & 
       bit_number */
    int1 = word_number * word_width + bit_number;

    ram_index = int1 >> 3;
    ram_offset = int1 & 7;
            
    /* retrieve entire base_address ram integer... */
    base = ram[ram_index];
                         
    /* for each offset, mask off the bits and store values */
    cm_mask_and_store(&base,ram_offset,out);
                          
    /* store modified base value */
    ram[ram_index] = base;                   

}




/*==============================================================================

FUNCTION cm_store_ram_value()

AUTHORS                      

    27 Jun 1991     Jeffrey P. Murray

MODIFICATIONS   

     9 Jul 1991     Jeffrey P. Murray
    30 Sep 1991     Jeffrey P. Murray

SUMMARY

    This function stores digital data into specific short
    integer array locations, after decoding address bits 
    passed to it (using cm_address_to_decimal routine).

INTERFACES       

    FILE                 ROUTINE CALLED     
 
    N/A                  N/A


RETURNED VALUE
    
    Returns updated ram[] value via *ram pointer.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== CM_STORE_RAM_VALUE ROUTINE ===*/

/************************************************
*      The following routine stores two-bit     *
*   data into short integer array "ram". The    *
*   integers are assumed to be at least two     *
*   bytes each, so each will hold eight two-    *
*   bit values. A sister routine, cm_get_       *
*   ram_value is used to retrieve the two-      *
*   bit values from the "ram" array.            *
*                                               *
*   Created 6/27/91               J.P.Murray    *
************************************************/


static void cm_store_ram_value(Digital_State_t out,int word_width,int bit_number,
                        Digital_State_t *address,int address_size,
                        short *ram)
{
    int       err,      /* error index value    */
             int1,      /* temp storage variable    */
      word_number,      /* particular word of interest...this value
                           is derived from the passed address bits  */
        ram_index,      /* ram base address at which word bits will
                           be found */
       ram_offset;      /* offset from ram base address at which bit[0]
                           of the required word can be found    */
        
    short    base;      /* variable to hold current base integer for
                           comparison purposes. */

    /** first obtain word_number from *address values **/
    err = cm_address_to_decimal(address,address_size,&word_number);
                                                
    if ( FALSE == err ) { /** valid data was returned...store value **/     

        /* obtain offset value from word_number, word_width & 
           bit_number */
        int1 = word_number * word_width + bit_number;

        ram_index = int1 >> 3;
        ram_offset = int1 & 7;
            
        /* retrieve entire base_address ram integer... */
        base = ram[ram_index];
                             
        /* for each offset, mask off the bits and store values */
        cm_mask_and_store(&base,ram_offset,out);
                              
        /* store modified base value */
        ram[ram_index] = base;                   

    }
}





/*==============================================================================

FUNCTION cm_get_ram_value()

AUTHORS                      

    27 Jun 1991     Jeffrey P. Murray

MODIFICATIONS   

    30 Sep 1991     Jeffrey P. Murray

SUMMARY

    This function retrieves digital data from specific short
    integer array locations, after decoding address bits 
    passed to it (using cm_address_to_decimal routine). This 
    is a sister routine to cm_store_ram_value.

INTERFACES       

    FILE                 ROUTINE CALLED     
 
    N/A                  N/A


RETURNED VALUE
    
    Returns output value via *out pointer.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== CM_GET_RAM_VALUE ROUTINE ===*/

/************************************************
*      The following routine retrieves two-bit  *
*   data from short integer array "ram". The    *
*   integers are assumed to be at least two     *
*   bytes each, so each will hold eight two-    *
*   bit values. A sister routine, cm_store_     *
*   ram_value is used to store the two-bit      *
*   values into the "ram" array.                *
*                                               *
*   Created 6/27/91               J.P.Murray    *
************************************************/

static Digital_State_t cm_get_ram_value(int word_width,int bit_number,Digital_State_t *address,
                 int address_size,short *ram)

{
    int       err,      /* error index value    */
             int1,      /* temp storage variable    */
      word_number,      /* particular word of interest...this value
                           is derived from the passed address bits  */
        ram_index,      /* ram base address at which word bits will
                           be found */
       ram_offset;      /* offset from ram base address at which bit[0]
                           of the required word can be found    */
        
    short    base;      /* variable to hold current base integer for
                           comparison purposes. */
                                        

    /** first obtain word_number from *address values **/
    err = cm_address_to_decimal(address,address_size,&word_number);
                                                
    if ( FALSE == err ) { /** valid data was returned **/     

        /* obtain offset value from word_number, word_width & 
           bit_number */
        int1 = word_number * word_width + bit_number;

        ram_index = int1 >> 3;
        ram_offset = int1 & 7;
                
        /* retrieve entire base_address ram integer... */
        base = ram[ram_index];
                             
        /* for each offset, mask off the bits and determine values */

        return cm_mask_and_retrieve(base,ram_offset);

    }
    else { /** incorrect data returned...return UNKNOWN values  **/
        return UNKNOWN;
    }
}




/*==============================================================================

FUNCTION cm_d_ram()

AUTHORS                      

    26 Jun 1991     Jeffrey P. Murray

MODIFICATIONS   

    30 Sep 1991     Jeffrey P. Murray

SUMMARY

    This function implements the d_ram code model.

INTERFACES       

    FILE                 ROUTINE CALLED     
 
    CMevt.c              void *cm_event_alloc()
                         void *cm_event_get_ptr()


RETURNED VALUE
    
    Returns inputs and outputs via ARGS structure.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== CM_D_RAM ROUTINE ===*/

/************************************************
*      The following is the model for the       *
*   digital M x N random access memory for the  *
*   ATESSE Version 2.0 system.                  *
*                                               *
*   Created 6/26/91               J.P.Murray    *
************************************************/


void cm_d_ram(ARGS) 

{
    int                    i,   /* generic loop counter index   */
                           j,   /* generic loop counter index   */
                    ram_size,   /* total number of words in ram */
             num_of_ram_ints,   /* actual number of "short" integer
                                   values necessary to store all of the
                                   ram_size integers */
                address_size,   /* total number of address lines    */
                 select_size,   /* total number of address lines    */
                select_value,   /* decimal value compared with 
                                   select inputs to confirm that the
                                   ram has indeed been activated    */
                  word_width,   /* width of each word in bits   */
                    int_test,   /* integer test variable        */
             address_changed,   /* TRUE if address is different from
                                   that on the previous call...FALSE
                                   otherwise    */
             address_unknown;   /* TRUE if currently-read address has
                                   at least one line which is an unknown
                                   value.   */

    short int           *ram,   /* storage words...note that the 
                                   total ram data will be stored in the
                                   two-bytes-per-Digital_State_t...since
                                   we require 2 bits per ram bit (for
                                   ZERO, ONE & UNKNOWN), we will store
                                   8 ram bits per Digital_State_t
                                   location   */
                    *ram_old;   /* previous values of storage words */

                        

    Digital_State_t     *address,   /* address line values  */
                    *address_old,   /* previous address line values */
                       *write_en,   /* write_en value    */
                   *write_en_old,   /* previous write_en value   */
                         *select,   /* current selected state of ram...
                                       note that this is derived from all
                                       of the select lines into the ram */
                     *select_old,   /* previous selected state of ram   */
                          d_test,   /* digital debugging variable       */
                             out;   /* current output bit   */





    /** retrieve device size values...& other parameters **/
    address_size = PORT_SIZE(address);    

    /* calculate ram word size from address size */
    ram_size = 1;
    for (i=0; i<address_size; i++) {
        ram_size *= 2;
    }

    select_size = PORT_SIZE(select);
    select_value = PARAM(select_value);
    word_width = PORT_SIZE(data_in);
                                                        

    num_of_ram_ints = (ram_size * word_width + 7) / 8;




    /*** Setup required state variables ***/
    if(INIT) {  /* initial pass */ 

        /* allocate storage */
        cm_event_alloc(0, address_size * (int) sizeof(Digital_State_t));
        cm_event_alloc(1, (int) sizeof(Digital_State_t));
        cm_event_alloc(2, select_size * (int) sizeof(Digital_State_t));


        /* allocate storage for ram memory */
        cm_event_alloc(3, num_of_ram_ints * (int) sizeof(short));

        /* declare load values */
        for (i=0; i<word_width; i++) {
            LOAD(data_in[i]) = PARAM(data_load);
        }

        for (i=0; i<address_size; i++) {
            LOAD(address[i]) = PARAM(address_load);
        }                        

        LOAD(write_en) = PARAM(enable_load);

        for (i=0; i<select_size; i++) {
            LOAD(select[i]) = PARAM(select_load);
        }

        /* retrieve storage for the outputs */
        address = address_old = (Digital_State_t *) cm_event_get_ptr(0,0);
        write_en = write_en_old = (Digital_State_t *) cm_event_get_ptr(1,0);
        select = select_old = (Digital_State_t *) cm_event_get_ptr(2,0);

        /* retrieve ram base addresses */
        ram = ram_old = (short *) cm_event_get_ptr(3,0);

    }
    else {      /* Retrieve previous values */
                                              
        /* retrieve storage for the outputs */
        address = (Digital_State_t *) cm_event_get_ptr(0,0);
        address_old = (Digital_State_t *) cm_event_get_ptr(0,1);
        write_en = (Digital_State_t *) cm_event_get_ptr(1,0);
        write_en_old = (Digital_State_t *) cm_event_get_ptr(1,1);
        select = (Digital_State_t *) cm_event_get_ptr(2,0);
        select_old = (Digital_State_t *) cm_event_get_ptr(2,1);

        /* retrieve ram base addresses */
        ram = (short *) cm_event_get_ptr(3,0);
        ram_old = (short *) cm_event_get_ptr(3,1);

        for(i=0;i<num_of_ram_ints;i++)
          ram[i] = ram_old[i];
    }
                                      


     

    /**** retrieve inputs ****/
    *write_en = INPUT_STATE(write_en);

    address_changed = FALSE;
    address_unknown = FALSE;
    for (i=0; i<address_size; i++) {
        address[i] = INPUT_STATE(address[i]);
        if (UNKNOWN == address[i]) address_unknown = TRUE;
        if (address[i] != address_old[i]) address_changed = TRUE; 
    }


    /** Determine whether we are selected or not... **/
    
    /* retrieve the bit equivalents for the select lines */

    *select = ONE;
    for (i=0; i<select_size; i++) { /* compare each bit in succession
                                       with the value of each of the 
                                       select input lines.          */
        if ( (d_test = INPUT_STATE(select[i])) != (int_test = (select_value & 0x0001)) ) {
            *select = ZERO;
        }
        /* shift the values select_value bits for next compare... */
        select_value = select_value>>1;
    }
                                                  

        
         



    /******* Determine analysis type and output appropriate values *******/

    if (0.0 == TIME) {   /****** DC analysis...output w/o delays ******/
                                  
        /** initialize ram to ic value **/

        out = (Digital_State_t) PARAM(ic);
        for (i=0; i<word_width; i++) {
            for (j=0; j<ram_size; j++) {
                cm_initialize_ram(out,word_width,i,j,ram);
            }
        }

        if ( (ONE==*select) && (ZERO==*write_en) ) {
            /* output STRONG initial conditions */
            for (i=0; i<word_width; i++) {
                OUTPUT_STATE(data_out[i]) = (Digital_State_t) PARAM(ic);
                OUTPUT_STRENGTH(data_out[i]) = STRONG;
            }
        }
        else { /* change output to high impedance */
            for (i=0; i<word_width; i++) {
                OUTPUT_STATE(data_out[i]) = UNKNOWN;
                OUTPUT_STRENGTH(data_out[i]) = HI_IMPEDANCE;
            }
        }
    }

    else {      /****** Transient Analysis ******/
                                 
        /***** Find input that has changed... *****/

        /**** Test select value for change ****/
        if ( *select != *select_old ) { /* either selected or de-selected */

            switch ( *select ) {

            case ONE: /** chip is selected **/
                if ( ZERO == *write_en ) {   /* need to retrieve new output */
                    for (i=0; i<word_width; i++) { 
                        /* for each output bit in the word, */
                        /* retrieve the state value.        */
                        out = cm_get_ram_value(word_width,i,address,address_size,ram);
                        OUTPUT_STATE(data_out[i]) = out;                              
                        OUTPUT_STRENGTH(data_out[i]) = STRONG;                              
                        OUTPUT_DELAY(data_out[i]) = PARAM(read_delay);
                    }
                }
                else {  /* store word & output current value */
                    for (i=0; i<word_width; i++) {
                        if (address_unknown) {
                            /** entire ram goes unknown!!! **/
                            for (j=0; j<ram_size; j++) {
                                cm_initialize_ram(UNKNOWN,word_width,i,j,ram);
                            }
                            OUTPUT_STATE(data_out[i]) = UNKNOWN;
                        }
                        else {
                            out = INPUT_STATE(data_in[i]);
                            cm_store_ram_value(out,word_width,i,address,
                                               address_size,ram);
                            OUTPUT_STATE(data_out[i]) = out;
                        }
                        OUTPUT_STRENGTH(data_out[i]) = HI_IMPEDANCE;
                        OUTPUT_DELAY(data_out[i]) = PARAM(read_delay);
                    }
                }
                break;


            case UNKNOWN:
            case ZERO: /* output goes tristate unless already so... */
                if ( ZERO == *write_en ) {   /* output needs tristating  */
                    for (i=0; i<word_width; i++) {
                        OUTPUT_STATE(data_out[i]) = UNKNOWN;
                        OUTPUT_STRENGTH(data_out[i]) = HI_IMPEDANCE;
                        OUTPUT_DELAY(data_out[i]) = PARAM(read_delay);
                    }
                }
                else {  /* output will not change   */
                    for (i=0; i<word_width; i++) {
                        OUTPUT_CHANGED(data_out[i]) = FALSE;
                    }
                }
                break;
            }
        } 
        else {

            /**** Either a change in write_en, a change in address 
                  or a change in data values must have occurred...
                  all of these are treated similarly.       ****/
            switch ( *write_en ) {

            case ONE:
                if ( (ONE == *select) ) {   /* need to output tristates 
                                             & store current data word */
                    for (i=0; i<word_width; i++) {
                        if (address_unknown) {
                            /** entire ram goes unknown!!! **/
                            for (j=0; j<ram_size; j++) {
                                cm_initialize_ram(UNKNOWN,word_width,i,j,ram);
                            }
                            OUTPUT_STATE(data_out[i]) = UNKNOWN;
                        }
                        else {
                            out = INPUT_STATE(data_in[i]);
                            cm_store_ram_value(out,word_width,i,address,
                                               address_size,ram);
                            OUTPUT_STATE(data_out[i]) = out;
                        }
                        OUTPUT_STRENGTH(data_out[i]) = HI_IMPEDANCE;
                        OUTPUT_DELAY(data_out[i]) = PARAM(read_delay);
                    }
                }
                else {  /* output will not change   */
                    for (i=0; i<word_width; i++) {
                        OUTPUT_CHANGED(data_out[i]) = FALSE;
                    }
                }
                break;

            case UNKNOWN:
            case ZERO:     /** output contents of current address if selected **/
                if ( ONE == *select) {   /* store word & output current value */
                    for (i=0; i<word_width; i++) {
                        if (address_unknown) {
                            /* output goes unknown */
                            OUTPUT_STATE(data_out[i]) = UNKNOWN;
                        }
                        else {
                            /* for each output bit in the word, */
                            /* retrieve the state value.        */
                            out = cm_get_ram_value(word_width,i,address,
                                             address_size,ram);
                            OUTPUT_STATE(data_out[i]) = out;
                        }
                        OUTPUT_STRENGTH(data_out[i]) = STRONG;
                        OUTPUT_DELAY(data_out[i]) = PARAM(read_delay);
                    }
                }
                else {  /* output will not change   */
                    for (i=0; i<word_width; i++) {
                        OUTPUT_CHANGED(data_out[i]) = FALSE;
                    }
                }
                break;
            }
        }
    }
} 

      



