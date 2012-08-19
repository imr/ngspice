/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE d_source/cfunc.mod

Copyright 1991
Georgia Tech Research Corporation, Atlanta, Ga. 30332
All Rights Reserved

PROJECT A-8503-405


AUTHORS

    6 June 1991     Jeffrey P. Murray


MODIFICATIONS

    30 Sept 1991    Jeffrey P. Murray
    19 Aug  2012    Holger Vogt

SUMMARY

    This file contains the model-specific routines used to
    functionally describe the <model_name> code model.


INTERFACES

    FILE                 ROUTINE CALLED

    CMmacros.h           cm_message_send();

    CMevt.c              void *cm_event_alloc()
                         void *cm_event_get_ptr()
                         int  cm_event_queue()



REFERENCED FILES

    Inputs from and outputs to ARGS structure.


NON-STANDARD FEATURES

    NONE

===============================================================================*/

/*=== INCLUDE FILES ====================*/

#include "d_source.h"    /*    ...contains macros & type defns.
                                     for this model.  6/13/90 - JPM */
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>


/*=== CONSTANTS ========================*/

#define MAX_STRING_SIZE 200


/*=== MACROS ===========================*/

#if defined(__MINGW32__) || defined(_MSC_VER)
#define DIR_PATHSEP    "\\"
#else
#define DIR_PATHSEP    "/"
#endif

#ifdef _MSC_VER
#define snprintf _snprintf
#endif

/*=== LOCAL VARIABLES & TYPEDEFS =======*/

typedef struct {
    int index,  /* current index into source tables                 */
        width,  /* width of table...equal to size of out port */
        depth;  /* depth of table...equal to size of
                   "timepoints" array, and to the total
                   number of vectors retrieved from the
                   source.in file.                                  */
} Source_Table_Info_t;

    double   *all_timepoints;   /* the storage array for the
                                   timepoints, as read from file.
                                   This will have size equal
                                   to "depth"   */

    char          **all_bits;   /* the storage array for the
                                   output bit representations;
                                   as read from file. This will have size equal
                                   to width*depth, one short will hold a
                                   12-state bit description.    */

/* Type definition for each possible token returned. */
typedef enum token_type_s {CNV_NO_TOK,CNV_STRING_TOK} Cnv_Token_Type_t;



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

#include <stdlib.h>

static char  *CNVgettok(char **s)

{

    char    *buf;       /* temporary storage to copy token into */
    /*char    *temp;*/      /* temporary storage to copy token into */
    char    *ret_str;   /* storage for returned string */

    int     i;

    /* allocate space big enough for the whole string */

    buf = (char *) malloc(strlen(*s) + 1);

    /* skip over any white space */

    while(isspace(**s) || (**s == '=') ||
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
               (! ( isspace(**s) || (**s == '=') ||
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

    while(isspace(**s) || (**s == '=') ||
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
        if( isalpha(c) && (c != 'E') && (c != 'e') )
            break;
        else if( isspace(c) )
            break;
        else
            val_str[i] = c;
    }
    val_str[i] = '\0';


    /* Determine the scale factor */

    if( (i >= len) || (! isalpha(c)) )
        scale_factor = 1.0;
    else {

        if(isupper(c))
            c = (char) tolower(c);

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

        case 'm':
            i++;
            if(i >= len) {
                scale_factor = 1.0e-3;
                break;
            }
            c1 = str[i];
            if(! isalpha(c1)) {
                scale_factor = 1.0e-3;
                break;
            }
            if(islower(c1))
                c1 = (char) toupper(c1);
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

FUNCTION cm_source_retrieve()

AUTHORS

    15 Jul 1991     Jeffrey P. Murray

MODIFICATIONS

    16 Jul 1991     Jeffrey P. Murray
    30 Sep 1991     Jeffrey P. Murray
    19 Aug 2012     H. Vogt

SUMMARY

    Retrieves a bit value from a char character

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

/*=== Static CM_SOURCE_RETRIEVE ROUTINE ===*/

/**************************************************
*      The following routine retrieves            *
*   the value passed to it by the out value.      *
*                                                 *
*   Created 7/15/91               J.P.Murray      *
**************************************************/

static void cm_source_retrieve(char val, Digital_t *out)
{


    int value = (int)val;

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


FUNCTION cm_get_source_value()

AUTHORS

    15 Jul 1991     Jeffrey P. Murray

MODIFICATIONS

    16 Jul 1991     Jeffrey P. Murray
    30 Sep 1991     Jeffrey P. Murray
    19 Aug 2012     H. Vogt

SUMMARY

    Retrieves a char per bit value from the array "all_bits".

INTERFACES

    FILE                 ROUTINE CALLED

    N/A                  N/A


RETURNED VALUE

    Returns data via *out pointer.

GLOBAL VARIABLES

    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== Static CM_GET_SOURCE_VALUE ROUTINE ===*/

/************************************************
*      The following routine retrieves a bit    *
*      from all_bits and stores them in out     *
*                                               *
*      Created 8/19/12               H. Vogt    *
************************************************/

static void cm_get_source_value(int row, int bit_number, char **all_bits, Digital_t *out)

{
    char val;

    val = all_bits[row][bit_number];

    cm_source_retrieve(val,out);

}



/*==============================================================================

FUNCTION cm_read_source()

AUTHORS

    15 Jul 1991     Jeffrey P. Murray

MODIFICATIONS

    19 Jul 1991     Jeffrey P. Murray
    30 Sep 1991     Jeffrey P. Murray
    19 Aug 2012     H. Vogt

SUMMARY

    This function reads the source file and stores the results
    for later output by the model.

INTERFACES

    FILE                 ROUTINE CALLED

    N/A                  N/A


RETURNED VALUE

    Returns output bits stored in "all_bits" array,
    time values in "all_timepoints" array,
    return != 0 code if error.

GLOBAL VARIABLES

    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== Static CM_READ_SOURCE ROUTINE ===*/


/**************************************************
*      The following routine reads the file       *
*   *source, parses the file, and stores          *
*   the values found there into the *bits &       *
*   *timepoints arrays.                           *
*                                                 *
*   Created 7/15/91               J.P.Murray      *
**************************************************/

static int cm_read_source(FILE *source,char **all_bits,double *all_timepoints,
                   Source_Table_Info_t *info)
{
    size_t      n;  /* loop index */
    int         i,  /* indexing variable    */
                j,  /* indexing variable    */
       num_tokens;  /* number of tokens in a given string    */

    Cnv_Token_Type_t type;  /* variable for testing token type returned.    */

    char   temp[MAX_STRING_SIZE],   /* holding string variable for testing
                                       input from source.in */
                              *s,   /* main string variable */
                   *base_address,   /* storage location for base address
                                       of string.   */
                          *token;   /* a particular token from the string   */


    double                number;   /* holding variable for timepoint values */

    char               bit_value;   /* holding variable for value read from
                                       source file which needs to be stored */

    i = 0;
    s = temp;
    while ( fgets(s,MAX_STRING_SIZE,source) != NULL) {

        /* Test this string to see if it is whitespace... */

        base_address = s;
        while(isspace(*s) || (*s == '*'))
              (s)++;
        if ( *s != '\0' ) {     /* This is not a blank line, so process... */
            s = base_address;

            if ( '*' != s[0] ) {

                /* Count up total number of tokens including \0... */
                j = 0;
                type = CNV_STRING_TOK;
                while ( type != CNV_NO_TOK ) {
                    token = CNVget_token(&s, &type);
                    j++;
                }
                num_tokens = j;

                /* If this number is incorrect, return with an error    */
                if ( (info->width + 2) != num_tokens) {
                    return 2;
                }

                /* reset s to beginning... */
                s = base_address;

                /* set storage space for bits in a row and set them to 0*/
                all_bits[i] = (char*)malloc(sizeof(char) * info->width);
                for (n = 0; n < (unsigned int)info->width; n++)
                    all_bits[i][n] = 0;

                /** Retrieve each token, analyze, and       **/
                /** store the timepoint and bit information **/
                for (j=0; j<(info->width + 1); j++) {

                    token = CNVget_token(&s, &type);

                    if ( 0 == j ) { /* obtain timepoint value... */

                        /* convert to a floating point number... */
                        cnv_get_spice_value(token,&number);

                        all_timepoints[i] = number;


                        /* provided this is not the first timepoint
                           to be written... */
                        if ( 0 != i ) {

                            /* if current timepoint value is not greater
                               than the previous value, then return with
                               an error message... */
                            if ( all_timepoints[i] <= all_timepoints[i-1] ) {
                                return 3;
                            }
                        }

                    }
                    else { /* obtain each bit value & set bits entry    */

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
                            return 4;
                        }
                        else { /* need to store this value in the all_bits[] array */
                            all_bits[i][j-1] = bit_value;
                        }
                    }
                }
                i++;
            }
            s = temp;
        }
    }
    return 0;
}





/*==============================================================================

FUNCTION cm_d_source()

AUTHORS

    13 Jun 1991     Jeffrey P. Murray

MODIFICATIONS

     8 Aug 1991     Jeffrey P. Murray
    30 Sep 1991     Jeffrey P. Murray
    19 Aug 2012     H. Vogt

SUMMARY

    This function implements the d_source code model.

INTERFACES

    FILE                 ROUTINE CALLED

    CMmacros.h           cm_message_send();

    CMevt.c              void *cm_event_alloc()
                         void *cm_event_get_ptr()
                         int  cm_event_queue()


RETURNED VALUE

    Returns inputs and outputs via ARGS structure.

GLOBAL VARIABLES

    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== CM_D_SOURCE ROUTINE ==============*/

/************************************************
*      The following is the new model for       *
*      the digital source                       *
*                                               *
*      Created 8/19/12            H. Vogt       *
************************************************/


void cm_d_source(ARGS)
{
    int                    i,   /* generic loop counter index */
                         err;   /* integer for storage of error status  */

    char               *bits,   /* the storage array for the
                                   output bit representations...
                                   this will have size equal to width,
                                   since one short will hold a 12-state
                                   bit description.    */
                   *bits_old;   /* the storage array for old bit values */

    double        *timepoint,   /* the storage array for the
                                   timepoints, a single point   */
              *timepoint_old;   /* the storage  for the old timepoint */

    volatile double             /* enforce 64 bit precision, (equality comparison) */
                 test_double;   /* test variable for doubles    */



    FILE                 *source;   /* pointer to the source.in input
                                       vector file */


    Source_Table_Info_t    *info,   /* storage location for source
                                       index and depth info. */
                       *info_old;   /* storage location for old info */


    Digital_t                out;   /* storage for each output bit */



    char   temp[MAX_STRING_SIZE],   /* holding string variable for testing
                                       input from source.in */
                              *s;   /* main string variable */


    char *loading_error = "\nERROR **\n D_SOURCE: source.in file was not read successfully. \n";


    /**** Setup required state variables ****/

    if(INIT) {  /* initial pass */

        /*** open file and count the number of vectors in it ***/
        char* filename = PARAM(input_file);
        source = fopen_with_path( filename, "r");

        if (!source) {
            char *lbuffer, *p;
            lbuffer = getenv("NGSPICE_INPUT_DIR");
            if (lbuffer && *lbuffer) {
                p = (char*) malloc(strlen(lbuffer) + strlen(DIR_PATHSEP) + strlen(PARAM(input_file)) + 1);
                sprintf(p, "%s%s%s", lbuffer, DIR_PATHSEP, PARAM(input_file));
                source = fopen(p, "r");
                free(p);
            }
            if (!source) {
                char msg[512];
                snprintf(msg, sizeof(msg), "cannot open file %s", PARAM(input_file));
                cm_message_send(msg);
            }
        }

        /* increment counter if not a comment until EOF reached... */
        i = 0;
        if (source) {
          s = temp;
          while ( fgets(s,MAX_STRING_SIZE,source) != NULL) {
              if ( '*' != s[0] ) {
                  while(isspace(*s) || (*s == '*'))
                        (s)++;
                  if ( *s != '\0' ) i++;
              }
              s = temp;
          }
        }

        /*** allocate storage for **all_bits, & *all_timepoints ***/
        all_timepoints = (double*)malloc(i * sizeof(double));
        all_bits = (char**)malloc(i * sizeof(char*));

        /*** allocate storage for *index, *bits & *timepoints ***/

                          cm_event_alloc(0,sizeof(Source_Table_Info_t));

        cm_event_alloc(1, PORT_SIZE(out) * (int) sizeof(char));

        cm_event_alloc(2, (int) sizeof(double));

        /**** Get all pointers again (to avoid realloc problems) ****/
        info = info_old = (Source_Table_Info_t *) cm_event_get_ptr(0,0);
        bits = bits_old = (char *) cm_event_get_ptr(1,0);
        timepoint = timepoint_old = (double *) cm_event_get_ptr(2,0);

        /* Initialize info values... */
        info->index = 0;
        info->depth = i;

        /* Retrieve width of the source */
        info->width = PORT_SIZE(out);

        /* Initialize *bits & *timepoints to zero */
        for (i=0; i<info->width; i++)
            bits[i] = 0;
        for (i=0; i<info->depth; i++)
            all_timepoints[i] = 0.0;

        /* Send file pointer and the two array storage pointers */
        /* to "cm_read_source()". This will return after        */
        /* reading the contents of source.in, and if no         */
        /* errors have occurred, the "*all_bits" and "*all_timepoints"  */
        /* vectors will be loaded and the width and depth       */
        /* values supplied.                                     */

        if (source) {
          rewind(source);
          err = cm_read_source(source,all_bits,all_timepoints,info);
        } else {
          err=1;
        }

        if (err) { /* problem occurred in load...send error msg. */
            cm_message_send(loading_error);

            /* Reset *bits & *timepoints to zero */
            for (i=0; i<info->width; i++) bits[i] = 0;
            for (i=0; i<info->depth; i++) all_timepoints[i] = 0;
            switch (err)
            {
            case 2:
                cm_message_send(" d_source word length and number of columns in file differ.\n" );
                break;
            case 3:
                cm_message_send(" Time values in first column have to increase monotonically.\n");
                break;
            case 4:
                cm_message_send(" Unknown bit value.\n");
                break;
            default:
                break;
            }
        }

        /* close source file */
        if (source)
            fclose(source);
    }
    else {      /*** Retrieve previous values ***/

        /** Retrieve info... **/
        info = (Source_Table_Info_t *) cm_event_get_ptr(0,0);
        info_old  = (Source_Table_Info_t *) cm_event_get_ptr(0,1);

        /* Set old values to new... */
        info->index = info_old->index;
        info->depth = info_old->depth;
        info->width = info_old->width;

        /** Retrieve bits... **/
        bits = (char *) cm_event_get_ptr(1,0);
        bits_old = (char *) cm_event_get_ptr(1,1);

        for (i=0; i<PORT_SIZE(out); i++) bits[i] = bits_old[i];

        /** Retrieve timepoints... **/
        timepoint = (double *) cm_event_get_ptr(2,0);
        timepoint_old = (double *) cm_event_get_ptr(2,1);

        /* Set old values to new... */
        timepoint = timepoint_old;
    }

    /*** For the case of TIME==0.0, set special breakpoint ***/

    if ( 0.0 == TIME ) {

        test_double = all_timepoints[info->index];
        if ( 0.0 == test_double && info->depth > 0 ) { /* Set DC value */

            /* reset current breakpoint */
            test_double = all_timepoints[info->index];
            cm_event_queue( test_double );

            /* Output new values... */
            for (i=0; i<info->width; i++) {

                /* retrieve output value */
                cm_get_source_value(info->index, i, all_bits, &out);

                OUTPUT_STATE(out[i]) = out.state;
                OUTPUT_STRENGTH(out[i]) = out.strength;
            }

            /* increment breakpoint */
            (info->index)++;


            /* set next breakpoint as long as depth
               has not been exceeded    */
            if ( info->index < info->depth ) {
                test_double = all_timepoints[info->index] - 1.0e-10;
                cm_event_queue( test_double );
            }

        }
        else { /* Set breakpoint for first time index */

            /* set next breakpoint as long as depth
               has not been exceeded    */
            if ( info->index < info->depth ) {
                test_double = all_timepoints[info->index] - 1.0e-10;
                cm_event_queue( test_double );
            }
            for(i=0; i<info->width; i++) {
              OUTPUT_STATE(out[i]) = UNKNOWN;
              OUTPUT_STRENGTH(out[i]) = UNDETERMINED;
            }
        }
    }
    else {

        /*** Retrieve last index value and branch to appropriate ***
         *** routine based on the last breakpoint's relationship ***
         *** to the current time value.                          ***/

        test_double = all_timepoints[info->index] - 1.0e-10;

        if ( TIME < test_double ) { /* Breakpoint has not occurred */

            /** Output hasn't changed...do nothing this time. **/
            for (i=0; i<info->width; i++) {
                OUTPUT_CHANGED(out[i]) = FALSE;
            }

            if ( info->index < info->depth ) {
                test_double = all_timepoints[info->index] - 1.0e-10;
                cm_event_queue( test_double );
            }

        }
        else  /* Breakpoint has been reached or exceeded  */

        if ( TIME == test_double ) { /* Breakpoint reached */

            /* reset current breakpoint */
            test_double = all_timepoints[info->index] - 1.0e-10;
            cm_event_queue( test_double );

            /* Output new values... */
            for (i=0; i<info->width; i++) {

                /* retrieve output value */
                cm_get_source_value(info->index, i , all_bits, &out);

                OUTPUT_STATE(out[i]) = out.state;
                OUTPUT_DELAY(out[i]) = 1.0e-10;
                OUTPUT_STRENGTH(out[i]) = out.strength;
            }

            /* increment breakpoint */
            (info->index)++;

            /* set next breakpoint as long as depth
               has not been exceeded    */
            if ( info->index < info->depth ) {
                test_double = all_timepoints[info->index] - 1.0e-10;
                cm_event_queue( test_double );
            }
        }

        else { /* Last source file breakpoint has been exceeded...
                      do not change the value of the output */

            for (i=0; i<info->width; i++) {
                OUTPUT_CHANGED(out[i]) = FALSE;
            }
        }
    }
}



