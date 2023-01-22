/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE d_source/cfunc.mod

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
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

#define MAX_STRING_SIZE 1024


/*=== MACROS ===========================*/

#if defined(__MINGW32__) || defined(_MSC_VER)
#define DIR_PATHSEP    "\\"
#else
#define DIR_PATHSEP    "/"
#endif


/*=== LOCAL VARIABLES & TYPEDEFS =======*/

typedef struct {
    int width,  /* width of table...equal to size of out port */
        depth,  /* depth of table...equal to size of
                   "timepoints" array, and to the total
                   number of vectors retrieved from the
                   source.in file. */
        imal;   /* array size malloced */

    double   *all_timepoints;   /* the storage array for the
                                   timepoints, as read from file.
                                   This will have size equal
                                   to "depth"   */

    char          **all_data;   /* the storage array for the
                                   output bit representations;
                                   as read from file. This will have size equal
                                   to width*depth, one short will hold a
                                   12-state bit description.    */

    bool   init_ok;             /* set to true if init is succeful */

} Local_Data_t;


/* Type definition for each possible token returned. */
typedef enum token_type_s {CNV_NO_TOK,CNV_STRING_TOK} Cnv_Token_Type_t;



/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/


static void free_local_data(Local_Data_t *loc);



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

        if(isupper_c(c))
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

    Retrieves a char per bit value from the array "all_data".

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
*      from all_data and stores them in out     *
*                                               *
*      Created 8/19/12               H. Vogt    *
************************************************/

static void cm_get_source_value(int row, int bit_number, char **all_data, Digital_t *out)

{
    char val;

    val = all_data[row][bit_number];

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

    Returns output bits stored in "all_data" array,
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

static int cm_read_source(FILE *source, Local_Data_t *loc)
{
    int         n;  /* loop index */
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
    loc->imal = 0;
    s = temp;
    while ( fgets(s,MAX_STRING_SIZE,source) != NULL) {

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
                num_tokens = j;

                /* If this number is incorrect, return with an error    */
                if ( (loc->width + 2) != num_tokens) {
                    return 2;
                }

                /* reset s to beginning... */
                s = base_address;

                /* set storage space for bits in a row and set them to 0*/
                loc->all_data[i] = (char*)malloc(sizeof(char) * (size_t) loc->width);
                loc->imal = i;
                for (n = 0; n < loc->width; n++)
                    loc->all_data[i][n] = 0;

                /** Retrieve each token, analyze, and       **/
                /** store the timepoint and bit information **/
                for (j=0; j<(loc->width + 1); j++) {

                    token = CNVget_token(&s, &type);

                    if (!token)
                        return 4;

                    if ( 0 == j ) { /* obtain timepoint value... */

                        /* convert to a floating point number... */
                        cnv_get_spice_value(token,&number);

                        loc->all_timepoints[i] = number;


                        /* provided this is not the first timepoint
                           to be written... */
                        if ( 0 != i ) {

                            /* if current timepoint value is not greater
                               than the previous value, then return with
                               an error message... */
                            if ( loc->all_timepoints[i] <= loc->all_timepoints[i-1] ) {
                                free(token);
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
                            free(token);
                            return 4;
                        }
                        else { /* need to store this value in the all_data[] array */
                            loc->all_data[i][j-1] = bit_value;
                        }
                    }
                    free(token);
                }
                i++;
            }
            s = temp;
        }
    }
    return 0;
}


static void cm_d_source_callback(ARGS,
        Mif_Callback_Reason_t reason)
{
    switch (reason) {
        case MIF_CB_DESTROY: {
            Local_Data_t *loc = STATIC_VAR(locdata);
            if (loc) {
                free_local_data(loc);
                STATIC_VAR(locdata) = loc = NULL;
            }
            break;
        } /* end of case MIF_CB_DESTROY */
    } /* end of switch over reason being called */
} /* end of function cm_d_source_callback */



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

/**** the state variables, memory allocation with cm_event_alloc *****/
    char               *bits,   /* the storage array for the
                                   output bit representations...
                                   this will have size equal to width,
                                   since one short will hold a 12-state
                                   bit description.    */
                   *bits_old;   /* the storage array for old bit values */

    int           *row_index,        /* current index into source tables */
                  *row_index_old;    /* previous index into source tables */

    double        *timepoint,   /* the storage array for the
                                   timepoints, a single point   */
              *timepoint_old;   /* the storage  for the old timepoint */
/*********************************************************************/

    volatile double             /* enforce 64 bit precision, (equality comparison) */
                 test_double;   /* test variable for doubles    */

    FILE                 *source;   /* pointer to the source.in input
                                       vector file */


    Digital_t                out;   /* storage for each output bit */


    char   temp[MAX_STRING_SIZE],   /* holding string variable for testing
                                       input from source.in */
                              *s;   /* main string variable */


    char *loading_error = "\nERROR **\n D_SOURCE: source.in file was not read successfully. \n";

    Local_Data_t *loc;        /* Pointer to local static data, not to be included
                                       in the state vector (save memory!) */

    /**** Setup required local and state variables ****/
    /* local data will be setup once during INIT and never change again */

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
            if (!source)
                cm_message_printf("cannot open file %s", PARAM(input_file));
        }

        /* increment counter if not a comment until EOF reached... */
        i = 0;
        if (source) {
          s = temp;
          while ( fgets(s,MAX_STRING_SIZE,source) != NULL) {
              if ( '*' != s[0] ) {
                  while(isspace_c(*s) || (*s == '*'))
                        (s)++;
                  if ( *s != '\0' ) i++;
              }
              s = temp;
          }
        }

        /*** allocate static storage for *loc ***/
        STATIC_VAR (locdata) = calloc (1 , sizeof ( Local_Data_t ));
        loc = STATIC_VAR (locdata);
        CALLBACK = cm_d_source_callback;

        /*** allocate storage for *index, *bits & *timepoint ***/

        cm_event_alloc(0, sizeof(int));

        cm_event_alloc(1, PORT_SIZE(out) * (int) sizeof(char));

        cm_event_alloc(2, (int) sizeof(double));

        /**** Get all pointers again (to avoid realloc problems) ****/
        row_index = row_index_old = (int *) cm_event_get_ptr(0,0);
        bits = bits_old = (char *) cm_event_get_ptr(1,0);
        timepoint = timepoint_old = (double *) cm_event_get_ptr(2,0);

        /* Initialize info values... */
        *row_index = 0;
        loc->depth = i;

        /* Retrieve width of the source */
        loc->width = PORT_SIZE(out);

        /*** allocate storage for **all_data, & *all_timepoints ***/
        loc->all_timepoints = (double*)calloc((size_t) i, sizeof(double));
        loc->all_data = (char**)calloc((size_t) i, sizeof(char*));

        /* Send file pointer and the two array storage pointers */
        /* to "cm_read_source()". This will return after        */
        /* reading the contents of source.in, and if no         */
        /* errors have occurred, the "*all_data" and "*all_timepoints"  */
        /* vectors will be loaded and the width and depth       */
        /* values supplied.                                     */

        if (source) {
          rewind(source);
          err = cm_read_source(source, loc);
        } else {
          err=1;
        }

        if (err) { /* problem occurred in load...send error msg. */
            cm_message_send(loading_error);

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

        if (err > 0) {
            loc->init_ok = FALSE;
            cm_message_send(" dsource will return only its initial state.\n");
            return;
        }
        else
            loc->init_ok = TRUE;
    }
    else {      /*** Retrieve previous values ***/

        /** Retrieve info... **/
        row_index = (int *) cm_event_get_ptr(0,0);
        row_index_old  = (int *) cm_event_get_ptr(0,1);

        loc = STATIC_VAR (locdata);

        if (!loc->init_ok)
            return;

        /* Set old values to new... */
        *row_index = *row_index_old;

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

        test_double = loc->all_timepoints[*row_index];
        if ( 0.0 == test_double && loc->depth > 0 ) { /* Set DC value */

            /* reset current breakpoint */
            test_double = loc->all_timepoints[*row_index];
            cm_event_queue( test_double );

            /* Output new values... */
            for (i=0; i<loc->width; i++) {

                /* retrieve output value */
                cm_get_source_value(*row_index, i, loc->all_data, &out);

                OUTPUT_STATE(out[i]) = out.state;
                OUTPUT_STRENGTH(out[i]) = out.strength;
            }

            /* increment breakpoint */
            (*row_index)++;


            /* set next breakpoint as long as depth
               has not been exceeded    */
            if ( *row_index < loc->depth ) {
                test_double = loc->all_timepoints[*row_index] - 1.0e-10;
                cm_event_queue( test_double );
            }

        }
        else { /* Set breakpoint for first time index */

            /* set next breakpoint as long as depth
               has not been exceeded    */
            if ( *row_index < loc->depth ) {
                test_double = loc->all_timepoints[*row_index] - 1.0e-10;
                cm_event_queue( test_double );
            }
            for(i=0; i<loc->width; i++) {
              OUTPUT_STATE(out[i]) = UNKNOWN;
              OUTPUT_STRENGTH(out[i]) = UNDETERMINED;
            }
        }
    }
    else {

        /*** Retrieve last index value and branch to appropriate ***
         *** routine based on the last breakpoint's relationship ***
         *** to the current time value.                          ***/

        test_double = loc->all_timepoints[*row_index] - 1.0e-10;

        if ( TIME < test_double ) { /* Breakpoint has not occurred */

            /** Output hasn't changed...do nothing this time. **/
            for (i=0; i<loc->width; i++) {
                OUTPUT_CHANGED(out[i]) = FALSE;
            }

            if ( *row_index < loc->depth ) {
                test_double = loc->all_timepoints[*row_index] - 1.0e-10;
                cm_event_queue( test_double );
            }

        }
        else  /* Breakpoint has been reached or exceeded  */

        if ( TIME == test_double ) { /* Breakpoint reached */

            /* reset current breakpoint */
            test_double = loc->all_timepoints[*row_index] - 1.0e-10;
            cm_event_queue( test_double );

            /* Output new values... */
            for (i=0; i<loc->width; i++) {

                /* retrieve output value */
                cm_get_source_value(*row_index, i , loc->all_data, &out);

                OUTPUT_STATE(out[i]) = out.state;
                OUTPUT_DELAY(out[i]) = 1.0e-10;
                OUTPUT_STRENGTH(out[i]) = out.strength;
            }

            /* increment breakpoint */
            (*row_index)++;

            /* set next breakpoint as long as depth
               has not been exceeded    */
            if ( *row_index < loc->depth ) {
                test_double = loc->all_timepoints[*row_index] - 1.0e-10;
                cm_event_queue( test_double );
            }
        }

        else { /* Last source file breakpoint has been exceeded...
                      do not change the value of the output */

            for (i=0; i<loc->width; i++) {
                OUTPUT_CHANGED(out[i]) = FALSE;
            }
        }
    }
}

/* Free memory allocations in Local_Data_t structure */
static void free_local_data(Local_Data_t *loc)
{
    if (loc == (Local_Data_t *) NULL) {
        return;
    }
    /* Free data table and related values */
    if (loc->all_timepoints) {
        free(loc->all_timepoints);
    }
    if (loc->all_data) {
        int i;
        for (i = 0; i <= loc->imal; i++)
            free(loc->all_data[i]);
        free(loc->all_data);
    }
    free(loc);
} /* end of function free_local_data */


