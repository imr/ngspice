/*============================================================================
FILE    MIF_INP2A.c

MEMBER OF process XSPICE

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503

AUTHORS

    9/12/91  Bill Kuhn

MODIFICATIONS

    11/19/12 H Vogt patch by Marcel (mhx) added

SUMMARY

    This file contains the main routine for parsing code model lines
    in the SPICE circuit description input deck.

INTERFACES

    MIF_INP2A()

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/

/* #include "prefix.h"   */  /* jgroves  */
#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/fteext.h"
#include "ngspice/ifsim.h"
#include "ngspice/inpdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/inpmacs.h"
#include "ngspice/fteext.h"

#include "ngspice/mifproto.h"
#include "ngspice/mifparse.h"
#include "ngspice/mifdefs.h"
#include "ngspice/mifcmdat.h"

#include "ngspice/evt.h"
#include "ngspice/evtproto.h"

extern int *DEVicesfl; /*flags for the devices */

static void  MIFinit_inst(MIFmodel *mdfast, MIFinstance *fast);

static void  MIFget_port_type(
    CKTcircuit       *ckt,      /* circuit structure to put mod/inst structs in */
    INPtables        *tab,      /* symbol table for node names, etc.            */
    struct card      *current,  /* MUST be named 'current' for spice macros     */
    char             **line,
    char             **next_token,
    Mif_Token_Type_t *next_token_type,
    Mif_Port_Type_t  *port_type,
    char             **port_type_str,
    Mif_Conn_Info_t  *conn_info,  /* for faster access to conn info struct */
    Mif_Status_t     *status);


static void MIFget_port(
    CKTcircuit       *ckt,      /* circuit structure to put mod/inst structs in */
    INPtables        *tab,      /* symbol table for node names, etc.            */
    struct card      *current,  /* MUST be named 'current' for spice macros     */
    MIFinstance      *fast,     /* pointer to instance struct */
    char             **line,
    char             **next_token,
    Mif_Token_Type_t *next_token_type,
    Mif_Port_Type_t  def_port_type,
    char             *def_port_type_str,
    Mif_Conn_Info_t  *conn_inf,   /* for faster access to conn info struct */
    int              conn_num,
    int              port_num,
    Mif_Status_t     *status);

/** A local garbage collector **
The functions copy, MIFgettok, and MIFget_token have been used virtuously,
without caring about memory leaks. This is a test with a local gc.
Add the list of malloced addresses alltokens.
Add a function copy_gc to copy and enter the address.
Add a function MIFgettok_gc like MIFgettok, but entering the address
Add a function MIFget_token_gc like MIFget_token, but entering the address
Add a function gc_end to delete all entries in alltokens.
Beware of addresses deleted elsewhere and use anew by malloc.
Some tokens should not be deleted here, they need another copying.
*/
static char *MIFgettok_gc(char **line);
static char *MIFget_token_gc(char **s, Mif_Token_Type_t *type);
static char *copy_gc(char *in);
static void gc_start(void);
static void gc_end(void);

#define MIFgettok MIFgettok_gc
#define MIFget_token MIFget_token_gc

static char *alltokens[BSIZE_SP];
static int curtoknr = 0;

/* ********************************************************************* */



/*
MIF_INP2A

This function is called by INPpas2() in SPICE to parse the new
``a'' type element cards and build the required circuit structures
for the associated code model device and instance.  It first
checks the model name at the end of the element card to be sure
the model was found in pass 1 of the parser.  If so, MIFgetMod is
called to process the .model card, creating the necessary
internal model structure and filling in the parameter value
information.  Next, the instance structure is allocated.
Finally, the connections on the element card are scanned and the
connection information is filled-in on the instance structure,
and error checks are performed.
*/

/*---------------  Quick summary of algorithm  -----------------------

1.  Get the spice card.  Place card string into variable 'line'.
2.  Get the name of the instance and add it to the symbol table
3.  Locate the last token on the line and assign it to variable 'model'
4.  Locate the model from pass 1.  If it hasn't been processed yet,
    allocate structure in ckt for it, process the parameters, and return a
    pointer to its structure in 'thismodel'
5.  Get model type.
6.  Create a new instance structure in ckt for this instance.
------  Process the connections: here's where it gets interesting. -----
7.  Reset 'line'.  Then read instance name again to go over it.
8.  Read initial token.
9   Start loop through port tokens:
10. If token is a %, then read next token to get port type
    and save this port type.  Otherwise, use default port type.
11. Check if connection is null.  If so, iterate to next conn.
12. Get next token.  Depending upon token type (scalar or array) do
    the following:
     -- Scalar:  Process it, then continue loop.
     -- Array:   Loop through all tokens until ] is found & process tokens.
                 If token is a %, then read next token to get port type
                 Then continue outer loop over port tokens.
13.  After looping through connection tokens, do error checks.
     At this point, nothing should be left in 'line'

-------------------------------------------------------------------------*/

void
MIF_INP2A (
    CKTcircuit   *ckt,      /* circuit structure to put mod/inst structs in */
    INPtables    *tab,      /* symbol table for node names, etc.            */
    struct card  *current ) /* the card we are to parse                     */
/* Must be called "current" for compatibility   */
/* with macros                                  */
{

    /* parse a code model instance card */
    /* Aname <connection list> <mname> */

    char    *line;      /* the text line for this card */
    char    *name;      /* the name of the instance */
    char    *model=NULL;     /* the name of the model */

    char    *def_port_type_str = NULL;  /* The default port type in string form */
    char    *next_token;         /* a token string */
    char    *tmp_token;         /* a token string */

    int     i;          /* a loop counter */
    int     j;          /* a loop counter */
    int     type;       /* the type of the model for this instance */
    /*    int     num_conn;    number of connections for this model */
    int     error;      /* for the IFC macro */

    MIFmodel     *mdfast;  /* pointer to model struct */
    MIFinstance  *fast[1];    /* pointer to instance struct */

    INPmodel  *thismodel;  /* pointer to model struct */

    Mif_Conn_Info_t  *conn_info;  /* for faster access to conn info struct */
    Mif_Param_Info_t  *param_info;  /* for faster access to param info struct */
    Mif_Port_Type_t  def_port_type = MIF_VOLTAGE;  /* the default port type */
    Mif_Status_t     status;         /* return status */
    Mif_Token_Type_t next_token_type; /* the type of the next token */

#ifdef TRACE
    /* SDB debug statement */
    printf("In MIF_INP2A, line to process = %s . . . \n", current->line);
#endif

    /* get the line text from the card struct */
    line = current->line;

    /* reset the garbage collector */
    gc_start();

    /* get the name of the instance and add it to the symbol table */
    name = copy(MIFgettok(&line));
    INPinsert(&name, tab);

    /* locate the last token on the line (i.e. model name) and put it into "model" */
    while(*line != '\0') {
        model = MIFgettok(&line);
    }

    /* make sure the model name was there. */
    if(model == NULL) {
        LITERR("Missing model on A type device");
        gc_end();
        return;
    }

    /* Locate model from pass 1.  If it hasn't been processed yet, */
    /* allocate a structure in ckt for it, process its parameters  */
    /* and return a pointer to its structure in 'thismodel'        */
    current->error = MIFgetMod(ckt, model, &thismodel, tab);
    if(current->error) {
        gc_end();
        return;
    }

    /* get the integer index into the DEVices data array for this  */
    /* model                                                       */
    type = thismodel->INPmodType;
    if((type >= DEVmaxnum) || DEVicesfl[type] == 0) {
        LITERR("Invalid model type for A type device");
        gc_end();
        return;
    }

    /* create a new structure for this instance in ckt */
    mdfast = (MIFmodel*) thismodel->INPmodfast;
    IFC(newInstance, (ckt, (GENmodel*)mdfast, (GENinstance **)fast, name));


    /* initialize the code model specific elements of the inst struct */
    MIFinit_inst(mdfast, fast[0]);


    /* *********************** */
    /* Process the connections */
    /* *********************** */

    /* reset 'line', and then read instance name again.  */
    line = current->line;

    tmp_token = MIFgettok(&line);  /* read instance name again . . . .*/

    /*  OK -- now &line points to the first token after
    the instance name and we are ready to process the connections
    (netnames)
    */

    /* now get next token.  It should be either a % token,
       a [, or a connection (netname) which might be 'null'.
    */
    next_token = MIFget_token(&line,&next_token_type);

    /* When we enter the loop, next_token holds the first thing *after* the instance name.
       Upon each iteration, we start the iteration with next_token holding
       the next *unprocessed* token.
       The loop proceeds through the fixed number of connections expected, as defined in the
       DEVices struct.
    */
    for(i = 0; i < DEVices[type]->DEVpublic.num_conn; i++) {

        /* Check that the line is not finished yet. */
        if(*line == '\0') {
            LITERR("Encountered end of line before all connections were found in model.");
            gc_end();
            return;
        }

        /* At this point, we have one of three possibilities:
           1.  next_token holds a %, and &line points to either the port type identifier (id, vd, etc)
           2.  next_token holds a netname and &line points to the thing *after* the first netname.
           3.  next_token holds a [ indicating the start of an array of ports.
        */

        /* prepare a pointer for fast access to info about this connection */
        conn_info = &(DEVices[type]->DEVpublic.conn[i]);

        /*  Now if we had a % token, get actual info about connection type.
            Otherwise use default info */
        if(next_token_type == MIF_PERCENT_TOK) {  /*  we found a %  */
            /* get the port type identifier and check it for validity */
            next_token = MIFget_token(&line, &next_token_type);
            /* Note that MIFget_port_type eats the next token and advances the token pointer in line */
            MIFget_port_type(ckt,
                             tab,
                             current,
                             &line,
                             &next_token,
                             &next_token_type,
                             &def_port_type,
                             &def_port_type_str,
                             conn_info,
                             &status);
            if (status == MIF_ERROR) {
                gc_end();
                return;
            }
        } else {   /* use the default port type for this connection */
            def_port_type = conn_info->default_port_type;
            def_port_type_str = copy_gc(conn_info->default_type);
        }

        /* At this point, next_token should be either a [ char, or should hold
           the the first connection (netname)
        */

        /* set analog and event_driven flags on instance and model */
        if((def_port_type == MIF_DIGITAL) ||
                (def_port_type == MIF_USER_DEFINED)) {
            fast[0]->event_driven = MIF_TRUE;
            mdfast->event_driven = MIF_TRUE;
        } else {
            fast[0]->analog = MIF_TRUE;
            mdfast->analog = MIF_TRUE;
        }


        /* check for a null connection and continue to next connection if found */
        if(next_token_type == MIF_NULL_TOK) {
            /* make sure null is allowed */
            if(! conn_info->null_allowed) {
                LITERR("NULL connection found where not allowed");
                gc_end();
                return;
            }

            /* set the null flag to true */
            fast[0]->conn[i]->is_null = MIF_TRUE;
            fast[0]->conn[i]->size = 0;

            /* eat the null token and continue to next connection */
            next_token = MIFget_token(&line,&next_token_type);
            continue;  /* iterate */
        } else {
            /* set the null flag to false */
            fast[0]->conn[i]->is_null = MIF_FALSE;
        }


        /* =====  process connection as appropriate for scalar or array  ====== */
        if(! conn_info->is_array) {     /* a scalar connection - the simpler case */
            /*  If we get to there, next_token should hold a netname in the port netlist.  */
            /*  First, do a couple of error checks */
            if(next_token_type == MIF_LARRAY_TOK) {
                LITERR("ERROR - Scalar connection expected, [ found");
                printf("ERROR - Scalar connection expected, [ found.  Returning . . .");
                gc_end();
                return;
            }
            if(next_token_type == MIF_RARRAY_TOK) {
                LITERR("ERROR - Unexpected ]");
                printf("ERROR - Unexpected ].  Returning . . .");
                gc_end();
                return;
            }

            /* If all OK, get the port data into the instance struct */
            /* allocating the port member of the instance struct as needed */
            /*  Note that MIFget_port eats next_token, and advances the &line pointer..  */
            MIFget_port(ckt,
                        tab,
                        current,
                        fast[0],
                        &line,
                        &next_token,
                        &next_token_type,
                        def_port_type,
                        def_port_type_str,
                        conn_info,
                        i,                  /* connection index */
                        0,                  /* port index for scalar connection */
                        &status);

            if (status == MIF_ERROR) {
                gc_end();
                return;
            }

            fast[0]->conn[i]->size = 1;

            /* when we leave here, next_token should hold the next, unprocessed netname */

        } else { /* ====== the connection is an array - much to be done ... ====== */
            /* At this point, the next_token should be a [ */
            /* check for required leading array delim character [ and eat it if found */
            if(next_token_type != MIF_LARRAY_TOK) {
                LITERR("Missing [, an array connection was expected");
                printf("Missing [, an array connection was expected.  Returning . . .");
                gc_end();
                return;
            } else /* eat the [  */
                next_token = MIFget_token(&line,&next_token_type);

            /*------ get and process ports until ] is encountered ------*/
            for(j = 0;
                    (next_token_type != MIF_RARRAY_TOK) &&
                    (*line != '\0');
                    j++) {
                /********** mhx Friday, August 19, 2011, 15:08 begin ***
                 Now if we had a % token, get actual info about connection type,
                 or else use the port type for this connection that was setup BEFORE the '[' token.
                 Do NOT use conn_info->default_port_type! */
                if (next_token_type == MIF_PERCENT_TOK) {
                    next_token = MIFget_token (&line,&next_token_type);
                    MIFget_port_type(ckt,
                                     tab,
                                     current,
                                     &line,
                                     &next_token,
                                     &next_token_type,
                                     &def_port_type,
                                     &def_port_type_str,
                                     conn_info,
                                     &status);
                    if (status == MIF_ERROR) {
                        gc_end();
                        return;
                    }
                }
                /* At this point, next_token should be either a [ or ] char (not allowed),
                or hold a non-null connection (netname) */
                if(next_token_type == MIF_NULL_TOK) {
                    LITERR("NULL connection found where not allowed");
                    printf("NULL connection found where not allowed. Returning . . .");
                    gc_end();
                    return;
                }
                if(next_token_type == MIF_LARRAY_TOK) {
                    LITERR("ERROR - Unexpected [ - Arrays of arrays not allowed");
                    printf("ERROR - Unexpected [ - Arrays of arrays not allowed. Returning . . .");
                    gc_end();
                    return;
                }
                if(next_token_type == MIF_RARRAY_TOK) {
                    LITERR("ERROR - Unexpected ]");
                    printf("ERROR - Unexpected ]. Returning . . .");
                    gc_end();
                    return;
                }
                /********** mhx Friday, August 19, 2011, 15:08 end ***/

                /* If all OK, get the port nodes into the instance struct */
                /* allocating the port member of the instance struct as needed */
                /* Note that MIFget_port eats next_token and advances &line by one.  */
                MIFget_port(ckt,
                            tab,
                            current,
                            fast[0],
                            &line,
                            &next_token,
                            &next_token_type,
                            def_port_type,
                            def_port_type_str,
                            conn_info,
                            i,                  /* connection index */
                            j,                  /* port index */
                            &status);

                if (status == MIF_ERROR) {
                    gc_end();
                    return;
                }
            } /*------ end of for loop until ] is encountered ------*/

            /*  At this point, next_token should hold the next token after the
            port netnames.  This token should be a ].  */

            /* make sure we exited because the end of the array connection
               was reached.
            */
            if(*line == '\0') {
                LITERR("Missing ] in array connection");
                gc_end();
                return;
            }

            /* record the number of ports found for this connection */
            if(j < 1) {
                LITERR("Array connection must have at least one port");
                gc_end();
                return;
            }
            fast[0]->conn[i]->size = j;

            /*  At this point, the next time we get_token, we should get a % or a net name.
            We'll do that now, since when we enter the loop, we expect next_token
            to hold the next unprocessed token.
            */
            next_token = MIFget_token(&line, &next_token_type);

        }  /* ======  array connection processing   ====== */

        /* be careful about putting stuff here, there is a 'continue' used */
        /* in the processing of NULL connections above                     */
        /*  At this point, next_token should hold the next unprocessed token.  */

    } /******* for number of connections *******/


    /* *********************** */
    /*      Error Checks       */
    /* *********************** */

    /* check for too many connections */

    /* At this point, we should have eaten all the net connections, and left
       next_token holding the model name.  &line should be empty.
    */

    if(strcmp(next_token, model) != 0) {
        LITERR("Too many connections -- expecting model name but encountered other tokens.");
        gc_end();
        return;
    }

    /* check connection constraints */

    for(i = 0; i < DEVices[type]->DEVpublic.num_conn; i++) {

        conn_info = &(DEVices[type]->DEVpublic.conn[i]);

        if( (fast[0]->conn[i]->is_null) &&
                (! conn_info->null_allowed) ) {
            LITERR("Null found for connection where not allowed");
            gc_end();
            return;
        }

        if(conn_info->has_lower_bound) {
            if(fast[0]->conn[i]->size < conn_info->lower_bound) {
                LITERR("Too few ports in connection");
                gc_end();
                return;
            }
        }

        if(conn_info->has_upper_bound) {
            if(fast[0]->conn[i]->size > conn_info->upper_bound) {
                LITERR("Too many ports in connection");
                gc_end();
                return;
            }
        }
    }

    /* check model parameter constraints */
    /* some of these should probably be done in MIFgetMod() */
    /* to prevent multiple error messages                   */

    for(i = 0; i < DEVices[type]->DEVpublic.num_param; i++) {

        param_info = &(DEVices[type]->DEVpublic.param[i]);

        if(mdfast->param[i]->is_null) {
            char* emessage;

            if(! param_info->has_default) {
                if (param_info->type == MIF_STRING)
                    continue;   // Allow NULL
                emessage = tprintf("Parameter %s on model %s has no default",
                                   param_info->name, mdfast->gen.GENmodName);
                LITERR(emessage);
                tfree(emessage);
                gc_end();
                return;
            }
        }
        if((! mdfast->param[i]->is_null) && (param_info->is_array)) {
            if(param_info->has_conn_ref) {
                if(fast[0]->conn[param_info->conn_ref]->size != fast[0]->param[i]->size) {
                    LITERR("Array parameter size on model does not match connection size");
                    gc_end();
                    return;
                }
            }
        }
    }
    gc_end();
}



/* ********************************************************************* */



/*
MIFinit_inst

This function initializes the code model specific elements of the inst struct.
*/


static void  MIFinit_inst(
    MIFmodel *mdfast,       /* The model the instance is derived from */
    MIFinstance *fast)      /* The instance to initialize */
{

    int  mod_type;  /* type of this model */

    Mif_Conn_Info_t  *conn_info;

    int     i;


    /* get an index into the DEVices information structure */

    mod_type = mdfast->MIFmodType;


    /* allocate code model connector data in instance struct */

    fast->num_conn = DEVices[mod_type]->DEVpublic.num_conn;
    fast->conn = TMALLOC(Mif_Conn_Data_t *, fast->num_conn);

    for(i = 0; i < fast->num_conn; i++)
        fast->conn[i] = TMALLOC(Mif_Conn_Data_t, 1);

    /* initialize code model connector data */
    for(i = 0; i < fast->num_conn; i++) {
        conn_info = &(DEVices[mod_type]->DEVpublic.conn[i]);
        fast->conn[i]->name = conn_info->name;
        fast->conn[i]->description = conn_info->description;
        fast->conn[i]->is_null = MIF_TRUE;
        fast->conn[i]->size = 0;
        fast->conn[i]->port = NULL;
        switch(conn_info->direction) {
        case MIF_INOUT:
            fast->conn[i]->is_input =  MIF_TRUE;
            fast->conn[i]->is_output = MIF_TRUE;
            break;
        case MIF_IN:
            fast->conn[i]->is_input =  MIF_TRUE;
            fast->conn[i]->is_output = MIF_FALSE;
            break;
        case MIF_OUT:
            fast->conn[i]->is_input =  MIF_FALSE;
            fast->conn[i]->is_output = MIF_TRUE;
            break;
        default:
            printf("\nERROR - Impossible direction type in MIFinit_inst\n");
            controlled_exit(1);
        }
    }


    /* allocate and copy instance variable data to the instance */

    fast->num_inst_var = DEVices[mod_type]->DEVpublic.num_inst_var;
    fast->inst_var = TMALLOC(Mif_Inst_Var_Data_t *, fast->num_inst_var);

    for(i = 0; i < fast->num_inst_var; i++) {

        fast->inst_var[i] = TMALLOC(Mif_Inst_Var_Data_t, 1);

        if(DEVices[mod_type]->DEVpublic.inst_var[i].is_array) {
            fast->inst_var[i]->size = 0;
            fast->inst_var[i]->element = NULL;
            /* The code model allocates space for the data and sets the size */
        } else {
            fast->inst_var[i]->size = 1;
            fast->inst_var[i]->element = TMALLOC(Mif_Value_t, 1);
        }
    }


    /* copy model parameter data to the instance */

    fast->num_param = mdfast->num_param;
    fast->param = mdfast->param;

    /* initialize any additional instance data */
    fast->initialized = MIF_FALSE;
    fast->analog = MIF_FALSE;
    fast->event_driven = MIF_FALSE;
    fast->inst_index = 0;
    fast->callback = NULL;
}



/* ********************************************************************* */



/*
MIFget_port_type

This function gets the port type identifier and checks it for validity.  It also
replaces false default information in conn_info with the real info based upon
the discovered port type string.

When we call it, we expect that the next token type (sent from above)
should be the port type (MIF_STRING_TOK type).  That is, we should be sitting right
on the def of the port type (i.e. %vnam, %vd, %id, etc.)  Note that the parser
should have stripped the % already.

Upon return, this fcn should leave next_token holding the token *after* the
port type (i.e. the thing after vnam, v, vd, id, etc).
*/


static void
MIFget_port_type(
    CKTcircuit       *ckt,      /* circuit structure to put mod/inst structs in */
    INPtables        *tab,      /* symbol table for node names, etc.            */
    struct card      *current,  /* MUST be named 'current' for spice macros     */
    char             **line,
    char             **next_token,
    Mif_Token_Type_t *next_token_type,
    Mif_Port_Type_t  *port_type,
    char             **port_type_str,
    Mif_Conn_Info_t  *conn_info,  /* for faster access to conn info struct */
    Mif_Status_t     *status)
{

    Mif_Boolean_t    found_type;
    char             *temp;

    int              i;

    NG_IGNORE(ckt);
    NG_IGNORE(tab);

    if(**line == '\0') {
        LITERR("Missing connections on A device");
        *status = MIF_ERROR;
        return;
    }

    if(*next_token_type != MIF_STRING_TOK) {
        LITERR("Invalid port type specifier");
        *status = MIF_ERROR;
        return;
    }

    /* OK, so get the port type string from the token and read next token */

    temp = *next_token;
    *next_token = MIFget_token(line, next_token_type);

    /* check port type for validity */

    found_type = MIF_FALSE;

    for(i = 0; i < conn_info->num_allowed_types; i++) {
        if(strcmp(temp, conn_info->allowed_type_str[i]) == 0) {
            found_type = MIF_TRUE;
            *port_type = conn_info->allowed_type[i];
            *port_type_str = temp;
            break;
        }
    }

    if(! found_type) {
        LITERR("Port type is invalid");
        *status = MIF_ERROR;
    } else {

        /* Fix by SDB so that the netlist parser uses the actual nature
        of the port instead of the default state to decide if it is an array.  */
        /*
        if ( (*port_type == MIF_DIFF_VOLTAGE) ||
         (*port_type == MIF_DIFF_CURRENT) ||
         (*port_type == MIF_DIFF_CONDUCTANCE) ||
         (*port_type == MIF_DIFF_RESISTANCE) ) {
        conn_info->is_array = 1;
             }
             */

        *status = MIF_OK;
    }
}




/* ********************************************************************* */


/*
MIFget_port

This function processes a port being parsed, either single ended,
or both connections of a differential.

When we call this fcn, next_token should be the *first* netname in the port
net list.  Depending upon the type of port, this fcn should eat the appropriate
number of net tokens.

When we leave this fcn, next_token should hold the next token after the last
netname processed.
*/




static void
MIFget_port(
    CKTcircuit       *ckt,      /* circuit structure to put mod/inst structs in */
    INPtables        *tab,      /* symbol table for node names, etc.            */
    struct card      *current,  /* MUST be named 'current' for spice macros     */
    MIFinstance      *fast,     /* pointer to instance struct */
    char             **line,
    char             **next_token,
    Mif_Token_Type_t *next_token_type,
    Mif_Port_Type_t  def_port_type,
    char             *def_port_type_str,
    Mif_Conn_Info_t  *conn_info,  /* for faster access to conn info struct */
    int              conn_num,
    int              port_num,
    Mif_Status_t     *status)

{


    CKTnode         *pos_node[1];          /* positive connection node */
    CKTnode         *neg_node[1];          /* negative connection node */

    char            *node;


    /* allocate space in the instance data struct for this port */
    if(port_num == 0) {
        fast->conn[conn_num]->port = TMALLOC(Mif_Port_Data_t *, 1);
        fast->conn[conn_num]->port[0] = TMALLOC(Mif_Port_Data_t, 1);
    } else {
        fast->conn[conn_num]->port = TREALLOC(Mif_Port_Data_t *, fast->conn[conn_num]->port, port_num + 1);
        fast->conn[conn_num]->port[port_num] = TMALLOC(Mif_Port_Data_t, 1);
    }


    /* store the port type information in the instance struct */
    fast->conn[conn_num]->port[port_num]->type = def_port_type;
    fast->conn[conn_num]->port[port_num]->type_str = copy(def_port_type_str);

    /* check for a leading tilde on digital ports */
    if(*next_token_type == MIF_TILDE_TOK) {
        if((def_port_type != MIF_DIGITAL) && (def_port_type != MIF_USER_DEFINED)) {
            LITERR("ERROR - Tilde not allowed on analog nodes");
            *status = MIF_ERROR;
            return;
        }
        fast->conn[conn_num]->port[port_num]->invert = MIF_TRUE;

        /* eat the tilde and get the next token */
        *next_token = MIFget_token(line, next_token_type);
        if(**line == '\0') {
            LITERR("ERROR - Not enough ports");
            *status = MIF_ERROR;
            return;
        }
    } else
        fast->conn[conn_num]->port[port_num]->invert = MIF_FALSE;


    /* check for null port */
    if(*next_token_type == MIF_NULL_TOK) {

        /* make sure null is allowed */
        if(! conn_info->null_allowed) {
            LITERR("NULL connection found where not allowed");
            *status = MIF_ERROR;
            return;
        }

        /* set the (port specific) null flag to true */
        fast->conn[conn_num]->port[port_num]->is_null = MIF_TRUE;

        /* set input value to zero in case user code model refers to it */
        fast->conn[conn_num]->port[port_num]->input.rvalue = 0.0;

        /* eat the null token and return */
        *next_token = MIFget_token(line, next_token_type);
        *status = MIF_OK;
        return;
    } else {
        /* set the (port specific) null flag to false */
        fast->conn[conn_num]->port[port_num]->is_null = MIF_FALSE;
    }


    /* next token must be a node/instance identifier ... */
    if(*next_token_type != MIF_STRING_TOK) {
        LITERR("ERROR - Expected node/instance identifier");
        *status = MIF_ERROR;
        return;
    }

    /* Get the first connection or the voltage source name */

    switch(def_port_type) {

    case MIF_VOLTAGE:
    case MIF_DIFF_VOLTAGE:
    case MIF_CURRENT:
    case MIF_DIFF_CURRENT:
    case MIF_CONDUCTANCE:
    case MIF_DIFF_CONDUCTANCE:
    case MIF_RESISTANCE:
    case MIF_DIFF_RESISTANCE:

        *next_token = copy(*next_token);
        /* Call the spice3c1 function to put this node in the node list in ckt */
        INPtermInsert(ckt, next_token, tab, pos_node);

        /* store the equation number and node identifier */
        /* This is the equivalent of what CKTbindNode() does in 3C1 */
        fast->conn[conn_num]->port[port_num]->pos_node_str = *next_token;
        fast->conn[conn_num]->port[port_num]->smp_data.pos_node = pos_node[0]->number;

        break;

    case MIF_VSOURCE_CURRENT:
        *next_token = copy(*next_token);
        /* Call the spice3c1 function to put this vsource instance name in */
        /* the symbol table                                                */
        INPinsert(next_token, tab);

        /* Now record the name of the vsource instance for processing */
        /* later by MIFsetup.  This is equivalent to what INPpName    */
        /* does in 3C1.  Checking to see if the source is present in  */
        /* the circuit is deferred to MIFsetup as is done in 3C1.     */
        fast->conn[conn_num]->port[port_num]->vsource_str = *next_token;

        break;

    case MIF_DIGITAL:
    case MIF_USER_DEFINED:
        /* Insert data into event-driven info structs */
        EVTtermInsert(ckt,
                      fast,
                      *next_token,
                      def_port_type_str,
                      conn_num,
                      port_num,
                      &(current->error));
        if(current->error) {
            *status = MIF_ERROR;
            return;
        }
        break;

    default:

        /* impossible connection type */
        LITERR("INTERNAL ERROR - Impossible connection type");
        *status = MIF_ERROR;
        return;
    }

    /* get the next token */
    *next_token = MIFget_token(line, next_token_type);

    /* get other node if appropriate */
    switch(def_port_type) {

    case MIF_VOLTAGE:
    case MIF_CURRENT:
    case MIF_CONDUCTANCE:
    case MIF_RESISTANCE:
        /* These are single ended types, so default other node to ground */
        // This don't work dickhead, INPtermInsert tries to FREE(&node) K.A. Feb 27, 2000
        // which was not allocted
        node = TMALLOC(char, 2);// added by K.A. march 5th 2000

        *node = '0';    // added by K.A. March 5th 2000
        node[1] ='\0';  // added by K.A. March 5th 2000

        INPtermInsert(ckt, &node, tab, neg_node);

        fast->conn[conn_num]->port[port_num]->neg_node_str = node;
        fast->conn[conn_num]->port[port_num]->smp_data.neg_node = neg_node[0]->number;
        break;

    case MIF_DIFF_VOLTAGE:
    case MIF_DIFF_CURRENT:
    case MIF_DIFF_CONDUCTANCE:
    case MIF_DIFF_RESISTANCE:
        /* These are differential types, so get the other node */
        if((**line == '\0') || (*next_token_type != MIF_STRING_TOK)) {
            LITERR("ERROR - Expected node identifier");
            *status = MIF_ERROR;
            return;
        }
        *next_token = copy(*next_token);
        INPtermInsert(ckt, next_token, tab, neg_node);
        fast->conn[conn_num]->port[port_num]->neg_node_str = *next_token;
        fast->conn[conn_num]->port[port_num]->smp_data.neg_node = neg_node[0]->number;
        *next_token = MIFget_token(line, next_token_type);
        break;

    default:
        /* must be vsource name, digital, or user defined, so there is no other node */
        break;

    }

    *status = MIF_OK;
    return;
}


#undef MIFgettok
#undef MIFget_token

static
char *MIFgettok_gc(char **line)
{
    char *newtok = MIFgettok(line);
    alltokens[curtoknr++] = newtok;
    return newtok;
}

static
char *MIFget_token_gc(char **s, Mif_Token_Type_t *type)
{
    char *newtok = MIFget_token(s, type);
    alltokens[curtoknr++] = newtok;
    return newtok;
}

static
void gc_start(void)
{
    int i;
    for (i = 0; i < BSIZE_SP; i++)
        alltokens[i] = NULL;
    curtoknr = 0;
}

static
void gc_end(void)
{
    int i, j;
    for (i = 0; i < BSIZE_SP; i++) {
        /* We have multiple entries with the same address */
        for (j = i + 1; j < curtoknr; j++)
            if (alltokens[i] == alltokens[j])
                alltokens[j] = NULL;
        tfree(alltokens[i]);
    }
}

char *
copy_gc(char* in)
{
    char *newtok = copy(in);
    alltokens[curtoknr++] = newtok;
    return newtok;
}




