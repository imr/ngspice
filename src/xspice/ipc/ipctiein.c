/*============================================================================
FILE    IPCtiein.c

MEMBER OF process XSPICE

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503

AUTHORS

    9/12/91  Bill Kuhn

MODIFICATIONS

    <date> <person name> <nature of modifications>

SUMMARY

    Provides a protocol independent interface between the simulator
    and the IPC method used to interface to CAE packages.

INTERFACES

    g_ipc   (global variable)

    ipc_handle_stop()
    ipc_handle_returni()
    ipc_handle_mintime()
    ipc_handle_vtrans()
    ipc_send_stdout()
    ipc_send_stderr()
    ipc_send_std_files()
    ipc_screen_name()
    ipc_get_devices()
    ipc_free_devices()
    ipc_check_pause_stop()

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/


#define CONFIG

#include "ngspice/ngspice.h"
#include "ngspice/inpdefs.h"
#include "ngspice/gendefs.h"
#include "ngspice/cktdefs.h"
#include "bjt/bjtdefs.h"
#include "jfet/jfetdefs.h"
#include "mos1/mos1defs.h"
#include "mos2/mos2defs.h"
#include "mos3/mos3defs.h"
#include "ngspice/mifproto.h"
#include "ngspice/ipc.h"
#include "ngspice/ipctiein.h"



/*
Global variable g_ipc is used by the SPICE mods that take care of
interprocess communications activities.
*/


Ipc_Tiein_t  g_ipc = {
    IPC_FALSE,                  /* enabled */
    IPC_MODE_INTERACTIVE,       /* mode */
    IPC_ANAL_DCOP,              /* analysis mode */
    IPC_FALSE,                  /* parse_error */
    IPC_FALSE,                  /* run_error */
    IPC_FALSE,                  /* errchk_sent */
    IPC_FALSE,                  /* returni */
    0.0,                        /* mintime */
    0.0,                        /* lasttime */
    0.0,                        /* cpu time */
    NULL,                       /* send array */
    NULL,                       /* log file */
    {                           /* vtrans struct */
        0,                          /* size */
        NULL,                       /* vsrc_name array */
        NULL,                       /* device_name array */
    },
    IPC_FALSE,                  /* stop analysis */
};



/*
ipc_handle_stop

This function sets a flag in the g_ipc variable to signal that
a stop message has been received over the IPC channel.
*/

void ipc_handle_stop(void)
{
    g_ipc.stop_analysis = IPC_TRUE;
}


/*
ipc_handle_returni

This function sets a flag in the g_ipc variable to signal that
a message has been received over the IPC channel specifying that
current values are to be returned in the results data sets.
*/

void ipc_handle_returni(void)
{
    g_ipc.returni = IPC_TRUE;
}


/*
ipc_handle_mintime

This function sets a value in the g_ipc variable that specifies
how often data is to be returned as it is computed.  If the
simulator takes timestep backups, data may still be returned
more often that that specified by 'mintime' so that glitches
are not missed.
*/

void ipc_handle_mintime(double time)
{
    g_ipc.mintime = time;
}



/*
ipc_handle_vtrans

This function processes arguments from a #VTRANS card received over
the IPC channel.  The data on the card specifies that a particular
zero-valued voltage source name should be translated to the specified
instance name for which it was setup to monitor currents.
*/

void ipc_handle_vtrans(
    char *vsrc,   /* The name of the voltage source to be translated */
    char *dev)    /* The device name the vsource name should be translated to */
{
    int i;
    int size;


    if(g_ipc.vtrans.size == 0) {
        g_ipc.vtrans.size = 1;
        g_ipc.vtrans.vsrc_name = TMALLOC(char *, 1);
        g_ipc.vtrans.device_name = TMALLOC(char *, 1);
        g_ipc.vtrans.vsrc_name[0] = MIFcopy(vsrc);
        g_ipc.vtrans.device_name[0] = MIFcopy(dev);
    }
    else {
        g_ipc.vtrans.size++;

        size = g_ipc.vtrans.size;
        i = g_ipc.vtrans.size - 1;

        g_ipc.vtrans.vsrc_name = TREALLOC(char *, g_ipc.vtrans.vsrc_name, size);
        g_ipc.vtrans.device_name = TREALLOC(char *, g_ipc.vtrans.device_name, size);
        g_ipc.vtrans.vsrc_name[i] = MIFcopy(vsrc);
        g_ipc.vtrans.device_name[i] = MIFcopy(dev);
    }
}



/*
ipc_send_stdout

This function sends the data written to stdout over the IPC channel.
This stream was previously redirected to a temporary file during
the simulation.
*/

void ipc_send_stdout(void)
{
    int c;
    int len;

    char    buf[IPC_MAX_LINE_LEN+1];

    /* rewind the redirected stdout stream */
    rewind(stdout);

    /* Begin reading from the top of file and send lines */
    /* over the IPC channel.                             */

    /* Don't send newlines.  Also, if line is > IPC_MAX_LINE_LEN chars */
    /* we must wrap it because Mspice can't handle it              */

    len = 0;
    while( (c=fgetc(stdout)) != EOF) {
        if(c != '\n') {
            buf[len] = (char) c;
            len++;
        }
        if((c == '\n') || (len == IPC_MAX_LINE_LEN)) {
            buf[len] = '\0';
            ipc_send_line(buf);
            len = 0;
        }
    }
    if(len > 0) {
        buf[len] = '\0';
        ipc_send_line(buf);
    }

    /* Finally, rewind file again to discard the data already sent */
    rewind(stdout);
}


/*
ipc_send_stderr

This function sends the data written to stderr over the IPC channel.
This stream was previously redirected to a temporary file during
the simulation.
*/

void ipc_send_stderr(void)
{
    int c;
    int len;

    char    buf[IPC_MAX_LINE_LEN+1];

    /* rewind the redirected stderr stream */
    rewind(stderr);

    /* Begin reading from the top of file and send lines */
    /* over the IPC channel.                             */

    /* Don't send newlines.  Also, if line is > IPC_MAX_LINE_LEN chars */
    /* we must wrap it because Mspice can't handle it              */

    len = 0;
    while( (c=fgetc(stderr)) != EOF) {
        if(c != '\n') {
            buf[len] = (char) c;
            len++;
        }
        if((c == '\n') || (len == IPC_MAX_LINE_LEN)) {
            buf[len] = '\0';
            ipc_send_line(buf);
            len = 0;
        }
    }
    if(len > 0) {
        buf[len] = '\0';
        ipc_send_line(buf);
    }

    /* Finally, rewind file again to discard the data already sent */
    rewind(stderr);
}


/*
ipc_send_std_files

This function sends the data written to stdout and stderr over the
IPC channel.  These streams were previously redirected to temporary
files during the simulation.
*/

Ipc_Status_t ipc_send_std_files(void)
{
    ipc_send_stdout();
    ipc_send_stderr();

    return(ipc_flush());
}



/*
ipc_screen_name

This function screens names of instances and nodes to limit the
data returned over the IPC channel.
*/

Ipc_Boolean_t ipc_screen_name(char *name, char *mapped_name)
{
    char    *endp;
    int     i;
    int     len;
    long    l;

    /* Return FALSE if name is in a subcircuit */
    for(i = 0; name[i] != '\0'; i++) {
        if(name[i] == ':')
            return(IPC_FALSE);
    }

    /* Determine if name is numeric and what value is */
    l = strtol(name, &endp, 10);

    /* If numeric */
    if(*endp == '\0') {
        /* Return FALSE if >100,000 -> added by ms_server in ATESSE 1.0 */
        if(l >= 100000)
            return(IPC_FALSE);
        /* Otherwise, copy name to mapped name and return true */
        else {
            strcpy(mapped_name,name);
            return(IPC_TRUE);
        }
    }

    /* If node is an internal node from a semiconductor (indicated by a      */
    /* trailing #collector, #source, ...), do not return its current.        */
    /* Otherwise, map to upper case and eliminate trailing "#branch" if any. */
    for(i = 0; name[i]; i++) {
        if(name[i] == '#') {
            if(strcmp(name + i, "#branch") == 0)
                break;
            else
                return(IPC_FALSE);
        }
        else {
            if(islower_c(name[i]))
                mapped_name[i] = toupper_c(name[i]);
            else
                mapped_name[i] = name[i];
        }
    }
    mapped_name[i] = '\0';
    len = i;

    /* If len != 8 or 6'th char not equal to $, then doesn't need vtrans */
    /* Otherwise, translate to device name that it monitors */
    if(len != 8)
        return(IPC_TRUE);
    else if(name[5] != '$')
        return(IPC_TRUE);
    else {
        /* Scan list of prefixes in VTRANS table and convert name */
        for(i = 0; i < g_ipc.vtrans.size; i++) {
            if(strncmp(mapped_name, g_ipc.vtrans.vsrc_name[i], 5) == 0) {
                strcpy(mapped_name, g_ipc.vtrans.device_name[i]);
                return(IPC_TRUE);
            }
        }
        return(IPC_TRUE);
    }

}



/*
ipc_get_devices

This function is used to setup the OUTinterface data structure that
determines what instances will have data returned over the IPC channel.
*/


int ipc_get_devices(
    CKTcircuit  *ckt,         /* The circuit structure */
    char        *device,      /* The device name as it appears in the info struct */
    char        ***names,     /* Array of name strings to be built */
    double      **modtypes)   /* Array of types to be built */
{
    int         index;
    int         num_instances;
    GENmodel    *model;
    GENinstance *here;
    char        *inst_name;
    int         inst_name_len;
    int         i;

    BJTmodel         *BJTmod;
    JFETmodel        *JFETmod;
    MOS1model        *MOS1mod;
    MOS2model        *MOS2mod;
    MOS3model        *MOS3mod;

    /* Initialize local variables */
    num_instances = 0;

    /* Get the index into the circuit structure linked list of models */
    index = INPtypelook(device);

    /* Iterate through all models of this type */
    for(model = ckt->CKThead[index]; model; model = model->GENnextModel) {

        /* Iterate through all instance of this model */
        for(here = model->GENinstances; here; here = here->GENnextInstance) {

            /* Get the name of the instance */
            inst_name = here->GENname;
            inst_name_len = (int) strlen(inst_name);

            /* Skip if it is a inside a subcircuit */
            for(i = 0; i < inst_name_len; i++)
                if(inst_name[i] == ':')
                    break;
            if(i < inst_name_len)
                continue;

            /* Otherwise, add the name to the list */
            num_instances++;
            if(num_instances == 1)
                *names = TMALLOC(char *, 1);
            else
                *names = TREALLOC(char *, *names, num_instances);
            (*names)[num_instances-1] = MIFcopy(inst_name);

            /* Then get the type if it is a Q J or M */
            if(num_instances == 1)
                *modtypes = TMALLOC(double, 1);
            else
                *modtypes = TREALLOC(double, *modtypes, num_instances);

            if(strcmp(device,"BJT") == 0) {
                BJTmod = (BJTmodel *) model;
                (*modtypes)[num_instances-1] = BJTmod->BJTtype;
            }
            else if(strcmp(device,"JFET") == 0) {
                JFETmod = (JFETmodel *) model;
                (*modtypes)[num_instances-1] = JFETmod->JFETtype;
            }
            else if(strcmp(device,"Mos1") == 0) {
                MOS1mod = (MOS1model *) model;
                (*modtypes)[num_instances-1] = MOS1mod->MOS1type;
            }
            else if(strcmp(device,"Mos2") == 0) {
                MOS2mod = (MOS2model *) model;
                (*modtypes)[num_instances-1] = MOS2mod->MOS2type;
            }
            else if(strcmp(device,"Mos3") == 0) {
                MOS3mod = (MOS3model *) model;
                (*modtypes)[num_instances-1] = MOS3mod->MOS3type;
            }
            else {
                (*modtypes)[num_instances-1] = 1.0;
            }

        } /* end for all instances */
    } /* end for all models */

    return(num_instances);
}



/*
ipc_free_devices

This function frees temporary data created by ipc_get_devices().
*/


void ipc_free_devices(
    int         num_items,    /* Number of things to free */
    char        **names,     /* Array of name strings to be built */
    double      *modtypes)   /* Array of types to be built */
{
    int         i;

    for(i = 0; i < num_items; i++)
	{
        FREE(names[i]);
		names[i] = NULL;
	}

    if(num_items > 0) 
	{
        FREE(names);
        FREE(modtypes);

		names = NULL;
		modtypes = NULL;
    }
}


/*
ipc_check_pause_stop

This function is called at various times during a simulation to check
for incoming messages of the form >STOP or >PAUSE signaling that
simulation should be stopped or paused.  Processing of the messages
is handled by ipc_get_line().
*/

void ipc_check_pause_stop(void)
{
    char        buf[1025];
    int         len;

    /* If already seen stop analysis, don't call ipc_get_line, just return. */
    /* This is provided so that the function can be called multiple times */
    /* during the process of stopping */
    if(g_ipc.stop_analysis)
        return;

    /* Otherwise do a non-blocking call to ipc_get_line() to check for messages. */
    /* We assume that the only possible messages at this point are >PAUSE */
    /* and >STOP, so we don't do anything with the returned text if any */
    ipc_get_line(buf, &len, IPC_NO_WAIT);
}

