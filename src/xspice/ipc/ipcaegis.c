/*============================================================================
FILE    IPCaegis.c

MEMBER OF process XSPICE

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503

AUTHORS

    9/12/91  Steve Tynor

MODIFICATIONS

    <date> <person name> <nature of modifications>

SUMMARY

    Provides compatibility for the new XSPICE simulator to both the MSPICE user
    interface and BCP via ATESSE v.1 style AEGIS mailboxes.

INTERFACES


REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/

#ifdef IPC_AEGIS_MAILBOXES 

#include <assert.h>
#include <apollo/base.h>
#include <apollo/mbx.h>
#include <apollo/error.h>
#include "ngspice/memory.h"

#include "ngspice/ipc.h"


typedef unsigned char Buffer_char_t;

static status_$t        status;
typedef enum {
   IPC_MBX_UNINITIALIZED,
   IPC_MBX_INITIALIZED,
   IPC_MBX_CONNECTED_TO_CLIENT,
} Ipc_Mbx_State_t;

static void                     *mbx_handle;
static Ipc_Mbx_State_t          mbx_state = IPC_MBX_UNINITIALIZED;
static mbx_$server_msg_t        mbx_send_msg_buf;
static mbx_$server_msg_t        mbx_recieve_msg_buf;
static mbx_$server_msg_t        *mbx_ret_ptr;
static int                      mbx_ret_len;
static short                    mbx_chan;

#include "ngspice/ipcproto.h"

/*---------------------------------------------------------------------------*/

/*
ipc_transport_initialize_server

This function creates an Aegis mailbox, and if successful,
calls ipc_get_line to wait for the first record sent which is
assumed to be the batch output filename.
*/



Ipc_Status_t ipc_transport_initialize_server (server_name, m, p,
                                              batch_filename)
     char               *server_name;    /* The mailbox pathname */
     Ipc_Mode_t         m;               /* Mode - interactive or batch */
     Ipc_Protocol_t     p;               /* Protocol type */
     char               *batch_filename; /* Batch filename returned */
{
   int len;
/*   extern void *malloc(); */

   assert (p == IPC_PROTOCOL_V1);
   
   mbx_$create_server (server_name, strlen (server_name), mbx_$serv_msg_max,
                       1, &mbx_handle, &status);

   if (status.all != status_$ok) {
      fprintf (stderr,
               "ERROR: IPC: Error creating mailbox server \"%s\"\n",
               server_name);
      error_$print (status);
      mbx_state = IPC_MBX_UNINITIALIZED;
      return IPC_STATUS_ERROR;
   } else {
      mbx_state = IPC_MBX_INITIALIZED;
      /*
       * First record is the name of the batch filename - whether we're in
       * batch mode or not:
       */
      return ipc_get_line (batch_filename, &len, IPC_WAIT);
   }
   /*
    * shouldn't get here
    */
   assert (0);         
   return IPC_STATUS_ERROR;
}
/*---------------------------------------------------------------------------*/
Ipc_Status_t extract_msg (str, len)
     char *str;
     int  *len;
{
   *len = mbx_ret_len - mbx_$serv_msg_hdr_len;
   assert (*len >= 0);

   /*
    * null terminate before copy:
    */
   mbx_ret_ptr->data [*len] = '\0';
   strcpy (str, mbx_ret_ptr->data);

   return IPC_STATUS_OK;
}

/*---------------------------------------------------------------------------*/

/*
ipc_transport_get_line

This function reads data sent by a client over the mailbox
channel.  It also handles the initial opening of the
mailbox channel when requested by a client.
*/



Ipc_Status_t ipc_transport_get_line (str, len, wait)
     char               *str;   /* The string text read from IPC channel */
     int                *len;   /* The length of str */
     Ipc_Wait_t         wait;   /* Blocking or non-blocking */
{
   if (mbx_state == IPC_MBX_UNINITIALIZED) {
      fprintf (stderr,
               "ERROR: IPC: Attempted to read from non-initialized mailbox\n");
      return IPC_STATUS_ERROR;
   }

   assert ((mbx_state == IPC_MBX_CONNECTED_TO_CLIENT) ||
           (mbx_state == IPC_MBX_INITIALIZED));
   
   for (;;) {
      if (wait == IPC_WAIT) {
         mbx_$get_rec (mbx_handle, &mbx_recieve_msg_buf, mbx_$serv_msg_max,
                       &mbx_ret_ptr, &mbx_ret_len, &status);
      } else {
         mbx_$get_conditional (mbx_handle, &mbx_recieve_msg_buf,
                               mbx_$serv_msg_max, &mbx_ret_ptr, &mbx_ret_len,
                               &status);
         if (status.all == mbx_$channel_empty) {
            return IPC_STATUS_NO_DATA;
         }
      }
      
      if (status.all != status_$ok) {
         fprintf (stderr, "ERROR: IPC: Error reading from mailbox\n");
         error_$print (status);
         return IPC_STATUS_ERROR;
      }

      switch (mbx_ret_ptr->mt) {
      case mbx_$channel_open_mt:
         if (mbx_state == IPC_MBX_CONNECTED_TO_CLIENT) {
            /*
             * we're already connected to a client... refuse the connection
             */
            mbx_send_msg_buf.mt = mbx_$reject_open_mt;
         } else {
            mbx_send_msg_buf.mt = mbx_$accept_open_mt;
            mbx_state = IPC_MBX_CONNECTED_TO_CLIENT;
         }
         mbx_send_msg_buf.cnt = mbx_$serv_msg_hdr_len;
         mbx_chan = mbx_ret_ptr->chan;
         mbx_send_msg_buf.chan = mbx_chan; 

         mbx_$put_rec (mbx_handle, &mbx_send_msg_buf, mbx_$serv_msg_hdr_len,
                       &status);

         if (status.all != status_$ok) {
            fprintf (stderr, "ERROR: IPC: Error writing to mailbox\n");
            error_$print (status);
            return IPC_STATUS_ERROR;
         }

         /*
          * check to see if there was a message buried in the open request:
          */
         if (mbx_ret_len > mbx_$serv_msg_hdr_len) {
            return extract_msg (str, len);
         }
         break;
      case mbx_$eof_mt:
         mbx_chan = mbx_ret_ptr->chan;
	 mbx_$deallocate(mbx_handle, mbx_chan, &status);

         if (status.all != status_$ok) {
            fprintf (stderr, "ERROR: IPC: Error deallocating mailbox\n");
            error_$print (status);
            return IPC_STATUS_ERROR;
         }

         mbx_state = IPC_MBX_INITIALIZED;
         return IPC_STATUS_EOF;
         break;
      case mbx_$data_mt:
         assert (mbx_state == IPC_MBX_CONNECTED_TO_CLIENT);
         return extract_msg (str, len);
         break;
      case mbx_$data_partial_mt:
         fprintf (stderr, "ERROR: IPC: Recieved partial data message - ignored\n");
         break;
      default:
         fprintf (stderr, "ERROR: IPC: Bad message type (0x%x) recieved\n",
                  mbx_ret_ptr->mt);
      }
   }
   return IPC_STATUS_ERROR;
}
 
/*---------------------------------------------------------------------------*/

/*
ipc_transport_terminate_server

This function calls ipc\_transport\_get\_line until it
receives an EOF from the client, which concludes the
communication.
*/


Ipc_Status_t ipc_transport_terminate_server ()
{
   char buffer[300];
   int len;
   Ipc_Status_t status;

   do {
      status = ipc_transport_get_line (buffer, &len, IPC_WAIT);
   } while ((status != IPC_STATUS_ERROR) &&
	    (status != IPC_STATUS_EOF));
   return status;
}

/*---------------------------------------------------------------------------*/

/*
ipc_transport_send_line

This function sends a message to the current client through
the mailbox channel.
*/


Ipc_Status_t ipc_transport_send_line (str, len)
     char *str;    /* The bytes to send */
     int len;      /* The number of bytes from str to send */
{
   long cnt;
   
   if (mbx_state != IPC_MBX_CONNECTED_TO_CLIENT) {
      fprintf (stderr,
               "ERROR: IPC: Attempted to write to non-open mailbox\n");
      return IPC_STATUS_ERROR;
   }

   mbx_send_msg_buf.mt = mbx_$data_mt;
   if (mbx_$serv_msg_hdr_len + len > mbx_$serv_msg_max) {
      fprintf (stderr,
	       "ERROR: IPC: send_line message too long - truncating\n");
      len = mbx_$serv_msg_max - mbx_$serv_msg_hdr_len;
   }
   
   mbx_send_msg_buf.cnt = mbx_$serv_msg_hdr_len + len;
   mbx_send_msg_buf.chan = mbx_chan;
   memcpy (mbx_send_msg_buf.data, str, len);

   cnt = mbx_send_msg_buf.cnt;
   mbx_$put_rec (mbx_handle, &mbx_send_msg_buf, cnt, &status);
   
   if (status.all != status_$ok) {
      fprintf (stderr, "ERROR: IPC: Error writing to mailbox\n");
      error_$print (status);
      return IPC_STATUS_ERROR;
   }   
   return IPC_STATUS_OK;
}

#else

int intDummy1;

#endif  /* IPC_AEGIS_MAILBOXES */
