/*
 * Steve Tynor
 *
 * Generic Interprocess Communication module
 *
 * Used for debugging in absense of IPC interface.
 *
 */

#include "ngspice/ngspice.h"

#ifdef IPC_DEBUG_VIA_STDIO

#include <stdio.h>


#include "ngspice/ipc.h"

#include "ngspice/ipcproto.h"

#include <assert.h>   /*   12/1/97 jg  */
 
/*---------------------------------------------------------------------------*/
Ipc_Status_t ipc_transport_initialize_server (
     char               *server_name,
     Ipc_Mode_t         m,
     Ipc_Protocol_t     p,
     char               *batch_filename )
{
   NG_IGNORE(server_name);
   NG_IGNORE(p);  
   NG_IGNORE(batch_filename);

   assert (m == IPC_MODE_INTERACTIVE);
   printf ("INITIALIZE_SERVER\n");
   return IPC_STATUS_OK;
}

/*---------------------------------------------------------------------------*/
Ipc_Status_t ipc_transport_get_line (
     char               *str,
     int                *len,
     Ipc_Wait_t         wait )
{
   NG_IGNORE(wait);

   printf ("GET_LINE\n");
   fgets (str, 512, stdin);
   char *tmp = strchr(str, '\n');
   if (tmp)
       *tmp = '\0';
   *len = (int) strlen (str);
   return IPC_STATUS_OK;
}

/*---------------------------------------------------------------------------*/
Ipc_Status_t ipc_transport_send_line (
     char *str,
     int len )
{
   int i;

   printf ("SEND_LINE: /");
   for (i = 0; i < len; i++)
      putchar (str[i]);
   printf ("/\n");
   return IPC_STATUS_OK;
}

/*---------------------------------------------------------------------------*/
Ipc_Status_t ipc_transport_terminate_server (void)
{
return IPC_STATUS_OK;
}


#endif  /* IPC_DEBUG_VIA_STDIO */
