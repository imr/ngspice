/*=============================================================================

  FILE IPCsockets.c 

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503

  AUTHOR
    Stefan Roth  July 1991

  MODIFICATIONS
    none

  SUMMARY
  Generic Interprocess Communication module
  Provides compatibility for the new SPICE simulator to both the MSPICE user
  interface and BCP (via ATESSE v.1 style AEGIS mailboxes) and the new ATESSE
  v.2 Simulator Interface and BCP (via BSD Sockets). This file contains the
  BSD sockets version.
  The Simulator is the server, while the SI and BCP will be the clients.
  

  INTERFACES       

    FILE                 ROUTINE CALLED     

    IPC.c                ipc_get_line();


  REFERENCED FILES

    Outputs to stderr.


=============================================================================*/


/*=============================================================================

  DESCRIPTION OF FUNCTIONALITY:

  Outline of Initialize_Server function:
    create socket;
    bind name to socket;
    getsockname;
    listen;
    sock_state = IPC_SOCK_INITIALIZED;
    return ipc_get_line ();


  Format of a message line:
    bytes  description
    -----  -------------------
       0   recognition character for beginning of line; value is BOL_CHAR.
     1-4   message length (not including bytes 0-4); 32 bits in htonl
           format;
           if value = -1, then EOF and socket should be closed.
    5-N+5  message body of length specified in bytes 1-4.

    The bytes before the message body are the message header. The header
    length is specified as SOCK_MSG_HDR_LEN bytes.


  Outline of Get_Line function:
      read 5 characters;
      verify that first char is BOL_CHAR;
      interpret message length (N) from bytes 1-4;
      do error checking on message header bytes;
      read N characters as message body;
      do error checking on message body read;


  Outline of Send_Line function:
      write BOL_CHAR;
      write 4-byte message body length
      write message body (N bytes)
      do error checking after each write operation


  Outline of Terminate_Server function:
      Continue to read lines (with ipc_transport_get_line) and ignore
	  them until socket EOF is reached;
      Close the socket.


=============================================================================*/

#include "ngspice/ngspice.h"

#ifdef IPC_UNIX_SOCKETS

/*=== INCLUDE FILES ===*/

#include <assert.h>
#include <errno.h>

#include "ngspice/ipc.h"
#include "ngspice/ipctiein.h"


/*=== TYPE DEFINITIONS ===*/  

typedef enum {
   IPC_SOCK_UNINITIALIZED,
   IPC_SOCK_INITIALIZED,
   IPC_SOCK_CONNECTED_TO_CLIENT
} Ipc_Sock_State_t;


/*=== LOCAL VARIABLES ===*/          

static int                     sock_desc;    /* socket descriptor */
static int                     msg_stream;   /* socket stream     */
static Ipc_Sock_State_t        sock_state = IPC_SOCK_UNINITIALIZED;


/*=== INCLUDE FILES ===*/

#include "ngspice/ipcproto.h"

/*=============================================================================

FUNCTION ipc_transport_initialize_server

AUTHORS                      

    July 1991   Stefan Roth

MODIFICATIONS   

    NONE

SUMMARY

    Creates and opens the BSD socket of the server. Listens for requests
    by a client and then reads the first line message.

INTERFACES
            
    Called by:  (IPC.c) ipc_initialize_server();

RETURNED VALUE
    
    Ipc_Status_t - returns status of the socket connection.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE
     
=============================================================================*/

Ipc_Status_t
ipc_transport_initialize_server (
     char               *server_name,     /* not used                      */
     Ipc_Mode_t         mode,             /* not used                      */
     Ipc_Protocol_t     protocol,         /* IN - only used in assert      */
     char               *batch_filename ) /* OUT - returns a value         */
     /* Note that unused parameters are required to maintain compatibility */
     /* with version 1 (mailboxes) functions of the same names.            */
{
  struct        sockaddr_in server;     /* Server specifications for socket*/
  socklen_t  server_length;             /* Size of server structure        */
  int  port_num;                        /* Port number converted from server_name */

  NG_IGNORE(mode);
  NG_IGNORE(protocol);

  /* assert (protocol == IPC_PROTOCOL_V2); */ /* allow v1 protocol - wbk */
  assert (sock_state == IPC_SOCK_UNINITIALIZED);

  /* convert server_name (from atesse_xspice invocation line) to a port */
  /* number */
  port_num = atoi(server_name);
  if((port_num > 0) && (port_num < 1024)) {
    /* Reserved port number */
    perror ("ERROR: IPC  Port numbers below 1024 are reserved");
    sock_state = IPC_SOCK_UNINITIALIZED;
    return IPC_STATUS_ERROR;
  }


  sock_desc = socket (AF_INET, SOCK_STREAM, 0);

  if (sock_desc < 0) {
    /* Unsuccessful socket opening */
    perror ("ERROR: IPC  Creating socket");
    sock_state = IPC_SOCK_UNINITIALIZED;
    return IPC_STATUS_ERROR;
  }

  /* Socket opened successfully */
  
  server.sin_family      = AF_INET;
  server.sin_addr.s_addr = INADDR_ANY;
  server.sin_port        = SOCKET_PORT;
     
  server_length = sizeof (server);
  if (bind (sock_desc, (struct sockaddr *)&server, server_length)
      < 0) {
    fprintf (stderr, "ERROR: IPC: Bind unsuccessful\n");
    perror ("ERROR: IPC");
    sock_state = IPC_SOCK_UNINITIALIZED;
    return IPC_STATUS_ERROR;
  }

  if (getsockname (sock_desc, (struct sockaddr *)&server, &server_length)
      < 0) {
    fprintf (stderr, "ERROR: IPC: getting socket name\n");
    perror ("ERROR: IPC");
    sock_state = IPC_SOCK_UNINITIALIZED;
    return IPC_STATUS_ERROR;
  }

  fprintf (stderr, "Socket port %d.\n", ntohs(server.sin_port));

  listen (sock_desc, 5);

  sock_state = IPC_SOCK_INITIALIZED;

  /* Socket ok to use now */

  /*
   * First record is the name of the batch filename if we're in batch mode.
   */
  
  if(g_ipc.mode == IPC_MODE_BATCH) {
    int len;
    return ipc_get_line (batch_filename, &len, IPC_WAIT);
  }

  /* Return success */
  return IPC_STATUS_OK;

}   /* end ipc_transport_initialize_server */



/*=============================================================================

FUNCTION bytes_to_integer

AUTHORS                      

    July 1991   Stefan Roth

MODIFICATIONS   

    NONE

SUMMARY

  Convert four bytes at START in the string STR
  to a 32-bit unsigned integer. The string is assumed
  to be in network byte order and the returned value
  is converted to host byte order (with ntohl).

INTERFACES

  Local to this file.
  Called by:  ipc_transport_get_line();

RETURNED VALUE
    
    u_long - unsigned 32 bit integer

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE
     
=============================================================================*/

/* FIXME,
 *   this is seriously broken,
 *   once it was probably based upon htonl(),
 *     yet with broken types
 *   then the game has changed and strtoul() was used
 *     with a ascii representation of the length
 *     (probably as a hacky workaround, because it proved unreliable)
 *     but the buffer is not terminated properly
 * Fix this when needed, currently this functionality looks like
 *   an unused ancient artefact
 * Fix it with regard to ipc_transport_get_line() and ipc_transport_send_line()
 *   and in concert with the actual user at the other side of the socket
 */
static u_long
bytes_to_integer (
     char   *str,	/* IN - string that contains the bytes to convert  */
     int    start )	/* IN - index into string where bytes are          */
{
  uint32_t u;           /* Value to be returned                            */
  char   buff[4];       /* Transfer str into buff to word align reqd data  */
  int    index;         /* Index into str and buff for transfer            */

  /* Get the str+start character and cast it into a u_long and put
     the value through the network-to-host-short converter and store
     it in the variable u.   */

  index = 0;
  while (index < (int) sizeof(u)) {
    buff[index] = str[index+start];
    index++;
  }
/*  u = ntohl (*((u_long *) buff)); */
  u = (uint32_t) strtoul(buff, NULL, 10);

  return u;
}   /* end bytes_to_integer */



/*=============================================================================

FUNCTION handle_socket_eof

AUTHORS                      

    July 1991   Stefan Roth

MODIFICATIONS   

    NONE

SUMMARY

  Do processing when the socket reaches EOF or when a message from the
  client states that EOF has been reached.

INTERFACES

  Local to this file.
  Called by:  ipc_transport_get_line();

RETURNED VALUE
    
  Ipc_Status_t - always IPC_STATUS_EOF

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE
     
=============================================================================*/


static Ipc_Status_t
handle_socket_eof (void)
{
  close (msg_stream);
  close (sock_desc);

  sock_state = IPC_SOCK_UNINITIALIZED;

  return IPC_STATUS_EOF;
}   /* handle_socket_eof */



/*=============================================================================

FUNCTION read_sock

AUTHORS                      

    July 1991   Stefan Roth

MODIFICATIONS   

    NONE

SUMMARY

  Read N bytes from a socket. Only returns when the read had an error,
  when 0 bytes (EOF) could be read, or LENGTH bytes are read.

INTERFACES

  Local to this file.
  Called by:  ipc_transport_get_line();

RETURNED VALUE
    
  int - Returns the total number of bytes read.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE
     
=============================================================================*/


static int
read_sock (
     int          stream,   /* IN - Socket stream                          */
     char         *buffer,  /* OUT - buffer to store incoming data         */
     int          length,   /* IN - Number of bytes to be read             */
     Ipc_Wait_t   wait,     /* IN - type of read operation                 */
     int          flags )   /* IN - Original socket flags for blocking read */
{
  int   count;			/* Number of bytes read with last `read`    */
  int   totalcount;		/* total number of bytes read               */
  char  *buf2;
  
/*  count = 0;                                                               */
/*  while (count < length) {                                                 */
/*    buffer[count] = 'x';                                                   */
/*    count++;                                                               */
/*  }                                                                        */
  count = (int) read (stream, buffer, (size_t) length);
  if (wait == IPC_NO_WAIT) {
    fcntl (stream, F_SETFL, flags); /* Revert to blocking read           */
  }
  if ((count <= 0) || (count == length)) {
    /* If error or if read in reqd number of bytes:                        */
    return count;
  } else {
    /* Only got some of the bytes requested                                */
    totalcount = count;
    buf2 = &buffer[totalcount];
    length = length - count;
    while (length > 0) {
      count = (int) read (stream, buf2, (size_t) length);
      if (count <= 0)		/* EOF or read error                 */
	break;
      totalcount = totalcount + count;
      buf2 = &buffer[totalcount];
      length = length - count;
    }
    if (length != 0) {
      fprintf (stderr, "WARNING: READ_SOCK read %d bytes instead of %d\n",
	       totalcount, totalcount + length);
    }
    return totalcount;
  }
}				/* end read_sock */



/*=============================================================================

FUNCTION ipc_transport_get_line

AUTHORS                      

    July 1991   Stefan Roth

MODIFICATIONS   

    NONE

SUMMARY

  Main function for reading one line from a socket. Requires that the
  socket be open. Lines are mostly SPICE code, but commands may also
  be embedded in the socket data and they are interpreted by this function.
  Therefore, this function may cause the socket to be closed.

INTERFACES
            
  Called by:  ipc_transport_terminate_server();
              (IPC.c) ipc_get_line();

RETURNED VALUE
    
  Ipc_Status_t - returns status of the read operation

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE
     
=============================================================================*/


Ipc_Status_t
ipc_transport_get_line (
     char               *str,    /* returns the result, null terminated    */
     int                *len,    /* length of str passed IN and passed OUT */
     Ipc_Wait_t         wait )   /* IN - wait or dont wait on incoming msg */
{
  int count = 0;                     /* number of bytes read                   */
  int message_length;            /* extracted from message header          */

  if (sock_state == IPC_SOCK_UNINITIALIZED) {
    fprintf (stderr,
	    "ERROR: IPC: Attempted to read from uninitialized socket\n");
    return IPC_STATUS_ERROR;
  }
  
  assert ((sock_state == IPC_SOCK_CONNECTED_TO_CLIENT) ||
	  (sock_state == IPC_SOCK_INITIALIZED));

  if (sock_state == IPC_SOCK_INITIALIZED) {
    /* We have an open socket but have not connected to a client.          */
    /* Accept a connection from a client.                                  */
    msg_stream = accept (sock_desc, (struct sockaddr *)0, (socklen_t*)0);

    if (msg_stream == -1) {
      fprintf (stderr, "ERROR: IPC: Server accepting request\n");
      perror ("ERROR: IPC");
      return IPC_STATUS_ERROR;
    }
    sock_state = IPC_SOCK_CONNECTED_TO_CLIENT;
  }
  /*-----------------------------------------------------------------------*/
  /* First read in the message header.                                     */
  {
    int flags;
    flags = fcntl(msg_stream, F_GETFL, NULL);    /* Blocking read mode     */

    if (wait == IPC_WAIT) {
      /* Block here and wait for the next message                          */
      count = read_sock (msg_stream, str, SOCK_MSG_HDR_LEN, wait, flags);
      if (count == 0) {
	/* EOF, will this ever happen?                                     */
	/* fprintf (stderr, "WARNING: IPC: Reached eof on socket\n");  */
	return handle_socket_eof ();
      }
    } else if (wait == IPC_NO_WAIT) {
      /* Read message, but do not wait if none available.                  */

      fcntl (msg_stream, F_SETFL, flags | O_NDELAY);
      count = read_sock (msg_stream, str, SOCK_MSG_HDR_LEN, wait, flags);

      if (count == 0) {
	/* EOF, will this ever happen?                                     */
	/* fprintf (stderr, "WARNING: IPC: Reached eof on socket\n");  */
	return handle_socket_eof ();
      } else if (count == -1) {
	if (errno == EWOULDBLOCK) {
	  return IPC_STATUS_NO_DATA;
	}
      }

    } else {
      /* Serious problem, since it is not reading anything.                */
      fprintf (stderr,
	       "ERROR: IPC: invalid wait arg to ipc_transport_get_line\n");
    }
  }
  
  /* Do more error checking on the read in values of the message header:   */
  if (count == -1) {
    fprintf (stderr, "ERROR: IPC: Reading from socket\n");
    perror ("ERROR: IPC");
    return IPC_STATUS_ERROR;
  } else if (str[0] != BOL_CHAR) {
    fprintf (stderr,
	     "ERROR: IPC: Did not find beginning of message header (%c)\n",
	     str[0]);
    return IPC_STATUS_ERROR;
  } else if ((message_length = (int) bytes_to_integer (str, 1)) == -1) {
    /* fprintf (stderr, "WARNING: IPC: Reached eof on socket\n");      */
    return handle_socket_eof ();
  } else if (message_length == 0) {
    *len = 0;
    return IPC_STATUS_NO_DATA;

/*  Invalid test... delete - wbk
  } else if (message_length > *len) {
    fprintf (stderr,
	     "ERROR: IPC: Buffer (%d) is too short for message (%d)\n",
	     *len, message_length);
    return IPC_STATUS_ERROR;
*/

  }

  /*-----------------------------------------------------------------------*/
  /* Now read in the message body.                                         */
  /* Always block here since the message header was already read and       */
  /* we must get the body.                                                 */

  *len = message_length;
  count = read_sock (msg_stream, str, message_length, IPC_WAIT, 0);
  if (count == 0) {
    /* EOF, will this ever happen? */
    /* fprintf (stderr,                                                    */
    /*          "WARNING: IPC: Reached eof in message body on socket\n");*/
    return handle_socket_eof ();
  } else if (count == -1) {
    fprintf (stderr, "ERROR: IPC: reading message body from socket\n");
    perror ("ERROR: IPC");
    return IPC_STATUS_ERROR;
  }
  
  /* Looks like we have a valid message here. Put in the string terminator. */
  *len = count;
  str[count] = '\0';

  return IPC_STATUS_OK;

}   /* end ipc_transport_get_line */
 


/*=============================================================================

FUNCTION ipc_transport_send_line

AUTHORS                      

    July 1991   Stefan Roth

MODIFICATIONS   

    NONE

SUMMARY
  Send a line of information. First sends a message header and
  then the actual message body.
  Error checking is done to make reasonably sure that the data was sent.
    

INTERFACES
            
  Called by:   (IPC.c)   ipc_flush ();

RETURNED VALUE

  Ipc_Status_t - returns status of the send operation (typically
  IPC_STATUS_ERROR or IPC_STATUS_OK).


GLOBAL VARIABLES
    
  NONE

NON-STANDARD FEATURES

  NONE
     
=============================================================================*/


Ipc_Status_t
ipc_transport_send_line (
     char *str,           /* IN - String to write                          */
     int len )            /* IN - Number of characters out of STR to write */
{
  int count;              /* Counts how many bytes were actually written   */
  u_long u;               /* 32-bit placeholder for transmission of LEN    */
  char hdr_buff[5];       /* Buffer for building header message in         */
  int  i;                 /* Temporary counter                             */
  char *char_ptr;         /* Pointer for int to bytes conversion           */

  if (sock_state != IPC_SOCK_CONNECTED_TO_CLIENT) {
    fprintf (stderr, "ERROR: IPC: Attempt to write to non-open socket\n");
    return IPC_STATUS_ERROR;
  }

  /* Write message body header with length: */
  hdr_buff[0] = BOL_CHAR;
  u = htonl ((uint32_t) len);
  char_ptr = (char *) &u;
  for(i = 0; i < 4; i++)
    hdr_buff[i+1] = char_ptr[i];

  count = (int) write (msg_stream, hdr_buff, 5);
  if (count != 5) {
    fprintf (stderr, "ERROR: IPC: (%d) send line error 1\n", count);
    return IPC_STATUS_ERROR;
  }

  /* Write message body: */
  count = (int) write (msg_stream, str, (size_t) len);
  if (count != len) {
    fprintf (stderr, "ERROR: IPC: (%d) send line error 2\n", count);
    return IPC_STATUS_ERROR;
  }

  return IPC_STATUS_OK;
}   /* end ipc_transport_send_line */



/*=============================================================================

FUNCTION ipc_transport_terminate_server

AUTHORS                      

    July 1991   Stefan Roth

MODIFICATIONS   

    NONE

SUMMARY

  This function reads all pending incoming messages and discards them.
  Reading continues until a read error occurs or EOF is reached, at which
  time the socket is closed.
  Note that this function does not actually close the socket. This is
  done in ipc_transport_get_line, which is called in this function.

  In this function, the incoming line length is limited. See buffer below.

INTERFACES
            
    Called by:  (IPC.c)   ipc_terminate_server();

RETURNED VALUE
    
    Ipc_Status_t - returns status of last read operation (always
    IPC_STATUS_ERROR or IPC_STATUS_EOF).

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE
      
=============================================================================*/


Ipc_Status_t
ipc_transport_terminate_server (void)
{
   char buffer[17000];		/* temp buffer for incoming data           */
   int len;			/* placeholder var to as arg to function   */
   Ipc_Status_t status;		/* value to be returned from function      */
   int max_size;		/* Max length of buffer                    */

   max_size = sizeof (buffer);
   do {
     len = max_size;
     status = ipc_transport_get_line (buffer, &len, IPC_WAIT);
   } while ((status != IPC_STATUS_ERROR) &&
	    (status != IPC_STATUS_EOF));
   return status;
}

#endif /* IPC_UNIX_SOCKETS */
