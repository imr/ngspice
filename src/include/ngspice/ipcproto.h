#ifndef ngspice_IPCPROTO_H
#define ngspice_IPCPROTO_H

/* IPC.c */
Ipc_Boolean_t kw_match (char *keyword , char *str );
Ipc_Status_t ipc_initialize_server (char *server_name , Ipc_Mode_t m , Ipc_Protocol_t p );
Ipc_Status_t ipc_terminate_server (void );
Ipc_Status_t ipc_get_line (char *str , int *len , Ipc_Wait_t wait );
Ipc_Status_t ipc_flush (void );
Ipc_Status_t ipc_send_line_binary (char *str , int len );
Ipc_Status_t ipc_send_line (char *str );
Ipc_Status_t ipc_send_data_prefix (double time );
Ipc_Status_t ipc_send_dcop_prefix (void );
Ipc_Status_t ipc_send_data_suffix (void );
Ipc_Status_t ipc_send_dcop_suffix (void );
Ipc_Status_t ipc_send_errchk (void );
Ipc_Status_t ipc_send_end (void );
int stuff_binary_v1 (double d1 , double d2 , int n , char *buf , int pos );
Ipc_Status_t ipc_send_double (char *tag , double value );
Ipc_Status_t ipc_send_complex (char *tag , Ipc_Complex_t value );
Ipc_Status_t ipc_send_int (char *tag , int value );
Ipc_Status_t ipc_send_boolean (char *tag , Ipc_Boolean_t value );
Ipc_Status_t ipc_send_string (char *tag , char *value );
Ipc_Status_t ipc_send_int_array (char *tag , int array_len , int *value );
Ipc_Status_t ipc_send_double_array (char *tag , int array_len , double *value );
Ipc_Status_t ipc_send_complex_array (char *tag , int array_len , Ipc_Complex_t *value );
Ipc_Status_t ipc_send_boolean_array (char *tag , int array_len , Ipc_Boolean_t *value );
Ipc_Status_t ipc_send_string_array (char *tag , int array_len , char **value );
Ipc_Status_t ipc_send_evtdict_prefix (void );
Ipc_Status_t ipc_send_evtdict_suffix (void );
Ipc_Status_t ipc_send_evtdata_prefix (void );
Ipc_Status_t ipc_send_evtdata_suffix (void );
Ipc_Status_t ipc_send_event(int, double, double, char *, void *, int);

/* IPCtiein.c */
void ipc_handle_stop (void );
void ipc_handle_returni (void );
void ipc_handle_mintime (double time );
void ipc_handle_vtrans (char *vsrc , char *dev );
void ipc_send_stdout (void );
void ipc_send_stderr (void );
Ipc_Status_t ipc_send_std_files (void );
Ipc_Boolean_t ipc_screen_name (char *name , char *mapped_name );
int ipc_get_devices (CKTcircuit *circuit , char *device , char ***names , double **modtypes );
void ipc_free_devices (int num_items , char **names , double *modtypes );
void ipc_check_pause_stop (void );

/* IPCaegis.c */
Ipc_Status_t ipc_transport_initialize_server (char *server_name , Ipc_Mode_t m , Ipc_Protocol_t p , char *batch_filename );
Ipc_Status_t extract_msg (char *str , int *len );
Ipc_Status_t ipc_transport_get_line (char *str , int *len , Ipc_Wait_t wait );
Ipc_Status_t ipc_transport_terminate_server (void );
Ipc_Status_t ipc_transport_send_line (char *str , int len );

#endif
