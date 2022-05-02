#ifndef INCLUDED_UDEVICES
#define INCLUDED_UDEVICES

BOOL u_process_instance(char *line);
BOOL u_process_model_line(char *line);
BOOL u_check_instance(char *line);
void create_model_xlator(void);
void cleanup_model_xlator(void);

#endif
