#ifndef _SPICE_SNDFILE_H
#define _SPICE_SNDFILE_H

void snd_configure(char*, int, int, double, double, int);
void snd_init(int);
void snd_close(void);
int snd_send(double, int, double);
int snd_format(char*);
double snd_get_samplerate(void);

#endif
