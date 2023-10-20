/*
   The organization of this file is modelled after the motorforce example
   developed by Uros Platise.
*/
#include <string.h>
#include <stdlib.h>
#include <fcntl.h>
#if !defined(_MSC_VER) && !defined(__MINGW64__)
#include <unistd.h>
#endif
#if defined(_MSC_VER)
#include <io.h>
#include <process.h>
#include <Windows.h>
#endif
#include "d_process.h"

#define DIGITAL_IN      1
#define DIGITAL_OUT     4

static int compute(
    uint8_t *datain, int insz, uint8_t *dataout, int outsz, double time
);

//#define ENABLE_DEBUGGING 1
#ifdef ENABLE_DEBUGGING
#include "debugging.h"
#endif

int main(int argc, char *argv[]) {    
    int inlen = D_PROCESS_DLEN(DIGITAL_IN);
    int outlen = D_PROCESS_DLEN(DIGITAL_OUT);

#if defined(_MSC_VER) || defined(__MINGW64__)
#pragma pack(push, 1)
    struct in_s {
        double time;
        uint8_t din[D_PROCESS_DLEN(DIGITAL_IN)];
    } in;
            
    struct out_s {
        uint8_t dout[D_PROCESS_DLEN(DIGITAL_OUT)];
    } out;
#pragma pack(pop)
#else
    struct in_s {
        double time;
        uint8_t din[D_PROCESS_DLEN(DIGITAL_IN)];
    } __attribute__((packed)) in;
            
    struct out_s {
        uint8_t dout[D_PROCESS_DLEN(DIGITAL_OUT)];
    } __attribute__((packed)) out;
#endif

    int pipein = 0; // default stdin to recv from ngspice
    int pipeout= 1; // default stdout to send to ngspice
#if defined(_MSC_VER) || defined(__MINGW64__)
    _setmode(0, _O_BINARY);
    _setmode(1, _O_BINARY);
#endif

#ifdef ENABLE_DEBUGGING
    debug_info(argc, argv);
#endif

    if (d_process_init(pipein, pipeout, DIGITAL_IN, DIGITAL_OUT) ) {
#if defined(_MSC_VER) || defined(__MINGW64__)
        while(_read(pipein, &in, sizeof(in)) == sizeof(in)) {
#else
        while(read(pipein, &in, sizeof(in)) == sizeof(in)) {
#endif

            if (!compute(in.din, inlen, out.dout, outlen, in.time)) {
                return 1;
            }

#if defined(_MSC_VER) || defined(__MINGW64__)
            _write(pipeout, &out, sizeof(out));
#else
            write(pipeout, &out, sizeof(out));
#endif
        }
        return 0;
    }
    return -1;
}

static int compute(
    uint8_t *datain, int insz, uint8_t *dataout, int outsz, double time
)
{
    static uint8_t next[2][16] = {
        {15, 14,  0,  1, 13, 12,  2,  3, 11, 10,  4,  5,  9,  8,  6,  7},
        { 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0}
    };
    uint8_t inbit0 = datain[0] & 1;
    static uint8_t count = 0;
    if (time < 0.0) {
        fprintf(stderr, "Reset prog1in4out at time %g\n", -time);
        count = 15;
    }
    if (count < 15) {
        count++;
    } else {
        count = 0;
    }
    dataout[0] = (next[inbit0][count]) & 0x0F;
    return 1;
}

