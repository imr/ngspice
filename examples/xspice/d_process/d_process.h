/**
 * \file
 * \brief NgSpice D_PROCESS Interface
 * \author Uros Platise <uros@isotel.eu>
 */

#ifndef __NGSPICE_D_PROCESS__
#define __NGSPICE_D_PROCESS__

#include <stdio.h>
#if !defined(_MSC_VER)
#include <unistd.h>
#endif
#include <stdint.h>

#define D_PROCESS_FORMAT_VERSION    0x01
#define D_PROCESS_DLEN(x)           (uint8_t)( ((x)==0) ? 0 : (((x)-1)/8 + 1) )

static inline int d_process_init(int pipein, int pipeout, uint8_t N_din, uint8_t N_dout) {
#if defined(_MSC_VER) || defined(__MINGW64__)
#pragma pack(push, 1)
    struct header_s {
        uint8_t version, N_din, N_dout;
    } header;
#pragma pack(pop)
#else
    struct header_s {
        uint8_t version, N_din, N_dout;
    } __attribute__((packed))header;
#endif

#if defined(_MSC_VER) || defined(__MINGW64__)
    if (_read(pipein, &header, sizeof(header)) != sizeof(header)) {
#else
    if (read(pipein, &header, sizeof(header)) != sizeof(header)) {
#endif
        fprintf(stderr, "Error: Incompatible ngspice d_process header size, requiring version %d\n", header.version);
        return 0;
    }
    if (header.version != D_PROCESS_FORMAT_VERSION || header.N_din != N_din || header.N_dout != N_dout) {
        fprintf(stderr, "Error: Incompatible ngspice d_process requiring version %d, number of inputs %d expected %d, and outputs %d expected %d.\n",
            header.version, header.N_din, N_din, header.N_dout, N_dout);
        return 0;
    }
#if defined(_MSC_VER) || defined(__MINGW64__)
    _write(pipeout, &header, sizeof(header));    // acknowledge connection by returning back the same header
#else
    write(pipeout, &header, sizeof(header));    // acknowledge connection by returning back the same header
#endif
    return 1;
}

#endif
