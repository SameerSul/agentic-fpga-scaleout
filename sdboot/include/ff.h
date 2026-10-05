/* sdboot: FatFs as the program calls it, served from DDR. U-Boot has
   already read each file off the card into the address the program loads
   it to, and written its size into a table (sdboot/rt.c); a "read" there
   copies nothing, and the program's own size checks still apply. */
#ifndef FF_H
#define FF_H
#include "xil_io.h"
typedef unsigned int UINT;
typedef struct { int unused; } FATFS;
typedef struct { const u8 *src; u32 size, pos; } FIL;
typedef enum { FR_OK = 0, FR_DISK_ERR = 1, FR_NO_FILE = 4 } FRESULT;
#define FA_READ 0x01
#define f_size(fp) ((fp)->size)
FRESULT f_mount(FATFS *fs, const char *path, unsigned char opt);
FRESULT f_open(FIL *fp, const char *path, unsigned char mode);
FRESULT f_read(FIL *fp, void *buf, UINT btr, UINT *br);
FRESULT f_close(FIL *fp);
#endif
