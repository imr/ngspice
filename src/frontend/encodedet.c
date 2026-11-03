/* Copyright Holger Vogt, 2026
   License 3-clause BSD
   with help from Mistral Small 4 119B */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "encodedet.h"

FileEncoding detect_file_encoding(FILE *file) {
    unsigned char buffer[4];
    size_t bytes_read = fread(buffer, 1, 4, file);
    rewind(file); // Reset file pointer

    // Check for BOMs first
    if (bytes_read >= 3 &&
        buffer[0] == 0xEF && buffer[1] == 0xBB && buffer[2] == 0xBF) {
        return ENCODING_UTF8_WITH_BOM;
    }

    if (bytes_read >= 2) {
        if (buffer[0] == 0xFF && buffer[1] == 0xFE) {
            return ENCODING_UTF16LE_WITH_BOM;
        }
        if (buffer[0] == 0xFE && buffer[1] == 0xFF) {
            // UTF-16BE with BOM (not in your requested list)
            return ENCODING_UTF16BE_WITH_BOM;
        }
    }

    /* Check for UTF-16LE without BOM (look for null bytes)
    if (bytes_read >= 2) {
        int likely_utf16le = 1;
        for (size_t i = 0; i < bytes_read; i++) {
            if (i + 1 >= bytes_read) {
                likely_utf16le = 0;
                break;
            }
            if (buffer[i] != 0x00 && (buffer[i] < 0x20 || buffer[i] > 0x7E)) {
                likely_utf16le = 0;
                break;
            }
        }
        if (likely_utf16le && bytes_read % 2 == 0) {
            return ENCODING_UTF16LE_NO_BOM;
        }
    } */

    // Check for UTF-16LE without BOM (look for null bytes)
    if (bytes_read >= 4) {
        if (buffer[1] == '\0' && buffer[3] == '\0') {
            return ENCODING_UTF16LE_NO_BOM;
        }
    }
    else if (bytes_read >= 2) {
        if (buffer[1] == '\0') {
            return ENCODING_UTF16LE_NO_BOM;
        }
    }

    // Check for UTF-8 without BOM
    int is_utf8 = 1;
    for (size_t i = 0; i < bytes_read; i++) {
        if (buffer[i] == 0x00) { // Null byte found (likely binary)
            is_utf8 = 0;
            break;
        }
        if ((buffer[i] & 0x80) == 0x00) { // 0xxxxxxx - ASCII
            continue;
        } else if ((buffer[i] & 0xE0) == 0xC0) { // 110xxxxx
            if (i + 1 >= bytes_read || (buffer[i+1] & 0xC0) != 0x80) {
                is_utf8 = 0;
                break;
            }
            i++; // Skip next byte
        } else if ((buffer[i] & 0xF0) == 0xE0) { // 1110xxxx
            if (i + 2 >= bytes_read || (buffer[i+1] & 0xC0) != 0x80 ||
                (buffer[i+2] & 0xC0) != 0x80) {
                is_utf8 = 0;
                break;
            }
            i += 2; // Skip next 2 bytes
        } else if ((buffer[i] & 0xF8) == 0xF0) { // 11110xxx
            if (i + 3 >= bytes_read || (buffer[i+1] & 0xC0) != 0x80 ||
                (buffer[i+2] & 0xC0) != 0x80 || (buffer[i+3] & 0xC0) != 0x80) {
                is_utf8 = 0;
                break;
            }
            i += 3; // Skip next 3 bytes
        } else {
            is_utf8 = 0;
            break;
        }
    }

    if (is_utf8) {
        return ENCODING_UTF8_NO_BOM;
    }

    // Check for ASCII
    int is_ascii = 1;
    for (size_t i = 0; i < bytes_read; i++) {
        if (buffer[i] > 0x7F) {
            is_ascii = 0;
            break;
        }
    }

    if (is_ascii) {
        return ENCODING_ASCII;
    }

    // Check for binary (contains null bytes or non-text characters)
    for (size_t i = 0; i < bytes_read; i++) {
        if (buffer[i] == 0x00 || buffer[i] < 0x09 || buffer[i] == 0x0B ||
            buffer[i] == 0x0C || (buffer[i] > 0x0E && buffer[i] < 0x20) ||
            buffer[i] > 0x7E) {
            return ENCODING_BINARY;
        }
    }

    return ENCODING_UNKNOWN;
}

const char* encoding_to_string(FileEncoding encoding) {
    switch (encoding) {
        case ENCODING_ASCII: return "ASCII";
        case ENCODING_UTF8_NO_BOM: return "UTF-8 without BOM";
        case ENCODING_UTF8_WITH_BOM: return "UTF-8 with BOM";
        case ENCODING_UTF16LE_WITH_BOM: return "UTF-16LE with BOM";
        case ENCODING_UTF16LE_NO_BOM: return "UTF-16LE without BOM";
        case ENCODING_BINARY: return "Binary";
        default: return "Unknown";
    }
}
/*
int main() {
    FILE *file = fopen("test.txt", "rb");
    if (!file) {
        perror("Failed to open file");
        return 1;
    }

    FileEncoding encoding = detect_file_encoding(file);
    printf("File encoding: %s\n", encoding_to_string(encoding));

    fclose(file);
    return 0;
}
*/
