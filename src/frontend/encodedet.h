
typedef enum {
    ENCODING_ASCII,
    ENCODING_UTF8_NO_BOM,
    ENCODING_UTF8_WITH_BOM,
    ENCODING_UTF16LE_WITH_BOM,
    ENCODING_UTF16LE_NO_BOM,
    ENCODING_UTF16BE_WITH_BOM,
    ENCODING_BINARY,
    ENCODING_UNKNOWN
} FileEncoding;

FileEncoding detect_file_encoding(FILE* file);
const char* encoding_to_string(FileEncoding encoding);

