#define WIN32_LEAN_AND_MEAN
#include <windows.h>

int victory_seh_call(void (*fn)(void *), void *ctx) {
    __try {
        fn(ctx);
        return 1;
    } __except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH) {
        return 0;
    }
}
