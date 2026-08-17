/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Licensed under the Apache License v2.0 with LLVM Exceptions.
 * See https://nvidia.github.io/NVTX/LICENSE.txt for license information.
 */

/*
 * Modified by T-Head Semiconductor.
 * Changes: Replaced all nvtx* symbols with hgtx* (function names, types,
 * macros, headers) while preserving full API compatibility.
 * Original source: https://github.com/NVIDIA/NVTX
 */

#ifndef HGTX_EXT_INIT_GUARD
#error Never include this file directly -- it is automatically included by hgToolsExt.h (except when HGTX_NO_IMPL is defined).
#endif

#if defined(HGTX_AS_SYSTEM_HEADER)
#if defined(__clang__)
#pragma clang system_header
#elif defined(__GNUC__) || defined(__HGCOMPILER)
#pragma GCC system_header
#elif defined(_MSC_VER)
#pragma system_header
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* ---- Platform-independent helper definitions and functions ---- */

/* Prefer macros over inline functions to reduce symbol resolution at link time */

#if defined(_WIN32)
#define HGTX_ATOMIC_WRITE_PTR(address, value) \
    InterlockedExchangePointer(HGTX_REINTERPRET_CAST(volatile PVOID*, (address)), \
        HGTX_REINTERPRET_CAST(PVOID, (value)))
#define HGTX_ATOMIC_CAS_PTR(old, address, exchange, comparand) \
    (old) = HGTX_REINTERPRET_CAST(intptr_t, InterlockedCompareExchangePointer( \
        HGTX_REINTERPRET_CAST(volatile PVOID*, (address)), \
        HGTX_REINTERPRET_CAST(PVOID, (exchange)), \
        HGTX_REINTERPRET_CAST(PVOID, (comparand))))
#elif defined(__GNUC__)
/* Ensure full memory barrier for atomics, to match Windows functions */
#define HGTX_ATOMIC_WRITE_PTR(address, value) \
    __sync_synchronize(); *address = value; __sync_synchronize()
#define HGTX_ATOMIC_CAS_PTR(old, address, exchange, comparand) \
    old = __sync_val_compare_and_swap(address, comparand, exchange)
#else
#error The library does not support your configuration!
#endif

#ifndef HGTX_SUPPORT_ALREADY_INJECTED_LIBRARY
/* Define this to 1 for platforms that where pre-injected libraries can be discovered. */
#if defined(_WIN32)
/* Windows has no process-wide table of dynamic library symbols, so this can't be supported. */
#define HGTX_SUPPORT_ALREADY_INJECTED_LIBRARY 0
#else
/* POSIX platforms allow calling dlsym on a null module to use the process-wide table.
 * Note: Still disabled in load sequence version 2.  Needs to support following the
 * RTLD_NEXT chain, and needs more testing before support can be enabled by default. */
#define HGTX_SUPPORT_ALREADY_INJECTED_LIBRARY 0
#endif
#endif

#ifndef HGTX_SUPPORT_ENV_VARS
/* Define this to 1 for platforms that support environment variables. */
/* TODO: Detect UWP, a.k.a. Windows Store app, and set this to 0. */
/* Try:  #if defined(WINAPI_FAMILY_PARTITION) && WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP) */
#define HGTX_SUPPORT_ENV_VARS 1
#endif

#ifndef HGTX_SUPPORT_DYNAMIC_INJECTION_LIBRARY
/* Define this to 1 for platforms that support dynamic/shared libraries */
#define HGTX_SUPPORT_DYNAMIC_INJECTION_LIBRARY 1
#endif

#ifndef HGTX_SUPPORT_ANDROID_INJECTION_LIBRARY_IN_PACKAGE
#if defined(__ANDROID__)
#define HGTX_SUPPORT_ANDROID_INJECTION_LIBRARY_IN_PACKAGE 1
#else
#define HGTX_SUPPORT_ANDROID_INJECTION_LIBRARY_IN_PACKAGE 0
#endif
#endif

#ifndef HGTX_SUPPORT_STATIC_INJECTION_LIBRARY
/* On platforms that support weak symbols (i.e. non-Windows), injection libraries may
*  be statically linked into an application.  This is useful for platforms where dynamic
*  injection is not available.  Weak symbols not marked extern are definitions, not just
*  declarations.  They are guaranteed to be initialized to zero if no normal definitions
*  are found by the linker to override them.  This means the HGTX load sequence can safely
*  detect the presence of a static injection -- if InitializeInjectionHgtxExtension_fnptr is zero,
*  there is no static injection. */
#if defined(__GNUC__) && !defined(_WIN32) && !defined(__CYGWIN__)
#define HGTX_SUPPORT_STATIC_INJECTION_LIBRARY 1
#else
#define HGTX_SUPPORT_STATIC_INJECTION_LIBRARY 0
#endif
#endif

#if HGTX_SUPPORT_STATIC_INJECTION_LIBRARY && !defined(HGTX_STATIC_INJECTION_IMPL)
/* To make an HGTX injection library support static injection, it must do these things:
*  - Define InitializeInjectionHgtxExtension_fnptr as a normal symbol (not weak), pointing to
*    the implementation of InitializeInjectionHgtxExtension (which does not need to be a
*    dynamic export if only supporting static injection).
*  - Define HGTX_STATIC_INJECTION_IMPL so the weak definition below is skipped.
*  - Compile the static injection files with -fPIC if they are to be linked with other
*    files compiled this way.  If you forget this, GCC will simply tell you to add it.
*  When building the application, there a few ways to link in a static injection:
*  - Compile the injection's source files normally, and include the .o files as inputs
*    to the linker.
*  - If the injection is provided as an archive (.a file), it will not resolve any
*    unresolved symbols, so the linker will skip it by default.  This can be fixed
*    by wrapping the static injection's name on the linker command line with options
*    to treat it differently.  For example:
*      gcc example.o libfoo.a -Wl,--whole-archive libinj-static.a -Wl,--no-whole-archive libbar.a
*    Note that libinj-static.a is bracketed by options to turn on "whole archive" and
*    then back off again afterwards, so libfoo.a and libbar.a are linked normally.
*  - In CMake, a static injection can be added with options like this:
*      target_link_libraries(app PRIVATE -Wl,--whole-archive inj-static -Wl,--no-whole-archive)
*/
__attribute__((weak)) HgtxExtInitializeInjectionFunc_t InitializeInjectionHgtxExtension_fnptr;
#endif

/* This function tries to find or load an HGTX injection library and get the
*  address of its InitializeInjectionExtension function.  If such a function pointer
*  is found, it is called, and passed the address of this HGTX instance's
*  hgtxGetExportTable function, so the injection can attach to this instance.
*  If the initialization fails for any reason, any dynamic library loaded will
*  be freed, and all HGTX implementation functions will be set to no-ops.  If
*  initialization succeeds, HGTX functions not attached to the tool will be set
*  to no-ops.  This is implemented as one function instead of several small
*  functions to minimize the number of weak symbols the linker must resolve.
*  Order of search is:
*  - Pre-injected library exporting InitializeInjectionHgtxExtension
*  - Loadable library exporting InitializeInjectionHgtxExtension
*      - Path specified by env var HGTX_INJECTION??_PATH (?? is 32 or 64)
*      - On Android, libHgtxInjection??.so within the package (?? is 32 or 64)
*  - Statically-linked injection library defining InitializeInjectionHgtxExtension_fnptr
*/
HGTX_LINKONCE_FWDDECL_FUNCTION int HGTX_VERSIONED_IDENTIFIER(hgtxExtLoadInjectionLibrary)(
    HgtxExtInitializeInjectionFunc_t* out_init_fnptr);
HGTX_LINKONCE_DEFINE_FUNCTION int HGTX_VERSIONED_IDENTIFIER(hgtxExtLoadInjectionLibrary)(
    HgtxExtInitializeInjectionFunc_t* out_init_fnptr)
{
    static const char initFuncName[] = "InitializeInjectionHgtxExtension";
#if HGTX_SUPPORT_ALREADY_INJECTED_LIBRARY
    static const char initFuncPreinjectName[] = "InitializeInjectionHgtxExtensionPreinject";
#endif
    HgtxExtInitializeInjectionFunc_t init_fnptr = HGTX_NULLPTR;
    HGTX_DLLHANDLE injectionLibraryHandle = HGTX_DLLDEFAULT;

    if (out_init_fnptr)
    {
        *out_init_fnptr = HGTX_NULLPTR;
    }

#if HGTX_SUPPORT_DYNAMIC_INJECTION_LIBRARY
    /* Try discovering dynamic injection library to load */
    {
#if HGTX_SUPPORT_ENV_VARS
        /* If env var HGTX_INJECTION64_PATH is set, it should contain the path
           to a 64-bit dynamic HGTX injection library (and similar for 32-bit). */
        const HGTX_PATHCHAR* const hgtxEnvVarName = (sizeof(void*) == 4)
            ? HGTX_STR("HGTX_INJECTION32_PATH")
            : HGTX_STR("HGTX_INJECTION64_PATH");
#endif /* HGTX_SUPPORT_ENV_VARS */
        HGTX_PATHCHAR injectionLibraryPathBuf[HGTX_BUFSIZE];
        const HGTX_PATHCHAR* injectionLibraryPath = HGTX_NULLPTR;

        /* Refer to this variable explicitly in case all references to it are #if'ed out. */
        (void)injectionLibraryPathBuf;

#if HGTX_SUPPORT_ENV_VARS
        /* Disable the warning for getenv & _wgetenv -- this usage is safe because
           these functions are not called again before using the returned value. */
#if defined(_MSC_VER)
#pragma warning( push )
#pragma warning( disable : 4996 )
#endif
        injectionLibraryPath = HGTX_GETENV(hgtxEnvVarName);
#if defined(_MSC_VER)
#pragma warning( pop )
#endif
#endif

#if HGTX_SUPPORT_ANDROID_INJECTION_LIBRARY_IN_PACKAGE
        if (!injectionLibraryPath)
        {
            const char *bits = (sizeof(void*) == 4) ? "32" : "64";
            char cmdlineBuf[32];
            char pkgName[PATH_MAX];
            int count;
            int pid;
            FILE *fp;
            size_t bytesRead;
            size_t pos;

            pid = HGTX_STATIC_CAST(int, getpid());
            count = snprintf(cmdlineBuf, sizeof(cmdlineBuf), "/proc/%d/cmdline", pid);
            if (count <= 0 || count >= HGTX_STATIC_CAST(int, sizeof(cmdlineBuf)))
            {
                HGTX_ERR("Path buffer too small for: /proc/%d/cmdline\n", pid);
                return HGTX_ERR_INIT_ACCESS_LIBRARY;
            }

            fp = fopen(cmdlineBuf, "r");
            if (!fp)
            {
                HGTX_ERR("File couldn't be opened: %s\n", cmdlineBuf);
                return HGTX_ERR_INIT_ACCESS_LIBRARY;
            }

            bytesRead = fread(pkgName, 1, sizeof(pkgName) - 1, fp);
            fclose(fp);
            if (bytesRead == 0)
            {
                HGTX_ERR("Package name couldn't be read from file: %s\n", cmdlineBuf);
                return HGTX_ERR_INIT_ACCESS_LIBRARY;
            }

            pkgName[bytesRead] = 0;

            /* String can contain colon as a process separator. In this case the
               package name is before the colon. */
            pos = 0;
            while (pos < bytesRead && pkgName[pos] != ':' && pkgName[pos] != '\0')
            {
                ++pos;
            }
            pkgName[pos] = 0;

            count = snprintf(injectionLibraryPathBuf, HGTX_BUFSIZE, "/data/data/%s/files/libHgtxInjection%s.so", pkgName, bits);
            if (count <= 0 || count >= HGTX_BUFSIZE)
            {
                HGTX_ERR("Path buffer too small for: /data/data/%s/files/libHgtxInjection%s.so\n", pkgName, bits);
                return HGTX_ERR_INIT_ACCESS_LIBRARY;
            }

            /* On Android, verify path is accessible due to aggressive file access restrictions. */
            /* For dlopen, if the filename contains a leading slash, then it is interpreted as a */
            /* relative or absolute pathname; otherwise it will follow the rules in ld.so. */
            if (injectionLibraryPathBuf[0] == '/')
            {
#if (__ANDROID_API__ < 21)
                int access_err = access(injectionLibraryPathBuf, F_OK | R_OK);
#else
                int access_err = faccessat(AT_FDCWD, injectionLibraryPathBuf, F_OK | R_OK, 0);
#endif
                if (access_err != 0)
                {
                    HGTX_ERR("Injection library path wasn't accessible [code=%s] [path=%s]\n", strerror(errno), injectionLibraryPathBuf);
                    return HGTX_ERR_INIT_ACCESS_LIBRARY;
                }
            }
            injectionLibraryPath = injectionLibraryPathBuf;
        }
#endif /* HGTX_SUPPORT_ANDROID_INJECTION_LIBRARY_IN_PACKAGE */

        /* At this point, `injectionLibraryPath` is specified if a dynamic
           injection library was specified by a tool. */
        if (injectionLibraryPath)
        {
            /* Load the injection library */
            injectionLibraryHandle = HGTX_DLLOPEN(injectionLibraryPath);
            if (!injectionLibraryHandle)
            {
                HGTX_ERR("Failed to load injection library\n");
                return HGTX_ERR_INIT_LOAD_LIBRARY;
            }
            else
            {
                /* Attempt to get the injection library's entry-point. */
                init_fnptr = HGTX_REINTERPRET_CAST(HgtxExtInitializeInjectionFunc_t, HGTX_DLLFUNC(injectionLibraryHandle, initFuncName));
                if (!init_fnptr)
                {
                    HGTX_DLLCLOSE(injectionLibraryHandle);
                    HGTX_ERR("Failed to get address of function %s from injection library\n", initFuncName);
                    return HGTX_ERR_INIT_MISSING_LIBRARY_ENTRY_POINT;
                }
            }
        }
    }
#endif /* HGTX_SUPPORT_DYNAMIC_INJECTION_LIBRARY */

#if HGTX_SUPPORT_ALREADY_INJECTED_LIBRARY
    if (!init_fnptr)
    {
        /* Use POSIX global symbol chain to query for init function from any module */
        init_fnptr = HGTX_REINTERPRET_CAST(HgtxExtInitializeInjectionFunc_t, HGTX_DLLFUNC(HGTX_DLLDEFAULT, initFuncPreinjectName));
    }
#endif

#if HGTX_SUPPORT_STATIC_INJECTION_LIBRARY
    if (!init_fnptr)
    {
        /* Check weakly-defined function pointer.  A statically-linked injection can define this
        *  as a normal symbol and set it to the address of the HGTX init function -- this will
        *  provide a non-null value here.  If there is no other definition of this symbol, it
        *  will be null here. */
        if (InitializeInjectionHgtxExtension_fnptr)
        {
            init_fnptr = InitializeInjectionHgtxExtension_fnptr;
        }
    }
#endif

    if (out_init_fnptr)
    {
        *out_init_fnptr = init_fnptr;
    }

    /* At this point, if `init_fnptr` is not set, no tool has specified an HGTX injection library.
       Non-success result is returned, so that all HGTX API functions will be set to no-ops. */
    if (!init_fnptr)
    {
        return HGTX_ERR_NO_INJECTION_LIBRARY_AVAILABLE;
    }

    return HGTX_SUCCESS;
}

/* Avoid warnings about missing prototypes. */
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_VERSIONED_IDENTIFIER(hgtxExtInitOnce) (
    hgtxExtModuleInfo_t* moduleInfo, intptr_t* moduleState);
HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_VERSIONED_IDENTIFIER(hgtxExtInitOnce) (
    hgtxExtModuleInfo_t* moduleInfo, intptr_t* moduleState)
{
    intptr_t old;

    HGTX_INFO( "%s\n", __FUNCTION__ );

    if (*moduleState == HGTX_EXTENSION_LOADED)
    {
        HGTX_INFO("Module loaded\n");
        return;
    }

    HGTX_ATOMIC_CAS_PTR(
        old,
        moduleState,
        HGTX_EXTENSION_STARTING,
        HGTX_EXTENSION_FRESH);
    if (old == HGTX_EXTENSION_FRESH)
    {
        HgtxExtInitializeInjectionFunc_t init_fnptr =
            HGTX_VERSIONED_IDENTIFIER(hgtxExtGlobals1).injectionFnPtr;
        int entryPointStatus = 0;
        int forceAllToNoops = 0;
        size_t s;

        /* Load and initialize injection library, which will assign the function pointers. */
        if (init_fnptr == HGTX_NULLPTR)
        {
            int result = 0;

            /* Try to load vanilla HGTX first. */
            hgtxInitialize(HGTX_NULLPTR);

            result = HGTX_VERSIONED_IDENTIFIER(hgtxExtLoadInjectionLibrary)(&init_fnptr);
            /* At this point `init_fnptr` will be either 0 or a real function. */

            if (result == HGTX_SUCCESS)
            {
                HGTX_VERSIONED_IDENTIFIER(hgtxExtGlobals1).injectionFnPtr = init_fnptr;
            }
            else
            {
                HGTX_ERR("Failed to load injection library\n");
            }
        }

        if (init_fnptr != HGTX_NULLPTR)
        {
            /* Invoke injection library's initialization function. If it returns
               0 (failure) and a dynamic injection was loaded, unload it. */
            entryPointStatus = init_fnptr(moduleInfo);
            if (entryPointStatus == 0)
            {
                HGTX_ERR("Failed to initialize injection library -- initialization function returned 0\n");
            }
        }

        /* Clean up any functions that are still uninitialized so that they are
           skipped. Set all to null if injection init function failed as well. */
        forceAllToNoops = (init_fnptr == HGTX_NULLPTR) || (entryPointStatus == 0);
        for (s = 0; s < moduleInfo->segmentsCount; ++s)
        {
            hgtxExtModuleSegment_t* segment = moduleInfo->segments + s;
            size_t i;
            for (i = 0; i < segment->slotCount; ++i)
            {
                if (forceAllToNoops || (segment->functionSlots[i] == HGTX_EXTENSION_FRESH))
                {
                    segment->functionSlots[i] = HGTX_EXTENSION_DISABLED;
                }
            }
        }

        HGTX_MEMBAR();

        /* Signal that initialization has finished and the assigned function
           pointers will be used. */
        HGTX_ATOMIC_WRITE_PTR(moduleState, HGTX_EXTENSION_LOADED);
    }
    else /* Spin-wait until initialization has finished. */
    {
        HGTX_MEMBAR();
        while (*moduleState != HGTX_EXTENSION_LOADED)
        {
            HGTX_YIELD();
            HGTX_MEMBAR();
        }
    }
}

#ifdef __cplusplus
}
#endif /* __cplusplus */
