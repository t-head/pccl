/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <limits.h>
#include <memory>
#include <new>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "alloc.h"
#include "comm.h"
#include "analyzer.h"
#include "comm_dumper.h"
#include "diagnostics_checks_common.h"
#include "diagnostics_runtime.h"
#include "transport.h"

namespace {

struct rasDiagnosticsDeviceRestore {
  int device;
  ncclResult_t& result;

  ~rasDiagnosticsDeviceRestore() {
    if (device == -1) return;
    hggcError_t err = hggcSetDevice(device);
    if (err != hggcSuccess) {
      WARN("RAS runtime diagnostics failed to restore device %d: %s", device, hggcGetErrorString(err));
      (void)hggcGetLastError();
      if (result == ncclSuccess) result = ncclUnhandledHggcError;
    }
  }
};

static ncclResult_t rasDiagnosticsRuntimeEmitReport(const struct rasDiagnosticsContext* ctx,
                                                    const struct rasDiagnosticsReporter* reporter,
                                                    const struct rasDiagnosticsRuntimeRecord* records,
                                                    int nRecords) {
  char* text = nullptr;
  size_t textBytes = 0;
  FILE* stream = open_memstream(&text, &textBytes);
  if (stream == nullptr) return ncclSystemError;
  auto closeAndFree = [&text](FILE* fp) {
    fclose(fp);
    free(text);
  };
  std::unique_ptr<FILE, decltype(closeAndFree)> streamOwner(stream, closeAndFree);
  int result = ctx->summaryOnly ? analyzerFormatRuntimeSummary(stream, records, nRecords) :
    analyzerFormatRuntimeReport(stream, records, nRecords, ctx->nLegTimeouts, -1,
                                "PCCL Runtime State Diagnostics");
  // Closing the stream finalizes text; only then transfer ownership of the buffer.
  int closeResult = fclose(streamOwner.release());
  std::unique_ptr<char, decltype(&free)> textOwner(text, free);
  if (result < 0 || closeResult != 0) return ncclSystemError;
  for (char* line = text; line != nullptr && *line != '\0';) {
    char* next = strchr(line, '\n');
    if (next != nullptr) *next++ = '\0';
    if (*line != '\0') NCCLCHECK(reporter->emit(reporter->target, line));
    line = next;
  }
  return ncclSuccess;
}

static ncclResult_t rasDiagnosticsRuntimeColdSnapshot(struct pcclComm* comm,
                                                      struct pcclCommDiagCollector* collector) {
  ncclResult_t ret = ncclSuccess;
  // Restore before returning ret, including on goto and exception paths.
  {
    rasDiagnosticsDeviceRestore restore = {-1, ret};
    HGGCCHECKGOTO(hggcGetDevice(&restore.device), ret, exit);
    HGGCCHECKGOTO(hggcSetDevice(comm->hggcDev), ret, exit);
    ret = rasDiagCollectRuntimeSnapshot(comm, collector);
  }
exit:
  return ret;
}

} // namespace

ncclResult_t rasDiagnosticsRuntimeStateCollectLocal(const struct rasDiagnosticsContext* ctx,
                                                    struct rasDiagnosticsLocalData* data) {
  if (data == nullptr || ctx == nullptr) return ncclInvalidArgument;
  memset(data, 0, sizeof(*data));
  try {
    std::vector<rasDiagnosticsRuntimeRecord> records;
    std::lock_guard<std::recursive_mutex> lock(ncclCommsMutex);
    std::unique_ptr<pcclCommDiagCollector, decltype(&rasDiagCollectorDestroy)> collector(
        rasDiagCollectorCreate(), rasDiagCollectorDestroy);
    if (!collector) return ncclSystemError;
    if (ctx->triggerLocalDump) {
      struct rasCollRequest dump = {};
      dump.type = RAS_BC_DUMP;
      dump.dump.dumpAllComms = !ctx->hasCommFilter;
      dump.dump.commHash = ctx->hasCommFilter ? ctx->commFilter.commHash : 0;
      dump.dump.seq = ctx->diagSeq;
      dump.dump.dumpLevel = ctx->dumpLevel;
      bool done = false;
      rasStateMonitorDump(&dump, &done, collector.get(), ctx);
    }
    for (int i = 0; i < nNcclComms; i++) {
      struct pcclComm* comm = ncclComms[i].comm;
      if (comm == nullptr || !__atomic_load_n(&comm->peerInfoValid, __ATOMIC_ACQUIRE) ||
          !rasDiagnosticsCommMatchesContext(ctx, comm)) continue;
      struct rasDiagnosticsRuntimeRecord rankRecord = {};
      rankRecord.type = RAS_DIAG_RUNTIME_RANK;
      rankRecord.channelId = -1;
      rankRecord.rank.commId = {comm->commHash, comm->peerInfo[0].hostHash, comm->peerInfo[0].pidHash};
      rankRecord.rank.commRank = comm->rank;
      rankRecord.rank.commNRanks = comm->nRanks;
      auto& rank = rankRecord.value.rankData;
      rank.version = RAS_DIAG_RUNTIME_WIRE_VERSION;
      rank.hggcDev = comm->hggcDev;
      rank.hgmlDev = comm->hgmlDev;
      if (comm->abortFlag != nullptr && __atomic_load_n(comm->abortFlag, __ATOMIC_ACQUIRE)) {
        rank.availability = RAS_DIAG_RUNTIME_ABORTED;
      } else {
        int nLocalRecords = 0;
        const auto* localRecords = rasDiagCollectorGetRuntime(collector.get(), comm, &nLocalRecords);
        if (localRecords == nullptr &&
            rasDiagnosticsRuntimeColdSnapshot(comm, collector.get()) == ncclSuccess)
          localRecords = rasDiagCollectorGetRuntime(collector.get(), comm, &nLocalRecords);
        if (localRecords != nullptr && nLocalRecords > 0) {
          if ((size_t)nLocalRecords > (size_t)INT_MAX / sizeof(rasDiagnosticsRuntimeRecord) - records.size())
            return ncclInvalidArgument;
          records.insert(records.end(), localRecords, localRecords + nLocalRecords);
          continue;
        }
        rank.availability = RAS_DIAG_RUNTIME_UNAVAILABLE;
      }
      if (records.size() >= (size_t)INT_MAX / sizeof(rasDiagnosticsRuntimeRecord)) return ncclInvalidArgument;
      records.push_back(rankRecord);
    }
    if (records.empty()) return ncclSuccess;
    size_t bytes = records.size() * sizeof(rasDiagnosticsRuntimeRecord);
    NCCLCHECK(ncclCalloc(&data->records, bytes));
    memcpy(data->records, records.data(), bytes);
    data->recordsBytes = (int)bytes;
    data->recordStride = sizeof(rasDiagnosticsRuntimeRecord);
    data->nRecords = (int)records.size();
    return ncclSuccess;
  } catch (const std::bad_alloc&) {
    WARN("RAS runtime diagnostics allocation failed");
    return ncclSystemError;
  } catch (...) {
    WARN("RAS runtime diagnostics collection failed");
    return ncclInternalError;
  }
}

ncclResult_t rasDiagnosticsRuntimeStateSummarize(const struct rasDiagnosticsContext* ctx,
                                                 const struct rasDiagnosticsReporter* reporter,
                                                 const char* data, int nData) {
  if (ctx == nullptr || reporter == nullptr || reporter->emit == nullptr || nData < 0 ||
      (size_t)nData % sizeof(rasDiagnosticsRuntimeRecord) != 0 || (nData > 0 && data == nullptr))
    return ncclInvalidArgument;
  int nRecords = nData / sizeof(rasDiagnosticsRuntimeRecord);
  const auto* records = reinterpret_cast<const rasDiagnosticsRuntimeRecord*>(data);
  for (int i = 0; i < nRecords; i++) {
    if (ctx->hasCommFilter &&
        (rasDiagnosticsCommIdCompare(&ctx->commFilter, &records[i].rank.commId) != 0 ||
         (ctx->commNRanks > 0 && ctx->commNRanks != records[i].rank.commNRanks))) return ncclInvalidArgument;
  }
  try {
    if (ctx->nLegTimeouts > 0)
      (void)rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                 "Incomplete gather: %d leg timeout(s)", ctx->nLegTimeouts);
    if (nRecords == 0)
      return rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO, "Runtime state: no rank snapshots were received");
    ncclResult_t ret = rasDiagnosticsRuntimeEmitReport(ctx, reporter, records, nRecords);
    if (ret == ncclSuccess && ctx->persistReport) {
      char path[PATH_MAX + 1];
      NCCLCHECK(rasDiagWriteRuntimeReport(records, nRecords, ctx->nLegTimeouts, path, sizeof(path)));
      NCCLCHECK(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                    "Runtime state: diagnostic output saved under %s", path));
    }
    return ret;
  } catch (const std::bad_alloc&) {
    return ncclSystemError;
  } catch (...) {
    return ncclInternalError;
  }
}
