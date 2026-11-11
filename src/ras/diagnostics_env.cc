/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "alloc.h"
#include "checks.h"
#include "diagnostics_checks.h"
#include "diagnostics_checks_common.h"

// *************************************************************************
// PCCL_*/NCCL_* environment strings, interpreted with PCCL's prefix priority.
// This is not a snapshot of cached parameters or effective communicator config.
// *************************************************************************
#define RAS_DIAG_ENV_BYTES 16384
#define RAS_DIAG_ENV_KEY_DISPLAY_BYTES 256
#define RAS_DIAG_ENV_VALUE_DISPLAY_BYTES 256

extern "C" char** environ;

struct rasDiagnosticsNcclEnvData {
  uint16_t bytesUsed;
  uint8_t truncated;
  char data[RAS_DIAG_ENV_BYTES];
};

struct rasDiagnosticsEnvKey {
  const char* key;
  uint16_t len;
  bool normalized;
};

static bool rasDiagnosticsEnvHasPrefix(const char* entry) {
  return strncmp(entry, "PCCL_", 5) == 0 || strncmp(entry, "NCCL_", 5) == 0;
}

static int rasDiagnosticsEnvEntryCompare(const void* p1, const void* p2) {
  const char* e1 = *(char* const*)p1;
  const char* e2 = *(char* const*)p2;
  return strcmp(e1, e2);
}

static int rasDiagnosticsEnvKeyCompare(const void* p1, const void* p2) {
  const struct rasDiagnosticsEnvKey* k1 = (const struct rasDiagnosticsEnvKey*)p1;
  const struct rasDiagnosticsEnvKey* k2 = (const struct rasDiagnosticsEnvKey*)p2;
  // For ordinary names, compare only suffixes so the two aliases share one key.
  // The 64-byte boundary matches pcclGetEnvWithDualPrefix's name buffers.
  if (k1->normalized != k2->normalized) return k1->normalized ? -1 : 1;
  size_t skip = k1->normalized ? 5 : 0;
  size_t minLen = k1->len < k2->len ? k1->len : k2->len;
  int cmp = strncmp(k1->key + skip, k2->key + skip, minLen - skip);

  if (cmp != 0) return cmp;
  return (k1->len < k2->len ? -1 : (k1->len > k2->len ? 1 : 0));
}

static bool rasDiagnosticsEnvNextEntry(const struct rasDiagnosticsNcclEnvData* envData, int* offset, const char** entry,
                                       size_t* entryLen) {
  const char* start;
  const void* end;

  if (envData == nullptr || offset == nullptr || entry == nullptr || entryLen == nullptr) return false;
  if (envData->bytesUsed > RAS_DIAG_ENV_BYTES || *offset < 0 || *offset >= envData->bytesUsed) return false;

  start = envData->data + *offset;
  end = memchr(start, '\0', (size_t)envData->bytesUsed - (size_t)*offset);
  if (end == nullptr) return false;

  *entry = start;
  *entryLen = (const char*)end - start;
  *offset += (int)*entryLen + 1;
  return true;
}

static const char* rasDiagnosticsEnvFindValue(const struct rasDiagnosticsNcclEnvData* envData, const char* key,
                                              size_t keyLen, bool* conflict = nullptr) {
  int offset = 0;
  const char* entry;
  size_t entryLen;
  const char* value = nullptr;

  if (conflict != nullptr) *conflict = false;

  while (rasDiagnosticsEnvNextEntry(envData, &offset, &entry, &entryLen)) {
    const char* eq = (const char*)memchr(entry, '=', entryLen);
    if (eq == nullptr) continue;
    if ((size_t)(eq - entry) != keyLen || strncmp(entry, key, keyLen) != 0) continue;
    if (value != nullptr && strcmp(value, eq + 1) != 0 && conflict != nullptr) *conflict = true;
    value = eq + 1;
  }
  return value;
}

static const char* rasDiagnosticsEnvSelectedValue(const struct rasDiagnosticsNcclEnvData* envData,
                                                 const struct rasDiagnosticsEnvKey* key, const char** source) {
  if (!key->normalized) {
    *source = "raw";
    return rasDiagnosticsEnvFindValue(envData, key->key, key->len);
  }
  char name[64];
  memcpy(name, key->key, key->len);
  name[key->len] = '\0';
  name[0] = 'P';
  const char* value = rasDiagnosticsEnvFindValue(envData, name, key->len);
  if (value != nullptr) {
    *source = "PCCL";
    return value; // Empty strings also shadow the NCCL alias.
  }
  name[0] = 'N';
  value = rasDiagnosticsEnvFindValue(envData, name, key->len);
  *source = value != nullptr ? "NCCL" : "unset";
  return value;
}

static bool rasDiagnosticsEnvValuesEqual(const char* v1, const char* v2) {
  // nullptr means the environment variable is not set on that rank.
  if (v1 == nullptr || v2 == nullptr) return v1 == v2;
  return strcmp(v1, v2) == 0;
}

static const char* rasDiagnosticsEnvKeyDisplay(const struct rasDiagnosticsEnvKey* key, char* buf, size_t bufLen) {
  size_t limit;
  size_t i;

  if (bufLen == 0) return "";
  limit = bufLen > 4 ? bufLen - 4 : bufLen - 1;
  for (i = 0; i < limit && i < key->len; i++) {
    unsigned char c = (unsigned char)key->key[i];
    buf[i] = (c < 32 || c == 127) ? '?' : key->key[i];
  }
  if (i > 0 && key->normalized) buf[0] = 'P';
  if (i < key->len && bufLen > 4) {
    memcpy(buf + i, "...", 4);
  } else {
    buf[i] = '\0';
  }
  return buf;
}

static const char* rasDiagnosticsEnvValueDisplay(const char* value, char* buf, size_t bufLen) {
  if (bufLen == 0) return "";
  if (value == nullptr) {
    snprintf(buf, bufLen, "(unset)");
    return buf;
  }
  // Keep one-line output and distinguish an empty string from an absent value.
  if (bufLen < 7) {
    buf[0] = '\0';
    return buf;
  }
  size_t pos = 0;
  size_t i = 0;
  buf[pos++] = '"';
  while (value[i] != '\0') {
    unsigned char c = (unsigned char)value[i];
    size_t needed = (c == '"' || c == '\\') ? 2 : 1;
    if (pos + needed + 5 > bufLen) break; // Reserve ellipsis, quote and NUL.
    if (needed == 2) buf[pos++] = '\\';
    buf[pos++] = (c < 32 || c == 127) ? '?' : (char)c;
    i++;
  }
  if (value[i] != '\0') {
    memcpy(buf + pos, "...", 3);
    pos += 3;
  }
  buf[pos++] = '"';
  buf[pos] = '\0';
  return buf;
}

static ncclResult_t rasDiagnosticsNcclEnvReportMismatch(const struct rasDiagnosticsReporter* reporter,
                                                        const struct rasDiagnosticsRankHeader* startRank,
                                                        const char* keyDisplay, bool normalized) {
  return rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                              "PCCL environment: %s differs across %d ranks in comm 0x%lx for %s",
                              normalized ? "prefix-priority view" : "raw environment",
                              startRank->commNRanks, startRank->commId.commHash, keyDisplay);
}

static ncclResult_t rasDiagnosticsNcclEnvReportValue(const struct rasDiagnosticsReporter* reporter,
                                                     const char* keyDisplay, const char* valueDisplay,
                                                     const char* rankSet, const char* source) {
  return rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                              "PCCL environment: %s=%s on rank(s) %s, source=%s", keyDisplay,
                              valueDisplay, rankSet, source);
}

static ncclResult_t rasDiagnosticsNcclEnvReportTruncated(const struct rasDiagnosticsReporter* reporter,
                                                         int nTruncated) {
  return rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                              "PCCL environment: %d rank(s) had >%d bytes of PCCL_*/NCCL_* env vars; "
                              "truncated snapshot, comparison skipped for this communicator",
                              nTruncated, RAS_DIAG_ENV_BYTES);
}

static ncclResult_t rasDiagnosticsNcclEnvReportConsistent(const struct rasDiagnosticsReporter* reporter, int commNRanks,
                                                          const struct rasCommId* commId, bool hasRawDifferences) {
  if (hasRawDifferences) {
    return rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                "PCCL environment: prefix-priority environment strings consistent across %d ranks "
                                "in comm 0x%lx; raw-name differences found (see above)", commNRanks,
                                commId->commHash);
  }
  return rasDiagnosticsReport(reporter, RAS_DIAG_TAG_OK,
                              "PCCL environment: prefix-priority environment strings consistent across %d ranks "
                              "in comm 0x%lx", commNRanks,
                              commId->commHash);
}

static ncclResult_t rasDiagnosticsNcclEnvReportMismatchSummary(const struct rasDiagnosticsReporter* reporter,
                                                               int nMismatch, const struct rasCommId* commId) {
  return rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                              "PCCL environment: %d environment key(s) differ across ranks in comm 0x%lx", nMismatch,
                              commId->commHash);
}

static const struct rasDiagnosticsNcclEnvData* rasDiagnosticsNcclEnvDataFromRecord(const char* record) {
  return (const struct rasDiagnosticsNcclEnvData*)(record + sizeof(struct rasDiagnosticsRankHeader));
}

static bool rasDiagnosticsNcclEnvDataValid(const struct rasDiagnosticsNcclEnvData* envData) {
  if (envData->bytesUsed > RAS_DIAG_ENV_BYTES) return false;
  if (envData->bytesUsed != 0 && envData->data[envData->bytesUsed - 1] != '\0') return false;
  int offset = 0;
  const char* entry;
  size_t entryLen;
  while (rasDiagnosticsEnvNextEntry(envData, &offset, &entry, &entryLen)) {
    const char* eq = (const char*)memchr(entry, '=', entryLen);
    if (eq == nullptr || eq - entry < 5 || !rasDiagnosticsEnvHasPrefix(entry)) return false;
  }
  return offset == envData->bytesUsed;
}

static bool rasDiagnosticsNcclEnvHasConflicts(const struct rasDiagnosticsNcclEnvData* envData) {
  int offset = 0;
  const char* entry;
  size_t entryLen;
  while (rasDiagnosticsEnvNextEntry(envData, &offset, &entry, &entryLen)) {
    const char* eq = (const char*)memchr(entry, '=', entryLen);
    bool conflict;
    (void)rasDiagnosticsEnvFindValue(envData, entry, (size_t)(eq - entry), &conflict);
    if (conflict) return true;
  }
  return false;
}

static int rasDiagnosticsEnvCountCandidateKeys(const char* records, size_t recordStride, int start, int end) {
  int nKeys = 0;

  for (int i = start; i < end; i++) {
    const struct rasDiagnosticsNcclEnvData* envData =
      rasDiagnosticsNcclEnvDataFromRecord(records + (size_t)i * recordStride);
    int offset = 0;
    const char* entry;
    size_t entryLen;

    while (rasDiagnosticsEnvNextEntry(envData, &offset, &entry, &entryLen)) {
      if (memchr(entry, '=', entryLen) != nullptr) nKeys++;
    }
  }
  return nKeys;
}

static void rasDiagnosticsEnvFillCandidateKeys(const char* records, size_t recordStride, int start, int end,
                                               struct rasDiagnosticsEnvKey* keys) {
  int nKeys = 0;

  for (int i = start; i < end; i++) {
    const struct rasDiagnosticsNcclEnvData* envData =
      rasDiagnosticsNcclEnvDataFromRecord(records + (size_t)i * recordStride);
    int offset = 0;
    const char* entry;
    size_t entryLen;

    while (rasDiagnosticsEnvNextEntry(envData, &offset, &entry, &entryLen)) {
      const char* eq = (const char*)memchr(entry, '=', entryLen);
      if (eq == nullptr) continue;
      keys[nKeys].key = entry;
      keys[nKeys].len = (uint16_t)(eq - entry);
      keys[nKeys].normalized = keys[nKeys].len > 5 && keys[nKeys].len < 64;
      nKeys++;
    }
  }
}

static ncclResult_t rasDiagnosticsNcclEnvFillLocalData(const struct rasDiagnosticsCommSnapshot* comm, void* checkData) {
  struct rasDiagnosticsNcclEnvData* envData = (struct rasDiagnosticsNcclEnvData*)checkData;
  char** entries = nullptr;
  int nEntries = 0;

  (void)comm;

  envData->bytesUsed = 0;
  envData->truncated = 0;

  // Like the upstream collector, this requires no concurrent mutation of environ.
  // The completed record owns its strings; no environment pointers survive this call.
  for (char** ep = environ; ep != nullptr && *ep != nullptr; ep++) {
    if (rasDiagnosticsEnvHasPrefix(*ep)) nEntries++;
  }
  if (nEntries == 0) return ncclSuccess;

  NCCLCHECK(ncclCalloc(&entries, nEntries));
  char** entry = entries;
  for (char** ep = environ; ep != nullptr && *ep != nullptr; ep++) {
    if (rasDiagnosticsEnvHasPrefix(*ep)) *(entry++) = *ep;
  }
  qsort(entries, nEntries, sizeof(*entries), rasDiagnosticsEnvEntryCompare);

  for (int i = 0; i < nEntries; i++) {
    size_t len = strlen(entries[i]) + 1;
    if (len > (size_t)RAS_DIAG_ENV_BYTES - envData->bytesUsed) {
      envData->truncated = 1;
      break;
    }
    memcpy(envData->data + envData->bytesUsed, entries[i], len);
    envData->bytesUsed += (uint16_t)len;
  }

  free(entries);
  return ncclSuccess;
}

ncclResult_t rasDiagnosticsNcclEnvCollectLocal(const struct rasDiagnosticsContext* ctx,
                                               struct rasDiagnosticsLocalData* data) {
  NCCLCHECK(rasDiagnosticsCollectLocalRecords(ctx, sizeof(struct rasDiagnosticsNcclEnvData),
                                              rasDiagnosticsNcclEnvFillLocalData, data));
  return ncclSuccess;
}

static ncclResult_t rasDiagnosticsNcclEnvReportKey(const struct rasDiagnosticsReporter* reporter, const char* records,
                                                   size_t recordStride, int start, int end,
                                                   const struct rasDiagnosticsRankHeader* startRank,
                                                   const struct rasDiagnosticsEnvKey* key, int* nMismatch) {
  ncclResult_t ret = ncclSuccess;
  bool anyDiff = false;
  int nGroupRanks = end - start;
  const char** values = nullptr;
  const char** sources = nullptr;
  int* covered = nullptr;
  int* groupRanks = nullptr;
  char keyDisplay[RAS_DIAG_ENV_KEY_DISPLAY_BYTES];

  NCCLCHECKGOTO(ncclCalloc(&values, nGroupRanks), ret, exit);
  NCCLCHECKGOTO(ncclCalloc(&sources, nGroupRanks), ret, exit);
  for (int i = 0; i < nGroupRanks; i++) {
    values[i] = rasDiagnosticsEnvSelectedValue(
      rasDiagnosticsNcclEnvDataFromRecord(records + (size_t)(start + i) * recordStride), key, sources + i);
    if (i > 0 && !rasDiagnosticsEnvValuesEqual(values[0], values[i])) {
      anyDiff = true;
    }
  }
  if (!anyDiff) goto exit;

  (*nMismatch)++;
  NCCLCHECKGOTO(ncclCalloc(&covered, nGroupRanks), ret, exit);
  NCCLCHECKGOTO(ncclCalloc(&groupRanks, nGroupRanks), ret, exit);
  rasDiagnosticsEnvKeyDisplay(key, keyDisplay, sizeof(keyDisplay));

  NCCLCHECKGOTO(rasDiagnosticsNcclEnvReportMismatch(reporter, startRank, keyDisplay, key->normalized), ret, exit);

  for (int i = 0; i < nGroupRanks; i++) {
    const char* value;
    int nGroup = 0;
    const char* source = sources[i];
    char rankSet[128];
    char valueDisplay[RAS_DIAG_ENV_VALUE_DISPLAY_BYTES];

    if (covered[i]) continue;
    value = values[i];
    for (int j = i; j < nGroupRanks; j++) {
      const struct rasDiagnosticsRankHeader* rank =
        rasDiagnosticsRankHeaderFromRecord(records + (size_t)(start + j) * recordStride);
      const char* otherValue = values[j];

      if (covered[j] || !rasDiagnosticsEnvValuesEqual(value, otherValue)) continue;
      covered[j] = 1;
      groupRanks[nGroup++] = rank->commRank;
      if (strcmp(source, sources[j]) != 0) source = "PCCL/NCCL";
    }

    rasDiagnosticsFormatRankSet(rankSet, sizeof(rankSet), groupRanks, nGroup, nGroup);
    rasDiagnosticsEnvValueDisplay(value, valueDisplay, sizeof(valueDisplay));
    NCCLCHECKGOTO(rasDiagnosticsNcclEnvReportValue(reporter, keyDisplay, valueDisplay, rankSet, source), ret, exit);
  }

exit:
  free(values);
  free(sources);
  free(covered);
  free(groupRanks);
  return ret;
}

static ncclResult_t rasDiagnosticsNcclEnvReportAliases(
  const struct rasDiagnosticsReporter* reporter, const char* records, size_t recordStride, int start, int end,
  const struct rasDiagnosticsRankHeader* startRank, const struct rasDiagnosticsEnvKey* key, bool* hasRawDifferences) {
  char name[64];
  char pcclDisplay[RAS_DIAG_ENV_VALUE_DISPLAY_BYTES];
  char ncclDisplay[RAS_DIAG_ENV_VALUE_DISPLAY_BYTES];
  char keyDisplay[RAS_DIAG_ENV_KEY_DISPLAY_BYTES];
  int nRawMismatch = 0;

  if (!key->normalized) return ncclSuccess;
  memcpy(name, key->key, key->len);
  name[key->len] = '\0';
  rasDiagnosticsEnvKeyDisplay(key, keyDisplay, sizeof(keyDisplay));
  for (int i = start; i < end; i++) {
    const char* record = records + (size_t)i * recordStride;
    const struct rasDiagnosticsRankHeader* rank = rasDiagnosticsRankHeaderFromRecord(record);
    const struct rasDiagnosticsNcclEnvData* envData = rasDiagnosticsNcclEnvDataFromRecord(record);
    name[0] = 'P';
    const char* pccl = rasDiagnosticsEnvFindValue(envData, name, key->len);
    name[0] = 'N';
    const char* nccl = rasDiagnosticsEnvFindValue(envData, name, key->len);
    if (pccl == nullptr || nccl == nullptr || strcmp(pccl, nccl) == 0) continue;
    rasDiagnosticsEnvValueDisplay(pccl, pcclDisplay, sizeof(pcclDisplay));
    rasDiagnosticsEnvValueDisplay(nccl, ncclDisplay, sizeof(ncclDisplay));
    NCCLCHECK(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                  "PCCL environment: rank %d in comm 0x%lx: %s=%s shadows NCCL_%s=%s",
                                  rank->commRank, rank->commId.commHash, keyDisplay, pcclDisplay,
                                  keyDisplay + 5, ncclDisplay));
  }
  // Keep raw-name differences visible even when the selected values are equal.
  // A plugin may read an NCCL_* variable directly instead of using PCCL's helper.
  struct rasDiagnosticsEnvKey rawKey = {name, key->len, false};
  name[0] = 'P';
  NCCLCHECK(rasDiagnosticsNcclEnvReportKey(reporter, records, recordStride, start, end, startRank,
                                         &rawKey, &nRawMismatch));
  name[0] = 'N';
  NCCLCHECK(rasDiagnosticsNcclEnvReportKey(reporter, records, recordStride, start, end, startRank,
                                         &rawKey, &nRawMismatch));
  if (nRawMismatch > 0) *hasRawDifferences = true;
  return ncclSuccess;
}

ncclResult_t rasDiagnosticsNcclEnvSummarize(
  const struct rasDiagnosticsContext* ctx, const struct rasDiagnosticsReporter* reporter, const char* data, int nData) {
  ncclResult_t ret = ncclSuccess;
  char* records = nullptr;
  struct rasDiagnosticsEnvKey* keys = nullptr;
  const size_t recordStride = rasDiagnosticsLocalRecordStride(sizeof(struct rasDiagnosticsNcclEnvData));
  int nRecords;

  (void)ctx;

  if (reporter == nullptr || reporter->emit == nullptr) {
    WARN("RAS diagnostics PCCL environment check received invalid reporter");
    return ncclInternalError;
  }
  if (nData == 0) {
    return rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                "PCCL environment: no rank data; comparison skipped");
  }
  if (data == nullptr) {
    WARN("RAS diagnostics PCCL environment check received null data with size %d", nData);
    return ncclInternalError;
  }
  if (nData < 0 || nData % (int)recordStride != 0) {
    WARN("RAS diagnostics PCCL environment check received malformed data size %d", nData);
    return ncclInternalError;
  }

  nRecords = nData / (int)recordStride;
  NCCLCHECK(ncclCalloc(&records, nData));
  memcpy(records, data, nData);
  qsort(records, nRecords, recordStride, rasDiagnosticsRankHeaderCompare);
  NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                    "PCCL environment: values use PCCL core prefix priority; not cached or "
                                    "effective config. Unknown variables and plugins may use other rules."), ret, exit);

  for (int start = 0; start < nRecords;) {
    const char* startRecord = records + (size_t)start * recordStride;
    const struct rasDiagnosticsRankHeader* startRank = rasDiagnosticsRankHeaderFromRecord(startRecord);
    int commNRanks = startRank->commNRanks;
    int end = start;
    int nTruncated = 0;
    int nMismatch = 0;
    int nCandidateKeys;
    bool ranksValid = commNRanks > 0;
    bool envConflicts = false;
    bool hasRawDifferences = false;

    while (end < nRecords) {
      const char* record = records + (size_t)end * recordStride;
      const struct rasDiagnosticsRankHeader* rank = rasDiagnosticsRankHeaderFromRecord(record);
      const struct rasDiagnosticsNcclEnvData* envData;

      if (end > start && rasDiagnosticsCommIdCompare(&startRank->commId, &rank->commId) != 0) break;
      if (rank->commNRanks != commNRanks || rank->commRank < 0 || rank->commRank >= commNRanks) ranksValid = false;
      if (end > start) {
        const struct rasDiagnosticsRankHeader* previous =
          rasDiagnosticsRankHeaderFromRecord(records + (size_t)(end - 1) * recordStride);
        if (previous->commRank == rank->commRank) ranksValid = false;
      }
      envData = rasDiagnosticsNcclEnvDataFromRecord(record);
      if (!rasDiagnosticsNcclEnvDataValid(envData)) {
        WARN("RAS diagnostics PCCL environment check received malformed env payload");
        ret = ncclInternalError;
        goto exit;
      }
      if (envData->truncated) nTruncated++;
      if (rasDiagnosticsNcclEnvHasConflicts(envData)) envConflicts = true;
      end++;
    }

    if (!ranksValid) {
      NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                        "PCCL environment: invalid or duplicate rank data in comm 0x%lx; "
                                        "comparison skipped", startRank->commId.commHash), ret, exit);
      start = end;
      continue;
    }
    if (end - start != commNRanks) {
      NCCLCHECKGOTO(rasDiagnosticsReportIncomplete(reporter, "PCCL environment", startRank, end - start), ret, exit);
      start = end;
      continue;
    }
    // Never interpret an omitted high-priority alias as absent or fall back to NCCL.
    if (nTruncated > 0) {
      NCCLCHECKGOTO(rasDiagnosticsNcclEnvReportTruncated(reporter, nTruncated), ret, exit);
      start = end;
      continue;
    }
    if (envConflicts) {
      NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                        "PCCL environment: conflicting duplicate environment names in comm 0x%lx; "
                                        "comparison skipped", startRank->commId.commHash), ret, exit);
      start = end;
      continue;
    }

    nCandidateKeys = rasDiagnosticsEnvCountCandidateKeys(records, recordStride, start, end);
    if (nCandidateKeys > 0) {
      NCCLCHECKGOTO(ncclCalloc(&keys, nCandidateKeys), ret, exit);
      rasDiagnosticsEnvFillCandidateKeys(records, recordStride, start, end, keys);
      qsort(keys, nCandidateKeys, sizeof(*keys), rasDiagnosticsEnvKeyCompare);

      for (int keyIdx = 0; keyIdx < nCandidateKeys; keyIdx++) {
        const struct rasDiagnosticsEnvKey* key = keys + keyIdx;

        if (keyIdx > 0 && rasDiagnosticsEnvKeyCompare(key - 1, key) == 0) continue;
        ret = rasDiagnosticsNcclEnvReportKey(reporter, records, recordStride, start, end, startRank, key, &nMismatch);
        if (ret != ncclSuccess) goto exit;
        ret = rasDiagnosticsNcclEnvReportAliases(reporter, records, recordStride, start, end, startRank, key,
                                                &hasRawDifferences);
        if (ret != ncclSuccess) goto exit;
      }
    }

    if (nMismatch == 0) {
      NCCLCHECKGOTO(rasDiagnosticsNcclEnvReportConsistent(reporter, commNRanks, &startRank->commId,
                                                         hasRawDifferences), ret, exit);
    } else if (nMismatch > 0) {
      NCCLCHECKGOTO(rasDiagnosticsNcclEnvReportMismatchSummary(reporter, nMismatch, &startRank->commId), ret, exit);
    }

    free(keys);
    keys = nullptr;
    start = end;
  }

exit:
  free(keys);
  free(records);
  return ret;
}
