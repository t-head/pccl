/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/
// Internal nccl* spellings for the pccl* device API surface. The public contract
// in pccl.h / pccl_device/ is pccl*; the implementation keeps the upstream nccl*
// names so it stays diffable against NCCL. Token macros, so no include ordering
// dependency on the public headers.

#ifndef PCCL_COMPAT_H_
#define PCCL_COMPAT_H_

#define NCCL_COMM_PROPERTIES_INITIALIZER         PCCL_COMM_PROPERTIES_INITIALIZER
#define NCCL_CONFIG_UNDEF_INT                    PCCL_CONFIG_UNDEF_INT
#define NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER   PCCL_DEV_COMM_REQUIREMENTS_INITIALIZER
#define NCCL_GIN_BACKEND_MASK_ALL                PCCL_GIN_BACKEND_MASK_ALL
#define NCCL_GIN_CONNECTION_FULL                 PCCL_GIN_CONNECTION_FULL
#define NCCL_GIN_CONNECTION_NONE                 PCCL_GIN_CONNECTION_NONE
#define NCCL_GIN_CONNECTION_RAIL                 PCCL_GIN_CONNECTION_RAIL
#define NCCL_GIN_MAX_CONNECTIONS                 PCCL_GIN_MAX_CONNECTIONS
#define NCCL_GIN_TYPE_GDAKI                      PCCL_GIN_TYPE_GDAKI
#define NCCL_GIN_TYPE_NONE                       PCCL_GIN_TYPE_NONE
#define NCCL_GIN_TYPE_PROXY                      PCCL_GIN_TYPE_PROXY
#define NCCL_NET_DEVICE_GIN_GDAKI                PCCL_NET_DEVICE_GIN_GDAKI
#define NCCL_NET_DEVICE_GIN_PROXY                PCCL_NET_DEVICE_GIN_PROXY
#define NCCL_NET_DEVICE_HOST                     PCCL_NET_DEVICE_HOST
#define NCCL_WIN_STRICT_ORDERING                 PCCL_WIN_STRICT_ORDERING
#define ncclCommProperties_t                     pcclCommProperties_t
#define ncclCoopCta                              pcclCoopCta
#define ncclDevComm                              pcclDevComm
#define ncclDevCommRequirements                  pcclDevCommRequirements
#define ncclDevCommRequirements_t                pcclDevCommRequirements_t
#define ncclDevCommWindowTable                   pcclDevCommWindowTable
#define ncclDevResourceHandle                    pcclDevResourceHandle
#define ncclDevResourceRequirements              pcclDevResourceRequirements
#define ncclGinConnectionType_t                  pcclGinConnectionType_t
#define ncclGinCounter_t                         pcclGinCounter_t
#define ncclGinGdakiMlxGPUContext                pcclGinGdakiMlxGPUContext
#define ncclGinGdakiMlxMemHandle                 pcclGinGdakiMlxMemHandle
#define ncclGinInboxA2AHandle                    pcclGinInboxA2AHandle
#define ncclGinOutboxHandle                      pcclGinOutboxHandle
#define ncclGinProxyGfdCompletion                pcclGinProxyGfdCompletion
#define ncclGinProxyGfdDstHandle                 pcclGinProxyGfdDstHandle
#define ncclGinProxyGfdDstOff                    pcclGinProxyGfdDstOff
#define ncclGinProxyGfdHeader                    pcclGinProxyGfdHeader
#define ncclGinProxyGfdHeaderExt                 pcclGinProxyGfdHeaderExt
#define ncclGinProxyGfdInlineHigh                pcclGinProxyGfdInlineHigh
#define ncclGinProxyGfdInlineLow                 pcclGinProxyGfdInlineLow
#define ncclGinProxyGfdQwords                    pcclGinProxyGfdQwords
#define ncclGinProxyGfdSignalVal                 pcclGinProxyGfdSignalVal
#define ncclGinProxyGfdSrcHandle                 pcclGinProxyGfdSrcHandle
#define ncclGinProxyGfdSrcOff                    pcclGinProxyGfdSrcOff
#define ncclGinProxyGfdVASignalHandle            pcclGinProxyGfdVASignalHandle
#define ncclGinProxyGfdVASignalOff               pcclGinProxyGfdVASignalOff
#define ncclGinProxyGfd_t                        pcclGinProxyGfd_t
#define ncclGinProxyGpuCtx_t                     pcclGinProxyGpuCtx_t
#define ncclGinProxyOpBaseMask                   pcclGinProxyOpBaseMask
#define ncclGinProxyOpFlush                      pcclGinProxyOpFlush
#define ncclGinProxyOpGet                        pcclGinProxyOpGet
#define ncclGinProxyOpPut                        pcclGinProxyOpPut
#define ncclGinProxyOpVASignal                   pcclGinProxyOpVASignal
#define ncclGinProxyOpWithCounter                pcclGinProxyOpWithCounter
#define ncclGinProxyOpWithInline                 pcclGinProxyOpWithInline
#define ncclGinProxyOpWithSignalAdd              pcclGinProxyOpWithSignalAdd
#define ncclGinProxyOpWithSignalInc              pcclGinProxyOpWithSignalInc
#define ncclGinProxyOp_t                         pcclGinProxyOp_t
#define ncclGinProxyQword_t                      pcclGinProxyQword_t
#define ncclGinSignal_t                          pcclGinSignal_t
#define ncclGinSyncHandle                        pcclGinSyncHandle
#define ncclGinType_t                            pcclGinType_t
#define ncclGinWindow_t                          pcclGinWindow_t
#define ncclGin_SignalAdd                        pcclGin_SignalAdd
#define ncclGin_SignalInc                        pcclGin_SignalInc
#define ncclLLA2AHandle                          pcclLLA2AHandle
#define ncclMultimemHandle_t                     pcclMultimemHandle_t
#define ncclNetDeviceHandle_t                    pcclNetDeviceHandle_t
#define ncclNetDeviceHandle_v11_t                pcclNetDeviceHandle_v11_t
#define ncclNetDeviceType                        pcclNetDeviceType
#define ncclSegmentWindow                        pcclSegmentWindow
#define ncclTeam                                 pcclTeam
#define ncclTeamRequirements                     pcclTeamRequirements
#define ncclTeam_t                               pcclTeam_t
#define ncclWindow_vidmem                        pcclWindow_vidmem

// Device API host entry points: NCCL_API + definition stay pccl* (the exported
// contract); internal call sites keep the nccl* spelling, same as the
// ncclCommRegister family in pccl_internal.h.
#define ncclCommQueryProperties             pcclCommQueryProperties
#define ncclCoopThread                      pcclCoopThread
#define ncclDevCommCreate                   pcclDevCommCreate
#define ncclDevCommDestroy                  pcclDevCommDestroy
#define ncclDevComm_t                       pcclDevComm_t
#define ncclGetLsaDevicePointer             pcclGetLsaDevicePointer
#define ncclGetLsaMultimemDevicePointer     pcclGetLsaMultimemDevicePointer
#define ncclGetMultimemDevicePointer        pcclGetMultimemDevicePointer
#define ncclGetPeerDevicePointer            pcclGetPeerDevicePointer
#define ncclGetResourceBuffer               pcclGetResourceBuffer
#define ncclGetResourceBufferLocalPointer   pcclGetResourceBufferLocalPointer
#define ncclGinBarrierCreateRequirement     pcclGinBarrierCreateRequirement
#define ncclGinInboxA2ACreateRequirement    pcclGinInboxA2ACreateRequirement
#define ncclGinOutboxCreateRequirement      pcclGinOutboxCreateRequirement
#define ncclGin_BackendMask                 pcclGin_BackendMask
#define ncclLLA2ACalcSlots                  pcclLLA2ACalcSlots
#define ncclLLA2ACreateRequirement          pcclLLA2ACreateRequirement
#define ncclLsaBarrierCreateRequirement     pcclLsaBarrierCreateRequirement
#define ncclSymPtr                          pcclSymPtr
#define ncclTeamInnerFactor                 pcclTeamInnerFactor
#define ncclTeamLsa                         pcclTeamLsa
#define ncclTeamRail                        pcclTeamRail
#define ncclTeamRankIsMember                pcclTeamRankIsMember
#define ncclTeamRankToLsa                   pcclTeamRankToLsa
#define ncclTeamRankToTeam                  pcclTeamRankToTeam
#define ncclTeamRankToWorld                 pcclTeamRankToWorld
#define ncclTeamWorld                       pcclTeamWorld
#define ncclWinGetUserPtr                   pcclWinGetUserPtr

#endif // PCCL_COMPAT_H_
