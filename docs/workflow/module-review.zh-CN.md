# PCCL 模块评审流程

本文是 [Module review workflow](module-review.md) 的中文版。

这套流程根据 PR 涉及的 PCCL 模块，自动邀请相应负责人评审，并在现有 CI 之外增加一项必需的模块审批检查。
**仅把文件放进仓库不会自动启用强制约束，管理员还需要完成下文的配置。**
受保护的目标分支是 GitHub 的 `main`，本方案不覆盖 `ppu1.7`。

本文中的 Maintainer 指模块主负责人，Backup 指该模块的后备负责人，Reviewer 指对应子模块或专项的评审人。

## 一、从提交到合入的流程

**贡献者不需要安装 Node.js，也不必运行本地分类命令。** 在 PR 中声明涉及的模块即可；工作流启用后，GitHub Actions 会自动核对路径覆盖并邀请评审人。

1. 按现有 `pccl-contribution` skill 和贡献规范准备修改，包括关联 Issue、保持一个贡献提交、添加 DCO 签署，以及完成相关测试。
2. 查看整个 PR 的改动，根据下方模块表和 [owners.json](../../.github/review/owners.json)，选择涉及的主模块与专项。
   可以让 `pccl-contribution` skill 协助分类，也可以直接填写，不要求使用 skill 或运行本地工具。
3. 在 PR 描述中填写 `PCCL-Modules:`，并按需填写 `PCCL-Areas:`，例如：

   ```text
   PCCL-Modules: device, network
   PCCL-Areas: device-gin, network-rdma
   ```

   这个例子表示改动涉及 GIN 的 RDMA 后端，需要 Device API 与网络两个主模块参与评审。
   若补充了其他受影响模块，应在 PR 的 Changes & Impact 部分解释原因。
4. 工作流将路径匹配结果、PR 描述中的模块声明，以及 `module/<id>`、`area/<id>` 标签取并集，确定评审范围。
   随后邀请模块 Maintainer 和匹配的专项 Reviewer，排除 PR 作者，并避免重复邀请或重复发送尚未处理的邀请。
5. **受影响模块的 Maintainer 或 Backup 中，任意一人批准当前 PR 的最新提交，即满足模块审批要求。**
   跨多个模块时也只需一人批准，此人不必同时负责所有受影响模块。
   评审邀请仍覆盖所有受影响模块及匹配的专项。
6. 完成现有必需 CI 和评审要求后，由有合入权限的团队成员核对当前模块范围、审批、最新提交和 CI 状态，再执行合入。

删除模块标签或修改 PR 描述，不能从评审范围中移除代码路径已经匹配出的模块。
路径规则也无法识别所有语义依赖，因此作者、skill 和模块负责人还需要补充跨模块影响。
模块分类本身也是评审的一部分，文件路径不能完整代表架构职责。

<details>
<summary>可选：在本地预览分类结果</summary>

如果本机已经有 Node.js 20 或更高版本，可以用以下工具预览路径分类：

```bash
node .github/review/classify.mjs <fetched-target-ref> [<pr-body-file>]
```

`<fetched-target-ref>` 是已经从远端获取并核对过的目标分支引用；`<pr-body-file>` 是可选的 PR 描述文件。
工具读取目标分支与当前分支共同祖先到 `HEAD` 之间已经提交的改动，只输出分类建议。
它不会修改 PR、邀请评审人，也不能识别所有语义依赖。不运行这个预览工具，不影响服务端的自动分类和审批要求。

</details>

## 二、主负责人、Backup 与专项 Reviewer

[owners.json](../../.github/review/owners.json) 是人员名单和评审分配规则的统一来源。
下表使用 GitHub 账号。

| 模块 ID | 范围 | Maintainer（主负责人） | Backup（后备负责人） |
| --- | --- | --- | --- |
| `runtime` | 运行时与任务调度 | `ctt284373` | `ck-milo` |
| `topology` | 拓扑发现与路径搜索 | `xiangzhixyc` | `ck-milo` |
| `tuning` | 算法与协议调优 | `ck-milo` | `honganlxh` |
| `network` | 网络通信 | `xiangzhixyc` | `snowdrift-zw` |
| `kernels` | 集合通信与 P2P 内核 | `ck-milo` | `xiangzhixyc` |
| `device` | Device API 与 GIN | `wensuyang` | `snowdrift-zw` |
| `profiler` | Profiler 与 Inspector | `levilee-lwl` | `whale494255` |
| `ras` | RAS 与诊断 | `ck-milo` | `QingxunLy` |
| `workflow` | 贡献流程：提交、评审与合入 | `honganlxh` | `wensuyang` |
| `build` | 构建、CI，以及未分类路径的初步分流 | `levilee-lwl` | `ck-milo` |

23 个专项及其 Reviewer 名单，保存在各模块配置的 `areas` 中。
Reviewer 在对应专项内提供评审意见；只有同时担任某个受影响主模块的 Maintainer 或 Backup，其批准才会计入模块审批。

**第一版不强制要求“负责人批准之外，再额外取得一位专项 Reviewer 的批准”。**
不过，具有仓库写权限的评审人如果提出了尚未撤回或解除的“要求修改”（Changes requested），模块审批检查仍然不会通过。

`zian.cza` 暂时保留为 EIC 的 Reviewer 候选。由于 GitHub 账号尚未确认，工作流暂不自动邀请，待补齐账号映射后再启用。
其他已配置账号正常参与自动邀请。

### Backup 如何接替

Backup 对本模块具有与 Maintainer 相同的审批效力，是否能批准不取决于最初邀请了谁。
默认邀请 Maintainer；如果 Maintainer 本人就是 PR 作者，则改为邀请 Backup。

Maintainer 不在岗时，协作者可以添加 `backup/<module-id>` 标签，例如 `backup/device` 或 `backup/network`，请求 Backup 评审。
工作流不会删除已经存在的人工评审邀请。这个标签只调整邀请对象，不会改变谁有资格批准。
如果要更换 Backup，需要通过经过评审的 PR 修改受保护的人员配置。
Maintainer 和 Backup 都仍须具备仓库写权限，其批准才会计入模块审批。

标签不是必需的：不预先创建模块和专项标签，也可以直接使用 PR 描述中的声明。
管理员或具有标签管理权限的协作者，可以按需创建标签，名称必须使用配置中的准确 ID。

### 模块匹配规则及边界

路径规则按配置顺序匹配，每个路径使用第一条命中的规则。文件重命名时，同时检查旧路径和新路径。例如：

- `src/graph/tuning.cc` 归入 `tuning`；其他拓扑发现、Graph 搜索文件归入 `topology`。
- GIN 的 RDMA 后端同时涉及 `device` 和 `network`；Device API 接口归入 `device`。
- 扩展内核对应 `kernels-4`；RedOp、Symmetric、CE / OKI / P2P DMA 分别有更具体的专项评审范围。
- 如果 EIC / SNIC 的修改发生在公共网络文件里，需要根据实际影响补充 `network-eic` 或 `network-snic` 专项声明。
- `src/misc/drv_extension.cc` 的派发与资源调优归入 `tuning`；如果同时改变运行时或内核行为，还要补充 `runtime` 或 `kernels`。
  归类依据是实际功能，不能仅因文件放在 `misc` 目录就确定其职责。
- 公共 Host 代码由 `runtime` 提供基础覆盖。没有命中路径规则的文件，会额外交给 `build` 负责人初步分流，并列在检查结果中。
  批准前应核对这些文件，确认是否还有其他受影响模块。

贡献规范、PR／Issue 模板、Agent 评审规范和 Skill、人员表、模块审批工作流及其脚本、测试与文档，
归入 `workflow` / `workflow-review`，由湘涵担任 Maintainer、苏洋担任 Backup。
构建脚本、打包及硬件 CI（包括 `pr-test-ppu.yml` 和配套脚本）归入 `build`，由未来担任 Maintainer、陈凯担任 Backup。
贡献流程的专用路径规则优先于 `.github/` 的通用 CI 规则。
检查程序使用当前受信任的 `main` 分支配置，PR 中提出的新配置不会用于批准该 PR 自身。
配置修改合入后，主分支的推送事件会按新配置重新检查仍然打开的 PR。
涉及人员账号变更时，应在该 PR 中邀请受影响模块的负责人评审，并在启用前核对仓库访问权限。

## 三、管理员如何启用

本版完全使用 GitHub Actions 及其内置的 **`GITHUB_TOKEN`** 邀请评审人，并发布 **PCCL Module Owner Approval** 检查。
无须另外创建 GitHub App、配置私钥、个人令牌（PAT）或专用环境。

1. 按现有批准流程合入本实现。先保护好 `main`，避免未经评审的直接推送，并核对仓库和组织的 Actions 策略允许检查任务声明的权限：
   **Contents: read**、**Actions: read**、**Pull requests: write**、**Checks: write**。
   写权限只授予检查任务；它没有代码内容写权限，也不负责合入 PR。
2. 准备开始验证前，仓库变量 `PCCL_REVIEW_ENABLED` 保持未设置或 `false`；准备好后再设为 `true`。
   任务运行前会读取这个仓库级开关。令牌由 GitHub 自动提供，不需要自行创建名为 `GITHUB_TOKEN` 的 Secret，也不需要提供个人令牌。
3. 选择一个测试 PR，在 Actions 中针对 `main` 手动运行 **PCCL Module Review**，输入 PR 编号。
   核对 **PCCL Module Owner Approval** 出现在 PR 的实际最新提交上，结果来源为 **GitHub Actions**。
   先完成下文的上线验收，再将其设为强制要求。
4. 在 `main` 的必需状态检查中加入 **PCCL Module Owner Approval**，将 **GitHub Actions 指定为预期来源**。
   必须选择这一项检查，不能用工作流的 **Reconcile module approvals** 任务代替。
   检查程序成功执行，也可能发布“不满足模块审批”的失败结果；执行成功不代表审批通过。
   同时要求 GitHub Actions 发布的 **Review policy tests** 通过，保留原有必需 CI、安全检查和原生评审保护。
5. 保留适用的 PR 审批要求、提交更新后取消旧批准、最近一次推送的审批、讨论解决要求，并要求合入前分支与目标分支保持最新。
   保留现有仓库和组织的保护规则。
6. 将对 `main` 的实际更新和合入操作，限制给指定的 Maintainer / Backup 合入团队。
   如果使用 Rulesets，应单独建立一个 **Restrict updates（限制更新）规则集**，只允许该团队通过 PR 合入时绕过这项限制，
   即使用 **For pull requests only**。
   PR 审批和必需检查放在另一个规则集中，该团队不应拥有对它的绕过权限。
   如果把所有要求放进同一个可绕过的规则集，团队也就能一并绕过审批和检查。
   平台支持时，也可以使用传统分支保护中的允许推送对象限制；具体可用设置需要在仓库中核对。
7. 分别用一位合入团队成员和一位团队外的专项 Reviewer 验证：两人都能评审，但当审批和 CI 都通过后，只有授权团队成员能够合入。

预期来源只能识别 GitHub Actions，不能进一步绑定某一个工作流文件。
将 **PCCL Module Owner Approval** 保留给本检查程序使用，其他任务、工作流或状态发布程序不要使用同名检查，并认真评审工作流及人员配置的修改。
这种来源设置不会把本检查与其他同样使用 GitHub Actions 身份的工作流隔离开。

仓库写权限作用于整个仓库，本身不表达“某个人只能合入哪个模块”。
模块审批检查决定**需要谁批准**，团队和分支限制决定**谁能执行合入**。
如果 Backup 也需要亲自合入，应把相应人员加入合入团队。
这些文件不会自动移除或限制现有协作者的写权限；上述配置需要管理员权限。

本实现没有新增 `CODEOWNERS` 文件。
如果后续单独添加 CODEOWNERS，它会作为另一套基于路径的审批要求继续生效，但不会读取 `PCCL-Modules`，也不会表达主负责人和 Backup 的邀请优先级。

## 四、触发事件、检查行为与限制

- `module-review.yml` 通过 `pull_request_target`、`workflow_run`、手动运行和主分支推送等事件执行受信任的主分支代码。
  它不会检出或执行 PR 中的代码，也不会下载 PR 生成的产物或使用其缓存。
- `module-review-events.yml` 在评审提交、编辑或撤回时发出无特权的通知，包含来自 Fork 的评审。
  该任务结束后唤醒受信任的检查程序，由检查程序重新读取 GitHub 上的 PR、文件、评审和权限。
  如果无法可靠关联 Fork 或合并引用对应的 PR，就重新检查仍然打开的目标分支 PR。通知任务的运行结果不代表审批通过。
- 大约每 15 分钟定期运行一次，补查遗漏事件并重新核对协作者权限。GitHub 可能延迟定时任务。
  检查结果没有及时更新时，可以在 Actions 中手动刷新。
- 检查程序会分页读取完整文件和评审列表，只认可针对当前 PR 最新提交的批准，排除作者自批和机器人批准，并在发布结果前再次核对实时状态。
  API 出错、文件列表不完整或 PR 过大时，都不会生成通过结果。超过 GitHub 3,000 个文件列表上限的 PR 需要拆分。
- Check Run 关联的是提交，并非某一个 PR 独有。因此，多个打开的 PR 共用同一 HEAD SHA 时会被拒绝；
  每个 PR 应使用独立提交，避免不同模块声明复用同一个通过结果。
- 评审邀请只表示请求某人参与，不是已经批准。邀请失败会显示在检查结果中，维护者可以手动邀请；
  邀请失败或成功，都不能代替缺失的负责人批准。
- 其他自动化使用 `GITHUB_TOKEN` 修改标签或 PR 描述时，可能不会再次触发工作流；此时手动刷新或等待定期检查。
  本流程不依赖自身发出的评审邀请来触发下一轮检查。
- Webhook、API 快照和检查结果更新是异步的。修改标签或描述、撤回批准、调整权限后，旧结果可能在重新检查前短暂保留。
  如果 GitHub 连新检查都无法接收，检查程序必须等服务恢复后才能使旧结果失效。
  这不是一个能把“验证审批”和“执行合入”合并为不可分割操作的服务。
  应保留原生评审保护，并在合入前核对最新范围、审批和检查；如果要求合入瞬间严格保证模块审批，第一版不适合用于无人值守的自动合入。
- 第一版**不支持合并队列（`merge_group`）**。补齐并验证合并队列检查之前，不要在启用本项必需检查的同时启用合并队列。
- 现有 `pccl checkin` 及其 Human Review Gate 保持原有职责。
  启动硬件 CI 的权限、批准模块改动的权限和执行合入的权限分别管理；这套工作流不负责版本发布权限。

## 五、验证与上线验收

使用 Node.js 20 或更高版本，运行本地审批逻辑和 API 模拟测试：

```bash
node --test tests/review/*.test.mjs
```

正式启用强制约束前，需要用真实测试 PR 验证：

- 没有模块负责人批准时不能通过；Maintainer 批准后通过；Backup 不加缺席标签也能有效批准。
- 只有专项 Reviewer 批准时不能通过；作者不能批准自己的模块改动。
- 跨模块 PR 只需某个受影响模块的 Maintainer 或 Backup 一人批准；无关模块负责人的批准不计入。
  删除标签不能从评审范围中移除路径已经匹配出的模块。
- 新提交、批准撤回、转回 Draft，以及模块范围变化后，检查结果正确刷新。
  新增模块不会使已有的有效批准失效；若批准人不再负责任何受影响模块，其批准不再计入。
- Fork 的评审能触发重新检查；通知任务延迟或不可用时，手动刷新仍然可用。
- PR 内修改人员配置或检查程序，不会改变该 PR 自身使用的审批规则。
- PR 最新提交上出现名称准确、来源为 GitHub Actions 的必需检查，且其他任务或工作流未使用同名检查。
  即使检查任务执行成功，缺少负责人批准仍然不能通过；其他 PR 不能复用同一 HEAD 的通过结果。
- 合入团队成员与团队外具有写权限的人员，实际受到预期的合入权限限制。

本地测试不能验证线上 GitHub 令牌权限、实际评审邀请、规则集的实际约束效果或 PPU CI。
完成管理员配置和真实 PR 验收后，才能将系统描述为已经启用。

## GitHub 官方参考资料

- [规则集支持的规则](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets)
- [仓库角色和权限](https://docs.github.com/en/organizations/managing-access-to-your-organizations-repositories/repository-roles-for-an-organization)
- [CODEOWNERS](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)
- [工作流触发事件](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows)
- [GITHUB_TOKEN](https://docs.github.com/en/actions/concepts/security/github_token)
- [工作流权限](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#permissions)
