#include "processor/operator/scan/scan_node_table.h"

#include "storage/local_storage/local_node_table.h"

using namespace kuzu::common;
using namespace kuzu::storage;

namespace kuzu {
namespace processor {

void ScanNodeTableSharedState::initialize(transaction::Transaction* transaction, NodeTable* table) {
    this->table = table;
    this->currentCommittedGroupIdx = 0;
    this->currentUnCommittedGroupIdx = 0;
    this->numCommittedNodeGroups = table->getNumCommittedNodeGroups();
    if (transaction->isWriteTransaction()) {
        if (const auto localTable = transaction->getLocalStorage()->getLocalTable(
                this->table->getTableID(), LocalStorage::NotExistAction::RETURN_NULL)) {
            localNodeGroups = ku_dynamic_cast<LocalTable*, LocalNodeTable*>(localTable)
                                  ->getTableData()
                                  ->getNodeGroups();
        }
    }
    committedNodeGroupVectorIdx.resize(numCommittedNodeGroups, 0);
    committedNodeGroupFinished.resize(numCommittedNodeGroups, false);
}

void ScanNodeTableSharedState::updateVectorIdx(storage::NodeTableScanState& scanState) {
    std::unique_lock lck{mtx};
    auto& dataScanState =
        ku_dynamic_cast<TableDataScanState&, NodeDataScanState&>(*scanState.dataScanState);
    if (scanState.source == TableScanSource::COMMITTED) {
        dataScanState.vectorIdx = committedNodeGroupVectorIdx[scanState.nodeGroupIdx]++;
        if (!dataScanState.updateNumRowsToScan()) {
            committedNodeGroupFinished[scanState.nodeGroupIdx] = true;
        }
    } else {
        dataScanState.vectorIdx++;
    }
}

void ScanNodeTableSharedState::nextMorsel(NodeTableScanState& scanState) {
    std::unique_lock lck{mtx};
    // First try to give separate node groups to different threads.
    if (currentCommittedGroupIdx < numCommittedNodeGroups) {
        scanState.nodeGroupIdx = currentCommittedGroupIdx++;
        scanState.source = TableScanSource::COMMITTED;
        return;
    }
    // Then try to give local node groups to different threads.
    if (currentUnCommittedGroupIdx < localNodeGroups.size()) {
        scanState.localNodeGroup = ku_dynamic_cast<LocalNodeGroup*, LocalNodeNG*>(
            localNodeGroups[currentUnCommittedGroupIdx++]);
        scanState.source = TableScanSource::UNCOMMITTED;
        return;
    }
    // Finally, give the node group with smallest vector index to different threads.
    if (currentCommittedGroupIdx == numCommittedNodeGroups) {
        // Find node group with the smallest vector index.
        auto minVectorIdx = common::INVALID_IDX;
        auto nodeGroupIdx = common::INVALID_NODE_GROUP_IDX;
        for (auto i = 0u; i < committedNodeGroupVectorIdx.size(); i++) {
            if (committedNodeGroupVectorIdx[i] < minVectorIdx && !committedNodeGroupFinished[i]) {
                minVectorIdx = committedNodeGroupVectorIdx[i];
                nodeGroupIdx = i;
            }
        }
        if (minVectorIdx != common::INVALID_IDX) {
            scanState.source = TableScanSource::COMMITTED;
            scanState.nodeGroupIdx = nodeGroupIdx;
            return;
        }
    }
    scanState.source = TableScanSource::NONE;
}

void ScanNodeTableInfo::initScanState(NodeSemiMask* semiMask) {
    localScanState =
        std::make_unique<NodeTableScanState>(columnIDs, copyVector(columnPredicates), semiMask);
}

std::vector<NodeSemiMask*> ScanNodeTable::getSemiMasks() {
    std::vector<NodeSemiMask*> result;
    for (auto& sharedState : sharedStates) {
        result.push_back(sharedState->getSemiMask());
    }
    return result;
}

void ScanNodeTable::initLocalStateInternal(ResultSet* resultSet, ExecutionContext* context) {
    ScanTable::initLocalStateInternal(resultSet, context);
    for (auto i = 0u; i < nodeInfos.size(); ++i) {
        auto& nodeInfo = nodeInfos[i];
        nodeInfo.initScanState(sharedStates[i]->getSemiMask());
        initVectors(*nodeInfo.localScanState, *resultSet);
    }
}

void ScanNodeTable::initGlobalStateInternal(ExecutionContext* context) {
    KU_ASSERT(sharedStates.size() == nodeInfos.size());
    for (auto i = 0u; i < nodeInfos.size(); i++) {
        sharedStates[i]->initialize(context->clientContext->getTx(), nodeInfos[i].table);
    }
}

bool ScanNodeTable::getNextTuplesInternal(ExecutionContext* context) {
    auto transaction = context->clientContext->getTx();
    while (currentTableIdx < nodeInfos.size()) {
        const auto& info = nodeInfos[currentTableIdx];
        auto& scanState = *info.localScanState;
        auto skipScan =
            transaction->isReadOnly() && scanState.zoneMapResult == ZoneMapCheckResult::SKIP_SCAN;
        if (!skipScan) {
            sharedStates[currentTableIdx]->updateVectorIdx(scanState);
            while (scanState.source != TableScanSource::NONE &&
                   info.table->scan(transaction, scanState)) {
                if (scanState.nodeIDVector->state->getSelVector().getSelSize() > 0) {
                    return true;
                }
                sharedStates[currentTableIdx]->updateVectorIdx(scanState);
            }
        }
        // TODO: Take care of skipScan!!
        sharedStates[currentTableIdx]->nextMorsel(scanState);
        if (scanState.source == TableScanSource::NONE) {
            currentTableIdx++;
        } else {
            info.table->initializeScanState(transaction, scanState);
        }
    }
    return false;
}

std::unique_ptr<PhysicalOperator> ScanNodeTable::clone() {
    return make_unique<ScanNodeTable>(info.copy(), copyVector(nodeInfos), sharedStates, id,
        printInfo->copy());
}

} // namespace processor
} // namespace kuzu
