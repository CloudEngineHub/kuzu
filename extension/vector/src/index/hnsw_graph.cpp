#include "index/hnsw_graph.h"

#include "storage/local_cached_column.h"
#include "storage/store/list_chunk_data.h"
#include "storage/store/node_table.h"
#include "storage/store/rel_table.h"
#include "transaction/transaction.h"

using namespace kuzu::storage;

namespace kuzu {
namespace vector_extension {

InMemEmbeddings::InMemEmbeddings(transaction::Transaction* transaction, EmbeddingTypeInfo typeInfo,
    common::table_id_t tableID, common::column_id_t columnID)
    : EmbeddingColumn{std::move(typeInfo)} {
    auto& cacheManager = transaction->getLocalCacheManager();
    const auto key = CachedColumn::getKey(tableID, columnID);
    if (cacheManager.contains(key)) {
        data = transaction->getLocalCacheManager().at(key).cast<CachedColumn>();
    } else {
        throw common::RuntimeException("Miss cached column, which should never happen.");
    }
}

float* InMemEmbeddings::getEmbedding(common::offset_t offset) const {
    auto [nodeGroupIdx, offsetInGroup] = StorageUtils::getNodeGroupIdxAndOffsetInChunk(offset);
    KU_ASSERT(nodeGroupIdx < data->columnChunks.size());
    const auto& listChunk = data->columnChunks[nodeGroupIdx]->cast<ListChunkData>();
    return &listChunk.getDataColumnChunk()
                ->getData<float>()[listChunk.getListStartOffset(offsetInGroup)];
}

bool InMemEmbeddings::isNull(common::offset_t offset) const {
    auto [nodeGroupIdx, offsetInGroup] = StorageUtils::getNodeGroupIdxAndOffsetInChunk(offset);
    KU_ASSERT(nodeGroupIdx < data->columnChunks.size());
    const auto& listChunk = data->columnChunks[nodeGroupIdx]->cast<ListChunkData>();
    return listChunk.isNull(offsetInGroup);
}

OnDiskEmbeddingScanState::OnDiskEmbeddingScanState(const transaction::Transaction* transaction,
    MemoryManager* mm, NodeTable& nodeTable, common::column_id_t columnID) {
    std::vector columnIDs{columnID};
    // The first ValueVector in scanChunk is reserved for nodeIDs.
    std::vector<common::LogicalType> types;
    types.emplace_back(common::LogicalType::INTERNAL_ID());
    types.emplace_back(nodeTable.getColumn(columnID).getDataType().copy());
    scanChunk = Table::constructDataChunk(mm, std::move(types));
    std::vector outVectors{&scanChunk.getValueVectorMutable(1)};
    scanState = std::make_unique<NodeTableScanState>(&scanChunk.getValueVectorMutable(0),
        outVectors, scanChunk.state);
    scanState->setToTable(transaction, &nodeTable, std::move(columnIDs));
}

OnDiskEmbeddings::OnDiskEmbeddings(EmbeddingTypeInfo typeInfo, NodeTable& nodeTable)
    : EmbeddingColumn{std::move(typeInfo)}, nodeTable{nodeTable} {}

float* OnDiskEmbeddings::getEmbedding(transaction::Transaction* transaction,
    NodeTableScanState& scanState, common::offset_t offset) const {
    scanState.nodeIDVector->setValue(0, common::internalID_t{offset, nodeTable.getTableID()});
    scanState.nodeIDVector->state->getSelVectorUnsafe().setToUnfiltered(1);
    scanState.source = TableScanSource::COMMITTED;
    scanState.nodeGroupIdx = StorageUtils::getNodeGroupIdx(offset);
    nodeTable.initScanState(transaction, scanState);
    const auto result = nodeTable.lookup(transaction, scanState);
    KU_ASSERT(result);
    KU_UNUSED(result);
    KU_ASSERT(scanState.outputVectors.size() == 1 &&
              scanState.outputVectors[0]->state->getSelVector()[0] == 0);
    const auto value = scanState.outputVectors[0]->getValue<common::list_entry_t>(0);
    KU_ASSERT(value.size == typeInfo.dimension);
    KU_UNUSED(value);
    const auto dataVector = common::ListVector::getDataVector(scanState.outputVectors[0]);
    return reinterpret_cast<float*>(dataVector->getData()) + value.offset;
}

InMemHNSWGraph::~InMemHNSWGraph() = default;

std::span<InMemHNSWGraph::atomic_offset_t> SparseInMemHNSWGraph::getNeighbors(
    common::offset_t nodeOffset) {
    std::shared_lock lock(dstNodesMutex[nodeOffset]);
    if (dstNodes[nodeOffset]) {
        return {dstNodes[nodeOffset]->data(), getCSRLength(nodeOffset)};
    }
    static std::vector<std::atomic<common::offset_t>> emptyNeighbors;
    return std::span(emptyNeighbors);
}

void SparseInMemHNSWGraph::setDstNode(common::offset_t nodeOffset, common::offset_t offsetInCSRList,
    common::offset_t dstNode) {
    std::unique_lock lock(dstNodesMutex[dstNode]);
    if (!dstNodes[nodeOffset]) {
        dstNodes[nodeOffset] = std::make_unique<atomic_offset_vec_t>(maxDegree);
        for (auto i = 0u; i < maxDegree; i++) {
            dstNodes[nodeOffset]->at(i).store(common::INVALID_OFFSET, std::memory_order_relaxed);
        }
    }
    KU_ASSERT(offsetInCSRList < maxDegree);
    dstNodes[nodeOffset]->data()[offsetInCSRList].store(dstNode, std::memory_order_relaxed);
}

common::offset_t SparseInMemHNSWGraph::getDstNode(common::offset_t nodeOffset,
    common::offset_t offsetInCSRList) {
    std::shared_lock lock(dstNodesMutex[nodeOffset]);
    if (dstNodes[nodeOffset]) {
        return dstNodes[nodeOffset]->data()[offsetInCSRList].load(std::memory_order_relaxed);
    }
    return common::INVALID_OFFSET;
}

// NOLINTNEXTLINE(readability-make-member-function-const): Semantically non-const function.
void InMemHNSWGraph::finalize(MemoryManager& mm, common::node_group_idx_t nodeGroupIdx,
    const processor::PartitionerSharedState& partitionerSharedState) {
    const auto& partitionBuffers = partitionerSharedState.partitioningBuffers[0]->partitions;
    auto numRels = 0u;
    const auto startNodeOffset = StorageUtils::getStartOffsetOfNodeGroup(nodeGroupIdx);
    const auto numNodesInGroup =
        std::min(common::StorageConfig::NODE_GROUP_SIZE, numNodes - startNodeOffset);
    for (auto i = 0u; i < numNodesInGroup; i++) {
        numRels += getCSRLength(startNodeOffset + i);
    }
    finalizeNodeGroup(mm, nodeGroupIdx, numRels, partitionerSharedState.srcNodeTable->getTableID(),
        partitionerSharedState.dstNodeTable->getTableID(),
        partitionerSharedState.relTable->getTableID(), *partitionBuffers[nodeGroupIdx]);
}

void InMemHNSWGraph::finalizeNodeGroup(MemoryManager& mm, common::node_group_idx_t nodeGroupIdx,
    uint64_t numRels, common::table_id_t srcNodeTableID, common::table_id_t dstNodeTableID,
    common::table_id_t relTableID, InMemChunkedNodeGroupCollection& partition) {
    const auto startNodeOffset = StorageUtils::getStartOffsetOfNodeGroup(nodeGroupIdx);
    const auto numNodesInGroup =
        std::min(common::StorageConfig::NODE_GROUP_SIZE, numNodes - startNodeOffset);
    // BOUND_ID, NBR_ID, REL_ID.
    std::vector<common::LogicalType> columnTypes;
    columnTypes.push_back(common::LogicalType::INTERNAL_ID());
    columnTypes.push_back(common::LogicalType::INTERNAL_ID());
    columnTypes.push_back(common::LogicalType::INTERNAL_ID());
    auto chunkedNodeGroup = std::make_unique<ChunkedNodeGroup>(mm, columnTypes,
        false /* enableCompression */, numRels, 0 /* startRowIdx */, ResidencyState::IN_MEMORY);

    auto currNumRels = 0u;
    auto& boundColumnChunk = chunkedNodeGroup->getColumnChunk(0).getData();
    auto& nbrColumnChunk = chunkedNodeGroup->getColumnChunk(1).getData();
    auto& relIDColumnChunk = chunkedNodeGroup->getColumnChunk(2).getData();
    boundColumnChunk.cast<InternalIDChunkData>().setTableID(srcNodeTableID);
    nbrColumnChunk.cast<InternalIDChunkData>().setTableID(dstNodeTableID);
    relIDColumnChunk.cast<InternalIDChunkData>().setTableID(relTableID);
    for (auto i = 0u; i < numNodesInGroup; i++) {
        const auto currNodeOffset = startNodeOffset + i;
        const auto csrLen = getCSRLength(currNodeOffset);
        for (auto j = 0u; j < csrLen; j++) {
            boundColumnChunk.setValue<common::offset_t>(currNodeOffset, currNumRels);
            relIDColumnChunk.setValue<common::offset_t>(currNumRels, currNumRels);
            const auto nbrOffset = getDstNode(currNodeOffset, j);
            KU_ASSERT(nbrOffset < numNodes);
            nbrColumnChunk.setValue<common::offset_t>(nbrOffset, currNumRels);
            currNumRels++;
        }
    }
    chunkedNodeGroup->setNumRows(currNumRels);

    for (auto i = 0u; i < nbrColumnChunk.getNumValues(); i++) {
        const auto offset = nbrColumnChunk.getValue<common::offset_t>(i);
        KU_ASSERT(offset < numNodes);
        KU_UNUSED(offset);
    }
    chunkedNodeGroup->setUnused(mm);
    partition.merge(std::move(chunkedNodeGroup));
}

void DenseInMemHNSWGraph::resetDstNodes() {
    for (common::offset_t i = 0; i < numNodes; i++) {
        for (auto j = 0u; j < maxDegree; j++) {
            setDstNode(i, j, common::INVALID_OFFSET);
        }
    }
}

} // namespace vector_extension
} // namespace kuzu
