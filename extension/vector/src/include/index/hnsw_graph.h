#pragma once

#include <cmath>
#include <shared_mutex>

#include "index/hnsw_config.h"
#include "processor/operator/partitioner.h"
#include "storage/buffer_manager/memory_manager.h"
#include "storage/local_cached_column.h"

namespace kuzu {
namespace storage {
struct NodeTableScanState;
}
namespace vector_extension {

struct EmbeddingTypeInfo {
    common::LogicalType elementType;
    common::length_t dimension;
};

class EmbeddingColumn {
public:
    explicit EmbeddingColumn(EmbeddingTypeInfo typeInfo) : typeInfo{std::move(typeInfo)} {}
    virtual ~EmbeddingColumn() = default;

    const EmbeddingTypeInfo& getTypeInfo() const { return typeInfo; }
    common::length_t getDimension() const { return typeInfo.dimension; }

protected:
    EmbeddingTypeInfo typeInfo;
};

class InMemEmbeddings final : public EmbeddingColumn {
public:
    InMemEmbeddings(transaction::Transaction* transaction, EmbeddingTypeInfo typeInfo,
        common::table_id_t tableID, common::column_id_t columnID);

    float* getEmbedding(common::offset_t offset) const;
    bool isNull(common::offset_t offset) const;

private:
    storage::CachedColumn* data;
};

struct OnDiskEmbeddingScanState {
    common::DataChunk scanChunk;
    std::unique_ptr<storage::NodeTableScanState> scanState;

    OnDiskEmbeddingScanState(const transaction::Transaction* transaction,
        storage::MemoryManager* mm, storage::NodeTable& nodeTable, common::column_id_t columnID);
};

class OnDiskEmbeddings final : public EmbeddingColumn {
public:
    OnDiskEmbeddings(EmbeddingTypeInfo typeInfo, storage::NodeTable& nodeTable);

    float* getEmbedding(transaction::Transaction* transaction,
        storage::NodeTableScanState& scanState, common::offset_t offset) const;

private:
    storage::NodeTable& nodeTable;
};

struct NodeWithDistance {
    common::offset_t nodeOffset;
    double_t distance;

    NodeWithDistance(common::offset_t nodeOffset, double_t distance)
        : nodeOffset{nodeOffset}, distance{distance} {}
};

struct HNSWGraphInfo {
    common::offset_t numNodes;
    EmbeddingColumn* embeddings;
    MetricType distFunc;

    HNSWGraphInfo(common::offset_t numNodes, EmbeddingColumn* embeddings, MetricType distFunc)
        : numNodes{numNodes}, embeddings{embeddings}, distFunc{distFunc} {}
};

class InMemHNSWGraph {
public:
    using atomic_offset_t = std::atomic<common::offset_t>;
    using atomic_offset_vec_t = std::vector<std::atomic<common::offset_t>>;

    InMemHNSWGraph(storage::MemoryManager* mm, common::offset_t numNodes,
        common::length_t maxDegree)
        : numNodes{numNodes}, maxDegree{maxDegree} {
        csrLengthBuffer = mm->allocateBuffer(true, numNodes * sizeof(std::atomic<uint16_t>));
        csrLengths = reinterpret_cast<std::atomic<uint16_t>*>(csrLengthBuffer->getData());
        resetCSRLengths();
    }
    virtual ~InMemHNSWGraph();

    common::length_t getMaxDegree() const { return maxDegree; }

    virtual std::span<atomic_offset_t> getNeighbors(common::offset_t nodeOffset) = 0;
    uint16_t getCSRLength(common::offset_t nodeOffset) const {
        return csrLengths[nodeOffset].load(std::memory_order_relaxed);
    }
    // NOLINTNEXTLINE(readability-make-member-function-const): Semantically non-const function.
    void setCSRLength(common::offset_t nodeOffset, uint16_t length) {
        csrLengths[nodeOffset].store(length, std::memory_order_relaxed);
    }
    // NOLINTNEXTLINE(readability-make-member-function-const): Semantically non-const function.
    uint16_t incrementCSRLength(common::offset_t nodeOffset) {
        return csrLengths[nodeOffset].fetch_add(1, std::memory_order_relaxed);
    }
    virtual void setDstNode(common::offset_t nodeOffset, common::offset_t offsetInCSRList,
        common::offset_t dstNode) = 0;
    virtual common::offset_t getDstNode(common::offset_t nodeOffset,
        common::offset_t offsetInCSRList) = 0;
    void finalize(storage::MemoryManager& mm, common::node_group_idx_t nodeGroupIdx,
        const processor::PartitionerSharedState& partitionerSharedState);

protected:
    void resetCSRLengths() {
        for (common::offset_t i = 0; i < numNodes; i++) {
            setCSRLength(i, 0);
        }
    }

private:
    void finalizeNodeGroup(storage::MemoryManager& mm, common::node_group_idx_t nodeGroupIdx,
        uint64_t numRels, common::table_id_t srcNodeTableID, common::table_id_t dstNodeTableID,
        common::table_id_t relTableID, storage::InMemChunkedNodeGroupCollection& partition);

protected:
    common::offset_t numNodes;
    // Max allowed degree of a node in the graph before shrinking.
    common::length_t maxDegree;
    std::unique_ptr<storage::MemoryBuffer> csrLengthBuffer;
    std::atomic<uint16_t>* csrLengths;
};

class SparseInMemHNSWGraph final : public InMemHNSWGraph {
public:
    SparseInMemHNSWGraph(storage::MemoryManager* mm, common::offset_t numNodes,
        common::length_t maxDegree)
        : InMemHNSWGraph{mm, numNodes, maxDegree}, dstNodes{numNodes}, dstNodesMutex{numNodes} {}

    std::span<std::atomic<common::offset_t>> getNeighbors(common::offset_t nodeOffset) override;
    void setDstNode(common::offset_t nodeOffset, common::offset_t offsetInCSRList,
        common::offset_t dstNode) override;
    common::offset_t getDstNode(common::offset_t nodeOffset,
        common::offset_t offsetInCSRList) override;

private:
    std::vector<std::unique_ptr<atomic_offset_vec_t>> dstNodes;
    std::vector<std::shared_mutex> dstNodesMutex;
};

class DenseInMemHNSWGraph final : public InMemHNSWGraph {
public:
    DenseInMemHNSWGraph(storage::MemoryManager* mm, common::offset_t numNodes,
        common::length_t maxDegree)
        : InMemHNSWGraph{mm, numNodes, maxDegree} {
        dstNodesBuffer =
            mm->allocateBuffer(false, numNodes * maxDegree * sizeof(std::atomic<common::offset_t>));
        dstNodes = reinterpret_cast<std::atomic<common::offset_t>*>(dstNodesBuffer->getData());
        resetDstNodes();
    }

    std::span<std::atomic<common::offset_t>> getNeighbors(common::offset_t nodeOffset) override {
        const auto numNbrs = getCSRLength(nodeOffset);
        return {&dstNodes[nodeOffset * maxDegree], numNbrs};
    }

    // NOLINTNEXTLINE(readability-make-member-function-const): Semantically non-const function.
    void setDstNode(common::offset_t nodeOffset, common::offset_t offsetInCSRList,
        common::offset_t dstNode) override {
        const auto csrOffset = nodeOffset * maxDegree + offsetInCSRList;
        dstNodes[csrOffset].store(dstNode, std::memory_order_relaxed);
    }

private:
    void resetDstNodes();

    common::offset_t getDstNode(common::offset_t nodeOffset,
        common::offset_t offsetInCSRList) override {
        const auto csrOffset = nodeOffset * maxDegree + offsetInCSRList;
        return dstNodes[csrOffset].load(std::memory_order_relaxed);
    }

private:
    std::unique_ptr<storage::MemoryBuffer> dstNodesBuffer;
    atomic_offset_t* dstNodes;
};

} // namespace vector_extension
} // namespace kuzu
