#pragma once

#include "processor/operator/sink.h"
#include "storage/store/table.h"

namespace kuzu {
namespace storage {
class MemoryManager;
class ChunkedNodeGroup;
} // namespace storage
namespace processor {

struct BatchInsertInfo {
    std::string tableName;
    catalog::TableCatalogEntry* tableEntry = nullptr;
    bool compressionEnabled = true;

    std::vector<common::LogicalType> warningColumnTypes;
    // column types include property and warning
    std::vector<common::LogicalType> columnTypes;
    std::vector<common::column_id_t> insertColumnIDs;
    std::vector<common::column_id_t> outputDataColumns;
    std::vector<common::column_id_t> warningDataColumns;

    BatchInsertInfo(std::string tableName, std::vector<common::LogicalType> warningColumnTypes)
        : tableName{std::move(tableName)}, warningColumnTypes{std::move(warningColumnTypes)} {}
    BatchInsertInfo(const BatchInsertInfo& other)
        : tableEntry{other.tableEntry},
          columnTypes{copyVector(other.columnTypes)}, insertColumnIDs{other.insertColumnIDs},
          outputDataColumns{other.outputDataColumns}, warningDataColumns{other.warningDataColumns} {
    }
    virtual ~BatchInsertInfo() = default;

    virtual std::unique_ptr<BatchInsertInfo> copy() const = 0;

    template<class TARGET>
    TARGET* ptrCast() {
        return common::ku_dynamic_cast<TARGET*>(this);
    }
};

struct KUZU_API BatchInsertSharedState {
    std::mutex mtx;
    std::atomic<common::row_idx_t> numRows;

    // Use a separate mutex for numErroredRows to avoid double-locking in local error handlers
    // As access to numErroredRows is independent of access to other shared state
    std::mutex erroredRowMutex;
    std::shared_ptr<common::row_idx_t> numErroredRows;

    storage::Table* table;

    BatchInsertSharedState()
        : numRows{0}, numErroredRows(std::make_shared<common::row_idx_t>(0)), table{nullptr} {}
    BatchInsertSharedState(const BatchInsertSharedState& other) = delete;

    virtual ~BatchInsertSharedState() = default;

    void incrementNumRows(common::row_idx_t numRowsToIncrement) {
        numRows.fetch_add(numRowsToIncrement);
    }
    common::row_idx_t getNumRows() const { return numRows.load(); }
    common::row_idx_t getNumErroredRows() {
        common::UniqLock lockGuard{erroredRowMutex};
        return *numErroredRows;
    }

    template<class TARGET>
    TARGET* ptrCast() {
        return common::ku_dynamic_cast<TARGET*>(this);
    }
};

struct BatchInsertLocalState {
    std::unique_ptr<storage::ChunkedNodeGroup> chunkedGroup;

    virtual ~BatchInsertLocalState() = default;

    template<class TARGET>
    TARGET* ptrCast() {
        return common::ku_dynamic_cast<TARGET*>(this);
    }
};

class BatchInsert : public Sink {
    static constexpr PhysicalOperatorType type_ = PhysicalOperatorType::BATCH_INSERT;

public:
    BatchInsert(std::unique_ptr<BatchInsertInfo> info, std::shared_ptr<SinkSharedState> sinkSharedState,
        std::shared_ptr<BatchInsertSharedState> sharedState, physical_op_id id,
        std::unique_ptr<OPPrintInfo> printInfo)
        : Sink{type_, id, std::move(printInfo)},
          info{std::move(info)}, sharedState{std::move(sharedState)} {}

    ~BatchInsert() override = default;

    std::unique_ptr<PhysicalOperator> copy() override = 0;

protected:
    std::unique_ptr<BatchInsertInfo> info;
    std::shared_ptr<BatchInsertSharedState> sharedState;
    std::unique_ptr<BatchInsertLocalState> localState;
};

} // namespace processor
} // namespace kuzu
