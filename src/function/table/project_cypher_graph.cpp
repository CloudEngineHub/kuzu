#include "function/table/standalone_call_function.h"
#include "function/table/bind_data.h"
#include "function/table/table_function.h"
#include "processor/execution_context.h"
#include "graph/graph_entry_set.h"

using namespace kuzu::common;
using namespace kuzu::graph;

namespace kuzu {
namespace function {

struct ProjectGraphCypherBindData final : TableFuncBindData {
    std::string graphName;
    std::string cypherQuery;

    ProjectGraphCypherBindData(std::string graphName, std::string cypherQuery)
        : graphName{std::move(graphName)}, cypherQuery{std::move(cypherQuery)} {}

    std::unique_ptr<TableFuncBindData> copy() const override {
        return std::make_unique<ProjectGraphCypherBindData>(graphName, cypherQuery);
    }
};

static offset_t tableFunc(const TableFuncInput& input, TableFuncOutput&) {
    const auto bindData = ku_dynamic_cast<ProjectGraphCypherBindData*>(input.bindData);
    auto& graphEntrySet = input.context->clientContext->getGraphEntrySetUnsafe();
    // if (graphEntrySet.hasGraph(bindData->graphName)) {
    //     throw RuntimeException(
    //         stringFormat("Project graph {} already exists.", bindData->graphName));
    // }

    graphEntrySet.validateGraphNotExist(bindData->graphName);

    auto entry = graph::ParsedCypherGraphEntry();
    entry.nodeInfos = bindData->nodeInfos;
    entry.relInfos = bindData->relInfos;
    // bind graph entry to check if input is valid or not. Ignore bind result.
    GDSFunction::bindGraphEntry(*input.context->clientContext, entry);
    graphEntrySet.addGraph(bindData->graphName, entry);
    return 0;
}


static std::unique_ptr<TableFuncBindData> bindFunc(const main::ClientContext*,
    const TableFuncBindInput* input) {
    auto graphName = input->getLiteralVal<std::string>(0);
    auto cypherQuery = input->getLiteralVal<std::string>(1);
    return std::make_unique<ProjectGraphCypherBindData>(graphName, cypherQuery);
}

function_set ProjectGraphCypherFunction::getFunctionSet() {
    function_set functionSet;
    auto func = std::make_unique<TableFunction>(name,
        std::vector{LogicalTypeID::STRING, LogicalTypeID::STRING});
    func->bindFunc = bindFunc;
    func->tableFunc = tableFunc;
    func->initSharedStateFunc = TableFunction::initEmptySharedState;
    func->initLocalStateFunc = TableFunction::initEmptyLocalState;
    func->canParallelFunc = []() { return false; };
    functionSet.push_back(std::move(func));
    return functionSet;
}

}
}
