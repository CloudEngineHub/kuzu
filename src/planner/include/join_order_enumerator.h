#pragma once

#include "src/binder/query/include/normalized_single_query.h"
#include "src/catalog/include/catalog.h"
#include "src/planner/include/join_order_enumerator_context.h"
#include "src/storage/store/include/nodes_metadata.h"

using namespace graphflow::catalog;

namespace graphflow {
namespace planner {

class Enumerator;
class JoinOrderEnumeratorContext;

/**
 * JoinOrderEnumerator is currently responsible for
 *      join order enumeration
 *      filter push down
 *      property push down
 */
class JoinOrderEnumerator {
    friend class Enumerator;

public:
    JoinOrderEnumerator(
        const Catalog& catalog, const NodesMetadata& nodesMetadata, Enumerator* enumerator)
        : catalog{catalog}, nodesMetadata{nodesMetadata},
          enumerator{enumerator}, context{make_unique<JoinOrderEnumeratorContext>()} {};

    vector<unique_ptr<LogicalPlan>> enumerateJoinOrder(const QueryGraph& queryGraph,
        const shared_ptr<Expression>& queryGraphPredicate,
        vector<unique_ptr<LogicalPlan>> prevPlans);

    inline void resetState() { context->resetState(); }

private:
    unique_ptr<JoinOrderEnumeratorContext> enterSubquery(expression_vector expressionsToScan);
    void exitSubquery(unique_ptr<JoinOrderEnumeratorContext> prevContext);

    void planResultScan();

    // Initial internal ID scan for node table.
    void planNodeScan();
    // Filter push down for node table.
    void planFiltersForNode(expression_vector& predicates, NodeExpression& node, LogicalPlan& plan);
    // Property push down for node table.
    void planPropertyScansForNode(NodeExpression& node, LogicalPlan& plan);

    void planCurrentLevel();

    // Plan index nested loop join.
    void planINLJoin();
    // Node table index nested loop join (random access).
    void planNodeINLJoin(const SubqueryGraph& prevSubgraph, uint32_t nodePos,
        vector<unique_ptr<LogicalPlan>>& prevPlans);
    // Edge table index nested join.
    void planRelINLJoin(const SubqueryGraph& prevSubgraph, uint32_t relPos,
        vector<unique_ptr<LogicalPlan>>& prevPlans);
    // Filter push down for rel table.
    void planFiltersForRel(expression_vector& predicates, RelExpression& rel,
        RelDirection direction, LogicalPlan& plan);
    // Property push down for rel table.
    void planPropertyScansForRel(RelExpression& rel, RelDirection direction, LogicalPlan& plan);

    inline void planHashJoin() {
        auto maxLeftLevel = floor(context->currentLevel / 2.0);
        for (auto leftLevel = 1; leftLevel <= maxLeftLevel; ++leftLevel) {
            auto rightLevel = context->currentLevel - leftLevel;
            planHashJoin(leftLevel, rightLevel);
        }
    }
    void planHashJoin(uint32_t leftLevel, uint32_t rightLevel);

    // Filter push down for hash join.
    void planFiltersForHashJoin(expression_vector& predicates, LogicalPlan& plan);

    void appendResultScan(const expression_vector& expressionsToSelect, LogicalPlan& plan);
    void appendScanNodeID(shared_ptr<NodeExpression> queryNode, LogicalPlan& plan);
    void appendScanNodeID(
        shared_ptr<NodeExpression> queryNode, node_offset_t filter, LogicalPlan& plan);

    void appendExtend(const RelExpression& queryRel, RelDirection direction, LogicalPlan& plan);

    void planHashJoin(
        shared_ptr<NodeExpression>& joinNode, LogicalPlan& probePlan, LogicalPlan& buildPlan);
    void appendSemiMasker(shared_ptr<NodeExpression>& joinNode, LogicalPlan& plan);
    void appendASPJoin(
        shared_ptr<NodeExpression>& joinNode, LogicalPlan& probePlan, LogicalPlan& buildPlan);
    void appendSJoin(
        shared_ptr<NodeExpression>& joinNode, LogicalPlan& probePlan, LogicalPlan& buildPlan);
    void appendHashJoin(
        shared_ptr<NodeExpression>& joinNode, LogicalPlan& probePlan, LogicalPlan& buildPlan);
    shared_ptr<LogicalOperator> createHashJoin(
        shared_ptr<NodeExpression> joinNode, LogicalPlan& probePlan, LogicalPlan& buildPlan);

    expression_vector getPropertiesForVariable(Expression& expression, Expression& variable);
    uint64_t getExtensionRate(label_t boundNodeLabel, label_t relLabel, RelDirection relDirection);

private:
    const catalog::Catalog& catalog;
    const storage::NodesMetadata& nodesMetadata;
    Enumerator* enumerator;
    unique_ptr<JoinOrderEnumeratorContext> context;
};

} // namespace planner
} // namespace graphflow
