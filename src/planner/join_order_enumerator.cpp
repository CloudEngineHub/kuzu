#include "src/planner/include/join_order_enumerator.h"

#include "src/binder/expression/include/function_expression.h"
#include "src/planner/include/enumerator.h"
#include "src/planner/logical_plan/logical_operator/include/logical_extend.h"
#include "src/planner/logical_plan/logical_operator/include/logical_hash_join.h"
#include "src/planner/logical_plan/logical_operator/include/logical_intersect.h"
#include "src/planner/logical_plan/logical_operator/include/logical_result_scan.h"
#include "src/planner/logical_plan/logical_operator/include/logical_scan_node_id.h"
#include "src/planner/logical_plan/logical_operator/include/logical_semi_masker.h"

namespace graphflow {
namespace planner {

static expression_vector getNewMatchedExpressions(const SubqueryGraph& prevSubgraph,
    const SubqueryGraph& newSubgraph, const expression_vector& expressions);
static expression_vector getNewMatchedExpressions(const SubqueryGraph& prevLeftSubgraph,
    const SubqueryGraph& prevRightSubgraph, const SubqueryGraph& newSubgraph,
    const expression_vector& expressions);
static shared_ptr<Expression> createNodeIDComparison(const shared_ptr<Expression>& left,
    const shared_ptr<Expression>& right, BuiltInVectorOperations* builtInFunctions);

// Rewrite a query rel that closes a cycle as a regular query rel for extend. This requires giving a
// different identifier to the node that will close the cycle. This identifier is created as rel
// name + node name.
static shared_ptr<RelExpression> rewriteQueryRel(const RelExpression& queryRel, bool isRewriteDst);

vector<unique_ptr<LogicalPlan>> JoinOrderEnumerator::enumerateJoinOrder(
    const QueryGraph& queryGraph, const shared_ptr<Expression>& queryGraphPredicate,
    vector<unique_ptr<LogicalPlan>> prevPlans) {
    context->init(queryGraph, queryGraphPredicate, move(prevPlans));
    context->hasExpressionsToScanFromOuter() ? planResultScan() : planNodeScan();
    context->currentLevel++;
    while (context->currentLevel < context->maxLevel) {
        planCurrentLevel();
        context->currentLevel++;
    }
    return move(context->getPlans(context->getFullyMatchedSubqueryGraph()));
}

unique_ptr<JoinOrderEnumeratorContext> JoinOrderEnumerator::enterSubquery(
    expression_vector expressionsToScan) {
    auto prevContext = move(context);
    context = make_unique<JoinOrderEnumeratorContext>();
    context->setExpressionsToScanFromOuter(move(expressionsToScan));
    return prevContext;
}

void JoinOrderEnumerator::exitSubquery(unique_ptr<JoinOrderEnumeratorContext> prevContext) {
    context = move(prevContext);
}

void JoinOrderEnumerator::planResultScan() {
    auto emptySubgraph = context->getEmptySubqueryGraph();
    auto plan = context->containPlans(emptySubgraph) ?
                    context->getPlans(emptySubgraph)[0]->shallowCopy() :
                    make_unique<LogicalPlan>();
    auto newSubgraph = emptySubgraph;
    for (auto& expression : context->getExpressionsToScanFromOuter()) {
        if (expression->dataType.typeID == NODE_ID) {
            assert(expression->expressionType == PROPERTY);
            newSubgraph.addQueryNode(context->getQueryGraph()->getQueryNodePos(
                expression->getChild(0)->getUniqueName()));
        }
    }
    appendResultScan(context->getExpressionsToScanFromOuter(), *plan);
    for (auto& expression :
        getNewMatchedExpressions(emptySubgraph, newSubgraph, context->getWhereExpressions())) {
        enumerator->appendFilter(expression, *plan);
    }
    context->addPlan(newSubgraph, move(plan));
}

void JoinOrderEnumerator::planNodeScan() {
    auto emptySubgraph = context->getEmptySubqueryGraph();
    if (context->getMatchedQueryNodes().count() == 1) {
        // If only single node has been previously enumerated, then join order is decided
        return;
    }
    auto queryGraph = context->getQueryGraph();
    for (auto nodePos = 0u; nodePos < queryGraph->getNumQueryNodes(); ++nodePos) {
        auto newSubgraph = context->getEmptySubqueryGraph();
        newSubgraph.addQueryNode(nodePos);
        auto plan = make_unique<LogicalPlan>();
        auto node = queryGraph->getQueryNode(nodePos);
        appendScanNodeID(node, *plan);
        auto predicates =
            getNewMatchedExpressions(emptySubgraph, newSubgraph, context->getWhereExpressions());
        planFiltersForNode(predicates, *node, *plan);
        planPropertyScansForNode(*node, *plan);
        context->addPlan(newSubgraph, move(plan));
    }
}

void JoinOrderEnumerator::planFiltersForNode(
    expression_vector& predicates, NodeExpression& node, LogicalPlan& plan) {
    for (auto& predicate : predicates) {
        auto propertiesToScan = getPropertiesForVariable(*predicate, node);
        enumerator->appendScanNodePropIfNecessarySwitch(propertiesToScan, node, plan);
        enumerator->appendFilter(predicate, plan);
    }
}

void JoinOrderEnumerator::planPropertyScansForNode(NodeExpression& node, LogicalPlan& plan) {
    auto properties = enumerator->getPropertiesForNode(node);
    enumerator->appendScanNodePropIfNecessarySwitch(properties, node, plan);
}

void JoinOrderEnumerator::planCurrentLevel() {
    assert(context->currentLevel > 1);
    planINLJoin();
    planHashJoin();
    context->subPlansTable->finalizeLevel(context->currentLevel);
}

void JoinOrderEnumerator::planINLJoin() {
    auto prevLevel = context->currentLevel - 1;
    for (auto& [subgraph, plans] : *context->subPlansTable->getSubqueryGraphPlansMap(prevLevel)) {
        auto nodeNbrPositions = subgraph.getNodeNbrPositions();
        for (auto& nodePos : nodeNbrPositions) {
            planNodeINLJoin(subgraph, nodePos, plans);
        }
        auto relNbrPositions = subgraph.getRelNbrPositions();
        for (auto& relPos : relNbrPositions) {
            planRelINLJoin(subgraph, relPos, plans);
        }
    }
}

void JoinOrderEnumerator::planNodeINLJoin(const SubqueryGraph& prevSubgraph, uint32_t nodePos,
    vector<unique_ptr<LogicalPlan>>& prevPlans) {
    auto newSubgraph = prevSubgraph;
    newSubgraph.addQueryNode(nodePos);
    auto node = context->mergedQueryGraph->getQueryNode(nodePos);
    auto predicates =
        getNewMatchedExpressions(prevSubgraph, newSubgraph, context->getWhereExpressions());
    for (auto& prevPlan : prevPlans) {
        auto plan = prevPlan->shallowCopy();
        planFiltersForNode(predicates, *node, *plan);
        planPropertyScansForNode(*node, *plan);
        plan->multiplyCost(EnumeratorKnobs::RANDOM_LOOKUP_PENALTY);
        context->addPlan(newSubgraph, move(plan));
    }
}

void JoinOrderEnumerator::planRelINLJoin(const SubqueryGraph& prevSubgraph, uint32_t relPos,
    vector<unique_ptr<LogicalPlan>>& prevPlans) {
    // Consider query MATCH (a)-[r1]->(b)-[r2]->(c)-[r3]->(d) WITH *
    // MATCH (d)->[r4]->(e)-[r5]->(f) RETURN *
    // First MATCH is enumerated normally. When enumerating second MATCH,
    // we first merge graph as (a)-[r1]->(b)-[r2]->(c)-[r3]->(d)->[r4]->(e)-[r5]->(f) and
    // enumerate from level 0 again. If we hit a query rel that has been previously matched
    // i.e. r1 & r2 & r3, we skip the plan. This guarantees DP only enumerate query rels in
    // the second MATCH. Note this is different from fully merged, since we don't generate
    // plans like build side QVO : a, b, c,  probe side QVO: f, e, d, c, HashJoin(c).
    if (context->matchedQueryRels[relPos]) {
        return;
    }
    auto newSubgraph = prevSubgraph;
    newSubgraph.addQueryRel(relPos);
    auto rel = context->mergedQueryGraph->getQueryRel(relPos);
    auto predicates =
        getNewMatchedExpressions(prevSubgraph, newSubgraph, context->getWhereExpressions());
    // Note isClosingRel check is different from checking src&dst connectivity.
    // Consider triangle query example (a)-[e1]->(b)-[e2]->(c), a-[e3]->(c), and prevSubgraph 'sg'
    // is a-e1-b-e2, a rel neighbour for 'sg' is e3. e3 is connected to 'sg' on node 'a' but not
    // node 'c' since 'c' is not part of 'sg'. However, we need to treat e3 as a closing edge. So
    // isClosingRel checks only for rel and ignoring whether a node has been matched or not.
    if (prevSubgraph.isClosingRel(relPos)) {
        for (auto direction : REL_DIRECTIONS) { // closing direction
            auto isCloseOnDst = direction == FWD;
            auto tmpRel = rewriteQueryRel(*rel, isCloseOnDst); // break cycle
            auto closeNode = isCloseOnDst ? rel->getDstNode() : rel->getSrcNode();
            auto tmpCloseNode = isCloseOnDst ? tmpRel->getDstNode() : tmpRel->getSrcNode();
            for (auto& prevPlan : prevPlans) { // filter-based solution
                auto plan = prevPlan->shallowCopy();
                appendExtend(*tmpRel, direction, *plan);
                auto idComparison = createNodeIDComparison(closeNode->getNodeIDPropertyExpression(),
                    tmpCloseNode->getNodeIDPropertyExpression(),
                    catalog.getBuiltInScalarFunctions());
                enumerator->appendFilter(idComparison, *plan);
                planFiltersForRel(predicates, *tmpRel, direction, *plan);
                planPropertyScansForRel(*tmpRel, direction, *plan);
                context->addPlan(newSubgraph, move(plan));
            }
            for (auto& prevPlan : prevPlans) { // intersect-based solution
                auto plan = prevPlan->shallowCopy();
                appendExtend(*tmpRel, direction, *plan);
                planFiltersForRel(predicates, *tmpRel, direction, *plan);
                planPropertyScansForRel(*tmpRel, direction, *plan);
                if (appendIntersect(
                        closeNode->getIDProperty(), tmpCloseNode->getIDProperty(), *plan)) {
                    context->addPlan(newSubgraph, move(plan));
                }
            }
        }
    } else {
        auto isSrcConnected = prevSubgraph.isSrcConnected(relPos);
        auto isDstConnected = prevSubgraph.isDstConnected(relPos);
        assert(isSrcConnected || isDstConnected);
        auto direction = isSrcConnected ? FWD : BWD;
        for (auto& prevPlan : prevPlans) {
            auto plan = prevPlan->shallowCopy();
            appendExtend(*rel, direction, *plan);
            planFiltersForRel(predicates, *rel, direction, *plan);
            planPropertyScansForRel(*rel, direction, *plan);
            context->addPlan(newSubgraph, move(plan));
        }
    }
}

void JoinOrderEnumerator::planFiltersForRel(
    expression_vector& predicates, RelExpression& rel, RelDirection direction, LogicalPlan& plan) {
    for (auto& predicate : predicates) {
        auto relPropertiesToScan = getPropertiesForVariable(*predicate, rel);
        enumerator->appendScanRelPropsIfNecessary(relPropertiesToScan, rel, direction, plan);
        enumerator->appendFilter(predicate, plan);
    }
}

void JoinOrderEnumerator::planPropertyScansForRel(
    RelExpression& rel, RelDirection direction, LogicalPlan& plan) {
    auto relProperties = enumerator->getPropertiesForRel(rel);
    enumerator->appendScanRelPropsIfNecessary(relProperties, rel, direction, plan);
}

void JoinOrderEnumerator::planHashJoin(uint32_t leftLevel, uint32_t rightLevel) {
    assert(leftLevel <= rightLevel);
    auto rightSubgraphPlansMap = context->subPlansTable->getSubqueryGraphPlansMap(rightLevel);
    for (auto& [rightSubgraph, rightPlans] : *rightSubgraphPlansMap) {
        for (auto& nbrSubgraph : rightSubgraph.getNbrSubgraphs(leftLevel)) {
            // Consider previous example in enumerateExtend(), when enumerating second MATCH, and
            // current level = 4 we get left subgraph as f, d, e (size = 2), and try to find a
            // connected right subgraph of size 2. A possible right graph could be b, c, d. However,
            // b, c, d is a subgraph enumerated in the first MATCH and has been cleared before
            // enumeration of second MATCH. So subPlansTable does not contain this subgraph.
            if (!context->containPlans(nbrSubgraph)) {
                continue;
            }
            auto joinNodePositions = rightSubgraph.getConnectedNodePos(nbrSubgraph);
            assert(joinNodePositions.size() == 1);
            auto joinNode = context->mergedQueryGraph->getQueryNode(joinNodePositions[0]);
            auto& leftPlans = context->getPlans(nbrSubgraph);
            auto newSubgraph = rightSubgraph;
            newSubgraph.addSubqueryGraph(nbrSubgraph);
            auto predicates = getNewMatchedExpressions(
                nbrSubgraph, rightSubgraph, newSubgraph, context->getWhereExpressions());
            for (auto& leftPlan : leftPlans) {
                for (auto& rightPlan : rightPlans) {
                    auto leftPlanProbeCopy = leftPlan->shallowCopy();
                    auto rightPlanBuildCopy = rightPlan->shallowCopy();
                    auto leftPlanBuildCopy = leftPlan->shallowCopy();
                    auto rightPlanProbeCopy = rightPlan->shallowCopy();
                    planHashJoin(joinNode, *leftPlanProbeCopy, *rightPlanBuildCopy);
                    planFiltersForHashJoin(predicates, *leftPlanProbeCopy);
                    context->addPlan(newSubgraph, move(leftPlanProbeCopy));
                    // flip build and probe side to get another HashJoin plan
                    if (leftLevel != rightLevel) {
                        planHashJoin(joinNode, *rightPlanProbeCopy, *leftPlanBuildCopy);
                        planFiltersForHashJoin(predicates, *rightPlanProbeCopy);
                        context->addPlan(newSubgraph, move(rightPlanProbeCopy));
                    }
                }
            }
        }
    }
}

void JoinOrderEnumerator::planFiltersForHashJoin(expression_vector& predicates, LogicalPlan& plan) {
    for (auto& predicate : predicates) {
        enumerator->appendFilter(predicate, plan);
    }
}

void JoinOrderEnumerator::appendResultScan(
    const expression_vector& expressionsToSelect, LogicalPlan& plan) {
    auto schema = plan.getSchema();
    auto groupPos = schema->createGroup();
    for (auto& expressionToSelect : expressionsToSelect) {
        schema->insertToGroupAndScope(expressionToSelect, groupPos);
    }
    auto resultScan = make_shared<LogicalResultScan>(expressionsToSelect);
    auto group = schema->getGroup(groupPos);
    group->setIsFlat(true);
    plan.appendOperator(move(resultScan));
}

void JoinOrderEnumerator::appendScanNodeID(
    shared_ptr<NodeExpression> queryNode, LogicalPlan& plan) {
    auto schema = plan.getSchema();
    assert(plan.isEmpty());
    auto groupPos = schema->createGroup();
    schema->insertToGroupAndScope(queryNode->getNodeIDPropertyExpression(), groupPos);
    auto numNodes = nodesMetadata.getNodeMetadata(queryNode->getLabel())->getMaxNodeOffset() + 1;
    auto scan = make_shared<LogicalScanNodeID>(move(queryNode));
    schema->getGroup(groupPos)->setMultiplier(numNodes);
    plan.appendOperator(move(scan));
}

void JoinOrderEnumerator::appendExtend(
    const RelExpression& queryRel, RelDirection direction, LogicalPlan& plan) {
    auto schema = plan.getSchema();
    auto boundNode = FWD == direction ? queryRel.getSrcNode() : queryRel.getDstNode();
    auto nbrNode = FWD == direction ? queryRel.getDstNode() : queryRel.getSrcNode();
    auto boundNodeID = boundNode->getIDProperty();
    auto nbrNodeID = nbrNode->getIDProperty();
    auto isColumnExtend = catalog.getReadOnlyVersion()->isSingleMultiplicityInDirection(
        queryRel.getLabel(), direction);
    auto boundNodeGroupPos = schema->getGroupPos(boundNodeID);
    uint32_t nbrGroupPos;
    // If the join is a single (1-hop) fixed-length column extend (e.g., over a relationship with
    // one-to-one multiplicity), then we put the nbrNode vector into the same
    // datachunk/factorization group as the boundNodeId. Otherwise (including a var-length join over
    // a column extend) to a separate data chunk, which will be unflat. However, note that a
    // var-length column join can still write a single value to this unflat nbrNode vector (i.e.,
    // the vector in this case can be an unflat vector with a single value in it).
    if (isColumnExtend && (queryRel.getLowerBound() == 1) &&
        (queryRel.getLowerBound() == queryRel.getUpperBound())) {
        nbrGroupPos = boundNodeGroupPos;
    } else {
        Enumerator::appendFlattenIfNecessary(boundNodeGroupPos, plan);
        nbrGroupPos = schema->createGroup();
        auto nbrNodeGroup = schema->getGroup(nbrGroupPos);
        nbrNodeGroup->setMultiplier(
            getExtensionRate(boundNode->getLabel(), queryRel.getLabel(), direction));
    }
    auto extend = make_shared<LogicalExtend>(boundNode, nbrNode, queryRel.getLabel(), direction,
        isColumnExtend, queryRel.getLowerBound(), queryRel.getUpperBound(), plan.getLastOperator());
    schema->insertToGroupAndScope(nbrNode->getNodeIDPropertyExpression(), nbrGroupPos);
    plan.increaseCost(plan.getCardinality());
    plan.appendOperator(move(extend));
}

static void collectScanNodeIDRecursive(LogicalOperator* op, vector<LogicalOperator*>& scanNodeIDs) {
    if (op->getLogicalOperatorType() == LOGICAL_SCAN_NODE_ID) {
        scanNodeIDs.push_back(op);
        return;
    }
    for (auto i = 0u; i < op->getNumChildren(); ++i) {
        collectScanNodeIDRecursive(op->getChild(i).get(), scanNodeIDs);
    }
}

static vector<LogicalOperator*> collectScanNodeID(LogicalOperator* op) {
    vector<LogicalOperator*> result;
    collectScanNodeIDRecursive(op, result);
    return result;
}

static bool tryPassMask(NodeExpression* joinNode, vector<LogicalOperator*>& scanNodeIDs) {
    for (auto& op : scanNodeIDs) {
        assert(op->getLogicalOperatorType() == LOGICAL_SCAN_NODE_ID);
        auto scanNodeID = (LogicalScanNodeID*)op;
        if (scanNodeID->getNodeExpression()->getUniqueName() == joinNode->getUniqueName()) {
            return true;
        }
    }
    return false;
}

void JoinOrderEnumerator::planHashJoin(
    shared_ptr<NodeExpression>& joinNode, LogicalPlan& probePlan, LogicalPlan& buildPlan) {
    auto buildSideScanNodeIDs = collectScanNodeID(buildPlan.getLastOperator().get());
    bool canPassMaskFromProbeToBuild = tryPassMask(joinNode.get(), buildSideScanNodeIDs);
    auto probeSideScanNodeIDs = collectScanNodeID(probePlan.getLastOperator().get());
    bool canPassMaskFromBuildToProbe = tryPassMask(joinNode.get(), probeSideScanNodeIDs);
    assert(!(canPassMaskFromProbeToBuild && canPassMaskFromBuildToProbe));
    if (canPassMaskFromProbeToBuild) {
        appendASPJoin(joinNode, probePlan, buildPlan);
    } else if (canPassMaskFromBuildToProbe) {
        appendSJoin(joinNode, probePlan, buildPlan);
    } else {
        appendHashJoin(joinNode, probePlan, buildPlan);
    }
}

static bool isJoinKeyUniqueOnBuildSide(const string& joinNodeID, LogicalPlan& buildPlan) {
    auto buildSchema = buildPlan.getSchema();
    auto numGroupsInScope = buildSchema->getGroupsPosInScope().size();
    bool hasProjectedOutGroups = buildSchema->getNumGroups() > numGroupsInScope;
    if (numGroupsInScope > 1 || hasProjectedOutGroups) {
        return false;
    }
    // Now there is a single factorization group, we need to further make sure joinNodeID comes from
    // ScanNodeID operator. Because if joinNodeID comes from a ColExtend we cannot guarantee the
    // reverse mapping is still many-to-one. We look for the most simple pattern where build plan is
    // linear.
    auto firstop = buildPlan.getLastOperator().get();
    while (firstop->getNumChildren() != 0) {
        if (firstop->getNumChildren() > 1) {
            return false;
        }
        firstop = firstop->getChild(0).get();
    }
    if (firstop->getLogicalOperatorType() != LOGICAL_SCAN_NODE_ID) {
        return false;
    }
    auto scanNodeID = (LogicalScanNodeID*)firstop;
    if (scanNodeID->getNodeExpression()->getIDProperty() != joinNodeID) {
        return false;
    }
    return true;
}

void JoinOrderEnumerator::appendSemiMasker(
    shared_ptr<NodeExpression>& joinNode, LogicalPlan& plan) {
    auto semiMasker = make_shared<LogicalSemiMasker>(joinNode, plan.getLastOperator());
    plan.appendOperator(move(semiMasker));
}

void JoinOrderEnumerator::appendASPJoin(
    shared_ptr<NodeExpression>& joinNode, LogicalPlan& probePlan, LogicalPlan& buildPlan) {
    appendSemiMasker(joinNode, probePlan);
    Enumerator::appendSink(probePlan);
    auto hashJoin =
        static_pointer_cast<LogicalHashJoin>(createHashJoin(joinNode, probePlan, buildPlan));
    hashJoin->setJoinType(HashJoinType::ASP_JOIN);
    probePlan.appendOperator(move(hashJoin));
}

void JoinOrderEnumerator::appendSJoin(
    shared_ptr<NodeExpression>& joinNode, LogicalPlan& probePlan, LogicalPlan& buildPlan) {
    appendSemiMasker(joinNode, buildPlan);
    auto hashJoin =
        static_pointer_cast<LogicalHashJoin>(createHashJoin(joinNode, probePlan, buildPlan));
    hashJoin->setJoinType(HashJoinType::S_JOIN);
    probePlan.appendOperator(move(hashJoin));
}

void JoinOrderEnumerator::appendHashJoin(
    shared_ptr<NodeExpression>& joinNode, LogicalPlan& probePlan, LogicalPlan& buildPlan) {
    auto hashJoin = createHashJoin(joinNode, probePlan, buildPlan);
    probePlan.appendOperator(move(hashJoin));
}

shared_ptr<LogicalOperator> JoinOrderEnumerator::createHashJoin(
    shared_ptr<NodeExpression> joinNode, LogicalPlan& probePlan, LogicalPlan& buildPlan) {
    auto joinNodeID = joinNode->getIDProperty();
    auto& buildSideSchema = *buildPlan.getSchema();
    auto buildSideKeyGroupPos = buildSideSchema.getGroupPos(joinNodeID);
    auto probeSideSchema = probePlan.getSchema();
    auto probeSideKeyGroupPos = probeSideSchema->getGroupPos(joinNodeID);
    probePlan.increaseCost(probePlan.getCardinality() + buildPlan.getCardinality());
    // Flat probe side key group if the build side contains more than one group or the build side
    // has projected out data chunks, which may increase the multiplicity of data chunks in the
    // build side. The core idea is to keep probe side key unflat only when we know that there is
    // only 0 or 1 match for each key.
    if (!isJoinKeyUniqueOnBuildSide(joinNodeID, buildPlan)) {
        Enumerator::appendFlattenIfNecessary(probeSideKeyGroupPos, probePlan);
        // Update probe side cardinality if build side does not guarantee to have 0/1 match
        probePlan.multiplyCardinality(
            buildPlan.getCardinality() * EnumeratorKnobs::PREDICATE_SELECTIVITY);
        probePlan.multiplyCost(EnumeratorKnobs::FLAT_PROBE_PENALTY);
    }
    // Merge key group from build side into probe side.
    for (auto& expression : buildSideSchema.getExpressionsInScope(buildSideKeyGroupPos)) {
        if (expression->getUniqueName() == joinNodeID) {
            continue;
        }
        probeSideSchema->insertToGroupAndScope(expression, probeSideKeyGroupPos);
    }
    // Merge build side payload groups to the result.
    unordered_set<uint32_t> buildSidePayloadGroupsPos;
    bool buildSideHasUnflatPayloads = false;
    for (auto& groupPos : buildSideSchema.getGroupsPosInScope()) {
        if (groupPos == buildSideKeyGroupPos) {
            continue;
        }
        buildSidePayloadGroupsPos.insert(groupPos);
        if (!buildSideSchema.getGroup(groupPos)->getIsFlat()) {
            buildSideHasUnflatPayloads = true;
        }
    }
    auto resultSchema = probeSideSchema;
    auto numGroupsBefore = resultSchema->getNumGroups();
    Enumerator::computeSchemaForSinkOperators(
        buildSidePayloadGroupsPos, buildSideSchema, *resultSchema);
    // When the build side payload has no unflat groups, and build side key has more than 1 vectors,
    // we flatten newly added groups from the build side in the output. This is because we cannot
    // guarantee a 1-1 mapping between the key and other payloads within the key data chunk (take
    // rel properties as an example). Notice that if there is any unflat groups in the build side,
    // they are already flatten in previous call `computeSchemaForSinkOperators`. Otherwise, they
    // may still stay unflat. Thus, we further apply this rule.
    bool flattenBuildSideOutput =
        !buildSideHasUnflatPayloads &&
        buildSideSchema.getExpressionsInScope(buildSideKeyGroupPos).size() > 1;
    vector<uint64_t> flatOutputGroupPositions;
    for (auto i = numGroupsBefore; i < resultSchema->getNumGroups(); ++i) {
        if (resultSchema->getGroup(i)->getIsFlat()) {
            flatOutputGroupPositions.push_back(i);
        } else if (flattenBuildSideOutput) {
            flatOutputGroupPositions.push_back(i);
            resultSchema->flattenGroup(i);
        }
    }
    // Here, for flat probe side key, we decide for the operator if the final output is a flat
    // tuple. Three cases are considered: 1) build side output is already flat, then the output is
    // also flat; 2) build side has no payloads, we also need to output a flat tuple because we lose
    // the multiplicity information from the build side in the final output; 3) build side contains
    // unflat payload, we must output a flat tuple for form a correct f-structure.
    bool isOutputAFlatTuple = false;
    if (probeSideSchema->getGroup(probeSideKeyGroupPos)->getIsFlat() &&
        (flattenBuildSideOutput || buildSidePayloadGroupsPos.empty() ||
            buildSideHasUnflatPayloads)) {
        isOutputAFlatTuple = true;
    }
    return make_shared<LogicalHashJoin>(joinNode, buildSideSchema.copy(), flatOutputGroupPositions,
        buildSideSchema.getExpressionsInScope(), isOutputAFlatTuple, probePlan.getLastOperator(),
        buildPlan.getLastOperator());
}

bool JoinOrderEnumerator::appendIntersect(
    const string& leftNodeID, const string& rightNodeID, LogicalPlan& plan) {
    auto schema = plan.getSchema();
    auto leftGroupPos = schema->getGroupPos(leftNodeID);
    auto rightGroupPos = schema->getGroupPos(rightNodeID);
    auto& leftGroup = *schema->getGroup(leftGroupPos);
    auto& rightGroup = *schema->getGroup(rightGroupPos);
    if (leftGroup.getIsFlat() || rightGroup.getIsFlat()) {
        // We should use filter close cycle if any group is flat.
        return false;
    }
    auto intersect = make_shared<LogicalIntersect>(leftNodeID, rightNodeID, plan.getLastOperator());
    plan.appendOperator(move(intersect));
    return true;
}

expression_vector JoinOrderEnumerator::getPropertiesForVariable(
    Expression& expression, Expression& variable) {
    expression_vector result;
    for (auto& propertyExpression : expression.getSubPropertyExpressions()) {
        if (propertyExpression->getChild(0)->getUniqueName() != variable.getUniqueName()) {
            continue;
        }
        result.push_back(propertyExpression);
    }
    return result;
}

uint64_t JoinOrderEnumerator::getExtensionRate(
    label_t boundNodeLabel, label_t relLabel, RelDirection relDirection) {
    auto numRels = catalog.getReadOnlyVersion()->getNumRelsForDirectionBoundLabel(
        relLabel, relDirection, boundNodeLabel);
    return ceil(
        (double)numRels / nodesMetadata.getNodeMetadata(boundNodeLabel)->getMaxNodeOffset() + 1);
}

expression_vector getNewMatchedExpressions(const SubqueryGraph& prevSubgraph,
    const SubqueryGraph& newSubgraph, const expression_vector& expressions) {
    expression_vector newMatchedExpressions;
    for (auto& expression : expressions) {
        auto includedVariables = expression->getDependentVariableNames();
        if (newSubgraph.containAllVariables(includedVariables) &&
            !prevSubgraph.containAllVariables(includedVariables)) {
            newMatchedExpressions.push_back(expression);
        }
    }
    return newMatchedExpressions;
}

expression_vector getNewMatchedExpressions(const SubqueryGraph& prevLeftSubgraph,
    const SubqueryGraph& prevRightSubgraph, const SubqueryGraph& newSubgraph,
    const expression_vector& expressions) {
    expression_vector newMatchedExpressions;
    for (auto& expression : expressions) {
        auto includedVariables = expression->getDependentVariableNames();
        if (newSubgraph.containAllVariables(includedVariables) &&
            !prevLeftSubgraph.containAllVariables(includedVariables) &&
            !prevRightSubgraph.containAllVariables(includedVariables)) {
            newMatchedExpressions.push_back(expression);
        }
    }
    return newMatchedExpressions;
}

shared_ptr<RelExpression> rewriteQueryRel(const RelExpression& queryRel, bool isRewriteDst) {
    auto& nodeToRewrite = isRewriteDst ? *queryRel.getDstNode() : *queryRel.getSrcNode();
    auto tmpNode = make_shared<NodeExpression>(
        nodeToRewrite.getUniqueName() + "_" + queryRel.getUniqueName(), nodeToRewrite.getLabel());
    return make_shared<RelExpression>(queryRel.getUniqueName(), queryRel.getLabel(),
        isRewriteDst ? queryRel.getSrcNode() : tmpNode,
        isRewriteDst ? tmpNode : queryRel.getDstNode(), queryRel.getLowerBound(),
        queryRel.getUpperBound());
}

shared_ptr<Expression> createNodeIDComparison(const shared_ptr<Expression>& left,
    const shared_ptr<Expression>& right, BuiltInVectorOperations* builtInFunctions) {
    expression_vector children;
    children.push_back(left);
    children.push_back(right);
    vector<DataType> childrenTypes;
    childrenTypes.push_back(left->dataType);
    childrenTypes.push_back(right->dataType);
    auto function = builtInFunctions->matchFunction(EQUALS_FUNC_NAME, childrenTypes);
    auto uniqueName = ScalarFunctionExpression::getUniqueName(EQUALS_FUNC_NAME, children);
    return make_shared<ScalarFunctionExpression>(EQUALS, DataType(BOOL), move(children),
        function->execFunc, function->selectFunc, uniqueName);
}

} // namespace planner
} // namespace graphflow
