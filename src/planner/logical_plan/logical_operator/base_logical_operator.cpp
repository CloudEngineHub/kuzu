#include "include/base_logical_operator.h"

namespace graphflow {
namespace planner {

LogicalOperator::LogicalOperator(shared_ptr<LogicalOperator> child) {
    children.push_back(move(child));
}

LogicalOperator::LogicalOperator(
    shared_ptr<LogicalOperator> left, shared_ptr<LogicalOperator> right) {
    children.push_back(move(left));
    children.push_back(move(right));
}

bool LogicalOperator::descendantsContainType(
    const unordered_set<LogicalOperatorType>& types) const {
    if (types.contains(getLogicalOperatorType())) {
        return true;
    }
    for (auto& child : children) {
        if (child->descendantsContainType(types)) {
            return true;
        }
    }
    return false;
}

string LogicalOperator::toString(uint64_t depth) const {
    auto padding = string(depth * 4, ' ');
    string result = padding;
    result += LogicalOperatorTypeNames[getLogicalOperatorType()] + "[" +
              getExpressionsForPrinting() + "]";
    if (children.size() == 1) {
        result += "\n" + children[0]->toString(depth);
    } else if (children.size() == 2) {
        result += "\n" + padding + "LEFT:\n" + children[0]->toString(depth + 1);
        result += "\n" + padding + "RIGHT:\n" + children[1]->toString(depth + 1);
    } else {
        for (auto& child : children) {
            result += "\n" + padding + "CHILD:\n" + child->toString(depth + 1);
        }
    }
    return result;
}

} // namespace planner
} // namespace graphflow
