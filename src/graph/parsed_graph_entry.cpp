#include "graph/parsed_graph_entry.h"
#include "common/string_format.h"

using namespace kuzu::common;

namespace kuzu {
namespace graph {

std::string ParsedNativeGraphTableInfo::toString() const {
    std::string result = "{";
    result += stringFormat("'table': '{}'", tableName);
    if (predicate != "") {
        result += stringFormat(",'predicate': '{}'", predicate);
    }
    result += "}";
    return result;
}

}
}
