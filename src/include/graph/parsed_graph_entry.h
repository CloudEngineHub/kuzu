#pragma once

#include <string>
#include <vector>

namespace kuzu {
namespace graph {

enum class GraphEntryType : uint8_t {
    NATIVE = 0,
    CYPHER = 1,
};

struct ParsedGraphEntry {
    GraphEntryType type;

    explicit ParsedGraphEntry(GraphEntryType type) : type{type} {}
    virtual ~ParsedGraphEntry() = default;
};

struct ParsedNativeGraphTableInfo {
    std::string tableName;
    std::string predicate;

    ParsedNativeGraphTableInfo(std::string tableName, std::string predicate)
        : tableName{std::move(tableName)}, predicate{std::move(predicate)} {}

    std::string toString() const;
};

struct ParsedNativeGraphEntry : ParsedGraphEntry  {
    std::vector<ParsedNativeGraphTableInfo> nodeInfos;
    std::vector<ParsedNativeGraphTableInfo> relInfos;

    ParsedNativeGraphEntry(std::vector<ParsedNativeGraphTableInfo> nodeInfos, std::vector<ParsedNativeGraphTableInfo> relInfos)
        : ParsedGraphEntry{GraphEntryType::NATIVE}, nodeInfos{std::move(nodeInfos)}, relInfos{std::move(relInfos)} {}
};

struct ParsedCypherGraphEntry : ParsedGraphEntry {
    std::string cypherQuery;

    explicit ParsedCypherGraphEntry(std::string cypherQuery) : ParsedGraphEntry{GraphEntryType::CYPHER}, cypherQuery{std::move(cypherQuery)} {}
};

}
}
