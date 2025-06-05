#pragma once

#include "parsed_graph_entry.h"

namespace kuzu {
namespace graph {

class GraphEntrySet {
public:

    void validateGraphNotExist(const std::string& name) const;
    void validateGraphExist(const std::string& name) const;

    bool hasGraph(const std::string& name) const { return nameToEntry.contains(name); }
    ParsedGraphEntry* getEntry(const std::string& name) const {
        KU_ASSERT(hasGraph(name));
        return nameToEntry.at(name).get();
    }
    void addGraph(const std::string& name, std::unique_ptr<ParsedGraphEntry> entry) {
        nameToEntry.insert({name, std::move(entry)});
    }
    void dropGraph(const std::string& name) { nameToEntry.erase(name); }

    // using iterator = std::unordered_map<std::string, ParsedNativeGraphEntry>::iterator;
    // using const_iterator = std::unordered_map<std::string, ParsedNativeGraphEntry>::const_iterator;
    //
    // iterator begin() { return nameToEntry.begin(); }
    // iterator end() { return nameToEntry.end(); }
    // const_iterator begin() const { return nameToEntry.begin(); }
    // const_iterator end() const { return nameToEntry.end(); }
    // const_iterator cbegin() const { return nameToEntry.cbegin(); }
    // const_iterator cend() const { return nameToEntry.cend(); }

private:
    std::unordered_map<std::string, std::unique_ptr<ParsedGraphEntry>> nameToEntry;
};

}
}
