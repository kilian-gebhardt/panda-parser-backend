//
// Created by kilian on 03/03/17.
//

#ifndef STERMPARSER_GRAMMARINFO_H
#define STERMPARSER_GRAMMARINFO_H
#include <vector>
#include <iostream>

namespace Trainer {

    class GrammarInfo2 {
    public:
        std::vector<std::vector<size_t>> rule_to_nonterminals;
        const std::vector<std::vector<size_t>> normalizationGroups;
        const size_t start;

        GrammarInfo2(
                const std::vector<std::vector<size_t>> rule_to_nonterminals
                , size_t start
        )
                : rule_to_nonterminals(rule_to_nonterminals)
                , normalizationGroups(GrammarInfo2::compute_normalization_groups(rule_to_nonterminals))
                , start(start) {
        }


        bool check_for_consistency(){
            // check that the number of rules matches
            size_t numberOfRules = 0;
            for(auto& group : normalizationGroups){
                numberOfRules += group.size();
            }
            return numberOfRules == rule_to_nonterminals.size();
        }

    private:
        static std::vector<std::vector<size_t>> compute_normalization_groups(
                const std::vector<std::vector<size_t>> & rule_to_nonterminals) {
            std::vector<std::vector<size_t>> normGroups;
            for (size_t rule_idx = 0; rule_idx < rule_to_nonterminals.size(); ++rule_idx) {
                if (rule_to_nonterminals[rule_idx].size() > 0) {
                    if (normGroups.size() <= rule_to_nonterminals[rule_idx][0]) {
                        normGroups.resize(rule_to_nonterminals[rule_idx][0] + 1);
                    }
                    normGroups[rule_to_nonterminals[rule_idx][0]].push_back(rule_idx);
                } else
                    std::cerr << "[Error] Rule " << rule_idx << " has 0 Nonterminals";
            }
            return normGroups;
        }
    };

    std::ostream &operator<<(std::ostream &os, const GrammarInfo2 &grammarInfo) {
        os << "Start symbol " << grammarInfo.start << std::endl;

        os << "Rule to nonterminals: " << std::endl;

        // Debug Output:
        unsigned i = 0;
        for (auto rtn : grammarInfo.rule_to_nonterminals) {
            os << i << ": ";
            unsigned j = 0;
            for (auto n : rtn) {
                if (j == 1) {
                    os << "-> ";
                }
                os << n << " ";
                ++j;
            }
            os << ";" << std::endl;
            ++i;
        }

        os << "normalization groups " << std::endl;
        for (unsigned i = 0; i < grammarInfo.normalizationGroups.size(); ++i) {
            os << i << " : { ";
            for (auto n : grammarInfo.normalizationGroups[i]) {
                os << n << " ";
            }
            os << "} ";
        }
        os << std::endl;

        return os;
    }
}

#endif //STERMPARSER_GRAMMARINFO_H
