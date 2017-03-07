//
// Created by kilian on 03/03/17.
//

#ifndef STERMPARSER_GRAMMARINFO_H
#define STERMPARSER_GRAMMARINFO_H
#include <vector>

namespace Trainer {

    class GrammarInfo2 {
    public:
        std::vector<std::vector<size_t>> rule_to_nonterminals;
        std::vector<std::vector<size_t>> normalizationGroups;
        const size_t start;

        GrammarInfo2(
                const std::vector<std::vector<size_t>> rule_to_nonterminals
                , size_t start
        )
                : rule_to_nonterminals(rule_to_nonterminals), start(start) {
            std::vector<std::vector<size_t>> normGroups;
            for (size_t rule_idx = 0; rule_idx < rule_to_nonterminals.size(); ++rule_idx) {
                if (rule_to_nonterminals[rule_idx].size() > 0) {
                    if (normGroups.size() <= rule_to_nonterminals[rule_idx][0]) {
                        normGroups.resize(rule_to_nonterminals[rule_idx][0] + 1);
                    }
                    normGroups[rule_to_nonterminals[rule_idx][0]].push_back(rule_idx);
                }
            }

            normalizationGroups = normGroups;

//        { // Debug Output:
//            unsigned i = 0;
//            for (auto rtn : rule_to_nonterminals) {
//                std::cerr << i << ": ";
//                unsigned j = 0;
//                for (auto n : rtn) {
//                    if (j == 1) {
//                        std::cerr << "-> ";
//                    }
//                    std::cerr << n << " ";
//                    ++j;
//                }
//                std::cerr << ";" << std::endl;
//                ++i;
//            }
//            for (unsigned i = 0; i < normalizationGroups.size(); ++i) {
//                std::cerr << i << " : { ";
//                for (auto n : normalizationGroups[i]) {
//                    std::cerr << n << " ";
//                }
//                std::cerr << "} ";
//            }
//            std::cerr << std::endl;
//        }

        }
    };
}

#endif //STERMPARSER_GRAMMARINFO_H
