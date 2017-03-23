//
// Created by kilian on 24/03/17.
//

#ifndef STERMPARSER_SMOOTHER_H
#define STERMPARSER_SMOOTHER_H

#include "LatentAnnotation.h"
#include "GrammarInfo.h"
#include <memory>

namespace Trainer {
    class Smoother {
        std::shared_ptr<const GrammarInfo2> grammarInfo;
        const double smoothingFactor;

    public:
        Smoother(std::shared_ptr<const GrammarInfo2> grammarInfo
                 , double smoothingFactor = 0.01)
                : grammarInfo(grammarInfo), smoothingFactor(smoothingFactor) {
            if (0.0 > smoothingFactor or 1.0 < smoothingFactor) {
                std::cerr << "A smoothing factor needs to be in the interval [0,1]," << std::endl
                          << "but was set to " << smoothingFactor << "." << std::endl;
                abort();
            }
        }

        void smooth(LatentAnnotation & latentAnnotation) {
            std::cerr << "Smoothing rules with factor " << smoothingFactor << "." << std::endl;
            for (RuleTensor<double> & rule : *(latentAnnotation.ruleWeights)) {
                smooth_rule(rule);
            }
            for (size_t nont = 0; nont < grammarInfo->normalizationGroups.size(); ++nont) {
                auto & group = grammarInfo->normalizationGroups[nont];
                Eigen::Tensor<double, 1> normalizationDivisor(latentAnnotation.nonterminalSplits[nont]);
                normalizationDivisor.setZero();
                for (size_t ruleId : group) {
                    compute_normalization_divisor(normalizationDivisor, (*latentAnnotation.ruleWeights)[ruleId]);
                }
                for (size_t ruleId : group) {
                    RuleTensor<double> & rule = (*latentAnnotation.ruleWeights)[ruleId];
                    normalize(rule, rule, normalizationDivisor);
                }
            }
        }

        double get_smoothing_factor() {
            return smoothingFactor;
        }

    private:

        void smooth_rule(RuleTensor<double> & rule) {
            switch (rule.which() + 1) {
                case 1:
                    smooth_rule_ranked<1>(rule);
                    break;
                case 2:
                    smooth_rule_ranked<2>(rule);
                    break;
                case 3:
                    smooth_rule_ranked<3>(rule);
                    break;
                case 4:
                    smooth_rule_ranked<4>(rule);
                    break;
                default:
                    std::cerr<< "Rule with rank " << rule.which() + 1 << " is not supported." << std::endl;
                    abort();
            }
        }

        template <size_t rank>
        void smooth_rule_ranked(RuleTensor<double> & rule) {
            auto & ruleRaw = boost::get<RuleTensorRaw<double, rank>>(rule);
            const Eigen::Tensor<double, 0> sumt = ruleRaw.sum();
            const double sum = sumt(0);
            const double sf = smoothingFactor;
            ruleRaw = ruleRaw.unaryExpr([sum, sf](double x) -> double {
                return x * (1 - sf) + sum * sf;
            });
        }
    };
}

#endif //STERMPARSER_SMOOTHER_H
