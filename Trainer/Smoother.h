//
// Created by kilian on 24/03/17.
//

#ifndef STERMPARSER_SMOOTHER_H
#define STERMPARSER_SMOOTHER_H

#include "LatentAnnotation.h"
#include "GrammarInfo.h"
#include <memory>

namespace Trainer {
    struct TensorSmoother : boost::static_visitor<void> {
        const double smoothingFactor;

        explicit TensorSmoother(const double smoothingFactor) : smoothingFactor(smoothingFactor) {}

        template <int rank>
        void operator()(RuleTensorRaw<double, rank> & ruleTensor) {
            const Eigen::Tensor<double, 0> sumt = ruleTensor.sum();
            const double sum = sumt(0);
            const double sf = smoothingFactor;
            ruleTensor = ruleTensor.unaryExpr([sum, sf](double x) -> double {
                return x * (1 - sf) + sum * sf;
            });
        }
    };

    class Smoother {
        std::shared_ptr<const GrammarInfo2> grammarInfo;
        const double smoothingFactor;

    public:
        explicit Smoother(std::shared_ptr<const GrammarInfo2> grammarInfo
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
            TensorSmoother tensorSmoother(smoothingFactor);
            for (RuleTensor<double> & rule : *(latentAnnotation.ruleWeights)) {
                boost::apply_visitor(tensorSmoother, rule);
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
    };
}

#endif //STERMPARSER_SMOOTHER_H
