//
// Created by Markus on 06.06.17.
//

#ifndef STERMPARSER_ANNOTATIONPROJECTION_H
#define STERMPARSER_ANNOTATIONPROJECTION_H

#include "TrainingCommon.h"
#include "LatentAnnotation.h"
#include "SplitMergeTrainer.h"

namespace Trainer {

    static const double IO_PRECISION_DEFAULT = 0.000001;
    static const unsigned int IO_CYCLE_LIMIT_DEFAULT = 200;


    template<typename Nonterminal>
    HypergraphPtr<Nonterminal> hypergraph_from_grammar(const GrammarInfo2& grammarInfo) {
        // build HG
        MAPTYPE<size_t, bool> nodes;
        std::vector<size_t> nLabels{};

        size_t labelCounter = 0;
        std::vector<size_t> eLabels{};


        for (const auto rule : grammarInfo.rule_to_nonterminals) {
            eLabels.push_back(labelCounter++);
            for (size_t nont : rule) {
                if (nodes.count(nont) == 0) {
                    nodes[nont] = true;
                    nLabels.push_back(nont);
                }
            }
        }

        const auto nLabelsPtr = std::make_shared<const std::vector<size_t>>(nLabels);
        const auto eLabelsPtr = std::make_shared<const std::vector<size_t>>(eLabels);

        HypergraphPtr<Nonterminal> hg = std::make_shared<Hypergraph<Nonterminal>>(nLabelsPtr, eLabelsPtr);

        MAPTYPE<size_t, Element<Node<size_t>>> nodeElements;
        for (size_t ruleNr = 0; ruleNr < grammarInfo.rule_to_nonterminals.size(); ++ruleNr) {
            std::vector<size_t> rule = grammarInfo.rule_to_nonterminals[ruleNr];
            std::vector<Element<Node<Nonterminal>>> rhs{};
            for (size_t i = 0; i < rule.size(); ++i) {
                size_t nont = rule[i];
                if (nodeElements.count(nont) == 0) {
                    nodeElements.insert(MAPTYPE<size_t, Element<Node<size_t>>>::value_type(nont, hg->create(nont)));
                }
                if (i != 0)
                    rhs.push_back(nodeElements.at(nont));
            }
            hg->add_hyperedge(ruleNr, nodeElements.at(rule[0]), rhs);
        }

        return hg;
    }


    template<typename Nonterminal>
    void io_weights_for_grammar(
            const HypergraphPtr<Nonterminal>& hg
            , const LatentAnnotation &annotation
            , const Element<Node<Nonterminal>>& initialNode
            , MAPTYPE <Element<Node<Nonterminal>>, Trainer::WeightVector> &insideWeights
            , MAPTYPE <Element<Node<Nonterminal>>, Trainer::WeightVector> &outsideWeights
            , const double ioPrecision = IO_PRECISION_DEFAULT
            , const unsigned int ioCycleLimit = IO_CYCLE_LIMIT_DEFAULT
            , const bool debug = false
    ) {

        std::shared_ptr<Trainer::TraceManager2<Nonterminal, size_t>> tMPtr = std::make_shared<Trainer::TraceManager2<
                Nonterminal
                , size_t>>(hg->get_node_labels(), hg->get_edge_labels());
        tMPtr->create(1, hg, initialNode);

        for (const auto n : *hg) {
            insideWeights[n] = Trainer::WeightVector(annotation.nonterminalSplits.at(n->get_label_id()));
            outsideWeights[n] = Trainer::WeightVector(annotation.nonterminalSplits.at(n->get_label_id()));
        }

        tMPtr->set_io_precision(ioPrecision);
        tMPtr->set_io_cycle_limit(ioCycleLimit);
        (*tMPtr)[0].io_weights_fixpoint_la(
                *(annotation.ruleWeights)
                , annotation.rootWeights
                , insideWeights
                , outsideWeights
                , false
                , debug
        );

    }


    template<typename Nonterminal>
    LatentAnnotation project_annotation(
            const LatentAnnotation &annotation
            , const GrammarInfo2 &grammarInfo
            , const double ioPrecision = IO_PRECISION_DEFAULT
            , const size_t ioCycleLimit = IO_CYCLE_LIMIT_DEFAULT
            , const bool debug = false
    ) {

        HypergraphPtr<Nonterminal> hg = hypergraph_from_grammar<Nonterminal>(grammarInfo);

        MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> insideWeights;
        MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> outsideWeights;

        io_weights_for_grammar<Nonterminal>(
                hg
                , annotation
                , hg->get_node_by_label(grammarInfo.start)
                , insideWeights
                , outsideWeights
                , ioPrecision
                , ioCycleLimit
                , debug
        );



        auto projRuleWeights = std::vector<RuleTensor<double>>();
        projRuleWeights.reserve(hg->get_edges().lock()->size());
        SizeOneTensorCreator odc(0.0);
        for(size_t ruleId = 0; ruleId < (*annotation.ruleWeights).size(); ++ruleId)
            projRuleWeights.push_back(boost::apply_visitor(odc, (*annotation.ruleWeights)[ruleId]));



        // do projection
        for (const auto &edge : *hg->get_edges().lock()) {
            size_t ruleId = edge->get_label();
            const std::vector<size_t> &rule = grammarInfo.rule_to_nonterminals[ruleId];
            const auto &ruleVariant = (*(annotation.ruleWeights))[ruleId];

            const auto normalisationCalc = insideWeights[edge->get_target()].contract(
                    outsideWeights[edge->get_target()]
                    , Eigen::array<Eigen::IndexPair<long>, 1>{Eigen::IndexPair<long>(0, 0)}
            );
            Eigen::Tensor<double, 0> normalisationVector = normalisationCalc;

            if (debug) {
                std::cerr << "rule " << ruleId << " normalizationVector " << normalisationVector << std::endl;
                std::cerr << "inside weight " << insideWeights[edge->get_target()] << std::endl;
                std::cerr << "outside weight " << outsideWeights[edge->get_target()] << std::endl;
                std::cerr << "rule weight before projection " << ruleVariant << std::endl;
            }

            if (std::abs(normalisationVector(0)) < std::exp(-50)) { // normalization is 0, apply a default value
                size_t norm = grammarInfo.normalizationGroups[rule[0]].size();
                SizeOneTensorCreator nvc(1.0/(double)norm);

                projRuleWeights[ruleId]  = boost::apply_visitor(nvc, ruleVariant);

            } else {
                InsideOutsideMultiplierAndNormalizer<Nonterminal> ioMultiplier(
                        edge
                        , insideWeights
                        , outsideWeights
                        , normalisationVector(0));
                projRuleWeights[ruleId] = boost::apply_visitor(ioMultiplier, ruleVariant);
            }

            if (debug) std::cerr << "rule weight after projection " << projRuleWeights[ruleId] << std::endl;
        }

        // calculate root weights
        WeightVector root(1);
        Eigen::Tensor<double, 0> rootval = annotation.rootWeights.sum();
        root.setValues({rootval(0)});


        LatentAnnotation result(
                std::vector<size_t>(annotation.nonterminalSplits.size(), 1)
                , std::move(root)
                , std::make_unique<std::vector<RuleTensor<double>>>(std::move(projRuleWeights)));


        // make the LA proper
        // (this is needed, since inside or outside values of some nonterminals might be 0)
        result.make_proper(grammarInfo);

        return result;
    }


    /**
     * Compute the weight of an edge in the hypergraph according according to a latent annotation.
     *
     * Precisely, for each edge e = (v -> σ(v_1, …, v_n)) of the hypergraph,
     * we compute
     *   q(e) / (out(v) * in(v))      if variational
     *   q(e) / in(root)              otherwise
     * where q is computed according to the formula in [Petrov and Klein (2007), Figure 3]
     * cf. http://aclweb.org/anthology/N07-1051
     *
     * A vector is returned that contains the projected weights in the order of edges in the hypergraph.
     */
    template<typename Nonterminal, typename TraceID>
    std::vector<double> edge_weight_projection(
        const LatentAnnotation &annotation
        , const GrammarInfo2 &grammarInfo
        , const Trace<Nonterminal, TraceID>& trace
        , const bool variational=false
    ) {
        HypergraphPtr<Nonterminal> hg {trace.get_hypergraph()};

        MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> insideWeights;
        MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> outsideWeights;

        for (const auto n : *hg) {
            insideWeights[n] = Trainer::WeightVector(annotation.nonterminalSplits.at(n->get_label_id()));
            outsideWeights[n] = Trainer::WeightVector(annotation.nonterminalSplits.at(n->get_label_id()));
        }

        trace.io_weights_la(*annotation.ruleWeights, annotation.rootWeights, insideWeights, outsideWeights);


        std::vector<double> projections;
        projections.reserve(hg->get_edges().lock()->size());

        const auto &rootInsideWeight = insideWeights.at(trace.get_goal());
        const auto &rootOutsideWeight = outsideWeights.at(trace.get_goal());
        Eigen::Tensor<double, 0> root_weight{(rootOutsideWeight * rootInsideWeight).sum()};

        double normalizer{root_weight(0)};

        // do projection
        for (const auto &edge : *hg->get_edges().lock()) {
            size_t ruleId = edge->get_label();
            const std::vector<size_t> &rule = grammarInfo.rule_to_nonterminals[ruleId];
            const auto &ruleVariant = (*(annotation.ruleWeights))[ruleId];

            if (variational) {
                const auto normalisationCalc = insideWeights[edge->get_target()].contract(
                        outsideWeights[edge->get_target()]
                        , Eigen::array<Eigen::IndexPair<long>, 1>{Eigen::IndexPair<long>(0, 0)}
                );
                Eigen::Tensor<double, 0> normalisationVector = normalisationCalc;
                normalizer = normalisationVector(0);
            }

            InsideOutsideMultiplierAndNormalizer<Nonterminal> ioMultiplier(
                        edge
                        , insideWeights
                        , outsideWeights
                        , normalizer);

            auto tensor {boost::apply_visitor(ioMultiplier, ruleVariant)};
            SizeOneTensorAccessor sota;
            double projected_weight {boost::apply_visitor(sota, tensor)};

            projections.push_back(projected_weight);
        }

        return projections;
    }


    /**
     * Merges according to externally provided Merge lists.
     * Merge weights are computed w.r.t. exected frequency of latently annotated nonterminals in the grammar.
     */
    template<typename Nonterminal>
    class MergeListMergePreparator {
    private:
        const GrammarInfo2& grammarInfo;
        bool debug;
        const double ioPrecision;
        const unsigned int ioCycleLimit;
    public:
        MergeListMergePreparator(const GrammarInfo2& grammarInfo
                                 , bool debug = false
                                 , const double ioPrecision=IO_PRECISION_DEFAULT
                                 , const unsigned ioCycleLimit=IO_CYCLE_LIMIT_DEFAULT)
                : grammarInfo(grammarInfo), debug(debug), ioPrecision(ioPrecision), ioCycleLimit(ioCycleLimit) {};

        MergeInfo merge_prepare(const LatentAnnotation &latentAnnotation,
                const std::vector<std::vector<std::vector<size_t>>> mergeSources
        ) {
            std::vector<std::vector<double>> mergeFactors;
            std::vector<size_t> new_splits;
            new_splits.reserve(mergeSources.size());

            MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> insideWeights;
            MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> outsideWeights;
            HypergraphPtr<Nonterminal> hg = hypergraph_from_grammar<Nonterminal>(grammarInfo);
            io_weights_for_grammar(hg
                                   , latentAnnotation
                                   , hg->get_node_by_label(grammarInfo.start)
                                   , insideWeights
                                   , outsideWeights
                                   , ioPrecision
                                   , ioCycleLimit
                                   , debug);

            size_t nont_id {0};
            for (auto sourceLists : mergeSources) {
                new_splits.push_back(sourceLists.size());
                std::vector<double> factors(latentAnnotation.nonterminalSplits[nont_id], 1.0);
                size_t count {0};

                auto inside = insideWeights.at(hg->get_node_by_label(nont_id));
                auto outside = outsideWeights.at(hg->get_node_by_label(nont_id));
                Eigen::Tensor<double, 1> io_prod = inside * outside;

                if (debug)
                    std::cerr << nont_id << " inside {" << inside << "} outside {" << outside << "} product {"
                              << io_prod << "}" << std::endl;

                for (auto sourceList : sourceLists){
                    count += sourceList.size();
                    if (sourceList.size() == 1)
                        continue;
                    if (sourceList.size() == 0) {
                        std::cerr << "Empty merge source lists are not permitted. nont_id: " << nont_id << std::endl;
                        abort();
                    }

                    Eigen::Tensor<double, 0> max = io_prod.maximum();
                    size_t safe_count = 3;
                    while (max(0) < exp(-100) and safe_count > 0) {
                        io_prod = io_prod * exp(100);
                        max = io_prod.maximum();
                        safe_count--;
                    }
                    Eigen::Tensor<double, 0> io_sum = io_prod.sum();
                    if (std::isnan(io_sum(0)) or std::isinf(io_sum(0))) {
                        for (size_t source : sourceList)
                            factors[source] = 1.0 / sourceList.size();
                    } else {
                        bool invalid = false;
                        for (size_t source : sourceList) {
                            factors[source] = io_prod(source) / io_sum(0);
                            if (std::isnan(factors[source]) or std::isinf(factors[source])) {
                                invalid = true;
                                break;
                            }
                        }
                        if (invalid)
                            for (size_t source : sourceList)
                                factors[source] = 1.0 / sourceList.size();
                    }
                }
                if (count != latentAnnotation.nonterminalSplits[nont_id]) {
                    std::cerr << "Non-exhaustive merge sources: nont_id: " << nont_id
                              << ", #la: " << latentAnnotation.nonterminalSplits[nont_id]
                              << " but counted: " << count << std::endl;
                    abort();
                }
                mergeFactors.push_back(factors);
                ++nont_id;
            }

            return MergeInfo(mergeSources, new_splits, mergeFactors);
        }
    };

    template<typename Nonterminal>
    LatentAnnotation project_annotation_by_merging(
            const LatentAnnotation &annotation
            , const GrammarInfo2 &grammarInfo
            , const std::vector<std::vector<std::vector<size_t>>> & merge_sources
            , const double ioPrecision = IO_PRECISION_DEFAULT
            , const unsigned int ioCycleLimit = IO_CYCLE_LIMIT_DEFAULT
            , const bool debug = false
    ) {
        MergeListMergePreparator<Nonterminal> mergePreparator(grammarInfo, debug, ioPrecision, ioCycleLimit);
        MergeInfo mi = mergePreparator.merge_prepare(annotation, merge_sources);
        if (debug) std::cerr << mi << std::endl;
        Merger merger(grammarInfo, std::make_shared<StorageManager>());
        return merger.merge(annotation, mi);
    }
}


#endif //STERMPARSER_ANNOTATIONPROJECTION_H
