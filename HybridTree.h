//
// Created by kilian on 18/11/16.
//
#ifndef STERMPARSER_HYBRIDTREE_H
#define STERMPARSER_HYBRIDTREE_H

#include <map>
#include <vector>
#include <iostream>
#include <tuple>
#include "SDCP.h"

template <typename Terminal, typename Position>
class HybridTree {
private:
    std::map<Position, Terminal> tree_label;
    std::map<Position, Terminal> string_label;
    std::map<Position, Position> previous, next, parent;
    std::map<Position, std::vector<Position>> children;
    std::vector<Position> linearization;
    Position entry, exit;
    void terminals_recur(std::vector<std::pair<Position, Terminal>> & terminals, const std::vector<Position> & positions){
        for (auto position : positions){
            if (!is_initial(position))
                terminals.emplace_back(std::pair<Position, Terminal>(position, get_tree_label(position)));
            terminals_recur(terminals, get_children(position));
        }
    };
public:
    const Terminal get_tree_label(const Position position) {
        return tree_label[position];
    }
    const Terminal get_string_label(const Position position) {
        return string_label[position];
    }
    const Position get_next(const Position position)  {
        return next[position];
    }
    const Position get_previous(const Position position)  {
        return previous[position];
    }
    const Position get_parent(const Position position)  {
        return parent[position];
    }
    const Position get_entry() {
        return entry;
    }
    const Position get_exit() {
        return exit;
    }
    const std::vector<Position> & get_children(const Position position)  {
        return children[position];
    }

    void add_node(Position position1, Terminal label, Position position2);

    void add_node(Position position1, Terminal tree_label, Terminal string_label, Position position2);

    void add_child(Position parent, Position child);

    void set_entry(Position position);

    void set_exit(Position position);

    bool is_initial(Position position){
        return previous[position] == -1;
    }

    bool is_final(Position position) {
        return next[position] == -1;
    }

    void output();
    void output_recur(Position position, int indent);

    const std::vector<std::pair<Position, Terminal>> terminals() {
        std::vector<std::pair<Position, Terminal>> terminals;
        Position position = get_entry();
        while (position != -1) {
            if (!is_initial(position))
                terminals.emplace_back(std::pair<Position, Terminal>(position, get_tree_label(position)));
            terminals_recur(terminals, get_children(position));
            position = get_next(position);
        }
        return terminals;
    };

    void set_linearization(std::vector<Position> linearization) {
        this->linearization = linearization;
    }

    const std::vector<Position> & get_linearization() {
        return linearization;
    }

};

template <typename Terminal, typename Position>
void HybridTree<Terminal, Position>::add_child(Position parent, Position child) {
    this->parent[child] = parent;
    this->children[parent].push_back(child);
}

template <typename Terminal, typename Position>
void HybridTree<Terminal, Position>::add_node(Position position1, Terminal label, Position position2) {
    add_node(position1, label, label, position2);
}

template <typename Terminal, typename Position>
void HybridTree<Terminal, Position>::add_node(Position position1, Terminal tree_label, Terminal string_label, Position position2) {
    this->children[position2] = std::vector<Position >();
    this->tree_label[position2] = tree_label;
    this->string_label[position2] = string_label;
    this->next[position1] = position2;
    this->previous[position2] = position1;
    try {
        previous.at(position1);
    }
    catch (const std::out_of_range& oor){
        previous[position1] = -1;
    }
    try {
        next.at(position2);
    }
    catch (const std::out_of_range& oor) {
        next[position2] = -1;
    }
}

template <typename Terminal, typename Position>
void HybridTree<Terminal, Position>::set_entry(Position position) {
    this->entry = position;
}

template <typename Terminal, typename Position>
void HybridTree<Terminal, Position>::set_exit(Position position) {
    this->exit = position;
}

template <typename Terminal, typename Position>
void HybridTree<Terminal, Position>::output() {
    output_recur(this->entry, 0);
    if (linearization.size()) {
        int i = 0;
        for (auto pos: linearization) {
            if (i++)
                std::cerr << " ";
            std::cerr << string_label[pos];

        }
        std::cerr << std::endl;
    }
}

template <typename Terminal, typename Position>
void HybridTree<Terminal, Position>::output_recur(Position position, int indent) {
    while (is_initial(position))
        position = get_next(position);
    while (position != - 1) {
        for (int i = 0; i < indent; ++i)
            std::cerr << "  ";
        std::cerr << get_previous(position) << " " << get_tree_label(position) << " " << position << std::endl;
        if (get_children(position).size() > 0) {
            output_recur(get_children(position)[0], indent + 1);
        }
        position = get_next(position);
    }
}

void output_helper(std::string s){
    std::cerr << s << std::endl;
}

#endif //STERMPARSER_HYBRIDTREE_H
