//
// Created by kilian on 18/11/16.
//
#ifndef STERMPARSER_HYBRIDTREE_H
#define STERMPARSER_HYBRIDTREE_H

#include <map>
#include <vector>
#include <iostream>

// typedef int Terminal;
// typedef int Position;

template <typename Terminal, typename Position>
class HybridTree {
private:
    std::map<Position, Terminal> label;
    std::map<Position, Position> previous, next, parent;
    std::map<Position, std::vector<Position>> children;
    Position entry, exit;
public:
    const Terminal get_label(Position position) {
        return label[position];
    }
    const Position get_next(Position position){
        return next[position];
    }
    const Position get_previous(Position position){
        return previous[position];
    }
    const Position get_parent(Position position){
        return parent[position];
    }
    const Position get_entry(){
        return entry;
    }
    const Position get_exit(){
        return exit;
    }
    const std::vector<Position> & get_children(Position position){
        return children[position];
    }

    void add_node(Position position1, Terminal label, Position position2);

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

};

template <typename Terminal, typename Position>
void HybridTree<Terminal, Position>::add_child(Position parent, Position child) {
    this->parent[child] = parent;
    this->children[parent].push_back(child);
}

template <typename Terminal, typename Position>
void HybridTree<Terminal, Position>::add_node(Position position1, Terminal label, Position position2) {
    this->children[position2] = std::vector<Position >();
    this->label[position2] = label;
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
}

template <typename Terminal, typename Position>
void HybridTree<Terminal, Position>::output_recur(Position position, int indent) {
    while (is_initial(position))
        position = get_next(position);
    while (position != - 1) {
        for (int i = 0; i < indent; ++i)
            std::cout << "  ";
        std::cout << get_label(position) << std::endl;
        if (get_children(position).size() > 0) {
            output_recur(get_children(position)[0], indent + 1);
        }
        position = get_next(position);
    }
}

#endif //STERMPARSER_HYBRIDTREE_H
