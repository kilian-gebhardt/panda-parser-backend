//
// Created by jamb on 28.02.17.
//

#ifndef STERMPARSER_STORAGEMANAGER_H
#define STERMPARSER_STORAGEMANAGER_H

#include <iostream>
#include <cassert>


class StorageManager {
private:
    bool self_malloc;
    double * start = nullptr;
    double * next = nullptr;
    double * max_mem = nullptr;
    unsigned the_size = 0; // 625000; // 5MB


public:
    StorageManager(bool selfmal = false): self_malloc(selfmal) {}


    bool reserve_memory(unsigned size) {
        if(!self_malloc)
            return true;

        std::cerr << "reserving " << size << std::endl;
        if (start == next) {
            if (start != nullptr and max_mem - start < size) {
                free(start);
            }
            if (start == nullptr or max_mem - start < size) {
                unsigned allocate = the_size > size ? the_size : size;
                std::cerr << "allocating " << allocate << std::endl;
                start = (double *) malloc(sizeof(double) * allocate);
                max_mem = start + allocate;
                next = start;
            }
            return true;
        } else
            return false;
    }

    double * get_region(unsigned size) {
        if (not self_malloc)
            return (double*) malloc(sizeof(double) * size);
        else {
            if (start == nullptr) {
                start = (double *) malloc(sizeof(double) * the_size);
                max_mem = start + the_size;
                next = start;
            }
            if (max_mem - next < size) {
                std::cerr << "Maximum size of double storage exceeded" << std::endl;
                std::cerr << "Required  " << size << std::endl;
                std::cerr << "Available " << (max_mem - next) / sizeof(double) << std::endl;
                abort();
            }
            double *return_ = next;
            next = return_ + size;
            return return_;
        }
    }

    bool free_region(double* const ptr, const unsigned size) {
        if (not self_malloc) {
            free(ptr);
            return true;
        } else {
            if (ptr + size == next) {
                assert(start <= ptr);
                next = ptr;
                return true;
            }
            return false;
        }
    }

};





#endif //STERMPARSER_STORAGEMANAGER_H
