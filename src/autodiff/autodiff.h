#ifndef __ZHOU_AUTODIFF_H__
#define __ZHOU_AUTODIFF_H__

#include "utils/type.h"
#include <memory>

namespace autodiff {

class Node;

class AutoDiff {
public:
    typedef std::shared_ptr<AutoDiff> ptr;

    AutoDiff(float_t value) : m_value(value), m_gradient(0), m_node(nullptr) {}
    AutoDiff(float_t value, std::shared_ptr<Node> node) : m_value(value), m_gradient(0), m_node(node) {}


public:
    void backward();

public:
    float_t m_value;
    float_t m_gradient;
    std::shared_ptr<Node> m_node;

};  // ! AutoDiff class

}   // ! autodiff namespace 


#endif // ! __ZHOU_AUTODIFF_H__
