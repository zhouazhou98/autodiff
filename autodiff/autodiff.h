#ifndef __ZHOU_AUTODIFF_H__
#define __ZHOU_AUTODIFF_H__

#include <memory>

namespace autodiff {

typedef double float_t;

class Node {
public:
    typedef std::shared_ptr<Node> ptr;

    virtual ~Node() {}
    virtual void backward(float_t gradient) = 0;
};

class AutoDiff {
public:
    std::shared_ptr<AutoDiff> ptr;

    AutoDiff(float_t value) : m_value(value), m_gradient(0), m_node(nullptr) {}
    AutoDiff(float_t value, Node::ptr node) : m_value(value), m_gradient(0), m_node(node) {}


public:
    void backward();

private:
    float_t m_value;
    float_t m_gradient;
    Node::ptr m_node;
};

}


#endif // ! __ZHOU_AUTODIFF_H__


