#ifndef __ZHOU_NODE_H__
#define __ZHOU_NODE_H__

#include "utils/type.h"
#include "autodiff.h"
#include <memory>
#include <iostream>

namespace autodiff {

class Node {
public:
    typedef std::shared_ptr<Node> ptr;

    virtual ~Node() {}
    virtual void backward(float_t gradient) = 0;
};

// --------------------- AddNode ---------------------

class AddNode : public Node {
public:
    AddNode(AutoDiff::ptr x1, AutoDiff::ptr x2) : m_x1(x1), m_x2(x2) {}

    void backward(float_t gradient) override {
        m_x1->m_gradient += gradient;
        m_x2->m_gradient += gradient;
        if (m_x1->m_node) m_x1->m_node->backward(gradient);
        if (m_x2->m_node) m_x2->m_node->backward(gradient);
    }

private:
    AutoDiff::ptr m_x1;
    AutoDiff::ptr m_x2;
};

AutoDiff::ptr operator+(AutoDiff::ptr x1, AutoDiff::ptr x2);


// --------------------- MulNode ---------------------

class MulNode : public Node {
public:
    MulNode(AutoDiff::ptr x1, AutoDiff::ptr x2) : m_x1(x1), m_x2(x2) {}

    void backward(float_t gradient) override {
        std::cout << "MulNode::backward(" << gradient << ")" << std::endl;
        m_x1->m_gradient += gradient * m_x2->m_value;
        m_x2->m_gradient += gradient * m_x1->m_value;
        std::cout << "\tgradient of x1: " << m_x1->m_gradient << ", gradient of x2: " << m_x2->m_gradient << std::endl;
        if (m_x1->m_node) m_x1->m_node->backward(gradient * m_x2->m_value);
        if (m_x2->m_node) m_x2->m_node->backward(gradient * m_x1->m_value);
    }

private:
    AutoDiff::ptr m_x1;
    AutoDiff::ptr m_x2;
};

AutoDiff::ptr operator*(AutoDiff::ptr x1, AutoDiff::ptr x2);


}   // ! autodiff namespace

#endif // ! __ZHOU_NODE_H__