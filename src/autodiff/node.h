#ifndef __ZHOU_NODE_H__
#define __ZHOU_NODE_H__

#include "utils/type.h"
#include <memory>

namespace autodiff {

class Node {
public:
    typedef std::shared_ptr<Node> ptr;

    virtual ~Node() {}
    virtual void backward(float_t gradient) = 0;
};

}   // ! autodiff namespace

#endif // ! __ZHOU_NODE_H__