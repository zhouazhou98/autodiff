#include "node.h"

namespace autodiff {

AutoDiff::ptr operator+(AutoDiff::ptr x1, AutoDiff::ptr x2) {
    return std::make_shared<AutoDiff>(x1->m_value + x2->m_value, std::make_shared<AddNode>(x1, x2));
}



AutoDiff::ptr operator*(AutoDiff::ptr x1, AutoDiff::ptr x2) {
    return std::make_shared<AutoDiff>(x1->m_value * x2->m_value, std::make_shared<MulNode>(x1, x2));
}

}
