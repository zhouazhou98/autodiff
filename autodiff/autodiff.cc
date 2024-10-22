#include "autodiff.h"

namespace autodiff {


void AutoDiff::backward() {
    m_gradient = 1.0;
    if (m_node) m_node->backward(m_gradient);
}



}

