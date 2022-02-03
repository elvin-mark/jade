package com.nn;

import com.data.Tensor;
import com.functions.F;

public class CrossEntropyLoss extends Loss {
    public CrossEntropyLoss() {
        super();
    }

    public Tensor criterion(Tensor input, Tensor target) {
        return F.cross_entropy_loss(input, target);
    }
}
