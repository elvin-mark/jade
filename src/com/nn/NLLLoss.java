package com.nn;

import com.data.Tensor;
import com.functions.F;

public class NLLLoss extends Loss {
    public NLLLoss() {
        super();
    }

    public Tensor criterion(Tensor input, Tensor target) {
        return F.nllloss(input, target);
    }
}
