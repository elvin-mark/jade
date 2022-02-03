package com.nn;

import com.data.Tensor;
import com.functions.F;

public class MSELoss extends Loss {
    public MSELoss() {
        super();
    }

    public Tensor criterion(Tensor input, Tensor target) {
        return F.mse_loss(input, target);
    }
}
