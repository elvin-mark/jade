package com.nn;

import com.data.Tensor;
import com.functions.F;

public class LeakyReLU extends NNModule {
    float alpha;

    public LeakyReLU(float alpha) {
        super();
        this.alpha = alpha;
        this.moduleName = "LeakyReLU";
    }

    public Tensor forward(Tensor input) {
        return F.leaky_relu(input, this.alpha);
    }

}
