package com.nn;

import com.data.Tensor;
import com.functions.F;

public class Sigmoid extends NNModule {
    public Sigmoid() {
        super();
        this.moduleName = "Sigmoid";
    }

    public Tensor forward(Tensor input) {
        return F.sigmoid(input);
    }
}
