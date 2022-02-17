package com.nn;

import com.data.Tensor;
import com.functions.F;

public class Dropout2d extends NNModule {
    float p;

    public Dropout2d(float p) {
        super();
        this.p = p;
        this.moduleName = "Dropout2d";
    }

    public Tensor forward(Tensor input) {
        if (this.training) {
            return F.dropout2d(input, p);
        }
        return input;
    }
}
