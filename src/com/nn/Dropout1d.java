package com.nn;

import com.data.Tensor;
import com.functions.F;

public class Dropout1d extends NNModule {
    float p;

    public Dropout1d(float p) {
        super();
        this.p = p;
        this.moduleName = "Dropout1d";
    }

    public Tensor forward(Tensor input) {
        if (this.training) {
            return F.dropout1d(input, p);
        }
        return input;
    }
}
