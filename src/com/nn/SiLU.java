package com.nn;

import com.data.Tensor;
import com.functions.F;

public class SiLU extends NNModule {
    public SiLU() {
        super();
        this.moduleName = "SiLU";
    }

    public Tensor forward(Tensor input) {
        return F.silu(input);
    }
}
