package com.nn;

import com.data.*;
import com.functions.*;

public class Linear extends NNModule {
    boolean bias;

    public Linear(int in_features, int out_features, boolean bias) {
        super();
        Tensor W = new Tensor(new Tensor(new int[] { in_features, out_features }));
        W.random(0.0f, 1.0f);
        W.requires_grad(true);
        this.params.add(W);

        this.bias = bias;
        this.moduleName = "Linear";

        if (bias) {
            Tensor b = new Tensor(new int[] { out_features });
            b.random(0.0f, 1.0f);
            b.requires_grad(true);
            this.params.add(b);
        }
    }

    public void init_random() {
        for (Tensor param : this.params) {
            param.random(0.0f, 1.0f);
        }
    }

    public void init_randn() {
        for (Tensor param : this.params) {
            param.randn(0.0f, 1.0f);
        }
    }

    public Tensor forward(Tensor input) {
        Tensor w = this.params.get(0);
        Tensor b, out;
        if (this.bias) {
            b = this.params.get(1);
            out = F.linear(input, w, b);
        } else {
            out = F.linear(input, w);
        }
        return out;
    }
}
