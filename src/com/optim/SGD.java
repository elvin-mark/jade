package com.optim;

import com.data.Tensor;
import java.util.*;

public class SGD extends Optimizer {
    public SGD(Tensor[] params, Map<String, Float> hyperParameters) {
        super(params, hyperParameters);
    }

    public void step() {
        Tensor lr = new Tensor(this.hyperParameters.get("lr"));
        for (Tensor param : params) {
            if (param.requires_grad_) {
                Tensor grad = param.grad;
                param.data = param.sub(grad.mul(lr)).data;
            }
        }
    }
}
