package com.optim;

import com.data.Tensor;
import java.util.*;

public class SGD extends Optimizer {
    public Tensor[] m = null;
    public Tensor lr;
    public Tensor momentum;

    public SGD(ArrayList<Tensor> params, Map<String, Float> hyperParameters) {
        super(params, hyperParameters);
        if (hyperParameters.containsKey("lr")) {
            lr = new Tensor(hyperParameters.get("lr"));
        } else {
            lr = new Tensor(0.01f);
        }

        if (hyperParameters.containsKey("momentum")) {
            momentum = new Tensor(hyperParameters.get("momentum"));
            m = new Tensor[params.size()];
            for (int i = 0; i < params.size(); i++) {
                m[i] = new Tensor(params.get(i).shape);
                m[i].zeros();
            }
        } else {
            momentum = new Tensor(0.0f);
        }
    }

    public void step() {
        for (int i = 0; i < params.size(); i++) {
            Tensor p = params.get(i);
            if (p.requires_grad_) {
                Tensor tmp = new Tensor(p.grad.shape);
                tmp.data = p.grad.data;
                if (momentum.item() != 0.0f) {
                    this.m[i].data = this.m[i].mul(momentum).add(tmp).data;
                    tmp.data = this.m[i].data;
                }
                p.data = p.sub(tmp.mul(lr)).data;
            }
        }
    }
}
