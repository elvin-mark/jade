package com.optim;

import com.data.Tensor;
import java.util.*;

public class Adam extends Optimizer {
    public Tensor[] m = null;
    public Tensor[] v = null;
    public Tensor lr;
    public Tensor momentum;
    public Tensor beta1;
    public Tensor beta2;
    public Tensor eps;
    public Tensor beta1t = new Tensor(1.0f);
    public Tensor beta2t = new Tensor(1.0f);

    public Adam(ArrayList<Tensor> params, Map<String, Float> hyperParameters) {
        super(params, hyperParameters);

        if (hyperParameters.containsKey("lr")) {
            lr = new Tensor(hyperParameters.get("lr"));
        } else {
            lr = new Tensor(0.001f);
        }

        if (hyperParameters.containsKey("momentum")) {
            momentum = new Tensor(hyperParameters.get("momentum"));
        } else {
            momentum = new Tensor(0.0f);
        }

        m = new Tensor[params.size()];
        for (int i = 0; i < params.size(); i++) {
            m[i] = new Tensor(params.get(i).shape);
            m[i].zeros();
        }

        v = new Tensor[params.size()];
        for (int i = 0; i < params.size(); i++) {
            v[i] = new Tensor(params.get(i).shape);
            v[i].zeros();
        }

        if (hyperParameters.containsKey("beta1")) {
            beta1 = new Tensor(hyperParameters.get("beta1"));
        } else {
            beta1 = new Tensor(0.9f);
        }

        if (hyperParameters.containsKey("beta2")) {
            beta2 = new Tensor(hyperParameters.get("beta2"));
        } else {
            beta2 = new Tensor(0.999f);
        }

        if (hyperParameters.containsKey("eps")) {
            eps = new Tensor(hyperParameters.get("eps"));
        } else {
            eps = new Tensor(1e-8f);
        }
    }

    public void step() {
        for (int i = 0; i < params.size(); i++) {
            Tensor p = params.get(i);
            if (p.requires_grad_) {
                this.m[i].data = this.m[i].mul(beta1).add(p.grad.mul(new Tensor(1.0f - beta1.item()))).data;
                this.v[i].data = this.v[i].mul(beta2).add(p.grad.pow(2).mul(new Tensor(1.0f - beta2.item()))).data;
                beta1t.data[0] = beta1t.data[0] * beta1.data[0];
                beta2t.data[0] = beta2t.data[0] * beta2.data[0];
                Tensor m_hat = this.m[i].div(new Tensor(1.0f - beta1t.item()));
                Tensor v_hat = this.v[i].div(new Tensor(1.0f - beta2t.item()));
                p.data = p.sub(m_hat.div(v_hat.pow(0.5f).add(eps)).mul(lr)).data;
            }
        }
    }
}
