package com.data;

public class LeakyReluBackward extends Node {
    float alpha;

    public LeakyReluBackward(Tensor t1, Tensor result, float alpha) {
        super();
        this.add_child(t1.node);
        this.tensor = result;
        this.alpha = alpha;
    }

    public void backward(Tensor loss) {
        Tensor t1 = this.children.get(0).tensor;
        Tensor new_loss = new Tensor(t1.shape);

        for (int i = 0; i < new_loss.size; i++) {
            new_loss.data[i] = this.tensor.data[i] > 0 ? this.alpha : 0;
        }
        t1.node.backward(new_loss);
    }

}
