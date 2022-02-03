package com.data;

public class ReluBackward extends Node {
    public ReluBackward(Tensor t1, Tensor result) {
        super();
        this.add_child(t1.node);
        this.tensor = result;
    }

    public void backward(Tensor loss) {
        Tensor t1 = this.children.get(0).tensor;
        Tensor new_loss = new Tensor(t1.shape);

        for (int i = 0; i < new_loss.size; i++) {
            new_loss.data[i] = this.tensor.data[i] > 0 ? 1 : 0;
        }
        t1.node.backward(new_loss);
    }

}
