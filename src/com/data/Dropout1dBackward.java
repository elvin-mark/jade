package com.data;

public class Dropout1dBackward extends Node {
    float p;

    public Dropout1dBackward(Tensor t1, Tensor result, float p) {
        super();
        this.add_child(t1.node);
        this.tensor = result;
        this.p = p;
    }

    public void backward(Tensor loss) {
        Tensor t1 = this.children.get(0).tensor;
        Tensor new_loss = new Tensor(t1.shape);
        for (int i = 0; i < new_loss.size; i++) {
            if (this.tensor.data[i] != 0) {
                new_loss.data[i] = loss.data[i] / p;
            }
        }
        t1.node.backward(new_loss);
    }
}
