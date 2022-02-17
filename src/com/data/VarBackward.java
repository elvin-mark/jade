package com.data;

public class VarBackward extends Node {
    boolean bias;

    public VarBackward(Tensor t1, Tensor t2, boolean bias, Tensor result) {
        super();
        this.add_child(t1.node);
        this.add_child(t2.node);
        this.tensor = result;
        this.bias = bias;
    }

    public void backward(Tensor loss) {
        Tensor t1 = this.children.get(0).tensor;
        Tensor t2 = this.children.get(1).tensor;

        Tensor new_loss = null;

        if (loss.size == 1) {
            new_loss = new Tensor(t1.shape);
            if (this.bias) {
                for (int i = 0; i < new_loss.size; i++) {
                    new_loss.data[i] = 2 * (t1.data[i] - t2.data[0]) * loss.item() / (t1.size);
                }
            } else {
                for (int i = 0; i < new_loss.size; i++) {
                    new_loss.data[i] = 2 * (t1.data[i] - t2.data[0]) * loss.item() / (t1.size - 1);
                }
            }

        } else {
            float scale;
            if (this.bias) {
                scale = 2.0f / (t1.size / loss.size);
            } else {
                scale = 2.0f / (t1.size / loss.size - 1.0f);
            }
            new_loss = t1.sub(t2).mul(loss).mul(new Tensor(scale));
        }

        t1.node.backward(new_loss);
    }
}
