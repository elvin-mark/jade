package com.data;

public class MeanBackward extends Node {
    public MeanBackward(Tensor t1, Tensor result) {
        super();
        this.add_child(t1.node);
        this.tensor = result;
    }

    public void backward(Tensor loss) {
        Tensor t1 = this.children.get(0).tensor;
        Tensor new_loss = new Tensor(t1.shape);
        if (loss.size == 1) {
            for (int i = 0; i < new_loss.size; i++) {
                new_loss.data[i] = loss.item() / t1.size;
            }
        } else {
            new_loss.ones();
            float scale = 1.0f / (t1.size / loss.size);
            new_loss = new_loss.mul(loss.mul(new Tensor(scale)));
        }

        t1.node.backward(new_loss);
    }
}
