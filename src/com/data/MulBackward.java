package com.data;

public class MulBackward extends Node {
    public MulBackward(Tensor t1, Tensor t2, Tensor result) {
        super();
        this.add_child(t1.node);
        this.add_child(t2.node);
        this.tensor = result;
    }

    public void backward(Tensor loss) {
        // implement this
    }
}
