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
        // implement this
    }

}
