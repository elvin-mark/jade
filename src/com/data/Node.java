package com.data;

import java.util.*;

public class Node {
    public ArrayList<Node> children;
    public Tensor tensor;

    public Node() {
        this.children = new ArrayList<Node>();
    }

    public Node(Tensor tensor) {
        this.children = new ArrayList<Node>();
        this.tensor = tensor;
    }

    public void add_child(Node node) {
        this.children.add(node);
    }

    public void backward(Tensor loss) {
        if (is_leaf() && this.tensor.requires_grad_) {
            Tensor new_grad = this.tensor.grad.add(loss);
            this.tensor.grad.data = new_grad.data;
        }
    }

    public boolean is_leaf() {
        return this.children.size() == 0;
    }
}
