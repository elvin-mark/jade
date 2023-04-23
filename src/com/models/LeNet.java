package com.models;

import com.nn.*;

public class LeNet extends NNModule {
    public LeNet() {
        super();
        this.add_module((NNModule) new Conv2d(1, 6, 5, true));
        this.add_module((NNModule) new Tanh());
        this.add_module((NNModule) new MaxPool2d(2));
        this.add_module((NNModule) new Conv2d(6, 16, 5, true));
        this.add_module((NNModule) new Tanh());
        this.add_module((NNModule) new MaxPool2d(2));
        this.add_module((NNModule) new Conv2d(16, 120, 5, true));
        this.add_module((NNModule) new Flatten());
        this.add_module((NNModule) new Linear(120, 84, true));
        this.add_module((NNModule) new Tanh());
        this.add_module((NNModule) new Linear(84, 10, true));
    }
}
