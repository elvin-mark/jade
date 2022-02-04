package com.nn;

import com.data.Tensor;
import com.functions.F;

public class LogSoftmax extends NNModule {
    public LogSoftmax() {
        super();
        this.moduleName = "LogSoftmax";
    }

    public Tensor forward(Tensor input) {
        return F.logsoftmax(input);
    }
}
