![jade_logo](https://raw.githubusercontent.com/elvin-mark/jade/main/images/jade_logo.png)
# <span style="color:#00aa22">**Ja**</span>va <span style="color:#00aa22">**De**</span>ep Learning Library
Deep Learning Library for Java inspired on PyTorch. (It is still not optimized). I really love PyTorch and I was fascinated at how it works, so I decided to implement in Java for fun (and to remember a little bit of Java)

## Layers implemented
- Linear (with bias)
- Conv1d [Coming Soon ...]
- Conv2d [Coming Soon ...]
- Sigmoid
- Tanh
- ReLU
- LeakyReLU
- Softmax [Coming Soon ...]

## Loss functions implemented
- Mean Square Error Loss (MSELosss)
- Cross Entropy Loss (CrossEntropyLoss) [Coming Soon ...]
- Negative Log Likelihood Loss (NLLLoss) [Coming Soon ...]

## Optimizers implemented
- SGD
- Adam [Coming Soon ...]

## Example

Simple Linear Regression example.

```Java
import com.nn.*;
import com.optim.*;
import com.data.*;
import java.util.*;

public class SampleLinearRegression {
    public static void main(String args[]) {
        float[] X = new float[] { 1, 2, 3, 4, 5 };
        float[] y = new float[] { 7, 9, 11, 13, 15 };

        Tensor x_train = new Tensor(new int[] { 5, 1 }, X);
        Tensor y_train = new Tensor(new int[] { 5, 1 }, y);

        NNModule seq = new Sequential();
        seq.add_module((NNModule) new Linear(1, 1, true));

        Loss loss_fn = new MSELoss();
        Map<String, Float> optim_params = new HashMap<String, Float>();
        optim_params.put("lr", 0.01f);
        Optimizer optim = new SGD(seq.parameters(), optim_params);

        for (int epoch = 0; epoch < 1000; epoch++) {
            Tensor o = seq.forward(x_train);
            Tensor loss = loss_fn.criterion(o, y_train);
            if (epoch % 100 == 0) {
                System.out.println(epoch + ": " + loss);
            }
            optim.zero_grad();
            loss.backward();
            optim.step();
        }

        for (Tensor param : seq.parameters()) {
            param.print();
        }
    }
}

``` 