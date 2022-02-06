![jade_logo](https://raw.githubusercontent.com/elvin-mark/jade/main/images/jade_logo.png)
# <span style="color:#00aa22">**Ja**</span>va <span style="color:#00aa22">**De**</span>ep Learning Library
Deep Learning Library for Java inspired on PyTorch. (It is still not optimized). I really love PyTorch and I was fascinated at how it works, so I decided to implement in Java for fun (and to remember a little bit of Java)

## Layers implemented
- Linear (with bias)
- Conv1d (with bias)
- Conv2d (with bias)
- BatchNorm2d [Coming Soon ...]
- MaxPool2d
- Dropout2d
- Sigmoid
- Tanh
- ReLU
- LeakyReLU
- Softmax 
- LogSoftmax

## Loss functions implemented
- Mean Square Error Loss (MSELosss)
- Cross Entropy Loss (CrossEntropyLoss) 
- Negative Log Likelihood Loss (NLLLoss) 

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

Simple Classification Example
```Java
import com.nn.*;
import com.optim.*;
import com.data.*;
import java.util.*;

public class SampleClassification {
    public static void main(String args[]) {
        Tensor x_train = new Tensor(new int[] { 4, 2 }, new float[] { 1.0f, 5.0f, 2.0f, 4.0f, -1.f, -4.f, -2.f, -3.f });
        Tensor y_train = new Tensor(new int[] { 4, 1 }, new float[] { 0.0f, 0.0f, 1.0f, 1.0f });

        NNModule seq = new Sequential();
        seq.add_module((NNModule) new Linear(2, 5, true));
        seq.add_module((NNModule) new Sigmoid());
        seq.add_module((NNModule) new Linear(5, 2, true));

        Loss loss_fn = new CrossEntropyLoss();
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
        System.out.println(seq.forward(x_train));
        for (Tensor param : seq.parameters()) {
            param.print();
        }
    }
}

```

Simple Classification using Convolutional Layers Example
```Java
import com.data.Tensor;
import com.utils.*;
import com.optim.*;
import com.nn.*;

import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

public class SampleConvClassification {
    public static void main(String args[]) {
        Sequential model = new Sequential();

        model.add_module((NNModule) new Conv2d(1, 8, 3, true));
        model.add_module((NNModule) new Tanh());
        model.add_module((NNModule) new Conv2d(8, 16, 3, true));
        model.add_module((NNModule) new Tanh());
        model.add_module((NNModule) new Conv2d(16, 32, 3, true));
        model.add_module((NNModule) new Tanh());
        model.add_module((NNModule) new Flatten());
        model.add_module((NNModule) new Linear(128, 10, true));

        String path_to_digits = System.getenv().get("PATH_TO_DIGITS");

        Tensor x_train = Misc.loadTensor(Paths.get(path_to_digits,
                "x_train_digits.bin").toString());
        Tensor y_train = Misc.loadTensor(Paths.get(path_to_digits,
                "y_train_digits.bin").toString());

        x_train = x_train.div(new Tensor(16.0f));
        y_train = y_train.div(new Tensor(16.0f));

        Dataset train_ds = new TensorDataset(x_train, y_train);
        DataLoader train_dl = new DataLoader(train_ds, 256, true);

        HashMap<String, Float> hyperparams = new HashMap<String, Float>();
        hyperparams.put("lr", 0.01f);
        Optimizer optim = new SGD(model.parameters(), hyperparams);
        Loss loss_fn = new CrossEntropyLoss();

        float tot_loss;
        for (int epoch = 0; epoch < 5; epoch++) {
            tot_loss = 0.0f;
            for (Tensor[] xy : train_dl) {
                Tensor y_pred = model.forward(xy[0]);
                Tensor loss = loss_fn.criterion(y_pred, xy[1]);
                tot_loss += loss.item();
                optim.zero_grad();
                loss.backward();
                optim.step();
            }
            tot_loss /= train_dl.size();
            System.out.println("Epoch: " + epoch + " Loss: " + tot_loss);
        }
    }

}


```