import com.data.Tensor;
import com.utils.*;
import com.optim.*;
import com.nn.*;

import java.nio.file.Path;
import java.util.HashMap;

public class SampleConvClassification {
    public static void print_shape(int[] shape) {
        for (int i : shape) {
            System.out.print(i + " ");
        }
        System.out.println();
    }

    public static void main(String args[]) {
        Sequential model = new Sequential();
        Tensor x = new Tensor(new int[] { 2, 1, 28, 28 });
        model.add_module((NNModule) new Conv2d(1, 8, 3, true));
        model.add_module((NNModule) new Tanh());
        model.add_module((NNModule) new Conv2d(8, 16, 3, true));
        model.add_module((NNModule) new Tanh());
        model.add_module((NNModule) new Flatten());
        model.add_module((NNModule) new Linear(16 * 4 * 4, 10, true));

        return;
        // Tensor x_train = Misc.loadTensor(""");
        // Tensor y_train = Misc.loadTensor("");

        // x_train = x_train.div(new Tensor(16.0f));
        // y_train = y_train.div(new Tensor(16.0f));

        // Dataset train_ds = new TensorDataset(x_train, y_train);
        // DataLoader train_dl = new DataLoader(train_ds, 64, true);

        // HashMap<String, Float> hyperparams = new HashMap<String, Float>();
        // hyperparams.put("lr", 0.01f);
        // Optimizer optim = new SGD(model.parameters(), hyperparams);
        // Loss loss_fn = new CrossEntropyLoss();
        // float tot_loss;
        // for (int epoch = 0; epoch < 10; epoch++) {
        // tot_loss = 0.0f;
        // for (Tensor[] xy : train_dl) {
        // Tensor y_pred = model.forward(xy[0]);
        // Tensor loss = loss_fn.criterion(y_pred, xy[1]);
        // System.out.println(loss.item());
        // tot_loss += loss.item();
        // optim.zero_grad();
        // loss.backward();
        // optim.step();
        // }
        // tot_loss /= train_dl.size();
        // System.out.println("Epoch: " + epoch + " Loss: " + tot_loss);
        // }

        // Tensor output = model.forward(x);
        // System.out.println(output);
        // for (int i : output.shape) {
        // System.out.print(i + " ");
        // }
    }

}
