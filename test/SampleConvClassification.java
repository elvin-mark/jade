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
