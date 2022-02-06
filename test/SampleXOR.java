import com.nn.*;
import com.optim.*;
import com.data.*;
import java.util.*;

public class SampleXOR {
    public static void main(String args[]) {
        float[] X = new float[] { 0, 0, 0, 1, 1, 0, 1, 1 };
        float[] y = new float[] { 1, 0, 0, 1, 0, 1, 1, 0 };

        Tensor x_train = new Tensor(new int[] { 4, 2 }, X);
        Tensor y_train = new Tensor(new int[] { 4, 2 }, y);

        NNModule seq = new Sequential();
        seq.add_module((NNModule) new Linear(2, 5, true));
        seq.add_module((NNModule) new Tanh());
        seq.add_module((NNModule) new Linear(5, 2, true));
        seq.add_module((NNModule) new Sigmoid());

        Loss loss_fn = new MSELoss();
        Map<String, Float> optim_params = new HashMap<String, Float>();
        optim_params.put("lr", 0.1f);
        Optimizer optim = new SGD(seq.parameters(), optim_params);

        for (int epoch = 0; epoch < 2000; epoch++) {
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
    }
}
