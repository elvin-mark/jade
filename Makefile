random_test:
	javac -d build/ src/com/data/*.java src/com/functions/F.java src/com/nn/*.java src/com/optim/*.java src/com/utils/*.java src/com/vision/*.java src/com/test/Test.java 
	java -classpath build/ com.test.Test

sample_linear_regression:
	javac -d build/ -cp releases/jade.jar test/SampleLinearRegression.java
	java -cp .:./build/:releases/jade.jar SampleLinearRegression

sample_xor:
	javac -d build/ -cp releases/jade.jar test/SampleXOR.java
	java -cp .:./build/:releases/jade.jar SampleXOR

sample_classification:
	javac -d build/ -cp releases/jade.jar test/SampleClassification.java
	java -cp .:./build/:releases/jade.jar SampleClassification

sample_conv_classification:
	javac -d build/ -cp releases/jade.jar test/SampleConvClassification.java
	java -cp .:./build/:releases/jade.jar SampleConvClassification

sample_mlp_classification:
	javac -d build/ -cp releases/jade.jar test/SampleMLPClassification.java
	java -cp .:./build/:releases/jade.jar SampleMLPClassification

# javac -d build/ src/com/data/*.java src/com/functions/F.java src/com/nn/*.java src/com/optim/*.java  src/com/utils/*.java src/com/vision/*.java 
trainer:
	mkdir -p build
	javac -d build/ src/com/gui/Trainer.java
	java -cp .:./build/ com.gui.Trainer

jade.jar:
	mkdir -p build
	javac -d build/ src/com/data/*.java src/com/functions/F.java src/com/nn/*.java src/com/optim/*.java  src/com/utils/*.java src/com/vision/*.java 
	jar cvf releases/jade.jar -C build/ .