random_test:
	javac -d build/ src/com/data/*.java src/com/functions/F.java src/com/nn/*.java src/com/optim/*.java src/com/utils/*.java src/com/test/Test.java 
	java -classpath build/ com.test.Test

sample_linear_regression:
	javac -d build/ -cp releases/jade.jar test/SampleLinearRegression.java
	java -classpath build/ -cp .:releases/jade.jar com.test.SampleLinearRegression

sample_xor:
	javac -d build/ -cp releases/jade.jar test/SampleXOR.java
	java -classpath build/ -cp .:releases/jade.jar com.test.SampleXOR

sample_classification:
	javac -d build/ -cp releases/jade.jar test/SampleClassification.java
	java -classpath build/ -cp .:releases/jade.jar com.test.SampleClassification

sample_conv_classification:
	javac -d build/ -cp releases/jade.jar test/SampleConvClassification.java
	java -classpath build/ -cp .:releases/jade.jar com.test.SampleConvClassification

jade.jar:
	mkdir -p build
	javac -d build/ src/com/data/*.java src/com/functions/F.java src/com/nn/*.java src/com/optim/*.java  src/com/utils/*.java src/com/utils/*.java
	jar cvf releases/jade.jar -C build/ .