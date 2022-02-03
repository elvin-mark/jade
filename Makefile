test:
	javac -d build/ src/com/data/*.java src/com/functions/F.java src/com/nn/*.java src/com/optim/*.java src/com/test/Test.java
	java -classpath build/ com.test.Test

sample_linear_regression:
	javac -d build/ src/com/data/*.java src/com/functions/F.java src/com/nn/*.java src/com/optim/*.java src/com/test/SampleLinearRegression.java
	java -classpath build/ com.test.SampleLinearRegression

sample_xor:
	javac -d build/ src/com/data/*.java src/com/functions/F.java src/com/nn/*.java src/com/optim/*.java src/com/test/SampleXOR.java
	java -classpath build/ com.test.SampleXOR

jade.jar:
	mkdir -p build
	javac -d build/ src/com/data/*.java src/com/functions/F.java src/com/nn/*.java src/com/optim/*.java 
	jar cvf jade.jar -C build/ .