test:
	javac -d build/ src/com/data/*.java src/com/functions/F.java src/com/nn/*.java src/com/test/Test.java
	java -classpath build/ com.test.Test