CXX=g++
CXXFLAGS=-g -std=c++11 -Wall -pedantic 
LDLIBS = -lstdc++fs
DEPS = BarAgent.hpp MultiNightBar.hpp 
OBJ = BarExperiment.o BarAgent.o MultiNightBar.o 

%.o: %.c $(DEPS) $(LDLIBS)
	$(CXX) -c -o $@ $< $(CFLAGS)

experiment: $(OBJ)
	g++ -o $@ $^ $(CFLAGS)

clean:
	rm *.o
	rm experiment