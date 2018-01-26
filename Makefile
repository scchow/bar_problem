CXX=g++
CXXFLAGS=-pg -std=c++11 -Wall -pedantic 
LDLIBS = -lstdc++fs
DEPS = BarAgent.hpp MultiNightBar.hpp 
OBJ = BarAgent.o MultiNightBar.o 
TARGETS = impactExp fixedExp BarAgent.o MultiNightBar.o

# %.o: %.c $(DEPS) $(LDLIBS)
# 	$(CXX) -c -o $@ $< $(CFLAGS)

# experiment: $(OBJ)
# 	g++ -o $@ $^ $(CFLAGS)

all: impact_exp fixed_exp

BarAgent.o: BarAgent.cpp BarAgent.hpp
	g++ -c $(CXXFLAGS) BarAgent.cpp -o BarAgent.o

MultiNightBar.o: MultiNightBar.cpp MultiNightBar.hpp
	g++ -c $(CXXFLAGS) MultiNightBar.cpp -o MultiNightBar.o

impact_exp: BarAgent.o MultiNightBar.o experiment_impact.cpp
	g++ $(CXXFLAGS) BarAgent.o MultiNightBar.o experiment_impact.cpp -o impactExp

fixed_exp: BarAgent.o MultiNightBar.o experiment_fixed.cpp
	g++ $(CXXFLAGS) BarAgent.o MultiNightBar.o experiment_fixed.cpp -o fixedExp

clean:
	rm -f ${TARGETS}