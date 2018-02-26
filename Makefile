CXX=g++
CXXFLAGS=-g -std=c++11 -Wall -pedantic 
LDLIBS = -lstdc++fs
# DEPS = BarAgent.hpp MultiNightBar.hpp 
# OBJ = BarAgent.o MultiNightBar.o 
TARGETS = impactExp fixedExp tempExp tempFixedExp impactStagExp impactNormExp rewardExp BarAgent.o MultiNightBar.o 

# %.o: %.c $(DEPS) $(LDLIBS)
# 	$(CXX) -c -o $@ $< $(CFLAGS)

# experiment: $(OBJ)
# 	g++ -o $@ $^ $(CFLAGS)

all: impact_exp fixed_exp temp_exp tempfix_exp impact_stag_exp impact_norm_exp reward_exp

BarAgent.o: BarAgent.cpp BarAgent.hpp
	g++ -c $(CXXFLAGS) BarAgent.cpp -o BarAgent.o

MultiNightBar.o: MultiNightBar.cpp MultiNightBar.hpp
	g++ -c $(CXXFLAGS) MultiNightBar.cpp -o MultiNightBar.o

impact_exp: BarAgent.o MultiNightBar.o experiment_impact.cpp
	g++ $(CXXFLAGS) BarAgent.o MultiNightBar.o experiment_impact.cpp -o impactExp

fixed_exp: BarAgent.o MultiNightBar.o experiment_fixed.cpp
	g++ $(CXXFLAGS) BarAgent.o MultiNightBar.o experiment_fixed.cpp -o fixedExp

temp_exp: BarAgent.o MultiNightBar.o experiment_temp.cpp
	g++ $(CXXFLAGS) BarAgent.o MultiNightBar.o experiment_temp.cpp -o tempExp

tempfix_exp: BarAgent.o MultiNightBar.o experiment_tempfix.cpp
	g++ $(CXXFLAGS) BarAgent.o MultiNightBar.o experiment_tempfix.cpp -o tempFixedExp

impact_stag_exp: BarAgent.o MultiNightBar.o experiment_staggered_impact.cpp
	g++ $(CXXFLAGS) BarAgent.o MultiNightBar.o experiment_staggered_impact.cpp -o impactStagExp

impact_norm_exp: BarAgent.o MultiNightBar.o experiment_norm_impact.cpp
	g++ $(CXXFLAGS) BarAgent.o MultiNightBar.o experiment_norm_impact.cpp -o impactNormExp

reward_exp: BarAgent.o MultiNightBar.o experiment_reward.cpp
	g++ $(CXXFLAGS) BarAgent.o MultiNightBar.o experiment_reward.cpp -o rewardExp

clean:
	rm -f ${TARGETS}
